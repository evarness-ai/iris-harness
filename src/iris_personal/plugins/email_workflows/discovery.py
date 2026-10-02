"""Category discovery for Phase 1 Track 1E.2.

Implements the (embed → HDBSCAN sweep → cohesion → name) pipeline
behind ``iris email bootstrap-categories``. Per ADR-0017 (dynamic
hierarchical categories) + ADR-0018 (CLI shape).

This commit lands the pure-function half of the module:

  * ``embed_corpus`` — sentence-transformers MiniLM-L6-v2,
    L2-normalized output. **Moved to ``iris_harness.llm.embeddings``**
    at OSS plan M3.1: it is a generic embedder that finance and the
    email semantic index also call, so it stays core while this module
    becomes part of the email-workflows plugin.
  * ``cluster_corpus`` — sklearn HDBSCAN sweep + best-run pick.
  * ``compute_cohesion`` — mean intra-cluster centroid cosine
    similarity. Replaces LLM self-reported confidence per ADR-0018 §3.
  * ``build_proposals`` — assemble ``CategoryProposal`` rows from
    rows + embeddings + labels (naming fields stay None — filled by
    the namer in a subsequent commit).

The namer (``name_clusters`` + ``LlamaServerClient``) and the
orchestrator (``bootstrap_categories``) ship in the next commits.

The embedder is **lazy-imported** inside ``embed_corpus`` (now in
``iris_harness.llm.embeddings``) so tests and ``iris --help`` don't pay
the torch/transformers cold-import cost.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import requests

from iris_harness.sdk.llm import EMBED_MODEL_DEFAULT, embed_corpus
from iris_harness.sdk.persistence import data_path
from iris_personal.email.contracts import CategoryProposal, CategoryRepresentative

# sklearn is imported inside ``cluster_corpus`` only. A top-level import made every
# iris-api process pay for sklearn + scipy + pandas + pyarrow (about 135 MiB RSS,
# measured 2026-09-22) at plugin mount, for a function that runs only during corpus
# discovery. The lazy import keeps that cost off the resident set until it is used.

logger = logging.getLogger(__name__)

LLAMA_SERVER_BASE_DEFAULT = "http://localhost:8090/v1"
LLAMA_MODEL_DEFAULT = "qwen3-30b-a3b"
LLAMA_REQUEST_TIMEOUT_S = 60.0

# Sweep grid — matches the spike's empirically-validated range.
# See docs/architecture/spikes/phase1_corpus_discovery.md §HDBSCAN sweep.
MIN_CLUSTER_SIZES = (5, 10, 15, 25, 40)
MIN_SAMPLES_VALUES = (3, 5, 10)

REPRESENTATIVES_PER_CLUSTER = 8
TOP_DOMAINS_PER_CLUSTER = 5

# Constrained taxonomy roots. Branches + leaves are free-form (per
# ADR-0017's dynamic-discovery stance). The model MUST pick from this
# list for the root; anything else is coerced to "other".
ALLOWED_ROOTS = (
    "shopping",
    "finance",
    "news",
    "social",
    "work",
    "personal",
    "learning",
    "travel",
    "jobs",
    "community",
    "automotive",
    "tools",
    "transactional",
    "other",
)

NAMING_SYSTEM_PROMPT = f"""You are a taxonomy classifier for personal email.
Given a cluster of emails grouped by topic, propose a 3-level hierarchical
category for the entire cluster: root.branch.leaf.

The ROOT must be exactly one of:
{', '.join(ALLOWED_ROOTS)}

Rules:
- BRANCH is a free-form 1-3 word category under that root.
- LEAF is a free-form 1-3 word specific topic under that branch.
- Use lowercase, dashes-not-spaces.
- LEAF preference rule: when one sender domain dominates the cluster
  (top domain >= 60% of cluster size), prefer the sender's BRAND or
  DOMAIN for the leaf. Use subject-derived leaves only when the
  cluster spans multiple senders posting the same topic.
- LEAF must NOT name specific seasonal campaigns. Avoid: mother-s-day,
  memorial-day, summer-sale, black-friday, cyber-monday. Use evergreen
  brand or topic names instead.

Examples of correct names:
  shopping.apparel.outlet-brand          (one dominant sender → brand leaf)
  finance.investing.indian-bonds        (multiple senders, same topic)
  social.facebook.friend-updates
  community.religious.place-of-worship      (cross-sender community theme)
  travel.flights.kayak

Examples of corrections to apply:
  an airline loyalty program → travel.airlines.airline
    (airline, not shopping, despite loyalty-program tone)
  a pharmacy's Mother's Day promo  → shopping.pharmacy.pharmacy-brand
    (brand leaf, NOT mother-s-day-deals — time-pegged)
  a furniture brand's Memorial Day sofas     → shopping.furniture.furniture-brand
    (brand leaf, NOT memorial-day-sale — time-pegged)

Return ONLY a JSON object with keys: root, branch, leaf, rationale.
No confidence field — we compute confidence from cluster cohesion.
RATIONALE must be ONE short sentence (12 words or fewer) — just enough to justify the
name, not a full explanation. A long rationale risks being cut off before the JSON's
closing brace, which silently drops the whole cluster.
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_NORMALIZE_RE = re.compile(r"[^a-z0-9-]+")


class NamingClient(Protocol):
    """Protocol the namer calls into. Lets tests stub the LLM."""

    def complete_json(self, system: str, user: str) -> str:
        """Return raw response content. The namer handles JSON parse."""
        ...


@dataclass(frozen=True)
class LlamaServerClient:
    """Transitional shim — talks directly to llama-server over its
    OpenAI-compatible HTTP API.

    Retires when Track 1H lands Tier-3-local in ``tier_router``. At
    that point this class collapses and ``name_clusters`` reaches the
    model through ``tier_router.get_llm_config("classify")`` instead.
    See ADR-0018 §5 for the retirement plan.
    """

    base_url: str = LLAMA_SERVER_BASE_DEFAULT
    model: str = LLAMA_MODEL_DEFAULT
    timeout_s: float = LLAMA_REQUEST_TIMEOUT_S

    def complete_json(self, system: str, user: str) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
                # Headroom past a typical root.branch.leaf + a one-sentence rationale,
                # so an unusually verbose response doesn't get cut before the closing
                # brace and silently drop the cluster (issue #68).
                "max_tokens": 384,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return cast("str", resp.json()["choices"][0]["message"]["content"])


# ``(texts, model name) -> L2-normalised vectors``: ``embed_corpus``'s shape.
Embedder = Callable[[list[str], str], "np.ndarray[Any, Any]"]

# The JSON the namer asks for: the root from the allowed list, the rest free-form.
NAMING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "root": {"type": "string", "enum": list(ALLOWED_ROOTS)},
        "branch": {"type": "string"},
        "leaf": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["root", "branch", "leaf", "rationale"],
}

# ``(system prompt, user message, JSON schema) -> JsonReply``: the judge's governed call.
JsonCall = Callable[[str, str, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class GovernedNamingClient:
    """A :class:`NamingClient` over a governed structured call (``judge.make_tier_llm``:
    the local ``email_judge`` tier). Each cluster is one governed call -- the pre-LLM
    hooks and an audit row fire, and the representatives never leave the local tier --
    where :class:`LlamaServerClient` posts to llama-server directly."""

    # A model call, not a tool: `.call` is reserved for tool handles (test_no_bypass).
    ask: JsonCall

    def complete_json(self, system: str, user: str) -> str:
        reply = self.ask(system, user, NAMING_SCHEMA)
        return json.dumps(reply.data)


@dataclass(frozen=True)
class CorpusRow:
    """One row of the embedding corpus — projection from email.db."""

    id: str
    subject: str
    from_address: str
    from_domain: str | None
    snippet: str

    def to_embedding_text(self) -> str:
        """Embedder input — sender first, since sender carries more
        semantic weight than subject for category formation in the
        spike's observation."""
        domain = self.from_domain or ""
        return f"From: {self.from_address} ({domain})\nSubject: {self.subject}\nSnippet: {self.snippet}"


# ─── Corpus loading ──────────────────────────────────────────────────────────


def load_corpus(db_path: Path, account_id: str, *, include_held: bool = False) -> list[CorpusRow]:
    """Pull every email for ``account_id`` from ``email.db``.

    Held mail (queued for the judge, not released yet) stays out unless
    ``include_held``: email setup discovers categories right after its first fetch,
    when most of the mailbox is still waiting for the judge. Discovery only proposes
    names for the owner to review; it releases nothing and emits nothing."""
    from iris_personal.email.store import visible_condition

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        visible = "1" if include_held else visible_condition(conn)
        rows = conn.execute(
            "SELECT id, subject, from_address, from_domain, snippet "  # noqa: S608
            f"FROM emails WHERE account_id = ? AND {visible} "
            "ORDER BY received_at DESC",
            (account_id,),
        ).fetchall()
    return [
        CorpusRow(
            id=r["id"],
            subject=r["subject"] or "",
            from_address=r["from_address"] or "",
            from_domain=r["from_domain"],
            snippet=r["snippet"] or "",
        )
        for r in rows
    ]


# ─── Clustering sweep ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SweepResult:
    """One (min_cluster_size, min_samples) run of HDBSCAN."""

    min_cluster_size: int
    min_samples: int
    n_clusters: int
    noise_pct: float
    median_cluster_size: int
    largest_cluster_size: int
    labels: np.ndarray[Any, Any]

    def to_summary(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "labels"}
        return d


def cluster_corpus(embeddings: np.ndarray[Any, Any]) -> tuple[SweepResult, list[SweepResult]]:
    """Run the HDBSCAN sweep grid; return (best_pick, full_sweep)."""
    # Lazy on purpose (see the module header). Untyped, and absent without the `ml` extra:
    # pyproject's mypy override for sklearn covers both.
    from sklearn.cluster import HDBSCAN

    results: list[SweepResult] = []
    for mcs in MIN_CLUSTER_SIZES:
        for ms in MIN_SAMPLES_VALUES:
            if ms > mcs:
                continue
            clusterer = HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric="euclidean",  # OK because we L2-normalized
                cluster_selection_method="eom",
                copy=True,
            )
            labels = clusterer.fit_predict(embeddings)
            counts = Counter(labels.tolist())
            noise = counts.get(-1, 0)
            cluster_counts = [c for lbl, c in counts.items() if lbl != -1]
            results.append(
                SweepResult(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    n_clusters=len(cluster_counts),
                    noise_pct=100.0 * noise / max(1, len(labels)),
                    median_cluster_size=(int(np.median(cluster_counts)) if cluster_counts else 0),
                    largest_cluster_size=max(cluster_counts) if cluster_counts else 0,
                    labels=labels,
                )
            )
    return _pick_best(results), results


def _pick_best(results: list[SweepResult]) -> SweepResult:
    """Cost: noise% + penalty for too-few or too-many clusters.

    Identical to the spike's pick function. Rationale captured in
    docs/architecture/spikes/phase1_corpus_discovery.md §HDBSCAN sweep.
    """

    def cost(r: SweepResult) -> float:
        c = r.noise_pct
        if r.n_clusters < 10:
            c += 50 * (10 - r.n_clusters)
        if r.n_clusters > 80:
            c += 5 * (r.n_clusters - 80)
        if r.median_cluster_size < 5:
            c += 20
        return c

    return min(results, key=cost)


# ─── Cohesion ────────────────────────────────────────────────────────────────


def compute_cohesion(embeddings: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]) -> float:
    """Mean cosine similarity of cluster members to the cluster centroid.

    Vectors are already L2-normalized → cosine sim = dot product.
    Centroid is the cluster mean (not unit-norm); we renormalize it
    for fair comparison. Single-member clusters return 1.0 by fiat.
    """
    if mask.sum() == 0:
        return 0.0
    vecs = embeddings[mask]
    if len(vecs) == 1:
        return 1.0
    centroid = vecs.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < 1e-12:
        return 0.0
    cnorm = centroid / norm
    sims = vecs @ cnorm  # cosine sims since vecs are unit-norm
    val = float(sims.mean())
    return max(0.0, min(1.0, val))  # clamp to contract range


# ─── Proposals ───────────────────────────────────────────────────────────────


def build_proposals(
    rows: list[CorpusRow],
    embeddings: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
) -> list[CategoryProposal]:
    """Assemble CategoryProposal objects from rows + embeddings + labels.

    Naming fields (proposed_root/branch/leaf, rationale) stay None —
    the namer fills them in. Representatives are the
    REPRESENTATIVES_PER_CLUSTER points closest to the cluster
    centroid. Top domains are aggregated across the full cluster
    membership.
    """
    proposals: list[CategoryProposal] = []
    unique = sorted({int(lbl) for lbl in labels if lbl != -1})
    for cid in unique:
        mask = labels == cid
        idxs = np.where(mask)[0]
        cluster_vecs = embeddings[idxs]
        centroid = cluster_vecs.mean(axis=0)
        cnorm = centroid / (np.linalg.norm(centroid) + 1e-12)
        sims = cluster_vecs @ cnorm
        order = np.argsort(-sims)
        rep_idxs = idxs[order[:REPRESENTATIVES_PER_CLUSTER]]
        reps = tuple(
            CategoryRepresentative(
                id=rows[i].id,
                subject=rows[i].subject[:200],
                from_address=rows[i].from_address[:200],
                snippet=rows[i].snippet[:160],
            )
            for i in rep_idxs
        )
        domain_counts = tuple(
            Counter((rows[i].from_domain or "(unknown)") for i in idxs).most_common(
                TOP_DOMAINS_PER_CLUSTER
            )
        )
        proposals.append(
            CategoryProposal(
                cluster_id=cid,
                size=int(mask.sum()),
                top_domains=domain_counts,
                representatives=reps,
                member_ids=tuple(rows[i].id for i in idxs),
                cohesion=compute_cohesion(embeddings, mask),
            )
        )
    proposals.sort(key=lambda p: -p.size)
    return proposals


# ─── Orchestrator ────────────────────────────────────────────────────────────


MIN_CORPUS_DEFAULT = 500
BULK_FETCH_MAX_MESSAGES = 2000
BULK_FETCH_COLD_START_DAYS = 180


def bootstrap_categories(
    account_id: str,
    *,
    db_path: Path | None = None,
    min_corpus: int = MIN_CORPUS_DEFAULT,
    fetch_if_low: bool = True,
    naming_client: NamingClient | None = None,
    embed_model: str = EMBED_MODEL_DEFAULT,
    embedder: Embedder | None = None,
    include_held: bool = False,
) -> list[CategoryProposal]:
    """End-to-end: corpus → embed → cluster → cohesion → name.

    Behavior per ADR-0018:
      * §2: if ``fetch_if_low`` and the local corpus has fewer than
        ``min_corpus`` rows for the account, run a one-shot bulk fetch
        before clustering (writes to email.db).
      * §5: a ``naming_client`` must be provided; production wires a
        ``LlamaServerClient``, tests pass a stub.

    ``embedder`` replaces the local MiniLM embedder (tests); ``include_held`` clusters
    mail still waiting for the judge too (see :func:`load_corpus`).

    Returns proposals sorted by size descending. Naming fields can be
    None on a per-cluster basis if the LLM call failed; check
    ``naming_rationale`` for ``error: …`` prefixes.

    Raises:
        ValueError: account_id has no rows in email.db AND fetch_if_low
            is False (or the bulk fetch yielded nothing).
        RuntimeError: bulk fetch raised (missing Gmail credentials,
            etc.) — caller surfaces to user.
    """
    db_path = db_path or data_path("email.db")

    rows = load_corpus(db_path, account_id, include_held=include_held)
    if len(rows) < min_corpus and fetch_if_low:
        logger.info("corpus has %d rows (< %d); running bulk fetch", len(rows), min_corpus)
        _bulk_fetch_for_bootstrap(account_id, db_path)
        rows = load_corpus(db_path, account_id, include_held=include_held)

    if not rows:
        raise ValueError(
            f"no emails for {account_id} in {db_path}. "
            f"Authenticate via `iris auth gmail login` and try again."
        )
    if len(rows) < min_corpus:
        logger.warning(
            "corpus is small (%d rows < %d) — clustering may be unstable",
            len(rows),
            min_corpus,
        )

    texts = [r.to_embedding_text() for r in rows]
    embeddings = (embedder or embed_corpus)(texts, embed_model)

    best, _sweep = cluster_corpus(embeddings)
    logger.info(
        "picked mcs=%d ms=%d → %d clusters, %.1f%% noise, median %d",
        best.min_cluster_size,
        best.min_samples,
        best.n_clusters,
        best.noise_pct,
        best.median_cluster_size,
    )

    proposals = build_proposals(rows, embeddings, best.labels)

    if naming_client is not None:
        proposals = name_clusters(proposals, naming_client)
    else:
        logger.info("no naming_client supplied — returning unnamed proposals")
    return proposals


def _bulk_fetch_for_bootstrap(account_id: str, db_path: Path) -> None:
    """Reset cursor + cold-start fetch up to 2000 messages.

    The cursor reset is bootstrap-specific behavior: a one-time, deliberate "I'm
    starting from scratch" signal. Other callers should NOT erase cursors.

    Goes through the core's mail-provider registry (``email.providers``), so this
    plugin never imports the gmail plugin: whichever provider serves the account
    does the reset and the fetch (OSS plan M5.7 track A).
    """
    from iris_personal.email.providers import mail_provider_for
    from iris_personal.email.store import EmailStore

    provider = mail_provider_for(account_id)
    if provider is None:
        raise RuntimeError(
            f"no mail provider is mounted for {account_id!r}; mount the provider plugin "
            "(e.g. `gmail`) before bootstrapping categories"
        )
    store = EmailStore(db_path=db_path)
    store.ensure_schema()
    provider.reset_cursor(account_id, store=store)
    result = provider.fetch_new(
        account_id,
        store=store,
        max_messages=BULK_FETCH_MAX_MESSAGES,
        cold_start_days=BULK_FETCH_COLD_START_DAYS,
    )
    logger.info("bulk fetch: %d messages", result.fetched)


# ─── Naming ──────────────────────────────────────────────────────────────────


def _normalize_token(value: str | None) -> str:
    if not value:
        return ""
    return _NORMALIZE_RE.sub("-", value.strip().lower()).strip("-")


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Best-effort JSON extraction. Model may wrap in code fences."""
    txt = raw.strip()
    try:
        return cast("dict[str, Any]", json.loads(txt))
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(txt)
    if not m:
        raise ValueError(f"no JSON object in response: {txt[:200]}")
    return cast("dict[str, Any]", json.loads(m.group(0)))


def _build_user_prompt(proposal: CategoryProposal) -> str:
    """Render a proposal as a per-cluster naming prompt."""
    top_total = sum(c for _, c in proposal.top_domains) or 1
    top = ", ".join(f"{d} ({c}, {100 * c // top_total}%)" for d, c in proposal.top_domains)
    lines = [
        f"Cluster size: {proposal.size}",
        f"Cluster cohesion: {proposal.cohesion:.3f}",
        f"Top sender domains: {top}",
        "Representative emails:",
    ]
    for r in proposal.representatives:
        lines.append(f"  - From: {r.from_address[:60]}")
        lines.append(f"    Subject: {r.subject[:120]}")
        if r.snippet:
            lines.append(f"    Snippet: {r.snippet[:140]}")
    return "\n".join(lines)


def name_clusters(
    proposals: list[CategoryProposal],
    client: NamingClient,
) -> list[CategoryProposal]:
    """Hand each proposal to the LLM; return new proposals with names set.

    Per-cluster failures are caught and recorded in ``naming_rationale``
    so a single broken response can't abort the whole pass. Roots
    outside ``ALLOWED_ROOTS`` are coerced to ``"other"``.
    """
    out: list[CategoryProposal] = []
    for i, proposal in enumerate(proposals):
        try:
            raw = client.complete_json(NAMING_SYSTEM_PROMPT, _build_user_prompt(proposal))
            parsed = _parse_json_response(raw)
            root = _normalize_token(parsed.get("root"))
            if root not in ALLOWED_ROOTS:
                logger.warning(
                    "cluster %d: model returned non-allowed root %r; coercing to 'other'",
                    proposal.cluster_id,
                    root,
                )
                root = "other"
            branch = _normalize_token(parsed.get("branch"))
            leaf = _normalize_token(parsed.get("leaf"))
            rationale = parsed.get("rationale") or None
            named = proposal.model_copy(
                update={
                    "proposed_root": root,
                    "proposed_branch": branch or None,
                    "proposed_leaf": leaf or None,
                    "naming_rationale": rationale,
                }
            )
        except Exception as exc:  # noqa: BLE001 — single-cluster failure must not abort
            logger.warning("cluster %d: naming failed: %s", proposal.cluster_id, exc)
            named = proposal.model_copy(update={"naming_rationale": f"error: {exc}"})
        logger.info(
            "[%d/%d] cluster %d (size %d, cohesion %.2f) → %s.%s.%s",
            i + 1,
            len(proposals),
            named.cluster_id,
            named.size,
            named.cohesion,
            named.proposed_root,
            named.proposed_branch,
            named.proposed_leaf,
        )
        out.append(named)
    return out


# ─── Acceptance ──────────────────────────────────────────────────────────────


def category_fields(raw: Mapping[str, Any], account_id: str) -> tuple[str | None, dict[str, Any]]:
    """Translate one proposal (a ``proposals.jsonl`` row) into ``Category`` fields.

    Returns ``(error_message, category_kwargs)``; a non-``None`` error means the row
    cannot be accepted (``iris email accept-categories`` aborts on it, email setup
    leaves it out and says why).
    """
    from iris_personal.email.category_store import ALLOWED_ROOTS as STORE_ROOTS
    from iris_personal.email.category_store import make_path

    cluster_id = raw.get("cluster_id")
    root = (raw.get("proposed_root") or "").strip().lower()
    branch = (raw.get("proposed_branch") or "").strip().lower()
    leaf = (raw.get("proposed_leaf") or "").strip().lower()

    if not root:
        return (f"cluster {cluster_id}: empty proposed_root", {})
    if root not in STORE_ROOTS:
        return (
            f"cluster {cluster_id}: root {root!r} not in ALLOWED_ROOTS",
            {},
        )
    if not branch:
        return (f"cluster {cluster_id}: empty proposed_branch", {})
    if not leaf:
        return (f"cluster {cluster_id}: empty proposed_leaf", {})

    path = make_path("email", root, branch, leaf)
    metadata = {
        "cluster_id": cluster_id,
        "top_domains": raw.get("top_domains", []),
        "naming_rationale": raw.get("naming_rationale"),
        "size_at_acceptance": raw.get("size"),
    }
    return (
        None,
        {
            "path": path,
            "type": "email",
            "root": root,
            "branch": branch,
            "leaf": leaf,
            "account_id": account_id,
            "cohesion": raw.get("cohesion"),
            "metadata": metadata,
        },
    )
