"""Email triage classifier — Phase 1 Track 1G.

Implements the hybrid kNN-then-LLM classification per ADR-0021:

  1. Load accepted categories for the account from ``categories``.
  2. Lazy-compute per-category centroids from the JSONL's ``member_ids``
     (re-embed the cluster representatives via ``embed_corpus``).
  3. For each incoming email:
       a. Embed (subject + from + from_domain + snippet).
       b. Cosine-similarity to every centroid → take top-3 candidates.
       c. Hand the 3 paths + email envelope to ``LlamaServerClient``
          with a constrained "pick exactly one" prompt.
       d. confidence = chosen.cohesion × 0.9 (ADR-0021 §6 heuristic;
          Track 1J replaces it).
       e. ``EmailStore.mark_classified`` writes the result;
          ``email.classified`` event fires.

Per-email soft-fail per ADR-0021 §7 — any error leaves the row
unclassified and logs WARN. One bad email cannot break the batch.

The ``LlamaServerClient`` reuse is marked transitional per ADR-0018 §5;
Track 1H retires the shim by plumbing Tier-3-local into
``tier_router.get_llm_config("classify")``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from iris_harness.sdk.config import workspace_dir
from iris_harness.sdk.content import wrap_external_content
from iris_harness.sdk.events import EventBus, get_default_bus
from iris_harness.sdk.llm import EMBED_MODEL_DEFAULT, embed_corpus
from iris_harness.sdk.persistence import data_path
from iris_harness.sdk.process_state import track_globals
from iris_personal.email.category_store import Category, CategoryStore
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.events import EMAIL_CLASSIFIED, EmailClassifiedPayload
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.discovery import (
    LLAMA_SERVER_BASE_DEFAULT,
    CorpusRow,
    LlamaServerClient,
    NamingClient,
)

logger = logging.getLogger(__name__)

# Public API — `embed_corpus` is re-exported (imported from the core embedder,
# `iris_harness.llm.embeddings`, at OSS plan M3.1; formerly from
# iris_personal.plugins.email_workflows.discovery) so sibling modules (knn_gate, sweep) and the
# embedder-injection seam can patch it via `iris_personal.plugins.email_workflows.triage.embed_corpus`.
__all__ = [
    "embed_corpus",
    "CategoryCentroid",
    "EmailTriageClassifier",
    "TriageResult",
    "build_centroids",
    "build_email_triage_classifier",
    "reset_lazy_classifier",
    "root_margin",
    "subscribe_email_triage",
    "PICKER_SYSTEM_PROMPT",
    "LLM_PICKER_RELIABILITY_PRIOR",
    "FALLBACK_CONFIDENCE",
    "CLASSIFIER_TAG",
    "CLASSIFIER_TAG_PURE_KNN",
    "KNN_GATE_MIN_COSINE",
    "KNN_GATE_MIN_MARGIN",
]

# Constrained picker prompt — the LLM's job here is local choice
# between 3 already-similar candidates, not free-form taxonomy.
# Token cost ~150/call vs ~500 for the discovery namer.
PICKER_SYSTEM_PROMPT = """You are an email classifier. Given an email envelope
and 3 candidate categories, pick EXACTLY ONE category path that best fits
the email. The categories are already pre-filtered as the closest matches
by embedding similarity — you are picking the best of three.

Return ONLY a JSON object with one key:
  {"chosen_path": "<one of the three exact paths>"}
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# ADR-0021 §6 heuristic — confidence = chosen.cohesion × this prior.
# Track 1J replaces this with a learned confidence model.
LLM_PICKER_RELIABILITY_PRIOR = 0.9
FALLBACK_CONFIDENCE = 0.5  # used when the LLM's answer doesn't match top-3
CLASSIFIER_TAG = "tier3-local-knn"
CLASSIFIER_TAG_PURE_KNN = "pure-knn"

# ADR-0022 §3 — confidence gate for the pure-kNN classifier. Both
# conditions required: top-1 similarity ≥ MIN, AND margin between top-1
# and top-2 ≥ MIN.
#
# Tuned 2026-05-26 by Track 1L's kNN-gate measurement against the
# user-labeled holdout — see docs/architecture/spikes/phase1_knn_gate.md.
# Original ADR-0022 defaults were (0.70, 0.05); 2026-05-26 retune picked
# (0.50, 0.02) for the leaf-level margin.
#
# Re-tuned 2026-06-12 for the ROOT-AWARE margin (ADR-0022 amendment):
# with cross-root margins the margin axis is informative again and the
# cos floor can drop — sweep picked (0.35, 0.08): 31/51 holdout gated
# (61%) at 90.3% root accuracy, vs 25/51 (49%) at 92.0% before. The +6
# answered / +1 wrong trade is accepted per ADR-0023 §5 (max coverage
# at the gated floor); wrong filings are recoverable via
# `iris email recategorize`, which feeds this measurement.
KNN_GATE_MIN_COSINE = 0.35
KNN_GATE_MIN_MARGIN = 0.08


# ─── Centroid cache ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CategoryCentroid:
    """One category's centroid + cohesion, ready for kNN lookup."""

    path: str
    cohesion: float
    centroid: np.ndarray[Any, Any]  # L2-normalized 384-dim float32
    # Set on correction exemplars only (ADR-0024 extension): the corrected
    # message's sender domain. A domain match gives the exemplar
    # PRECEDENCE over cluster centroids — a correction is a sender-level
    # routing fact ("this sender belongs to X"), and must win even when
    # stale cluster categories for the same sender tie across roots.
    from_domain: str | None = None


def _path_root(path: str) -> str:
    """Return the root segment of an ``email/root/branch/leaf`` path."""
    parts = path.split("/")
    return parts[1] if len(parts) > 1 else path


def root_margin(sims: Any, ordered: Sequence[Any]) -> float:
    """cos_top1 minus the best candidate from a DIFFERENT root.

    ADR-0022 amendment (Phase 3): the gate's ambiguity signal must measure
    cross-root competition only — sibling leaves within one root are an
    acceptable tie for filing purposes. Returns 1.0 when every centroid
    shares the top root (no cross-root competitor exists).
    """
    if len(ordered) == 0:
        return 0.0
    top_root = _path_root(ordered[0].path)
    top1 = float(sims[0])
    for sim, centroid in zip(sims[1:], ordered[1:], strict=False):
        if _path_root(centroid.path) != top_root:
            return top1 - float(sim)
    return 1.0


def _proposal_path(workspace_dir: Path, account_id: str) -> Path:
    """Translate account_id → workspace JSONL path. Matches the CLI slug."""
    slug = account_id.replace(":", "-").replace("@", "-at-")
    return workspace_dir / "email" / slug / "proposals.jsonl"


def _row_from_representative(rep: dict[str, Any]) -> CorpusRow:
    """Project a JSONL representative into the shape embed_corpus consumes."""
    from_addr = rep.get("from") or rep.get("from_address") or ""
    return CorpusRow(
        id=rep.get("id", ""),
        subject=rep.get("subject", ""),
        from_address=from_addr,
        from_domain=(
            from_addr.split("@", 1)[1].split(">", 1)[0].strip().lower()
            if "@" in from_addr
            else None
        ),
        snippet=rep.get("snippet", ""),
    )


def build_centroids(
    workspace_dir: Path,
    account_id: str,
    *,
    active_paths: set[str],
    embed_model: str = EMBED_MODEL_DEFAULT,
) -> list[CategoryCentroid]:
    """Read proposals.jsonl, re-embed cluster representatives, return
    one CategoryCentroid per active accepted category.

    A category is *active* when its path is in ``active_paths`` (the
    set CategoryStore.list(active_only=True) returned for the account).
    JSONL entries for paths not in that set are skipped — they were
    proposed but never accepted.
    """
    jsonl = _proposal_path(workspace_dir, account_id)
    if not jsonl.exists():
        raise FileNotFoundError(
            f"proposals.jsonl missing for {account_id}: {jsonl}. "
            f"Run `iris email bootstrap-categories` first."
        )

    proposals: list[dict[str, Any]] = []
    for line in jsonl.read_text().splitlines():
        if line.strip():
            proposals.append(json.loads(line))

    # Bucket reps per (root, branch, leaf) and only keep ones the user accepted.
    selected: list[tuple[str, float, list[CorpusRow]]] = []
    for p in proposals:
        root = p.get("proposed_root")
        branch = p.get("proposed_branch")
        leaf = p.get("proposed_leaf")
        if not all((root, branch, leaf)):
            continue
        path = f"email/{root}/{branch}/{leaf}"
        if path not in active_paths:
            continue
        reps = [_row_from_representative(r) for r in (p.get("representatives") or [])]
        if not reps:
            logger.warning("category %s has no representatives in JSONL — skipping", path)
            continue
        cohesion = float(p.get("cohesion") or 0.0)
        selected.append((path, cohesion, reps))

    if not selected:
        logger.warning(
            "no overlap between proposals.jsonl and active categories for %s",
            account_id,
        )
        return []

    # One embed pass for all reps across all categories — much faster than
    # per-category model loads.
    all_texts: list[str] = []
    boundaries: list[int] = []
    cursor = 0
    for _, _, reps in selected:
        boundaries.append(cursor)
        for r in reps:
            all_texts.append(r.to_embedding_text())
            cursor += 1
    boundaries.append(cursor)

    vecs = embed_corpus(all_texts, embed_model)

    centroids: list[CategoryCentroid] = []
    for idx, (path, cohesion, _) in enumerate(selected):
        start, end = boundaries[idx], boundaries[idx + 1]
        block = vecs[start:end]
        centroid = block.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-12:
            logger.warning("centroid for %s is zero-norm; skipping", path)
            continue
        centroid /= norm
        centroids.append(
            CategoryCentroid(
                path=path,
                cohesion=cohesion,
                centroid=centroid.astype(np.float32),
            )
        )

    logger.info("built %d centroids for %s", len(centroids), account_id)
    return centroids


# ─── Picker prompt + LLM ─────────────────────────────────────────────────────


def _build_picker_prompt(email: EmailMessage, candidates: list[CategoryCentroid]) -> str:
    domain = email.from_domain or ""
    paths = "\n".join(
        f"  {i + 1}. {c.path}  (cohesion {c.cohesion:.2f})" for i, c in enumerate(candidates)
    )
    # The envelope is text a third party wrote (issue #148): marked and redacted as data.
    envelope = wrap_external_content(
        f"Email envelope:\n"
        f"  From: {email.from_address} ({domain})\n"
        f"  Subject: {email.subject}\n"
        f"  Snippet: {email.snippet}",
        source="email",
        tool="triage",
    )
    return (
        f"{envelope}\n\n"
        f"Candidate categories (closest by embedding distance):\n{paths}\n\n"
        f"Pick exactly one path."
    )


def _parse_picker_response(raw: str) -> str | None:
    """Return the picked path or None if the response is unparseable."""
    txt = raw.strip()
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(txt)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    chosen = obj.get("chosen_path")
    return chosen if isinstance(chosen, str) else None


# ─── Classifier ──────────────────────────────────────────────────────────────


@dataclass
class TriageResult:
    """One classification outcome — surfaces to the CLI summary table.

    Per ADR-0022, a result can be one of:
      * classified  — category_path + confidence set, queued=False
      * queued      — queued=True, category_path=None
                       (kNN was ambiguous; deferred to batch review)
      * error       — error set, queued=False, category_path=None
      * no-op       — all None (categories not configured for account)
    """

    message_id: str
    category_path: str | None
    confidence: float | None
    error: str | None = None
    queued: bool = False
    classifier: str = CLASSIFIER_TAG_PURE_KNN


@dataclass
class EmailTriageClassifier:
    """Hybrid kNN-then-LLM email classifier per ADR-0021.

    Construct with ``classifier = EmailTriageClassifier()`` for production
    defaults (loads from the data dir's iris.db, $IRIS_HOME/workspace, llama-server on
    :8090). Tests pass a stub ``naming_client`` and an ``embedder`` to
    avoid loading MiniLM + torch.
    """

    workspace_dir: Path = field(default_factory=workspace_dir)
    db_path: Path = field(default_factory=lambda: data_path("iris.db"))
    email_db_path: Path = field(default_factory=lambda: data_path("email.db"))
    naming_client: NamingClient | None = None
    bus: EventBus | None = None
    embed_model: str = EMBED_MODEL_DEFAULT
    # Optional injection point for tests: a callable
    # ``(texts: list[str], model_name: str) -> np.ndarray``.
    embedder: Any = None
    # Filled by load_categories.
    _centroids_by_account: dict[str, list[CategoryCentroid]] = field(default_factory=dict)

    def _embed(self, texts: list[str]) -> np.ndarray[Any, Any]:
        if self.embedder is not None:
            return cast("np.ndarray[Any, Any]", self.embedder(texts, self.embed_model))
        return embed_corpus(texts, self.embed_model)

    def _resolve_naming_client(self) -> NamingClient:
        if self.naming_client is not None:
            return self.naming_client
        # ADR-0018 §5 transitional — retires when Track 1H wires tier_router.
        return LlamaServerClient(base_url=LLAMA_SERVER_BASE_DEFAULT)

    def load_categories(self, account_id: str) -> list[CategoryCentroid]:
        """Build the centroid cache for an account. Idempotent — second
        call returns the cached list."""
        if account_id in self._centroids_by_account:
            return self._centroids_by_account[account_id]
        store = CategoryStore(db_path=self.db_path)
        store.ensure_schema()
        active: list[Category] = store.list(type="email", account_id=account_id, active_only=True)
        active_paths = {c.path for c in active}
        if not active_paths:
            self._centroids_by_account[account_id] = []
            logger.warning("no active categories for %s; cannot classify", account_id)
            return []
        # Patch the embedder lookup so build_centroids uses our injection.
        if self.embedder is not None:
            from iris_personal.plugins.email_workflows import triage as _self_mod

            original = _self_mod.embed_corpus
            _self_mod.embed_corpus = self.embedder
            try:
                centroids = build_centroids(
                    self.workspace_dir, account_id, active_paths=active_paths
                )
            finally:
                _self_mod.embed_corpus = original
        else:
            centroids = build_centroids(
                self.workspace_dir,
                account_id,
                active_paths=active_paths,
                embed_model=self.embed_model,
            )
        centroids = centroids + self._correction_exemplars(account_id, active_paths)
        self._centroids_by_account[account_id] = centroids
        return centroids

    # ADR-0024 extension (Phase 3 follow-up): every user correction becomes
    # an exemplar centroid, so a single `iris email recategorize` makes that
    # sender's future mail classifiable — the only lever that reaches
    # unclustered one-off mail (all remaining holdout abstentions are
    # cos_top1 < 0.35 to every cluster centroid). The holdout labels are
    # deliberately NOT used here: they are evaluation-only. Note that the
    # knn-gate measurement's --include-corrections option now overlaps
    # training data and should not be combined with exemplar centroids.
    EXEMPLAR_COHESION = 0.95  # user-confirmed label — highest-trust centroid

    def _correction_exemplars(
        self, account_id: str, active_paths: set[str]
    ) -> list[CategoryCentroid]:
        """Embed user-corrected emails into per-message exemplar centroids."""
        import sqlite3

        exemplars: list[CategoryCentroid] = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT payload FROM categories_history "
                    "WHERE source='user-classification-correction' AND op='update'"
                ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("correction exemplars unavailable: %s", exc)
            return []

        email_store = EmailStore(db_path=self.email_db_path)
        email_store.ensure_schema()
        for (payload_raw,) in rows:
            try:
                payload = json.loads(payload_raw or "{}")
                if payload.get("account_id") != account_id:
                    continue
                new_path = payload.get("new_path") or ""
                if new_path not in active_paths:
                    continue
                message = email_store.get(str(payload.get("message_id") or ""))
                if message is None:
                    continue
                vec = self._embed_email(message)
                exemplars.append(
                    CategoryCentroid(
                        path=new_path,
                        cohesion=self.EXEMPLAR_COHESION,
                        centroid=vec.astype(np.float32),
                        from_domain=message.from_domain,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — per-correction soft-fail
                logger.warning("skipping correction exemplar: %s", exc)
        if exemplars:
            logger.info(
                "loaded %d correction exemplar centroid(s) for %s",
                len(exemplars),
                account_id,
            )
        return exemplars

    def classify(self, email: EmailMessage) -> TriageResult:
        """Default per-email classifier per ADR-0022: pure kNN only.

        Soft-fails on any error (LLM unreachable doesn't even apply
        here because we don't call an LLM). Returns a TriageResult
        with either a classification (gate passed) or queued=True
        (gate failed; deferred to ``iris email triage-batch``).
        """
        return self.classify_pure_knn(email)

    def classify_pure_knn(self, email: EmailMessage) -> TriageResult:
        """Pure-kNN classifier with confidence gate per ADR-0022 §3.

        Gate: ``cos_top1 ≥ KNN_GATE_MIN_COSINE AND
              root_margin ≥ KNN_GATE_MIN_MARGIN``

        where ``root_margin`` is cos_top1 minus the best candidate from a
        DIFFERENT root (ADR-0022 amendment, Phase 3 category expansion):
        the original top1−top2 margin conflated "unsure which leaf" with
        "unsure which root" — adding sibling categories within a root
        (e.g. a second ``learning`` leaf) crushed margins and *reduced*
        coverage, measured live at 46% → 20% on the holdout when the
        taxonomy grew 47 → 83. Within-root ambiguity is harmless for
        filing; only cross-root ambiguity should abstain.

          * Gate passes → category_path = top1.path,
            confidence = cos_top1, classifier = 'pure-knn'.
          * Gate fails  → queued=True, error="queued: cos1=X root_margin=Y".
          * No centroids → error="no active categories for account".

        Zero LLM calls. ~50 ms/email on M4 Max.
        """
        try:
            centroids = self.load_categories(email.account_id)
        except FileNotFoundError as exc:
            return TriageResult(email.id, None, None, error=str(exc))

        if not centroids:
            return TriageResult(email.id, None, None, error="no active categories for account")

        try:
            email_vec = self._embed_email(email)
            # Correction precedence: when the user corrected mail from this
            # sender, only their exemplars compete for it. The embedding
            # gate (cos >= MIN) still applies as a sanity check; the margin
            # does not — the user already resolved the ambiguity.
            domain_exemplars = [
                c for c in centroids if c.from_domain and c.from_domain == email.from_domain
            ]
            candidates = domain_exemplars or centroids
            sims, ordered = self._knn_top_k_with_sims(email_vec, candidates, k=len(candidates))
        except Exception as exc:  # noqa: BLE001 — per-email soft-fail
            logger.warning("classify_pure_knn failed for %s: %s", email.id, exc)
            return TriageResult(email.id, None, None, error=str(exc))

        top1_sim = float(sims[0])
        margin = 1.0 if domain_exemplars else root_margin(sims, ordered)

        if top1_sim < KNN_GATE_MIN_COSINE or margin < KNN_GATE_MIN_MARGIN:
            note = f"queued: cos1={top1_sim:.3f} root_margin={margin:.3f}"
            return TriageResult(
                email.id,
                None,
                None,
                error=note,
                queued=True,
                classifier=CLASSIFIER_TAG_PURE_KNN,
            )

        chosen = ordered[0]
        return TriageResult(
            email.id,
            chosen.path,
            top1_sim,
            classifier=CLASSIFIER_TAG_PURE_KNN,
        )

    def classify_with_llm(self, email: EmailMessage, *, top_k: int = 3) -> TriageResult:
        """Original hybrid kNN-then-LLM classifier per ADR-0021 §1.

        Used by the batch path (``iris email triage-batch``) to drain
        the pending_review queue, and by ``iris email triage --use-llm``
        for direct invocation. NOT called per email on the auto-fire
        path — that's pure-kNN now per ADR-0022.

        ``classifier`` is tagged ``CLASSIFIER_TAG`` ("tier3-local-knn")
        so downstream consumers can distinguish the two paths.
        """
        try:
            centroids = self.load_categories(email.account_id)
        except FileNotFoundError as exc:
            return TriageResult(email.id, None, None, error=str(exc))

        if not centroids:
            return TriageResult(email.id, None, None, error="no active categories for account")

        try:
            email_vec = self._embed_email(email)
            candidates = self._knn_top_k(email_vec, centroids, k=top_k)
            chosen, confidence = self._pick_via_llm(email, candidates)
            return TriageResult(email.id, chosen.path, confidence, classifier=CLASSIFIER_TAG)
        except Exception as exc:  # noqa: BLE001 — per-email soft-fail
            logger.warning("classify_with_llm failed for %s: %s", email.id, exc)
            return TriageResult(email.id, None, None, error=str(exc))

    def _embed_email(self, email: EmailMessage) -> np.ndarray[Any, Any]:
        row = CorpusRow(
            id=email.id,
            subject=email.subject,
            from_address=email.from_address,
            from_domain=email.from_domain,
            snippet=email.snippet,
        )
        vecs = self._embed([row.to_embedding_text()])
        return cast("np.ndarray[Any, Any]", vecs[0])

    def _knn_top_k(
        self,
        email_vec: np.ndarray[Any, Any],
        centroids: list[CategoryCentroid],
        *,
        k: int,
    ) -> list[CategoryCentroid]:
        _, ordered = self._knn_top_k_with_sims(email_vec, centroids, k=k)
        return ordered

    def _knn_top_k_with_sims(
        self,
        email_vec: np.ndarray[Any, Any],
        centroids: list[CategoryCentroid],
        *,
        k: int,
    ) -> tuple[np.ndarray[Any, Any], list[CategoryCentroid]]:
        """Return (sorted_sims, sorted_centroids) for the top-k matches.

        Used by classify_pure_knn so it can evaluate the confidence gate
        without re-computing similarities.
        """
        matrix = np.stack([c.centroid for c in centroids])
        sims = matrix @ email_vec  # both L2-normalized → cosine sim
        order = np.argsort(-sims)
        top = order[: min(k, len(centroids))]
        return sims[top], [centroids[i] for i in top]

    def _pick_via_llm(
        self, email: EmailMessage, candidates: list[CategoryCentroid]
    ) -> tuple[CategoryCentroid, float]:
        client = self._resolve_naming_client()
        raw = client.complete_json(PICKER_SYSTEM_PROMPT, _build_picker_prompt(email, candidates))
        chosen_path = _parse_picker_response(raw)
        path_to_cand = {c.path: c for c in candidates}
        if chosen_path is None or chosen_path not in path_to_cand:
            # Fallback: take kNN top-1; mark with lower confidence.
            logger.warning(
                "picker returned %r not in candidate set; falling back to kNN top-1",
                chosen_path,
            )
            return candidates[0], FALLBACK_CONFIDENCE
        chosen = path_to_cand[chosen_path]
        # ADR-0021 §6 — Track 1J replaces this heuristic.
        confidence = max(0.0, min(1.0, chosen.cohesion * LLM_PICKER_RELIABILITY_PRIOR))
        return chosen, confidence

    # ------------------------------------------------------------------
    # Batch + side-effect-bearing entry points
    # ------------------------------------------------------------------

    def classify_and_persist(
        self,
        email: EmailMessage,
        *,
        store: EmailStore | None = None,
        use_llm: bool = False,
    ) -> TriageResult:
        """Classify one email AND apply the right state to ``email.db``.

        Per ADR-0022 routing:
          * gate-passed kNN / LLM-classified → mark_classified
            (sets triage_state='classified') + emit ``email.classified``.
          * gate-failed kNN → mark_pending_review (triage_state=
            'pending_review'); NO event emitted.
          * other soft-fail → mark_triage_error (triage_state='error');
            NO event.

        ``use_llm=True`` routes through ``classify_with_llm`` instead
        of the default pure-kNN path. Used by ``triage-batch`` and
        ``iris email triage --use-llm``.
        """
        result = self.classify_with_llm(email) if use_llm else self.classify(email)
        s = store or EmailStore(db_path=self.email_db_path)
        s.ensure_schema()

        if result.queued:
            s.mark_pending_review(result.message_id)
            return result

        if result.category_path is not None and result.confidence is not None:
            updated = s.mark_classified(
                result.message_id,
                category=result.category_path,
                confidence=result.confidence,
            )
            if not updated:
                logger.warning(
                    "classified ghost id %s — message not in email.db",
                    result.message_id,
                )
                return result
            bus = self.bus if self.bus is not None else get_default_bus()
            bus.emit_sync(
                EMAIL_CLASSIFIED,
                EmailClassifiedPayload(
                    id=result.message_id,
                    account_id=email.account_id,
                    category_path=result.category_path,
                    confidence=result.confidence,
                    classifier=result.classifier,
                ),
            )
            return result

        if result.error is not None:
            s.mark_triage_error(result.message_id)
        return result

    def classify_unclassified(
        self,
        account_id: str,
        *,
        limit: int = 20,
        store: EmailStore | None = None,
        use_llm: bool = False,
    ) -> list[TriageResult]:
        """Pull the N most-recent unclassified emails for the account
        and classify each via the pure-kNN path (or LLM if requested).

        Used by ``iris email triage`` (manual) and the event subscriber.
        """
        s = store or EmailStore(db_path=self.email_db_path)
        s.ensure_schema()
        emails = s.list_unclassified(account_id, limit=limit)
        return [self.classify_and_persist(e, store=s, use_llm=use_llm) for e in emails]

    def classify_pending_review_batch(
        self,
        account_id: str,
        *,
        limit: int = 20,
        store: EmailStore | None = None,
    ) -> list[TriageResult]:
        """Drain the pending_review queue via the LLM picker per ADR-0022.

        Called by ``iris email triage-batch``. Each pending-review email
        runs through ``classify_with_llm``; classified rows are persisted
        and emit the event, dropping out of the queue. Rows that still
        fail (LLM unreachable, off-menu pick) stay queued or move to
        error state depending on the failure mode (see
        ``classify_and_persist`` routing).
        """
        s = store or EmailStore(db_path=self.email_db_path)
        s.ensure_schema()
        emails = s.list_pending_review(account_id, limit=limit)
        return [self.classify_and_persist(e, store=s, use_llm=True) for e in emails]


def build_email_triage_classifier() -> EmailTriageClassifier:
    """Factory used by bootstrap when subscribing to email.new_arrived."""
    return EmailTriageClassifier(bus=get_default_bus())


# ─── Module-level lazy singleton + event subscription ────────────────────────
#
# Per ADR-0021 §4 + the in-session Q2 decision: the runtime subscribes to
# email.new_arrived via this module's lazy singleton so MiniLM + torch don't
# load on `iris --help`. The classifier is built on first event dispatch, not
# at import time.

_LAZY_CLASSIFIER: EmailTriageClassifier | None = None


def _get_lazy_classifier() -> EmailTriageClassifier:
    """One-per-process triage classifier for the event subscriber path.

    Tests reset via ``reset_lazy_classifier()``.
    """
    global _LAZY_CLASSIFIER
    if _LAZY_CLASSIFIER is None:
        _LAZY_CLASSIFIER = build_email_triage_classifier()
    return _LAZY_CLASSIFIER


def reset_lazy_classifier() -> None:
    """Test seam — force the next ``_get_lazy_classifier`` call to rebuild."""
    global _LAZY_CLASSIFIER
    _LAZY_CLASSIFIER = None


def _handle_email_new_arrived(payload: Any) -> None:
    """Event handler — subscribed in ``subscribe_email_triage``.

    For each new message id, fetch from email.db and run the classifier.
    Per-message soft-fail: a single broken message can't break the batch
    (the classifier itself enforces this; we additionally guard against
    bad id lookups here).
    """
    from iris_personal.email.events import EmailNewArrivedPayload  # local import → no cycle risk

    if not isinstance(payload, EmailNewArrivedPayload):
        logger.warning("email-triage subscriber got unexpected payload %r", type(payload))
        return
    classifier = _get_lazy_classifier()
    store = EmailStore(db_path=classifier.email_db_path)
    store.ensure_schema()
    for msg_id in payload.new_message_ids:
        message = store.get(msg_id)
        if message is None:
            logger.warning("email-triage: ghost id %s in email.new_arrived", msg_id)
            continue
        classifier.classify_and_persist(message, store=store)


def subscribe_email_triage(bus: EventBus | None = None) -> None:
    """Wire the triage handler to ``email.new_arrived`` on the given bus
    (default: the singleton bus from ``get_default_bus``)."""
    from iris_personal.email.events import EMAIL_NEW_ARRIVED  # local import → cheap, avoids cycle

    target_bus = bus if bus is not None else get_default_bus()
    target_bus.on(EMAIL_NEW_ARRIVED, _handle_email_new_arrived)
    logger.info("email-triage subscribed to %s", EMAIL_NEW_ARRIVED)


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_LAZY_CLASSIFIER")
