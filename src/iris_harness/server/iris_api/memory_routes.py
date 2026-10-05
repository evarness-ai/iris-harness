"""Memory: facts, contradictions, history, the wiki and graph, logs, RAG documents, and reversible removal (ADR-0119).

    GET    /memory/contradictions
    POST   /memory/contradictions/ack
    GET    /memory/facts
    GET    /memory/review
    POST   /memory/review/{proposal_id}/approve
    POST   /memory/review/{proposal_id}/reject
    GET    /memory/terms
    POST   /memory/terms/{name}/reject
    POST   /memory/terms/{name}/activate
    GET    /memory/terms/{name}/promotion
    GET    /memory/entities
    POST   /memory/entities/{decision_id}/same
    POST   /memory/entities/{decision_id}/distinct
    POST   /memory/entities/{decision_id}/unmerge
    POST   /memory/facts/{key}/confirm
    GET    /memory/lessons
    POST   /memory/lessons/{proposal_id}/approve
    POST   /memory/lessons/{proposal_id}/reject
    POST   /memory/review/expire
    GET    /memory/profile
    PUT    /memory/profile
    POST   /memory/facts/{key}/forget
    POST   /memory/facts/{key}/restore
    PATCH  /memory/facts/{key}
    GET    /memory/facts/{key}/history
    GET    /memory/review/test-sessions
    GET    /memory/sessions
    GET    /memory/context/{session_id}
    POST   /memory/export
    GET    /memory/graph
    GET    /memory/housekeeping
    POST   /memory/housekeeping/run
    GET    /logs/archive
    POST   /logs/archive/restore
    POST   /memory/forget
    POST   /memory/sessions/purge
    GET    /memory/removed
    POST   /memory/removed/preview
    POST   /memory/removed
    POST   /memory/removed/delete
    POST   /memory/removed/{removal_id}/restore
    GET    /memory/{session_id}
    DELETE /rag/documents/{file_id}
    GET    /rag/documents
    POST   /rag/search
    GET    /knowledge/graph
    GET    /memory/history/retention
    POST   /memory/history/prune

Moved out of ``create_app`` unchanged (review item: split the god function); the route
table and OpenAPI schema are identical before and after. The write guard in ``main``
still gates the mutating routes.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from iris_harness.foundation.paths import repo_root
from iris_harness.runtime import IrisRuntime
from iris_harness.server.iris_api.runtime_access import runtime_or_503 as _runtime_or_503
from iris_harness.services.rag.documents import DocumentCatalog, document_catalog
from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.ingest import ingest_path
from iris_harness.services.rag.ingest_source import current_ingest_source
from iris_harness.services.rag.retrieve import search_documents
from iris_harness.services.rag.store import DocumentStore

REPO_ROOT = repo_root()
logger = logging.getLogger(__name__)


class RemovalTargetsRequest(BaseModel):
    """Things to remove from memory (ADR-0119): ``{"kind": ..., "id": ...}`` each."""

    targets: list[dict[str, Any]] = Field(default_factory=list)


class RemovalDeleteRequest(BaseModel):
    ids: list[str] = Field(default_factory=list)
    confirm: str = ""


class PruneHistoryRequest(BaseModel):
    """Request body for ``POST /memory/history/prune`` — reviewed entry ids to remove."""

    entry_ids: list[str] = Field(..., min_length=1, max_length=10000)


class ProfileWriteRequest(BaseModel):
    """Body for ``PUT /memory/profile`` — the curated head of USER.md."""

    profile: str = Field(..., max_length=100_000)


class FactCorrectRequest(BaseModel):
    """Body for ``PATCH /memory/facts/{key}`` — the owner's value for a fact."""

    value: str = Field(..., min_length=1, max_length=2_000)
    # Which value to replace when the key holds several (a fact id from GET
    # /memory/facts). Without it an edit of a card would be ADDED beside the others.
    id: str | None = Field(default=None, max_length=128)


class AckContradictionsRequest(BaseModel):
    """Request body for ``POST /memory/contradictions/ack`` — reviewed conflict ids."""

    ids: list[str] = Field(..., min_length=1, max_length=10000)


RAG_UPLOAD_DIR = Path(
    os.getenv("IRIS_RAG_UPLOAD_DIR", str(REPO_ROOT / "data" / "rag_uploads"))
).expanduser()


RAG_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


# The upload is read in pieces this size and refused as soon as it passes the cap, so
# an oversized file never sits in memory whole.
RAG_UPLOAD_CHUNK_BYTES = 1024 * 1024

# Room for the multipart envelope (boundaries, part headers, the filename) on top of
# the file itself, when judging a request by its Content-Length before parsing it.
RAG_UPLOAD_FORM_OVERHEAD_BYTES = 64 * 1024


def _upload_too_large(method: str, path: str, content_length: str | None) -> bool:
    """Whether a RAG upload declares a body that cannot fit under the cap.

    The route cannot refuse it itself: the form is parsed (and the file spooled to a
    temporary file) before the handler runs, so a 2 GB body would be received in full
    and only then turned away. A missing or unreadable header is left to the route's
    own capped read."""
    if method != "POST" or path != RAG_UPLOAD_PATH or not content_length:
        return False
    try:
        declared = int(content_length)
    except ValueError:
        return False
    return declared > RAG_MAX_UPLOAD_BYTES + RAG_UPLOAD_FORM_OVERHEAD_BYTES


RAG_UPLOAD_PATH = "/rag/upload"


RAG_SUPPORTED_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".text",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
    ".docx",
}


class RagSearchRequest(BaseModel):
    """Request body for ``POST /rag/search``."""

    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


def _rag_handles(app: FastAPI) -> tuple[DocumentStore, DocumentIndex]:
    """Build (once) and cache the RAG store and index.

    ``DocumentIndex`` spins up a persistent ChromaDB client, so we keep one set
    of handles on ``app.state`` rather than re-opening it per request.
    """
    handles: tuple[DocumentStore, DocumentIndex] | None = getattr(app.state, "rag_handles", None)
    if handles is None:
        store = DocumentStore()
        store.ensure_schema()
        handles = (store, DocumentIndex())
        app.state.rag_handles = handles
    return handles


def _rag_catalog(store: DocumentStore) -> DocumentCatalog:
    """The document list: a file domain's catalog when one registered, else RAG's own.

    Read per request, not cached with the handles: the plugin that registers a catalog
    is set up when the runtime is built, which may be after the first request.
    """
    return document_catalog(store)


# Cap auxiliary (fact/signal) nodes so a large memory store can't blow up the
# graph payload; the full counts are always reported in `stats`.
KNOWLEDGE_MAX_AUX = 200


def _kg_norm(text: str) -> str:
    return text.strip().lower()


def _kg_mentions(haystack: str, term: str) -> bool:
    """Whole-token match (boundaries) — 'ml' won't match 'html'. term lowercased."""
    if len(term) < 2:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack) is not None


def _wiki_tags(frontmatter: dict[str, object]) -> list[str]:
    raw = frontmatter.get("tags")
    if isinstance(raw, str):
        return [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(t).strip() for t in raw if str(t).strip()]
    return []


def _build_knowledge_graph(app: FastAPI) -> dict[str, Any]:
    """Assemble one node/edge graph unifying semantic memory (wiki + facts +
    signals) and the RAG document corpus. Tags are the cross-corpus bridge;
    [[wikilinks]] resolve across wiki AND documents by title. Pure enumeration —
    no embedding / Ollama calls."""
    rt = _runtime_or_503(app)
    store, _index = _rag_handles(app)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    title_map: dict[str, str] = {}  # normalized title/slug/stem -> node id
    tag_nodes: dict[str, str] = {}  # lowercased tag -> node id
    edge_seen: set[str] = set()

    def add_node(nid: str, kind: str, label: str, meta: dict[str, Any]) -> None:
        if nid in node_ids:
            return
        node_ids.add(nid)
        nodes.append({"id": nid, "kind": kind, "label": label, "meta": meta})

    def add_edge(src: str, tgt: str, kind: str) -> None:
        if src == tgt:
            return
        eid = f"{src}|{tgt}|{kind}"
        if eid in edge_seen:
            return
        edge_seen.add(eid)
        edges.append({"id": eid, "source": src, "target": tgt, "kind": kind})

    def tag_node(tag: str) -> str:
        key = _kg_norm(tag)
        nid = tag_nodes.get(key)
        if nid is None:
            nid = f"tag:{key}"
            tag_nodes[key] = nid
            add_node(nid, "tag", f"#{tag}", {})
        return nid

    # --- wiki pages (structured semantic memory) ---
    wiki_pages = []
    wiki = getattr(rt, "wiki", None)
    if wiki is not None:
        try:
            wiki_pages = wiki._pages.load_all()
        except Exception:  # a broken/empty wiki must not 500 the map
            logger.exception("knowledge graph: wiki enumeration failed")
            wiki_pages = []
    for p in wiki_pages:
        nid = f"wiki:{p.slug}"
        add_node(nid, "wiki", p.title, {"page_type": str(p.page_type), "slug": p.slug})
        title_map[_kg_norm(p.title)] = nid
        title_map[_kg_norm(p.slug)] = nid

    # --- RAG documents (corpus) ---
    sources = store.list_sources()
    for s in sources:
        nid = f"doc:{s.id}"
        label = s.title or Path(s.path).name
        add_node(nid, "document", label, {"kind": str(s.kind), "path": s.path})
        if s.title:
            title_map.setdefault(_kg_norm(s.title), nid)
        title_map.setdefault(_kg_norm(Path(s.path).stem), nid)

    # --- edges: wikilinks + tags (resolve across BOTH corpora) ---
    for p in wiki_pages:
        src = f"wiki:{p.slug}"
        for link in p.wikilinks:
            tgt = title_map.get(_kg_norm(link.split("|", 1)[0].split("#", 1)[0]))
            if tgt:
                add_edge(src, tgt, "link")
        for tag in _wiki_tags(p.frontmatter):
            add_edge(src, tag_node(tag), "tag")

    for s in sources:
        src = f"doc:{s.id}"
        for link in s.links:
            tgt = title_map.get(_kg_norm(link))
            if tgt:
                add_edge(src, tgt, "link")
        for tag in s.tags:
            add_edge(src, tag_node(tag), "tag")

    # --- facts + signals: attach to a tag when they mention it ---
    tag_keys = list(tag_nodes.keys())
    facts = rt.memory_store.fetch_all_user_facts()
    facts_unlinked = 0
    for f in facts[:KNOWLEDGE_MAX_AUX]:
        text = f"{f.key} {f.value}".lower()
        matched = [t for t in tag_keys if _kg_mentions(text, t)]
        if not matched:
            facts_unlinked += 1
            continue
        nid = f"fact:{f.key}"
        add_node(
            nid, "fact", f.key, {"value": f.value, "confidence": f.confidence, "source": f.source}
        )
        for t in matched:
            add_edge(nid, tag_nodes[t], "tag")

    signals = rt.memory_store.fetch_learning_signals()
    signals_unlinked = 0
    for sg in signals[:KNOWLEDGE_MAX_AUX]:
        text = f"{sg.query} {sg.domain} {sg.outcome}".lower()
        matched = [t for t in tag_keys if _kg_mentions(text, t)]
        if not matched:
            signals_unlinked += 1
            continue
        nid = f"signal:{sg.id}"
        add_node(
            nid,
            "signal",
            (sg.query[:48] or sg.domain),
            {"domain": sg.domain, "outcome": sg.outcome},
        )
        for t in matched:
            add_edge(nid, tag_nodes[t], "tag")

    degree: dict[str, int] = {}
    for e in edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1
    for n in nodes:
        n["degree"] = degree.get(n["id"], 0)

    def count(kind: str) -> int:
        return sum(1 for n in nodes if n["kind"] == kind)

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "wiki": count("wiki"),
            "documents": count("document"),
            "tags": len(tag_nodes),
            "facts": count("fact"),
            "signals": count("signal"),
            "edges": len(edges),
            "facts_total": len(facts),
            "facts_unlinked": facts_unlinked,
            "signals_total": len(signals),
            "signals_unlinked": signals_unlinked,
        },
    }


def _upload_limit_detail() -> str:
    return f"file exceeds {RAG_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"


def install_memory_routes(app: FastAPI, runtime: Callable[[], Any]) -> None:
    """Register these routes. ``runtime`` returns the live runtime or raises 503."""

    @app.get("/memory/contradictions")
    def memory_contradictions(include_acknowledged: bool = False) -> dict[str, Any]:
        """Detected same-key value conflicts for review (most recent first).

        Read-only. The web UI surfaces these so the user can correct the fact and/or
        acknowledge the conflict via POST /memory/contradictions/ack.
        """
        store = getattr(runtime(), "memory_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="memory store unavailable")
        rows = store.fetch_contradictions(include_acknowledged=include_acknowledged)
        return {
            "count": len(rows),
            "contradictions": [
                {
                    "id": c.id,
                    "key": c.key,
                    "stored_value": c.stored_value,
                    "stored_confidence": c.stored_confidence,
                    "incoming_value": c.incoming_value,
                    "incoming_confidence": c.incoming_confidence,
                    "resolution": c.resolution,
                    "source": c.source,
                    "detected_at": c.detected_at.isoformat(),
                    "seen_count": c.seen_count,
                }
                for c in rows
            ],
        }

    @app.post("/memory/contradictions/ack")
    def memory_contradictions_ack(request: AckContradictionsRequest) -> dict[str, Any]:
        """Mark conflicts reviewed (clears them from the default queue). Write-gated."""
        store = getattr(runtime(), "memory_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="memory store unavailable")
        acknowledged = store.acknowledge_contradictions(request.ids)
        return {"requested": len(request.ids), "acknowledged": acknowledged}

    # Owner-confirmed recall: the store is the truth, the queue is what it owes the
    # user. Registered BEFORE the greedy ``/memory/{session_id}`` route below.
    def _memory_store_or_503() -> Any:
        store = getattr(runtime(), "memory_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="memory store unavailable")
        return store

    def _fact_coordinator() -> Any:
        from iris_harness.memory.coordinator import FactCoordinator

        rt = runtime()
        return FactCoordinator(_memory_store_or_503(), getattr(rt, "semantic_index", None))

    @app.get("/memory/facts")
    def memory_facts(confirmed_only: bool = True) -> dict[str, Any]:
        """Stored facts. Confirmed ones are what actually reach a prompt."""
        facts = _memory_store_or_503().fetch_all_user_facts(confirmed_only=confirmed_only)
        return {
            "count": len(facts),
            "facts": [
                {
                    # The fact's own id: a key can hold several values (two cards),
                    # and Edit / Forget must name one of them, not the key.
                    "id": f.statement_id,
                    "key": f.key,
                    "value": f.value,
                    "confidence": f.confidence,
                    "source": f.source,
                    "confirmed": f.confirmed,
                    "first_seen": f.first_seen.isoformat(),
                    "last_confirmed": f.last_confirmed.isoformat(),
                    "times_confirmed": f.times_confirmed,
                }
                for f in facts
            ],
        }

    @app.get("/memory/review")
    def memory_review(limit: int = 100) -> dict[str, Any]:
        """The review queue: every proposed fact, newest first.

        Since memris plan PR 2c a fact stored before confirmation existed IS a proposal
        (a proposed statement), so there is one queue, not two.
        """
        store = _memory_store_or_503()
        proposals = store.fetch_fact_proposals(status="pending", limit=limit)
        return {
            "pending_count": store.count_pending_review(),
            "proposals": [
                {
                    "id": p.id,
                    "key": p.key,
                    "value": p.value,
                    "current_value": p.current_value,
                    "confidence": p.confidence,
                    "source": p.source,
                    "evidence": p.evidence,
                    "created_at": p.created_at.isoformat(),
                    "kind": "changed" if p.current_value else "new",
                    # Whom it is about when not the owner — one hop away (memris PR 3b).
                    "subject": p.subject,
                }
                for p in proposals
            ],
        }

    @app.post("/memory/review/{proposal_id}/approve")
    def memory_review_approve(proposal_id: str) -> dict[str, Any]:
        """Approve a proposal — it becomes a confirmed fact everywhere. Write-gated."""
        proposal = _memory_store_or_503().fetch_fact_proposal(proposal_id)
        fact = _fact_coordinator().approve_proposal(proposal_id)
        if fact is None:
            raise HTTPException(status_code=404, detail="no pending proposal with that id")
        subject = proposal.subject if proposal is not None else None
        return {"approved": True, "key": fact.key, "value": fact.value, "subject": subject}

    @app.post("/memory/review/{proposal_id}/reject")
    def memory_review_reject(proposal_id: str) -> dict[str, Any]:
        """Reject a proposal. Nothing is written to the fact store. Write-gated."""
        if not _fact_coordinator().reject_proposal(proposal_id):
            raise HTTPException(status_code=404, detail="no pending proposal with that id")
        return {"rejected": True, "id": proposal_id}

    # Look-alike entities (ADR-0115 decision 4): pairs of similar names kept apart until
    # someone decides. Chat asks about one per conversation; these are the rest.
    def _entity_decision_json(graph: Any, d: Any) -> dict[str, Any]:
        def label(entity_id: str) -> str:
            entity = graph.get_entity(entity_id)
            return entity.label if entity is not None else entity_id

        return {
            "id": d.id,
            "decision": d.decision,
            "a": {"id": d.a, "label": label(d.a)},
            "b": {"id": d.b, "label": label(d.b)},
            "score": d.score,
            "evidence_count": len(d.evidence),
            "decided_by": d.decided_by,
            "decided_at": d.decided_at.isoformat(),
            "asked_at": d.asked_at.isoformat() if d.asked_at else None,
        }

    # Learned vocabulary (ADR-0115 decision 7): words memory picked up on its own, as
    # rows in the memory database. Reject = never learned again; promote = a YAML snippet
    # for a human to commit (the runtime never writes config).
    def _term_json(t: Any) -> dict[str, Any]:
        return {
            "name": t.name,
            "kind": t.kind,
            "label": t.label,
            "status": t.status,
            "alias_of": t.alias_of,
            "observations": t.observations,
            "conversations": len(t.episodes),
            "examples": list(t.examples),
            "first_seen": t.first_seen.isoformat(),
            "last_seen": t.last_seen.isoformat(),
            "activated_at": t.activated_at.isoformat() if t.activated_at else None,
            "decided_by": t.decided_by,
        }

    @app.get("/memory/terms")
    def memory_terms(status: str | None = None) -> dict[str, Any]:
        """Learned terms — all, or one status (candidate, alias, active, rejected)."""
        terms = _memory_store_or_503().vocabulary().terms(status)
        return {"count": len(terms), "terms": [_term_json(t) for t in terms]}

    @app.post("/memory/terms/{name}/reject")
    def memory_term_reject(name: str) -> dict[str, Any]:
        """Never learn this word; what was said with it leaves memory. Write-gated."""
        rejected = _memory_store_or_503().vocabulary().reject(name)
        if rejected is None:
            raise HTTPException(status_code=404, detail="no learned term by that name")
        return _term_json(rejected)

    @app.post("/memory/terms/{name}/activate")
    def memory_term_activate(name: str) -> dict[str, Any]:
        """Learn this word now, without waiting for it to recur. Write-gated."""
        active = _memory_store_or_503().vocabulary().activate(name)
        if active is None:
            raise HTTPException(status_code=404, detail="no candidate term by that name")
        return _term_json(active)

    @app.get("/memory/terms/{name}/promotion")
    def memory_term_promotion(name: str) -> dict[str, Any]:
        """The YAML that would declare this term, for a human to review and commit."""
        from iris_harness.memory.vocabulary import promotion

        vocabulary = _memory_store_or_503().vocabulary()
        wanted = vocabulary.qualified(name)
        term = next((t for t in vocabulary.terms() if t.name == wanted), None)
        if term is None or term.status == "alias":
            raise HTTPException(status_code=404, detail="no learned term to promote")
        return {"name": term.name, "yaml": promotion(term, vocabulary.graph.declared)}

    @app.get("/memory/entities")
    def memory_entities(decision: str = "candidate") -> dict[str, Any]:
        """Entity decisions: open look-alikes by default; ``decision=all`` for every one."""
        graph = _memory_store_or_503().memory_graph()
        rows = graph.open_candidates() if decision == "candidate" else graph.decisions()
        if decision not in ("candidate", "all"):
            rows = [d for d in rows if d.decision == decision]
        return {"count": len(rows), "decisions": [_entity_decision_json(graph, d) for d in rows]}

    def _decide_entities(decision_id: str, action: str) -> dict[str, Any]:
        from memris.graph import StatementError

        graph = _memory_store_or_503().memory_graph()
        try:
            if action == "same":
                done = graph.accept_candidate(decision_id, decided_by="owner")
            elif action == "distinct":
                done = graph.reject_candidate(decision_id, decided_by="owner")
            else:
                done = graph.unmerge(decision_id, decided_by="owner")
        except StatementError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _entity_decision_json(graph, done)

    @app.post("/memory/entities/{decision_id}/same")
    def memory_entities_same(decision_id: str) -> dict[str, Any]:
        """A look-alike pair is one thing: merge it (the older name kept). Write-gated."""
        return _decide_entities(decision_id, "same")

    @app.post("/memory/entities/{decision_id}/distinct")
    def memory_entities_distinct(decision_id: str) -> dict[str, Any]:
        """A look-alike pair is two things: never proposed again. Write-gated."""
        return _decide_entities(decision_id, "distinct")

    @app.post("/memory/entities/{decision_id}/unmerge")
    def memory_entities_unmerge(decision_id: str) -> dict[str, Any]:
        """Undo a merge: both names stand alone again. Write-gated."""
        return _decide_entities(decision_id, "unmerge")

    @app.post("/memory/facts/{key}/confirm")
    def memory_fact_confirm(key: str) -> dict[str, Any]:
        """Confirm a stored fact (how the pre-review rows are cleared). Write-gated."""
        fact = _fact_coordinator().confirm(key)
        if fact is None:
            raise HTTPException(status_code=404, detail="no fact with that key")
        return {"confirmed": True, "key": fact.key, "value": fact.value}

    @app.get("/memory/lessons")
    def memory_lessons(status: str = "pending", limit: int = 100) -> dict[str, Any]:
        """Lessons waiting for review. Approving one writes a behavior file."""
        proposals = _memory_store_or_503().fetch_lesson_proposals(status=status, limit=limit)
        return {
            "count": len(proposals),
            "lessons": [
                {
                    "id": p.id,
                    "trigger": p.trigger,
                    "lesson": p.lesson,
                    "evidence": p.evidence,
                    "source": p.source,
                    "created_at": p.created_at.isoformat(),
                }
                for p in proposals
            ],
        }

    @app.post("/memory/lessons/{proposal_id}/approve")
    def memory_lesson_approve(proposal_id: int) -> dict[str, Any]:
        """Approve a lesson — it becomes a behavior, matched from the next turn on."""
        from iris_harness.memory.lessons import LessonCurator

        name = LessonCurator(_memory_store_or_503()).approve(proposal_id)
        if name is None:
            raise HTTPException(status_code=404, detail="no pending lesson with that id")
        return {"approved": True, "behavior": name}

    @app.post("/memory/lessons/{proposal_id}/reject")
    def memory_lesson_reject(proposal_id: int) -> dict[str, Any]:
        """Reject a lesson. Nothing is written. Write-gated."""
        from iris_harness.memory.lessons import LessonCurator

        if not LessonCurator(_memory_store_or_503()).reject(proposal_id):
            raise HTTPException(status_code=404, detail="no pending lesson with that id")
        return {"rejected": True, "id": proposal_id}

    @app.post("/memory/review/expire")
    def memory_review_expire(older_than_days: int = 30) -> dict[str, Any]:
        """Expire proposals nobody reviewed. Write-gated."""
        store = _memory_store_or_503()
        expired = store.expire_fact_proposals(older_than_days=older_than_days)
        expired_lessons = store.expire_lesson_proposals(older_than_days=older_than_days)
        return {
            "expired": expired,
            "expired_lessons": expired_lessons,
            "older_than_days": older_than_days,
        }

    def _retention_or_503() -> Any:
        service = getattr(runtime(), "retention", None)
        if service is None:
            raise HTTPException(status_code=503, detail="retention service unavailable")
        return service

    @app.get("/memory/profile")
    def memory_profile() -> dict[str, Any]:
        """The hand-curated head of USER.md — what the owner wrote, not the auto block."""
        from iris_harness.memory.identity import load_curated_profile

        text = load_curated_profile()
        return {"profile": text, "chars": len(text)}

    @app.put("/memory/profile")
    def memory_profile_write(request: ProfileWriteRequest) -> dict[str, Any]:
        """Replace the curated head of USER.md. The auto-detected block is untouched."""
        from iris_harness.memory.identity import write_curated_profile

        path = write_curated_profile(request.profile)
        return {"written": True, "path": str(path), "chars": len(request.profile)}

    @app.post("/memory/facts/{key}/forget")
    def memory_fact_forget(key: str, id: str | None = None) -> dict[str, Any]:
        """Forget a fact — store, index and USER.md together. Reversible via /restore.

        ``id`` (a fact id from GET /memory/facts) forgets that one value of a key that
        holds several (one of two cards); without it, every value of the key goes.
        """
        if not _fact_coordinator().forget(key, statement_id=id):
            raise HTTPException(status_code=404, detail="no fact with that key")
        return {"forgotten": True, "key": key, "id": id}

    @app.post("/memory/facts/{key}/restore")
    def memory_fact_restore(key: str) -> dict[str, Any]:
        """Restore a fact's previous value from its history."""
        from iris_harness.memory.fact_statements import FactKeyError

        try:
            restored = _fact_coordinator().restore(key)
        except FactKeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if restored is None:
            raise HTTPException(status_code=404, detail="nothing to restore for that key")
        return {"restored": True, "key": key, "value": restored}

    @app.patch("/memory/facts/{key}")
    def memory_fact_correct(key: str, request: FactCorrectRequest) -> dict[str, Any]:
        """Set a fact's value. An owner edit is confirmed by definition."""
        from iris_harness.memory.fact_statements import FactKeyError

        coordinator = _fact_coordinator()
        try:
            coordinator.correct(key, request.value, statement_id=request.id)
        except FactKeyError as exc:  # the vocabulary is closed (ADR-0115 decision 6)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        coordinator.confirm(key)
        return {"updated": True, "key": key, "value": request.value}

    @app.get("/memory/facts/{key}/history")
    def memory_fact_history(key: str, limit: int = 50) -> dict[str, Any]:
        """How a fact's value changed, newest first."""
        entries = _memory_store_or_503().fetch_fact_history(key, limit=limit)
        return {
            "key": key,
            "count": len(entries),
            "history": [
                {
                    "id": e.id,
                    "old_value": e.old_value,
                    "new_value": e.new_value,
                    "reason": e.reason,
                    "changed_at": e.changed_at.isoformat(),
                }
                for e in entries
            ],
        }

    @app.get("/memory/review/test-sessions")
    def memory_review_test_sessions() -> dict[str, Any]:
        """Conversations whose id looks like a test run, for a one-time bulk removal.

        Read-only. ``retention.yaml``'s ``test_session_review`` says what a real id looks
        like; everything else is offered here. Removing goes through
        ``POST /memory/removed`` (kind ``session``), so it is listed and restorable in
        the Removed tab (ADR-0119, Map cleanup plan decision 7).
        """
        from iris_harness.memory.retention import (
            flag_test_sessions,
            real_session_shapes,
        )

        store = _memory_store_or_503()
        return {
            "sessions": [f.as_dict() for f in flag_test_sessions(store)],
            "real_shapes": real_session_shapes(),
        }

    @app.get("/memory/sessions")
    def memory_sessions(limit: int = 100) -> dict[str, Any]:
        """Stored conversations: summary, size and whether the text is still kept."""
        from iris_harness.memory.retention import is_ephemeral_session, retention_config

        store = _memory_store_or_503()
        hot_days = int((retention_config().get("conversations") or {}).get("hot_days", 90))
        cutoff = datetime.now(UTC) - timedelta(days=hot_days)
        rows = []
        removed = store.removed_session_ids()
        for session_id, last_ts, turns in store.session_activity():
            if session_id in removed:  # the Removed tab lists it (ADR-0119)
                continue
            try:
                last = datetime.fromisoformat(last_ts)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=UTC)
            except ValueError:
                last = datetime.now(UTC)
            rows.append(
                {
                    "session_id": session_id,
                    "last_activity": last.isoformat(),
                    "turns": turns,
                    "summary": store.load_conversation_summary(session_id),
                    "state": "hot" if last > cutoff else "cold-eligible",
                    "is_run": is_ephemeral_session(session_id),
                }
            )
        rows.sort(key=lambda r: str(r["last_activity"]), reverse=True)
        return {"count": len(rows), "hot_days": hot_days, "sessions": rows[:limit]}

    @app.get("/memory/context/{session_id}")
    def memory_context_inspect(session_id: str, message: str | None = None) -> dict[str, Any]:
        """What the next turn in this session would carry, block by block.

        The anti-"shipped unwired" view: every L0 block and L1 pointer with its token
        cost, so a block that is not reaching the model is visible rather than assumed.
        The memory-graph block depends on what the message names, so it is computed for
        ``message`` — by default the session's last user message (memris PR 6).
        """
        rt = runtime()
        sessions = getattr(rt, "sessions", None)
        if sessions is None:
            raise HTTPException(status_code=503, detail="session memory unavailable")
        from iris_harness.llm.budget import estimate_tokens

        if message is None:
            history = getattr(sessions, "conversations", {}).get(session_id, [])
            said = [t.content for t in history if getattr(t, "role", "") == "user"]
            message = said[-1] if said else ""
        ctx = sessions.build_memory_context(message, session_id=session_id, intent="general")
        blocks = [
            ("SOUL (core)", ctx.soul or ""),
            ("USER.md (curated)", ctx.user_profile or ""),
            ("confirmed facts", ", ".join(f"{f.key}={f.value}" for f in ctx.user_facts)),
            ("memory graph (names in the message)", ctx.linked or ""),
            ("active items", ctx.active or ""),
            ("episodic digest", ctx.episodic_digest or ""),
            (f"behavior: {ctx.behavior_name or '—'}", ctx.behavior or ""),
            ("session summary", ctx.summary or ""),
            ("from earlier sessions", "\n".join(ctx.related_turns)),
            ("recent turns", "\n".join(ctx.recent_turns)),
        ]
        rendered: list[dict[str, Any]] = [
            {
                "block": name,
                "tokens": estimate_tokens(text),
                "present": bool(text.strip()),
                "preview": " ".join(text.split())[:240],
            }
            for name, text in blocks
        ]
        return {
            "session_id": session_id,
            "message": message,
            "blocks": rendered,
            "pointers": list(ctx.pointers),
            "total_tokens": sum(estimate_tokens(text) for _name, text in blocks),
            "compaction_in_flight": sessions.compaction_in_flight(session_id),
        }

    @app.post("/memory/export")
    def memory_export(out_dir: str, include_unconfirmed: bool = False) -> dict[str, Any]:
        """Export the memory graph as a linked markdown vault (Obsidian-compatible).

        One way: nothing in the vault is read back. Write-gated, and the response
        repeats where the data landed — the folder holds personal memory, and moving it
        to a cloud notebook is an egress decision the caller makes knowingly.
        """
        from pathlib import Path as _Path

        from iris_harness.memory.export import export_memory_vault

        target = _Path(out_dir).expanduser()
        if not target.is_absolute():
            raise HTTPException(status_code=400, detail="out_dir must be an absolute path")
        result = export_memory_vault(
            _memory_store_or_503(), target, include_unconfirmed=include_unconfirmed
        )
        payload = result.as_dict()
        payload["note"] = (
            "This folder contains personal memory. An Obsidian vault stays on this "
            "machine; uploading it to a cloud notebook does not."
        )
        return payload

    @app.get("/memory/graph")
    def memory_graph(
        focus: str | None = None,
        depth: int = 1,
        kinds: str | None = None,
        confirmed_only: bool = False,
        cap: int = 150,
        as_of: datetime | None = None,
    ) -> dict[str, Any]:
        """The memory graph — computed on request, never stored.

        Opens on "You"; pass ``focus=<node id>`` to expand one node's neighbours. What
        the cap leaves out comes back as a "+N more" node rather than vanishing.
        ``as_of`` (ISO-8601) draws what memory held true at that moment.
        """
        from iris_harness.memory.graph import build_memory_graph

        if as_of is not None and as_of.tzinfo is None:
            raise HTTPException(status_code=422, detail="as_of needs a timezone (e.g. …Z)")
        kind_set = {k.strip() for k in kinds.split(",") if k.strip()} if kinds else None
        return build_memory_graph(
            _memory_store_or_503(),
            focus=focus,
            depth=max(1, min(depth, 3)),
            kinds=kind_set,
            confirmed_only=confirmed_only,
            node_cap=max(10, min(cap, 500)),
            as_of=as_of,
        )

    @app.get("/memory/housekeeping")
    def memory_housekeeping_runs(limit: int = 20) -> dict[str, Any]:
        """What the daily retention pass has done. Read-only."""
        runs: list[dict[str, Any]] = _retention_or_503().history(limit=limit)
        return {"runs": runs}

    @app.post("/memory/housekeeping/run")
    def memory_housekeeping_run(dry_run: bool = False) -> dict[str, Any]:
        """Run the retention pass now. ``dry_run`` reports without deleting. Write-gated."""
        report: dict[str, Any] = _retention_or_503().run(dry_run=dry_run).as_dict()
        return report

    @app.get("/logs/archive")
    def logs_archive() -> dict[str, Any]:
        """The encrypted session-log archive, month by month. Read-only."""
        result: dict[str, Any] = _retention_or_503().archived_logs()
        return result

    @app.post("/logs/archive/restore")
    def logs_archive_restore(
        session: str | None = None, month: str | None = None
    ) -> dict[str, Any]:
        """Restore one session's log (``session``) or a whole month (``month=YYYY-MM``)
        from the archive into the live log directory. Write-gated. Anything but
        exactly one of the two is a 400 (the archive refuses it)."""
        try:
            restored = _retention_or_503().restore_logs(session=session, month=month)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"restored": restored, "count": len(restored)}

    @app.post("/memory/forget")
    def memory_forget(needle: str, confirm: bool = False) -> dict[str, Any]:
        """Forget everything matching ``needle``.

        Without ``confirm`` this only PREVIEWS what would go — turns, summaries and
        facts — so the caller can show it before anything is deleted. Write-gated.
        """
        if not needle.strip():
            raise HTTPException(status_code=400, detail="needle must not be empty")
        result: dict[str, Any] = _retention_or_503().forget_matching(needle, confirm=confirm)
        return result

    @app.post("/memory/sessions/purge")
    def memory_sessions_purge(
        session_ids: str | None = None, confirm: bool = False
    ) -> dict[str, Any]:
        """Purge sessions entirely; defaults to every playground / eval / test session.

        Without ``confirm`` it lists what would go. Write-gated.
        """
        ids = [s.strip() for s in session_ids.split(",")] if session_ids else None
        purged: dict[str, Any] = _retention_or_503().purge_sessions(ids, confirm=confirm)
        return purged

    # -- Removed (ADR-0119): reversible removal from memory -----------------------
    # Registered BEFORE the greedy ``/memory/{session_id}`` route.

    def _removal() -> Any:
        from iris_harness.memory.removal import MemoryRemoval

        store = _memory_store_or_503()

        def purge_session(session_id: str) -> None:
            _retention_or_503().purge_sessions([session_id], confirm=True)
            store.delete_conversation_summary(session_id)  # a cooled session has no turns

        return MemoryRemoval(
            store, rederive=_fact_coordinator().rederive, purge_session=purge_session
        )

    def _removal_targets(request: RemovalTargetsRequest) -> list[Any]:
        from iris_harness.memory.removal import RemovalError, Target

        if not request.targets:
            raise HTTPException(status_code=400, detail="targets must not be empty")
        try:
            return [Target.of(t) for t in request.targets]
        except RemovalError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def _removal_call(fn: Callable[[], Any]) -> Any:
        from iris_harness.memory.removal import NotFoundError, RemovalError

        try:
            return fn()
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RemovalError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/memory/removed")
    def memory_removed() -> dict[str, Any]:
        """The Removed list: every removal and every forgotten fact, newest first."""
        return {"items": _removal().items()}

    @app.post("/memory/removed/preview")
    def memory_removed_preview(request: RemovalTargetsRequest) -> dict[str, Any]:
        """What removing each target would do — nothing is changed, so it is ungated."""
        targets = _removal_targets(request)
        return {"effects": _removal_call(lambda: _removal().preview(targets))}

    @app.post("/memory/removed")
    def memory_removed_add(request: RemovalTargetsRequest) -> dict[str, Any]:
        """Remove each target from memory — the Map, recall and the prompt together."""
        targets = _removal_targets(request)
        return {"items": _removal_call(lambda: _removal().remove(targets))}

    @app.post("/memory/removed/delete")
    def memory_removed_delete(request: RemovalDeleteRequest) -> dict[str, Any]:
        """Delete removed things for good; ``confirm`` must be the word ``delete``."""
        result: dict[str, Any] = _removal_call(
            lambda: _removal().delete(request.ids, confirm=request.confirm)
        )
        return result

    @app.post("/memory/removed/{removal_id}/restore")
    def memory_removed_restore(removal_id: str) -> dict[str, Any]:
        """Undo one removal exactly."""
        return {"restored": _removal_call(lambda: _removal().restore(removal_id))}

    @app.get("/memory/{session_id}")
    def get_memory(session_id: str) -> dict[str, Any]:
        rt: IrisRuntime | None = app.state.runtime
        if rt is None:
            raise HTTPException(status_code=503, detail="runtime unavailable")
        turns = rt.sessions.conversations.get(session_id, [])
        return {
            "session_id": session_id,
            "turn_count": len(turns),
            "turns": [{"role": t.role, "content": t.content} for t in turns],
        }

    @app.post(RAG_UPLOAD_PATH)
    async def rag_upload(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        """Save an uploaded document and index it.

        Async only for the capped read. Writing the file and indexing it are blocking
        (disk, SQLite, embeddings) and run in the thread pool: on the event loop they
        stalled every other request, streaming chat included, for as long as a large
        PDF took to index."""
        store, index = _rag_handles(app)

        name = Path(file.filename or "").name
        if not name:
            raise HTTPException(status_code=400, detail="missing filename")
        suffix = Path(name).suffix.lower()
        if suffix not in RAG_SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported file type '{suffix}'. supported: "
                + ", ".join(sorted(RAG_SUPPORTED_SUFFIXES)),
            )

        # Read in pieces and stop one piece past the cap: the whole-body read this
        # replaces held any size of upload in memory before checking it.
        chunks: list[bytes] = []
        received = 0
        while chunk := await file.read(RAG_UPLOAD_CHUNK_BYTES):
            received += len(chunk)
            if received > RAG_MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=_upload_limit_detail())
            chunks.append(chunk)
        if not received:
            raise HTTPException(status_code=400, detail="empty file")
        raw = b"".join(chunks)

        def save_and_ingest() -> tuple[Path, Any, Any]:
            RAG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            dest = (RAG_UPLOAD_DIR / name).resolve()
            try:
                dest.write_bytes(raw)
            except OSError as exc:
                raise HTTPException(
                    status_code=500, detail=f"could not save upload: {exc}"
                ) from exc
            try:
                result = ingest_path(dest, store=store, index=index, source=current_ingest_source())
            except Exception as exc:  # surface ingest failure to the client
                logger.exception("rag ingest failed for %s", dest)
                raise HTTPException(status_code=500, detail=f"ingest failed: {exc}") from exc
            doc = next(
                (d for d in _rag_catalog(store).list_documents() if d.storage_path == str(dest)),
                None,
            )
            return dest, result, doc

        _dest, result, doc = await run_in_threadpool(save_and_ingest)
        return {
            "filename": name,
            "index_ready": index.is_ready,
            "sources_added": result.sources_added,
            "sources_updated": result.sources_updated,
            "sources_skipped": result.sources_skipped,
            "sources_denied": result.sources_denied,
            "chunks_indexed": result.chunks_indexed,
            "summary": result.summary(),
            "document": doc.to_payload() if doc is not None else None,
        }

    @app.delete("/rag/documents/{file_id}")
    def rag_delete(file_id: str) -> dict[str, Any]:
        store, index = _rag_handles(app)
        catalog = _rag_catalog(store)
        entry = catalog.get_document(file_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"rag document '{file_id}' not found")

        path = entry.storage_path
        # De-index everywhere (vector index + chunk store).
        source = next((s for s in store.list_sources() if s.path == path), None)
        if source is not None:
            index.delete_source(source.id)
            store.delete_source(source.id)
        catalog.forget_document(file_id)

        # ADR-0067: only unlink on-disk bytes IRIS itself created (uploads under
        # RAG_UPLOAD_DIR). A file indexed in place (iCloud / os_local) is the
        # user's — de-index only, never delete a file we don't own.
        file_removed = False
        try:
            resolved = Path(path).resolve()
            if resolved.is_relative_to(RAG_UPLOAD_DIR.resolve()) and resolved.exists():
                resolved.unlink()
                file_removed = True
        except OSError:
            logger.exception("rag delete: could not remove file %s", path)

        return {
            "deleted": True,
            "file_id": file_id,
            "de_indexed": source is not None,
            "file_removed": file_removed,
        }

    @app.get("/rag/documents")
    def rag_documents() -> dict[str, Any]:
        store, index = _rag_handles(app)
        entries = _rag_catalog(store).list_documents()
        return {
            "count": len(entries),
            "index_ready": index.is_ready,
            "documents": [e.to_payload() for e in entries],
        }

    @app.post("/rag/search")
    def rag_search(request: RagSearchRequest) -> dict[str, Any]:
        store, index = _rag_handles(app)
        hits = search_documents(request.query, store=store, index=index, limit=request.limit)
        return {
            "count": len(hits),
            "index_ready": index.is_ready,
            "results": [
                {
                    "text": h.text,
                    "score": h.score,
                    "citation": h.citation.label(),
                    "page": h.citation.page,
                }
                for h in hits
            ],
        }

    @app.get("/knowledge/graph")
    def knowledge_graph() -> dict[str, Any]:
        return _build_knowledge_graph(app)

    @app.get("/memory/history/retention")
    def memory_history_retention(older_than_days: int = 180, limit: int = 200) -> dict[str, Any]:
        """Fact-history entries older than ``older_than_days`` — the human review queue.

        Read-only (never deletes). Oldest first. The web UI surfaces these so the user
        can choose which to prune via POST /memory/history/prune.
        """
        store = getattr(runtime(), "memory_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="memory store unavailable")
        entries = store.fetch_history_retention_candidates(
            older_than_days=max(0, older_than_days), limit=max(1, min(limit, 2000))
        )
        return {
            "older_than_days": older_than_days,
            "total": store.count_fact_history(),
            "count": len(entries),
            "entries": [
                {
                    "id": e.id,
                    "key": e.key,
                    "old_value": e.old_value,
                    "new_value": e.new_value,
                    "source": e.source,
                    "reason": e.reason,
                    "changed_at": e.changed_at.isoformat(),
                }
                for e in entries
            ],
        }

    @app.post("/memory/history/prune")
    def memory_history_prune(request: PruneHistoryRequest) -> dict[str, Any]:
        """Remove the reviewed history entries by id (terminal). Write-gated.

        The user must have surfaced + chosen these via the retention review; this is the
        human-actioned outcome, mirroring `iris facts prune`.
        """
        store = getattr(runtime(), "memory_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="memory store unavailable")
        removed = store.prune_history_entries(request.entry_ids)
        return {"requested": len(request.entry_ids), "removed": removed}
