"""The self-learning loop's API: experiments, proposals, metrics, flags.

    GET    /learning/experiments
    GET    /learning/intelligence
    GET    /learning/analysis
    POST   /learning/recommendations/{index}/promote
    GET    /learning/behaviors
    POST   /learning/behaviors/{pattern_id}/approve
    POST   /learning/behaviors/{pattern_id}/reject
    GET    /learning/signals
    GET    /learning/intentions
    POST   /learning/intentions/{intention_id}/approve
    POST   /learning/intentions/{intention_id}/dismiss
    GET    /learning/proposal-quality
    GET    /learning/behaviors/preview
    GET    /learning/intentions/preview
    GET    /learning/flags
    POST   /learning/flags/{name}
    POST   /learning/flags/{name}/run

Moved out of ``create_app`` unchanged (review item: split the god function). The
capability is the runtime's learning service; these routes, the ``learning_*`` ReAct
tools and ``iris learning`` are surfaces. The write guard in ``main`` gates the
mutating ones, as before.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.server.iris_api.settings_routes import actor_of


class LearningFlagRequest(BaseModel):
    """Body for ``POST /learning/flags/{name}`` — hot-toggle a learning capability."""

    enabled: bool


def install_learning_routes(app: FastAPI, runtime: Callable[[], Any]) -> None:
    """Register the learning routes. ``runtime`` returns the live runtime or raises 503."""

    @app.get("/learning/experiments")
    def learning_experiments() -> dict[str, Any]:
        """Self-learning experiments (fast-follow #4): each carries a hypothesis,
        a baseline vs current metric, and a lifecycle status (pending/running/
        kept/discarded/...). Surfaces existing telemetry; pure read."""
        runtime = getattr(app.state, "runtime", None)
        store = getattr(runtime, "learning_store", None)
        if store is None:
            return {"experiments": [], "status_counts": {}}

        def _iso(value: Any) -> str | None:
            return value.isoformat() if value is not None else None

        experiments = [
            {
                "id": e.id,
                "domain": e.domain,
                "hypothesis": e.hypothesis,
                "variant_description": e.variant_description,
                "baseline_metric": e.baseline_metric,
                "current_metric": e.current_metric,
                "status": e.status,
                "created_at": _iso(e.created_at),
                "started_at": _iso(e.started_at),
                "evaluated_at": _iso(e.evaluated_at),
                "evaluation_window_hours": e.evaluation_window_hours,
                # Sandbox pre-flight verdict (ADR-0070), when one has run.
                "sandbox": getattr(e, "config_changes", {}).get("__sandbox__"),
            }
            for e in store.list_experiments()
        ]
        return {"experiments": experiments, "status_counts": store.experiment_status_counts()}

    @app.get("/learning/intelligence")
    def learning_intelligence() -> dict[str, Any]:
        """Measured learning-intelligence snapshot (ADR-0069 #4, slice 1).

        Deterministic aggregates over the existing self-learning signals: signal
        accuracy/integrity (escalation-judge precision, drop rate) + a per-(intent,
        tier) outcome matrix (completion / correction / reuse / cost). Pure read,
        no LLM; the agentic analyst interprets these numbers, not this endpoint."""
        from iris_harness.services.learning.intelligence import build_intelligence

        runtime = getattr(app.state, "runtime", None)
        store = getattr(runtime, "learning_store", None)
        if store is None:
            return {"available": False}
        report = build_intelligence(store)
        return {"available": True, **report.as_dict()}

    @app.get("/learning/analysis")
    def learning_analysis() -> dict[str, Any]:
        """Latest learning-analyst recommendations (ADR-0069 #4, slice 2).

        Returns the most recent advisory analysis persisted by the opt-in
        learning_analysis heartbeat (summary + ranked recommendations). Pure read;
        does not run the LLM. {available: false} when the analyst hasn't run."""
        runtime = getattr(app.state, "runtime", None)
        store = getattr(runtime, "learning_store", None)
        if store is None:
            return {"available": False}
        payload = store.latest_analysis()
        if not payload:
            return {"available": False}
        return {"available": True, **payload}

    @app.post("/learning/recommendations/{index}/promote")
    def promote_recommendation_endpoint(index: int) -> dict[str, Any]:
        """Promote recommendation #index to a tracked experiment (ADR-0069 #4 s3).

        HITL loop-closer: captures the recommendation's measured baseline and starts
        re-measuring it. Does NOT apply the change. Write-gated by IRIS_WEBUI_ALLOW_WRITES
        (see _is_gated_write). Returns the created experiment, or 404 if no such rec."""
        from iris_harness.services.learning.promote import promote_recommendation

        runtime = getattr(app.state, "runtime", None)
        store = getattr(runtime, "learning_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="learning store unavailable")
        result = promote_recommendation(store, index=index)
        if result is None:
            raise HTTPException(status_code=404, detail=f"no recommendation #{index} to promote")
        exp = result.experiment
        return {
            "experiment_id": exp.id,
            "hypothesis": exp.hypothesis,
            "baseline_metric": exp.baseline_metric,
            "measurable": result.measurable,
            "note": result.note,
        }

    # ------------------------------------------------------------------
    # Digital-twin review surfaces (HITL). Three layers, all propose-only:
    #   1. behaviors — mined recurring habits (-> episodic memory on approve)
    #   2. signals   — how the user steers the assistant (read-only ground truth)
    #   3. intentions — rolled-up longitudinal goals (-> ACTIVE identity on approve)
    # Reads are open; the approve/reject/dismiss writes are gated by
    # IRIS_WEBUI_ALLOW_WRITES (see _is_gated_write). Logic mirrors the
    # `iris behaviors` / `iris intentions` CLIs — the web UI is a pure renderer.
    # ------------------------------------------------------------------

    def _learning_store_or_503() -> Any:
        store = getattr(getattr(app.state, "runtime", None), "learning_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="learning store unavailable")
        return store

    @app.get("/learning/behaviors")
    def learning_behaviors(status: str = "pending") -> dict[str, Any]:
        """Mined behavior-pattern proposals awaiting review (digital-twin layer 1).

        Read-only. Approve via POST /learning/behaviors/{id}/approve (appends to
        episodic memory); reject via .../reject. status: pending | approved | rejected."""
        store = getattr(getattr(app.state, "runtime", None), "learning_store", None)
        if store is None:
            return {"status": status, "count": 0, "behaviors": []}
        rows = store.list_behavior_proposals(status=status)
        return {
            "status": status,
            "count": len(rows),
            "behaviors": [
                {
                    "pattern_id": b.pattern_id,
                    "text": b.text,
                    "confidence": b.confidence,
                    "evidence": list(b.evidence),
                    "status": b.status,
                    "created_at": b.created_at,
                }
                for b in rows
            ],
        }

    @app.post("/learning/behaviors/{pattern_id}/approve")
    def learning_behaviors_approve(pattern_id: str) -> dict[str, Any]:
        """Approve a mined habit: append it to durable episodic memory and clear the
        proposal. Write-gated. Returns the approved text, or 404 if not pending."""
        from iris_harness.memory.identity.loader import append_episodic_pattern

        store = _learning_store_or_503()
        proposal = store.get_behavior_proposal(pattern_id)
        if proposal is None or proposal.status != "pending":
            raise HTTPException(status_code=404, detail=f"no pending behavior: {pattern_id}")
        append_episodic_pattern(proposal.text)
        store.resolve_behavior_proposal(pattern_id, "approved")
        store.record_user_behavior_signal(
            "pattern_confirmed", subject=proposal.text, detail=proposal.confidence
        )
        return {"pattern_id": pattern_id, "status": "approved", "text": proposal.text}

    @app.post("/learning/behaviors/{pattern_id}/reject")
    def learning_behaviors_reject(pattern_id: str) -> dict[str, Any]:
        """Reject a proposed habit (won't be re-proposed). Write-gated."""
        store = _learning_store_or_503()
        proposal = store.get_behavior_proposal(pattern_id)
        if not store.resolve_behavior_proposal(pattern_id, "rejected"):
            raise HTTPException(status_code=404, detail=f"no behavior: {pattern_id}")
        store.record_user_behavior_signal(
            "pattern_dismissed", subject=proposal.text if proposal else pattern_id
        )
        return {"pattern_id": pattern_id, "status": "rejected"}

    @app.get("/learning/signals")
    def learning_signals(kind: str | None = None, limit: int = 100) -> dict[str, Any]:
        """Recent user-behavior signals — how the user steers the assistant (layer 2).

        Read-only ground truth (corrections, dismissals, confirmations). Optionally
        filter by kind; `summary` is the count by kind across all signals."""
        store = getattr(getattr(app.state, "runtime", None), "learning_store", None)
        if store is None:
            return {"count": 0, "signals": [], "summary": {}}
        rows = store.list_user_behavior_signals(kind=kind, limit=limit)
        return {
            "count": len(rows),
            "signals": [
                {
                    "id": s.id,
                    "kind": s.kind,
                    "subject": s.subject,
                    "detail": s.detail,
                    "created_at": s.created_at,
                }
                for s in rows
            ],
            "summary": store.user_behavior_summary(),
        }

    @app.get("/learning/intentions")
    def learning_intentions(status: str = "proposed") -> dict[str, Any]:
        """Rolled-up intentions awaiting review (digital-twin layer 3).

        Read-only. Approve via POST /learning/intentions/{id}/approve (writes the goal
        to the ACTIVE identity layer); dismiss via .../dismiss. status: proposed |
        active | dismissed."""
        store = getattr(getattr(app.state, "runtime", None), "learning_store", None)
        if store is None:
            return {"status": status, "count": 0, "intentions": []}
        rows = store.list_intentions(status=status)
        return {
            "status": status,
            "count": len(rows),
            "intentions": [
                {
                    "intention_id": i.intention_id,
                    "title": i.title,
                    "summary": i.summary,
                    "supporting": list(i.supporting),
                    "status": i.status,
                    "created_at": i.created_at,
                }
                for i in rows
            ],
        }

    @app.post("/learning/intentions/{intention_id}/approve")
    def learning_intentions_approve(intention_id: str) -> dict[str, Any]:
        """Approve an intention: write it to the ACTIVE identity layer (so the agent
        works toward it) and clear the proposal. Write-gated. 404 if not proposed."""
        from iris_harness.memory.identity.loader import add_active_item

        store = _learning_store_or_503()
        proposal = store.get_intention(intention_id)
        if proposal is None or proposal.status != "proposed":
            raise HTTPException(status_code=404, detail=f"no proposed intention: {intention_id}")
        text = proposal.title if not proposal.summary else f"{proposal.title} — {proposal.summary}"
        add_active_item(text)
        store.resolve_intention(intention_id, "active")
        store.record_user_behavior_signal("intention_approved", subject=proposal.title)
        return {"intention_id": intention_id, "status": "active", "title": proposal.title}

    @app.post("/learning/intentions/{intention_id}/dismiss")
    def learning_intentions_dismiss(intention_id: str) -> dict[str, Any]:
        """Dismiss a proposed intention (won't be re-proposed). Write-gated."""
        store = _learning_store_or_503()
        proposal = store.get_intention(intention_id)
        if not store.resolve_intention(intention_id, "dismissed"):
            raise HTTPException(status_code=404, detail=f"no intention: {intention_id}")
        store.record_user_behavior_signal(
            "intention_dismissed", subject=proposal.title if proposal else intention_id
        )
        return {"intention_id": intention_id, "status": "dismissed"}

    @app.get("/learning/proposal-quality")
    def learning_proposal_quality() -> dict[str, Any]:
        """Accept/reject tallies + acceptance rate for both miners' proposal queues
        (digital-twin layers 1 & 3). Read-only, deterministic. The precision proxy: watch
        it after enabling a miner to see whether the user keeps what it proposes."""
        from iris_harness.services.learning.proposal_quality import (
            build_proposal_quality,
        )

        store = getattr(getattr(app.state, "runtime", None), "learning_store", None)
        if store is None:
            return {"available": False}
        return {"available": True, **build_proposal_quality(store).as_dict()}

    @app.get("/learning/behaviors/preview")
    def learning_behaviors_preview() -> dict[str, Any]:
        """Dry-run the behavior miner on real history and return what it WOULD propose,
        persisting nothing — validate proposal quality before enabling the 24h tick.
        Runs the miner even when ``IRIS_BEHAVIOR_MINER`` is off (explicit, read-only)."""
        rt = getattr(app.state, "runtime", None)
        if rt is None:
            return {"available": False, "patterns": []}
        preview: dict[str, Any] = rt.learning.preview_behavior_mining()
        return preview

    @app.get("/learning/intentions/preview")
    def learning_intentions_preview() -> dict[str, Any]:
        """Dry-run the intention rollup on the user's current activity and return what it
        WOULD propose, persisting nothing. Runs even when ``IRIS_INTENTION_ROLLUP`` is
        off (explicit, read-only)."""
        rt = getattr(app.state, "runtime", None)
        if rt is None:
            return {"available": False, "intentions": []}
        preview: dict[str, Any] = rt.learning.preview_intention_rollup()
        return preview

    @app.get("/learning/flags")
    def learning_flags() -> dict[str, Any]:
        """State of the hot-toggleable learning capabilities (ADR-0083): behavior_miner,
        intention_rollup, learning_analyst. Each shows live ``enabled`` + the ``env_default``
        a restart reverts to. Read-only."""
        rt = getattr(app.state, "runtime", None)
        if rt is None:
            return {"available": False, "flags": {}}
        return {"available": True, "flags": rt.learning.learning_flags()}

    @app.post("/learning/flags/{name}")
    def learning_flag_set(
        name: str, body: LearningFlagRequest, http_request: Request
    ) -> dict[str, Any]:
        """Hot-toggle a learning capability WITHOUT a restart. Write-gated. Returns the
        resulting state (may stay off if the LLM can't be built). The resulting state is
        saved as the setting behind it (ADR-0120), so a restart keeps it — it used to
        revert to the env default."""
        from iris_harness.foundation.settings.env_overrides import (
            set_env_override,
        )
        from iris_harness.runtime.learning_controls import learning_flag_env

        rt = runtime()
        try:
            # Name the setting first: a control without one must not flip unsaved.
            env_key = learning_flag_env(name)
            enabled = rt.learning.set_learning_flag(name, body.enabled)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        set_env_override(
            env_key,
            "1" if enabled else "0",
            actor=actor_of(http_request),
            store=SettingsStore(db_path=rt.data_dir / SETTINGS_DB_NAME),
        )
        return {"name": name, "enabled": enabled, "saved_as": env_key}

    @app.post("/learning/flags/{name}/run")
    def learning_flag_run(name: str) -> dict[str, Any]:
        """Run a learning capability now instead of waiting for its heartbeat — the fast
        half of the tweak loop. Write-gated. Returns the count it produced."""
        rt = runtime()
        try:
            count = rt.learning.run_learning_now(name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return {"name": name, "ran": True, "count": count}
