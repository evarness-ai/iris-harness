"""Governance state, flags, cost and the HITL approval queue.

    GET    /governance/state
    GET    /cost
    GET    /governance/audit
    GET    /governance/pii-shadow
    GET    /governance/proof-bundle/check
    GET    /governance/approvals
    POST   /governance/approvals/{approval_id}/respond

Moved out of ``create_app`` unchanged (review item: split the god function); the route
table and OpenAPI schema are identical before and after. The write guard in ``main``
still gates the mutating routes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from iris_harness.foundation.env import env_flag
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.audit.view import audit_view, redaction_view
from iris_harness.kernel.governance.external_content import (
    EXTERNAL_CONTENT_FLOOR_FLAG,
    floor_enabled,
)
from iris_harness.kernel.governance.plugins.owner_pii_shadow import (
    ENV_FLAG as OWNER_PII_FLAG,
)
from iris_harness.kernel.governance.plugins.owner_pii_shadow import (
    owner_pii_mode_from_env,
    owner_pii_shadow_summary,
)
from iris_harness.kernel.governance.wiring import (
    SIDE_EFFECT_LEDGER_ALL_ENV,
    SIDE_EFFECT_LEDGER_ENV,
    side_effect_ledger_settings_from_env,
)


class ApprovalRespondRequest(BaseModel):
    """Body for ``POST /governance/approvals/{approval_id}/respond``."""

    status: str  # "approved" | "rejected"
    actor: str | None = None


# Curated governance posture flags (key, human label, secure default). Surfaced
# read-only so the operator can see what's shadow vs enforced; flips stay an
# operator action (env / runbook), never a UI write.
_GOVERNANCE_FLAGS: tuple[tuple[str, str, bool], ...] = (
    ("IRIS_GOVERNANCE_ENABLED", "Kernel enabled", True),
    ("IRIS_GOVERNANCE_COMMAND_SANDBOX", "Command sandbox", True),
    ("IRIS_GOVERNANCE_FS_JAIL", "Filesystem jail", True),
    ("IRIS_GOVERNANCE_NETWORK_EGRESS", "Network egress control", True),
    ("IRIS_GOVERNANCE_MCP_ALLOWLIST", "MCP allowlist", True),
    ("IRIS_GOVERNANCE_REDACTION_ENABLED", "Secret redaction", False),
    ("IRIS_GOVERNANCE_LOOP_DETECT_ENABLED", "Loop detection", False),
    ("IRIS_GOVERNANCE_COST_LIMITER_ENABLED", "Cost limiter", False),
    ("IRIS_GOVERNANCE_PROMPT_GUARD", "Prompt / threat guard", False),
    ("IRIS_GOVERNANCE_INPUT_SAFETY", "Input safety screen", False),
)


def _audit_log() -> AuditLog:
    raw = os.getenv("IRIS_GOVERNANCE_AUDIT_DB_PATH")
    return AuditLog(db_path=Path(raw).expanduser() if raw else None)


def _flag_payload(spec: tuple[tuple[str, str, bool], ...]) -> list[dict[str, Any]]:
    return [{"key": k, "label": lbl, "on": _env_flag(k, default=d)} for k, lbl, d in spec]


def _external_content_floor_flag() -> dict[str, Any]:
    """The external-content floor, read by the parser the kernel build uses (a blank value
    is on here, unlike the shared ``env_flag``), so the posture shown is the posture built."""
    return {
        "key": EXTERNAL_CONTENT_FLOOR_FLAG,
        "label": "External-content floor (marker + instruction tripwire, no model)",
        "on": floor_enabled(),
    }


def _owner_pii_flag() -> dict[str, Any]:
    """The owner-PII mode: not a boolean (``off`` | ``shadow``), so it carries ``value``,
    read by the parser the kernel build uses -- the two cannot disagree on a value."""
    setting = owner_pii_mode_from_env()
    return {
        "key": OWNER_PII_FLAG,
        "label": "Owner-PII guards",
        "on": setting.mode != "off",
        "value": setting.mode,
    }


def _side_effect_ledger_flags() -> list[dict[str, Any]]:
    """The two ledger booleans, read by the parser the kernel build uses so the two cannot
    disagree on a spelling or an unrecognised value. ``record_all`` is false while the
    ledger is off: it has no effect then."""
    setting = side_effect_ledger_settings_from_env()
    return [
        {
            "key": SIDE_EFFECT_LEDGER_ENV,
            "label": "Side-effect ledger (high-risk calls; off denies destructive tools)",
            "on": setting.enabled,
        },
        {
            "key": SIDE_EFFECT_LEDGER_ALL_ENV,
            "label": "Side-effect ledger: also record every non-read call",
            "on": setting.record_all,
        },
    ]


def _model_guard(runtime: Callable[[], Any]) -> dict[str, Any]:
    """The model guard's posture, from the probe Health uses (issue #136), so the two cannot
    disagree. ``external_tools_mounted`` is None when the runtime is not up."""
    from iris_harness.kernel.governance.threat.availability import model_guard_state

    state = model_guard_state()
    mounted: int | None
    try:
        from iris_harness.runtime.external_tools import mounted_external_tools

        mounted = len(mounted_external_tools(runtime()))
    except Exception:  # noqa: BLE001 - a runtime that is not up must not 500 the state page
        mounted = None
    return {
        "on": state.on,
        "classifier": state.classifier if state.on else None,
        "reason": state.reason,
        "fix": state.fix,
        "external_tools_mounted": mounted,
    }


def _allow_state() -> dict[str, Any]:
    """The owner's external-content allow-list (issue #139): how many entries are live, and
    why the file is being ignored when it is (then nothing is allowed)."""
    from iris_harness.kernel.governance.external_content_allow import allow_status

    entries, problem = allow_status()
    return {"entries": entries, "problem": problem}


def _env_flag(name: str, *, default: bool = False) -> bool:
    """Thin alias for the shared reader, keeping this module's semantics.

    One of six copies M6.3 found in three disagreeing variants; see
    ``iris_harness.foundation.env`` for what they disagreed about.
    """
    return env_flag(name, default=default)


def install_governance_routes(app: FastAPI, runtime: Callable[[], Any]) -> None:
    """Register these routes. ``runtime`` returns the live runtime or raises 503."""

    @app.get("/governance/state")
    def governance_state() -> dict[str, Any]:
        audit = _audit_log()
        try:
            audit_count = audit.count()
        except Exception:  # noqa: BLE001 — a missing/locked audit db must not 500
            audit_count = 0
        return {
            "enabled": _env_flag("IRIS_GOVERNANCE_ENABLED", default=True),
            "audit_db": str(audit.db_path),
            "audit_count": audit_count,
            "flags": [
                *_flag_payload(_GOVERNANCE_FLAGS),
                *_side_effect_ledger_flags(),
                _external_content_floor_flag(),
                _owner_pii_flag(),
            ],
            "model_guard": _model_guard(runtime),
            "external_content_allow": _allow_state(),
        }

    @app.get("/cost")
    def cost() -> dict[str, Any]:
        """What IRIS has spent on LLM calls (Track 2 PR 6, plan decision 30).

        A thin renderer over the governance cost ledger; the rollup lives in
        ``kernel.governance.cost.summary``. It carries ``recording`` because the
        ledger is only written while ``CostLimiter`` is registered, and a
        truthful 0.00 from a ledger nobody writes reads exactly like a real
        0.00 — the caller has to be able to tell those apart.

        This is LLM spend only. The Azure resource-group bill (the VM, storage,
        the Foundry account) needs the ``azure_cost`` plugin and a managed
        identity, which is track 4.
        """
        from iris_harness.kernel.governance.cost import cost_summary

        user_id = (os.getenv("IRIS_USER_ID", "") or "local").strip() or "local"
        return cost_summary(user_id=user_id).as_dict()

    @app.get("/governance/audit")
    def governance_audit(
        limit: int = 100, decision: str | None = None, caller: str | None = None
    ) -> dict[str, Any]:
        """The newest ledger rows, newest first: a thin renderer over ``audit_view``.

        ``caller`` filters by who made the call, exactly or by namespace (``mcp:``).
        Each entry carries the documented payload fields only (caller, deterministic
        handler, tool, digest key, session) and a reason with email addresses masked.
        """
        audit = _audit_log()
        try:
            return audit_view(audit, decision=decision, caller=caller, limit=limit)
        except Exception:  # noqa: BLE001 — empty/missing ledger -> empty view, not 500
            return {
                "count": 0,
                "total": 0,
                "audit_db": str(audit.db_path),
                "callers": [],
                "entries": [],
            }

    @app.get("/governance/redactions")
    def governance_redactions(limit: int = 100) -> dict[str, Any]:
        """What the external-content floor redacted and the allow-list kept (issue #139):
        a thin renderer over ``redaction_view``. Pattern ids and counts only, never text."""
        audit = _audit_log()
        try:
            return redaction_view(audit, limit=limit)
        except Exception:  # noqa: BLE001 — empty/missing ledger -> empty view, not 500
            return {"count": 0, "audit_db": str(audit.db_path), "entries": []}

    @app.get("/governance/pii-shadow")
    def governance_pii_shadow(days: float = 7.0) -> dict[str, Any]:
        """What the owner-PII guards would have done, over the last ``days`` (ADR-0125 PR 4).

        Counts by hook point x guard x kind x action from the shadow hook's audit rows --
        never a literal. A thin renderer over ``owner_pii_shadow_summary``; ``mode`` says
        whether shadow is recording, because an empty summary from a kernel that is not
        observing reads exactly like a quiet one.
        """
        if not 0 < days <= 366:
            raise HTTPException(status_code=422, detail="days must be in (0, 366]")
        try:
            summary = owner_pii_shadow_summary(_audit_log(), days=days).as_dict()
        except Exception:  # noqa: BLE001 — empty/missing ledger -> empty view, not 500
            summary = {"since": "", "rows": 0, "checked": {}, "unobserved": {}, "cells": []}
        return {"mode": owner_pii_mode_from_env().mode, **summary}

    @app.get("/governance/proof-bundle/check")
    def governance_proof_bundle_check(days: float = 7.0) -> dict[str, Any]:
        """The R14 proof bundle of the last ``days``, verified, per invariant.

        A thin renderer over ``proof_bundle.check_window`` (``iris governance proof-bundle
        check`` prints the same): the ledger window and the session logs' model calls and
        answers, exported in memory -- nothing is written -- and verified offline. Each
        invariant carries the evidence it was judged on, so one that holds over nothing
        is not mistaken for one that held over a week of calls.
        """
        from datetime import UTC, datetime, timedelta

        from iris_harness.foundation.observability.session_log import session_log_dir
        from iris_harness.kernel.governance.audit import proof_bundle

        if not 0 < days <= 366:
            raise HTTPException(status_code=422, detail="days must be in (0, 366]")
        since = datetime.now(UTC) - timedelta(days=days)
        return proof_bundle.check_window(_audit_log(), since=since, session_logs=session_log_dir())

    # ── HITL approval queue ────────────────────────────────────────────────────
    #
    # The queue itself is channel-agnostic and so is the capability behind these two
    # endpoints: `governance.approvals.service` holds the decision + audit + resume,
    # and the CLI (`iris approvals`) and the Telegram handler call the same functions.
    # These are the HTTP surface, not the implementation — a core capability is never
    # isolated in one channel. Pending approvals also reach `GET /actions`, so any
    # channel reading the Action Center sees that a run is waiting.

    @app.get("/governance/approvals")
    def governance_approvals(due_only: bool = False) -> dict[str, Any]:
        """Approvals still waiting on a human. Pure read."""
        from datetime import UTC, datetime

        from iris_harness.kernel.governance.approvals.service import (
            parse_checkpoint_id,
            pending_approvals,
        )

        rows = pending_approvals(due_only=due_only)
        now = datetime.now(UTC).isoformat()
        return {
            "count": len(rows),
            "approvals": [
                {
                    "approval_id": r.approval_id,
                    "run_id": r.run_id,
                    "signal": r.signal,
                    "context_summary": r.context_summary,
                    "requested_at": r.requested_at,
                    "timeout_at": r.timeout_at,
                    "channel": r.channel,
                    "status": r.status,
                    "checkpoint_id": r.checkpoint_id,
                    # Whether approving this can actually continue the run, so the UI can
                    # say which button does what instead of promising a resume it cannot do.
                    "resumable": parse_checkpoint_id(r.checkpoint_id) is not None,
                    # Past its deadline but not yet swept. `approval_timeout_tick` runs
                    # every 60s, so this is a short window — but a read must not show a
                    # dead request as live, which is exactly what it used to do forever.
                    "overdue": r.timeout_at <= now,
                    "session_id": r.session_id,
                    # ADR-0118: "destructive" when the row pins tool calls (it deletes or
                    # overwrites data), else "evaluator" (a paused run). The card is how it
                    # reads to the owner; items are the exact calls approving will run.
                    "kind": "destructive" if r.items is not None else "evaluator",
                    "card": r.card.to_dict() if r.card is not None else None,
                    "items": [{"tool": i.tool, "args": i.args} for i in (r.items or ())],
                }
                for r in rows
            ],
        }

    @app.post("/governance/approvals/{approval_id}/respond")
    def respond_approval(approval_id: str, request: ApprovalRespondRequest) -> dict[str, Any]:
        """Approve or reject a pending approval; approving continues the halted run.

        The resume runs the turn pipeline, so it can take as long as a chat turn — this
        is deliberately synchronous, like ``POST /chat``, and the answer comes back in
        ``detail``. It also lands in the session log, so the conversation the run belongs
        to shows it whether or not this caller waits.
        """
        from iris_harness.kernel.governance.approvals.service import respond_to_approval
        from iris_harness.kernel.governance.approvals.store import (
            ApprovalAlreadyAnsweredError,
            ApprovalNotFoundError,
        )

        if request.status not in ("approved", "rejected"):
            raise HTTPException(status_code=400, detail="status must be 'approved' or 'rejected'")
        rt = runtime()
        try:
            outcome = respond_to_approval(
                approval_id,
                status=request.status,
                actor=request.actor or "web",
                resumer=rt,
                # A code caller's approved call runs here (plugin-capabilities decision 1).
                executor=rt.tool_service,
            )
        except ApprovalNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ApprovalAlreadyAnsweredError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return outcome.as_dict()
