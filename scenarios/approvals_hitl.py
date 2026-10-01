"""Phase 3 — HITL approval scenarios.

Two layers, both production code paths:

- **Chat-level routine approvals** (the user-facing surface): draft via
  natural language → refine → approve / cancel, against an isolated data
  dir with the real skill registry.
- **Governance approval gate** (the pause/resume surface used by the
  evaluator and tool gates): enqueue → approve / reject / timeout through
  ``ApprovalStore``/``ApprovalGate``.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from scenarios.common import REPO_ROOT, ScenarioRecord, Timer


def _runtime(tmp: Path) -> Any:
    from iris_harness.runtime import build_runtime

    data_dir = tmp / "data"
    data_dir.mkdir(exist_ok=True)
    runtime = build_runtime(
        config_dir=REPO_ROOT / "config",
        data_dir=data_dir,
        use_background_scheduler=False,
    )
    runtime.skill_registry.discover()
    return runtime


def _chat_case(
    records: list[ScenarioRecord],
    case_id: str,
    ok: bool,
    expected: str,
    actual: str,
    latency_ms: float,
) -> None:
    records.append(
        ScenarioRecord(
            scenario="approvals-hitl",
            case_id=case_id,
            verdict="pass" if ok else "fail",
            expected=expected,
            actual=actual,
            latency_ms=round(latency_ms, 2),
        )
    )


def run_chat_approval_lifecycle() -> list[ScenarioRecord]:
    records: list[ScenarioRecord] = []
    with tempfile.TemporaryDirectory(prefix="iris-scenario-hitl-") as tmp:
        runtime = _runtime(Path(tmp))
        session = "hitl-approve"

        with Timer() as timer:
            drafted = runtime.chat(
                "send me a morning briefing everyday at 9 am with reminders " "and active items",
                session_id=session,
            )
        pending = runtime.routine_store.get_pending_approval_request(session)
        _chat_case(
            records,
            "draft-requires-approval",
            drafted.metadata.get("routine_action") == "drafted" and pending is not None,
            "drafted + approval request pending",
            f"action={drafted.metadata.get('routine_action')} pending={pending is not None}",
            timer.elapsed_ms,
        )

        with Timer() as timer:
            approved = runtime.chat("approve it", session_id=session)
        finalized = runtime.routine_store.load(pending.routine_id) if pending else None
        _chat_case(
            records,
            "approve-schedules-routine",
            approved.metadata.get("routine_action") == "approved"
            and finalized is not None
            and str(finalized.approval_status) == "scheduled",
            "approved + status scheduled",
            f"action={approved.metadata.get('routine_action')} "
            f"status={finalized and str(finalized.approval_status)}",
            timer.elapsed_ms,
        )

        # Cancel path in a fresh session.
        session = "hitl-cancel"
        runtime.chat(
            "send me a morning briefing every weekday at 8 am with reminders",
            session_id=session,
        )
        pending = runtime.routine_store.get_pending_approval_request(session)
        with Timer() as timer:
            cancelled = runtime.chat("cancel", session_id=session)
        retired = runtime.routine_store.load(pending.routine_id) if pending else None
        _chat_case(
            records,
            "cancel-retires-draft",
            pending is not None
            and retired is not None
            and str(retired.approval_status) in ("retired", "cancelled"),
            "cancel retires the draft",
            f"action={cancelled.metadata.get('routine_action')} "
            f"status={retired and str(retired.approval_status)}",
            timer.elapsed_ms,
        )

        # Unapproved drafts must never execute: no approval -> not scheduled.
        session = "hitl-no-answer"
        runtime.chat(
            "send me a morning briefing every sunday at 10 am with reminders",
            session_id=session,
        )
        pending = runtime.routine_store.get_pending_approval_request(session)
        spec = runtime.routine_store.load(pending.routine_id) if pending else None
        _chat_case(
            records,
            "unanswered-stays-unscheduled",
            spec is not None and str(spec.approval_status) != "scheduled",
            "no approval -> never scheduled",
            f"status={spec and str(spec.approval_status)}",
            0.0,
        )
    return records


def run_gate_cases() -> list[ScenarioRecord]:
    from iris_harness.governance.approvals.gate import (
        ApprovalGate,
        ApprovalRejectedError,
        ApprovalTimedOutError,
    )
    from iris_harness.governance.approvals.queue import ApprovalQueue
    from iris_harness.governance.approvals.store import ApprovalStore

    records: list[ScenarioRecord] = []

    async def _run() -> None:
        with tempfile.TemporaryDirectory(prefix="iris-scenario-gate-") as tmp:
            store = ApprovalStore(db_path=Path(tmp) / "approvals.db")
            queue = ApprovalQueue(store=store)
            gate = ApprovalGate(store=store)

            # Approve path.
            approval_id = queue.enqueue("run-a", None, "goal_drift", "scenario", channel="cli")

            async def _approve() -> None:
                await asyncio.sleep(0.05)
                store.respond(approval_id, status="approved", actor="cli:scenario")

            task = asyncio.create_task(_approve())
            with Timer() as timer:
                row = await gate.await_approval(approval_id, poll_interval=0.02)
            await task
            _chat_case(
                records,
                "gate-approve-resumes",
                row.status == "approved",
                "approved row",
                f"status={row.status}",
                timer.elapsed_ms,
            )

            # Reject path.
            approval_id = queue.enqueue("run-r", None, "loop_detect", "scenario", channel="cli")

            async def _reject() -> None:
                await asyncio.sleep(0.05)
                store.respond(approval_id, status="rejected", actor="cli:scenario")

            task = asyncio.create_task(_reject())
            outcome = "no-exception"
            with Timer() as timer:
                try:
                    await gate.await_approval(approval_id, poll_interval=0.02)
                except ApprovalRejectedError:
                    outcome = "rejected-error"
            await task
            _chat_case(
                records,
                "gate-reject-raises",
                outcome == "rejected-error",
                "ApprovalRejectedError",
                outcome,
                timer.elapsed_ms,
            )

            # Timeout path (fail-closed when nobody answers).
            approval_id = queue.enqueue("run-t", None, "cost_cap", "scenario", channel="cli")
            outcome = "no-exception"
            with Timer() as timer:
                try:
                    await gate.await_approval(approval_id, poll_interval=0.02, timeout=0.2)
                except ApprovalTimedOutError:
                    outcome = "timeout-error"
            _chat_case(
                records,
                "gate-timeout-failsclosed",
                outcome == "timeout-error",
                "ApprovalTimedOutError",
                outcome,
                timer.elapsed_ms,
            )

    asyncio.run(_run())
    return records


def run() -> list[ScenarioRecord]:
    return run_chat_approval_lifecycle() + run_gate_cases()
