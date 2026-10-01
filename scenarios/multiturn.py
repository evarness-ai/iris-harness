"""Phase 3 — multi-turn scenarios.

Everything graded so far has been single-turn; these scripts drive real
multi-turn conversations through ``runtime.chat`` (isolated data dir, real
config + skills, real local LLMs) and grade context carry-over, follow-up
handling, and session isolation.
"""

from __future__ import annotations

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


def run() -> list[ScenarioRecord]:
    records: list[ScenarioRecord] = []
    with tempfile.TemporaryDirectory(prefix="iris-scenario-mt-") as tmp:
        runtime = _runtime(Path(tmp))

        # 1. Deterministic follow-up: time then date in one session.
        session = "mt-time-date"
        runtime.chat("what time is it?", session_id=session)
        with Timer() as timer:
            followup = runtime.chat("and what's today's date?", session_id=session)
        records.append(
            ScenarioRecord(
                scenario="multiturn",
                case_id="deterministic-followup",
                verdict="pass" if "date:" in followup.response.lower() else "fail",
                expected="date answer on the follow-up turn",
                actual=followup.response[:80],
                latency_ms=round(timer.elapsed_ms, 2),
            )
        )

        # 2. Context carry: a fact stated in turn 1 is recalled in turn 2.
        session = "mt-recall"
        runtime.chat(
            "for this conversation, note that my favorite color is teal",
            session_id=session,
        )
        with Timer() as timer:
            recall = runtime.chat("what is my favorite color?", session_id=session)
        records.append(
            ScenarioRecord(
                scenario="multiturn",
                case_id="context-recall",
                verdict="pass" if "teal" in recall.response.lower() else "fail",
                expected="'teal' recalled from turn 1",
                actual=recall.response[:80],
                latency_ms=round(timer.elapsed_ms, 2),
            )
        )

        # 3. Scope qualifier honored: turn 1 said "for this conversation",
        #    so the fact must NOT surface in a fresh session (Phase 3
        #    follow-up fix). In-session recall above still works.
        with Timer() as timer:
            fresh = runtime.chat("what is my favorite color?", session_id="mt-fresh")
        records.append(
            ScenarioRecord(
                scenario="multiturn",
                case_id="scope-qualifier-honored",
                verdict="pass" if "teal" not in fresh.response.lower() else "fail",
                expected="'for this conversation' fact does NOT leak to a fresh session",
                actual=(
                    "leaked cross-session"
                    if "teal" in fresh.response.lower()
                    else "correctly not recalled cross-session"
                ),
                latency_ms=round(timer.elapsed_ms, 2),
            )
        )

        # 3b. Control: a durable fact (no scope qualifier) SHOULD cross
        #     sessions — the single-user memory feature still works.
        runtime.chat("my name is Robin", session_id="mt-durable")
        durable = runtime.chat("what is my name?", session_id="mt-durable-fresh")
        records.append(
            ScenarioRecord(
                scenario="multiturn",
                case_id="durable-fact-crosses-sessions",
                verdict="pass" if "robin" in durable.response.lower() else "fail",
                expected="an unscoped fact still crosses sessions (by design)",
                actual=durable.response[:80],
                latency_ms=0.0,
            )
        )

        # 4. Clarify-then-answer: vague routine ask -> clarification -> reply
        #    resolves to a draft (multi-turn slot filling).
        session = "mt-clarify"
        first = runtime.chat(
            "set up a routine for me every morning at 7",
            session_id=session,
        )
        with Timer() as timer:
            second = runtime.chat("the morning briefing", session_id=session)
        pending = runtime.routine_store.get_pending_approval_request(session)
        records.append(
            ScenarioRecord(
                scenario="multiturn",
                case_id="vague-routine-clarifies",
                verdict=(
                    "pass"
                    if first.metadata.get("routine_action") in ("clarify", "drafted")
                    else "fail"
                ),
                expected="turn 1 clarifies (or drafts) instead of falling through",
                actual=f"t1={first.metadata.get('routine_action')}",
                latency_ms=round(timer.elapsed_ms, 2),
            )
        )
        # Stateful capability clarify (Phase 3 follow-up fix): turn 1 asks
        # "which capability?" and remembers the request; turn 2 names it
        # ("the morning briefing") and the two combine into a draft — no
        # need to restate the schedule.
        records.append(
            ScenarioRecord(
                scenario="multiturn",
                case_id="clarify-reply-resolves-capability",
                verdict=(
                    "pass"
                    if second.metadata.get("routine_action") == "drafted" and pending is not None
                    else "fail"
                ),
                expected="naming the capability on turn 2 drafts the routine",
                actual=(
                    f"t2={second.metadata.get('routine_action')} pending={pending is not None}"
                ),
                latency_ms=0.0,
            )
        )

        # 5. History length: the conversation store carries all turns.
        history = runtime._conversations.get("mt-recall", [])
        records.append(
            ScenarioRecord(
                scenario="multiturn",
                case_id="history-persisted",
                verdict="pass" if len(history) == 4 else "fail",
                expected="4 entries (2 user + 2 assistant)",
                actual=f"{len(history)} entries",
                latency_ms=0.0,
            )
        )
    return records
