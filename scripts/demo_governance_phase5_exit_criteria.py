"""Manual demo harness for Governance Phase 5 exit criteria.

Run:
    python3 scripts/demo_governance_phase5_exit_criteria.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iris_harness.agent.agent_executor import AgentResult
from iris_harness.agent.response_curator import ResponseCurator
from iris_harness.kernel.governance.audit import AuditLog

try:
    from iris_harness.kernel.governance.audit.archive import (
        AuditArchive,
        AuditCompactor,
        AuditQueryEngine,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - manual operator path
    raise SystemExit(
        "Phase 5 demo requires optional deps (pyarrow + duckdb). "
        "Install project dependencies and re-run."
    ) from exc


@dataclass
class _DemoFaithfulnessJudge:
    """Deterministic async judge used to exercise retry-budget behavior."""

    verdicts: list[dict[str, Any]]

    async def judge(self, *, query: str, response: str) -> str:
        if self.verdicts:
            payload = self.verdicts.pop(0)
        else:
            payload = {
                "addresses_question": True,
                "confidence": 0.8,
                "rationale": "response addresses the question",
            }
        return json.dumps(payload)


def _seed_audit_rows(audit_log: AuditLog, *, now: datetime) -> None:
    old_ts = now - timedelta(days=45)
    hot_ts = now - timedelta(days=2)
    for idx, ts in enumerate((old_ts, hot_ts), start=1):
        audit_log.record(
            run_id=f"demo-run-{idx}",
            step_id=None,
            agent_type="chat",
            hook_point="pre_response",
            plugin="demo_seed",
            decision="allow",
            severity="info",
            reason=f"seed row {idx}",
            classification="policy/demo",
            payload={"seed": idx},
            ts=ts,
        )


def _assert(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"[PASS] {name}")


def _demo_curator_signals() -> None:
    print("\n== Curator demo ==")

    safety_curator = ResponseCurator()
    blocked = safety_curator.curate(
        [AgentResult(agent_type="system", output="Use AKIA1234567890ABCDEF.", success=True)],
        query="show me credentials",
    )
    _assert(
        "safety halt blocks unsafe output", blocked.has_errors and "unsafe" in blocked.text.lower()
    )

    schema_curator = ResponseCurator()
    schema_retry = schema_curator.curate(
        [
            AgentResult(
                agent_type="system",
                output='Result:\n```json\n{"status":"ok"}\n```',
                success=True,
                metadata={"expected_schema": {"required": ["status"]}},
            )
        ],
        query="Return status JSON",
        strict=True,
    )
    bundle = schema_retry.metadata.get("judge_bundle", {})
    _assert(
        "schema retry succeeds on retry-1",
        isinstance(bundle, dict) and int(bundle.get("retries_used", 0)) >= 1,
    )

    budget_curator = ResponseCurator(
        retry_budget=2,
        faithfulness_judge=_DemoFaithfulnessJudge(
            verdicts=[
                {
                    "addresses_question": False,
                    "confidence": 0.1,
                    "rationale": "off-topic",
                },
                {
                    "addresses_question": False,
                    "confidence": 0.1,
                    "rationale": "still off-topic",
                },
            ]
        ),
    )
    budget_case = budget_curator.curate(
        [
            AgentResult(
                agent_type="system",
                output='{"foo":"bar"}',
                success=True,
                metadata={"expected_schema": {"required": ["status"]}},
            )
        ],
        query="Return my order status as JSON.",
        strict=True,
    )
    budget_bundle = budget_case.metadata.get("judge_bundle", {})
    _assert(
        "retry budget is exhausted",
        isinstance(budget_bundle, dict) and int(budget_bundle.get("retries_used", 0)) == 2,
    )
    _assert("warning banner shown", "Governance warning" in budget_case.text)


def _demo_archive_and_query(tmp_dir: Path) -> None:
    print("\n== Archive demo ==")
    audit_db = tmp_dir / "audit.db"
    archive_root = tmp_dir / "audit-archive"
    now = datetime.now(UTC)

    audit_log = AuditLog(db_path=audit_db)
    _seed_audit_rows(audit_log, now=now)

    archive = AuditArchive(root=archive_root)
    compactor = AuditCompactor(audit_log=audit_log, archive=archive, retention_days=30)
    first = compactor.compact(now=now)
    second = compactor.compact(now=now)
    _assert(
        "first compaction moved one old row", first.archived_rows == 1 and first.deleted_rows == 1
    )
    _assert(
        "second compaction is idempotent", second.archived_rows == 0 and second.deleted_rows == 0
    )

    query_engine = AuditQueryEngine(audit_db_path=audit_db, archive_root=archive_root)
    result = query_engine.query(
        "SELECT COUNT(*) AS n FROM audit_archive WHERE classification = 'policy/demo'"
    )
    total = int(result.rows[0][0]) if result.rows else 0
    _assert("duckdb sees hot+cold rows transparently", total == 2)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="iris-phase5-demo-") as raw:
        _demo_curator_signals()
        _demo_archive_and_query(Path(raw))
    print("\nPhase 5 exit-criteria demo completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
