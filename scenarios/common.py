"""Shared plumbing for the Phase 2 scenario harness.

Scenarios are *measurements*, not pass/fail CI tests: a routing accuracy of
70% is a baseline to improve, not a red build. Results append to JSONL files
under ``docs/testing-program/results/`` (committed — they are the proof the
testing program documents), and each run prints a human summary.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "docs" / "testing-program" / "results"


@dataclass
class ScenarioRecord:
    """One graded case inside a scenario run."""

    scenario: str
    case_id: str
    verdict: str  # "pass" | "fail" | "info"
    expected: str
    actual: str
    latency_ms: float
    detail: dict[str, Any] = field(default_factory=dict)


def run_stamp() -> str:
    """UTC timestamp used as the run id in result filenames."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_results(scenario: str, stamp: str, records: list[ScenarioRecord]) -> Path:
    """Append-only JSONL persistence, one file per scenario run."""
    out_dir = RESULTS_DIR / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stamp}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), default=str, sort_keys=True) + "\n")
    return path


def summarize(records: list[ScenarioRecord]) -> str:
    """Plain-text summary: verdict counts + failure table."""
    counts = Counter(r.verdict for r in records)
    graded = [r for r in records if r.verdict in ("pass", "fail")]
    lines = [
        f"cases={len(records)} pass={counts.get('pass', 0)} "
        f"fail={counts.get('fail', 0)} info={counts.get('info', 0)}"
    ]
    if graded:
        accuracy = counts.get("pass", 0) / len(graded)
        lines.append(f"accuracy={accuracy:.1%} over {len(graded)} graded cases")
    failures = [r for r in records if r.verdict == "fail"]
    if failures:
        lines.append("failures:")
        for r in failures:
            lines.append(f"  {r.case_id}: expected={r.expected} actual={r.actual}")
    return "\n".join(lines)


class Timer:
    """Tiny monotonic stopwatch for per-case latency."""

    def __enter__(self) -> Timer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000.0
