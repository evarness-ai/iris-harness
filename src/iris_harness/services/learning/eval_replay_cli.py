"""In-container entrypoint for the crystallized replay-eval pre-flight (ADR-0070).

Runs as ``python -m iris_harness.services.learning.eval_replay_cli --intent <X> …`` inside the
``iris-eval`` Docker/gVisor image: it builds the effect-contained pre-flight for
one intent, replays the workload through an isolated runtime, and prints a single
JSON verdict line to stdout. The host (``learning.eval_docker``) parses that line.

Running it inside the container is the *outer* isolation (process/filesystem/
kernel via gVisor, no host creds mounted); ``sandbox="effect"`` here is the inner
belt-and-suspenders (no external writes even if a tool tried).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from iris_harness.services.learning.eval_harness import EvalQuery
from iris_harness.services.learning.preflight import (
    DEFAULT_MIN_COMPLETION_RATE,
    DEFAULT_MIN_QUERIES,
    PreflightVerdict,
    build_proposal_preflight,
)
from iris_harness.tools.skills.models import SkillProposal

VERDICT_PREFIX = "PREFLIGHT_VERDICT "


def workload_to_json(workload: list[EvalQuery]) -> str:
    """Serialize a frozen workload so the host can pass it into the container."""
    return json.dumps(
        [{"query": q.query, "intent": q.intent, "expected_tool": q.expected_tool} for q in workload]
    )


def workload_from_json(text: str) -> list[EvalQuery]:
    rows = json.loads(text)
    out: list[EvalQuery] = []
    for r in rows:
        if isinstance(r, dict) and r.get("query"):
            out.append(
                EvalQuery(
                    query=str(r["query"]),
                    intent=str(r.get("intent", "")),
                    expected_tool=r.get("expected_tool") or None,
                )
            )
    return out


def verdict_to_json(verdict: PreflightVerdict) -> str:
    """Serialize a verdict to a single JSON line (prefixed so the host can find it
    amid any incidental stdout)."""
    payload = {
        "passed": verdict.passed,
        "completion_rate": verdict.completion_rate,
        "runs": verdict.runs,
        "queries": verdict.queries,
        "reason": verdict.reason,
    }
    return VERDICT_PREFIX + json.dumps(payload)


def verdict_from_stdout(stdout: str) -> PreflightVerdict | None:
    """Recover a verdict from container stdout (the last VERDICT line), or None."""
    line = next(
        (
            ln[len(VERDICT_PREFIX) :]
            for ln in reversed(stdout.splitlines())
            if ln.startswith(VERDICT_PREFIX)
        ),
        None,
    )
    if line is None:
        return None
    try:
        obj: dict[str, Any] = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    return PreflightVerdict(
        passed=bool(obj.get("passed", False)),
        completion_rate=float(obj.get("completion_rate", 0.0)),
        runs=int(obj.get("runs", 0)),
        queries=int(obj.get("queries", 0)),
        reason=str(obj.get("reason", "")),
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="iris_harness.services.learning.eval_replay_cli", description=__doc__
    )
    parser.add_argument("--intent", required=True, help="intent to pre-flight, e.g. calendar")
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--min-queries", type=int, default=DEFAULT_MIN_QUERIES)
    parser.add_argument("--min-rate", type=float, default=DEFAULT_MIN_COMPLETION_RATE)
    parser.add_argument(
        "--workload",
        default=None,
        help="path to a frozen workload JSON (host-sourced); replayed verbatim. "
        "Without it, the workload is sourced from this environment's traces.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv if argv is not None else sys.argv[1:])
    proposal = SkillProposal(
        proposal_id=f"prop-eval-{ns.intent}",
        task_id=f"signal:{ns.intent}",
        skill_name=f"{ns.intent} (eval)",
        skill_slug=f"{ns.intent}-eval",
        scope="learning",
        source_description=f"replay-eval pre-flight for intent '{ns.intent}'",
        proposal_dir=f"config/skills/auto/{ns.intent}-eval",
        manifest_path=f"config/skills/auto/{ns.intent}-eval/manifest.yaml",
        source_kind="crystallized",
    )
    # A frozen workload is passed in by the host (the container can't see host
    # traces); sandbox="off" since the container itself is the isolation and
    # IRIS_DISABLE_EXTERNAL_WRITES is set on the container env.
    if ns.workload:
        frozen = workload_from_json(Path(ns.workload).read_text(encoding="utf-8"))
        preflight = build_proposal_preflight(
            Path("."),
            sandbox="off",
            repeats=ns.repeats,
            max_items=ns.max_items,
            min_queries=ns.min_queries,
            min_completion_rate=ns.min_rate,
            workload_loader=lambda _intent, _limit: frozen,
        )
    else:
        preflight = build_proposal_preflight(
            Path("."),
            sandbox="effect",
            repeats=ns.repeats,
            max_items=ns.max_items,
            min_queries=ns.min_queries,
            min_completion_rate=ns.min_rate,
        )
    verdict = preflight(proposal)
    print(verdict_to_json(verdict))  # noqa: T201 — CLI output (JSON on stdout)
    return 0 if verdict.passed else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
