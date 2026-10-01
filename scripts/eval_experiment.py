"""Run a learning experiment by hand: baseline vs a tier-routing variant.

This is the trial-and-error entry point for ADR-0070's replay-eval harness. It
replays a frozen workload of your real queries (for one intent) through an
ISOLATED eval runtime under two arms — the current config (baseline) and a
variant that routes that intent to a different tier — and prints which did better
on the offline-reproducible metric. Nothing in production is touched: the eval
runtime uses a throwaway temp data dir, so its learning signals never reach your
real ``learning.db``, and the variant is applied only to that scratch runtime.

It needs a live model backend (Ollama) — it actually runs the model on each query.

Usage:
    IRIS_AUTH_SECRET=... poetry run python scripts/eval_experiment.py \\
        --intent email --to-tier tier2 --repeats 3 --max-items 10

    # try the metric that ignores prose quality (needs a labelled workload):
    poetry run python scripts/eval_experiment.py --intent email --to-tier tier2 \\
        --workload my_email_workload.json --metric tool_correctness
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", required=True, help="intent to experiment on, e.g. email")
    parser.add_argument(
        "--to-tier", required=True, help="tier to route the intent to in the variant, e.g. tier2"
    )
    parser.add_argument("--repeats", type=int, default=3, help="runs per query (avg out variance)")
    parser.add_argument("--max-items", type=int, default=10, help="cap on workload size")
    parser.add_argument(
        "--metric",
        default="completion_rate",
        choices=["completion_rate", "tool_correctness"],
        help="completion_rate needs no labels; tool_correctness needs a labelled --workload",
    )
    parser.add_argument(
        "--min-improvement", type=float, default=5.0, help="percent gain to count as 'better'"
    )
    parser.add_argument(
        "--workload",
        type=Path,
        default=None,
        help="optional frozen workload JSON; default snapshots recent traces for the intent",
    )
    parser.add_argument(
        "--save-workload",
        type=Path,
        default=None,
        help="snapshot the workload to this JSON and exit (no eval) — for labelling/reuse",
    )
    args = parser.parse_args()

    from iris_harness.services.learning.eval_adapter import build_eval_runtime, run_preflight
    from iris_harness.services.learning.eval_harness import VariantConfig
    from iris_harness.services.learning.eval_workload import (
        build_workload_from_traces,
        load_workload,
        save_workload,
    )

    # 1. Frozen workload — replay the same queries under both arms.
    if args.workload is not None:
        workload = load_workload(args.workload)
    else:
        workload = build_workload_from_traces(intent=args.intent, max_items=args.max_items)
    if not workload:
        print(
            f"No workload for intent '{args.intent}'. Have you chatted with that intent? "
            "Try a different --intent or pass --workload a hand-written JSON file."
        )
        return 1

    if args.save_workload is not None:
        save_workload(workload, args.save_workload)
        print(f"Saved {len(workload)} queries to {args.save_workload} (hand-label expected_tool).")
        return 0

    # 2. The variant: route this intent to a different tier. Baseline = no override.
    variant = VariantConfig(
        label=f"{args.intent}@{args.to_tier}",
        intent_tier_priors={args.intent: args.to_tier},
    )
    print(
        f"Replaying {len(workload)} '{args.intent}' queries x {args.repeats} "
        f"under baseline vs {variant.label} (metric={args.metric})...\n"
    )

    # 3. Isolated eval runtime (throwaway data dir) → run baseline + variant → compare.
    runtime, scratch = build_eval_runtime()
    try:
        comparison = run_preflight(
            runtime,
            workload,
            variant,
            repeats=args.repeats,
            metric=args.metric,
            min_improvement_pct=args.min_improvement,
        )
    finally:
        try:
            runtime.shutdown()
        except Exception:  # noqa: BLE001, S110 — best-effort teardown
            pass
        shutil.rmtree(scratch, ignore_errors=True)

    print(json.dumps(comparison.as_dict(), indent=2))
    print(f"\nVerdict: {comparison.verdict.upper()} — {comparison.note}")
    if comparison.verdict == "better":
        print(
            f"\nThe variant won the pre-flight. To apply it for real, set '{args.intent}' to "
            f"'{args.to_tier}' in config/llm_tiers.yaml (move the intent's use_for tag), then let "
            "the live re-measure confirm it. The harness does NOT apply the change for you."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
