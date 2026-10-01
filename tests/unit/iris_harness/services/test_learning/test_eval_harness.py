"""Tests for the offline replay-eval harness core (ADR-0070, slice 1).

The harness is pure given an injected RunQuery, so these drive it with fake
executions: deterministic scoring, the counterfactual comparison, the
inconclusive guards (no labels, too few runs), repeats, and the frozen-workload
round-trip.
"""

from __future__ import annotations

from pathlib import Path

from iris_harness.services.learning.eval_harness import (
    COMPLETION_RATE,
    TOOL_CORRECTNESS,
    EvalQuery,
    RunOutcome,
    VariantConfig,
    compare,
    evaluate,
    score_arm,
)
from iris_harness.services.learning.eval_workload import (
    build_workload_from_traces,
    load_workload,
    save_workload,
)


def _wl() -> list[EvalQuery]:
    return [EvalQuery(query="q1", intent="email"), EvalQuery(query="q2", intent="email")]


def test_score_arm_completion_rate() -> None:
    # q1 completes, q2 doesn't -> 50%.
    def run(item: EvalQuery, _cfg: VariantConfig) -> RunOutcome:
        return RunOutcome(completed=item.query == "q1")

    score = score_arm(_wl(), run_query=run, config=VariantConfig(label="baseline"))
    assert score.runs == 2
    assert score.completion_rate == 0.5
    assert score.tool_correctness is None  # no labels


def test_score_arm_tool_correctness_only_over_labelled() -> None:
    wl = [
        EvalQuery(query="q1", intent="email", expected_tool="read_email"),
        EvalQuery(query="q2", intent="email"),  # unlabelled -> ignored for tool metric
    ]

    def run(item: EvalQuery, _cfg: VariantConfig) -> RunOutcome:
        return RunOutcome(completed=True, tools_called=("read_email",))

    score = score_arm(wl, run_query=run, config=VariantConfig(label="v"))
    assert score.labelled_runs == 1
    assert score.tool_correctness == 1.0


def test_score_arm_repeats_average_variance() -> None:
    calls = {"n": 0}

    def run(_item: EvalQuery, _cfg: VariantConfig) -> RunOutcome:
        calls["n"] += 1
        return RunOutcome(completed=calls["n"] % 2 == 0)  # alternate pass/fail

    score = score_arm(
        [EvalQuery(query="q", intent="x")],
        run_query=run,
        config=VariantConfig(label="v"),
        repeats=4,
    )
    assert score.runs == 4
    assert score.completion_rate == 0.5


def test_exception_in_run_counts_as_noncompletion() -> None:
    def run(_item: EvalQuery, _cfg: VariantConfig) -> RunOutcome:
        raise RuntimeError("boom")

    score = score_arm(_wl(), run_query=run, config=VariantConfig(label="v"))
    assert score.runs == 2
    assert score.completion_rate == 0.0


def test_evaluate_variant_beats_baseline() -> None:
    # Baseline (no priors) fails; variant (email->tier2) completes.
    def run(_item: EvalQuery, cfg: VariantConfig) -> RunOutcome:
        return RunOutcome(completed=not cfg.is_baseline)

    cmp = evaluate(
        _wl(),
        run_query=run,
        variant=VariantConfig(label="email@tier2", intent_tier_priors={"email": "tier2"}),
        metric=COMPLETION_RATE,
    )
    assert cmp.verdict == "better"
    assert cmp.improvement_pct is not None and cmp.improvement_pct > 0
    assert cmp.baseline.completion_rate == 0.0
    assert cmp.variant.completion_rate == 1.0


def test_evaluate_variant_worse() -> None:
    def run(_item: EvalQuery, cfg: VariantConfig) -> RunOutcome:
        return RunOutcome(completed=cfg.is_baseline)  # variant regresses

    cmp = evaluate(
        _wl(),
        run_query=run,
        variant=VariantConfig(label="v", intent_tier_priors={"email": "tier3"}),
    )
    assert cmp.verdict == "worse"


def test_tool_correctness_inconclusive_without_labels() -> None:
    base = score_arm(
        _wl(), run_query=lambda i, c: RunOutcome(True), config=VariantConfig(label="b")
    )
    var = score_arm(_wl(), run_query=lambda i, c: RunOutcome(True), config=VariantConfig(label="v"))
    cmp = compare(baseline=base, variant=var, metric=TOOL_CORRECTNESS)
    assert cmp.verdict == "inconclusive"
    assert cmp.improvement_pct is None


def test_compare_inconclusive_on_small_change() -> None:
    def run(_item: EvalQuery, cfg: VariantConfig) -> RunOutcome:
        # 2/2 baseline vs 2/2 variant -> 0% change -> inconclusive at 5% threshold.
        return RunOutcome(completed=True)

    cmp = evaluate(
        _wl(), run_query=run, variant=VariantConfig(label="v", intent_tier_priors={"e": "t2"})
    )
    assert cmp.verdict == "inconclusive"


def test_as_dict_shapes() -> None:
    base = score_arm(
        _wl(), run_query=lambda i, c: RunOutcome(True), config=VariantConfig(label="b")
    )
    var = score_arm(_wl(), run_query=lambda i, c: RunOutcome(True), config=VariantConfig(label="v"))
    payload = compare(baseline=base, variant=var).as_dict()
    assert set(payload) == {"metric", "baseline", "variant", "improvement_pct", "verdict", "note"}
    assert payload["baseline"]["label"] == "b"


# ---- workload loader -------------------------------------------------------


def _write_session(log_dir: Path, name: str, turns: list[tuple[str, str, bool]]) -> None:
    import json

    lines: list[str] = []
    for query, intent, has_errors in turns:
        lines.append(json.dumps({"kind": "user_message", "text": query}))
        lines.append(
            json.dumps(
                {
                    "kind": "agent_response",
                    "session_id": "s1",
                    "turn_id": "t",
                    "intent": intent,
                    "agent_type": intent,
                    "response": "ok",
                    "has_errors": has_errors,
                }
            )
        )
    (log_dir / f"session-{name}.jsonl").write_text("\n".join(lines), encoding="utf-8")


def test_build_workload_filters_intent_and_dedupes(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        "a",
        [
            ("email q1", "email", False),
            ("email q1", "email", False),
            ("finance q", "finance", False),
        ],
    )
    wl = build_workload_from_traces(intent="email", log_dir=tmp_path)
    assert [q.query for q in wl] == ["email q1"]  # filtered + deduped
    assert wl[0].intent == "email"


def test_workload_round_trip_with_label(tmp_path: Path) -> None:
    items = [EvalQuery(query="q", intent="email", expected_tool="read_email")]
    path = tmp_path / "wl.json"
    save_workload(items, path)
    loaded = load_workload(path)
    assert loaded == items


def test_load_missing_workload_is_empty(tmp_path: Path) -> None:
    assert load_workload(tmp_path / "nope.json") == []
