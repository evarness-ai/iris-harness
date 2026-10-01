"""ADR-0070 IRIS_EVAL_SANDBOX=docker — host driver argv + workload/verdict protocol.

The real container run needs Docker + the iris-eval image (validated on a Docker
host). Here we test the seams without Docker: argv construction, the frozen-workload
round-trip, and stdout→verdict parsing via an injected runner."""

from __future__ import annotations

import pytest

from iris_harness.services.learning.eval_docker import (
    RunResult,
    build_eval_docker_argv,
    run_eval_in_docker,
)
from iris_harness.services.learning.eval_harness import EvalQuery
from iris_harness.services.learning.eval_replay_cli import (
    PreflightVerdict,
    parse_args,
    verdict_from_stdout,
    verdict_to_json,
    workload_from_json,
    workload_to_json,
)
from iris_harness.services.learning.preflight import SandboxUnavailable


def _wl(n: int) -> list[EvalQuery]:
    return [EvalQuery(query=f"q{i}", intent="calendar") for i in range(n)]


# ── entrypoint protocol ──────────────────────────────────────────────────────


def test_workload_json_round_trip() -> None:
    wl = [EvalQuery(query="any meeting tomorrow?", intent="calendar", expected_tool="calendar")]
    back = workload_from_json(workload_to_json(wl))
    assert back[0].query == "any meeting tomorrow?"
    assert back[0].intent == "calendar"
    assert back[0].expected_tool == "calendar"


def test_verdict_stdout_round_trip() -> None:
    v = PreflightVerdict(True, 0.9, 10, 5, "ok")
    out = f"some log line\n{verdict_to_json(v)}\ntrailing\n"
    # trailing line isn't a verdict; parser scans for the prefixed line
    recovered = verdict_from_stdout(out + verdict_to_json(v))
    assert recovered is not None and recovered.passed is True
    assert recovered.completion_rate == pytest.approx(0.9)


def test_verdict_from_stdout_none_on_junk() -> None:
    assert verdict_from_stdout("no verdict here\njust noise") is None


def test_parse_args_defaults() -> None:
    ns = parse_args(["--intent", "calendar"])
    assert ns.intent == "calendar"
    assert ns.workload is None


# ── host docker argv ─────────────────────────────────────────────────────────


def test_build_argv_has_hardening_and_workload_mount() -> None:
    argv = build_eval_docker_argv(
        "calendar", gvisor=True, workload_host_path="/tmp/wl.json", image="iris-eval:latest"
    )
    joined = " ".join(argv)
    assert "--runtime=runsc" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert "iris-eval:latest" in argv
    assert "iris_harness.services.learning.eval_replay_cli" in joined
    assert "--intent calendar" in joined
    assert "/tmp/wl.json:/eval/workload.json:ro" in argv
    assert "--workload" in argv
    assert any("OLLAMA_BASE_URL=" in a for a in argv)


def test_build_argv_no_gvisor_omits_runtime() -> None:
    argv = build_eval_docker_argv("calendar", gvisor=False)
    assert "--runtime=runsc" not in argv


# ── run_eval_in_docker with an injected runner ───────────────────────────────


def test_run_in_docker_parses_verdict_from_runner() -> None:
    v = PreflightVerdict(True, 1.0, 6, 3, "replay 100%")

    def fake_runner(argv: list[str], *, timeout: int) -> RunResult:
        assert "--workload" in argv  # workload was mounted + passed
        return RunResult(stdout=verdict_to_json(v), stderr="", exit_code=0)

    out = run_eval_in_docker("calendar", _wl(3), runner=fake_runner)
    assert out.passed is True
    assert out.completion_rate == 1.0


def test_run_in_docker_raises_when_no_verdict() -> None:
    def broken_runner(argv: list[str], *, timeout: int) -> RunResult:
        return RunResult(stdout="boom, no image", stderr="image not found", exit_code=125)

    with pytest.raises(SandboxUnavailable, match="no verdict"):
        run_eval_in_docker("calendar", _wl(3), runner=broken_runner)
