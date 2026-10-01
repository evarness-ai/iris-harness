"""Unit tests for OllamaArbiter — eviction logic without hitting a real daemon."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from iris_harness.foundation.observability.host_pressure import PressureSnapshot
from iris_harness.llm.arbiter import Mode, OllamaArbiter, ResourceGovernor


@dataclass
class _StubResponse:
    payload: dict[str, Any]

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


@dataclass
class _StubHttp:
    """Records get/post calls; returns a configurable /api/ps payload."""

    ps_payload: dict[str, Any] = field(default_factory=lambda: {"models": []})
    get_calls: list[str] = field(default_factory=list)
    post_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def get(self, url: str) -> _StubResponse:
        self.get_calls.append(url)
        return _StubResponse(self.ps_payload)

    def post(self, url: str, json: dict[str, Any]) -> _StubResponse:
        self.post_calls.append((url, json))
        return _StubResponse({})

    def close(self) -> None:
        return None


def _gb(value: float) -> int:
    return int(value * (1024**3))


def test_acquire_no_op_when_under_budget() -> None:
    http = _StubHttp(ps_payload={"models": [{"name": "llama3.2:3b", "size": _gb(3)}]})
    arb = OllamaArbiter(base_url="http://localhost:11434", http=http, budget_gb=22.0)

    arb.acquire("qwen2.5-coder:7b")  # 3 + 5 = 8 GB resident, well under 22

    assert http.post_calls == []


def test_acquire_evicts_largest_first_when_over_budget() -> None:
    # Two models resident: 20 GB + 5 GB. Adding 7 GB target would reach 32 GB.
    # Budget 22 GB → must evict the 20 GB model.
    http = _StubHttp(
        ps_payload={
            "models": [
                {"name": "qwen3-coder:30b", "size": _gb(20)},
                {"name": "qwen2.5-coder:7b", "size": _gb(5)},
            ]
        }
    )
    arb = OllamaArbiter(base_url="http://localhost:11434", http=http, budget_gb=22.0)

    arb.acquire("llama3.2:3b")

    assert len(http.post_calls) == 1
    url, body = http.post_calls[0]
    assert url.endswith("/api/generate")
    assert body == {"model": "qwen3-coder:30b", "keep_alive": 0}


def test_acquire_does_not_evict_target_model() -> None:
    # Target is already resident — no eviction needed even if at budget.
    http = _StubHttp(ps_payload={"models": [{"name": "qwen3.6:27b", "size": _gb(18)}]})
    arb = OllamaArbiter(base_url="http://localhost:11434", http=http, budget_gb=22.0)

    arb.acquire("qwen3.6:27b")

    assert http.post_calls == []


def test_acquire_silently_swallows_ps_failures() -> None:
    class _Broken:
        def get(self, url: str) -> _StubResponse:
            raise RuntimeError("ollama down")

        def post(self, url: str, json: dict[str, Any]) -> _StubResponse:
            return _StubResponse({})

    arb = OllamaArbiter(base_url="http://localhost:11434", http=_Broken(), budget_gb=22.0)
    # Must not raise — arbiter is best-effort.
    arb.acquire("qwen2.5-coder:7b")


def test_size_for_falls_back_to_base_name_match() -> None:
    arb = OllamaArbiter(base_url="http://localhost:11434", http=_StubHttp())
    # llama3.2 base in defaults — unknown tag should still resolve.
    assert arb._size_for("llama3.2:1b") == 3.0
    # Truly unknown model gets the conservative default.
    assert arb._size_for("totally-new-model:42b") == 8.0


def test_acquire_strips_v1_suffix_when_calling_native_api() -> None:
    http = _StubHttp(ps_payload={"models": [{"name": "qwen3-coder:30b", "size": _gb(20)}]})
    arb = OllamaArbiter(base_url="http://localhost:11434/v1", http=http, budget_gb=22.0)

    arb.acquire("llama3.2:3b")

    assert http.get_calls == ["http://localhost:11434/api/ps"]
    assert http.post_calls[0][0] == "http://localhost:11434/api/generate"


# ---------------------------------------------------------------------------
# ResourceGovernor + PressureSnapshot tests
# ---------------------------------------------------------------------------


def _snap(
    *,
    ram_free_gb: float = 16.0,
    cpu_speed_limit: int = 100,
    cpu_percent: float = 25.0,
) -> PressureSnapshot:
    return PressureSnapshot(
        ram_free_gb=ram_free_gb,
        cpu_percent=cpu_percent,
        cpu_speed_limit=cpu_speed_limit,
        thermal_throttled=cpu_speed_limit < 100,
        sampled_at=datetime.now(UTC),
    )


def _governor(snapshots: list[PressureSnapshot], *, adaptive: bool = True) -> ResourceGovernor:
    arb = OllamaArbiter(base_url="http://localhost:11434", http=_StubHttp())
    queue = list(snapshots)

    def _sampler() -> PressureSnapshot:
        return queue.pop(0)

    return ResourceGovernor(arbiter=arb, adaptive=adaptive, sampler=_sampler)


def test_governor_enters_thermal_after_three_pressure_polls() -> None:
    pressure = _snap(ram_free_gb=2.0)  # below 4 GB threshold
    gov = _governor([pressure, pressure, pressure])

    gov.poll()
    assert gov.mode() == Mode.ACTIVE
    gov.poll()
    assert gov.mode() == Mode.ACTIVE
    gov.poll()
    assert gov.mode() == Mode.THERMAL


def test_governor_exits_thermal_after_five_clear_polls() -> None:
    pressure = _snap(ram_free_gb=2.0)
    clear = _snap(ram_free_gb=20.0)
    gov = _governor([pressure] * 3 + [clear] * 5)

    for _ in range(3):
        gov.poll()
    assert gov.mode() == Mode.THERMAL

    for i in range(4):
        gov.poll()
        assert gov.mode() == Mode.THERMAL, f"flipped early after {i + 1} clear polls"
    gov.poll()
    assert gov.mode() == Mode.ACTIVE


def test_thermal_throttling_alone_triggers_thermal_mode() -> None:
    throttled = _snap(ram_free_gb=20.0, cpu_speed_limit=70)  # ram fine, but throttled
    gov = _governor([throttled] * 3)
    for _ in range(3):
        gov.poll()
    assert gov.mode() == Mode.THERMAL


def test_recommend_tier_passthrough_when_adaptive_off() -> None:
    gov = _governor([_snap(ram_free_gb=2.0)] * 3, adaptive=False)
    for _ in range(3):
        gov.poll()
    # Even after pressure, adaptive=False means no downshift recommendation.
    assert gov.recommend_tier_name("tier3") == "tier3"


def test_recommend_tier_downshifts_to_tier1_under_thermal() -> None:
    gov = _governor([_snap(ram_free_gb=2.0)] * 3)
    for _ in range(3):
        gov.poll()
    assert gov.mode() == Mode.THERMAL
    assert gov.recommend_tier_name("tier3") == "tier1"
    # Already at the downshift target → passthrough.
    assert gov.recommend_tier_name("tier1") == "tier1"


def test_pin_overrides_automatic_mode() -> None:
    gov = _governor([_snap(ram_free_gb=20.0)])
    gov.set_pin(Mode.IDLE)
    gov.poll()
    assert gov.mode() == Mode.IDLE
    # Pin survives a clear poll — automatic transitions are bypassed.
    assert gov.recommend_tier_name("tier3") == "tier1"


def test_releasing_pin_restores_automatic_mode() -> None:
    gov = _governor([_snap(ram_free_gb=20.0), _snap(ram_free_gb=20.0)])
    gov.set_pin(Mode.THERMAL)
    gov.poll()
    assert gov.mode() == Mode.THERMAL
    gov.set_pin(None)
    # Auto mode never advanced because the pin suppressed _update_mode; back to ACTIVE default.
    assert gov.mode() == Mode.ACTIVE


def test_governor_acquire_delegates_to_arbiter() -> None:
    arb = OllamaArbiter(
        base_url="http://localhost:11434",
        http=_StubHttp(ps_payload={"models": [{"name": "qwen3-coder:30b", "size": _gb(20)}]}),
        budget_gb=22.0,
    )
    gov = ResourceGovernor(arbiter=arb)
    gov.acquire("llama3.2:3b")
    assert arb.http.post_calls  # type: ignore[attr-defined]
    assert arb.http.post_calls[0][1]["model"] == "qwen3-coder:30b"  # type: ignore[attr-defined]
