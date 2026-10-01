"""Unit tests for the reworked (success-driven + synthesis) crystallizer."""

from __future__ import annotations

from pathlib import Path

import yaml

from iris_harness.services.learning.crystallizer import SkillCrystallizer
from iris_harness.services.learning.store import LearningMetricsStore
from iris_harness.tools.skills.skill_synthesizer import SkillExample, SynthesizedSkill


def _store(tmp_path: Path) -> LearningMetricsStore:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    return store


def _successes(store: LearningMetricsStore, *, intent: str, agent: str, count: int) -> None:
    for _ in range(count):
        store.record_signal(
            source="chat",
            metric_name="task_completed",
            value=1.0,
            success=True,
            metadata={"intent": intent},
            resolved_agent=agent,
        )


def _corrections(store: LearningMetricsStore, *, intent: str, count: int) -> None:
    for _ in range(count):
        store.record_signal(
            source="chat",
            metric_name="user_correction",
            value=1.0,
            success=False,
            metadata={"intent": intent},
        )


def _crystallizer(store: LearningMetricsStore, tmp_path: Path, **kw: object) -> SkillCrystallizer:
    # No example mining by default in tests (avoid touching real session logs).
    kw.setdefault("turn_reader", lambda: [])
    return SkillCrystallizer(store=store, repo_root=tmp_path, min_occurrences=5, **kw)  # type: ignore[arg-type]


def test_proposes_on_clean_recurring_successes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _successes(store, intent="finance_summary", agent="finance", count=6)

    report = _crystallizer(store, tmp_path).crystallize()

    assert len(report.proposals) == 1
    manifest_path = tmp_path / report.proposals[0].manifest_path
    assert manifest_path.exists()


def test_does_not_propose_from_errors_alone(tmp_path: Path) -> None:
    # Errors are no longer the trigger — only clean successes are.
    store = _store(tmp_path)
    for _ in range(6):
        store.record_signal(
            source="chat",
            metric_name="response_has_errors",
            value=1.0,
            success=False,
            metadata={"intent": "broken", "agent_type": "email"},
        )
    report = _crystallizer(store, tmp_path).crystallize()
    assert report.proposals == ()


def test_quality_gate_skips_frequently_corrected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _successes(store, intent="flaky", agent="general", count=6)
    _corrections(store, intent="flaky", count=6)  # 50% correction rate > 0.25 gate
    report = _crystallizer(store, tmp_path).crystallize()
    assert report.proposals == ()


def test_low_correction_rate_passes_gate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _successes(store, intent="solid", agent="general", count=10)
    _corrections(store, intent="solid", count=1)  # ~9% < 0.25
    report = _crystallizer(store, tmp_path).crystallize()
    assert len(report.proposals) == 1


def test_skips_existing_manifest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _successes(store, intent="finance_summary", agent="finance", count=6)
    cz = _crystallizer(store, tmp_path)
    cz.crystallize()
    second = cz.crystallize()
    assert second.proposals == ()
    assert len(second.skipped_existing) == 1


def test_ignores_low_occurrence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _successes(store, intent="rare", agent="email", count=2)
    report = _crystallizer(store, tmp_path).crystallize()
    assert report.proposals == ()


class _FakeSynth:
    def synthesize(self, *, intent, agent_type, examples):  # type: ignore[no-untyped-def]
        return SynthesizedSkill(
            name="finance-summary",
            description="Summarize the user's finances on request.",
            when_to_use="when the user asks for a finance summary",
            tool_description="Produce a per-currency net-worth summary.",
            trigger_keywords=("finance", "summary"),
        )


def test_synthesis_writes_real_manifest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _successes(store, intent="finance_summary", agent="finance", count=6)
    report = _crystallizer(store, tmp_path, synthesizer=_FakeSynth()).crystallize()

    assert len(report.proposals) == 1
    manifest = yaml.safe_load((tmp_path / report.proposals[0].manifest_path).read_text())
    assert manifest["description"] == "Summarize the user's finances on request."
    assert manifest["when_to_use"] == "when the user asks for a finance summary"
    assert manifest["tools"][0]["description"] == "Produce a per-currency net-worth summary."
    # the synthesized name carries into the proposal
    assert report.proposals[0].skill_name == "finance-summary"


def test_mines_examples_for_synthesis(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _successes(store, intent="finance_summary", agent="finance", count=6)
    seen: dict[str, object] = {}

    class _CapturingSynth:
        def synthesize(self, *, intent, agent_type, examples):  # type: ignore[no-untyped-def]
            seen["examples"] = list(examples)
            return None  # fall back to scaffold

    from iris_harness.foundation.observability.session_log import TurnRecord

    turns = [
        TurnRecord(
            session_id="s",
            turn_id="t",
            intent="finance_summary",
            agent_type="finance",
            query="how's my net worth?",
            response="Up 3% this month.",
            has_errors=False,
        )
    ]
    cz = SkillCrystallizer(
        store=store,
        repo_root=tmp_path,
        min_occurrences=5,
        synthesizer=_CapturingSynth(),
        turn_reader=lambda: turns,
    )
    cz.crystallize()
    examples = seen.get("examples")
    assert examples and isinstance(examples[0], SkillExample)
    assert examples[0].query == "how's my net worth?"


def test_real_path_scans_per_intent_with_filter(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # With no turn_reader injected, the crystallizer mines examples via an
    # intent-FILTERED scan per target intent (so a low-volume intent isn't crowded
    # out). Assert iter_recent_turns is invoked with the intent kwarg.
    import iris_harness.services.learning.crystallizer as cz_mod
    from iris_harness.foundation.observability.session_log import TurnRecord

    store = _store(tmp_path)
    _successes(store, intent="calendar", agent="calendar", count=6)

    calls: list[str | None] = []

    def fake_scan(*, limit, intent=None, scan_cap=100_000, log_dir=None, **kw):  # type: ignore[no-untyped-def]
        calls.append(intent)
        return [
            TurnRecord(
                session_id="s",
                turn_id="t",
                intent=intent or "",
                agent_type="calendar",
                query="any meeting tomorrow?",
                response="yes",
                has_errors=False,
            )
        ]

    monkeypatch.setattr(cz_mod, "iter_recent_turns", fake_scan)

    captured: dict[str, object] = {}

    class _Synth:
        def synthesize(self, *, intent, agent_type, examples):  # type: ignore[no-untyped-def]
            captured["examples"] = list(examples)
            return None

    cz = SkillCrystallizer(store=store, repo_root=tmp_path, min_occurrences=5, synthesizer=_Synth())
    cz.crystallize()

    assert calls == ["calendar"]  # filtered scan, not an unfiltered bulk read
    examples = captured.get("examples")
    assert examples and examples[0].query == "any meeting tomorrow?"


# ── replay-eval gate (ADR-0070) ──────────────────────────────────────────────


def _verdict(*, passed: bool, runs: int, reason: str = "v"):  # type: ignore[no-untyped-def]
    from iris_harness.services.learning.preflight import PreflightVerdict

    return PreflightVerdict(
        passed=passed, completion_rate=1.0 if passed else 0.0, runs=runs, queries=3, reason=reason
    )


def test_preflight_pass_lands_as_proposed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _successes(store, intent="finance_summary", agent="finance", count=6)
    report = _crystallizer(
        store,
        tmp_path,
        preflight=lambda _p: _verdict(passed=True, runs=2, reason="ok 100%"),
    ).crystallize()

    assert len(report.proposals) == 1
    assert report.rejected == ()
    assert report.proposals[0].status == "proposed"
    assert report.proposals[0].preflight_reason == "ok 100%"


def test_preflight_quality_failure_is_quarantined(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _successes(store, intent="finance_summary", agent="finance", count=6)
    report = _crystallizer(
        store,
        tmp_path,
        preflight=lambda _p: _verdict(passed=False, runs=3, reason="completion 30% < 80%"),
    ).crystallize()

    assert report.proposals == ()
    assert len(report.rejected) == 1
    assert report.rejected[0].status == "preflight_failed"
    assert "30%" in (report.rejected[0].preflight_reason or "")
    # persisted to disk with the failed status (inert; registry excludes auto/)
    manifest = yaml.safe_load((tmp_path / report.rejected[0].manifest_path).read_text())
    assert manifest is not None


def test_preflight_inconclusive_does_not_block(tmp_path: Path) -> None:
    # runs == 0 -> no workload / insufficient evidence, a data gap, not a quality
    # signal. The proposal still lands as 'proposed'.
    store = _store(tmp_path)
    _successes(store, intent="finance_summary", agent="finance", count=6)
    report = _crystallizer(
        store,
        tmp_path,
        preflight=lambda _p: _verdict(passed=False, runs=0, reason="no replay workload"),
    ).crystallize()

    assert len(report.proposals) == 1
    assert report.rejected == ()
    assert report.proposals[0].status == "proposed"
    assert "inconclusive" in (report.proposals[0].preflight_reason or "")


def test_preflight_error_does_not_block(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _successes(store, intent="finance_summary", agent="finance", count=6)

    def boom(_p):  # type: ignore[no-untyped-def]
        raise RuntimeError("ollama down")

    report = _crystallizer(store, tmp_path, preflight=boom).crystallize()

    assert len(report.proposals) == 1
    assert report.rejected == ()
    assert report.proposals[0].status == "proposed"
