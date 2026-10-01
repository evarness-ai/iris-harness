"""Tests for the agentic learning analyst (ADR-0069 #4, slice 2).

The analyst is the interpretation layer: these cover tolerant JSON parsing of the
model's response, the advisory data model, the round-trip used for persistence,
and the latest-run store. No real LLM — ``analyze_learning`` takes an injected
invoke.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from iris_harness.services.learning.analyst import (
    LearningAnalysis,
    Recommendation,
    analyze_learning,
    parse_learning_analysis,
    render_text,
)
from iris_harness.services.learning.intelligence import build_intelligence
from iris_harness.services.learning.store import LearningMetricsStore

_NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


def _parse(raw: str) -> LearningAnalysis | None:
    return parse_learning_analysis(raw, generated_at=_NOW, window_hours=168.0, model="test-model")


def test_parse_plain_json() -> None:
    raw = json.dumps(
        {
            "summary": "email on tier1 looks weak.",
            "recommendations": [
                {
                    "title": "Route email to tier2",
                    "finding": "email @ tier1 completion 60% (n=20)",
                    "action": "set email intent start tier to tier2",
                    "evidence": ["email@tier1: 60% done"],
                    "confidence": "high",
                }
            ],
        }
    )
    analysis = _parse(raw)
    assert analysis is not None
    assert analysis.summary.startswith("email")
    assert len(analysis.recommendations) == 1
    rec = analysis.recommendations[0]
    assert rec.title == "Route email to tier2"
    assert rec.confidence == "high"
    assert rec.evidence == ("email@tier1: 60% done",)


def test_parse_strips_code_fence_and_prose() -> None:
    raw = (
        "Here is my analysis:\n```json\n"
        '{"summary": "ok", "recommendations": []}\n'
        "```\nHope that helps."
    )
    analysis = _parse(raw)
    assert analysis is not None
    assert analysis.summary == "ok"
    assert analysis.recommendations == ()


def test_parse_returns_none_without_json() -> None:
    assert _parse("I could not find anything to report.") is None


def test_bad_confidence_defaults_low_and_string_evidence_coerced() -> None:
    raw = json.dumps(
        {
            "summary": "s",
            "recommendations": [
                {
                    "title": "t",
                    "finding": "f",
                    "action": "a",
                    "evidence": "single",
                    "confidence": "wat",
                }
            ],
        }
    )
    rec = _parse(raw).recommendations[0]  # type: ignore[union-attr]
    assert rec.confidence == "low"
    assert rec.evidence == ("single",)


def test_parses_structured_target() -> None:
    raw = json.dumps(
        {
            "summary": "s",
            "recommendations": [
                {
                    "title": "t",
                    "finding": "f",
                    "action": "a",
                    "confidence": "medium",
                    "target": {"metric": "completion_rate", "intent": "email", "tier": "tier1"},
                }
            ],
        }
    )
    rec = _parse(raw).recommendations[0]  # type: ignore[union-attr]
    assert rec.target is not None
    assert rec.target.metric == "completion_rate"
    assert rec.target.intent == "email"
    # An unknown metric is dropped (None), not trusted.
    raw2 = json.dumps(
        {
            "summary": "s",
            "recommendations": [
                {
                    "title": "t",
                    "finding": "f",
                    "action": "a",
                    "confidence": "low",
                    "target": {"metric": "made_up_metric"},
                }
            ],
        }
    )
    assert _parse(raw2).recommendations[0].target is None  # type: ignore[union-attr]


def test_recommendations_capped() -> None:
    recs = [
        {"title": f"r{i}", "finding": "f", "action": "a", "confidence": "low"} for i in range(10)
    ]
    analysis = _parse(json.dumps({"summary": "s", "recommendations": recs}))
    assert analysis is not None
    assert len(analysis.recommendations) <= 5


def test_analyze_learning_uses_injected_invoke(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    captured: dict[str, str] = {}

    def fake_invoke(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps({"summary": "all good", "recommendations": []})

    report = build_intelligence(store, now=_NOW)
    analysis = analyze_learning(report, invoke=fake_invoke, model="local-x")
    assert analysis is not None
    assert analysis.summary == "all good"
    assert analysis.model == "local-x"
    # The report's measured numbers are handed to the model.
    assert "Learning intelligence" in captured["user"]


def test_round_trip_as_dict_from_dict() -> None:
    analysis = LearningAnalysis(
        generated_at=_NOW,
        window_hours=168.0,
        model="m",
        summary="s",
        recommendations=(
            Recommendation(
                title="t", finding="f", action="a", evidence=("e1", "e2"), confidence="medium"
            ),
        ),
    )
    restored = LearningAnalysis.from_dict(analysis.as_dict())
    assert restored == analysis


def test_render_text_lists_recommendations() -> None:
    analysis = LearningAnalysis(
        generated_at=_NOW,
        window_hours=168.0,
        model="m",
        summary="overview",
        recommendations=(
            Recommendation(
                title="Route email up",
                finding="weak tier1",
                action="tier2",
                evidence=("x",),
                confidence="high",
            ),
        ),
    )
    text = render_text(analysis)
    assert "overview" in text
    assert "Route email up" in text
    assert "[high]" in text


def test_store_save_and_latest_analysis(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    assert store.latest_analysis() is None

    payload = {"generated_at": _NOW.isoformat(), "summary": "first", "recommendations": []}
    store.save_analysis(payload)
    assert store.latest_analysis() == payload

    # Singleton: a second save replaces, not appends.
    payload2 = {"generated_at": _NOW.isoformat(), "summary": "second", "recommendations": []}
    store.save_analysis(payload2)
    assert store.latest_analysis() == payload2
