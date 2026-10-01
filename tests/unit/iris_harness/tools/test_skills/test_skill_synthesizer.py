"""Unit tests for the skill synthesizer data shapes + parser."""

from __future__ import annotations

from iris_harness.tools.skills.skill_synthesizer import (
    SkillExample,
    SynthesizedSkill,
    build_skill_synth_prompt,
    parse_synthesized_skill,
)


def test_parse_valid() -> None:
    s = parse_synthesized_skill(
        '{"name": "weather-now", "description": "Current weather lookup.",'
        ' "when_to_use": "user asks the weather", "tool_description": "fetch weather",'
        ' "trigger_keywords": ["weather", "forecast"]}'
    )
    assert s is not None
    assert s.name == "weather-now"
    assert s.trigger_keywords == ("weather", "forecast")


def test_parse_tolerates_prose() -> None:
    s = parse_synthesized_skill('here:\n{"name": "x", "description": "y"}\nok')
    assert s is not None and s.name == "x"


def test_parse_requires_name_and_description() -> None:
    assert parse_synthesized_skill('{"name": "x"}') is None
    assert parse_synthesized_skill('{"description": "y"}') is None
    assert parse_synthesized_skill("not json") is None


def test_manifest_overrides() -> None:
    s = SynthesizedSkill(
        name="n", description="d", when_to_use="w", tool_description="t", trigger_keywords=("k",)
    )
    ov = s.manifest_overrides(slug="my-skill")
    assert ov["description"] == "d"
    assert ov["when_to_use"] == "w"
    assert ov["trigger_keywords"] == ["k"]
    assert ov["tools"][0]["name"] == "my-skill_tool"
    assert ov["tools"][0]["description"] == "t"


def test_prompt_includes_examples() -> None:
    prompt = build_skill_synth_prompt(
        intent="finance_summary",
        agent_type="finance",
        examples=[SkillExample(query="net worth?", response="up 3%")],
    )
    assert "finance_summary" in prompt
    assert "net worth?" in prompt
    assert "up 3%" in prompt
