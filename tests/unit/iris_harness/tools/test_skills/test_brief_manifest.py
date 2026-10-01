"""Tests for the `kind: brief` skill manifest extension."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from iris_harness.tools.skills.loader import load_skill_manifest, load_skill_package
from iris_harness.tools.skills.models import (
    BriefLiteralSlot,
    BriefSpec,
    BriefToolSlot,
    SkillManifest,
)


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    skill_dir = tmp_path / "test-brief"
    skill_dir.mkdir()
    (skill_dir / "manifest.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    return skill_dir


def _valid_payload() -> dict:
    return {
        "name": "test-brief",
        "version": "0.1.0",
        "description": "Test brief skill.",
        "author": "iris",
        "license": "Apache-2.0",
        "kind": "brief",
        "brief": {
            "subject": "Test Subject",
            "recipient": "user",
            "uses": ["other-skill"],
            "layout": "Hello {{greeting}}",
            "slots": {
                "greeting": {"kind": "literal", "value": "world"},
            },
        },
    }


def test_brief_manifest_round_trips_from_yaml(tmp_path: Path) -> None:
    skill_dir = _write_manifest(tmp_path, _valid_payload())
    manifest = load_skill_manifest(skill_dir)
    assert manifest.kind == "brief"
    assert manifest.brief is not None
    assert manifest.brief.subject == "Test Subject"
    assert manifest.brief.uses == ("other-skill",)
    assert list(manifest.brief.slots) == ["greeting"]
    slot = manifest.brief.slots["greeting"]
    assert isinstance(slot, BriefLiteralSlot)
    assert slot.value == "world"


def test_brief_manifest_rejects_kind_brief_without_brief_section(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["brief"]
    skill_dir = _write_manifest(tmp_path, payload)
    with pytest.raises(ValueError, match="kind='brief'"):
        load_skill_manifest(skill_dir)


def test_brief_manifest_rejects_brief_section_without_kind(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["kind"] = None
    skill_dir = _write_manifest(tmp_path, payload)
    with pytest.raises(ValueError, match="kind='brief'"):
        load_skill_manifest(skill_dir)


def test_brief_spec_rejects_tool_slot_outside_uses() -> None:
    with pytest.raises(ValueError, match="not in brief.uses"):
        BriefSpec(
            subject="x",
            uses=("allowed-skill",),
            layout="{{out}}",
            slots={
                "out": BriefToolSlot(
                    kind="tool",
                    skill="other-skill",
                    tool="some_tool",
                ),
            },
        )


def test_brief_spec_rejects_layout_placeholder_with_no_slot() -> None:
    with pytest.raises(ValueError, match="unknown slots"):
        BriefSpec(
            subject="x",
            layout="{{missing}}",
            slots={},
        )


def test_brief_spec_accepts_tool_slot_inside_uses() -> None:
    spec = BriefSpec(
        subject="x",
        uses=("allowed",),
        layout="{{out}}",
        slots={
            "out": BriefToolSlot(kind="tool", skill="allowed", tool="some_tool"),
        },
    )
    assert spec.uses == ("allowed",)


def test_brief_package_is_loadable_without_tools_py(tmp_path: Path) -> None:
    skill_dir = _write_manifest(tmp_path, _valid_payload())
    package = load_skill_package(tmp_path, skill_dir)
    assert package.is_loadable is True
    assert package.tool_classes == ()
    assert package.manifest.kind == "brief"


def test_non_brief_manifest_remains_unaffected() -> None:
    manifest = SkillManifest(
        name="regular",
        version="0.1.0",
        description="A regular skill.",
        author="iris",
        license="Apache-2.0",
    )
    assert manifest.kind is None
    assert manifest.brief is None
