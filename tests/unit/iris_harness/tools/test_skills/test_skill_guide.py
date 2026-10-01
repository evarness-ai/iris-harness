"""The worked skill in docs/guides/writing-a-skill.md builds and runs.

The unit-converter skill used to live in ``examples/skills/`` beside two other toy
skills; OSS plan R18 folded it into the guide and removed the others. The guide's two
code blocks (``# manifest.yaml`` and ``# tools.py``) are the skill: this test writes
them into a skill directory, loads it the way the ``SkillRegistry`` does and runs the
tool, so the guide cannot rot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from iris_harness.tools.skills.loader import load_skill_manifest, load_skill_tool_classes

_ROOT = Path(__file__).resolve().parents[5]
_GUIDE = _ROOT / "docs" / "guides" / "writing-a-skill.md"
_BLOCK = re.compile(r"```(?:yaml|python)\n# (manifest\.yaml|tools\.py)\n(.*?)```", re.DOTALL)


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    blocks = dict(_BLOCK.findall(_GUIDE.read_text(encoding="utf-8")))
    assert set(blocks) == {"manifest.yaml", "tools.py"}, "the guide's two skill files"
    directory = tmp_path / "unit-converter"
    directory.mkdir()
    for name, body in blocks.items():
        (directory / name).write_text(body, encoding="utf-8")
    return directory


def test_the_guides_skill_loads(skill_dir: Path) -> None:
    manifest = load_skill_manifest(skill_dir)
    assert manifest.name == "unit-converter"
    [tool_cls] = load_skill_tool_classes(skill_dir)
    tool = tool_cls()
    assert tool.name == "convert_units"
    assert tool.args_schema is not None


def test_the_guides_skill_runs(skill_dir: Path) -> None:
    tool = load_skill_tool_classes(skill_dir)[0]()
    assert tool._run(value=100, from_unit="km", to_unit="mi")["value"] == pytest.approx(62.137119)
    assert tool._run(value=0, from_unit="c", to_unit="f")["value"] == pytest.approx(32.0)
    assert "error" in tool._run(value=1, from_unit="m", to_unit="c")  # cross-kind rejected
