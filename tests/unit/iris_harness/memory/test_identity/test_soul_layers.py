"""SOUL.md is split at read time: L0 core in every prompt, the rest via iris_doc.

The file is the user's, so nothing here rewrites it — the split is by section
heading, driven by config/identity/soul_layers.yaml. Anything not listed as
extended stays in L0: silently hiding a section someone wrote themselves would be
the wrong default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.memory.identity import loader

SOUL = """---
name: IRIS
---

# IRIS — Soul

I am the preamble.

## Identity
Core identity text.

## Operational primer
Pipeline internals that go stale.

## Security & privacy
Never leak secrets.

## Tool selection policy
Long tool policy prose.

## Do not
Never invent facts.
"""


@pytest.fixture
def souled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / "workspace").mkdir(parents=True)
    (home / "workspace" / "SOUL.md").write_text(SOUL, encoding="utf-8")
    monkeypatch.setattr(loader, "SOUL_PATH", home / "workspace" / "SOUL.md")
    monkeypatch.setattr(loader, "_SOUL_LAYERS_CACHE", None)

    cfg = tmp_path / "config" / "identity"
    cfg.mkdir(parents=True)
    (cfg / "soul_layers.yaml").write_text(
        "extended_sections:\n"
        "  - Operational primer\n"
        "  - Tool selection policy\n"
        "tool_policy_summary: |\n"
        "  ## Tool policy (summary)\n"
        "  - Prefer real data over memory.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(tmp_path / "config"))
    return home


def test_core_keeps_identity_and_hard_rules(souled: Path) -> None:
    core = loader.load_soul_core() or ""

    assert "I am the preamble." in core
    assert "Core identity text." in core
    assert "Never leak secrets." in core
    assert "Never invent facts." in core


def test_core_drops_the_extended_sections(souled: Path) -> None:
    core = loader.load_soul_core() or ""

    assert "Pipeline internals that go stale." not in core
    assert "Long tool policy prose." not in core


def test_core_carries_the_short_tool_policy_from_config(souled: Path) -> None:
    core = loader.load_soul_core() or ""

    assert "## Tool policy (summary)" in core
    assert "Prefer real data over memory." in core


def test_extended_has_exactly_what_the_prompt_lost(souled: Path) -> None:
    extended = loader.load_soul_extended() or ""

    assert "Pipeline internals that go stale." in extended
    assert "Long tool policy prose." in extended
    assert "Core identity text." not in extended


def test_the_split_saves_tokens(souled: Path) -> None:
    from iris_harness.llm.budget import estimate_tokens

    full = loader.load_soul() or ""
    core = loader.load_soul_core() or ""

    assert estimate_tokens(core) < estimate_tokens(full)


def test_an_unlisted_section_stays_in_the_prompt(souled: Path, tmp_path: Path) -> None:
    """A user's own section is never hidden by a config that doesn't mention it."""
    path = tmp_path / "home" / "workspace" / "SOUL.md"
    path.write_text(SOUL + "\n## My own rules\nAlways answer in Tamil first.\n", encoding="utf-8")

    assert "Always answer in Tamil first." in (loader.load_soul_core() or "")


def test_missing_config_keeps_the_whole_soul(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No config = pre-split behavior, not an empty prompt."""
    home = tmp_path / "home2"
    (home / "workspace").mkdir(parents=True)
    (home / "workspace" / "SOUL.md").write_text(SOUL, encoding="utf-8")
    monkeypatch.setattr(loader, "SOUL_PATH", home / "workspace" / "SOUL.md")
    monkeypatch.setattr(loader, "_SOUL_LAYERS_CACHE", None)
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(tmp_path / "nonexistent"))
    # ...and no shipped copy to fall back to (checkout or package: foundation/paths.py)
    monkeypatch.setattr(loader, "default_config_dir", lambda: tmp_path / "nonexistent")

    core = loader.load_soul_core() or ""

    assert "Pipeline internals that go stale." in core
    assert loader.load_soul_extended() is None


def test_no_soul_file_means_no_core_and_no_extended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "SOUL_PATH", tmp_path / "absent.md")
    monkeypatch.setattr(loader, "_LEGACY_SOUL_PATH", tmp_path / "absent-legacy.md")
    monkeypatch.setattr(loader, "_SOUL_LAYERS_CACHE", None)

    assert loader.load_soul_core() is None
    assert loader.load_soul_extended() is None
