"""Unit tests for ``append_user_fact_to_md`` (Slice 5).

The appender must preserve user-curated content while idempotently managing
an ``## Auto-detected`` section for LLM-extracted facts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.memory.identity import loader


@pytest.fixture()
def iris_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / ".iris"
    monkeypatch.setattr(loader, "IRIS_HOME", home)
    monkeypatch.setattr(loader, "MEMORY_DIR", home / "memory")
    monkeypatch.setattr(loader, "USER_MD_PATH", home / "memory" / "user.md")
    return home


def test_creates_file_with_first_fact(iris_home: Path) -> None:
    assert loader.append_user_fact_to_md("name", "Robin", 0.9)
    text = (iris_home / "memory" / "user.md").read_text(encoding="utf-8")
    assert "# User Profile" in text
    assert "## Auto-detected" in text
    assert "- **name**: Robin  <!-- auto: confidence=0.90 -->" in text


def test_preserves_user_curated_content_above(iris_home: Path) -> None:
    user_md = iris_home / "memory" / "user.md"
    user_md.parent.mkdir(parents=True, exist_ok=True)
    user_md.write_text(
        "# User Profile\n\n" "## My Notes\n" "I prefer concise answers without preamble.\n",
        encoding="utf-8",
    )
    loader.append_user_fact_to_md("city", "Springfield", 0.85)
    text = user_md.read_text(encoding="utf-8")
    assert "## My Notes" in text
    assert "I prefer concise answers without preamble." in text
    assert "## Auto-detected" in text
    assert "- **city**: Springfield" in text


def test_idempotent_upsert_replaces_in_place(iris_home: Path) -> None:
    loader.append_user_fact_to_md("name", "Robin", 0.9)
    loader.append_user_fact_to_md("name", "Robin K", 0.95)
    text = (iris_home / "memory" / "user.md").read_text(encoding="utf-8")
    # Only one bullet for "name", and it has the new value.
    assert text.count("- **name**:") == 1
    assert "Robin K" in text
    assert "confidence=0.95" in text


def test_multiple_keys_coexist(iris_home: Path) -> None:
    loader.append_user_fact_to_md("name", "Robin", 0.9)
    loader.append_user_fact_to_md("city", "Springfield", 0.8)
    loader.append_user_fact_to_md("employer", "Acme", 0.7)
    text = (iris_home / "memory" / "user.md").read_text(encoding="utf-8")
    assert "- **name**: Robin" in text
    assert "- **city**: Springfield" in text
    assert "- **employer**: Acme" in text


def test_empty_key_or_value_is_noop(iris_home: Path) -> None:
    assert not loader.append_user_fact_to_md("", "Robin", 0.9)
    assert not loader.append_user_fact_to_md("name", "   ", 0.9)
    assert not (iris_home / "memory" / "user.md").exists()


def test_confidence_clamped(iris_home: Path) -> None:
    loader.append_user_fact_to_md("a", "x", 5.0)
    loader.append_user_fact_to_md("b", "y", -0.3)
    text = (iris_home / "memory" / "user.md").read_text(encoding="utf-8")
    assert "confidence=1.00" in text
    assert "confidence=0.00" in text


def test_curated_user_fact_blocks_auto_conflict(iris_home: Path) -> None:
    user_md = iris_home / "memory" / "user.md"
    user_md.parent.mkdir(parents=True, exist_ok=True)
    user_md.write_text(
        "# User Profile\n\n"
        "## Curated\n"
        "- **name**: Robin K\n\n"
        "## Auto-detected\n\n"
        "- **name**: Auto Name  <!-- auto: confidence=0.90 -->\n"
        "- **city**: Springfield  <!-- auto: confidence=0.80 -->\n\n"
        "## Notes\n"
        "Keep this section.\n",
        encoding="utf-8",
    )

    assert loader.append_user_fact_to_md("name", "Different", 0.95)
    text = user_md.read_text(encoding="utf-8")
    assert "- **name**: Robin K" in text
    assert "Auto Name" not in text
    assert "Different" not in text
    assert "- **city**: Springfield" in text
    assert "## Notes" in text
    assert "Keep this section." in text


def test_trailing_sections_survive_auto_upsert(iris_home: Path) -> None:
    user_md = iris_home / "memory" / "user.md"
    user_md.parent.mkdir(parents=True, exist_ok=True)
    user_md.write_text(
        "# User Profile\n\n"
        "## Auto-detected\n\n"
        "- **name**: Robin  <!-- auto: confidence=0.90 -->\n\n"
        "## Notes\n"
        "Manual note.\n",
        encoding="utf-8",
    )

    assert loader.append_user_fact_to_md("city", "Springfield", 0.85)
    text = user_md.read_text(encoding="utf-8")
    assert "- **name**: Robin" in text
    assert "- **city**: Springfield" in text
    assert "## Notes" in text
    assert "Manual note." in text


def test_inline_header_mention_in_intro_is_not_treated_as_boundary(iris_home: Path) -> None:
    """A curated intro that mentions ``## Auto-detected`` in prose must not be
    truncated when a fact is appended. The splitter used to partition on the first
    substring occurrence, mangling the user's curated head on every fact write."""
    user_md = iris_home / "memory" / "user.md"
    user_md.parent.mkdir(parents=True, exist_ok=True)
    user_md.write_text(
        "# User Profile\n\n"
        "_Everything above the `## Auto-detected` heading is curated by you._\n\n"
        "## Identity\n"
        "- **Name:** Robin\n"
        "- **Location:** Springfield, Illinois\n",
        encoding="utf-8",
    )

    assert loader.append_user_fact_to_md("greeting", "hello", 0.6)
    text = user_md.read_text(encoding="utf-8")
    # The curated intro + Identity survive intact (not truncated at the mention).
    assert "_Everything above the `## Auto-detected` heading is curated by you._" in text
    assert "## Identity" in text
    assert "- **Name:** Robin" in text
    assert "- **Location:** Springfield, Illinois" in text
    # And the fact still lands in a real Auto-detected section.
    assert "- **greeting**: hello" in text


def test_load_soul_strips_frontmatter_and_loads_agent_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    soul_path = tmp_path / "soul.md"
    soul_path.write_text(
        "---\nname: Atlas\n---\n\n# Atlas - Soul\n\nYou are Atlas.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "SOUL_PATH", soul_path)
    monkeypatch.setattr(loader, "_SOUL_DEFAULT", tmp_path / "missing.default.md")

    assert loader.load_agent_name() == "Atlas"
    soul = loader.load_soul()
    assert soul is not None
    assert soul.startswith("# Atlas - Soul")
    assert "name: Atlas" not in soul
