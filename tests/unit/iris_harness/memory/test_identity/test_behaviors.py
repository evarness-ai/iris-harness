"""Unit tests for behaviors loader + matcher (Slice 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.memory.identity import loader


@pytest.fixture()
def iris_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect ~/.iris/ to a tmp dir for isolated tests."""
    home = tmp_path / ".iris"
    monkeypatch.setattr(loader, "IRIS_HOME", home)
    monkeypatch.setattr(loader, "IDENTITY_DIR", home / "identity")
    monkeypatch.setattr(loader, "MEMORY_DIR", home / "memory")
    monkeypatch.setattr(loader, "BEHAVIORS_DIR", home / "behaviors")
    monkeypatch.setattr(loader, "SOUL_PATH", home / "identity" / "soul.md")
    monkeypatch.setattr(loader, "USER_MD_PATH", home / "memory" / "user.md")
    monkeypatch.setattr(loader, "ACTIVE_MD_PATH", home / "memory" / "active.md")
    monkeypatch.setattr(loader, "EPISODIC_MD_PATH", home / "memory" / "episodic.md")
    return home


def _write_behavior(home: Path, name: str, body: str = "Recipe body.") -> Path:
    path = home / "behaviors" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class TestBootstrapBehaviorsDir:
    def test_bootstrap_creates_dir_and_copies_defaults(self, iris_home: Path) -> None:
        loader.bootstrap_identity_files()
        assert (iris_home / "behaviors").is_dir()
        # Repo ships at least reminders.default.md — should land as reminders.md.
        assert (iris_home / "behaviors" / "reminders.md").exists()

    def test_bootstrap_does_not_overwrite_existing_behavior(self, iris_home: Path) -> None:
        target = iris_home / "behaviors" / "reminders.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("CUSTOM", encoding="utf-8")
        loader.bootstrap_identity_files()
        assert target.read_text(encoding="utf-8") == "CUSTOM"


class TestWriteBehavior:
    def test_round_trips_through_matcher(self, iris_home: Path) -> None:
        # A behavior written by write_behavior is matchable on the next read —
        # this is the contract the teach-a-preference capture relies on.
        loader.write_behavior(
            "day-overview",
            "Show the top 10 emails plus today's dues, to-dos and calendar events.",
            match_keywords=("how is my day", "how is my day today"),
            description="User-taught rule",
            source="taught",
        )
        match = loader.match_behavior("planner", "so how is my day today?")
        assert match is not None
        assert match.name == "day-overview"
        assert "top 10 emails" in match.body
        assert match.match_keywords == ("how is my day", "how is my day today")

    def test_reteaching_overwrites_in_place(self, iris_home: Path) -> None:
        loader.write_behavior("day-overview", "Old body.", match_keywords=("how is my day",))
        loader.write_behavior("day-overview", "New body.", match_keywords=("how is my day",))
        files = list((iris_home / "behaviors").glob("day-overview.md"))
        assert len(files) == 1
        assert "New body." in files[0].read_text(encoding="utf-8")
        assert "Old body." not in files[0].read_text(encoding="utf-8")

    def test_records_source_in_frontmatter(self, iris_home: Path) -> None:
        path = loader.write_behavior("r", "Body.", match_keywords=("k",), source="taught")
        assert "source: taught" in path.read_text(encoding="utf-8")


class TestListBehaviors:
    def test_empty_when_no_behaviors_dir(self, iris_home: Path) -> None:
        assert loader.list_behaviors() == []

    def test_skips_files_without_frontmatter(self, iris_home: Path) -> None:
        _write_behavior(iris_home, "broken", body="No frontmatter here.")
        assert loader.list_behaviors() == []

    def test_parses_frontmatter(self, iris_home: Path) -> None:
        _write_behavior(
            iris_home,
            "reminders",
            body=(
                "---\n"
                "name: reminders\n"
                "description: handle reminder asks\n"
                "match_intents: [calendar]\n"
                "match_keywords: [remind, follow up]\n"
                "---\n\n"
                "# Body\n\nDo the thing."
            ),
        )
        items = loader.list_behaviors()
        assert len(items) == 1
        b = items[0]
        assert b.name == "reminders"
        assert b.description == "handle reminder asks"
        assert b.match_intents == ("calendar",)
        assert b.match_keywords == ("remind", "follow up")
        assert "Do the thing." in b.body

    def test_keywords_lowercased(self, iris_home: Path) -> None:
        _write_behavior(
            iris_home,
            "x",
            body="---\nname: x\nmatch_keywords: [REMIND, Follow-Up]\n---\n\nbody",
        )
        b = loader.list_behaviors()[0]
        assert b.match_keywords == ("remind", "follow-up")


class TestMatchBehavior:
    @pytest.fixture(autouse=True)
    def _seed(self, iris_home: Path) -> None:
        _write_behavior(
            iris_home,
            "reminders",
            body=(
                "---\n"
                "name: reminders\n"
                "match_intents: [calendar]\n"
                "match_keywords: [remind, follow up]\n"
                "---\n\nrecipe"
            ),
        )

    def test_intent_match(self) -> None:
        b = loader.match_behavior("calendar", "anything")
        assert b is not None and b.name == "reminders"

    def test_keyword_match(self) -> None:
        b = loader.match_behavior("general", "please remind me tomorrow")
        assert b is not None and b.name == "reminders"

    def test_multi_word_keyword(self) -> None:
        b = loader.match_behavior("general", "let's follow up next week")
        assert b is not None and b.name == "reminders"

    def test_no_match(self) -> None:
        assert loader.match_behavior("general", "what is the weather") is None

    def test_intent_match_is_case_insensitive(self) -> None:
        b = loader.match_behavior("CALENDAR", "x")
        assert b is not None and b.name == "reminders"

    def test_empty_intent_falls_back_to_keywords(self) -> None:
        b = loader.match_behavior("", "remind me")
        assert b is not None and b.name == "reminders"
