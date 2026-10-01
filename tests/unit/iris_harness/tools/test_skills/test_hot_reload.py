"""Unit tests for :mod:`iris_harness.tools.skills.hot_reload`."""

from pathlib import Path
from unittest.mock import Mock

from watchdog.events import DirModifiedEvent, FileModifiedEvent

from iris_harness.tools.skills.hot_reload import ENV_FLAG, SkillHotReloader, _SkillChangeHandler
from iris_harness.tools.skills.registry import SkillRegistry


def test_hot_reloader_start_returns_false_when_env_disabled(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path)

    reloader = SkillHotReloader(registry=registry, repo_root=tmp_path)

    assert reloader.start() is False


def test_hot_reloader_start_returns_false_when_skills_root_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = SkillRegistry(tmp_path)
    monkeypatch.setenv(ENV_FLAG, "1")

    reloader = SkillHotReloader(registry=registry, repo_root=tmp_path)

    assert reloader.start() is False


def test_change_handler_rediscovers_only_for_watched_files(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path)
    discover = Mock(return_value=())
    handler = _SkillChangeHandler(registry=registry)

    directory_event = DirModifiedEvent(str(tmp_path / "config" / "skills"))
    ignored_file_event = FileModifiedEvent(str(tmp_path / "README.md"))
    manifest_event = FileModifiedEvent(
        str(tmp_path / "config" / "skills" / "demo" / "manifest.yaml")
    )

    discover.assert_not_called()
    registry.discover = discover

    handler.on_any_event(directory_event)
    handler.on_any_event(ignored_file_event)
    discover.assert_not_called()

    handler.on_any_event(manifest_event)
    discover.assert_called_once_with()
