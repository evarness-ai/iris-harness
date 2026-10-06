"""Tests for heartbeat config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.heartbeat import HeartbeatConfigError, load_heartbeats


def test_load_heartbeats_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert load_heartbeats(tmp_path / "missing.yaml") == []


def test_load_heartbeats_returns_empty_for_empty_file(tmp_path: Path) -> None:
    cfg = tmp_path / "heartbeats.yaml"
    cfg.write_text("", encoding="utf-8")
    assert load_heartbeats(cfg) == []


def test_load_heartbeats_parses_definitions(tmp_path: Path) -> None:
    cfg = tmp_path / "heartbeats.yaml"
    cfg.write_text(
        """
heartbeats:
  - name: morning_briefing
    handler: morning_briefing
    schedule: "0 7 * * *"
    enabled: true
    description: Daily summary
    params:
      include_calendar: true
  - name: lint
    handler: wiki_lint
    schedule: "interval:60"
""",
        encoding="utf-8",
    )
    defs = load_heartbeats(cfg)
    assert len(defs) == 2
    assert defs[0].name == "morning_briefing"
    assert defs[0].schedule == "0 7 * * *"
    assert defs[0].params == {"include_calendar": True}
    assert defs[1].schedule == "interval:60"
    assert defs[1].enabled is True


def test_load_heartbeats_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    cfg = tmp_path / "heartbeats.yaml"
    cfg.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(HeartbeatConfigError):
        load_heartbeats(cfg)


def test_load_heartbeats_requires_name_handler_schedule(tmp_path: Path) -> None:
    cfg = tmp_path / "heartbeats.yaml"
    cfg.write_text(
        "heartbeats:\n  - name: only_name\n",
        encoding="utf-8",
    )
    with pytest.raises(HeartbeatConfigError):
        load_heartbeats(cfg)


def test_repository_heartbeats_yaml_loads() -> None:
    """The shipped config/heartbeats.yaml must be parseable."""
    repo_root = Path(__file__).resolve().parents[5]
    defs = load_heartbeats(repo_root / "config" / "heartbeats.yaml")
    names = {d.name for d in defs}
    assert "wiki_lint" in names
    assert "routine_tick" in names
    assert "notification_reminder_tick" in names
    assert "reminder_tick" not in names  # the markdown store's tick is retired (D14)


def test_load_heartbeats_reads_platforms(tmp_path: Path) -> None:
    path = tmp_path / "heartbeats.yaml"
    path.write_text(
        "heartbeats:\n"
        "  - {name: a, handler: a, schedule: 'interval:60', platforms: [darwin]}\n"
        "  - {name: b, handler: b, schedule: 'interval:60'}\n",
        encoding="utf-8",
    )
    a, b = load_heartbeats(path)
    assert a.platforms == ("darwin",)
    assert b.platforms == ()


def test_load_heartbeats_rejects_bad_platforms(tmp_path: Path) -> None:
    path = tmp_path / "heartbeats.yaml"
    path.write_text(
        "heartbeats:\n  - {name: a, handler: a, schedule: 'interval:60', platforms: darwin}\n",
        encoding="utf-8",
    )
    with pytest.raises(HeartbeatConfigError, match="platforms"):
        load_heartbeats(path)


def test_load_heartbeats_reads_the_owning_plugin(tmp_path: Path) -> None:
    cfg = tmp_path / "heartbeats.yaml"
    cfg.write_text(
        """
heartbeats:
  - {name: a, handler: a, schedule: "interval:60", plugin: " owner "}
  - {name: b, handler: b, schedule: "interval:60"}
""",
        encoding="utf-8",
    )
    a, b = load_heartbeats(cfg)
    assert (a.plugin, b.plugin) == ("owner", "")


def test_load_heartbeats_rejects_a_non_string_plugin(tmp_path: Path) -> None:
    cfg = tmp_path / "heartbeats.yaml"
    cfg.write_text(
        'heartbeats:\n  - {name: a, handler: a, schedule: "interval:60", plugin: [x]}\n',
        encoding="utf-8",
    )
    with pytest.raises(HeartbeatConfigError, match="'plugin'"):
        load_heartbeats(cfg)
