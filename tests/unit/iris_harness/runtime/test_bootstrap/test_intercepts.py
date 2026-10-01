"""Unit tests for the declared intercept chain loader (Phase 2)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.runtime.intercepts import (
    DEFAULT_INTERCEPTS,
    InterceptSpec,
    load_intercept_chain,
)


def test_committed_yaml_matches_default_chain() -> None:
    # The shipped config/intercepts.yaml must stay in sync with the hardcoded
    # fallback so behavior is identical whether or not the file is present.
    chain = load_intercept_chain(Path("config/intercepts.yaml"))
    assert [s.name for s in chain] == [s.name for s in DEFAULT_INTERCEPTS]
    assert [s.handler for s in chain] == [s.handler for s in DEFAULT_INTERCEPTS]
    assert [s.trace_text for s in chain] == [s.trace_text for s in DEFAULT_INTERCEPTS]
    assert [s.trace_fields for s in chain] == [s.trace_fields for s in DEFAULT_INTERCEPTS]
    assert [s.passes_channel for s in chain] == [s.passes_channel for s in DEFAULT_INTERCEPTS]


def test_missing_file_falls_back_to_default(tmp_path: Path) -> None:
    assert load_intercept_chain(tmp_path / "nope.yaml") == DEFAULT_INTERCEPTS


def test_malformed_yaml_falls_back_to_default(tmp_path: Path) -> None:
    bad = tmp_path / "intercepts.yaml"
    bad.write_text("intercepts: [unclosed\n", encoding="utf-8")
    assert load_intercept_chain(bad) == DEFAULT_INTERCEPTS


def test_no_intercepts_key_falls_back(tmp_path: Path) -> None:
    empty = tmp_path / "intercepts.yaml"
    empty.write_text("other: 1\n", encoding="utf-8")
    assert load_intercept_chain(empty) == DEFAULT_INTERCEPTS


def test_disabled_entries_are_dropped(tmp_path: Path) -> None:
    cfg = tmp_path / "intercepts.yaml"
    cfg.write_text(
        "intercepts:\n" "  - name: time_date\n" "  - name: dues_request\n" "    enabled: false\n",
        encoding="utf-8",
    )
    chain = load_intercept_chain(cfg)
    assert [s.name for s in chain] == ["time_date"]
    # handler defaults to _handle_<name>_turn when omitted.
    assert chain[0].handler == "_handle_time_date_turn"


def test_custom_order_is_honored(tmp_path: Path) -> None:
    cfg = tmp_path / "intercepts.yaml"
    cfg.write_text(
        "intercepts:\n  - name: dues_request\n  - name: time_date\n",
        encoding="utf-8",
    )
    chain = load_intercept_chain(cfg)
    assert [s.name for s in chain] == ["dues_request", "time_date"]


def test_spec_defaults() -> None:
    spec = InterceptSpec("x", "_handle_x_turn")
    assert spec.enabled is True
    assert spec.passes_channel is False
    assert spec.trace_text is None
    assert spec.trace_fields == ()
