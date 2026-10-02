"""``setup_state``: ``$IRIS_HOME/setup.json`` -- recording, resuming, resetting."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.system import setup_state


@pytest.fixture()
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("IRIS_HOME", str(home))
    return home


def test_fresh_state_has_no_steps_and_next_is_first(home: Path) -> None:
    state = setup_state.load_state()
    assert state.steps == {}
    assert state.next_step() == "preflight"
    assert state.mandatory_done() is False


def test_record_step_persists_and_reloads(home: Path) -> None:
    setup_state.record_step("preflight", "done")
    reloaded = setup_state.load_state()
    assert reloaded.is_recorded("preflight")
    assert reloaded.steps["preflight"].status == "done"
    assert reloaded.next_step() == "home_secret"


def test_mandatory_done_requires_both_mandatory_steps(home: Path) -> None:
    setup_state.record_step("preflight", "done")
    assert setup_state.load_state().mandatory_done() is False
    setup_state.record_step("home_secret", "done")
    assert setup_state.load_state().mandatory_done() is True


def test_skipped_optional_step_is_recorded_and_not_next(home: Path) -> None:
    for name in ("preflight", "home_secret", "services"):
        setup_state.record_step(name, "done")
    setup_state.record_step("telegram", "skipped", detail="declined")
    state = setup_state.load_state()
    assert state.is_recorded("telegram")
    assert state.next_step() == "email"


def test_needs_run_is_true_with_no_record_or_a_failed_one(home: Path) -> None:
    state = setup_state.load_state()
    assert state.needs_run("telegram") is True  # no record yet

    state = setup_state.record_step("telegram", "failed", detail="no message received")
    assert state.needs_run("telegram") is True  # failed, not a choice -- retry it


def test_needs_run_is_false_once_done_or_declined(home: Path) -> None:
    state = setup_state.record_step("telegram", "done", detail="chat_id=1")
    assert state.needs_run("telegram") is False

    state = setup_state.record_step("email", "skipped", detail="declined")
    assert state.needs_run("email") is False


def test_next_step_skips_a_declined_step_but_returns_a_failed_one(home: Path) -> None:
    setup_state.record_step("preflight", "done")
    setup_state.record_step("home_secret", "done")
    setup_state.record_step("services", "skipped", detail="declined")
    state = setup_state.record_step("telegram", "failed", detail="no message received")
    # services was declined (skip it); telegram failed, so it's next, not email.
    assert state.next_step() == "telegram"


def test_reset_clears_every_recorded_step(home: Path) -> None:
    setup_state.record_step("preflight", "done")
    setup_state.clear_state()
    state = setup_state.load_state()
    assert state.steps == {}
    assert not setup_state.setup_marker_path().exists()


def test_reset_with_no_prior_state_is_a_no_op(home: Path) -> None:
    setup_state.clear_state()  # must not raise when there's nothing to clear
    assert setup_state.load_state().steps == {}


def test_malformed_marker_file_is_treated_as_empty(home: Path) -> None:
    path = setup_state.setup_marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    assert setup_state.load_state().steps == {}


def test_unknown_status_in_file_is_dropped(home: Path) -> None:
    import json

    path = setup_state.setup_marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"steps": {"preflight": {"status": "bogus", "at": "x"}}}), encoding="utf-8"
    )
    assert setup_state.load_state().steps == {}
