"""The owner's IRIS_* changes over the deploy's environment, and the restart request."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.foundation.settings.catalog import SettingDeclaration
from iris_harness.foundation.settings.env_overrides import (
    ENV_SECTION,
    SettingValueError,
    _reset_for_tests,
    apply_env_overrides,
    clear_env_override,
    deploy_value,
    normalize,
    set_env_override,
)
from iris_harness.foundation.settings.restart import (
    request_restart,
    restart_requested_after,
    watch_for_restart,
)

ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture
def store(tmp_path: Path) -> SettingsStore:
    return SettingsStore(db_path=tmp_path / SETTINGS_DB_NAME)


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _reset_for_tests()
    for name in ("IRIS_TEST_A", "IRIS_TEST_B"):
        monkeypatch.delenv(name, raising=False)
    yield
    _reset_for_tests()


def _decl(kind: str) -> SettingDeclaration:
    return SettingDeclaration.model_validate(
        {
            "kind": kind,
            "applies": "now",
            "label": "x",
            "description": "x.",
            "tab": "advanced",
        }
    )


@pytest.mark.parametrize(
    ("kind", "raw", "out"),
    [
        ("bool", True, "1"),
        ("bool", "off", "0"),
        ("bool", "YES", "1"),
        ("int", "12", "12"),
        ("int", 7, "7"),
        ("float", "0.5", "0.5"),
        ("str", "  gpt-5-mini ", "gpt-5-mini"),
        ("list", " a, ,b ,c", "a,b,c"),
        ("list", " , ", ""),
    ],
)
def test_values_are_normalised_to_what_the_readers_expect(kind: str, raw: object, out: str) -> None:
    assert normalize(_decl(kind), raw) == out


@pytest.mark.parametrize(
    ("kind", "raw"),
    [
        ("bool", "maybe"),
        ("int", "1.5"),
        ("int", True),
        ("float", "lots"),
        ("str", "a\nb"),
        ("str", ""),
    ],
)
def test_values_that_do_not_fit_are_refused(kind: str, raw: object) -> None:
    with pytest.raises(SettingValueError):
        normalize(_decl(kind), raw)


def test_set_applies_now_and_reset_returns_to_the_deploys_value(
    store: SettingsStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_TEST_A", "from-server-env")

    set_env_override("IRIS_TEST_A", "from-app", actor="device:x", store=store)
    assert os.environ["IRIS_TEST_A"] == "from-app"
    assert store.get(ENV_SECTION, "IRIS_TEST_A") == "from-app"
    assert deploy_value("IRIS_TEST_A") == "from-server-env"

    change = clear_env_override("IRIS_TEST_A", actor="device:x", store=store)
    assert change is not None and change.new == "from-server-env"
    assert os.environ["IRIS_TEST_A"] == "from-server-env"
    assert clear_env_override("IRIS_TEST_A", actor="s", store=store) is None


def test_reset_of_a_setting_the_deploy_never_set_unsets_it(store: SettingsStore) -> None:
    set_env_override("IRIS_TEST_B", "1", actor="s", store=store)
    clear_env_override("IRIS_TEST_B", actor="s", store=store)
    assert "IRIS_TEST_B" not in os.environ


def test_saved_overrides_apply_to_a_new_process(
    store: SettingsStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart or fresh container: a new process imports the server package."""
    set_env_override("IRIS_TEST_A", "saved", actor="s", store=store)
    env = {**os.environ, "IRIS_DATA_DIR": str(store.db_path.parent), "IRIS_TEST_A": "deploy"}
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    # Fixed argv: this interpreter and a constant program.
    argv = [
        sys.executable,
        "-c",
        "import os, iris_harness.server; print(os.environ['IRIS_TEST_A'])",
    ]
    out = subprocess.run(  # noqa: S603 - this interpreter and a constant program
        argv,  # this interpreter and a constant program
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "saved"


def test_apply_ignores_non_iris_names_and_survives_a_broken_store(
    store: SettingsStore, tmp_path: Path
) -> None:
    store.set(ENV_SECTION, "PATH", "/evil", old=None, actor="s")
    store.set(ENV_SECTION, "IRIS_TEST_A", "ok", old=None, actor="s")
    path_before = os.environ["PATH"]

    apply_env_overrides(store)

    assert os.environ["PATH"] == path_before
    assert os.environ["IRIS_TEST_A"] == "ok"
    broken = SettingsStore(db_path=tmp_path / "not-a-db")
    broken.db_path.write_text("garbage")
    assert apply_env_overrides(broken) == {}


def test_a_restart_request_is_seen_only_by_processes_started_before_it(
    store: SettingsStore,
) -> None:
    before = datetime.now(UTC) - timedelta(minutes=1)
    assert not restart_requested_after(store, before)
    at = request_restart(store, actor="device:x")
    assert restart_requested_after(store, before)
    assert not restart_requested_after(store, at + timedelta(seconds=1))


def test_the_watcher_exits_the_process_when_a_restart_is_requested(
    store: SettingsStore,
) -> None:
    exited: list[int] = []
    started = datetime.now(UTC) - timedelta(seconds=5)
    request_restart(store, actor="s")

    thread = watch_for_restart(
        store, started_at=started, exit_process=exited.append, poll_seconds=0.01
    )
    thread.join(timeout=2)

    assert exited == [0]


@pytest.mark.parametrize(
    ("supervised", "opt", "exits"),
    [
        ("1", None, True),
        ("1", "1", True),
        ("1", "0", False),
        ("0", None, False),
        (None, None, False),
    ],
)
def test_only_a_supervised_process_not_opted_out_exits_on_request(
    monkeypatch: pytest.MonkeyPatch, supervised: str | None, opt: str | None, exits: bool
) -> None:
    from iris_harness.foundation.settings.restart import exits_on_restart_request

    for name, value in (("IRIS_SUPERVISED", supervised), ("IRIS_RESTART_ON_REQUEST", opt)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert exits_on_restart_request() is exits
