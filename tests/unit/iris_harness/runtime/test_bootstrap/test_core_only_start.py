"""A core-only start is clean, and real faults still are not (issue #110).

The suite runs with the ``email`` extra installed, so a core-only install is simulated
explicitly: the profile is pinned to ``default`` (no ``email_workflows``, and none of the
domain plugins the shipped heartbeats.yaml names), and every Google package is hidden from
``importlib.metadata`` and ``sys.modules``.
"""

from __future__ import annotations

import importlib.metadata
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from iris_harness.runtime.bootstrap import build_runtime

_GOOGLE_DISTS = {"google-api-python-client", "google-auth", "google-auth-oauthlib"}


@pytest.fixture
def core_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("IRIS_PROFILE", "default")  # pinned: never the email profile
    real_version = importlib.metadata.version

    def version(name: str) -> str:
        if name in _GOOGLE_DISTS:
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    for module in ("googleapiclient", "google_auth_oauthlib"):
        monkeypatch.setitem(sys.modules, module, None)
    # Set aside the Gmail modules an earlier test may have imported (they would mask the
    # missing package), by plain dict operations: monkeypatch.delitem raises KeyError at
    # teardown if another test already removed the entry (seen under xdist).
    held = {m: sys.modules.pop(m) for m in list(sys.modules) if _is_gmail(m)}
    try:
        yield
    finally:
        for name in [m for m in sys.modules if _is_gmail(m)]:
            del sys.modules[name]
        sys.modules.update(held)


def _is_gmail(name: str) -> bool:
    return name.startswith("iris_personal.plugins.gmail")


def _start(tmp_path: Path):  # type: ignore[no-untyped-def]
    runtime = build_runtime(data_dir=tmp_path, use_background_scheduler=False)
    runtime.startup()
    return runtime


def test_core_only_start_emits_no_error_traceback_or_unknown_handler(
    core_only: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        runtime = _start(tmp_path)

    assert runtime.plugin_registry.unmounted_reason("email_workflows") is not None
    assert runtime.plugin_registry.unmounted_reason("system") is None
    assert runtime.skill_registry.load_failures == {}
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert [r.getMessage() for r in caplog.records if r.exc_info] == []
    assert [r.getMessage() for r in caplog.records if "unknown handler" in r.getMessage()] == []
    assert [
        r.getMessage() for r in caplog.records if "registered no handler" in r.getMessage()
    ] == []
    # The jobs of the plugins that are absent are accounted for in one INFO line naming them.
    summary = [r for r in caplog.records if "plugin not mounted" in r.getMessage()]
    assert len(summary) == 1 and summary[0].levelno == logging.INFO
    assert "email_workflows" in summary[0].getMessage()
    assert "email_sweep" in summary[0].getMessage()
    # ...and the app still lists them, locked, with the reason.
    sweep = next(d for d in runtime.heartbeats.all_definitions() if d.name == "email_sweep")
    reason = runtime.heartbeats.unavailable_reason(sweep)
    assert reason is not None and "email_workflows" in reason


def test_core_only_start_still_reports_a_broken_skill_and_a_mistyped_heartbeat(
    core_only: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The quiet start must not have become a blind one."""
    repo = Path(__file__).resolve().parents[5]
    config = tmp_path / "config"
    # a private copy of the shipped config with one broken skill and one mistyped heartbeat
    import shutil

    shutil.copytree(repo / "config", config)
    broken = config / "skills" / "zz_broken"
    broken.mkdir()
    (broken / "manifest.yaml").write_text(
        "name: zz_broken\nversion: 1.0.0\ndescription: d\nauthor: t\nlicense: Apache-2.0\n"
        "tools:\n  - name: zz_tool\n    description: d\n    governor_route: system/read\n",
        encoding="utf-8",
    )
    (broken / "tools.py").write_text("raise RuntimeError('broken on purpose')\n", encoding="utf-8")
    with (config / "heartbeats.yaml").open("a", encoding="utf-8") as fh:
        fh.write(
            "\n  - name: typo_job\n    handler: wiki_lnit\n"
            '    schedule: "interval:3600"\n    description: "typo"\n'
        )

    with caplog.at_level(logging.DEBUG):
        runtime = build_runtime(
            config_dir=config, data_dir=tmp_path / "data", use_background_scheduler=False
        )
        runtime.startup()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR and r.exc_info]
    assert any("zz_broken" in r.getMessage() for r in errors)
    assert any(
        r.levelno == logging.WARNING and "unknown handler wiki_lnit" in r.getMessage()
        for r in caplog.records
    )
