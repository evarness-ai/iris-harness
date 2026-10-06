"""Skill discovery keeps real faults loud and absent-extra skills quiet (issue #110).

A skill whose *declared* ``requires.packages`` are missing is unavailable: one INFO line
naming the extra to install, no traceback. A skill that fails to load for any other
reason (a bug in its tools.py, an import its manifest does not declare) is a fault: logged
once with its traceback, listed as failed, and still isolated so its siblings load.
"""

from __future__ import annotations

import importlib.metadata
import logging
import sys
from pathlib import Path

import pytest

from iris_harness.tools.skills.registry import SkillRegistry

REPO_ROOT = Path(__file__).resolve().parents[5]
_LOGGER = "iris_harness.tools.skills.registry"

_GOOD_TOOLS = (
    "from langchain_core.tools import BaseTool\n"
    "class T(BaseTool):\n"
    "    name: str = '{name}_tool'\n"
    "    description: str = 'd'\n"
    "    def _run(self) -> str:\n"
    "        return 'ok'\n"
    "SKILL_TOOLS = [T]\n"
)


def _write_skill(
    root: Path,
    name: str,
    *,
    tools_src: str | None = None,
    packages: tuple[str, ...] = (),
    extra: str | None = None,
    env_vars: tuple[str, ...] = (),
    config_files: tuple[str, ...] = (),
) -> None:
    skill_dir = root / "config" / "skills" / name
    skill_dir.mkdir(parents=True)
    requires = "requires:\n  python: '>=3.12'\n"
    if packages:
        requires += "  packages:\n" + "".join(f"    - {p}\n" for p in packages)
    if extra:
        requires += f"  extra: {extra}\n"
    if env_vars:
        requires += "  env_vars:\n" + "".join(f"    - {v}\n" for v in env_vars)
    if config_files:
        requires += "  config_files:\n" + "".join(f"    - {c}\n" for c in config_files)
    (skill_dir / "manifest.yaml").write_text(
        f"name: {name}\nversion: 1.0.0\ndescription: d\nauthor: t\nlicense: Apache-2.0\n"
        f"tools:\n  - name: {name}_tool\n    description: d\n    governor_route: system/read\n"
        + requires,
        encoding="utf-8",
    )
    (skill_dir / "tools.py").write_text(
        tools_src if tools_src is not None else _GOOD_TOOLS.format(name=name), encoding="utf-8"
    )


def _records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _LOGGER]


def test_skill_blocked_by_a_declared_package_is_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_skill(tmp_path, "aaa_good")
    # tools.py would raise if imported: it must not be once a declared package is missing.
    _write_skill(
        tmp_path,
        "mmm_extra",
        tools_src="import not_an_installed_module_xyz\n",
        packages=("not-an-installed-dist-xyz",),
        extra="demo",
    )
    _write_skill(tmp_path, "zzz_good")
    registry = SkillRegistry(tmp_path)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        packages = registry.discover()
        registry.discover()  # every turn re-runs discovery: the line must not repeat

    by_name = {p.manifest.name: p for p in packages}
    assert by_name["aaa_good"].is_loadable and by_name["zzz_good"].is_loadable
    assert not by_name["mmm_extra"].is_loadable
    assert by_name["mmm_extra"].missing_prerequisites == ("package:not-an-installed-dist-xyz",)
    assert registry.load_failures == {}
    records = _records(caplog)
    assert [r.levelno for r in records] == [logging.INFO]
    assert records[0].exc_info is None
    message = records[0].getMessage()
    assert "mmm_extra" in message and "not-an-installed-dist-xyz" in message
    assert "iris-harness[demo]" in message


def test_blocked_skill_without_a_declared_extra_still_logs_one_info_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_skill(tmp_path, "needs", packages=("not-an-installed-dist-xyz",))
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        SkillRegistry(tmp_path).discover()
    (record,) = _records(caplog)
    assert record.levelno == logging.INFO and "pip install" not in record.getMessage()


def test_genuine_skill_bug_logs_one_traceback_and_siblings_still_load(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_skill(tmp_path, "aaa_good")
    # No declared requirement: an import failing here is the skill's own bug.
    _write_skill(tmp_path, "mmm_broken", tools_src="import not_an_installed_module_xyz\n")
    _write_skill(tmp_path, "nnn_raises", tools_src="raise RuntimeError('tools.py is broken')\n")
    _write_skill(tmp_path, "zzz_good")
    registry = SkillRegistry(tmp_path)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        packages = registry.discover()
        registry.discover()  # again: the traceback is for the operator once, not per turn

    assert [p.manifest.name for p in packages] == ["aaa_good", "zzz_good"]  # not aborted
    assert {d.name for d in registry.load_failures} == {"mmm_broken", "nnn_raises"}
    errors = [r for r in _records(caplog) if r.levelno == logging.ERROR]
    assert len(errors) == 2
    assert all(r.exc_info is not None for r in errors)  # the traceback is kept
    exceptions = {type(r.exc_info[1]).__name__ for r in errors if r.exc_info}
    assert exceptions == {"ModuleNotFoundError", "RuntimeError"}


def test_unreadable_manifest_is_logged_with_traceback_and_isolated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_skill(tmp_path, "aaa_good")
    bad = tmp_path / "config" / "skills" / "bad_manifest"
    bad.mkdir(parents=True)
    (bad / "manifest.yaml").write_text("- not\n- a mapping\n", encoding="utf-8")
    registry = SkillRegistry(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        packages = registry.discover()
    assert [p.manifest.name for p in packages] == ["aaa_good"]
    assert any(r.levelno == logging.ERROR and r.exc_info for r in _records(caplog))


_GOOGLE_DISTS = {"google-api-python-client", "google-auth", "google-auth-oauthlib"}
_GOOGLE_MODULES = ("googleapiclient", "google_auth_oauthlib")


def _is_gmail_module(name: str) -> bool:
    return name.startswith("iris_personal.plugins.gmail")


def test_shipped_skills_lose_nothing_and_log_nothing_loud_without_the_email_extra(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The repo's own skills with every Google package hidden: no warning, no skill lost."""
    real_version = importlib.metadata.version

    def version(name: str) -> str:
        if name in _GOOGLE_DISTS:
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    for module in _GOOGLE_MODULES:
        monkeypatch.setitem(sys.modules, module, None)  # import raises ModuleNotFoundError
    # Hold aside any fetcher module an earlier test imported (it would mask the missing
    # package) and put it back by plain dict operations: monkeypatch.delitem would raise
    # KeyError at teardown when another xdist-ordered test already removed the entry.
    cached = {m: sys.modules.pop(m) for m in list(sys.modules) if _is_gmail_module(m)}
    registry = SkillRegistry(REPO_ROOT)
    try:
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            packages = registry.discover()
    finally:
        for name in [m for m in sys.modules if _is_gmail_module(m)]:
            del sys.modules[name]
        sys.modules.update(cached)

    by_name = {p.manifest.name: p for p in packages}
    gmail = by_name["gmail-inbox"]
    assert not gmail.is_loadable
    assert gmail.missing_prerequisites == tuple(f"package:{d}" for d in sorted(_GOOGLE_DISTS))
    assert by_name["web-fetch"].is_loadable  # sorts after gmail-inbox: not lost to it
    assert registry.load_failures == {}
    assert [r for r in _records(caplog) if r.levelno >= logging.WARNING] == []
    info = [r.getMessage() for r in _records(caplog) if r.levelno == logging.INFO]
    assert len(info) == 1 and "iris-harness[email]" in info[0]


def test_shipped_skills_all_load_when_the_extra_is_present() -> None:
    pytest.importorskip("googleapiclient")
    registry = SkillRegistry(REPO_ROOT)
    by_name = {p.manifest.name: p for p in registry.discover()}
    assert by_name["gmail-inbox"].is_loadable
    assert registry.load_failures == {}


# -- env / config prerequisites: blocked exactly like a missing package ---------------


def test_skill_blocked_by_a_missing_env_var_logs_one_info_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_TEST_SKILL_KEY", raising=False)
    _write_skill(tmp_path, "needs_env", env_vars=("IRIS_TEST_SKILL_KEY",))
    registry = SkillRegistry(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        (package,) = registry.discover()
        registry.discover()  # every turn re-runs discovery: still one line

    assert not package.is_loadable
    assert package.missing_prerequisites == ("env:IRIS_TEST_SKILL_KEY",)
    assert registry.load_failures == {}
    (record,) = _records(caplog)
    assert record.levelno == logging.INFO and record.exc_info is None
    assert record.getMessage() == "skill needs_env unavailable: missing env IRIS_TEST_SKILL_KEY"


def test_skill_blocked_by_a_missing_config_file_logs_one_info_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_skill(tmp_path, "needs_cfg", config_files=("config/absent-xyz.yaml",))
    registry = SkillRegistry(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        (package,) = registry.discover()
        registry.discover()

    assert package.missing_prerequisites == ("config:config/absent-xyz.yaml",)
    (record,) = _records(caplog)
    assert record.levelno == logging.INFO
    assert record.getMessage() == (
        "skill needs_cfg unavailable: missing config config/absent-xyz.yaml"
    )


def test_every_missing_prerequisite_is_named_in_one_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_TEST_SKILL_KEY", raising=False)
    _write_skill(
        tmp_path,
        "needs_all",
        packages=("not-an-installed-dist-xyz",),
        extra="demo",
        env_vars=("IRIS_TEST_SKILL_KEY",),
        config_files=("config/absent-xyz.yaml",),
    )
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        SkillRegistry(tmp_path).discover()
    (record,) = _records(caplog)
    assert record.getMessage() == (
        "skill needs_all unavailable: missing package not-an-installed-dist-xyz, "
        "env IRIS_TEST_SKILL_KEY, config config/absent-xyz.yaml"
        "; install it with: pip install 'iris-harness[demo]'"
    )


def test_env_var_set_in_the_process_environment_satisfies_the_skill(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default environment is the process's, read per discover (hot reload sees it)."""
    monkeypatch.delenv("IRIS_TEST_SKILL_KEY", raising=False)
    _write_skill(tmp_path, "needs_env", env_vars=("IRIS_TEST_SKILL_KEY",))
    registry = SkillRegistry(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        (blocked,) = registry.discover()
        monkeypatch.setenv("IRIS_TEST_SKILL_KEY", "s3cret-value-must-never-be-logged")
        (ready,) = registry.discover()

    assert not blocked.is_loadable and ready.is_loadable
    assert ready.missing_prerequisites == ()
    assert "s3cret-value-must-never-be-logged" not in caplog.text
    assert len(_records(caplog)) == 1  # the earlier block only


def test_an_explicit_environment_overrides_the_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_TEST_SKILL_KEY", "x")
    _write_skill(tmp_path, "needs_env", env_vars=("IRIS_TEST_SKILL_KEY",))
    (package,) = SkillRegistry(tmp_path, environment={}).discover()
    assert package.missing_prerequisites == ("env:IRIS_TEST_SKILL_KEY",)


def test_skills_list_shows_env_and_config_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.cli.commands import _list_skill_entries

    monkeypatch.delenv("IRIS_TEST_SKILL_KEY", raising=False)
    _write_skill(
        tmp_path,
        "needs_both",
        env_vars=("IRIS_TEST_SKILL_KEY",),
        config_files=("config/absent-xyz.yaml",),
    )
    (entry,) = _list_skill_entries(tmp_path)
    assert entry.status == "blocked: env:IRIS_TEST_SKILL_KEY, config:config/absent-xyz.yaml"
