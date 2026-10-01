"""The web console inside the wheel (OSS plan R6): ``scripts/bundle_webui.py`` and the
wiring around it.

What it pins: the script copies a build into the directory ``static_ui`` serves from and
refuses a directory that is no build; pyproject ships that directory in the sdist and the
wheel and git ignores it; the release workflow bundles before it builds and checks the
path ``static_ui`` reads. The wheel build itself is the release workflow's own check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import tomllib
import yaml

from iris_harness.server.iris_api import static_ui

ROOT = Path(__file__).resolve().parents[3]
_PRIVATE_LAYOUT = ROOT / ".github" / "public"
RELEASE = (_PRIVATE_LAYOUT if _PRIVATE_LAYOUT.is_dir() else ROOT / ".github") / (
    "workflows/release.yml"
)


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bundle_webui = _load("bundle_webui")
_PACKAGED_REL = static_ui.PACKAGED_WEBUI_DIST.relative_to(ROOT / "src")


def _build(path: Path) -> Path:
    (path / "assets").mkdir(parents=True)
    (path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (path / "assets" / "app-abc123.js").write_text("1", encoding="utf-8")
    return path


def test_the_script_targets_the_directory_static_ui_serves() -> None:
    assert bundle_webui.PACKAGED == static_ui.PACKAGED_WEBUI_DIST
    assert bundle_webui.DEFAULT_SOURCE == ROOT / "webui" / "dist"


def test_the_build_is_copied_whole(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "webui_dist"
    count = bundle_webui.bundle(_build(tmp_path / "dist"), target)
    assert count == 2
    assert (target / "index.html").is_file()
    assert (target / "assets" / "app-abc123.js").is_file()


def test_a_stale_build_does_not_survive_the_next_one(tmp_path: Path) -> None:
    target = tmp_path / "webui_dist"
    (target / "assets").mkdir(parents=True)
    (target / "assets" / "app-old.js").write_text("old", encoding="utf-8")
    bundle_webui.bundle(_build(tmp_path / "dist"), target)
    assert not (target / "assets" / "app-old.js").exists()


def test_a_directory_without_an_index_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "dist"
    (source / "assets").mkdir(parents=True)
    target = tmp_path / "webui_dist"
    with pytest.raises(FileNotFoundError):
        bundle_webui.bundle(source, target)
    assert not target.exists()


def test_the_cli_fails_on_a_missing_build(tmp_path: Path) -> None:
    assert bundle_webui.main(["--source", str(tmp_path / "absent")]) == 1


def test_pyproject_ships_the_packaged_build_in_both_formats() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entries = pyproject["tool"]["poetry"]["include"]
    wanted = f"src/{_PACKAGED_REL.as_posix()}/**/*"
    (entry,) = [e for e in entries if isinstance(e, dict) and e["path"] == wanted]
    assert set(entry["format"]) == {"sdist", "wheel"}


def test_git_ignores_the_packaged_build() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert f"src/{_PACKAGED_REL.as_posix()}/" in lines


def _build_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(RELEASE.read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = workflow["jobs"]["build"]["steps"]
    return steps


def test_the_release_bundles_after_the_npm_build_and_before_the_wheel() -> None:
    runs = [str(step.get("run", "")) for step in _build_steps()]
    npm = next(i for i, r in enumerate(runs) if "npm run build" in r)
    bundle = next(i for i, r in enumerate(runs) if "scripts/bundle_webui.py" in r)
    wheel = next(i for i, r in enumerate(runs) if "uv build" in r)
    assert npm < bundle < wheel


def test_the_release_checks_the_path_static_ui_reads() -> None:
    (check,) = [s for s in _build_steps() if s.get("name") == "The wheel carries the web console"]
    assert f'"{_PACKAGED_REL.as_posix()}/"' in check["run"]
