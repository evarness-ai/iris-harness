"""The TestPyPI build's version (issue #25): ``scripts/stamp_dev_version.py``.

What it pins: a dispatch build is ``<pyproject version>.dev<run number>``; both version
lines move together; the real files are the ones the script edits, and it refuses what
it cannot stamp. The workflow wiring is pinned in test_public_ci.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stamp_dev_version = _load("stamp_dev_version")


def _copies(tmp_path: Path) -> tuple[Path, Path]:
    pyproject = tmp_path / "pyproject.toml"
    init = tmp_path / "__init__.py"
    pyproject.write_text((ROOT / "pyproject.toml").read_text(encoding="utf-8"), "utf-8")
    init.write_text(
        (ROOT / "src" / "iris_harness" / "__init__.py").read_text(encoding="utf-8"), "utf-8"
    )
    return pyproject, init


def test_it_edits_the_real_version_files() -> None:
    assert stamp_dev_version.PYPROJECT == ROOT / "pyproject.toml"
    assert stamp_dev_version.INIT == ROOT / "src" / "iris_harness" / "__init__.py"


def test_the_dispatch_version_is_a_dev_release_of_pyprojects() -> None:
    # PEP 440: X.devN is a pre-release that sorts before X, so it never shadows the
    # release and no resolver picks it without an exact pin.
    assert stamp_dev_version.dev_version("0.1.0", 7) == "0.1.0.dev7"


@pytest.mark.parametrize("base", ["0.1.0.dev3", "0.1.0rc1", "0.1.0.post1", "0.1.0+local"])
def test_only_a_final_release_is_stamped(base: str) -> None:
    with pytest.raises(ValueError, match="not a final release"):
        stamp_dev_version.dev_version(base, 1)


def test_the_run_number_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        stamp_dev_version.dev_version("0.1.0", 0)


def test_both_version_lines_move_together(tmp_path: Path) -> None:
    pyproject, init = _copies(tmp_path)
    base = tomllib.loads(pyproject.read_text("utf-8"))["tool"]["poetry"]["version"]
    version = stamp_dev_version.stamp(42, pyproject=pyproject, init=init)
    assert version == f"{base}.dev42"
    assert tomllib.loads(pyproject.read_text("utf-8"))["tool"]["poetry"]["version"] == version
    namespace: dict[str, object] = {}
    exec(init.read_text("utf-8"), namespace)  # noqa: S102 -- a copy of our own file
    assert namespace["__version__"] == version


def test_main_takes_exactly_one_run_number(capsys: pytest.CaptureFixture[str]) -> None:
    assert stamp_dev_version.main([]) == 2
    assert stamp_dev_version.main(["x"]) == 2
    assert "usage" in capsys.readouterr().err
