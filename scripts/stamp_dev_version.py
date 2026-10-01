#!/usr/bin/env python3
"""Give a TestPyPI build its own version: ``<pyproject version>.dev<N>`` (OSS plan R11).

release.yml's ``workflow_dispatch`` path runs it in the runner's checkout, before the
build, with ``N`` = the workflow's run number. Nothing is committed: the stamp lives only
in that runner's copy of pyproject.toml and ``iris_harness/__init__.py`` (both, so
``importlib.metadata`` and ``iris_harness.__version__`` agree inside the wheel).

Why ``.devN``: TestPyPI refuses a second upload of a version, so every dispatch needs a
new one, and the run number is unique and increasing per workflow. A dev release sorts
BEFORE the release it names (``0.1.0.dev7 < 0.1.0``) and is a pre-release, so no
resolver picks it unless asked for it by exact pin. ``.postN`` would sort after the
release and win a plain ``pip install``; a local version (``+N``) is refused by every
index. Standard library only::

    python scripts/stamp_dev_version.py 42      # 0.1.0 -> 0.1.0.dev42; prints the version
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "iris_harness" / "__init__.py"
_RELEASE = re.compile(r"^\d+(\.\d+)*$")  # a final release: no pre/post/dev/local part


def dev_version(base: str, run_number: int) -> str:
    if run_number < 1:
        raise ValueError(f"run number {run_number} is not a positive integer")
    if not _RELEASE.match(base):
        raise ValueError(f"pyproject's version {base!r} is not a final release to stamp")
    return f"{base}.dev{run_number}"


def _replace_once(text: str, pattern: str, value: str, where: Path) -> str:
    new, count = re.subn(pattern, rf'\g<1>"{value}"', text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"{where}: no version line matching {pattern!r}")
    return new


def stamp(run_number: int, pyproject: Path = PYPROJECT, init: Path = INIT) -> str:
    """Rewrite both version lines to the dev version; return it."""
    base = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["poetry"]["version"]
    version = dev_version(base, run_number)
    # The [tool.poetry] table opens the file, so its `version =` is the first one.
    text = _replace_once(
        pyproject.read_text(encoding="utf-8"), r'^(version = )"[^"]*"', version, pyproject
    )
    if tomllib.loads(text)["tool"]["poetry"]["version"] != version:
        raise ValueError(f"{pyproject}: the first `version =` is not [tool.poetry]'s")
    pyproject.write_text(text, encoding="utf-8")
    init_text = init.read_text(encoding="utf-8")
    init.write_text(
        _replace_once(init_text, r'^(__version__ = )"[^"]*"', version, init), encoding="utf-8"
    )
    return version


def main(argv: list[str]) -> int:
    if len(argv) != 1 or not argv[0].isdigit():
        print("usage: stamp_dev_version.py <run number>", file=sys.stderr)
        return 2
    print(stamp(int(argv[0])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
