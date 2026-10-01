#!/usr/bin/env python3
"""Put the web console's production build inside the package (OSS plan R6).

The release workflow runs it between ``npm run build`` (in ``webui/``) and the wheel
build: it replaces ``src/iris_harness/server/iris_api/webui_dist/`` with a copy of
``webui/dist/``, and pyproject's ``include`` ships that directory in the sdist and the
wheel. ``static_ui.resolve_webui_dist()`` serves it when neither ``IRIS_WEBUI_DIST`` nor
the server image's ``/app/webui/dist`` holds a build, so ``pip install iris-harness``
serves the console with no checkout and no Node.

The packaged directory is gitignored: git holds the console's source once, in
``webui/``, never a build of it. Standard library only::

    python scripts/bundle_webui.py                  # webui/dist -> the package
    python scripts/bundle_webui.py --source <dir>   # another build directory

A source without an ``index.html`` is not a build, and the copy fails rather than
packaging a console that cannot load.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "webui" / "dist"
PACKAGED = ROOT / "src" / "iris_harness" / "server" / "iris_api" / "webui_dist"


def bundle(source: Path, target: Path = PACKAGED) -> int:
    """Replace ``target`` with a copy of the build at ``source``; return the file count."""
    if not (source / "index.html").is_file():
        raise FileNotFoundError(f"{source} has no index.html: not a web console build")
    if target.exists():
        shutil.rmtree(target)  # a stale build's hashed assets must not ship beside this one
    shutil.copytree(source, target)
    return sum(1 for p in target.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy the web console build into the package (OSS plan R6)."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args(argv)
    try:
        count = bundle(args.source.resolve())
    except FileNotFoundError as exc:
        print(f"bundle_webui: {exc}", file=sys.stderr)
        return 1
    print(f"bundle_webui: {count} files from {args.source} -> {PACKAGED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
