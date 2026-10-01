"""Attribute resident memory to the packages a Python entry point imports.

Usage:
    python scripts/mem_import_profile.py <module-to-import> [--top N] [--json out.json]

Wraps every module loader so the RSS growth during each module's exec is
recorded as that module's *self* cost (children subtracted), then aggregated
by top-level package. This is an attribution of RSS at import time only; it
does not include memory allocated later at runtime.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time


def rss_kb() -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


SELF: dict[str, int] = {}
TOTAL: dict[str, int] = {}
ORDER: list[str] = []
_stack: list[list[int]] = []  # each frame: [children_kb]
PARENT: dict[str, str] = {}
_names: list[str] = []


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        if hasattr(self.inner, "create_module"):
            return self.inner.create_module(spec)
        return None

    def exec_module(self, module):
        name = module.__name__
        before = rss_kb()
        PARENT[name] = _names[-1] if _names else "<entry>"
        _names.append(name)
        _stack.append([0])
        try:
            self.inner.exec_module(module)
        finally:
            frame = _stack.pop()
            _names.pop()
            total = rss_kb() - before
            self_kb = total - frame[0]
            SELF[name] = SELF.get(name, 0) + self_kb
            TOTAL[name] = total
            ORDER.append(name)
            if _stack:
                _stack[-1][0] += total

    def __getattr__(self, item):
        return getattr(self.inner, item)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                spec.loader = _Loader(spec.loader)
            return spec
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--json")
    args = ap.parse_args()

    base = rss_kb()
    sys.meta_path.insert(0, _Finder())
    t0 = time.perf_counter()
    importlib.import_module(args.module)
    elapsed = time.perf_counter() - t0
    after = rss_kb()

    by_pkg: dict[str, int] = {}
    for name, kb in SELF.items():
        top = name.split(".")[0]
        by_pkg[top] = by_pkg.get(top, 0) + kb
    ranked = sorted(by_pkg.items(), key=lambda kv: -kv[1])

    print(f"entry: {args.module}")
    print(
        f"rss before import: {base/1024:.1f} MiB   after: {after/1024:.1f} MiB   "
        f"delta: {(after-base)/1024:.1f} MiB   import time: {elapsed:.2f}s"
    )
    print(f"modules imported: {len(SELF)}")
    print(f"{'top-level package':40s} {'self MiB':>9s}")
    for name, kb in ranked[: args.top]:
        print(f"{name:40s} {kb/1024:9.1f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "entry": args.module,
                    "rss_before_kb": base,
                    "rss_after_kb": after,
                    "import_seconds": elapsed,
                    "module_count": len(SELF),
                    "by_package_kb": dict(ranked),
                    "pid": os.getpid(),
                },
                fh,
                indent=1,
            )


if __name__ == "__main__":
    main()
