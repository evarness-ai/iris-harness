#!/usr/bin/env python3
"""Count package-to-package imports under src/ (feeds docs/architecture/blueprints).

Maps ``iris_harness.X`` to ``iris_harness.<X>`` and any other root (memris,
iris_personal, iris_code) to its first segment. Counts ``import`` and
``from ... import`` statements at module level and inside functions; relative
imports stay inside their package and are not counted. Self-edges are dropped.

Usage: python scripts/import_graph.py [--json]
"""

from __future__ import annotations

import ast
import collections
import json
import os
import sys

ROOTS = ("iris_harness", "memris", "iris_personal", "iris_code")


def package_of(module: str) -> str:
    parts = module.split(".")
    if parts[0] == "iris_harness" and len(parts) > 1:
        return "iris_harness." + parts[1]
    return parts[0]


def scan(
    src_dir: str = "src",
) -> tuple[collections.Counter[tuple[str, str]], collections.Counter[str]]:
    edges: collections.Counter[tuple[str, str]] = collections.Counter()
    files: collections.Counter[str] = collections.Counter()
    for dirpath, _dirs, names in os.walk(src_dir):
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, src_dir).replace(os.sep, ".")[:-3]
            source_pkg = package_of(rel)
            if source_pkg.endswith("__init__"):
                continue
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except SyntaxError:
                continue
            files[source_pkg] += 1
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    modules = [node.module]
                for module in modules:
                    if module.split(".")[0] not in ROOTS:
                        continue
                    target = package_of(module)
                    if target != source_pkg:
                        edges[(source_pkg, target)] += 1
    return edges, files


def main() -> int:
    edges, files = scan()
    rows = [{"from": a, "to": b, "n": n} for (a, b), n in edges.most_common()]
    if "--json" in sys.argv:
        json.dump({"files": dict(files), "edges": rows}, sys.stdout, indent=1)
        return 0
    print(f"{sum(files.values())} py files, {len(rows)} edges")
    for row in rows:
        print(f"{row['n']:4d}  {row['from']} -> {row['to']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
