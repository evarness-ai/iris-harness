"""Attribute iris-api's startup memory: import phase vs lifespan-startup phase.

Runs the ASGI lifespan in-process (no uvicorn), with the same import hook as
mem_import_profile.py still armed, so lazy imports made during startup are
attributed too. Also takes a tracemalloc snapshot to show the Python-heap side.

Usage: python scripts/mem_startup_attribution.py [--top 25]
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import importlib
import os
import sys
import tracemalloc

sys.path.insert(0, os.path.dirname(__file__))
import mem_import_profile as mip  # noqa: E402


def by_pkg(self_map: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, kb in self_map.items():
        top = name.split(".")[0]
        out[top] = out.get(top, 0) + kb
    return out


def show(title: str, m: dict[str, int], top: int) -> None:
    print(f"\n-- {title}")
    for name, kb in sorted(m.items(), key=lambda kv: -kv[1])[:top]:
        if kb / 1024 < 0.3:
            break
        print(f"  {name:38s} {kb/1024:8.1f} MiB")


async def run(top: int) -> None:
    r0 = mip.rss_kb()
    sys.meta_path.insert(0, mip._Finder())
    tracemalloc.start()
    mod = importlib.import_module("iris_harness.server.iris_api.main")
    app = mod.app
    r1 = mip.rss_kb()
    import_pkgs = by_pkg(mip.SELF)
    imported_at_import = set(mip.SELF)
    tm_import = tracemalloc.get_traced_memory()[0]

    async with app.router.lifespan_context(app):
        gc.collect()
        r2 = mip.rss_kb()
        tm_start = tracemalloc.get_traced_memory()[0]
        startup_only = {k: v for k, v in mip.SELF.items() if k not in imported_at_import}
        startup_pkgs = by_pkg(startup_only)
        snap = tracemalloc.take_snapshot()

        print(
            f"rss: bare {r0/1024:.0f} MiB -> after import {r1/1024:.0f} MiB "
            f"(+{(r1-r0)/1024:.0f}) -> after lifespan startup {r2/1024:.0f} MiB (+{(r2-r1)/1024:.0f})"
        )
        print(
            f"python heap (tracemalloc): after import {tm_import/2**20:.0f} MiB, "
            f"after startup {tm_start/2**20:.0f} MiB"
        )
        print(
            f"modules: {len(imported_at_import)} at import, +{len(startup_only)} lazily during startup"
        )
        show("import-phase RSS by package (self)", import_pkgs, top)
        show("startup-phase lazy-import RSS by package (self)", startup_pkgs, top)
        print("\n-- python heap by file (tracemalloc, top)")
        for stat in snap.statistics("filename")[:top]:
            fn = stat.traceback[0].filename
            fn = fn.split("site-packages/")[-1].split("project-iris/")[-1]
            print(f"  {stat.size/2**20:7.1f} MiB  {fn}")
        # lifespan shutdown follows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    asyncio.run(run(args.top))


if __name__ == "__main__":
    main()
