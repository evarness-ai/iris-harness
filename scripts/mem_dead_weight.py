"""Which installed distributions does a service never import at idle?

Starts the app's lifespan in-process, then compares the top-level packages in
sys.modules against every distribution in site-packages. Prints the never-imported
distributions with their on-disk size: they cost disk and image size, not RAM.
Distributions that ARE imported are listed with the RSS the import profiler charged.

Usage: python scripts/mem_dead_weight.py iris_harness.server.iris_api.main
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata as md
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import mem_import_profile as mip  # noqa: E402


def dist_size(dist: md.Distribution) -> int:
    total = 0
    base = Path(str(dist.locate_file("")))
    for f in dist.files or []:
        p = base / str(f)
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


async def run(entry: str) -> None:
    sys.meta_path.insert(0, mip._Finder())
    mod = importlib.import_module(entry)
    app = getattr(mod, "app", None)
    if app is not None and hasattr(app, "router"):
        async with app.router.lifespan_context(app):
            report(entry)
    else:
        report(entry)


def report(entry: str) -> None:
    loaded_tops = {m.split(".")[0] for m in sys.modules}
    used: list[tuple[str, int, int]] = []
    unused: list[tuple[str, int]] = []
    for dist in md.distributions():
        name = dist.metadata["Name"]
        tops = set()
        try:
            txt = dist.read_text("top_level.txt") or ""
            tops = {t.strip() for t in txt.splitlines() if t.strip()}
        except Exception:  # noqa: BLE001 — no top_level.txt: fall back to the dist name below
            tops = set()
        if not tops:
            tops = {(name or "").replace("-", "_")}
        size = dist_size(dist)
        hit = tops & loaded_tops
        if hit:
            kb = sum(v for k, v in mip.SELF.items() if k.split(".")[0] in hit)
            used.append((name, size, kb))
        else:
            unused.append((name, size))
    unused.sort(key=lambda t: -t[1])
    used.sort(key=lambda t: -t[2])
    print(f"entry: {entry}")
    print(
        f"distributions installed: {len(used)+len(unused)}  imported at idle: {len(used)}  never imported: {len(unused)}"
    )
    print(
        f"disk of never-imported distributions: {sum(s for _, s in unused)/2**20:.0f} MiB "
        f"(of {sum(s for _, s, _ in used)/2**20 + sum(s for _, s in unused)/2**20:.0f} MiB total)"
    )
    print("\n-- never imported at idle (top 25 by disk)")
    for name, size in unused[:25]:
        print(f"  {size/2**20:7.1f} MiB  {name}")
    print("\n-- imported at idle (top 20 by import-time RSS)")
    for name, size, kb in used[:20]:
        print(f"  {kb/1024:7.1f} MiB rss  {size/2**20:7.1f} MiB disk  {name}")


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1]))
