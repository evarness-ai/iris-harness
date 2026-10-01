#!/usr/bin/env python3
# ruff: noqa: S603, S607 - fixed `npx` mermaid-cli invocation over our own temp files
"""Diagram CI check: validate the architecture diagrams stay render-clean.

Two layers:
- **drawio**: every ``docs/**/*.drawio`` must be well-formed XML with >=1 ``<diagram>``.
- **mermaid**: every ```mermaid block in ``docs/**/*.md`` must (a) start with a known
  diagram keyword (fast lint, always), and (b) actually render via the official
  ``@mermaid-js/mermaid-cli`` (full render, only with ``--mermaid`` — needs node/npx +
  a headless browser, so it runs in GitHub Actions, not the local pre-push gate).

Exit code 0 = all good; non-zero = at least one diagram is broken.

Usage:
    python scripts/check_diagrams.py            # drawio validity + mermaid lint (fast, no deps)
    python scripts/check_diagrams.py --mermaid  # + full mermaid render (CI; needs npx)
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import xml.dom.minidom
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

# Mermaid diagram openers GitHub supports (first non-empty/non-comment line must match).
_MERMAID_KEYWORDS = (
    "flowchart",
    "graph",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "erDiagram",
    "journey",
    "gantt",
    "pie",
    "mindmap",
    "timeline",
    "gitGraph",
    "quadrantChart",
)


def _fail(msg: str, failures: list[str]) -> None:
    print(f"  FAIL: {msg}")
    failures.append(msg)


def check_drawio(failures: list[str]) -> int:
    files = sorted(DOCS.rglob("*.drawio"))
    print(f"[drawio] checking {len(files)} file(s)")
    for f in files:
        rel = f.relative_to(REPO_ROOT)
        try:
            dom = xml.dom.minidom.parse(str(f))  # noqa: S318 - our own repo .drawio files
        except Exception as exc:  # noqa: BLE001 - report any parse error
            _fail(f"{rel}: not well-formed XML ({exc})", failures)
            continue
        pages = dom.getElementsByTagName("diagram")
        if not pages:
            _fail(f"{rel}: no <diagram> pages", failures)
            continue
        print(f"  OK: {rel} ({len(pages)} page(s))")
    return len(files)


def _mermaid_blocks() -> list[tuple[Path, int, str]]:
    blocks: list[tuple[Path, int, str]] = []
    for md in sorted(DOCS.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        for i, m in enumerate(re.finditer(r"```mermaid\n(.*?)```", text, re.S), start=1):
            blocks.append((md.relative_to(REPO_ROOT), i, m.group(1)))
    return blocks


def check_mermaid_lint(blocks: list[tuple[Path, int, str]], failures: list[str]) -> None:
    print(f"[mermaid:lint] checking {len(blocks)} block(s)")
    for rel, idx, body in blocks:
        opener = next(
            (
                ln.strip()
                for ln in body.splitlines()
                if ln.strip() and not ln.lstrip().startswith("%%")
            ),
            "",
        )
        if not opener.startswith(_MERMAID_KEYWORDS):
            _fail(
                f"{rel} block #{idx}: first line {opener!r} is not a known mermaid diagram type",
                failures,
            )


def check_mermaid_render(blocks: list[tuple[Path, int, str]], failures: list[str]) -> None:
    if not _have_npx():
        _fail("--mermaid requested but `npx` not found (need Node.js for mermaid-cli)", failures)
        return
    print(f"[mermaid:render] rendering {len(blocks)} block(s) via @mermaid-js/mermaid-cli")
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        for rel, idx, body in blocks:
            src = tmpd / f"{rel.name}.{idx}.mmd"
            out = tmpd / f"{rel.name}.{idx}.svg"
            src.write_text(body, encoding="utf-8")
            proc = subprocess.run(
                ["npx", "-y", "@mermaid-js/mermaid-cli", "-i", str(src), "-o", str(out)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
                _fail(f"{rel} block #{idx}: mermaid render failed: {' '.join(tail)}", failures)
            else:
                print(f"  OK: {rel} block #{idx}")


def _have_npx() -> bool:
    from shutil import which

    return which("npx") is not None


def main() -> int:
    do_render = "--mermaid" in sys.argv[1:]
    failures: list[str] = []

    check_drawio(failures)
    blocks = _mermaid_blocks()
    check_mermaid_lint(blocks, failures)
    if do_render:
        check_mermaid_render(blocks, failures)
    else:
        print("[mermaid:render] skipped (pass --mermaid for full render; runs in CI)")

    print()
    if failures:
        print(f"DIAGRAM CHECK FAILED: {len(failures)} issue(s)")
        return 1
    print("DIAGRAM CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
