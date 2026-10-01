"""Run the getting-started pages' shell blocks, as a new user would (OSS plan R7).

R7: "every getting-started page is exercised by CI". This is how. Each page under
``docs/getting-started/`` is a sequence of fenced ``bash`` blocks; this script runs them
in page order, in one shell environment the caller provides, and fails on the first
block that exits with a code the page does not allow.

A block may carry one annotation, an HTML comment on the line before its fence (the site
does not render it):

    <!-- ci: install -->           the install command. Not run here: the caller already
                                   installed this tree the same way (scripts/ci_quickstart.sh
                                   runs `uv tool install "<tree>[email]"`, timed), and
                                   tests/unit/test_docs/test_getting_started.py holds the
                                   block to the distribution's name and the email extra.
    <!-- ci: skip <reason> -->     not run: it needs something CI does not have (your
                                   mailbox, a browser, a model download). Its `iris`
                                   commands are still parsed against the CLI by the unit
                                   test, so a renamed command or option fails there.
    <!-- ci: exit 0,1 -->          run; any of these exit codes passes (default: 0 only).

Every other ``bash``/``sh`` block runs and must exit 0. A page with no block that runs
fails: a page CI never runs is a page that can rot.

Usage (scripts/ci_quickstart.sh, inside the fresh user's environment):
    python3 scripts/getting_started.py run [--docs DIR] [--workdir DIR]
    python3 scripts/getting_started.py list [--docs DIR]   # what would run, and why not

Standard library only, Python 3.9+: it runs under the system ``python3`` of a fresh
runner, outside the installed tool.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "getting-started"

# The pages, in the order a reader takes them (and mkdocs.yml's nav lists them). A page
# on disk missing here, or listed here and missing on disk, fails: a new page cannot
# slip past CI by not being named.
PAGE_ORDER = ("install.md", "demo.md", "connect-your-mailbox.md")

SHELL_LANGS = frozenset({"bash", "sh", "shell"})
_FENCE = re.compile(r"^(\s*)(```+|~~~+)\s*([\w+-]*)\s*$")
_ANNOTATION = re.compile(r"^\s*<!--\s*ci:\s*(.*?)\s*-->\s*$")
BLOCK_TIMEOUT_S = 600


@dataclass(frozen=True)
class Block:
    page: str
    line: int  # 1-based line of the opening fence
    lang: str
    annotation: str  # "" | "install" | "skip <reason>" | "exit <codes>"
    body: str

    @property
    def where(self) -> str:
        return f"{self.page}:{self.line}"

    def action(self) -> tuple[str, tuple[int, ...], str]:
        """(``run`` | ``install`` | ``skip``, allowed exit codes, reason)."""
        note = self.annotation
        if not note:
            return "run", (0,), ""
        if note == "install":
            return "install", (), "the caller installed this tree in its place"
        if note.startswith("skip "):
            reason = note[len("skip ") :].strip()
            if reason:
                return "skip", (), reason
        if note.startswith("exit "):
            codes = note[len("exit ") :].replace(" ", "")
            if re.fullmatch(r"\d+(,\d+)*", codes):
                return "run", tuple(int(c) for c in codes.split(",")), ""
        raise ValueError(
            f"{self.where}: unknown annotation `ci: {note}` "
            "(install | skip <reason> | exit <codes>)"
        )


def parse_blocks(text: str, page: str) -> list[Block]:
    """Every fenced shell block in ``text``, with the annotation on the line before it."""
    blocks: list[Block] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = _FENCE.match(lines[i])
        if not match:
            i += 1
            continue
        fence, lang = match.group(2), match.group(3).lower()
        start = i
        body: list[str] = []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith(fence):
            body.append(lines[i])
            i += 1
        i += 1  # the closing fence
        if lang not in SHELL_LANGS:
            continue
        annotation = ""
        prev = start - 1
        while prev >= 0 and not lines[prev].strip():
            prev -= 1
        if prev >= 0:
            note = _ANNOTATION.match(lines[prev])
            if note:
                annotation = note.group(1)
        blocks.append(Block(page, start + 1, lang, annotation, "\n".join(body) + "\n"))
    return blocks


def pages(docs: Path) -> list[Path]:
    on_disk = sorted(p.name for p in docs.glob("*.md"))
    missing = [p for p in PAGE_ORDER if p not in on_disk]
    unlisted = [p for p in on_disk if p not in PAGE_ORDER]
    if missing or unlisted:
        raise SystemExit(
            f"getting_started: PAGE_ORDER is out of step with {docs}: "
            f"missing on disk {missing}, not in PAGE_ORDER {unlisted}"
        )
    return [docs / name for name in PAGE_ORDER]


def all_blocks(docs: Path) -> list[Block]:
    found: list[Block] = []
    for page in pages(docs):
        page_blocks = parse_blocks(page.read_text(encoding="utf-8"), page.name)
        if not any(b.action()[0] == "run" for b in page_blocks):
            raise SystemExit(f"getting_started: {page.name} has no block CI runs")
        found += page_blocks
    return found


def iris_commands(body: str) -> list[list[str]]:
    """The ``iris ...`` command lines in a block body, as argv lists (comments dropped,
    ``\\`` continuations joined). Used by the unit test to parse them against the CLI."""
    import shlex

    commands: list[list[str]] = []
    joined = body.replace("\\\n", " ")
    for line in joined.splitlines():
        argv = shlex.split(line, comments=True)
        if argv and argv[0] == "iris":
            commands.append(argv[1:])
    return commands


def run(docs: Path, workdir: Path | None) -> int:
    blocks = all_blocks(docs)
    cwd = str(workdir) if workdir else None
    ran = 0
    for block in blocks:
        kind, allowed, reason = block.action()
        if kind != "run":
            print(f"getting-started: {block.where}: {kind} ({reason})", flush=True)
            continue
        print(f"getting-started: {block.where}: run", flush=True)
        for line in block.body.rstrip().splitlines():
            print(f"  $ {line}", flush=True)
        try:
            done = subprocess.run(  # noqa: S603 -- the docs pages are the input, by design
                ["bash", "-e", "-o", "pipefail", "-c", block.body],  # noqa: S607 -- the user's bash
                cwd=cwd,
                timeout=BLOCK_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"getting-started: {block.where}: timed out", file=sys.stderr)
            return 1
        if done.returncode not in allowed:
            print(
                f"getting-started: {block.where}: exited {done.returncode}, "
                f"the page allows {list(allowed)}",
                file=sys.stderr,
            )
            return 1
        ran += 1
    print(f"getting-started: OK ({ran} block(s) run, {len(blocks) - ran} not run)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "list"))
    parser.add_argument("--docs", type=Path, default=DOCS, help="the getting-started dir")
    parser.add_argument("--workdir", type=Path, help="where the blocks run (default: cwd)")
    args = parser.parse_args(argv)
    if args.command == "list":
        for block in all_blocks(args.docs):
            kind, allowed, reason = block.action()
            detail = reason or "exit " + ",".join(str(c) for c in allowed)
            print(f"{block.where}\t{kind}\t{detail}")
        return 0
    return run(args.docs, args.workdir)


if __name__ == "__main__":
    sys.exit(main())
