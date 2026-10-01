"""The public agent files (OSS plan R20) stay short and point only at real paths.

The public repo carries a root ``AGENTS.md`` (with a ``CLAUDE.md`` that points to it) and a
short ``AGENTS.md`` in five folders. The private repo has its own, larger ``CLAUDE.md`` files
at some of the same places, so the public ones are authored as ``AGENTS.public.md`` /
``CLAUDE.public.md`` and the export renames them -- the mechanism ``README.public.md``
already uses. This test holds them to what R20 promises: each exists, each is short, and
every path it names in backticks resolves. It runs in both trees: in the private repo it
reads the ``*.public.md`` sources, in the export the renamed files.
"""

from __future__ import annotations

import re
from pathlib import Path

from iris_harness.foundation.paths import repo_root

REPO_ROOT = repo_root()

# Exported names: the root pair, then the five folders R20 names.
ROOT_FILES = ("AGENTS.md", "CLAUDE.md")
FOLDER_FILES = (
    "src/iris_harness/kernel/AGENTS.md",
    "src/iris_harness/sdk/AGENTS.md",
    "src/iris_harness/plugins_builtin/AGENTS.md",
    "webui/AGENTS.md",
    "tests/AGENTS.md",
)
ROOT_MAX_LINES = 120
FOLDER_MAX_LINES = 30

# A backticked token is a path when it has a slash and nothing that marks it as a pattern,
# a command or a placeholder.
_BACKTICKED = re.compile(r"`([^`\s]+)`")
_NOT_A_PATH = re.compile(r"[<>*{}$:=]|\.\.\.|^https?:|^-")

# Where a path in these files may be rooted: the file's own folder, the repo root, or a
# source root (the root file's layout list names packages relative to their root).
_SOURCE_ROOTS = ("src/iris_harness", "src/iris_personal")


def _source(exported: str) -> Path:
    """The file to check: the ``*.public.md`` source where it exists (the private repo),
    else the exported name (the public tree, where the private CLAUDE.md files are gone)."""
    path = REPO_ROOT / exported
    public = path.with_name(path.stem + ".public.md")
    return public if public.is_file() else path


def _paths_named(doc: Path) -> list[str]:
    found = []
    for token in _BACKTICKED.findall(doc.read_text(encoding="utf-8")):
        if "/" in token and not _NOT_A_PATH.search(token):
            found.append(token)
    return found


def _resolves(token: str, doc: Path) -> bool:
    bases = [doc.parent, REPO_ROOT, *(REPO_ROOT / r for r in _SOURCE_ROOTS)]
    return any((base / token.rstrip("/")).exists() for base in bases)


def test_every_public_agent_file_exists() -> None:
    missing = [f for f in (*ROOT_FILES, *FOLDER_FILES) if not _source(f).is_file()]
    assert not missing, f"public agent files missing: {missing}"


def test_public_agent_files_stay_short() -> None:
    too_long = []
    for rel, limit in [(f, ROOT_MAX_LINES) for f in ROOT_FILES] + [
        (f, FOLDER_MAX_LINES) for f in FOLDER_FILES
    ]:
        lines = len(_source(rel).read_text(encoding="utf-8").splitlines())
        if lines > limit:
            too_long.append(f"{rel}: {lines} lines (limit {limit})")
    assert not too_long, "\n".join(too_long)


def test_every_path_a_public_agent_file_names_exists() -> None:
    broken = []
    checked = 0
    for rel in (*ROOT_FILES, *FOLDER_FILES):
        doc = _source(rel)
        for token in _paths_named(doc):
            checked += 1
            if not _resolves(token, doc):
                broken.append(f"{doc.relative_to(REPO_ROOT)} -> {token}")
    assert checked > 20, f"the scan found only {checked} paths; the pattern is broken"
    assert not broken, "public agent files name paths that do not exist:\n  " + "\n  ".join(broken)


def test_no_agents_md_beside_its_public_source() -> None:
    """The export renames ``AGENTS.public.md`` to ``AGENTS.md``; a tree that already had an
    ``AGENTS.md`` beside one would lose it there without a word."""
    clashes = [
        rel
        for rel in (ROOT_FILES[0], *FOLDER_FILES)
        if _source(rel) != REPO_ROOT / rel and (REPO_ROOT / rel).exists()
    ]
    assert not clashes, f"AGENTS.md already exists beside its public source: {clashes}"
