"""Every `src/...` and `tests/...` path a current-state doc names must exist.

M6.2 moved every package in the harness, and each layer left doc pointers aimed at
files that were no longer there -- 112 of them, found only by writing this scan. A
pointer that does not resolve is worse than no pointer: it sends a reader (or an
agent reading CLAUDE.md) to a path that does not exist, and nothing says so.

Scope is deliberate, and the line is one rule: **a doc that describes the tree as it
is now is in; a dated ledger of what was decided or built is out.** Rewriting a path
inside a ledger makes the record say something that was not true when it was written
(the M6.2 layer-4 notes have the case that set this rule -- a sweep reached ADR-0025
and rewrote a 2026 plan to name a package that did not exist yet).

In: `docs/architecture/*.md`, the docs site's pages, the CLAUDE.md files, the public AGENTS files, README,
CONTRIBUTING.
Out: `docs/architecture/adrs/` and the whole of `docs/{issues,learning,testing-program,
planner}/`, plus the three ledgers named in `_LEDGERS` below.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from iris_harness.foundation.paths import repo_root

# Asked for, not counted. The first draft of this file said `parents[4]`, which is one
# level too high: REPO_ROOT became /home/user, every glob returned nothing, and the test
# passed by scanning an empty corpus. A mutation probe caught it -- see
# `test_the_scan_is_not_vacuous`, which exists so that failure mode cannot come back.
REPO_ROOT = repo_root()

# A repo-relative path into one of the two source roots, as it appears in prose OR as a
# markdown link target. The `(?:\.\./)*` is not decoration: a link is written
# `[label](../../src/x/y.py)`, so a sweep that only rewrites prose leaves the HREF stale
# while the visible label reads correctly -- which is worse than both being wrong,
# because the page looks right. Nine links in these docs were in exactly that state.
_PATH = re.compile(
    r"(?:(?<=\()|(?<![\w/.]))(?:\.\./)*((?:src|tests)/[A-Za-z0-9_./-]*[A-Za-z0-9_/])"
)

# Not rot: webui-relative paths (that doc quotes the web client's own tree), explicit
# `foo/bar` examples, and the truncation left by brace-expansion prose such as
# `tests/unit/iris_harness/foundation/test_{persistence,observability}/`.
_NOT_A_REPO_PATH = re.compile(r"^src/(lib|screens|components|routes)\b|/(foo|bar)\b|[<>{}*]")


# Source roots the public export does not carry (scripts/oss_export_exclude.txt drops the
# coding agent and the domains, with their tests). The current-state docs describe the
# whole product and name paths inside them. Where the root is in the tree -- the private
# repo -- every such path is checked like any other; in a tree without the root there is
# nothing to resolve it against, so it is not rot there. A path outside these roots is
# always checked: the export rewrites (scripts/oss_rewrites.txt) neutralise the few other
# pointers at files it removes, and this test, run on the exported tree, holds them to it.
_ROOTS_A_TREE_MAY_LACK = (
    "src/iris_code",
    "src/iris_personal",
    "tests/unit/iris_code",
    "tests/unit/iris_personal",
)


def _in_a_root_this_tree_lacks(path: str) -> bool:
    return any(
        (path == root or path.startswith(root + "/")) and not (REPO_ROOT / root).is_dir()
        for root in _ROOTS_A_TREE_MAY_LACK
    )


def _is_placeholder(path: str) -> bool:
    if _NOT_A_REPO_PATH.search(path):
        return True
    # an explicit `foo`/`bar` example, including `test_foo/test_bar.py`
    if re.search(r"\b(test_)?(foo|bar)\b", path):
        return True
    # a reader's own package in an example, e.g. `check_stable_imports([Path("src/my_plugin")])`
    if re.search(r"/my[_-]\w", path):
        return True
    # a brace list truncated at the `{`, e.g. ".../test_" or ".../plugins_builtin/test_"
    return path.endswith("test_") or path.endswith("/tools")


def _resolves(path: str) -> bool:
    """True when the path names something real -- a file, a package, or a MODULE.

    The module case is the one worth spelling out: `iris_harness/server/iris_api/main`
    is how a uvicorn target is written (`...main:app`), and it is a correct reference
    even though no such file exists without the suffix.
    """
    target = REPO_ROOT / path
    if target.exists():
        return True
    return target.with_suffix(".py").is_file()


# Ledgers: each is a dated record of what was decided or built, not a description of
# today. Their entries are allowed -- required -- to name paths as they were then.
_LEDGERS = frozenset(
    {
        "DECISIONS.md",  # the dated ADR ledger
        "DOCS-DRIFT-AUDIT.md",  # an audit of drift at one moment
        "OSS-PLUGIN-HARNESS-PLAN.md",  # its milestone table records what each M landed
    }
)


def _design_docs() -> list[Path]:
    docs = [
        d
        for d in sorted((REPO_ROOT / "docs" / "architecture").glob("*.md"))
        if d.name not in _LEDGERS
    ]
    docs += [
        REPO_ROOT / "README.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "CLAUDE.md",
        REPO_ROOT / "src" / "CLAUDE.md",
        REPO_ROOT / "tests" / "CLAUDE.md",
        REPO_ROOT / "src" / "iris_harness" / "server" / "CLAUDE.md",
    ]
    # The public agent files (OSS plan R20) describe the tree as it is too: authored here
    # as `AGENTS.public.md`, shipped as `AGENTS.md` (scripts/oss_export.sh renames them).
    # In the exported tree they are most of the agent-facing map, so they are scanned in
    # both trees, not only where the private CLAUDE.md files are.
    docs += _agent_files()
    return [d for d in docs if d.exists()]


# The docs site's pages (OSS plan R7, mkdocs.yml): what a user reads about today's tree.
_SITE_DIRS = ("getting-started", "guides", "concepts", "reference")


def _site_docs() -> list[Path]:
    docs = [REPO_ROOT / "docs" / "index.md"]
    for site_dir in _SITE_DIRS:
        docs += sorted((REPO_ROOT / "docs" / site_dir).glob("*.md"))
    return [d for d in docs if d.exists()]


def _current_state_docs() -> list[Path]:
    return _design_docs() + _site_docs()


# Never walked: VCS and tool trees, vendored packages, runtime state, and `.claude/`
# (agent worktrees of this very repo live under it -- whole copies of the tree).
_WALK_SKIP = {".git", "node_modules", ".venv", ".claude", "data", "htmlcov", "__pycache__"}


def _agent_files() -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _WALK_SKIP and not d.startswith(".")]
        found += [Path(dirpath) / f for f in filenames if f in ("AGENTS.md", "AGENTS.public.md")]
    return sorted(found)


def test_every_source_path_named_in_a_current_state_doc_exists() -> None:
    broken: list[str] = []
    for doc in _current_state_docs():
        rel_doc = doc.relative_to(REPO_ROOT)
        for line_no, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for match in _PATH.finditer(line):
                path = match.group(1).rstrip("./")
                if not path or _is_placeholder(path) or _in_a_root_this_tree_lacks(path):
                    continue
                if not _resolves(path):
                    broken.append(f"{rel_doc}:{line_no} -> {path}")
    assert not broken, (
        "these docs point at source paths that do not exist "
        "(a move without a doc sweep):\n  " + "\n  ".join(broken)
    )


def test_the_scan_is_not_vacuous() -> None:
    """A green scan must mean "checked and clean", never "found nothing to check".

    Both ways this file can go quiet are failures that look like passes: a wrong
    REPO_ROOT (the first draft's) globs no docs, and a rotted regex matches no paths.
    Either leaves the test above asserting over an empty list. So assert the corpus is
    real: the docs resolve, and they name source paths in the hundreds.
    """
    docs = _design_docs()
    assert len(docs) > 15, f"only {len(docs)} docs resolved -- REPO_ROOT is wrong: {REPO_ROOT}"
    if (REPO_ROOT / "mkdocs.yml").is_file():
        assert len(_site_docs()) > 10, "the docs site's pages are not where the scan looks"

    # A density, not a count: the private tree's corpus names ~3.7 distinct source paths
    # per doc (220 over 59, 2026-09-30) and the public export's curated one ~4.2 (89 over
    # 21). A fixed 150 held only for the private corpus; a rotted regex fails either way.
    # Measured over the design docs and agent files only: the site's pages are written
    # for users and name a source path now and then, so they would dilute the measure
    # without saying anything about the regex.
    seen = {m.group(1) for doc in docs for m in _PATH.finditer(doc.read_text(encoding="utf-8"))}
    assert len(seen) > 3 * len(
        docs
    ), f"the scan matched only {len(seen)} source paths in {len(docs)} docs -- regex rotted"

    # and it must be looking at THIS repo, not some other checkout that also parses
    assert (REPO_ROOT / "src" / "iris_harness" / "runtime" / "bootstrap.py").is_file()
