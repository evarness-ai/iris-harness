"""Every path `.gitleaks.toml` allowlists must still exist.

The allowlist names the test files that carry deliberately fake secrets, so gitleaks
does not report their fixtures as leaks. The entries are **anchored** paths, so a
package move silently un-allowlists a file: the fixtures start reading as real
findings and release gate 6 fails -- but only on a machine where gitleaks is
installed, which the day-to-day dev box is not. All three entries were stale when
this test was written (M6.1b moved one, M6.2 layers 2 and 5 moved the others) and the
repo had no way to notice.

This test needs no gitleaks. It reads the config and checks the tree.
"""

from __future__ import annotations

import re

from iris_harness.foundation.paths import repo_root

REPO_ROOT = repo_root()
CONFIG = REPO_ROOT / ".gitleaks.toml"

# Entries anchored to a concrete file: `'''^some/path/file.py$'''`.
_ANCHORED_FILE = re.compile(r"'''\^(?P<path>[A-Za-z0-9_./\\-]+?)\$'''")


def _anchored_paths() -> list[str]:
    """Anchored entries under `tests/` -- the fake-secret fixtures.

    Deliberately not every anchored entry: `^\\.env$` names a gitignored local file that
    is *correctly* absent from a clean clone, and asserting it exists would fail on CI
    for the right reason at the wrong time. The entries that rot are the ones naming
    files in the tree, and those all live under `tests/`.
    """
    text = CONFIG.read_text(encoding="utf-8")
    return [
        path
        for m in _ANCHORED_FILE.finditer(text)
        if (path := m.group("path").replace("\\.", ".")).startswith("tests/")
    ]


def test_every_anchored_allowlist_path_exists() -> None:
    paths = _anchored_paths()
    missing = [p for p in paths if not (REPO_ROOT / p).exists()]
    assert not missing, (
        "`.gitleaks.toml` allowlists paths that no longer exist, so their fake-secret "
        "fixtures now read as real findings:\n  " + "\n  ".join(missing)
    )


def test_the_allowlist_actually_has_anchored_file_entries() -> None:
    """Guard against a green run that checked nothing.

    If the config's quoting changes, the regex stops matching, `missing` is empty and
    the test above passes while asserting over zero paths -- the same vacuous-pass
    shape that `tests/unit/test_docs` was written with and had to be fixed.
    """
    assert CONFIG.is_file(), f"no .gitleaks.toml at {CONFIG}"
    paths = _anchored_paths()
    # >= 2, not 3: the public export drops the entry that names a private test file.
    assert len(paths) >= 2, f"parsed only {len(paths)} anchored entries -- the regex rotted"
    assert any(p.startswith("tests/") for p in paths)
