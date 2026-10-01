"""No git environment reaches a test (tests/conftest.py strips every ``GIT_*``).

On 2026-09-26 the pre-push gate ran the suite from inside a git hook, which exports
GIT_DIR (and GIT_INDEX_FILE, ...). A test's ``git init`` in a temp dir inherited it,
re-initialised the owner's REAL repository instead, and set ``core.bare=true`` on it.
These tests pin the fix. The "victim" is always a scratch repo in ``tmp_path``, never
this checkout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"
GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(GIT is None, reason="git is not installed")


def _git(*args: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - git with fixed arguments
        [str(GIT), *args],  # git with fixed arguments
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def test_no_git_variable_reaches_a_test() -> None:
    assert [k for k in os.environ if k.startswith("GIT_")] == []


def test_a_git_init_after_the_conftest_leaves_a_hook_s_repo_alone(tmp_path: Path) -> None:
    """Load tests/conftest.py in a child that has a hook's GIT_DIR pointing at a victim
    repo, then ``git init`` a fresh directory, as the roll_vm test did. With the strip,
    the new repo is made where asked and the victim is untouched; without it, git
    re-initialises the victim and marks it bare -- the incident."""
    victim = tmp_path / "victim"
    victim.mkdir()
    assert _git("init", "-q", cwd=victim, env=_clean_env()).returncode == 0
    assert _git("config", "--get", "core.bare", cwd=victim, env=_clean_env()).stdout.strip() == (
        "false"
    )
    fresh = tmp_path / "fresh"
    fresh.mkdir()

    child = (
        "import runpy, subprocess, sys\n"
        f"runpy.run_path({str(CONFTEST)!r})\n"
        f"subprocess.run([{str(GIT)!r}, 'init', '-q'], cwd={str(fresh)!r}, check=True)\n"
    )
    hook_env = {
        **_clean_env(),
        "GIT_DIR": str(victim / ".git"),
        "GIT_INDEX_FILE": str(victim / ".git" / "index"),
        "GIT_PREFIX": "",
    }
    result = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-c", child],  # this interpreter, a fixed script
        env=hook_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    bare = _git("config", "--get", "core.bare", cwd=victim, env=_clean_env()).stdout.strip()
    assert bare == "false", "the hook's repo was re-initialised through an inherited GIT_DIR"
    assert (fresh / ".git").is_dir(), "git init did not create the repo it was asked for"
