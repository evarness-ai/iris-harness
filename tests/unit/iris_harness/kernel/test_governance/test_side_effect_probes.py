"""Tests for built-in side-effect verification probes (story 12.gov-4.10)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from iris_harness.kernel.governance.side_effects.probes import (
    get_probe,
    run_probe,
)

# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_all_builtin_probes_registered() -> None:
    for name in ("git_commit", "git_push", "github_create_pr", "write_file"):
        assert get_probe(name) is not None, f"probe {name!r} not registered"


def test_run_probe_unknown_returns_ambiguous() -> None:
    result = run_probe("nonexistent_probe", "some-id", {})
    assert result == "ambiguous"


# ---------------------------------------------------------------------------
# git_commit probe
# ---------------------------------------------------------------------------


def test_git_commit_completed_when_commit_exists(tmp_path: Path) -> None:
    fake_sha = "abc1234def5678901234567890123456789012ab"
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "commit\n"

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = run_probe("git_commit", fake_sha, {"repo_path": str(tmp_path)})

    assert result == "completed"
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "cat-file" in cmd
    assert fake_sha in cmd


def test_git_commit_not_completed_when_missing(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 128
    mock_result.stdout = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_probe("git_commit", "deadbeef" * 5, {"repo_path": str(tmp_path)})

    assert result == "not_completed"


def test_git_commit_ambiguous_on_timeout(tmp_path: Path) -> None:
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=10)):
        result = run_probe("git_commit", "deadbeef", {})
    assert result == "ambiguous"


def test_git_commit_ambiguous_when_git_missing() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
        result = run_probe("git_commit", "deadbeef", {})
    assert result == "ambiguous"


# ---------------------------------------------------------------------------
# git_push probe
# ---------------------------------------------------------------------------


def test_git_push_completed_when_ref_exists(tmp_path: Path) -> None:
    commit_sha = "abc1234def5678"
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = f"{commit_sha}fedc\trefs/heads/feat/my-branch\n"

    with patch("subprocess.run", return_value=mock_result):
        result = run_probe(
            "git_push",
            "origin/feat/my-branch",
            {"repo_path": str(tmp_path), "commit_sha": commit_sha},
        )

    assert result == "completed"


def test_git_push_not_completed_when_no_ref(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_probe("git_push", "origin/feat/my-branch", {})

    assert result == "not_completed"


def test_git_push_ambiguous_on_malformed_id() -> None:
    result = run_probe("git_push", "no-slash-here", {})
    assert result == "ambiguous"


def test_git_push_ambiguous_on_timeout() -> None:
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=15)):
        result = run_probe("git_push", "origin/main", {})
    assert result == "ambiguous"


# ---------------------------------------------------------------------------
# github_create_pr probe
# ---------------------------------------------------------------------------


def test_github_create_pr_completed_when_pr_exists() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = '{"state":"OPEN"}'

    with patch("subprocess.run", return_value=mock_result):
        result = run_probe(
            "github_create_pr",
            "https://github.com/org/repo/pull/42",
            {},
        )

    assert result == "completed"


def test_github_create_pr_not_completed_when_missing() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_probe(
            "github_create_pr",
            "https://github.com/org/repo/pull/99",
            {},
        )

    assert result == "not_completed"


def test_github_create_pr_ambiguous_on_non_https() -> None:
    result = run_probe("github_create_pr", "not-a-url", {})
    assert result == "ambiguous"


def test_github_create_pr_ambiguous_when_gh_missing() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError("gh not found")):
        result = run_probe("github_create_pr", "https://github.com/org/repo/pull/1", {})
    assert result == "ambiguous"


# ---------------------------------------------------------------------------
# write_file probe
# ---------------------------------------------------------------------------


def test_write_file_completed_when_file_exists_and_sha_matches(tmp_path: Path) -> None:
    content = b"hello world"
    sha = hashlib.sha256(content).hexdigest()
    f = tmp_path / "output.txt"
    f.write_bytes(content)

    result = run_probe("write_file", f"{f}:{sha}", {})
    assert result == "completed"


def test_write_file_not_completed_when_file_missing(tmp_path: Path) -> None:
    result = run_probe("write_file", str(tmp_path / "nonexistent.txt"), {})
    assert result == "not_completed"


def test_write_file_ambiguous_when_sha_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_bytes(b"actual content")
    wrong_sha = "0" * 64

    result = run_probe("write_file", f"{f}:{wrong_sha}", {})
    assert result == "ambiguous"


def test_write_file_completed_without_sha(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_bytes(b"any content")

    result = run_probe("write_file", str(f), {})
    assert result == "completed"


# ---------------------------------------------------------------------------
# run_probe exception handling
# ---------------------------------------------------------------------------


def test_run_probe_exception_in_probe_returns_ambiguous() -> None:
    from iris_harness.kernel.governance.side_effects import probes as _probes

    original = _probes._REGISTRY.get("write_file")
    try:
        _probes._REGISTRY["write_file"] = lambda _id, _meta: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        result = run_probe("write_file", "anything", {})
        assert result == "ambiguous"
    finally:
        if original is not None:
            _probes._REGISTRY["write_file"] = original
        else:
            _probes._REGISTRY.pop("write_file", None)


def test_no_probe_is_ambiguous_and_cannot_be_registered() -> None:
    """A non-read tool that declares no ``verify:`` probe is recorded with ``NO_PROBE``:
    resume cannot tell whether it landed, so the owner approves before a retry."""
    import pytest

    from iris_harness.kernel.governance.side_effects.probes import (
        NO_PROBE,
        probe_names,
        register_probe,
    )

    assert run_probe(NO_PROBE, "run-1:0:abc", {}) == "ambiguous"
    assert NO_PROBE not in probe_names()
    assert {"git_commit", "git_push", "github_create_pr", "write_file"} <= probe_names()
    with pytest.raises(ValueError, match="needs a name"):
        register_probe(NO_PROBE, lambda sid, meta: "completed")
