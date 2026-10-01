"""Tests for ``iris_harness.tools.skills.loader._resolve_config_path``.

The resolver decides where a manifest ``config_files`` entry actually
lives on disk. Bug history: the original implementation prefixed all
relative paths with ``config/``, so ``data/email.db`` resolved to
``<repo>/config/data/email.db`` (always missing). Phase 2's email
skills tripped on that and the brief render failed with
``tool not found`` because the loader skipped tool classes when
prereqs were "missing". This file documents the corrected behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.tools.skills.loader import (
    _resolve_config_path,
    validate_skill_prerequisites,
)
from iris_harness.tools.skills.models import SkillManifest, SkillRequirements


def test_resolves_bare_filename_under_config(tmp_path: Path) -> None:
    """Backward-compat — `message_templates.yaml` lives under config/."""
    assert (
        _resolve_config_path(tmp_path, "message_templates.yaml")
        == tmp_path / "config" / "message_templates.yaml"
    )


def test_resolves_path_starting_with_config_under_repo_root(tmp_path: Path) -> None:
    """Existing behavior — explicit `config/...` is repo-relative."""
    assert (
        _resolve_config_path(tmp_path, "config/channels.yaml")
        == tmp_path / "config" / "channels.yaml"
    )


def test_resolves_data_path_under_repo_root(tmp_path: Path) -> None:
    """The bug — `data/email.db` should resolve to <repo>/data/email.db,
    NOT <repo>/config/data/email.db.
    """
    assert _resolve_config_path(tmp_path, "data/email.db") == tmp_path / "data" / "email.db"


@pytest.mark.parametrize(
    "rel_path",
    [
        "scripts/serve.sh",
        "src/iris_harness/main.py",
        "tests/conftest.py",
        "docs/usage-guides/email-setup-and-workflow.md",
    ],
)
def test_resolves_known_top_level_repo_dirs_repo_relative(tmp_path: Path, rel_path: str) -> None:
    """Other repo-root directories share the data/ fix."""
    assert _resolve_config_path(tmp_path, rel_path) == tmp_path / rel_path


def test_expands_tilde_paths(tmp_path: Path) -> None:
    """`~/.iris/workspace/...` should expand to the user's home."""
    resolved = _resolve_config_path(tmp_path, "~/.iris/workspace/email/x/proposals.jsonl")
    assert str(resolved).startswith(str(Path.home()))
    assert resolved.parts[-1] == "proposals.jsonl"


def test_absolute_paths_pass_through(tmp_path: Path) -> None:
    """Absolute paths are returned untouched (no repo prefix)."""
    abs_path = "/etc/iris/something.yaml"
    assert _resolve_config_path(tmp_path, abs_path) == Path(abs_path)


def test_unknown_top_level_falls_back_to_config(tmp_path: Path) -> None:
    """An unknown first component preserves the legacy fallback —
    treated as a bare filename under config/."""
    assert (
        _resolve_config_path(tmp_path, "weird-dir/file.yaml")
        == tmp_path / "config" / "weird-dir" / "file.yaml"
    )


# ---------------------------------------------------------------------------
# Integration with validate_skill_prerequisites — the bug's user-facing impact
# ---------------------------------------------------------------------------


def _manifest_with_config_files(*files: str) -> SkillManifest:
    return SkillManifest(
        name="t",
        version="0.1.0",
        description="t",
        author="iris",
        license="Apache-2.0",
        requires=SkillRequirements(python=">=3.12", config_files=files),
    )


def test_validate_prereqs_finds_existing_data_file(tmp_path: Path) -> None:
    """An existing `data/<file>` is no longer reported missing."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "email.db").write_bytes(b"")

    manifest = _manifest_with_config_files("data/email.db")
    assert validate_skill_prerequisites(tmp_path, manifest) == ()


def test_validate_prereqs_reports_missing_data_file(tmp_path: Path) -> None:
    """A missing `data/<file>` is still flagged — just at the right path."""
    manifest = _manifest_with_config_files("data/email.db")
    missing = validate_skill_prerequisites(tmp_path, manifest)
    assert missing == ("config:data/email.db",)


def test_validate_prereqs_finds_existing_home_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`~/...` prereqs honor a redirected HOME via expanduser."""
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".iris" / "workspace" / "email" / "x" / "proposals.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("[]")

    manifest = _manifest_with_config_files("~/.iris/workspace/email/x/proposals.jsonl")
    assert validate_skill_prerequisites(tmp_path, manifest) == ()
