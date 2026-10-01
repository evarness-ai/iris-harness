"""Tests for SessionWorkspace artifact tracking + path safety."""

from __future__ import annotations

from pathlib import Path

from iris_harness.tools.sandbox.workspace import SessionWorkspace


def test_workspace_dir_created(tmp_path: Path) -> None:
    ws = SessionWorkspace("session1", root=tmp_path)
    assert ws.path.exists()
    assert ws.path.is_dir()


def test_session_id_sanitized(tmp_path: Path) -> None:
    ws = SessionWorkspace("../evil/../../etc", root=tmp_path)
    assert ".." not in ws.path.name
    assert "/" not in ws.path.name
    assert ws.path.parent == tmp_path


def test_empty_session_id_falls_back_to_default(tmp_path: Path) -> None:
    ws = SessionWorkspace("", root=tmp_path)
    assert ws.path.name == "default"


def test_snapshot_returns_only_new_files(tmp_path: Path) -> None:
    ws = SessionWorkspace("s", root=tmp_path)
    (ws.path / "first.txt").write_text("a")
    new = ws.snapshot_artifacts()
    assert any(p.endswith("first.txt") for p in new)

    (ws.path / "second.txt").write_text("b")
    new2 = ws.snapshot_artifacts()
    assert any(p.endswith("second.txt") for p in new2)
    assert not any(p.endswith("first.txt") for p in new2)


def test_snapshot_returns_absolute_host_paths(tmp_path: Path) -> None:
    ws = SessionWorkspace("s", root=tmp_path)
    (ws.path / "out.pdf").write_text("pdf")
    paths = ws.snapshot_artifacts()
    assert all(Path(p).is_absolute() for p in paths)
