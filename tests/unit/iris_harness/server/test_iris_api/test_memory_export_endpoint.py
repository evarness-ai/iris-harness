"""POST /memory/export writes a named folder under one export root, never to a path."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.memory.export import export_root, resolve_export_dir
from iris_harness.memory.store import MemoryStore
from iris_harness.server.iris_api.main import create_app

pytestmark = pytest.mark.usefixtures("test_vocabulary")


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    r = tmp_path / "exports-root"
    monkeypatch.setenv("IRIS_EXPORT_DIR", str(r))
    return r


@pytest.fixture
def client(tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.ensure_schema()
    runtime = SimpleNamespace(memory_store=store, semantic_index=None)
    app = create_app(runtime=runtime, auto_start_runtime=False)
    with TestClient(app, headers=auth_headers()) as c:
        yield c


def test_a_good_name_writes_under_the_root(client: TestClient, root: Path) -> None:
    resp = client.post("/memory/export", params={"name": "vault"})
    assert resp.status_code == 200, resp.text
    assert (root / "vault" / "index.md").is_file()
    assert Path(resp.json()["out_dir"]) == (root / "vault").resolve()
    assert root.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    "bad",
    ["../escape", "a/b", "a\\b", "/etc", "..", ".", "x" * 65, ""],
)
def test_names_that_are_not_one_plain_folder_name_are_refused(
    client: TestClient, root: Path, bad: str
) -> None:
    resp = client.post("/memory/export", params={"name": bad})
    assert resp.status_code == 400
    assert not (root.parent / "escape").exists()


def test_a_symlink_that_leaves_the_root_is_refused(
    client: TestClient, root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir(parents=True)
    os.symlink(outside, root / "link")
    resp = client.post("/memory/export", params={"name": "link"})
    assert resp.status_code == 400
    assert list(outside.iterdir()) == []


def test_an_existing_file_is_refused(client: TestClient, root: Path) -> None:
    root.mkdir(parents=True)
    (root / "taken").write_text("x")
    assert client.post("/memory/export", params={"name": "taken"}).status_code == 400


def test_the_old_absolute_path_parameter_is_gone(client: TestClient) -> None:
    assert client.post("/memory/export", params={"out_dir": "/tmp/x"}).status_code == 422


def test_root_defaults_under_iris_home_and_can_be_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_EXPORT_DIR", raising=False)
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home"))
    assert export_root() == tmp_path / "home" / "exports"
    monkeypatch.setenv("IRIS_EXPORT_DIR", str(tmp_path / "vaults"))
    assert resolve_export_dir("v") == (tmp_path / "vaults" / "v").resolve()
