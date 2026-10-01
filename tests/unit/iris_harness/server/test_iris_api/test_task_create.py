"""POST /tasks (Track 2 PR 7, plan decision 34).

The store could always create a task; only the harness could reach it. This
closes the owner's "every capability needs an API, never UI-only" rule for the
task store's main verb.

What is worth protecting: the route is write-gated, it cannot be used to forge
a task that claims to come from a subsystem, and creating one is visible to
everything that reads tasks — a create nobody can see is the same as no create.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.tasks import TaskStore


@pytest.fixture()
def runtime(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(data_dir=tmp_path)


@pytest.fixture()
def client(runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    return TestClient(create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers())


def _store(runtime: SimpleNamespace) -> TaskStore:
    store = TaskStore(db_path=runtime.data_dir / "tasks.db")
    store.ensure_schema()
    return store


def test_creating_a_task_persists_it(client: TestClient, runtime: SimpleNamespace) -> None:
    with client:
        resp = client.post("/tasks", json={"title": "Book the dentist"})

    assert resp.status_code == 201
    task = resp.json()["task"]
    assert task["title"] == "Book the dentist"
    assert task["status"] == "open"

    # Not just echoed back: it is in the store the rest of the app reads.
    assert [t.title for t in _store(runtime).list()] == ["Book the dentist"]


def test_a_created_task_reaches_the_list_endpoint(client: TestClient) -> None:
    # The Chat panel and the Action Center badge both read GET /tasks, so a
    # create that does not show up there is invisible where it matters.
    with client:
        client.post("/tasks", json={"title": "Rotate the Telegram token"})
        listed = client.get("/tasks").json()

    assert listed["count"] == 1
    assert listed["tasks"][0]["title"] == "Rotate the Telegram token"


def test_optional_fields_are_optional(client: TestClient) -> None:
    with client:
        body = client.post(
            "/tasks",
            json={
                "title": "Pay the electricity bill",
                "description": "Octopus, account 123456789012",
                "due_at": "2026-09-24T09:00:00Z",
            },
        ).json()["task"]

    assert body["description"] == "Octopus, account 123456789012"
    assert body["due_at"].startswith("2026-09-24T09:00:00")


def test_the_route_cannot_forge_a_subsystem_task(client: TestClient) -> None:
    """`source_kind` is forced to manual, whatever the caller sends.

    A task claiming `finance-bills` would sit in the finance pipeline's own
    views as though that pipeline had raised it.
    """
    with client:
        body = client.post(
            "/tasks",
            json={"title": "Not from finance", "source_kind": "finance-bills"},
        ).json()["task"]

    assert body["source_kind"] == "manual"


def test_the_route_cannot_attach_a_remediation_action(client: TestClient) -> None:
    # A Task with `action` set is a pending action in the Action Center, and a
    # `copy_command` one renders a command to run. Not from an untrusted body.
    with client:
        body = client.post(
            "/tasks",
            json={
                "title": "Looks helpful",
                "action": {"kind": "copy_command", "label": "Fix", "command": "rm -rf /"},
            },
        ).json()["task"]

    assert body["action"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {},  # no title
        {"title": ""},  # empty title
        {"title": "   "},  # whitespace only
        {"title": "x" * 501},  # past the column's limit
    ],
)
def test_a_task_needs_a_real_title(client: TestClient, payload: dict[str, object]) -> None:
    with client:
        assert client.post("/tasks", json=payload).status_code == 422


def test_creating_a_task_is_write_gated(
    runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The global kill switch still applies to the shared secret (#563): a
    # service credential may not write while writes are off.
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    with TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    ) as gated:
        assert gated.post("/tasks", json={"title": "nope"}).status_code == 403


def test_creating_a_task_needs_a_credential(runtime: SimpleNamespace) -> None:
    with TestClient(create_app(runtime=runtime, auto_start_runtime=False)) as anon:
        assert anon.post("/tasks", json={"title": "nope"}).status_code == 401
