"""The stored-digest read routes (loop-proof plan PR 2).

A push notification's tap lands on ``/digest/<id>``; the web view reads the
body through these routes. Pinned: latest vs by-id, 404s, and that "latest" is
never mistaken for an id.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.digests import DigestStore, FailedSection, shared_digest_store


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DigestStore:
    """The same shared store the brief handler writes, in a throwaway data dir."""
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    return shared_digest_store()


@pytest.fixture()
def client(store: DigestStore) -> TestClient:
    return TestClient(
        create_app(runtime=SimpleNamespace(), auto_start_runtime=False), headers=auth_headers()
    )


def test_latest_is_404_before_any_digest(client: TestClient) -> None:
    with client:
        resp = client.get("/api/digest/latest")
    assert resp.status_code == 404


def test_latest_and_by_id_return_the_full_body(client: TestClient, store: DigestStore) -> None:
    store.save("older", skill_id="morning-brief")
    newest = store.save(
        "## Bills due\n- AT&T [👎](iris:not-useful/a%40b.com)",
        skill_id="morning-brief",
        subject="Morning brief",
        failed_sections=(FailedSection(name="portfolio", title="Portfolio", reason="timeout"),),
    )
    with client:
        latest = client.get("/api/digest/latest").json()
        by_id = client.get(f"/api/digest/{newest.id}").json()
        listing = client.get("/api/digest").json()
    assert latest == by_id
    assert latest["id"] == newest.id
    # The stored copy keeps the action link: the web view turns it into a button.
    assert "iris:not-useful/a%40b.com" in latest["body"]
    assert latest["failed_sections"] == [
        {"name": "portfolio", "title": "Portfolio", "reason": "timeout"}
    ]
    assert [d["id"] for d in listing["digests"]][0] == newest.id
    assert listing["digests"][0]["failed_sections"] == 1


def test_unknown_or_malformed_ids_are_404(client: TestClient) -> None:
    with client:
        assert client.get(f"/api/digest/{'a' * 32}").status_code == 404
        assert client.get("/api/digest/../etc").status_code == 404


def test_the_routes_need_auth(store: DigestStore) -> None:
    with TestClient(create_app(runtime=SimpleNamespace(), auto_start_runtime=False)) as bare:
        assert bare.get("/api/digest/latest").status_code == 401
