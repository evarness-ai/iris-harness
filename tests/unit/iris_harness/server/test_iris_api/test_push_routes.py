"""The three push routes (track 2b PR 9).

The decisions worth pinning: subscribing is NOT a governed write, the
endpoint never comes back out, and a subscription is tied to the device that
made it so revoking the device takes it with them.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api import main as api_main
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.channels.web_push import PushSubscriptionStore

UA_PUBLIC = (
    "BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4"
)
AUTH = "BTBZMqHH6r4Tts7J_aSIgg"
ENDPOINT = "https://push.example/subscription/capability-token"

BODY = {"endpoint": ENDPOINT, "p256dh": UA_PUBLIC, "auth": AUTH, "label": "iPhone"}


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PushSubscriptionStore:
    """A throwaway store and key directory, never the developer's own."""
    monkeypatch.setenv("IRIS_HOME", str(tmp_path))
    built = PushSubscriptionStore(db_path=tmp_path / "push.db")
    monkeypatch.setattr(api_main, "_push_subscription_store", built)
    return built


@pytest.fixture()
def client(store: PushSubscriptionStore) -> TestClient:
    return TestClient(
        create_app(runtime=SimpleNamespace(), auto_start_runtime=False), headers=auth_headers()
    )


def test_the_key_route_hands_out_a_subscribable_public_key(client: TestClient) -> None:
    with client:
        body = client.get("/api/v1/push/key").json()
    # An uncompressed P-256 point in base64url is 87 characters and starts
    # with the 0x04 tag — "BP..." / "BC...". A browser refuses anything else.
    assert len(body["public_key"]) == 87
    assert body["public_key"].startswith("B")
    assert body["subscriptions"] == 0


def test_subscribing_persists_the_browser(client: TestClient, store: PushSubscriptionStore) -> None:
    with client:
        resp = client.post("/api/v1/push/subscribe", json=BODY)
    assert resp.status_code == 201
    saved = store.get(ENDPOINT)
    assert saved is not None and saved.label == "iPhone"


def test_the_endpoint_never_comes_back_out(client: TestClient) -> None:
    """It is a bearer capability: whoever holds it can push to that browser.

    The caller already has it, so echoing it only creates another copy to
    leak — through a log, a screenshot or an error report.
    """
    with client:
        payload = client.post("/api/v1/push/subscribe", json=BODY).json()
    assert ENDPOINT not in str(payload)


def test_subscribing_is_not_a_governed_write(
    store: PushSubscriptionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking to RECEIVE what IRIS already decided to say is not a state change.

    A read-only phone that cannot be told its calendar credential broke is a
    read-only phone nobody looks at.
    """
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    with TestClient(
        create_app(runtime=SimpleNamespace(), auto_start_runtime=False), headers=auth_headers()
    ) as client:
        assert client.post("/api/v1/push/subscribe", json=BODY).status_code == 201


def test_subscribing_still_needs_a_credential(store: PushSubscriptionStore) -> None:
    # Ungated is not unauthenticated: a stranger must not be able to attach a
    # destination to the owner's harness.
    with TestClient(create_app(runtime=SimpleNamespace(), auto_start_runtime=False)) as anon:
        assert anon.post("/api/v1/push/subscribe", json=BODY).status_code == 401


def test_resubscribing_replaces_rather_than_duplicates(
    client: TestClient, store: PushSubscriptionStore
) -> None:
    with client:
        client.post("/api/v1/push/subscribe", json=BODY)
        client.post("/api/v1/push/subscribe", json={**BODY, "label": "iPhone 16"})
    rows = store.list()
    assert len(rows) == 1 and rows[0].label == "iPhone 16"


def test_unsubscribing_is_idempotent(client: TestClient) -> None:
    with client:
        client.post("/api/v1/push/subscribe", json=BODY)
        first = client.request("DELETE", "/api/v1/push/subscribe", json={"endpoint": ENDPOINT})
        second = client.request("DELETE", "/api/v1/push/subscribe", json={"endpoint": ENDPOINT})
    assert first.json() == {"removed": True}
    assert second.status_code == 200 and second.json() == {"removed": False}


@pytest.mark.parametrize(
    "bad",
    [
        {**BODY, "p256dh": "too-short"},
        {**BODY, "auth": "too-short"},
        {**BODY, "endpoint": ""},
        {k: v for k, v in BODY.items() if k != "endpoint"},
    ],
)
def test_a_malformed_subscription_is_refused_at_the_boundary(client: TestClient, bad: dict) -> None:
    # Better a 422 than a stored row that fails at every send, forever.
    with client:
        assert client.post("/api/v1/push/subscribe", json=bad).status_code == 422
