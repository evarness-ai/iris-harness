"""``/api/v1/email/judgments`` — list, summary and the web correction (loop-proof PR 5)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governance.devices import DeviceService, DeviceStore
from iris_harness.server.iris_api.main import create_app
from iris_personal.plugins.email_workflows.judge_api import build_router
from iris_personal.plugins.email_workflows.judge_config import EMAIL_JUDGMENT_CORRECTED

from .judge_fixtures import DAY, Inbox, seed_day

BASE = "/api/v1/email/judgments"


@pytest.fixture
def inbox(tmp_path: Path) -> Inbox:
    box = Inbox(tmp_path)
    seed_day(box, DAY)
    box.judged(
        "m-old",
        "Harbor Bank <care@harborbank.example>",
        "Card activated",
        "fyi",
        at=DAY - timedelta(days=3),
        fields={"last4": "0000"},
    )
    return box


@pytest.fixture
def client(
    inbox: Inbox, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, list[tuple[str, object]]]]:
    from iris_harness.runtime.api_routes import clear_api_routers, register_api_router

    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    monkeypatch.setenv("IRIS_TZ", "UTC")
    emitted: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "iris_personal.plugins.email_workflows.judge_view.default_emit",
        lambda topic, payload: emitted.append((topic, payload)),
    )
    clear_api_routers()
    register_api_router("email_judgments", lambda: build_router(lambda: inbox.data_dir))
    service = DeviceService(store=DeviceStore(db_path=inbox.data_dir / "devices.db"))
    app = create_app(
        runtime=SimpleNamespace(),  # type: ignore[arg-type]
        auto_start_runtime=False,
        device_service=service,
    )
    with TestClient(app, base_url="http://iris.test") as c:
        yield c, emitted
    clear_api_routers()


def test_list_newest_first_with_email_fields_and_the_buckets(client) -> None:  # type: ignore[no-untyped-def]
    c, _ = client
    body = c.get(BASE, headers=auth_headers()).json()
    rows = body["judgments"]
    assert [r["message_id"] for r in rows] == [
        "m-plan",
        "m-case",
        "m-dinner",
        "m-dental",
        "m-bill",
        "m-old",
    ]
    plan = rows[0]
    assert plan["sender"] == "Nimbus Utilities"
    assert plan["subject"] == "Important information about your account"
    assert plan["bucket"] == "unsure" and plan["bucket_name"] == "Unsure"
    assert plan["confidence"] == 0.52
    assert plan["received_at"].startswith("2026-09-26T13:00")
    assert rows[-1]["figures"] == {"last4": "0000"}
    assert [b["key"] for b in body["buckets"]] == [
        "bill",
        "event",
        "needs_reply",
        "fyi",
        "unsure",
        "promo",
    ]
    assert body["buckets"][-1]["name"] == "Promo"


def test_list_filters_by_effective_bucket_limit_and_since(client, inbox: Inbox) -> None:  # type: ignore[no-untyped-def]
    c, _ = client
    inbox.correct("m-case", "needs_reply", source="gmail", at=DAY + timedelta(hours=20))
    got = c.get(f"{BASE}?bucket=needs_reply", headers=auth_headers()).json()["judgments"]
    assert [r["message_id"] for r in got] == ["m-case", "m-dinner"]
    assert got[0]["judge_bucket"] == "fyi" and got[0]["owner_source"] == "gmail"
    assert len(c.get(f"{BASE}?limit=2", headers=auth_headers()).json()["judgments"]) == 2
    since = c.get(f"{BASE}?since=2026-09-25", headers=auth_headers()).json()["judgments"]
    assert "m-old" not in {r["message_id"] for r in since} and len(since) == 5
    assert c.get(f"{BASE}?bucket=nope", headers=auth_headers()).status_code == 422
    assert c.get(f"{BASE}?since=soon", headers=auth_headers()).status_code == 422


def test_summary_counts_one_local_day(client, inbox: Inbox) -> None:  # type: ignore[no-untyped-def]
    c, _ = client
    inbox.judgments.mark_waiting("gmail:owner@example.com", ["w1", "w2"])
    body = c.get(f"{BASE}/summary?day=2026-09-26", headers=auth_headers()).json()
    assert body == {
        "day": "2026-09-26",
        "total": 5,
        "counts": {"bill": 1, "event": 1, "needs_reply": 1, "fyi": 1, "unsure": 1},
        "unsure": 1,
        "waiting": 2,
    }
    assert c.get(f"{BASE}/summary?day=2026-9-x", headers=auth_headers()).status_code == 422


def test_post_moves_the_bucket_through_apply_correction(client, inbox: Inbox) -> None:  # type: ignore[no-untyped-def]
    c, emitted = client
    resp = c.post(f"{BASE}/m-case/bucket", json={"bucket": "needs_reply"}, headers=auth_headers())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["previous"], body["changed"]) == ("fyi", True)
    assert body["judgment"]["bucket"] == "needs_reply"
    assert body["judgment"]["owner_source"] == "web"
    ((topic, payload),) = emitted
    assert topic == EMAIL_JUDGMENT_CORRECTED and payload.source == "web"

    # Undo is the same call with the previous bucket.
    back = c.post(f"{BASE}/m-case/bucket", json={"bucket": "fyi"}, headers=auth_headers())
    assert back.json()["judgment"]["bucket"] == "fyi"
    # The same bucket again: 200, nothing changed, nothing emitted.
    same = c.post(f"{BASE}/m-case/bucket", json={"bucket": "fyi"}, headers=auth_headers())
    assert same.json()["changed"] is False and len(emitted) == 2
    # Promo is an owner's bucket too.
    promo = c.post(f"{BASE}/m-case/bucket", json={"bucket": "promo"}, headers=auth_headers())
    assert promo.json()["judgment"]["bucket_name"] == "Promo"


def test_post_404_unknown_email_and_422_unknown_bucket(client) -> None:  # type: ignore[no-untyped-def]
    c, emitted = client
    missing = c.post(f"{BASE}/nope/bucket", json={"bucket": "fyi"}, headers=auth_headers())
    assert missing.status_code == 404
    bad = c.post(f"{BASE}/m-case/bucket", json={"bucket": "spam-ish"}, headers=auth_headers())
    assert bad.status_code == 422
    assert c.post(f"{BASE}/m-case/bucket", json={}, headers=auth_headers()).status_code == 422
    assert emitted == []


def test_the_post_is_a_gated_write(client, monkeypatch: pytest.MonkeyPatch, inbox: Inbox) -> None:  # type: ignore[no-untyped-def]
    c, _ = client
    # A read-only paired phone reads the list but cannot correct.
    code = c.post(
        "/api/v1/devices/pair/start", json={"scope": "read"}, headers=auth_headers()
    ).json()["code"]
    token = c.post(
        "/api/v1/devices/pair/claim", json={"code": code, "name": "Phone", "kind": "app"}
    ).json()["token"]
    c.cookies.clear()
    phone = {"Authorization": f"Bearer {token}"}
    assert c.get(BASE, headers=phone).status_code == 200
    refused = c.post(f"{BASE}/m-case/bucket", json={"bucket": "bill"}, headers=phone)
    assert refused.status_code == 403
    # Without the writes switch the service secret cannot either.
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    blocked = c.post(f"{BASE}/m-case/bucket", json={"bucket": "bill"}, headers=auth_headers())
    assert blocked.status_code == 403
    assert inbox.judgments.get("m-case").owner_bucket is None  # type: ignore[union-attr]
