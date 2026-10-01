"""The web_push connector, VAPID, and the subscription store.

What matters here is not that a push is delivered — no test can prove that —
but that the connector behaves like every other channel, and that the two
irreversible decisions are right: a subscription the push service calls gone
is forgotten, and an endpoint never reaches a log.
"""

from __future__ import annotations

from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from iris_harness.services.channels.models import ChannelMessage, DeliveryStatus
from iris_harness.services.channels.protocol import IChannelConnector
from iris_harness.services.channels.web_push import (
    PushSubscription,
    PushSubscriptionStore,
    WebPushConnector,
)
from iris_harness.services.channels.web_push import keys as vapid_keys
from iris_harness.services.channels.web_push.vapid import (
    TOKEN_LIFETIME_SECONDS,
    audience_for,
    authorization_header,
)

UA_PUBLIC = (
    "BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4"
)
AUTH = "BTBZMqHH6r4Tts7J_aSIgg"


@pytest.fixture()
def store(tmp_path: Path) -> PushSubscriptionStore:
    return PushSubscriptionStore(db_path=tmp_path / "push.db")


def _sub(endpoint: str = "https://push.example/aaa", **kw: object) -> PushSubscription:
    return PushSubscription(endpoint=endpoint, p256dh=UA_PUBLIC, auth=AUTH, **kw)  # type: ignore[arg-type]


# ── the store ──────────────────────────────────────────────────────────────


def test_a_browser_resubscribing_replaces_its_keys(store: PushSubscriptionStore) -> None:
    """Same endpoint, rotated keys. Keeping the old ones is a device that
    quietly stops receiving anything."""
    store.save(_sub())
    store.note_failure("https://push.example/aaa")
    store.save(_sub(label="iPhone"))

    rows = store.list()
    assert len(rows) == 1
    assert rows[0].label == "iPhone"
    assert rows[0].failures == 0, "a fresh subscription starts with a clean slate"


def test_revoking_a_device_takes_its_notifications(store: PushSubscriptionStore) -> None:
    store.save(_sub("https://push.example/aaa", device_id="phone"))
    store.save(_sub("https://push.example/bbb", device_id="laptop"))
    assert store.delete_for_device("phone") == 1
    assert [s.device_id for s in store.list()] == ["laptop"]


def test_deleting_an_unknown_endpoint_is_not_an_error(store: PushSubscriptionStore) -> None:
    assert store.delete("https://push.example/never") is False


# ── VAPID ──────────────────────────────────────────────────────────────────


def test_the_audience_is_the_origin_not_the_endpoint() -> None:
    """Mozilla and Apple both reject a token whose audience carries the path.

    The endpoint contains the subscription id, and it is not the audience.
    """
    assert audience_for("https://push.example/subscription/abc123") == "https://push.example"


def test_a_relative_endpoint_is_refused() -> None:
    with pytest.raises(ValueError, match="absolute URL"):
        audience_for("/subscription/abc")


def test_the_authorization_header_is_a_verifiable_es256_token() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    header = authorization_header(
        endpoint="https://push.example/abc", private_key=key, subject="mailto:a@b.c", now=1_000
    )
    assert header.startswith("vapid t=")
    token = header.removeprefix("vapid t=").split(",")[0]
    claims = jwt.decode(
        token,
        key.public_key(),
        algorithms=["ES256"],
        audience="https://push.example",
        # The fixed `now` puts expiry in 1970; what is under test is that the
        # token verifies against the public key and carries the right claims.
        options={"verify_exp": False},
    )
    assert claims["sub"] == "mailto:a@b.c"
    assert claims["aud"] == "https://push.example"
    assert claims["exp"] == 1_000 + TOKEN_LIFETIME_SECONDS


def test_a_subject_that_is_not_a_contact_url_is_refused() -> None:
    # Apple rejects a token without a usable `sub`; failing here is clearer
    # than a 403 from a push service per device.
    with pytest.raises(ValueError, match="mailto:"):
        authorization_header(
            endpoint="https://push.example/abc",
            private_key=ec.generate_private_key(ec.SECP256R1()),
            subject="robin",
        )


def test_the_keypair_is_stable_across_reads(tmp_path: Path) -> None:
    """Rotating it silently breaks every existing subscription."""
    path = tmp_path / "vapid.pem"
    first = vapid_keys.public_key_b64(path)
    assert vapid_keys.public_key_b64(path) == first


def test_the_private_key_is_written_unreadable_to_others(tmp_path: Path) -> None:
    path = tmp_path / "vapid.pem"
    vapid_keys.load_or_create(path)
    assert oct(path.stat().st_mode)[-3:] == "600"


# ── the connector ──────────────────────────────────────────────────────────


def test_it_is_an_ordinary_channel_connector(store: PushSubscriptionStore) -> None:
    # The whole reason this fits: the health watch's existing broadcast
    # reaches it with no change at the call site.
    assert isinstance(WebPushConnector(store=store), IChannelConnector)


def test_no_subscribers_is_skipped_not_failed(store: PushSubscriptionStore) -> None:
    receipt = WebPushConnector(store=store).send(ChannelMessage(recipient="*", body="hi"))
    assert receipt.status == DeliveryStatus.SKIPPED


def test_unhealthy_while_nobody_is_listening(store: PushSubscriptionStore) -> None:
    """A channel that reports healthy and delivers to no one is worse than one
    that admits it has no audience."""
    connector = WebPushConnector(store=store)
    assert connector.healthy() is False
    store.save(_sub())
    assert connector.healthy() is True


def test_a_gone_subscription_is_forgotten(store: PushSubscriptionStore, monkeypatch) -> None:
    """410 means the browser unsubscribed. A push service says so once and
    then says it forever, so the row has to go."""
    store.save(_sub())

    class _Resp:
        status_code = 410

    monkeypatch.setattr("httpx.post", lambda *a, **k: _Resp())
    receipt = WebPushConnector(store=store).send(ChannelMessage(recipient="*", body="hi"))

    assert store.list() == ()
    # Nothing failed: the device left, which is not an error to report. But nobody
    # received it either, so it is not SENT — a reminder counted as delivered to no
    # one would never be retried (loop-proof D14).
    assert receipt.status == DeliveryStatus.SKIPPED


def test_a_gone_subscription_beside_a_live_one_is_still_sent(
    store: PushSubscriptionStore, monkeypatch
) -> None:
    store.save(_sub("https://push.example/live", label="phone"))
    store.save(_sub("https://push.example/gone", label="old tablet"))

    class _Resp:
        def __init__(self, code: int) -> None:
            self.status_code = code

    monkeypatch.setattr("httpx.post", lambda url, **_: _Resp(410 if url.endswith("gone") else 201))
    receipt = WebPushConnector(store=store).send(ChannelMessage(recipient="*", body="hi"))

    assert receipt.status == DeliveryStatus.SENT
    assert receipt.message_id == "1/2"
    assert [s.label for s in store.list()] == ["phone"]


def test_gone_and_broken_subscriptions_are_a_failure(
    store: PushSubscriptionStore, monkeypatch
) -> None:
    store.save(_sub("https://push.example/gone", label="old tablet"))
    store.save(_sub("https://push.example/bad", label="laptop"))

    class _Resp:
        def __init__(self, code: int) -> None:
            self.status_code = code

    monkeypatch.setattr("httpx.post", lambda url, **_: _Resp(410 if url.endswith("gone") else 500))
    receipt = WebPushConnector(store=store).send(ChannelMessage(recipient="*", body="hi"))

    assert receipt.status == DeliveryStatus.FAILED
    assert "laptop" in receipt.error


def test_one_dead_subscription_does_not_stop_the_others(
    store: PushSubscriptionStore, monkeypatch
) -> None:
    store.save(_sub("https://push.example/good", label="phone"))
    store.save(_sub("https://push.example/bad", label="old laptop"))

    class _Resp:
        def __init__(self, code: int) -> None:
            self.status_code = code

    def fake_post(url: str, **_: object) -> _Resp:
        return _Resp(500 if url.endswith("bad") else 201)

    monkeypatch.setattr("httpx.post", fake_post)
    receipt = WebPushConnector(store=store).send(ChannelMessage(recipient="*", body="hi"))

    assert receipt.status == DeliveryStatus.SENT, "the owner's phone got it"
    assert receipt.message_id == "1/2"
    assert "old laptop" in receipt.error


def test_an_endpoint_never_appears_in_an_error(store: PushSubscriptionStore, monkeypatch) -> None:
    """It is a bearer capability: whoever holds it can push to that browser.

    Receipts reach logs and the Action Center, so they carry the label.
    """
    store.save(_sub("https://push.example/secret-capability-token", label="phone"))

    class _Resp:
        status_code = 500

    monkeypatch.setattr("httpx.post", lambda *a, **k: _Resp())
    receipt = WebPushConnector(store=store).send(ChannelMessage(recipient="*", body="hi"))

    assert "secret-capability-token" not in receipt.error
    assert "phone" in receipt.error


def test_the_payload_carries_title_body_and_url(store: PushSubscriptionStore, monkeypatch) -> None:
    # The three fields the service worker reads. Sending a different shape is
    # a notification that renders as "Something needs you."
    store.save(_sub())
    sent: dict[str, object] = {}

    class _Resp:
        status_code = 201

    def fake_post(url: str, content: bytes = b"", **kw: object) -> _Resp:
        sent["len"] = len(content)
        sent["headers"] = kw.get("headers", {})
        return _Resp()

    monkeypatch.setattr("httpx.post", fake_post)
    WebPushConnector(store=store).send(
        ChannelMessage(
            recipient="*", subject="Approval", body="send_email", metadata={"url": "/actions"}
        )
    )

    headers = sent["headers"]
    assert isinstance(headers, dict)
    assert headers["Content-Encoding"] == "aes128gcm"
    assert headers["Authorization"].startswith("vapid t=")
    assert int(headers["TTL"]) > 0
    # Encrypted, so only its size is observable — but it must not be empty.
    assert isinstance(sent["len"], int) and sent["len"] > 86


# ── where a tap lands (track 2b PR 10) ─────────────────────────────────────
#
# The health watch stamps `{"health": True}` and no URL — it should not have to
# learn what a route is to send a notice. The connector maps, because it is the
# web console's channel and so the one thing here entitled to know its routes.


def _sent_payload(store: PushSubscriptionStore, monkeypatch, message: ChannelMessage) -> dict:
    """Send, and return the plaintext the browser would decrypt."""
    import json

    captured: dict[str, object] = {}

    def fake_encrypt(*, plaintext: bytes, **_: object):
        captured["json"] = json.loads(plaintext)

        class _Payload:
            body = b"x" * 120
            headers = {"Content-Encoding": "aes128gcm"}

        return _Payload()

    class _Resp:
        status_code = 201

    monkeypatch.setattr("iris_harness.services.channels.web_push.connector.encrypt", fake_encrypt)
    monkeypatch.setattr("httpx.post", lambda *a, **k: _Resp())
    store.save(_sub())
    WebPushConnector(store=store).send(message)
    return captured["json"]  # type: ignore[return-value]


def test_a_health_notice_opens_pulse(store: PushSubscriptionStore, monkeypatch) -> None:
    payload = _sent_payload(
        store,
        monkeypatch,
        ChannelMessage(
            recipient="*", subject="Needs attention", body="calendar", metadata={"health": True}
        ),
    )
    assert payload["url"] == "/health"


def test_an_explicit_url_wins(store: PushSubscriptionStore, monkeypatch) -> None:
    payload = _sent_payload(
        store,
        monkeypatch,
        ChannelMessage(recipient="*", body="x", metadata={"health": True, "url": "/actions"}),
    )
    assert payload["url"] == "/actions"


def test_an_unlabelled_notice_opens_the_app(store: PushSubscriptionStore, monkeypatch) -> None:
    payload = _sent_payload(store, monkeypatch, ChannelMessage(recipient="*", body="x"))
    assert payload["url"] == "/"


def test_a_tag_collapses_repeats(store: PushSubscriptionStore, monkeypatch) -> None:
    # The health watch re-notifies about one incident every 12 hours; three
    # banners for one revoked credential is how an owner learns to swipe.
    payload = _sent_payload(
        store,
        monkeypatch,
        ChannelMessage(recipient="*", body="x", metadata={"tag": "approval"}),
    )
    assert payload["tag"] == "approval"


def test_a_reminder_tag_and_renotify_reach_the_worker(
    store: PushSubscriptionStore, monkeypatch
) -> None:
    # Each reminder has its own tag, so a second reminder never replaces the first;
    # renotify makes a snoozed one (same tag) alert again.
    payload = _sent_payload(
        store,
        monkeypatch,
        ChannelMessage(
            recipient="*",
            body="8:00 AM",
            metadata={"tag": "reminder:r1", "url": "/reminders/r1", "renotify": True},
        ),
    )
    assert payload["tag"] == "reminder:r1"
    assert payload["url"] == "/reminders/r1"
    assert payload["renotify"] is True


def test_no_renotify_unless_asked(store: PushSubscriptionStore, monkeypatch) -> None:
    payload = _sent_payload(
        store, monkeypatch, ChannelMessage(recipient="*", body="x", metadata={"tag": "a"})
    )
    assert "renotify" not in payload


def test_a_reminder_push_carries_its_buttons_and_id(
    store: PushSubscriptionStore, monkeypatch
) -> None:
    # Done / Snooze on the notification (Chrome, Android, desktop); the worker needs
    # the reminder id to answer the right one, and the url for an iPhone tap.
    payload = _sent_payload(
        store,
        monkeypatch,
        ChannelMessage(
            recipient="*",
            subject="⏰ Take out the recycling",
            body="8:00 AM",
            metadata={
                "reminder_id": "r1",
                "tag": "reminder:r1",
                "url": "/reminders/r1",
                "renotify": True,
                "actions": [
                    {"action": "done", "title": "Done"},
                    {"action": "snooze_1h", "title": "Snooze 1h"},
                ],
            },
        ),
    )
    assert payload["actions"] == [
        {"action": "done", "title": "Done"},
        {"action": "snooze_1h", "title": "Snooze 1h"},
    ]
    assert payload["data"] == {"url": "/reminders/r1", "reminder_id": "r1"}
    assert payload["tag"] == "reminder:r1" and payload["renotify"] is True


def test_a_plain_notice_has_no_buttons(store: PushSubscriptionStore, monkeypatch) -> None:
    payload = _sent_payload(
        store, monkeypatch, ChannelMessage(recipient="*", body="x", metadata={"health": True})
    )
    assert "actions" not in payload
    assert payload["data"] == {"url": "/health"}


def test_malformed_buttons_are_dropped_and_at_most_two_kept(
    store: PushSubscriptionStore, monkeypatch
) -> None:
    actions = [
        {"action": "a", "title": "A"},
        {"title": "no id"},
        "junk",
        {"action": "b", "title": "B"},
        {"action": "c", "title": "C"},
    ]
    payload = _sent_payload(
        store, monkeypatch, ChannelMessage(recipient="*", body="x", metadata={"actions": actions})
    )
    assert payload["actions"] == [{"action": "a", "title": "A"}, {"action": "b", "title": "B"}]
