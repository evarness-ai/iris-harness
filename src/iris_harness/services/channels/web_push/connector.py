"""The ``web_push`` channel connector.

Three methods, like every other connector — which is the whole reason this
fits: the health watch already calls ``gateway.broadcast(...)``, so wiring
push in changes nothing at the call sites (ADR-0116, plan decision 37).

What is different from Telegram: one logical message fans out to every
subscribed browser, each with its own encryption. A device that has
unsubscribed answers 404 or 410, and the RIGHT response to that is to forget
it — a push service says "gone" exactly once and keeps saying it forever.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from iris_harness.services.channels.models import ChannelMessage, DeliveryReceipt, DeliveryStatus
from iris_harness.services.channels.web_push import keys
from iris_harness.services.channels.web_push.encryption import b64url_decode, encrypt
from iris_harness.services.channels.web_push.store import PushSubscription, PushSubscriptionStore
from iris_harness.services.channels.web_push.vapid import authorization_header

logger = logging.getLogger(__name__)

#: How long a push service should hold the message for a phone that is off.
DEFAULT_TTL_SECONDS = 12 * 60 * 60

#: A subscription the push service has declared gone. 404 = never existed,
#: 410 = expired or unsubscribed; both mean stop trying.
_GONE_STATUSES = frozenset({404, 410})

_TIMEOUT_SECONDS = 10.0

#: ``_send_one``'s answer for a subscription the push service called gone. Compared
#: by identity, so no real error text can be mistaken for it.
_GONE = "gone"


@dataclass
class WebPushConnector:
    """Deliver a notification to every subscribed browser."""

    store: PushSubscriptionStore
    name: str = "web_push"
    ttl_seconds: int = DEFAULT_TTL_SECONDS

    def healthy(self) -> bool:
        """Configured and with somewhere to send.

        Deliberately false when nobody has subscribed: a channel that reports
        healthy and silently delivers to no one is worse than one that admits
        it has no audience.
        """
        try:
            return bool(self.store.list())
        except Exception:  # noqa: BLE001 - an unreadable store is unhealthy, not fatal
            return False

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        subscriptions = self.store.list()
        if not subscriptions:
            return DeliveryReceipt(
                channel=self.name,
                status=DeliveryStatus.SKIPPED,
                error="no browser has subscribed to notifications",
            )

        payload = self._payload(message)
        sent, gone, failures = 0, 0, []
        for sub in subscriptions:
            error = self._send_one(sub, payload)
            if error is None:
                sent += 1
                self.store.note_sent(sub.endpoint)
            elif error is _GONE:
                gone += 1
            else:
                failures.append(error)

        if sent:
            # A partial delivery is a delivery: the owner's phone got it even
            # if a stale desktop subscription did not.
            return DeliveryReceipt(
                channel=self.name,
                status=DeliveryStatus.SENT,
                message_id=f"{sent}/{len(subscriptions)}",
                error="; ".join(failures),
            )
        if not failures:
            # Every browser had left. Nobody received it, so it is not SENT (a
            # reminder counted as delivered to no one would never be retried), and
            # nothing broke either: the same answer as having no subscription.
            return DeliveryReceipt(
                channel=self.name,
                status=DeliveryStatus.SKIPPED,
                error=f"every subscribed browser has unsubscribed ({gone} forgotten)",
            )
        return DeliveryReceipt(
            channel=self.name, status=DeliveryStatus.FAILED, error="; ".join(failures)
        )

    def _payload(self, message: ChannelMessage) -> bytes:
        """What the service worker receives. Kept small and boring.

        `url` is where a tap lands. Adding to this dict is safe and removing is
        not: a worker installed by an older version reads `title`, `body`, `url`.

        `data` is what the worker keeps on the notification for its click
        handler: the url, plus `reminder_id` when the message is a reminder, so a
        Done / Snooze button knows which reminder to answer. `actions` are the
        notification's buttons (browsers that cannot show them ignore them).
        """
        url = _destination(message)
        data: dict[str, object] = {"url": url}
        reminder_id = message.metadata.get("reminder_id")
        if reminder_id:
            data["reminder_id"] = str(reminder_id)
        body: dict[str, object] = {
            "title": message.subject or "IRIS",
            "body": message.body,
            "url": url,
            "data": data,
        }
        actions = _actions(message.metadata.get("actions"))
        if actions:
            body["actions"] = actions
        tag = message.metadata.get("tag")
        if tag:
            # Collapses repeats of one subject rather than stacking them: the
            # health watch re-notifies about the same incident every 12 hours.
            # Reminders tag per reminder (`reminder:<id>`), so two never collapse.
            body["tag"] = str(tag)
        if message.metadata.get("renotify"):
            # Alert again when a notification replaces one under the same tag (a
            # snoozed reminder coming back), instead of swapping it in silently.
            body["renotify"] = True
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def _send_one(self, sub: PushSubscription, payload: bytes) -> str | None:
        """Returns None on success, ``_GONE`` for a forgotten subscription, or a short
        reason."""
        import httpx  # keep the import off the startup path

        try:
            encrypted = encrypt(
                plaintext=payload,
                ua_public_key=b64url_decode(sub.p256dh),
                auth_secret=b64url_decode(sub.auth),
            )
            headers = {
                **encrypted.headers,
                "TTL": str(self.ttl_seconds),
                "Urgency": "normal",
                "Authorization": authorization_header(
                    endpoint=sub.endpoint,
                    private_key=keys.load_or_create(),
                    subject=keys.subject(),
                ),
            }
            response = httpx.post(
                sub.endpoint, content=encrypted.body, headers=headers, timeout=_TIMEOUT_SECONDS
            )
        except Exception as exc:  # noqa: BLE001 - one bad subscription is not an outage
            self.store.note_failure(sub.endpoint)
            return f"{_label(sub)}: {exc}"

        if response.status_code in _GONE_STATUSES:
            # Not an error worth reporting: the browser told the push service
            # it was done, and keeping the row would fail forever.
            self.store.delete(sub.endpoint)
            logger.info("web push: %s is gone (%s); forgot it", _label(sub), response.status_code)
            return _GONE
        if response.status_code >= 400:
            self.store.note_failure(sub.endpoint)
            return f"{_label(sub)}: HTTP {response.status_code}"
        return None


def _destination(message: ChannelMessage) -> str:
    """Where a tap should land.

    The connector owns this mapping because it is the web console's channel,
    and so the one component here entitled to know the console's routes. The
    health watch stamps ``{"health": True}`` and no URL — it should not have
    to learn what a route is to send a notice.
    """
    explicit = message.metadata.get("url")
    if explicit:
        return str(explicit)
    if message.metadata.get("health"):
        return "/health"
    return "/"


def _actions(raw: object) -> list[dict[str, str]]:
    """Notification buttons as the worker expects them; anything malformed is dropped.

    Each is ``{"action": id, "title": label}``. Only the first two are kept:
    that is all Chrome shows, and the order says which two matter.
    """
    if not isinstance(raw, list | tuple):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("action") and item.get("title"):
            out.append({"action": str(item["action"]), "title": str(item["title"])})
    return out[:2]


def _label(sub: PushSubscription) -> str:
    """Never the endpoint: it is a bearer capability and this reaches logs."""
    return sub.label or (sub.device_id or "unknown device")


__all__ = ["DEFAULT_TTL_SECONDS", "WebPushConnector"]
