"""The ``web`` channel connector.

Delivery means "put this in the web chat the user is looking at". The harness
already has that path: a finished Activity appends an assistant turn to the
session, which the web UI renders from history — ``ActivityNotices.inject_system_notice``,
exposed to plugins as ``HarnessServices.deliver_in_chat``. This connector is that
path wearing the ``IChannelConnector`` interface, so proactive deliveries address
``web`` the same way they address ``telegram``.

``recipient`` is the session id. Empty means the shared web session
(``IRIS_WEB_SESSION``, default ``default``) — which is what a heartbeat sends,
since a brief is addressed to the user, not to a conversation they happen to
have open.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable

from iris_harness.sdk.channels import ChannelMessage, DeliveryReceipt, DeliveryStatus

DEFAULT_WEB_SESSION = "default"


class WebConnector:
    """Deliver a message into a web chat session's history."""

    def __init__(
        self,
        deliver_in_chat: Callable[[str, str], None],
        *,
        name: str = "web",
        default_session: str | None = None,
    ) -> None:
        self.name = name
        self._deliver = deliver_in_chat
        self._default_session: str = (
            default_session or os.getenv("IRIS_WEB_SESSION") or DEFAULT_WEB_SESSION
        )

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        session_id = message.recipient.strip() or self._default_session
        body = f"{message.subject}\n\n{message.body}" if message.subject else message.body
        self._deliver(session_id, body)
        return DeliveryReceipt(
            channel=self.name,
            status=DeliveryStatus.SENT,
            message_id=uuid.uuid4().hex,
        )

    def healthy(self) -> bool:
        """True whenever the harness gave us a delivery path.

        There is nothing to reach out to — the session store is local — so this
        is a wiring check, not a reachability one.
        """
        return callable(self._deliver)
