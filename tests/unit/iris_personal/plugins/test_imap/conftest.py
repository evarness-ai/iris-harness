"""Fixtures for the IMAP provider: a fake server, a synthetic account, a provider bound to
both, and synthetic RFC 822 messages. No real mailbox, no real credential."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage as MIMEMessage
from email.policy import SMTP
from pathlib import Path

import pytest

from iris_personal.email.store import EmailStore
from iris_personal.plugins.imap.account import ImapAccount, save_account
from iris_personal.plugins.imap.provider import ImapProvider
from iris_personal.plugins.imap.state import ImapState

from .fake_imap_server import FakeImapServer, FakeMailbox

USER = "owner@example.test"
PASSWORD = "synthetic-app-password-1234"


def build_message(
    *,
    subject: str = "Hello",
    sender: str = "Sender <sender@shop.example>",
    to: str = USER,
    text: str | None = "Plain body text.",
    html: str | None = None,
    attachments: list[tuple[str, str, bytes]] | None = None,
    message_id: str | None = None,
    date: datetime | None = None,
    extra_headers: dict[str, str] | None = None,
) -> bytes:
    msg = MIMEMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = (date or datetime.now(UTC)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    if message_id is not None:
        msg["Message-ID"] = message_id
    for key, value in (extra_headers or {}).items():
        msg[key] = value
    if text is not None:
        msg.set_content(text)
        if html is not None:
            msg.add_alternative(html, subtype="html")
    elif html is not None:
        msg.set_content(html, subtype="html")
    for filename, mime, data in attachments or []:
        maintype, subtype = mime.split("/", 1)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
    return msg.as_bytes(policy=SMTP)


@pytest.fixture(autouse=True)
def _own_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A data dir per test: the accounts table, the plugin state and the email library's
    write approvals all resolve there, so an approval never leaks between tests."""
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))


@pytest.fixture
def mailbox() -> FakeMailbox:
    return FakeMailbox(users={USER: PASSWORD})


@pytest.fixture
def server(mailbox: FakeMailbox) -> Iterator[FakeImapServer]:
    with FakeImapServer(mailbox) as srv:
        yield srv


@pytest.fixture
def account(server: FakeImapServer) -> ImapAccount:
    acct = ImapAccount(
        address=USER,
        host=server.host,
        username=USER,
        password=PASSWORD,
        port=server.port,
        security="plain",
    )
    save_account(acct)  # the test keyring (tests/conftest.py), never the OS store
    return acct


@pytest.fixture
def state(tmp_path: Path) -> ImapState:
    return ImapState(tmp_path / "imap_state.db")


@pytest.fixture
def provider(state: ImapState) -> ImapProvider:
    return ImapProvider(state=state, timeout=5.0)


@pytest.fixture
def store(tmp_path: Path) -> EmailStore:
    s = EmailStore(db_path=tmp_path / "email.db")
    s.ensure_schema()
    return s


def seed(mailbox: FakeMailbox, count: int, *, start: int = 0, days_ago: int = 1) -> list[str]:
    """``count`` plain messages; returns their Message-IDs."""
    ids = []
    for n in range(start, start + count):
        mid = f"<msg-{n}@shop.example>"
        mailbox.add_message(
            build_message(subject=f"Order {n}", message_id=mid, text=f"Your order {n} shipped."),
            internaldate=datetime.now(UTC) - timedelta(days=days_ago),
        )
        ids.append(mid)
    return ids
