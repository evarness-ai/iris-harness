"""``send_email`` (ADR-0118 amendment): validate, describe, and the send itself, over a
real mail store with a fake provider. Nothing here reaches Gmail."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agentic_core import ToolSpec
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.send_tools import build_send_tools
from iris_personal.email.store import EmailStore

ACCT = "gmail:owner@gmail.com"


class _Provider:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self.fail = fail

    def send_message(self, account_id: str, **kwargs: Any) -> str:
        if self.fail is not None:
            raise self.fail
        self.sent.append({"account_id": account_id, **kwargs})
        return "sent-1"


def _msg(mid: str, *, account: str = ACCT, **kw: Any) -> EmailMessage:
    base: dict[str, Any] = {
        "id": mid,
        "provider": "gmail",
        "account_id": account,
        "thread_id": "t-" + mid,
        "from_address": "Alice <alice@example.com>",
        "subject": "Dinner plans",
        "received_at": datetime(2026, 9, 20, 9, 0, tzinfo=UTC),
        "headers_subset": {"Message-ID": "<abc@mail.example.com>", "References": "<r0@x>"},
    }
    base.update(kw)
    return EmailMessage(**base)


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert_many([_msg("m1")])
    return tmp_path


# What the owner typed this turn: the recipient guard lets through an address that is in
# the request or already in the mail (alice@example.com is, as m1's sender).
_ASKED = "email bob@example.com, carol@example.org and dave@example.com"


def _tool(data_dir: Path, provider: Any, *, asked: str = _ASKED) -> ToolSpec:
    (tool,) = build_send_tools(
        data_dir=data_dir, provider_for=lambda _a: provider, current_query=lambda: asked
    )
    return tool


_OK = {"to": ["bob@example.com"], "subject": "Lunch", "body": "Friday at 1?\n\nRobin"}


# --- validate ---------------------------------------------------------------------------


def test_a_well_formed_message_passes(data_dir: Path) -> None:
    tool = _tool(data_dir, _Provider())
    assert tool.validate is not None and tool.validate(_OK) is None
    assert tool.validate({**_OK, "to": "Bob <bob@example.com>, carol@example.org"}) is None


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({**_OK, "to": ["Bob"]}, "not an email address: Bob"),
        ({**_OK, "to": ["bob@example"]}, "not an email address"),
        ({**_OK, "cc": ["me"]}, "not an email address: me"),
        ({**_OK, "to": ["bob@example.com\nBcc: eve@evil.com"]}, "not an email address"),
        ({**_OK, "body": "   "}, "body; it is empty"),
        ({k: v for k, v in _OK.items() if k != "body"}, "body; it is empty"),
        ({k: v for k, v in _OK.items() if k != "to"}, "needs a recipient"),
        ({k: v for k, v in _OK.items() if k != "subject"}, "needs a subject"),
        ({**_OK, "subject": "Hi\r\nBcc: eve@evil.com"}, "one line"),
        ({**_OK, "reply_to_id": "nope"}, "'nope' is not in the owner's mail"),
        ({**_OK, "attachments": ["a.pdf"]}, "plain text only"),
        ({**_OK, "html": "<b>hi</b>"}, "plain text only"),
        ({**_OK, "bcc": ["x@example.com"]}, "plain text only"),
    ],
)
def test_a_message_the_owner_could_only_approve_to_fail_is_refused(
    data_dir: Path, args: dict[str, Any], expected: str
) -> None:
    tool = _tool(data_dir, _Provider())
    assert tool.validate is not None
    assert expected in (tool.validate(args) or "")


def test_no_connected_account_is_refused(tmp_path: Path) -> None:
    tool = _tool(tmp_path, _Provider())  # an empty store
    assert tool.validate is not None
    assert "no mail account" in (tool.validate(_OK) or "")


# --- describe ---------------------------------------------------------------------------


def test_the_card_shows_the_whole_message_in_plain_words(data_dir: Path) -> None:
    tool = _tool(data_dir, _Provider())
    assert tool.describe is not None
    card = tool.describe({**_OK, "cc": ["carol@example.org"]})

    assert card.title == "Send email to bob@example.com — Lunch"
    assert card.lines == (
        "From: owner@gmail.com",
        "To: bob@example.com",
        "Cc: carol@example.org",
        "Subject: Lunch",
        "Message:",
        "Friday at 1?",
        "Robin",
    )


def test_a_reply_card_names_the_email_it_answers(data_dir: Path) -> None:
    tool = _tool(data_dir, _Provider())
    assert tool.describe is not None
    card = tool.describe({"reply_to_id": "m1", "body": "Count me in."})

    assert card.title == "Send email to Alice <alice@example.com> — Re: Dinner plans"
    assert "In reply to: Dinner plans — Alice <alice@example.com> · 20 Sep" in card.lines
    assert "To: Alice <alice@example.com>" in card.lines


def test_several_recipients_are_counted_in_the_title(data_dir: Path) -> None:
    tool = _tool(data_dir, _Provider())
    assert tool.describe is not None
    card = tool.describe({**_OK, "to": ["bob@example.com", "carol@example.org"]})
    assert card.title == "Send email to bob@example.com and 1 more — Lunch"


# --- send -------------------------------------------------------------------------------


def test_send_passes_exactly_the_message_to_the_provider(data_dir: Path) -> None:
    provider = _Provider()
    out = _tool(data_dir, provider).call({**_OK, "cc": ["carol@example.org", "bob@example.com"]})

    assert out == "Sent to bob@example.com: Lunch."
    assert provider.sent == [
        {
            "account_id": ACCT,
            "to": ["bob@example.com"],
            "cc": ["carol@example.org"],  # a To address is not copied again
            "subject": "Lunch",
            "body": "Friday at 1?\n\nRobin",
            "in_reply_to": None,
            "references": None,
            "thread_id": None,
        }
    ]


def test_a_reply_threads_and_fills_the_recipient_and_subject(data_dir: Path) -> None:
    provider = _Provider()
    out = _tool(data_dir, provider).call({"reply_to_id": "m1", "body": "Count me in."})

    assert out == "Reply sent to Alice <alice@example.com>: Re: Dinner plans."
    (sent,) = provider.sent
    assert sent["account_id"] == ACCT  # the account the email arrived in answers it
    assert sent["to"] == ["Alice <alice@example.com>"]
    assert sent["subject"] == "Re: Dinner plans"
    assert sent["in_reply_to"] == "<abc@mail.example.com>"
    assert sent["references"] == "<r0@x>"
    assert sent["thread_id"] == "t-m1"


def test_a_reply_keeps_an_explicit_recipient_and_an_existing_re(tmp_path: Path) -> None:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert_many([_msg("m2", subject="Re: Dinner plans", account="gmail:other@gmail.com")])
    provider = _Provider()
    _tool(tmp_path, provider).call(
        {"reply_to_id": "m2", "to": ["dave@example.com"], "body": "See you."}
    )
    (sent,) = provider.sent
    assert (sent["account_id"], sent["to"], sent["subject"]) == (
        "gmail:other@gmail.com",
        ["dave@example.com"],
        "Re: Dinner plans",
    )


def test_invalid_arguments_never_reach_the_provider(data_dir: Path) -> None:
    provider = _Provider()
    out = _tool(data_dir, provider).call({**_OK, "to": ["Bob"]})
    assert out.startswith("Error: not an email address") and out.endswith("Nothing was sent.")
    assert provider.sent == []


def test_a_read_only_grant_says_how_to_fix_it(data_dir: Path) -> None:
    provider = _Provider(fail=PermissionError("connected read-only; log in again"))
    assert _tool(data_dir, provider).call(_OK) == "Not sent: connected read-only; log in again"


def test_no_provider_that_can_send_is_an_error(data_dir: Path) -> None:
    out = _tool(data_dir, object()).call(_OK)
    assert out == f"Error: no mail provider can send from {ACCT}. Nothing was sent."


# --- the recipient guard and the sending account (owner, 2026-09-22) -------------------


def test_an_address_the_owner_did_not_give_is_refused(data_dir: Path) -> None:
    """The real case: the owner typed one address and the model sent to a blend of
    their two account names, which exists nowhere in their mail."""
    tool = _tool(data_dir, _Provider(), asked="send an email to me.ks@gmail.com")
    problem = tool.validate({**_OK, "to": ["me.ksoman@gmail.com"]})  # type: ignore[misc]
    assert problem is not None and "me.ksoman@gmail.com is not in the owner's request" in problem
    assert tool.validate({**_OK, "to": ["me.ks@gmail.com"]}) is None  # type: ignore[misc]


def test_an_address_already_in_the_mail_needs_no_typing(data_dir: Path) -> None:
    tool = _tool(data_dir, _Provider(), asked="write back to Alice")
    assert tool.validate({**_OK, "to": ["Alice <alice@example.com>"]}) is None  # type: ignore[misc]
    assert tool.validate({**_OK, "cc": ["zed@example.com"]}) is not None  # type: ignore[misc]


def test_a_known_address_is_matched_whole(data_dir: Path) -> None:
    store = EmailStore(db_path=data_dir / "email.db")
    assert store.address_known("alice@example.com")
    assert store.address_known("ALICE@example.com")
    assert not store.address_known("lice@example.com")
    assert not store.address_known("x.alice@example.com")
    assert not store.address_known("alice@example_com")  # "_" is not a wildcard


@pytest.fixture()
def two_accounts(data_dir: Path) -> Path:
    store = EmailStore(db_path=data_dir / "email.db")
    # The second account received mail most recently: it must not become the sender.
    store.upsert_many(
        [_msg("z1", account="gmail:zed@gmail.com", received_at=datetime(2026, 9, 22, tzinfo=UTC))]
    )
    return data_dir


def _sender(data_dir: Path, args: dict[str, Any]) -> str:
    provider = _Provider()
    _tool(data_dir, provider).call({**_OK, **args})
    return str(provider.sent[0]["account_id"])


def test_the_default_sender_setting_picks_the_account(
    two_accounts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_EMAIL_DEFAULT_SENDER", "zed@gmail.com")
    assert _sender(two_accounts, {}) == "gmail:zed@gmail.com"
    monkeypatch.setenv("IRIS_EMAIL_DEFAULT_SENDER", "gmail:owner@gmail.com")
    assert _sender(two_accounts, {}) == ACCT


def test_without_a_setting_the_sender_is_fixed_not_the_latest_mailbox(
    two_accounts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_EMAIL_DEFAULT_SENDER", raising=False)
    assert _sender(two_accounts, {}) == ACCT  # first in order, not zed (newest mail)


def test_from_names_a_connected_account_or_is_refused(
    two_accounts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_EMAIL_DEFAULT_SENDER", "owner@gmail.com")
    assert _sender(two_accounts, {"from": "zed@gmail.com"}) == "gmail:zed@gmail.com"
    tool = _tool(two_accounts, _Provider())
    problem = tool.validate({**_OK, "from": "nobody@gmail.com"})  # type: ignore[misc]
    assert problem is not None and "not a connected account" in problem


def test_a_setting_naming_no_connected_account_blocks_sending(
    two_accounts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_EMAIL_DEFAULT_SENDER", "gone@gmail.com")
    problem = _tool(two_accounts, _Provider()).validate(_OK)  # type: ignore[misc]
    assert problem is not None and "IRIS_EMAIL_DEFAULT_SENDER names gone@gmail.com" in problem


def test_a_reply_still_goes_from_the_account_that_received_it(
    two_accounts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_EMAIL_DEFAULT_SENDER", "owner@gmail.com")
    assert _sender(two_accounts, {"reply_to_id": "z1", "to": None}) == "gmail:zed@gmail.com"
