"""Tests for the email-sweep heartbeat handler (Track 1D)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.foundation.eventbus import EventBus
from iris_harness.services.heartbeat.models import HeartbeatDefinition, HeartbeatStatus
from iris_personal.email.accounts import EmailAccountStore
from iris_personal.email.events import (
    EMAIL_LABELS_CHANGED,
    EMAIL_NEW_ARRIVED,
    EMAIL_SWEPT,
    EmailLabelsChangedPayload,
    EmailNewArrivedPayload,
)
from iris_personal.email.providers import FetchResult, clear_mail_providers, register_mail_provider
from iris_personal.email.store import EmailStore
from iris_personal.email.sweep import EmailSweepHandler

# ─── Helpers ──────────────────────────────────────────────────────────


def _make_definition(*, max_messages: int = 50, name: str = "email_sweep") -> HeartbeatDefinition:
    return HeartbeatDefinition(
        name=name,
        handler="email_sweep",
        schedule="interval:600",
        enabled=True,
        params={"max_messages": max_messages},
    )


class _GmailStub:
    """A registered provider for the ``gmail`` kind; the tests inject the fetch itself."""

    name = "gmail"

    def fetch_new(self, account_id, *, store=None, max_messages=100, cold_start_days=30):  # type: ignore[no-untyped-def]
        raise AssertionError("tests inject `fetcher=`; the provider's fetch must not run")


@pytest.fixture(autouse=True)
def _gmail_provider_registered() -> None:
    """The sweep selects accounts by registered provider (M5.7 track A)."""
    clear_mail_providers()
    register_mail_provider(_GmailStub())  # type: ignore[arg-type]
    yield  # type: ignore[misc]
    clear_mail_providers()


@pytest.fixture
def accounts_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmailAccountStore:
    """EmailAccountStore pointed at a tmp data/iris.db."""
    db_path = tmp_path / "iris.db"

    def fake_init(self, db_path=db_path):  # type: ignore[no-untyped-def]
        self.db_path = db_path

    monkeypatch.setattr(EmailAccountStore, "__init__", fake_init)
    s = EmailAccountStore()
    s.ensure_schema()
    return s


@pytest.fixture
def email_store(tmp_path: Path) -> EmailStore:
    s = EmailStore(db_path=tmp_path / "email.db")
    s.ensure_schema()
    return s


@pytest.fixture
def captured_events() -> tuple[EventBus, list]:
    """A real EventBus plus a list that captures every payload on EMAIL_SWEPT (the
    judge's queue turns it into email.new_arrived as each email is released)."""
    bus = EventBus()
    captured: list = []
    bus.on(EMAIL_SWEPT, captured.append)
    return bus, captured


# ─── Behaviour ────────────────────────────────────────────────────────


def test_skips_when_no_active_accounts(
    accounts_store: EmailAccountStore, email_store: EmailStore
) -> None:
    handler = EmailSweepHandler(bus=None, accounts_store=accounts_store, email_store=email_store)
    run = handler(_make_definition())
    assert run.status is HeartbeatStatus.SKIPPED
    assert "no active accounts with a mounted mail provider" in run.output


def test_skips_when_only_inactive_accounts(
    accounts_store: EmailAccountStore, email_store: EmailStore
) -> None:
    acct = accounts_store.add(provider="gmail", address="user@gmail.com")
    accounts_store.deactivate(acct.id)
    handler = EmailSweepHandler(bus=None, accounts_store=accounts_store, email_store=email_store)
    run = handler(_make_definition())
    assert run.status is HeartbeatStatus.SKIPPED


def test_filters_to_accounts_with_a_mounted_provider(
    accounts_store: EmailAccountStore, email_store: EmailStore
) -> None:
    """An account whose provider plugin is not mounted is skipped, and named."""
    accounts_store.add(provider="outlook", address="x@outlook.com")

    handler = EmailSweepHandler(bus=None, accounts_store=accounts_store, email_store=email_store)
    run = handler(_make_definition())
    assert run.status is HeartbeatStatus.SKIPPED
    assert "outlook" in (run.output or "")


def test_with_no_provider_registered_nothing_is_swept_and_it_says_so(
    accounts_store: EmailAccountStore, email_store: EmailStore
) -> None:
    """The gmail plugin unmounted: the sweep must not crash and must not go quiet."""
    clear_mail_providers()
    accounts_store.add(provider="gmail", address="user@gmail.com")

    handler = EmailSweepHandler(bus=None, accounts_store=accounts_store, email_store=email_store)
    run = handler(_make_definition())
    assert run.status is HeartbeatStatus.SKIPPED
    assert "no provider mounted for: gmail" in (run.output or "")


@pytest.fixture
def _calendar_declared_non_mailbox():  # type: ignore[no-untyped-def]
    from iris_personal.email import accounts as email_accounts

    email_accounts.clear_non_mailbox_providers()
    email_accounts.declare_non_mailbox_provider("GCalendar ")  # normalised on both sides
    yield
    email_accounts.clear_non_mailbox_providers()


def test_declared_non_mailbox_accounts_are_passed_by_silently(
    accounts_store: EmailAccountStore, email_store: EmailStore, _calendar_declared_non_mailbox
) -> None:
    """Calendar rows share the accounts table; the sweep says nothing about them."""
    accounts_store.add(provider="gmail", address="a@gmail.com")
    accounts_store.add(provider="gcalendar", address="a@gmail.com")
    accounts_store.add(provider="outlook", address="x@outlook.com")  # a mailbox, unmounted

    def fake_fetcher(account_id, *, store, max_messages):  # type: ignore[no-untyped-def]
        return FetchResult(account_id, 0, (), "1", False)

    handler = EmailSweepHandler(
        bus=None, accounts_store=accounts_store, email_store=email_store, fetcher=fake_fetcher
    )
    run = handler(_make_definition())
    assert run.status is HeartbeatStatus.SUCCESS
    assert run.output == (
        "swept 1 account(s), total +0: gmail:a@gmail.com: +0; no provider mounted for: outlook"
    )


def test_only_non_mailbox_accounts_is_a_plain_skip(
    accounts_store: EmailAccountStore, email_store: EmailStore, _calendar_declared_non_mailbox
) -> None:
    accounts_store.add(provider="gcalendar", address="a@gmail.com")
    handler = EmailSweepHandler(bus=None, accounts_store=accounts_store, email_store=email_store)
    run = handler(_make_definition())
    assert run.status is HeartbeatStatus.SKIPPED
    assert run.output == "no active accounts with a mounted mail provider"


def test_fetches_per_account_and_aggregates_counts(
    accounts_store: EmailAccountStore,
    email_store: EmailStore,
    captured_events: tuple[EventBus, list],
) -> None:
    accounts_store.add(provider="gmail", address="a@gmail.com")
    accounts_store.add(provider="gmail", address="b@gmail.com")
    bus, captured = captured_events

    calls: list[str] = []

    def fake_fetcher(account_id, *, store, max_messages):  # type: ignore[no-untyped-def]
        calls.append(account_id)
        n = 2 if account_id.endswith("a@gmail.com") else 5
        return FetchResult(
            account_id=account_id,
            fetched=n,
            new_message_ids=tuple(f"{account_id}-msg-{i}" for i in range(n)),
            new_cursor="123",
            fell_back_to_cold_start=False,
        )

    handler = EmailSweepHandler(
        bus=bus,
        accounts_store=accounts_store,
        email_store=email_store,
        fetcher=fake_fetcher,
    )
    run = handler(_make_definition())

    assert run.status is HeartbeatStatus.SUCCESS
    assert "+7" in run.output  # 2 + 5
    assert len(calls) == 2
    assert len(captured) == 2
    assert {p.account_id for p in captured} == {"gmail:a@gmail.com", "gmail:b@gmail.com"}


def test_emits_payload_with_correct_shape(
    accounts_store: EmailAccountStore,
    email_store: EmailStore,
    captured_events: tuple[EventBus, list],
) -> None:
    accounts_store.add(provider="gmail", address="user@gmail.com")
    bus, captured = captured_events
    released: list = []
    bus.on(EMAIL_NEW_ARRIVED, released.append)

    def fake_fetcher(account_id, *, store, max_messages):  # type: ignore[no-untyped-def]
        return FetchResult(
            account_id=account_id,
            fetched=3,
            new_message_ids=("m1", "m2", "m3"),
            new_cursor="abc",
            fell_back_to_cold_start=True,
        )

    handler = EmailSweepHandler(
        bus=bus,
        accounts_store=accounts_store,
        email_store=email_store,
        fetcher=fake_fetcher,
    )
    handler(_make_definition())

    # Loop-proof PR 5: the sweep only says it stored mail; new mail reaches the rest
    # of IRIS (email.new_arrived) when the judge's queue releases it, never from here.
    assert released == []
    assert len(captured) == 1
    payload = captured[0]
    assert isinstance(payload, EmailNewArrivedPayload)
    assert payload.account_id == "gmail:user@gmail.com"
    assert payload.new_message_ids == ("m1", "m2", "m3")
    assert payload.count == 3
    assert payload.fell_back_to_cold_start is True


def test_no_event_when_fetched_zero(
    accounts_store: EmailAccountStore,
    email_store: EmailStore,
    captured_events: tuple[EventBus, list],
) -> None:
    """Delta sync with no new mail must NOT emit email.swept."""
    accounts_store.add(provider="gmail", address="user@gmail.com")
    bus, captured = captured_events

    def fake_fetcher(account_id, *, store, max_messages):  # type: ignore[no-untyped-def]
        return FetchResult(
            account_id=account_id,
            fetched=0,
            new_message_ids=(),
            new_cursor="abc",
            fell_back_to_cold_start=False,
        )

    handler = EmailSweepHandler(
        bus=bus,
        accounts_store=accounts_store,
        email_store=email_store,
        fetcher=fake_fetcher,
    )
    run = handler(_make_definition())
    assert run.status is HeartbeatStatus.SUCCESS
    assert len(captured) == 0


def test_bus_none_means_silent(accounts_store: EmailAccountStore, email_store: EmailStore) -> None:
    """Construct without a bus → operate silently, no errors."""
    accounts_store.add(provider="gmail", address="user@gmail.com")

    def fake_fetcher(account_id, *, store, max_messages):  # type: ignore[no-untyped-def]
        return FetchResult(
            account_id=account_id,
            fetched=4,
            new_message_ids=("a", "b", "c", "d"),
            new_cursor="x",
            fell_back_to_cold_start=False,
        )

    handler = EmailSweepHandler(
        bus=None,
        accounts_store=accounts_store,
        email_store=email_store,
        fetcher=fake_fetcher,
    )
    run = handler(_make_definition())
    assert run.status is HeartbeatStatus.SUCCESS  # fetch happened
    assert "+4" in run.output


def test_per_account_failure_does_not_abort_sweep(
    accounts_store: EmailAccountStore,
    email_store: EmailStore,
    captured_events: tuple[EventBus, list],
) -> None:
    """If one account's fetch raises, the sweep continues with the next account."""
    accounts_store.add(provider="gmail", address="good@gmail.com")
    accounts_store.add(provider="gmail", address="bad@gmail.com")
    bus, captured = captured_events

    def fake_fetcher(account_id, *, store, max_messages):  # type: ignore[no-untyped-def]
        if "bad@" in account_id:
            raise RuntimeError("quota exceeded")
        return FetchResult(
            account_id=account_id,
            fetched=1,
            new_message_ids=("ok-1",),
            new_cursor="x",
            fell_back_to_cold_start=False,
        )

    handler = EmailSweepHandler(
        bus=bus,
        accounts_store=accounts_store,
        email_store=email_store,
        fetcher=fake_fetcher,
    )
    run = handler(_make_definition())

    assert run.status is HeartbeatStatus.FAILED  # because at least one failed
    assert "quota exceeded" in run.error
    assert "+1" in run.output  # good account still fetched
    assert len(captured) == 1
    assert captured[0].account_id == "gmail:good@gmail.com"


def test_max_messages_param_threaded_through(
    accounts_store: EmailAccountStore, email_store: EmailStore
) -> None:
    """definition.params.max_messages reaches the fetcher."""
    accounts_store.add(provider="gmail", address="user@gmail.com")

    captured_kwargs: dict = {}

    def fake_fetcher(account_id, *, store, max_messages):  # type: ignore[no-untyped-def]
        captured_kwargs["max_messages"] = max_messages
        return FetchResult(
            account_id=account_id,
            fetched=0,
            new_message_ids=(),
            new_cursor=None,
            fell_back_to_cold_start=False,
        )

    handler = EmailSweepHandler(
        bus=None,
        accounts_store=accounts_store,
        email_store=email_store,
        fetcher=fake_fetcher,
    )
    handler(_make_definition(max_messages=37))
    assert captured_kwargs["max_messages"] == 37


def test_factory_wires_default_bus() -> None:
    """build_email_sweep_handler() pulls in the module-level singleton bus."""
    from iris_harness.foundation.eventbus import get_default_bus
    from iris_personal.email.sweep import build_email_sweep_handler

    handler = build_email_sweep_handler()
    assert handler.bus is get_default_bus()


def test_label_changes_go_out_on_their_own_topic(
    accounts_store: EmailAccountStore,
    email_store: EmailStore,
    captured_events: tuple[EventBus, list],
) -> None:
    """Loop-proof PR 5: a relabel read back is ``email.labels_changed``, not new mail."""
    accounts_store.add(provider="gmail", address="user@example.com")
    bus, new_mail = captured_events
    relabels: list = []
    bus.on(EMAIL_LABELS_CHANGED, relabels.append)
    changes = (("old-1", ("INBOX", "Label_7")),)

    def fake_fetcher(account_id, *, store, max_messages):  # type: ignore[no-untyped-def]
        return FetchResult(account_id, 0, (), "9", False, label_changes=changes)

    EmailSweepHandler(
        bus=bus, accounts_store=accounts_store, email_store=email_store, fetcher=fake_fetcher
    )(_make_definition())

    assert new_mail == []
    assert relabels == [
        EmailLabelsChangedPayload(account_id="gmail:user@example.com", changes=changes)
    ]


def test_no_label_changes_no_event(
    accounts_store: EmailAccountStore,
    email_store: EmailStore,
    captured_events: tuple[EventBus, list],
) -> None:
    accounts_store.add(provider="gmail", address="user@example.com")
    bus, _ = captured_events
    relabels: list = []
    bus.on(EMAIL_LABELS_CHANGED, relabels.append)

    def fake_fetcher(account_id, *, store, max_messages):  # type: ignore[no-untyped-def]
        return FetchResult(account_id, 1, ("m1",), "9", False)

    EmailSweepHandler(
        bus=bus, accounts_store=accounts_store, email_store=email_store, fetcher=fake_fetcher
    )(_make_definition())
    assert relabels == []


# ─── The setup gate (owner decision 2026-09-30) ──────────────────────


def _recording_fetcher(calls: list[str]):  # type: ignore[no-untyped-def]
    def fetch(account_id, *, store, max_messages):  # type: ignore[no-untyped-def]
        calls.append(account_id)
        return FetchResult(account_id, 0, (), "1", False)

    return fetch


def _sweep_audit_rows() -> list:  # type: ignore[type-arg]
    from iris_harness.sdk.audit import AuditLog, audit_db_path

    return [r for r in AuditLog(db_path=audit_db_path()).query() if r.plugin == "email_sweep"]


def test_an_account_setup_holds_is_not_swept_and_says_why_once(
    accounts_store: EmailAccountStore, email_store: EmailStore
) -> None:
    from iris_personal.email.sweep_gate import SweepGate

    accounts_store.add(provider="gmail", address="new@gmail.com")
    accounts_store.add(provider="gmail", address="old@gmail.com")
    SweepGate(db_path=email_store.db_path).hold("gmail:new@gmail.com", "setup unfinished")
    calls: list[str] = []
    handler = EmailSweepHandler(
        bus=None,
        accounts_store=accounts_store,
        email_store=email_store,
        fetcher=_recording_fetcher(calls),
    )
    first = handler(_make_definition())
    second = handler(_make_definition())
    # The held account is passed by; the one with no gate row is swept as before.
    assert calls == ["gmail:old@gmail.com", "gmail:old@gmail.com"]
    assert "waiting for email setup: gmail:new@gmail.com" in first.output
    assert "waiting for email setup: gmail:new@gmail.com" in second.output
    # Logged and audited once, not every tick.
    (row,) = [r for r in _sweep_audit_rows() if "new@gmail.com" in r.payload_json]
    assert (row.hook_point, row.decision) == ("email_sweep", "deny")
    assert '"account": "gmail:new@gmail.com"' in row.payload_json
    # The reason names the provider, never the address (the payload holds the id).
    assert row.reason == "email sweep: a gmail account waits for email setup to turn the sweep on"
    assert all("new@" not in r.reason and "old@" not in r.reason for r in _sweep_audit_rows())


def test_a_released_account_is_swept_again(
    accounts_store: EmailAccountStore, email_store: EmailStore
) -> None:
    from iris_personal.email.sweep_gate import SweepGate

    accounts_store.add(provider="gmail", address="fresh@gmail.com")
    gate = SweepGate(db_path=email_store.db_path)
    gate.hold("gmail:fresh@gmail.com", "setup unfinished")
    calls: list[str] = []
    handler = EmailSweepHandler(
        bus=None,
        accounts_store=accounts_store,
        email_store=email_store,
        fetcher=_recording_fetcher(calls),
    )
    held = handler(_make_definition())
    assert held.status is HeartbeatStatus.SKIPPED and calls == []
    assert held.output == "no account to sweep; waiting for email setup: gmail:fresh@gmail.com"
    gate.release("gmail:fresh@gmail.com", "setup turned the sweep on")
    swept = handler(_make_definition())
    assert swept.status is HeartbeatStatus.SUCCESS and calls == ["gmail:fresh@gmail.com"]
    assert "waiting for email setup" not in swept.output
