"""The judge's IRIS/* Gmail labels (loop-proof PR 5): writing them in bucket groups and
reading the owner's relabels back. A fake provider stands in for Gmail."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from iris_personal.email.events import EmailLabelsChangedPayload
from iris_personal.email.store import EmailStore
from iris_personal.email.write_approvals import approve_mailbox_writes, revoke_mailbox_writes
from iris_personal.plugins.email_workflows.judge_config import (
    EMAIL_JUDGMENT_CORRECTED,
    JudgeConfig,
    JudgmentCorrectedPayload,
)
from iris_personal.plugins.email_workflows.judge_corrections import apply_correction
from iris_personal.plugins.email_workflows.judge_labels import (
    LabelHandlers,
    read_back,
    sync_labels,
)
from iris_personal.plugins.email_workflows.judgments import PROMO, JudgmentStore

A = "gmail:owner@example.com"
B = "gmail:other@example.com"


class FakeGmail:
    """Labels by id; ``modify_labels`` records each call and applies it."""

    def __init__(self, *, refuse: Exception | None = None) -> None:
        self.ids: dict[str, str] = {}
        self.calls: list[tuple[str, list[str], list[str], list[str]]] = []
        self.on: dict[str, set[str]] = {}
        self._refuse = refuse
        self.attempts = 0

    def ensure_labels(self, account_id: str, names: Sequence[str]) -> dict[str, str]:
        for name in names:
            self.ids.setdefault(name, f"Label_{len(self.ids) + 1}")
        return {n: self.ids[n] for n in names}

    def modify_labels(
        self,
        account_id: str,
        message_ids: Sequence[str],
        add_ids: Sequence[str],
        remove_ids: Sequence[str],
    ) -> int:
        self.attempts += 1
        if self._refuse is not None:
            raise self._refuse
        self.calls.append((account_id, list(message_ids), list(add_ids), list(remove_ids)))
        for mid in message_ids:
            labels = self.on.setdefault(mid, {"INBOX", "UNREAD"})
            labels.difference_update(remove_ids)
            labels.update(add_ids)
        return len(message_ids)

    def label(self, name: str) -> str:
        return self.ids[name]


class NoLabels:
    """A mailbox provider without the labelling capability."""


@pytest.fixture(autouse=True)
def writes_approved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both accounts approved mailbox writes (R4) unless a test revokes one; the approvals
    live in this test's own data dir."""
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    approve_mailbox_writes(A, "test")
    approve_mailbox_writes(B, "test")


@pytest.fixture
def config() -> JudgeConfig:
    return JudgeConfig.load()


@pytest.fixture
def store(tmp_path: Path) -> JudgmentStore:
    db = tmp_path / "email.db"
    EmailStore(db_path=db).ensure_schema()
    s = JudgmentStore(db_path=db)
    s.ensure_schema()
    return s


def _judge(store: JudgmentStore, account: str, **buckets: str) -> None:
    for mid, bucket in buckets.items():
        store.record(mid, account, bucket=bucket, confidence=0.9)


# ─── sync_labels ────────────────────────────────────────────────────────────


def test_one_call_per_bucket_adds_it_and_removes_the_other_iris_labels(
    store: JudgmentStore, config: JudgeConfig
) -> None:
    _judge(store, A, m1="bill", m2="bill", m3="fyi")
    gmail = FakeGmail()

    result = sync_labels(store, config, {A: gmail})

    assert (result.written, result.removed, result.failed) == (3, 0, 0)
    assert len(gmail.calls) == 2
    every = {gmail.label(n) for n in config.labels.values()}
    for _acct, ids, add, remove in gmail.calls:
        bucket = "bill" if ids == ["m1", "m2"] else "fyi"
        assert add == [gmail.label(config.labels[bucket])]
        assert set(remove) == every - set(add)
        assert not {"INBOX", "UNREAD"} & set(remove)
    # Exactly one IRIS label each, INBOX and UNREAD untouched.
    assert gmail.on["m1"] == {"INBOX", "UNREAD", gmail.label("IRIS/Bill")}
    assert [j.label_bucket for j in map(store.get, ["m1", "m2", "m3"])] == ["bill", "bill", "fyi"]
    assert store.labels_due(A) == []
    # Nothing due: nothing sent.
    assert sync_labels(store, config, {A: gmail}).written == 0
    assert len(gmail.calls) == 2


def test_a_correction_moves_the_label(store: JudgmentStore, config: JudgeConfig) -> None:
    _judge(store, A, m1="fyi")
    gmail = FakeGmail()
    sync_labels(store, config, {A: gmail})
    apply_correction(store, config, "m1", "needs_reply", source="chat")

    sync_labels(store, config, {A: gmail})

    assert gmail.on["m1"] == {"INBOX", "UNREAD", gmail.label("IRIS/Needs-Reply")}
    assert store.get("m1").label_bucket == "needs_reply"  # type: ignore[union-attr]


def test_promo_takes_every_iris_label_off(store: JudgmentStore, config: JudgeConfig) -> None:
    _judge(store, A, m1="fyi")
    gmail = FakeGmail()
    sync_labels(store, config, {A: gmail})
    apply_correction(store, config, "m1", PROMO, source="card")

    result = sync_labels(store, config, {A: gmail})

    assert (result.written, result.removed) == (0, 1)
    _acct, ids, add, remove = gmail.calls[-1]
    assert (ids, add) == (["m1"], [])
    assert set(remove) == {gmail.label(n) for n in config.labels.values()}
    assert gmail.on["m1"] == {"INBOX", "UNREAD"}
    assert store.get("m1").label_bucket is None  # type: ignore[union-attr]
    assert store.labels_due(A) == []


def test_a_scope_error_on_one_account_does_not_stop_the_other(
    store: JudgmentStore, config: JudgeConfig
) -> None:
    _judge(store, A, m1="bill", m2="fyi")
    _judge(store, B, m3="event")
    refused = FakeGmail(refuse=PermissionError("Run `iris auth gmail login --user x` again"))
    ok = FakeGmail()

    result = sync_labels(store, config, {A: refused, B: ok})

    assert result.accounts[A].failed == 2
    assert refused.attempts == 1  # the grant refuses every group: not tried again
    assert "iris auth gmail login" in result.accounts[A].error
    assert result.accounts[B].written == 1
    assert store.get("m1").label_bucket is None  # type: ignore[union-attr]
    assert store.get("m3").label_bucket == "event"  # type: ignore[union-attr]
    assert "iris auth gmail login" in result.summary()


class CountingGmail(FakeGmail):
    """Counts every provider call, reads included."""

    def __init__(self) -> None:
        super().__init__()
        self.ensure_calls = 0

    def ensure_labels(self, account_id: str, names: Sequence[str]) -> dict[str, str]:
        self.ensure_calls += 1
        return super().ensure_labels(account_id, names)


def test_labels_on_but_writes_not_approved_writes_nothing_and_says_why(
    store: JudgmentStore, config: JudgeConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """IRIS_EMAIL_JUDGE_LABELS on (the default) is not enough: without the account's
    write approval the provider is never asked, the rows stay due, and the log names the
    one command that approves it."""
    revoke_mailbox_writes(A, actor="test", agent_type="test")
    _judge(store, A, m1="bill", m2="fyi")
    _judge(store, B, m3="event")
    gmail_a, gmail_b = CountingGmail(), CountingGmail()

    with caplog.at_level("WARNING"):
        result = sync_labels(store, config, {A: gmail_a, B: gmail_b})

    assert (gmail_a.ensure_calls, gmail_a.attempts) == (0, 0)
    assert result.accounts[A].failed == 2
    assert f"iris email writes approve --account {A}" in result.accounts[A].error
    assert f"iris email writes approve --account {A}" in caplog.text
    assert "2 left unlabelled" in caplog.text
    assert {m for m in ("m1", "m2") if store.get(m).label_bucket} == set()  # type: ignore[union-attr]
    assert len(store.labels_due(A)) == 2  # still due: labelled once approved
    assert result.accounts[B].written == 1  # the approved account is not held back


def test_approving_later_labels_what_was_held(store: JudgmentStore, config: JudgeConfig) -> None:
    revoke_mailbox_writes(A, actor="test", agent_type="test")
    _judge(store, A, m1="bill")
    gmail = FakeGmail()
    assert sync_labels(store, config, {A: gmail}).written == 0

    approve_mailbox_writes(A, "iris email writes approve")
    assert sync_labels(store, config, {A: gmail}).written == 1
    assert gmail.on["m1"] == {"INBOX", "UNREAD", gmail.label("IRIS/Bill")}

    revoke_mailbox_writes(A, actor="test", agent_type="test")
    apply_correction(store, config, "m1", "fyi", source="chat")
    assert sync_labels(store, config, {A: gmail}).failed == 1
    assert gmail.on["m1"] == {"INBOX", "UNREAD", gmail.label("IRIS/Bill")}  # not moved


def test_a_provider_that_cannot_label_is_skipped(store: JudgmentStore, config: JudgeConfig) -> None:
    _judge(store, A, m1="bill")
    result = sync_labels(store, config, {A: NoLabels()})
    assert result.skipped == [A]
    assert store.labels_due(A)  # still due for a provider that can


# ─── read_back ──────────────────────────────────────────────────────────────


def _labelled(store: JudgmentStore, config: JudgeConfig, **buckets: str) -> FakeGmail:
    _judge(store, A, **buckets)
    gmail = FakeGmail()
    sync_labels(store, config, {A: gmail})
    return gmail


def _names(gmail: FakeGmail) -> dict[str, str]:
    return {i: n for n, i in gmail.ids.items()}


def test_owner_relabel_in_gmail_is_a_correction(store: JudgmentStore, config: JudgeConfig) -> None:
    gmail = _labelled(store, config, m1="fyi")
    events: list[tuple[str, Any]] = []
    changes = [("m1", ["INBOX", gmail.label("IRIS/Bill")])]

    got = read_back(
        store, config, changes, lambda t, p: events.append((t, p)), label_names=_names(gmail)
    )

    assert [(c.previous, c.judgment.owner_bucket, c.judgment.owner_source) for c in got] == [
        ("fyi", "bill", "gmail")
    ]
    assert events[0][0] == EMAIL_JUDGMENT_CORRECTED
    assert events[0][1].source == "gmail"
    # Gmail already shows exactly the owner's label: nothing left to write.
    assert store.labels_due(A) == []


def test_owner_added_a_second_label_prefers_the_new_one(
    store: JudgmentStore, config: JudgeConfig
) -> None:
    gmail = _labelled(store, config, m1="fyi")
    changes = [("m1", [gmail.label("IRIS/FYI"), gmail.label("IRIS/Event")])]

    got = read_back(store, config, changes, None, label_names=_names(gmail))

    assert got[0].judgment.owner_bucket == "event"
    # IRIS's old label is still on it: the next sync takes it off.
    sync_labels(store, config, {A: gmail})
    _acct, ids, add, remove = gmail.calls[-1]
    assert add == [gmail.label("IRIS/Event")] and gmail.label("IRIS/FYI") in remove


def test_iris_own_label_echo_is_not_a_correction(store: JudgmentStore, config: JudgeConfig) -> None:
    gmail = _labelled(store, config, m1="fyi")
    events: list[Any] = []
    got = read_back(
        store,
        config,
        [("m1", ["INBOX", "UNREAD", gmail.label("IRIS/FYI")])],
        lambda t, p: events.append(p),
        label_names=_names(gmail),
    )
    assert got == [] and events == []
    assert store.get("m1").owner_bucket is None  # type: ignore[union-attr]


def test_a_stale_echo_does_not_undo_a_chat_correction(
    store: JudgmentStore, config: JudgeConfig
) -> None:
    """Corrected in chat before the label moved: Gmail still shows IRIS's old label,
    and that echo is not the owner putting it back."""
    gmail = _labelled(store, config, m1="fyi")
    apply_correction(store, config, "m1", "bill", source="chat")
    got = read_back(
        store, config, [("m1", ["INBOX", gmail.label("IRIS/FYI")])], None, label_names=_names(gmail)
    )
    assert got == []
    row = store.get("m1")
    assert row is not None and (row.owner_bucket, row.owner_source) == ("bill", "chat")


def test_all_iris_labels_removed_records_nothing(store: JudgmentStore, config: JudgeConfig) -> None:
    gmail = _labelled(store, config, m1="bill")
    got = read_back(store, config, [("m1", ["INBOX"])], None, label_names=_names(gmail))
    assert got == []
    row = store.get("m1")
    assert row is not None and row.owner_bucket is None and row.label_bucket == "bill"


def test_unjudged_mail_is_ignored(store: JudgmentStore, config: JudgeConfig) -> None:
    store.mark_waiting(A, ["w1"])
    got = read_back(store, config, [("w1", ["IRIS/Bill"]), ("nobody", ["IRIS/Bill"])], None)
    assert got == []


# ─── The bus handlers ───────────────────────────────────────────────────────


def _handlers(store: JudgmentStore, gmail: Any, events: list[Any]) -> LabelHandlers:
    return LabelHandlers(
        store_factory=lambda: store,
        provider_for=lambda account_id: gmail,
        emit=lambda topic, payload: events.append((topic, payload)),
    )


def test_labels_changed_event_reads_back(store: JudgmentStore, config: JudgeConfig) -> None:
    gmail = _labelled(store, config, m1="fyi")
    events: list[Any] = []
    payload = EmailLabelsChangedPayload(
        account_id=A, changes=(("m1", ("INBOX", gmail.label("IRIS/Needs-Reply"))),)
    )

    got = _handlers(store, gmail, events).on_labels_changed(payload)

    assert [c.judgment.owner_bucket for c in got] == ["needs_reply"]
    assert events and events[0][1].source == "gmail"


def test_a_correction_event_moves_that_accounts_label_now(
    store: JudgmentStore, config: JudgeConfig
) -> None:
    gmail = _labelled(store, config, m1="fyi")
    _judge(store, B, m9="bill")  # another account: not this event's business
    apply_correction(store, config, "m1", "bill", source="chat")
    payload = JudgmentCorrectedPayload(
        message_id="m1", account_id=A, bucket="bill", previous="fyi", source="chat"
    )
    accounts: list[str] = []
    handlers = _handlers(store, gmail, [])
    handlers.provider_for = lambda account_id: accounts.append(account_id) or gmail

    result = handlers.on_corrected(payload)

    assert result is not None and result.written == 1
    assert accounts == [A]
    assert gmail.on["m1"] == {"INBOX", "UNREAD", gmail.label("IRIS/Bill")}
    assert store.get("m9").label_bucket is None  # type: ignore[union-attr]


def test_a_gmail_correction_or_labels_off_does_not_sync(
    store: JudgmentStore, config: JudgeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    gmail = _labelled(store, config, m1="fyi")
    calls = len(gmail.calls)
    handlers = _handlers(store, gmail, [])
    gmail_payload = JudgmentCorrectedPayload(
        message_id="m1", account_id=A, bucket="bill", previous="fyi", source="gmail"
    )
    assert handlers.on_corrected(gmail_payload) is None

    monkeypatch.setenv("IRIS_EMAIL_JUDGE_LABELS", "0")
    chat_payload = JudgmentCorrectedPayload(
        message_id="m1", account_id=A, bucket="bill", previous="fyi", source="chat"
    )
    assert handlers.on_corrected(chat_payload) is None
    changed = EmailLabelsChangedPayload(account_id=A, changes=(("m1", ("IRIS/Bill",)),))
    assert handlers.on_labels_changed(changed) == []
    assert len(gmail.calls) == calls
