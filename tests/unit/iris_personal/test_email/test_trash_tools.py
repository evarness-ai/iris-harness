"""trash_email / restore_email over a real store and a fake mailbox (ADR-0118 step 5)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore
from iris_personal.email.trash_tools import build_trash_tools

ACCT = "gmail:owner@gmail.com"


def _msg(mid: str, subject: str, sender: str = "Store X <deals@storex.com>") -> EmailMessage:
    return EmailMessage(
        id=mid,
        provider="gmail",  # type: ignore[arg-type]
        account_id=ACCT,
        from_address=sender,
        subject=subject,
        received_at=datetime(2026, 9, 20, 9, 0, tzinfo=UTC),
        snippet=f"{subject} snippet",
    )


class _Mailbox:
    """A fake Gmail: remembers what it trashed and can give it back."""

    def __init__(self, messages: list[EmailMessage], *, read_only: bool = False) -> None:
        self.by_id = {m.id: m for m in messages}
        self.trashed: list[str] = []
        self.read_only = read_only

    def trash_messages(self, account_id: str, ids: Any) -> list[str]:
        if self.read_only:
            raise PermissionError(
                "owner@gmail.com was connected read-only. Run `iris auth gmail login`."
            )
        done = [i for i in ids if i in self.by_id and i not in self.trashed]
        self.trashed.extend(done)
        return done

    def restore_messages(
        self, account_id: str, ids: Any, *, labels_before: Any = None
    ) -> list[EmailMessage]:
        self.labels_before = dict(labels_before or {})
        back = [self.by_id[i] for i in ids if i in self.trashed]
        self.trashed = [i for i in self.trashed if i not in ids]
        return back


@pytest.fixture()
def world(tmp_path: Path) -> tuple[EmailStore, _Mailbox, dict[str, Any]]:
    messages = [
        _msg("m1", "Your weekly deals are here"),
        _msg("m2", "Last chance: 40% off", "Shop Y <hi@shopy.com>"),
        _msg("m3", "Your flight itinerary", "Airline <no-reply@air.com>"),
    ]
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert_many(messages)
    mailbox = _Mailbox(messages)
    tools = {
        t.name: t for t in build_trash_tools(data_dir=tmp_path, provider_for=lambda a: mailbox)
    }
    return store, mailbox, tools


def test_the_card_names_each_email_by_subject_and_sender(world: Any) -> None:
    _store, _mailbox, tools = world
    card = tools["trash_email"].describe({"ids": ["m1", "m2", "gone"]})
    assert card.title == "Trash 3 emails"
    assert card.lines == (
        "Your weekly deals are here — Store X <deals@storex.com> · 20 Sep",
        "Last chance: 40% off — Shop Y <hi@shopy.com> · 20 Sep",
        "(not in your mail any more: gone)",
    )


def test_trashing_moves_mail_to_trash_and_out_of_search(world: Any) -> None:
    store, mailbox, tools = world
    out = tools["trash_email"].call({"ids": ["m1", "m2"]})

    assert mailbox.trashed == ["m1", "m2"]
    assert out.startswith("Moved 2 emails to Trash (kept 30 days")
    assert store.get("m1") is None and store.get("m3") is not None
    assert [h.id for h in store.search("deals")] == []  # gone from search too
    assert {e[0] for e in store.trashed(["m1", "m2"])} == {"m1", "m2"}


def test_undo_restores_the_last_batch_and_it_is_searchable_again(world: Any) -> None:
    store, mailbox, tools = world
    tools["trash_email"].call({"ids": ["m1"]})
    tools["trash_email"].call({"ids": ["m2"]})

    out = tools["restore_email"].call({})  # "undo that": the most recent batch only

    assert out.startswith("Restored 1 email from Trash.")
    assert mailbox.trashed == ["m1"]
    assert store.get("m2") is not None and store.get("m1") is None
    assert [h.id for h in store.search("chance")] == ["m2"]


def test_restore_by_id(world: Any) -> None:
    store, _mailbox, tools = world
    tools["trash_email"].call({"ids": ["m1", "m2"]})
    tools["restore_email"].call({"ids": ["m1"]})
    assert store.get("m1") is not None and store.get("m2") is None


def test_a_read_only_grant_trashes_nothing_and_says_how_to_fix_it(tmp_path: Path) -> None:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_msg("m1", "Deals"))
    mailbox = _Mailbox([_msg("m1", "Deals")], read_only=True)
    tools = {
        t.name: t for t in build_trash_tools(data_dir=tmp_path, provider_for=lambda a: mailbox)
    }

    out = tools["trash_email"].call({"ids": ["m1"]})

    assert out.startswith("Not trashed (") and "iris auth gmail login" in out
    assert store.get("m1") is not None  # nothing moved locally either


def test_unknown_ids_and_empty_calls_are_reported(world: Any) -> None:
    _store, mailbox, tools = world
    assert tools["trash_email"].call({}).startswith("Error: give the emails to trash")
    out = tools["trash_email"].call({"ids": ["m3", "nope"]})
    assert "Not found, so not trashed: nope" in out and mailbox.trashed == ["m3"]
    assert (
        tools["trash_email"]
        .call({"ids": [f"x{i}" for i in range(51)]})
        .startswith("Error: at most 50")
    )


def test_nothing_to_restore_says_so(world: Any) -> None:
    _store, _mailbox, tools = world
    assert tools["restore_email"].call({}).startswith("Nothing to restore")


# --- validate: the arguments are checked before the owner is asked --------------------


def test_real_ids_pass_validation(world: Any) -> None:
    _store, _mailbox, tools = world
    assert tools["trash_email"].validate({"ids": ["m1", "m2"]}) is None


def test_invented_ids_are_refused_by_name(world: Any) -> None:
    _store, _mailbox, tools = world
    problem = tools["trash_email"].validate({"ids": ["m1", "17vq6d3-120928"]})
    assert problem is not None
    assert "1 email named here is not in the owner's mail: 17vq6d3-120928." in problem
    assert "search_inbox" in problem


def test_a_query_instead_of_ids_is_refused(world: Any) -> None:
    _store, _mailbox, tools = world
    problem = tools["trash_email"].validate({"query": "promotions this month"})
    assert problem is not None
    assert "this is NOT a search result" in problem
    assert "First find the emails with search_inbox or list_by_category" in problem


def test_a_refusal_never_offers_nothing_to_trash_as_an_answer(world: Any) -> None:
    # Phone test 2026-09-21 (image 8b222dfb): the model called trash_email with no ids,
    # the refusal said "if the search finds nothing, tell the owner there is nothing to
    # trash", and it did exactly that, having searched nothing. 170 promos were there.
    _store, _mailbox, tools = world
    for args in ({"ids": []}, {"ids": ["invented-1"]}):
        problem = tools["trash_email"].validate(args)
        assert problem is not None
        assert "nothing to trash" not in problem


def test_too_many_ids_are_refused(world: Any) -> None:
    _store, _mailbox, tools = world
    problem = tools["trash_email"].validate({"ids": [f"x{i}" for i in range(51)]})
    assert problem == "trash_email takes at most 50 emails at a time."


# ── trash_category: code picks the emails, no card (ADR-0118 amendment, 2026-09-22) ──


def _dated(mid: str, days_ago: int, *, labels: tuple[str, ...] = ()) -> EmailMessage:
    from datetime import timedelta

    return EmailMessage(
        id=mid,
        provider="gmail",  # type: ignore[arg-type]
        account_id=ACCT,
        from_address="Shop <hi@shop.com>",
        subject=f"Deal {mid}",
        received_at=datetime.now(UTC) - timedelta(days=days_ago, hours=1),
        labels=labels,
    )


@pytest.fixture()
def promos(tmp_path: Path) -> tuple[EmailStore, _Mailbox, Path]:
    promo = ("CATEGORY_PROMOTIONS",)
    messages = [
        _dated("p-today", 0, labels=promo),
        _dated("p-3d", 3, labels=promo),
        # Refiled by triage: only the Gmail label still says it is a promotion.
        _dated("p-refiled", 4, labels=promo),
        _dated("p-old", 30, labels=promo),
        _dated("bill", 1),
    ]
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert_many(messages)
    for mid in ("p-today", "p-3d", "p-old"):
        store.mark_classified(mid, category="email/promotions", confidence=0.5)
    store.mark_classified("p-refiled", category="email/shopping/apparel/gap", confidence=0.9)
    store.mark_classified("bill", category="email/finance/bills", confidence=0.9)
    return store, _Mailbox(messages), tmp_path


def _category_tools(
    promos: Any, *, window_days: Any = None, provider: Any = "default"
) -> dict[str, Any]:
    _store, mailbox, data_dir = promos
    return {
        t.name: t
        for t in build_trash_tools(
            data_dir=data_dir,
            provider_for=lambda a: mailbox if provider == "default" else provider,
            labels_for=lambda a: {"CATEGORY_PROMOTIONS": "email/promotions"},
            window_days=window_days,
        )
    }


def test_category_trash_moves_every_match_in_the_window_and_nothing_else(promos: Any) -> None:
    store, mailbox, _ = promos
    out = _category_tools(promos)["trash_category"].call({"category": "promo", "since_days": 7})

    assert sorted(mailbox.trashed) == ["p-3d", "p-refiled", "p-today"]
    assert store.get("bill") is not None and store.get("p-old") is not None
    assert out.startswith("Moved 3 emails filed under 'promo'")
    assert "from the last 7 days" in out and "restore_email" in out


def test_category_trash_is_undone_by_restore_with_no_ids(promos: Any) -> None:
    store, mailbox, _ = promos
    tools = _category_tools(promos)
    tools["trash_category"].call({"category": "promo", "since_days": 7})

    out = tools["restore_email"].call({})

    assert out.startswith("Restored 3 emails")
    assert mailbox.trashed == []
    assert store.get("p-refiled") is not None


def test_nothing_in_the_window_says_nothing_to_trash_and_raises_nothing(promos: Any) -> None:
    _store, mailbox, _ = promos
    tool = _category_tools(promos)["trash_category"]
    tool.call({"category": "promo", "since_days": 7})
    trashed = list(mailbox.trashed)

    out = tool.call({"category": "promo", "since_days": 7})

    assert out == "No promo emails from the last 7 days; nothing to trash."
    assert mailbox.trashed == trashed


def test_an_unknown_category_lists_the_real_ones_and_trashes_nothing(promos: Any) -> None:
    _store, mailbox, _ = promos
    out = _category_tools(promos)["trash_category"].call({"category": "zzz"})

    assert out.startswith("No category matches 'zzz', so nothing was trashed.")
    assert "promotions (3)" in out
    assert mailbox.trashed == []


def test_a_name_covering_every_category_is_refused(promos: Any) -> None:
    _store, mailbox, _ = promos
    out = _category_tools(promos)["trash_category"].call({"category": "emails"})

    assert "covers every category" in out and "nothing was trashed" in out
    assert mailbox.trashed == []


def test_more_than_the_cap_moves_the_newest_and_says_how_many_remain(
    promos: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris_personal.email.trash_tools as tt

    monkeypatch.setattr(tt, "_MAX_PER_CATEGORY", 2)
    _store, mailbox, _ = promos
    out = _category_tools(promos)["trash_category"].call({"category": "promo"})

    assert mailbox.trashed == ["p-today", "p-3d"]  # newest first
    assert "2 more promo emails are still in the inbox; asking again moves the next 2." in out


def test_the_window_comes_from_the_owners_words_when_the_model_omits_it(promos: Any) -> None:
    _store, mailbox, _ = promos
    tools = _category_tools(promos, window_days=lambda args: 7)
    tools["trash_category"].call({"category": "promo"})
    assert "p-old" not in mailbox.trashed


def test_no_provider_trashes_nothing(promos: Any) -> None:
    store, _mailbox, _ = promos
    out = _category_tools(promos, provider=None)["trash_category"].call({"category": "promo"})
    assert out.startswith("Error: no mail provider can trash mail for")
    assert store.get("p-today") is not None


def test_category_trash_needs_a_category(promos: Any) -> None:
    out = _category_tools(promos)["trash_category"].call({})
    assert out.startswith("Error: trash_category needs")


def test_the_mailbox_label_decides_so_a_misfiled_receipt_is_spared(promos: Any) -> None:
    """2026-09-22 probe: "promo" also matched IRIS's email/shopping/deals-promotions/amazon,
    where triage had filed order and delivery notices Gmail itself called Updates."""
    store, mailbox, _ = promos
    extra = [
        _dated("order-notice", 1, labels=("CATEGORY_UPDATES",)),
        _dated("amazon-deal", 1, labels=("CATEGORY_PROMOTIONS",)),
    ]
    store.upsert_many(extra)
    mailbox.by_id.update({m.id: m for m in extra})
    for mid in ("order-notice", "amazon-deal"):
        store.mark_classified(
            mid, category="email/shopping/deals-promotions/amazon", confidence=0.9
        )

    out = _category_tools(promos)["trash_category"].call({"category": "promo", "since_days": 7})

    assert "amazon-deal" in mailbox.trashed
    assert "order-notice" not in mailbox.trashed and store.get("order-notice") is not None
    assert "(email/promotions)" in out  # named by the mailbox bucket it used


def test_a_category_with_no_mailbox_label_trashes_by_path(promos: Any) -> None:
    store, mailbox, _ = promos
    out = _category_tools(promos)["trash_category"].call({"category": "bills"})

    assert mailbox.trashed == ["bill"]
    assert "(email/finance/bills)" in out


def test_a_read_only_account_is_skipped_and_the_others_still_move(tmp_path: Path) -> None:
    """2026-09-22, the owner's first real run: the second account was connected
    read-only, came first, and stopped the whole category trash, so none of the main
    account's promotions moved. Now it is skipped with its fix and the rest move."""
    other = "gmail:second@gmail.com"
    promo = ("CATEGORY_PROMOTIONS",)
    main_msgs = [_dated("main-1", 1, labels=promo), _dated("main-2", 2, labels=promo)]
    ro_msg = _dated("ro-1", 0, labels=promo).model_copy(update={"account_id": other})  # newest
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert_many([*main_msgs, ro_msg])
    writable, read_only = _Mailbox(main_msgs), _Mailbox([ro_msg], read_only=True)
    tools = {
        t.name: t
        for t in build_trash_tools(
            data_dir=tmp_path,
            provider_for=lambda a: read_only if a == other else writable,
            labels_for=lambda a: {"CATEGORY_PROMOTIONS": "email/promotions"},
        )
    }

    out = tools["trash_category"].call({"category": "promo", "since_days": 7})

    assert sorted(writable.trashed) == ["main-1", "main-2"]
    assert store.get("ro-1") is not None
    assert out.startswith("Moved 2 emails")
    assert "Not trashed (1 email): owner@gmail.com was connected read-only" in out


def test_the_labels_read_before_the_trash_come_back_on_restore(tmp_path: Path) -> None:
    """The provider's labels at trash time (not the store's, which can be stale) are
    kept in the ledger and handed to restore, which puts back what the trash removed."""
    msgs = [_dated("p1", 1, labels=("CATEGORY_PROMOTIONS",))]
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert_many(msgs)

    class _WithLabels(_Mailbox):
        def current_labels(self, account_id: str, ids: Any) -> dict[str, list[str]]:
            return {mid: ["INBOX", "CATEGORY_PROMOTIONS"] for mid in ids}

    mailbox = _WithLabels(msgs)
    tools = {
        t.name: t
        for t in build_trash_tools(
            data_dir=tmp_path,
            provider_for=lambda a: mailbox,
            labels_for=lambda a: {"CATEGORY_PROMOTIONS": "email/promotions"},
        )
    }
    tools["trash_category"].call({"category": "promo", "since_days": 7})
    assert store.trashed_labels(["p1"]) == {"p1": ["INBOX", "CATEGORY_PROMOTIONS"]}

    tools["restore_email"].call({})

    assert mailbox.labels_before == {"p1": ["INBOX", "CATEGORY_PROMOTIONS"]}


def test_a_ledger_from_before_gains_the_labels_column(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "email.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE trashed (id TEXT PRIMARY KEY, account_id TEXT NOT NULL, "
            "subject TEXT NOT NULL, from_address TEXT NOT NULL, trashed_at TEXT NOT NULL, "
            "batch_id TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO trashed VALUES ('old', 'a', 's', 'f', 't', 'b')")
    store = EmailStore(db_path=db)
    store.ensure_schema()
    assert store.trashed_labels(["old"]) == {}  # an old row restores as Gmail leaves it


def test_a_message_whose_labels_could_not_be_read_is_not_trashed(tmp_path: Path) -> None:
    """Its restore would not know what to put back (Gmail's trash drops INBOX)."""
    msgs = [_dated("read", 1, labels=("CATEGORY_PROMOTIONS",)), _dated("unread", 2)]
    msgs[1] = msgs[1].model_copy(update={"labels": ("CATEGORY_PROMOTIONS",)})
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert_many(msgs)

    class _Partial(_Mailbox):
        def current_labels(self, account_id: str, ids: Any) -> dict[str, list[str]]:
            return {"read": ["INBOX"]}  # "unread" was rate-limited

    mailbox = _Partial(msgs)
    tools = {
        t.name: t
        for t in build_trash_tools(
            data_dir=tmp_path,
            provider_for=lambda a: mailbox,
            labels_for=lambda a: {"CATEGORY_PROMOTIONS": "email/promotions"},
        )
    }
    out = tools["trash_category"].call({"category": "promo", "since_days": 7})

    assert mailbox.trashed == ["read"]
    assert store.get("unread") is not None
    assert "1 more promo emails" in out


def test_a_big_restore_names_a_sample_not_every_email(
    promos: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restore reply is the answer; the whole batch of 200 cost a 79 s model call."""
    import iris_personal.email.trash_tools as tt

    tools = _category_tools(promos)
    tools["trash_category"].call({"category": "promo"})
    monkeypatch.setattr(tt, "_REPLY_SAMPLE", 2)
    out = tools["restore_email"].call({})
    assert out.startswith("Restored 4 emails")
    assert out.count("\n- ") == 2 and out.endswith("(+2 more)")
