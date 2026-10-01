"""The gmail plugin's tab table (``vendor_categories.yaml``) and the one mapping
function both the fetcher and ``iris email label-from-vendor`` use (promo grill, PR 2)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore
from iris_personal.plugins.gmail import gmail_fetch
from iris_personal.plugins.gmail.vendor_categories import load_table, vendor_category_for

ACCOUNT = "gmail:user@gmail.com"


def test_shipped_table_maps_the_four_bulk_tabs_and_not_primary() -> None:
    table = load_table()
    assert table == {
        "CATEGORY_PROMOTIONS": "email/promotions",
        "CATEGORY_SOCIAL": "email/social",
        "CATEGORY_UPDATES": "email/updates",
        "CATEGORY_FORUMS": "email/forums",
        "CATEGORY_PERSONAL": None,
    }


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (("INBOX", "CATEGORY_PROMOTIONS", "UNREAD"), "email/promotions"),
        (("CATEGORY_SOCIAL",), "email/social"),
        (("CATEGORY_UPDATES",), "email/updates"),
        (("CATEGORY_FORUMS",), "email/forums"),
        (("INBOX", "CATEGORY_PERSONAL"), None),
        (("INBOX",), None),
        ((), None),
        # Two tab labels resolve by table order, every time.
        (("CATEGORY_UPDATES", "CATEGORY_PROMOTIONS"), "email/promotions"),
    ],
)
def test_vendor_category_for(labels: tuple[str, ...], expected: str | None) -> None:
    assert vendor_category_for(labels) == expected


def test_table_is_read_from_yaml(tmp_path: Path) -> None:
    """The label names are config: a different table maps differently."""
    custom = tmp_path / "tabs.yaml"
    custom.write_text("tabs:\n  MY_LABEL: email/custom\n", encoding="utf-8")
    assert vendor_category_for(["MY_LABEL"], table_path=custom) == "email/custom"
    assert vendor_category_for(["CATEGORY_PROMOTIONS"], table_path=custom) is None


def test_malformed_table_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("tabs:\n  X: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="topic path or null"):
        load_table(bad)


def test_parse_message_sets_vendor_category() -> None:
    payload = {
        "id": "m-1",
        "threadId": "t-1",
        "labelIds": ["INBOX", "CATEGORY_PROMOTIONS"],
        "internalDate": "1767225600000",
        "snippet": "50% off",
        "payload": {"headers": [{"name": "From", "value": "Shop <deals@shop.example>"}]},
    }
    msg = gmail_fetch._parse_message(payload, account_id=ACCOUNT)
    assert msg.vendor_category == "email/promotions"
    payload["labelIds"] = ["INBOX", "CATEGORY_PERSONAL"]
    assert gmail_fetch._parse_message(payload, account_id=ACCOUNT).vendor_category is None


# ─── iris email label-from-vendor ─────────────────────────────────────


def _seed(db: Path) -> EmailStore:
    store = EmailStore(db_path=db)
    store.ensure_schema()

    def msg(id: str, *labels: str) -> EmailMessage:
        return EmailMessage(
            id=id,
            provider="gmail",
            account_id=ACCOUNT,
            from_address="x@example.com",
            received_at=datetime(2026, 9, 1, tzinfo=UTC),
            labels=labels,
        )

    # Stored before the fetcher mapped tabs: labels present, no classification.
    store.upsert_many(
        [
            msg("p1", "INBOX", "CATEGORY_PROMOTIONS"),
            msg("p2", "CATEGORY_PROMOTIONS"),
            msg("s1", "CATEGORY_SOCIAL"),
            msg("primary", "INBOX", "CATEGORY_PERSONAL"),
            msg("iris", "CATEGORY_PROMOTIONS"),
        ]
    )
    store.mark_classified("iris", category="email/shopping/apparel", confidence=0.9)
    return store


def _run(db: Path, *extra: str) -> tuple[int, str]:
    result = CliRunner().invoke(
        app,
        ["email", "label-from-vendor", "--account", ACCOUNT, "--email-db", str(db), *extra],
    )
    return result.exit_code, result.output


def test_label_from_vendor_dry_run_writes_nothing(tmp_path: Path) -> None:
    db = tmp_path / "email.db"
    store = _seed(db)
    code, out = _run(db, "--dry-run")
    assert code == 0, out
    assert "would label" in out and "3" in out
    assert "email/promotions: 2" in out
    assert "dry run" in out
    assert store.list_by_category(ACCOUNT, "email/promotions") == []


def test_label_from_vendor_writes_then_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "email.db"
    store = _seed(db)
    code, out = _run(db)
    assert code == 0, out
    assert "labelled" in out
    promos = {m.id for m in store.list_by_category(ACCOUNT, "email/promotions")}
    assert promos == {"p1", "p2"}  # the IRIS-classified promo is left alone
    assert {m.id for m in store.list_by_category(ACCOUNT, "email/social")} == {"s1"}
    iris = store.get("iris")
    assert iris is not None and iris.classified_category == "email/shopping/apparel"

    code, out = _run(db)
    assert code == 0, out
    assert "labelled 0" in out
    assert "already labelled 3" in out


def test_label_from_vendor_exits_2_without_a_db(tmp_path: Path) -> None:
    code, out = _run(tmp_path / "missing.db")
    assert code == 2
    assert "email.db not found" in out


def test_the_provider_hands_the_email_tools_its_tab_labels() -> None:
    """The email library matches a tab's mail by label without knowing Gmail's names:
    the provider hands it the table, Primary (no path) left out (promo grill, PR 3b)."""
    from iris_personal.email.categories import provider_labels
    from iris_personal.email.providers import clear_mail_providers, register_mail_provider
    from iris_personal.plugins.gmail.provider import GmailProvider

    expected = {
        "CATEGORY_PROMOTIONS": "email/promotions",
        "CATEGORY_SOCIAL": "email/social",
        "CATEGORY_UPDATES": "email/updates",
        "CATEGORY_FORUMS": "email/forums",
    }
    assert GmailProvider().category_labels() == expected
    clear_mail_providers()
    register_mail_provider(GmailProvider())
    try:
        assert provider_labels(ACCOUNT) == expected
    finally:
        clear_mail_providers()
    assert provider_labels(ACCOUNT) == {}
