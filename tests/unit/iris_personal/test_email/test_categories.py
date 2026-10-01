"""Plain category names resolve to the categories that exist (promo grill, decision 5).

"promo" is what the owner says; ``email/promotions`` is what the store keeps. Before
this, a category was a raw prefix on the path, so "promo" matched nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_personal.email.categories import category_menu, resolve_category, resolve_paths
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import CategoryFilter, EmailStore

ACCT = "gmail:owner@gmail.com"
PATHS = [
    "email/promotions",
    "email/shopping/deals-promotions/amazon",
    "email/shopping/apparel/gap",
    "email/news/ai-industry/google",
    "email/finance/credit-cards/nerdwallet",
    "email/social",
    "Newsletters/AI",
]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("promo", ["email/promotions", "email/shopping/deals-promotions/amazon"]),
        # Filler and the root every path shares narrow nothing.
        ("my promo emails", ["email/promotions", "email/shopping/deals-promotions/amazon"]),
        ("promotional", ["email/promotions", "email/shopping/deals-promotions/amazon"]),
        ("credit cards", ["email/finance/credit-cards/nerdwallet"]),
        ("Newsletters", ["Newsletters/AI"]),
        # A shorter path word is not a match for a longer asked word.
        ("newsletters", ["Newsletters/AI"]),
        ("shoes", []),
        ("xyz", []),
        # A name with a slash keeps the old prefix meaning.
        (
            "email/shopping",
            ["email/shopping/apparel/gap", "email/shopping/deals-promotions/amazon"],
        ),
    ],
)
def test_names_resolve_by_word_prefix(name: str, expected: list[str]) -> None:
    assert resolve_paths(name, PATHS) == expected


def test_a_word_every_path_shares_resolves_to_all_of_them() -> None:
    # "email" names the whole mailbox; the trash tool refuses exactly this.
    assert resolve_paths("email", PATHS) == sorted(p for p in PATHS if p.startswith("email/"))


def _msg(mid: str, *, labels: tuple[str, ...] = (), day: int = 20) -> EmailMessage:
    return EmailMessage(
        id=mid,
        provider="gmail",  # type: ignore[arg-type]
        account_id=ACCT,
        from_address="Shop <hi@shop.com>",
        subject=f"subject {mid}",
        received_at=datetime(2026, 9, day, 9, 0, tzinfo=UTC),
        labels=labels,
    )


@pytest.fixture()
def store(tmp_path: Path) -> EmailStore:
    s = EmailStore(db_path=tmp_path / "email.db")
    s.ensure_schema()
    s.upsert_many(
        [
            _msg("vendor-promo", labels=("CATEGORY_PROMOTIONS",)),
            _msg("refiled-promo", labels=("CATEGORY_PROMOTIONS",)),
            _msg("deal"),
            _msg("bill"),
        ]
    )
    s.mark_classified("vendor-promo", category="email/promotions", confidence=0.5)
    # Triage refiled this promotion under a path of its own: only the label says promo.
    s.mark_classified("refiled-promo", category="email/shopping/apparel/gap", confidence=0.9)
    s.mark_classified("deal", category="email/shopping/deals-promotions/amazon", confidence=0.9)
    s.mark_classified("bill", category="email/finance/bills", confidence=0.9)
    return s


def _labels(_account: str) -> dict[str, str]:
    return {"CATEGORY_PROMOTIONS": "email/promotions", "CATEGORY_SOCIAL": "email/social"}


def test_a_resolved_category_carries_the_provider_label(store: EmailStore) -> None:
    resolved = resolve_category(store, "promo", ACCT, labels_for=_labels)

    assert resolved.filter == CategoryFilter(
        paths=("email/promotions", "email/shopping/deals-promotions/amazon"),
        labels=("CATEGORY_PROMOTIONS",),
    )
    ids = {m.id for m in store.list_by_category(ACCT, resolved.filter)}
    # The refiled promotion is found by its label; the bill is not touched.
    assert ids == {"vendor-promo", "refiled-promo", "deal"}
    assert store.count_by_category(ACCT, resolved.filter) == 3


def test_a_provider_path_with_no_stored_mail_still_resolves(store: EmailStore) -> None:
    resolved = resolve_category(store, "social", ACCT, labels_for=_labels)
    assert resolved.found
    assert resolved.filter.labels == ("CATEGORY_SOCIAL",)
    assert store.list_by_category(ACCT, resolved.filter) == []


def test_without_a_provider_only_paths_match(store: EmailStore) -> None:
    resolved = resolve_category(store, "promo", ACCT, labels_for=lambda _a: {})
    ids = {m.id for m in store.list_by_category(ACCT, resolved.filter)}
    assert ids == {"vendor-promo", "deal"}


def test_the_menu_groups_categories_two_levels_deep(store: EmailStore) -> None:
    assert category_menu(store) == (
        "Categories in the mail: shopping (2), finance (1), promotions (1)."
    )


def test_the_menu_on_an_uncategorised_store(tmp_path: Path) -> None:
    empty = EmailStore(db_path=tmp_path / "empty.db")
    empty.ensure_schema()
    assert category_menu(empty) == "No email has a category yet."


def test_a_filter_path_matches_itself_and_children_not_siblings(store: EmailStore) -> None:
    store.upsert(_msg("sibling"))
    store.mark_classified("sibling", category="email/promotionsx", confidence=0.9)

    ids = {m.id for m in store.list_by_category(ACCT, CategoryFilter(paths=("email/promotions",)))}

    assert ids == {"vendor-promo"}


def test_an_empty_filter_matches_nothing(store: EmailStore) -> None:
    assert store.list_by_category(ACCT, CategoryFilter()) == []
    assert store.search("subject", category_prefix=CategoryFilter()) == []


def test_search_honours_the_label_half_of_a_filter(store: EmailStore) -> None:
    hits = store.search(
        "subject", account_id=ACCT, category_prefix=CategoryFilter(labels=("CATEGORY_PROMOTIONS",))
    )
    assert {h.id for h in hits} == {"vendor-promo", "refiled-promo"}


def test_count_and_list_respect_the_window(store: EmailStore) -> None:
    store.upsert(_msg("old-promo", labels=("CATEGORY_PROMOTIONS",), day=1))
    promo = CategoryFilter(labels=("CATEGORY_PROMOTIONS",))
    since = datetime(2026, 9, 15, tzinfo=UTC)

    assert store.count_by_category(ACCT, promo) == 3
    assert store.count_by_category(ACCT, promo, since=since) == 2
    assert "old-promo" not in {m.id for m in store.list_by_category(ACCT, promo, since=since)}


def test_a_local_time_window_is_compared_in_utc(store: EmailStore) -> None:
    """The window is built in local time, the mail is stored in UTC; compared as text,
    a -05:00 start let five extra hours in (2026-09-22 end-to-end run)."""
    from datetime import timedelta, timezone

    cdt = timezone(timedelta(hours=-5))
    store.upsert(_msg("inside", labels=("CATEGORY_PROMOTIONS",), day=21))
    # 20 Sep 09:00 UTC is 04:00 CDT: before a window that starts at 20 Sep 08:00 CDT.
    since = datetime(2026, 9, 20, 8, 0, tzinfo=cdt)
    promo = CategoryFilter(labels=("CATEGORY_PROMOTIONS",))

    ids = {m.id for m in store.list_by_category(ACCT, promo, since=since)}

    assert ids == {"inside"}
    assert store.count_by_category(ACCT, promo, since=since) == 1
