"""Senders whose mail is mostly promotions stay off the memory Map (ADR-0119, decision 9).

Every "check my email" chat lists the senders it read, so a daily shop recurred in every
summary and the Map drew it next to the people the owner knows. The rule: at least
``share`` of a sender's stored mail in the configured categories, over at least
``min_messages``. Synthetic senders only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.map_exclusions import (
    CONFIG_PATH,
    ExclusionRule,
    build_provider,
    load_rule,
    promotional_senders,
)

ACCT = "gmail:user@example.com"
PROMO = "email/promotions"
RULE = ExclusionRule(categories=("promotions",), share=0.8, min_messages=3, cache_seconds=0)


def _no_labels(_account: str) -> dict[str, str]:
    return {}


@pytest.fixture
def store(tmp_path: Path) -> EmailStore:
    s = EmailStore(db_path=tmp_path / "email.db")
    s.ensure_schema()
    return s


_counter = iter(range(1_000_000))


def _mail(
    store: EmailStore,
    sender: str,
    category: str | None,
    *,
    n: int = 1,
    labels: tuple[str, ...] = (),
    account: str = ACCT,
) -> None:
    for _ in range(n):
        i = next(_counter)
        store.upsert(
            EmailMessage(
                id=f"m{i}",
                provider="gmail",  # type: ignore[arg-type]
                account_id=account,
                from_address=sender,
                to=("user@example.com",),
                subject=f"s{i}",
                received_at=datetime(2026, 9, 1, tzinfo=UTC),
                labels=labels,
            )
        )
        if category is not None:
            store.mark_classified(f"m{i}", category=category, confidence=0.9)


def _senders(store: EmailStore, rule: ExclusionRule = RULE, **kw) -> list[str]:
    return promotional_senders(store, rule, labels_for=kw.get("labels_for", _no_labels))


SHOP = "Shop <deals@mail.shop.example>"
FRIEND = "Petra Sutton <petra@friends.example>"


class TestTheShare:
    def test_a_sender_at_exactly_the_share_is_left_off(self, store: EmailStore) -> None:
        _mail(store, SHOP, PROMO, n=8)
        _mail(store, SHOP, "email/updates", n=2)

        assert "Shop" in _senders(store)

    def test_a_sender_just_under_the_share_is_drawn(self, store: EmailStore) -> None:
        # 79 of 100 — one message short of the line.
        _mail(store, SHOP, PROMO, n=79)
        _mail(store, SHOP, "email/updates", n=21)

        assert "Shop" not in _senders(store)

    def test_a_correspondent_with_some_promotions_is_drawn(self, store: EmailStore) -> None:
        _mail(store, FRIEND, PROMO, n=2)
        _mail(store, FRIEND, "email/personal", n=6)

        assert "Petra Sutton" not in _senders(store)

    def test_too_few_messages_is_not_a_pattern(self, store: EmailStore) -> None:
        _mail(store, FRIEND, PROMO, n=2)

        assert _senders(store) == []

    def test_min_messages_is_inclusive(self, store: EmailStore) -> None:
        _mail(store, SHOP, PROMO, n=3)

        assert "Shop" in _senders(store)

    def test_unclassified_mail_counts_against_the_share(self, store: EmailStore) -> None:
        _mail(store, SHOP, PROMO, n=3)
        _mail(store, SHOP, None, n=3)

        assert "Shop" not in _senders(store)


class TestWhatCountsAsPromotional:
    def test_a_path_below_the_category_counts(self, store: EmailStore) -> None:
        _mail(store, SHOP, PROMO, n=1)
        _mail(store, SHOP, "email/promotions/shoes", n=3)

        assert "Shop" in _senders(store)

    def test_a_refiled_promotion_counts_by_its_provider_label(self, store: EmailStore) -> None:
        # Triage moved it under a shop path; the mailbox still labels it a promotion.
        _mail(store, SHOP, PROMO, n=1)
        _mail(store, SHOP, "email/shopping/pharmacy", n=3, labels=("CATEGORY_PROMOTIONS",))

        found = _senders(store, labels_for=lambda _a: {"CATEGORY_PROMOTIONS": PROMO})

        assert "Shop" in found

    def test_no_categories_configured_leaves_nothing_off(self, store: EmailStore) -> None:
        _mail(store, SHOP, PROMO, n=5)
        rule = ExclusionRule(categories=(), share=0.8, min_messages=3, cache_seconds=0)

        assert _senders(store, rule) == []


class TestNamesAndDomains:
    def test_the_domain_is_left_off_with_the_name(self, store: EmailStore) -> None:
        _mail(store, SHOP, PROMO, n=3)

        assert _senders(store) == ["Shop", "mail.shop.example"]

    def test_a_shared_domain_is_judged_on_all_its_mail(self, store: EmailStore) -> None:
        _mail(store, "Shop <deals@common.example>", PROMO, n=3)
        _mail(store, "Petra <petra@common.example>", "email/personal", n=3)

        found = _senders(store)

        assert "Shop" in found
        assert "common.example" not in found

    def test_a_name_is_judged_across_accounts(self, store: EmailStore) -> None:
        _mail(store, FRIEND, PROMO, n=3)
        _mail(store, FRIEND, "email/personal", n=3, account="gmail:other@example.com")

        assert "Petra Sutton" not in _senders(store)

    def test_a_bare_address_is_named_by_the_address(self, store: EmailStore) -> None:
        _mail(store, "deals@mail.shop.example", PROMO, n=3)

        assert "deals@mail.shop.example" in _senders(store)


class TestTheProvider:
    def test_it_caches_until_the_window_passes(self, store: EmailStore) -> None:
        now = [0.0]
        rule = ExclusionRule(
            categories=("promotions",), share=0.8, min_messages=3, cache_seconds=60
        )
        provider = build_provider(lambda: store, rule_loader=lambda: rule, clock=lambda: now[0])

        assert provider() == []
        _mail(store, SHOP, PROMO, n=3)
        now[0] = 59.0
        assert provider() == []
        now[0] = 60.0
        assert "Shop" in provider()

    def test_a_missing_mail_store_leaves_nothing_off(self, tmp_path: Path) -> None:
        provider = build_provider(
            lambda: EmailStore(db_path=tmp_path / "absent.db"), rule_loader=lambda: RULE
        )

        assert provider() == []
        assert not (tmp_path / "absent.db").exists()

    def test_the_shipped_rule_is_eighty_percent_of_promotions_and_news(self) -> None:
        """Owner, 2026-09-22: newsletters (news) stay off the Map, but not `updates` —
        Gmail files bank and brokerage notices there too."""
        rule = load_rule(CONFIG_PATH)

        assert rule == ExclusionRule(
            categories=("promotions", "news"), share=0.8, min_messages=3, cache_seconds=600
        )


def test_the_plugin_registers_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.memory import map_exclusions
    from iris_personal.plugins.email_workflows import plugin

    registered: dict[str, object] = {}
    monkeypatch.setattr(
        map_exclusions, "_providers", registered
    )  # the SDK re-exports the same function, which reads this dict

    class _Api:
        def subscribe(self, *_a: object, **_k: object) -> None: ...
        def register_heartbeat(self, *_a: object, **_k: object) -> None: ...
        def register_api_router(self, *_a: object, **_k: object) -> None: ...
        def register_credential_check(self, *_a: object, **_k: object) -> None: ...

    monkeypatch.setattr(plugin, "_register_agent", lambda _api: None)
    monkeypatch.setattr("iris_personal.plugins.email_workflows.tools.register", lambda _api: None)
    monkeypatch.setattr(
        "iris_personal.plugins.email_workflows.judge_surfaces.register", lambda _api: None
    )
    plugin.setup(_Api())  # type: ignore[arg-type]

    assert "email_workflows.promotional_senders" in registered
