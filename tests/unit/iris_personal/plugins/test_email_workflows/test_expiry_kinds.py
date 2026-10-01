"""Email's two expiry kinds are its manifest's, and the owner sees the same policy
(core/SDK boundary plan PR 5, email slice step 4)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_harness.runtime.plugin_host.manifest import load_manifest
from iris_harness.services.digest import expiry
from iris_harness.services.digest.expiry import expiry_view
from iris_harness.services.digest.settings import load_defaults
from iris_personal.plugins.email_workflows.judge_cards import EmailJudgeActionProvider

from .judge_fixtures import DAY, Inbox, seed_day

REPO = Path(__file__).resolve().parents[5]


def test_the_manifest_declares_the_two_kinds_with_the_owner_approved_defaults() -> None:
    manifest = load_manifest(REPO / "src/iris_personal/plugins/email_workflows/manifest.yaml")
    assert {key: kind.default for key, kind in manifest.expiry.items()} == {
        "needs_reply_days": 3,
        "unsure_card_days": 7,
    }


def test_the_owner_sees_the_same_expiry_policy_as_before_the_move() -> None:
    """The view ``main`` showed at f6259600, when the two keys were core fields."""
    assert expiry_view(load_defaults(REPO / "config").settings.expiry) == {
        "prep_after_event": 0,
        "event_after_end": 0,
        "task_overdue_days": 3,
        "bill_ask_days": 3,
        "bill_overdue_digest_days": 7,
        "inbox_notice_days": 3,
        "reminder_missed_digests": 1,
        "needs_reply_days": 3,
        "unsure_card_days": 7,
    }


def test_the_owner_still_tunes_them_in_digest_yaml(tmp_path: Path) -> None:
    (tmp_path / "digest.yaml").write_text(
        "expiry:\n  needs_reply_days: 5\n  unsure_card_days: 2\n", encoding="utf-8"
    )
    assert expiry.expiry_days("needs_reply_days", tmp_path) == 5
    assert expiry.expiry_days("unsure_card_days", tmp_path) == 2


def test_an_undeclared_card_kind_fails_the_provider_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: drop the manifest's declaration and the Action Center provider raises
    (charged to the plugin by the pending-action fault boundary), rather than quietly
    keeping cards open for a guessed number of days."""
    inbox = Inbox(tmp_path)
    seed_day(inbox, DAY)
    monkeypatch.setattr(expiry, "declared_expiry_kinds", dict)
    provider = EmailJudgeActionProvider(
        inbox.data_dir, now=lambda: datetime(2026, 9, 27, tzinfo=UTC), tz=UTC, emit=inbox.emit
    )
    with pytest.raises(KeyError, match="unsure_card_days"):
        provider.desired_actions()
