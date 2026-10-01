"""Tests for the email_inbox_summary brief tool (Phase 2 Track 2C).

The other email-triage tools (ClassifyEmailByIdTool, RunEmailTriageTool)
require a Tier 3 local LLM and are exercised end-to-end elsewhere.
EmailInboxSummaryTool is pure SQL over email.db, so it can be tested
without external dependencies.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore


def _load_tools_module():
    repo_root = Path(__file__).resolve().parents[5]
    module_path = repo_root / "config" / "skills" / "email" / "email-triage" / "tools.py"
    spec = importlib.util.spec_from_file_location("email_triage_tools_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def email_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "data").mkdir()
    target = tmp_path / "data" / "email.db"
    # The tool reads email.db under IRIS_DATA_DIR, like every store.
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    store = EmailStore(db_path=target)
    store.ensure_schema()
    return target


def _seed(db_path: Path) -> None:
    store = EmailStore(db_path=db_path)
    store.upsert(
        EmailMessage(
            id="a",
            provider="gmail",
            account_id="gmail:user@a.com",
            from_address="x@a.com",
            received_at=datetime(2026, 5, 1, tzinfo=UTC),
        )
    )
    store.upsert(
        EmailMessage(
            id="b",
            provider="gmail",
            account_id="gmail:user@a.com",
            from_address="y@a.com",
            received_at=datetime(2026, 5, 2, tzinfo=UTC),
        )
    )
    store.upsert(
        EmailMessage(
            id="c",
            provider="gmail",
            account_id="gmail:user@b.com",
            from_address="z@b.com",
            received_at=datetime(2026, 5, 3, tzinfo=UTC),
        )
    )
    store.mark_classified("a", category="email/personal/work", confidence=0.9)
    store.mark_pending_review("b")
    # c stays unclassified


def test_inbox_summary_returns_empty_when_no_emails(email_db: Path) -> None:
    mod = _load_tools_module()
    tool = mod.EmailInboxSummaryTool()
    assert tool._run() == []


def test_inbox_summary_aggregates_across_accounts(email_db: Path) -> None:
    _seed(email_db)
    mod = _load_tools_module()
    tool = mod.EmailInboxSummaryTool()
    out = tool._run()
    assert len(out) == 1
    row = out[0]
    assert row["scope"] == "all accounts"
    assert row["classified"] == "1"
    assert row["pending"] == "1"
    assert row["unclassified"] == "1"
    assert row["total"] == "3"


def test_inbox_summary_filters_by_account(email_db: Path) -> None:
    _seed(email_db)
    mod = _load_tools_module()
    tool = mod.EmailInboxSummaryTool()
    out = tool._run(account_id="gmail:user@a.com")
    assert len(out) == 1
    assert out[0]["scope"] == "gmail:user@a.com"
    assert out[0]["total"] == "2"
    assert out[0]["classified"] == "1"
    assert out[0]["pending"] == "1"


def test_inbox_summary_returns_empty_when_account_unknown(email_db: Path) -> None:
    _seed(email_db)
    mod = _load_tools_module()
    tool = mod.EmailInboxSummaryTool()
    assert tool._run(account_id="gmail:ghost@nowhere.com") == []


def test_inbox_summary_last_24h_view_is_one_line_per_mailbox(email_db: Path) -> None:
    """The digest's view: each mailbox's last 24 h grouped by triage category."""
    from datetime import timedelta

    store = EmailStore(db_path=email_db)
    now = datetime.now(UTC)
    for mid, account, category, hours_ago in (
        ("a1", "gmail:owner@a.com", "email/updates/github", 1),
        ("a2", "gmail:owner@a.com", "email/updates/bank", 2),
        ("a3", "gmail:owner@a.com", "email/finance/cards", 3),
        ("a4", "gmail:owner@a.com", None, 4),
        ("a-old", "gmail:owner@a.com", "email/finance/cards", 30),
        ("b1", "gmail:owner@b.com", "email/promo", 0.5),
    ):
        store.upsert(
            EmailMessage(
                id=mid,
                provider="gmail",
                account_id=account,
                from_address="x@example.com",
                received_at=now - timedelta(hours=hours_ago),
            )
        )
        if category:
            store.mark_classified(mid, category=category, confidence=0.9)

    out = _load_tools_module().EmailInboxSummaryTool()._run(view="last_24h")

    assert [(r["account"], r["summary"]) for r in out] == [
        ("owner@b.com", "1 in the last 24 h — 1 promo"),
        ("owner@a.com", "4 in the last 24 h — 2 updates · 1 finance · unsorted 1"),
    ]


def test_morning_briefing_email_summary_slot_uses_the_last_24h_view() -> None:
    """The digest slot asks for the per-mailbox view and its template fits the rows."""
    import yaml

    repo_root = Path(__file__).resolve().parents[5]
    manifest = yaml.safe_load(
        (repo_root / "config/skills/builtin/morning-briefing/manifest.yaml").read_text()
    )
    slot = manifest["brief"]["slots"]["email_summary"]
    assert slot["tool"] == "email_inbox_summary"
    assert slot["args"] == {"view": "last_24h"}
    row = {
        "account": "owner@a.com",
        "account_id": "gmail:owner@a.com",
        "total": "1",
        "summary": "1 in the last 24 h — 1 promo",
    }
    assert slot["item_template"].format(**row) == "owner@a.com: 1 in the last 24 h — 1 promo"


# ─── email_focus: the digest's Focus section (loop-proof PR 2, D17) ──────────


def test_email_focus_tool_reads_settings_and_hides_not_useful_senders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import timedelta

    from iris_harness.sdk import digest as digest_settings
    from iris_harness.services.learning.suppression import (
        EMAIL_FOCUS_SURFACE,
        EMAIL_SEARCH_SUBSYSTEM,
        NOT_USEFUL,
        SurfaceFeedbackStore,
        email_focus_dims,
    )

    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        digest_settings,
        "load_digest_settings",
        lambda _data_dir, _config_dir=None: digest_settings.DigestSettings(
            focus_categories=("personal", "finance"), focus_limit=10, focus_per_account=5
        ),
    )
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    now = datetime.now(UTC)
    for mid, sender, category in (
        ("keep", "School <office@school.example>", "email/personal"),
        ("hide", "GoldenPi <news@goldenpi.example>", "email/finance/investing"),
        ("promo", "Shop <deals@shop.example>", "email/promotions"),
    ):
        store.upsert(
            EmailMessage(
                id=mid,
                provider="gmail",
                account_id="gmail:owner@example.com",
                from_address=sender,
                subject=f"subject {mid}",
                received_at=now - timedelta(hours=1),
            )
        )
        store.mark_classified(mid, category=category, confidence=0.9)
    ledger = SurfaceFeedbackStore(db_path=tmp_path / "learning.db")
    ledger.ensure_schema()
    ledger.record(
        EMAIL_SEARCH_SUBSYSTEM,
        EMAIL_FOCUS_SURFACE,
        email_focus_dims("news@goldenpi.example"),
        NOT_USEFUL,
        emit_signal=False,
    )

    text = _load_tools_module().EmailFocusTool()._run()

    assert text.splitlines() == [
        "## Focus — personal · finance, newest 5 per inbox",
        "- School · subject keep · personal [👎](iris:not-useful/office%40school.example)",
    ]


def test_email_focus_is_declared_in_the_manifest() -> None:
    from iris_harness.tools.skills.loader import load_skill_manifest

    repo_root = Path(__file__).resolve().parents[5]
    manifest = load_skill_manifest(repo_root / "config" / "skills" / "email" / "email-triage")
    assert "email_focus" in {t.name for t in manifest.tools}
    assert "email_focus" in {cls().name for cls in _load_tools_module().SKILL_TOOLS}
