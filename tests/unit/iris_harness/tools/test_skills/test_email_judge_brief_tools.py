"""The email-triage skill's digest tools for the email judge (loop-proof PR 5):
``email_needs_reply`` and ``email_judged_yesterday``, the morning brief's
``needs_reply`` and ``judged_yesterday`` slots."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from iris_harness.services.digest.learned import previous_local_day
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.judgments import JudgmentStore

REPO = Path(__file__).resolve().parents[5]


def _tools() -> Any:
    path = REPO / "config" / "skills" / "email" / "email-triage" / "tools.py"
    spec = importlib.util.spec_from_file_location("email_triage_judge_tools_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("IRIS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("IRIS_TZ", "UTC")
    monkeypatch.delenv("IRIS_CONFIG_DIR", raising=False)
    return data_dir


def _judged(data_dir: Path, mid: str, sender: str, subject: str, bucket: str, at: datetime) -> None:
    db = data_dir / "email.db"
    emails = EmailStore(db_path=db)
    emails.ensure_schema()
    emails.upsert(
        EmailMessage(
            id=mid,
            provider="gmail",
            account_id="gmail:owner@example.com",
            thread_id=f"t-{mid}",
            from_address=sender,
            subject=subject,
            received_at=at,
        )
    )
    store = JudgmentStore(db_path=db)
    store.ensure_schema()
    store.record(mid, "gmail:owner@example.com", bucket=bucket, confidence=0.9)


def test_the_brief_declares_both_slots_in_the_inbox_group() -> None:
    manifest = yaml.safe_load(
        (REPO / "config/skills/builtin/morning-briefing/manifest.yaml").read_text()
    )
    slots = manifest["brief"]["slots"]
    assert slots["needs_reply"]["tool"] == "email_needs_reply"
    assert slots["judged_yesterday"]["tool"] == "email_judged_yesterday"
    digest = yaml.safe_load((REPO / "config/digest.yaml").read_text())
    (inbox,) = (g for g in digest["groups"] if g["id"] == "inbox")
    assert inbox["sections"][:3] == ["needs_reply", "focus", "judged_yesterday"]
    assert inbox["push"] == "Focus {focus}"  # the push headline is unchanged


def test_needs_reply_tool_renders_the_section(data: Path) -> None:
    tools = _tools()
    assert tools.EmailNeedsReplyTool()._run() == "## Needs reply\nNothing waiting on you."
    _judged(
        data,
        "m1",
        "Petra Sample <petra@mail.example>",
        "dinner Saturday?",
        "needs_reply",
        datetime.now(UTC),
    )
    assert tools.EmailNeedsReplyTool()._run() == (
        "## Needs reply (1)\n- Petra Sample: dinner Saturday?"
    )


def test_judged_yesterday_tool_renders_the_line(data: Path) -> None:
    tools = _tools()
    assert tools.EmailJudgedYesterdayTool()._run() == ""
    start, _ = previous_local_day(datetime.now(UTC), UTC)  # type: ignore[arg-type]
    _judged(data, "m1", "A <a@mail.example>", "One", "bill", datetime.now(UTC))
    from unittest.mock import patch

    with patch(
        "iris_personal.plugins.email_workflows.judgments._now",
        lambda: (start + timedelta(hours=12)).isoformat(),
    ):
        _judged(data, "m2", "B <b@mail.example>", "Two", "fyi", start)
    assert tools.EmailJudgedYesterdayTool()._run() == "Judged yesterday: 1 · 1 fyi"
