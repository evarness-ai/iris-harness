"""Tests for the harness agent-console composer (ADR-0074)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from iris_harness.runtime.agent_console import (
    agent_pending_actions,
    agent_settings,
    list_agents,
    render_agents_overview,
)
from iris_harness.services.tasks import TaskAction, TaskStore


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    s = TaskStore(db_path=tmp_path / "tasks.db")
    s.ensure_schema()
    return s


def test_list_agents_from_catalog() -> None:
    names = {a["name"] for a in list_agents()}
    assert {"finance", "email", "system"} <= names
    finance = next(a for a in list_agents() if a["name"] == "finance")
    assert finance["source_kind"] == "finance-statements"


def test_list_agents_registry_wins() -> None:
    agents = list_agents(registered={"finance"})
    assert [a["name"] for a in agents] == ["finance"]


def test_agent_pending_actions_filters_by_source_kind(store: TaskStore) -> None:
    store.create(
        title="Wingtip card needs a password",
        source_kind="finance-statements",
        action=TaskAction(kind="copy_command", label="Set", command="iris finance secret ..."),
    )
    store.create(title="buy milk")  # not an action, not finance
    assert len(agent_pending_actions("finance", store)) == 1
    assert agent_pending_actions("email", store) == []  # email owns a different source_kind


def _finance_catalog():
    """The catalog with a finance plugin loaded: its switches come from the plugin's
    manifest (ADR-0120), not from a list in the core. The manifest is inline -- the
    same declaration shape the finance plugin ships -- so this runs in any tree."""
    from iris_harness.foundation.settings.catalog import SettingDeclaration
    from iris_harness.runtime.plugin_host.manifest import PluginManifest
    from iris_harness.runtime.settings_catalog import registry_catalog

    manifest = PluginManifest(
        name="finance_workflows",
        settings={
            "IRIS_FINANCE_AUTO_INGEST": SettingDeclaration(
                kind="bool",
                default=True,
                applies="next_run",
                label="Daily finance auto-ingest",
                description="When on, the daily heartbeat ingests statements.",
                tab="agents",
                agent="finance",
            )
        },
    )
    return registry_catalog(SimpleNamespace(plugins=lambda: [SimpleNamespace(manifest=manifest)]))


def test_agent_settings_reflects_tier_toggles_and_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_FINANCE_AUTO_INGEST", "0")
    tier_router = SimpleNamespace(
        get_tier=lambda intent: SimpleNamespace(
            name="Fallback", model="granite4:latest", provider="ollama"
        )
    )
    defs = [SimpleNamespace(name="finance_ingest_tick", schedule="0 7 * * *", enabled=True)]
    s = agent_settings(
        "finance", tier_router=tier_router, heartbeat_defs=defs, catalog=_finance_catalog()
    )
    assert s is not None
    assert s["llm"][0]["intent"] == "finance" and s["llm"][0]["model"] == "granite4:latest"
    by_key = {t["key"]: t for t in s["toggles"]}
    assert by_key["IRIS_FINANCE_AUTO_INGEST"]["enabled"] is False
    assert by_key["IRIS_FINANCE_AUTO_INGEST"]["default"] is True
    assert [h["name"] for h in s["heartbeats"]] == ["finance_ingest_tick"]


def test_agent_settings_unknown_agent_is_none() -> None:
    assert (
        agent_settings(
            "nope", tier_router=SimpleNamespace(), heartbeat_defs=[], catalog=_finance_catalog()
        )
        is None
    )


def test_an_agent_without_a_plugin_loaded_has_no_toggles() -> None:
    """The core lists no plugin's switches: with no plugin catalog, finance has none."""
    from iris_harness.runtime.settings_catalog import registry_catalog

    tier_router = SimpleNamespace(
        get_tier=lambda intent: SimpleNamespace(name="T", model="m", provider="ollama")
    )
    s = agent_settings(
        "finance", tier_router=tier_router, heartbeat_defs=[], catalog=registry_catalog(None)
    )
    assert s is not None and s["toggles"] == []


def test_render_agents_overview_counts(store: TaskStore) -> None:
    agents = [{"name": "finance", "title": "Finance", "description": "money", "source_kind": "x"}]
    text = render_agents_overview(agents, {"finance": 2})
    assert "Finance (finance)" in text and "2 pending action(s)" in text


# ── Per-agent metrics rollup (ADR-0074) ────────────────────────────────────

from datetime import datetime, timezone  # noqa: E402


def _cell(intent, tier, samples, done, *, corr_n=0, tokens=None, reuse=0):
    return SimpleNamespace(
        intent=intent,
        tier=tier,
        samples=samples,
        completion_rate=done,
        correction_rate=(corr_n / samples if samples else None),
        correction_samples=corr_n,
        reuse_count=reuse,
        avg_tokens=tokens,
    )


def _report(*cells):
    return SimpleNamespace(
        matrix=tuple(cells),
        window_hours=168.0,
        sampled_at=datetime(2026, 6, 22, tzinfo=timezone.utc),
    )


def test_agent_metrics_rolls_up_over_intents() -> None:
    from iris_harness.runtime.agent_console import agent_metrics

    # finance owns intent "finance"; the email cell must be excluded.
    report = _report(
        _cell("finance", "tier1", 8, 1.0, tokens=100),
        _cell("finance", "tier2", 2, 0.5, corr_n=1, tokens=300),
        _cell("communication", "tier2", 5, 0.0),  # email — excluded from finance
    )
    m = agent_metrics("finance", report)
    assert m is not None
    assert m["volume"] == 10
    assert m["success_rate"] == (8 * 1.0 + 2 * 0.5) / 10  # 0.9
    assert m["correction_rate"] == 1 / 10
    assert m["avg_tokens"] == (8 * 100 + 2 * 300) / 10  # 140
    assert {t["tier"] for t in m["by_tier"]} == {"tier1", "tier2"}


def test_agent_metrics_no_data_is_zero_volume() -> None:
    from iris_harness.runtime.agent_console import agent_metrics

    m = agent_metrics("finance", _report(_cell("communication", "tier2", 3, 1.0)))
    assert m is not None and m["volume"] == 0 and m["success_rate"] is None


def test_agent_metrics_unknown_agent_none() -> None:
    from iris_harness.runtime.agent_console import agent_metrics

    assert agent_metrics("nope", _report()) is None


def test_render_agent_metrics_text() -> None:
    from iris_harness.runtime.agent_console import agent_metrics, render_agent_metrics

    m = agent_metrics("finance", _report(_cell("finance", "tier2", 4, 0.75, tokens=200)))
    text = render_agent_metrics(m)
    assert "success: 75%" in text and "4 turn(s)" in text
