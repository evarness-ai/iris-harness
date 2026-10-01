"""Agent console composition — the harness home for the per-agent overview.

Mirrors ``iris_harness.runtime.action_center``: the logic lives here so the API, a chat ReAct tool
(``agents``), and the CLI (``iris agents``) all share it — the console is never
web-only. Most of it composes from config + the task store, so it works without a
live runtime (the API passes the live registry + run history for the richer view).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from iris_harness.agent.agent_metadata import AGENT_CATALOG, agent_meta
from iris_harness.foundation.paths import config_dir as resolve_config_dir
from iris_harness.foundation.paths import data_dir as resolve_data_dir
from iris_harness.services.tasks import TaskStore
from iris_harness.services.tasks.pending_actions import PendingAction, pending_action_from_task


def _truthy(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def config_dir() -> Path:
    return resolve_config_dir()


def data_dir() -> Path:
    return resolve_data_dir()


def list_agents(registered: set[str] | None = None) -> list[dict[str, Any]]:
    """Agents with presentation metadata. When ``registered`` (the live runtime
    registry) is given it wins; otherwise fall back to the catalog — the agents
    IRIS knows about — so the CLI/tool work without a running runtime."""
    names = sorted(registered) if registered is not None else sorted(AGENT_CATALOG)
    out: list[dict[str, Any]] = []
    for name in names:
        m = agent_meta(name)
        out.append(
            {
                "name": name,
                "title": m.title if m else name,
                "description": m.description if m else "",
                "source_kind": m.source_kind if m else None,
            }
        )
    return out


def agent_pending_actions(name: str, task_store: TaskStore) -> list[PendingAction]:
    """The agent's open pending actions (empty if it owns no source_kind)."""
    m = agent_meta(name)
    if m is None or m.source_kind is None:
        return []
    return [
        pending_action_from_task(t)
        for t in task_store.list(source_kind=m.source_kind, has_action=True, limit=200)
        if t.status in ("open", "doing")
    ]


def agent_settings(
    name: str, *, tier_router: Any, heartbeat_defs: list[Any], catalog: Any
) -> dict[str, Any] | None:
    """Read-only reflection of an agent's existing settings (ADR-0074): LLM tier(s)
    per intent, env-flag toggles (value/default/apply-timing), heartbeat cadence.
    The toggles are the catalog's on/off settings that name this agent (ADR-0120)."""
    from iris_harness.runtime.agent_settings_store import (
        APPLIES_TEXT,
        toggles_for_agent,
    )

    meta = agent_meta(name)
    if meta is None:
        return None
    llm = [
        {
            "intent": intent,
            "tier": tier_router.get_tier(intent).name,
            "model": tier_router.get_tier(intent).model,
            "provider": tier_router.get_tier(intent).provider,
        }
        for intent in meta.intents
    ]
    toggles = [
        {
            "key": entry.name,
            "label": entry.declaration.label,
            "enabled": _truthy(os.getenv(entry.name), bool(entry.declaration.default)),
            "default": bool(entry.declaration.default),
            "applies": APPLIES_TEXT[entry.declaration.applies],
            "guarded": entry.declaration.guarded,
        }
        for entry in toggles_for_agent(name, catalog).values()
    ]
    defs = {d.name: d for d in heartbeat_defs}
    heartbeats = [
        {"name": d.name, "schedule": d.schedule, "enabled": d.enabled}
        for hb in meta.heartbeats
        if (d := defs.get(hb)) is not None
    ]
    return {"llm": llm, "toggles": toggles, "heartbeats": heartbeats}


def agent_metrics(name: str, report: Any) -> dict[str, Any] | None:
    """Roll up the learning-intelligence outcome matrix (per intent x tier) to ONE
    agent, over its intents (ADR-0074 metrics). Reuses measured telemetry — no new
    instrumentation, no Phoenix. ``report`` is a build_intelligence() result."""
    meta = agent_meta(name)
    if meta is None:
        return None
    cells = [c for c in report.matrix if c.intent in meta.intents]
    base: dict[str, Any] = {
        "window_hours": report.window_hours,
        "sampled_at": report.sampled_at.isoformat(),
        "volume": 0,
        "success_rate": None,
        "correction_rate": None,
        "reuse_count": 0,
        "avg_tokens": None,
        "by_tier": [],
    }
    volume = sum(c.samples for c in cells)
    if volume == 0:
        return base

    success = sum(c.completion_rate * c.samples for c in cells) / volume
    corrections = sum(c.correction_samples for c in cells)
    tok_cells = [c for c in cells if c.avg_tokens is not None]
    tok_n = sum(c.samples for c in tok_cells)
    avg_tokens = (
        sum((c.avg_tokens or 0.0) * c.samples for c in tok_cells) / tok_n if tok_n else None
    )

    by_tier: dict[str, list[Any]] = {}
    for c in cells:
        by_tier.setdefault(c.tier, []).append(c)
    tiers = [
        {
            "tier": tier,
            "volume": sum(c.samples for c in tcells),
            "success_rate": (
                sum(c.completion_rate * c.samples for c in tcells) / sum(c.samples for c in tcells)
            ),
        }
        for tier, tcells in sorted(by_tier.items())
    ]

    base.update(
        volume=volume,
        success_rate=success,
        correction_rate=corrections / volume,
        reuse_count=sum(c.reuse_count for c in cells),
        avg_tokens=avg_tokens,
        by_tier=tiers,
    )
    return base


def render_agent_metrics(m: dict[str, Any]) -> str:
    """Plain-text metrics block for chat / CLI."""
    if not m or m.get("volume", 0) == 0:
        return f"No measured turns in the last {m.get('window_hours', 0):.0f}h."
    lines = [f"Metrics (last {m['window_hours']:.0f}h, {m['volume']} turn(s)):"]
    lines.append(f"- success: {m['success_rate']:.0%}")
    if m.get("correction_rate") is not None:
        lines.append(f"- user-correction: {m['correction_rate']:.0%}")
    if m.get("avg_tokens") is not None:
        lines.append(f"- avg tokens/turn: {m['avg_tokens']:.0f}")
    if m.get("reuse_count"):
        lines.append(f"- downstream reuse: {m['reuse_count']}")
    for t in m.get("by_tier", []):
        lines.append(f"  · {t['tier']}: {t['success_rate']:.0%} done (n={t['volume']})")
    return "\n".join(lines)


# ── Local (no-runtime) builders for the CLI / chat tool ────────────────────────


def _load_tier_router() -> Any:
    from iris_harness.llm.tier_router import TierRouter

    return TierRouter.load_from_yaml(config_dir() / "llm_tiers.yaml")


def _load_heartbeat_defs() -> list[Any]:
    from iris_harness.services.heartbeat.config import load_heartbeats

    path = config_dir() / "heartbeats.yaml"
    return load_heartbeats(path) if path.exists() else []


def _local_agent_metrics(name: str, *, window_days: int = 7) -> dict[str, Any] | None:
    """Build the agent's metrics from the local learning.db (best-effort; None if
    unavailable). Lets the CLI / chat detail show metrics without a live runtime."""
    from datetime import timedelta

    db = data_dir() / "learning.db"
    if not db.exists():
        return None
    try:
        from iris_harness.services.learning.intelligence import build_intelligence
        from iris_harness.services.learning.store import LearningMetricsStore

        store = LearningMetricsStore(db_path=db)
        report = build_intelligence(store, window=timedelta(days=window_days))
        return agent_metrics(name, report)
    except Exception:  # noqa: BLE001 — metrics are advisory; never break the detail view
        return None


# ── Renderers (channel-agnostic text for chat / CLI) ──────────────────────────


def render_agents_overview(agents: list[dict[str, Any]], counts: dict[str, int]) -> str:
    if not agents:
        return "No agents registered."
    lines = [f"{len(agents)} agent(s):"]
    for a in agents:
        n = counts.get(a["name"], 0)
        suffix = f" — {n} pending action(s)" if n else ""
        lines.append(f"- {a['title']} ({a['name']}): {a['description']}{suffix}")
    return "\n".join(lines)


def render_agent_detail(name: str) -> str:
    """Full text view for one agent (settings + pending actions), composed locally."""
    from iris_harness.runtime.action_center import render_pending_actions

    meta = agent_meta(name)
    if meta is None:
        return f"No such agent: {name}"

    ts = TaskStore(db_path=data_dir() / "tasks.db")
    ts.ensure_schema()
    pending = agent_pending_actions(name, ts)
    from iris_harness.runtime.settings_catalog import installed_catalog

    settings = agent_settings(
        name,
        tier_router=_load_tier_router(),
        heartbeat_defs=_load_heartbeat_defs(),
        catalog=installed_catalog(config_dir()),
    )

    lines = [f"{meta.title} ({name}) — {meta.description}", ""]
    lines.append(render_pending_actions(pending))
    if settings:
        if settings["llm"]:
            lines.append("")
            lines.append("LLM tier:")
            for x in settings["llm"]:
                lines.append(f"- {x['intent']}: {x['tier']} ({x['model']})")
        if settings["toggles"]:
            lines.append("")
            lines.append("Toggles:")
            for t in settings["toggles"]:
                state = "on" if t["enabled"] else "off"
                dflt = "on" if t["default"] else "off"
                lines.append(
                    f"- {t['label']} [{t['key']}]: {state} "
                    f"(default {dflt}, applies {t['applies']})"
                )
        if settings["heartbeats"]:
            lines.append("")
            lines.append("Schedule:")
            for h in settings["heartbeats"]:
                lines.append(f"- {h['name']}: {h['schedule']}")
    metrics = _local_agent_metrics(name)
    if metrics is not None:
        lines.append("")
        lines.append(render_agent_metrics(metrics))
    return "\n".join(lines)


def render_agents_overview_local() -> str:
    """The agents overview composed locally (catalog + task store) for chat/CLI."""
    ts = TaskStore(db_path=data_dir() / "tasks.db")
    ts.ensure_schema()
    agents = list_agents()
    counts = {a["name"]: len(agent_pending_actions(a["name"], ts)) for a in agents}
    return render_agents_overview(agents, counts)
