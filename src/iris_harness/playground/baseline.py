"""Baseline snapshot + diff — the before/after regression workflow.

Snapshot a suite's outcomes before a refactor, diff after. The snapshot keeps
the behaviorally-meaningful fields (passed, intent, agent_type, handler,
sources) so a diff flags exactly the turns whose behavior moved — the safety
belt for the bootstrap decomposition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import SuiteResult

_TRACKED = ("passed", "intent", "agent_type", "handler", "sources")


def snapshot(result: SuiteResult) -> dict[str, dict[str, object]]:
    """Reduce a suite result to a comparable per-scenario snapshot."""
    out: dict[str, dict[str, object]] = {}
    for r in result.results:
        out[r.scenario_name] = {
            "passed": r.passed,
            "intent": r.intent,
            "agent_type": r.agent_type,
            "handler": r.handler,
            "sources": list(r.sources),
        }
    return out


def write_baseline(path: Path, result: SuiteResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"suite": result.suite_name, "scenarios": snapshot(result)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_baseline(path: Path) -> dict[str, dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios = raw.get("scenarios", raw)
    if not isinstance(scenarios, dict):
        raise ValueError(f"{path}: malformed baseline (no scenarios map)")
    return scenarios


@dataclass(frozen=True)
class ScenarioDelta:
    """One scenario's change between baseline and current."""

    name: str
    status: str  # "added" | "removed" | "changed"
    changes: tuple[tuple[str, object, object], ...] = ()  # (field, before, after)


def diff(
    baseline: dict[str, dict[str, object]],
    current: dict[str, dict[str, object]],
) -> list[ScenarioDelta]:
    """Return the scenarios that were added, removed, or changed behavior."""
    deltas: list[ScenarioDelta] = []
    for name in sorted(set(baseline) | set(current)):
        if name not in current:
            deltas.append(ScenarioDelta(name, "removed"))
            continue
        if name not in baseline:
            deltas.append(ScenarioDelta(name, "added"))
            continue
        before, after = baseline[name], current[name]
        changed = tuple(
            (field, before.get(field), after.get(field))
            for field in _TRACKED
            if before.get(field) != after.get(field)
        )
        if changed:
            deltas.append(ScenarioDelta(name, "changed", changed))
    return deltas
