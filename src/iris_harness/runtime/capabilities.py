"""What this install actually has — read off the live registry, not prose.

SOUL.md carried a hand-written "Operational primer" that named `llama3.2:3b` and
`qwen2.5-coder:7b` as tier 1 and tier 2 while the configured models were `granite4`
and `qwen2.5:7b-instruct`, and it rode in every prompt. A description of the system
that the system does not generate goes stale and then lies.

Two shapes, same source:

- :func:`capability_line` — one line for the prompt (L1): what exists, and how to
  ask for detail.
- :func:`capability_report` — the detail, served by ``iris_doc("CAPABILITIES")``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_TIER_INTENTS = ("general", "communication", "code_exec")


def _plugin_names(runtime: Any) -> list[str]:
    registry = getattr(runtime, "plugin_registry", None)
    if registry is None:
        return []
    names: list[str] = []
    for record in registry.plugins():
        status = getattr(record.status, "value", record.status)
        if str(status).lower() in {"failed", "not_found", "missing"}:
            continue
        names.append(record.name)
    return sorted(names)


def _tier_models(runtime: Any) -> list[tuple[str, str]]:
    router = getattr(runtime, "tier_router", None)
    if router is None:
        return []
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for intent in _TIER_INTENTS:
        try:
            tier = router.get_tier(intent)
        except Exception:  # a partial router must not break a turn
            logger.debug("tier lookup failed for intent %s", intent, exc_info=True)
            continue
        if tier is None or tier.name in seen:
            continue
        seen.add(tier.name)
        out.append((tier.name, tier.model))
    return out


def capability_line(runtime: Any, tool_names: list[str] | None = None) -> str | None:
    """One line naming what is loaded, plus where to get the detail."""
    try:
        plugins = _plugin_names(runtime)
        tiers = _tier_models(runtime)
    except Exception:
        logger.debug("capability line failed", exc_info=True)
        return None
    if not plugins and not tiers:
        return None
    bits: list[str] = []
    if plugins:
        shown = plugins[:8]
        more = f" (+{len(plugins) - len(shown)} more)" if len(plugins) > len(shown) else ""
        bits.append("plugins loaded: " + ", ".join(shown) + more)
    if tiers:
        bits.append("models: " + ", ".join(f"{name} {model}" for name, model in tiers))
    if tool_names:
        bits.append(f"{len(tool_names)} tools on this turn's menu")
    return (
        "Your current setup — "
        + "; ".join(bits)
        + '. For the full list call iris_doc("CAPABILITIES"); '
        + 'for your operating detail call iris_doc("OPERATING").'
    )


def capability_report(runtime: Any, tool_names: list[str] | None = None) -> str:
    """The long form behind ``iris_doc("CAPABILITIES")``."""
    lines = ["# Capabilities (live, read from the running registry)", ""]

    tiers = _tier_models(runtime)
    if tiers:
        lines.append("## Models")
        lines.extend(f"- {name}: {model}" for name, model in tiers)
        lines.append("")

    registry = getattr(runtime, "plugin_registry", None)
    if registry is not None:
        lines.append("## Plugins")
        for record in sorted(registry.plugins(), key=lambda r: r.name):
            status = getattr(record.status, "value", record.status)
            kinds = sorted({str(getattr(r, "kind", "")) for r in record.registrations if r})
            detail = f" — registers: {', '.join(k for k in kinds if k)}" if kinds else ""
            lines.append(f"- {record.name} ({status}){detail}")
        lines.append("")
        tools = registry.tools()
        if tools:
            lines.append("## Plugin tools")
            lines.extend(f"- {t.name}" for t in sorted(tools, key=lambda t: t.name))
            lines.append("")

    if tool_names:
        lines.append("## Tools offered on the last turn")
        lines.extend(f"- {name}" for name in tool_names)
        lines.append("")
        lines.append(
            "Tools are shortlisted per turn, so this menu changes; a tool missing here "
            "still exists and can be called by name."
        )

    return "\n".join(lines).strip()
