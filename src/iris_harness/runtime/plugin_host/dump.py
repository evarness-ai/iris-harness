"""``iris --dump-config``: the effective profile and plugin tree, without booting.

Discovery only — manifests are read, ``setup`` is not called — so the command
is fast and safe to run anywhere. The post-boot view (what each plugin
actually registered, its subscriptions and seams, its YAML, and its health) is
``inventory.py`` behind ``GET /plugins`` and ``iris plugins``, plus ``GET /health`` and the playground drift panel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .loader import describe_sources
from .profile import list_profiles, load_profile


def dump_config(config_dir: Path, *, profile_name: str | None = None) -> dict[str, Any]:
    profile = load_profile(config_dir, profile_name)
    plugins = describe_sources(profile)
    return {
        "profile": profile.as_dict(),
        "available_profiles": list_profiles(config_dir),
        "plugins": plugins,
        "capabilities": capability_graph(plugins),
    }


def capability_graph(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """``{capability: {provides|uses|requires: [plugin, ...]}}`` over the enabled plugins.

    Who provides and who uses what (plugin-capabilities §2), from the manifests alone: a
    capability someone uses with nobody providing it reads ``provided by -`` here, before
    anything boots.
    """
    graph: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        if not row.get("enabled") or not row.get("source"):
            continue
        for role, names in (row.get("capabilities") or {}).items():
            for name in names:
                roles = graph.setdefault(name, {"provides": [], "uses": [], "requires": []})
                roles[role].append(row["name"])
    return dict(sorted(graph.items()))


def render_text(tree: dict[str, Any]) -> str:
    profile = tree["profile"]
    lines = [f"profile: {profile['name']}"]
    if profile.get("description"):
        lines.append(f"  {profile['description']}")
    lines.append("layers (in apply order):")
    for layer in profile["layers"]:
        lines.append(f"  - {layer}")
    lines.append("plugins:")
    if not tree["plugins"]:
        lines.append("  (none)")
    for row in tree["plugins"]:
        flag = "on " if row["enabled"] else "off"
        if row.get("source"):
            provides = ", ".join(row.get("provides") or []) or "-"
            lines.append(
                f"  [{flag}] {row['name']:<20} {row['source']:<40} v{row.get('version', '?')}"
                f"  trust={row.get('trust')}  party={row.get('party')}  provides={provides}  (set by {row['set_by']})"
            )
            if row.get("identity"):
                lines.append(f"        owner identity: {', '.join(row['identity'])}")
            egress = row.get("egress") or {}
            if egress.get("open_web"):
                lines.append("        egress: any host (open_web)")
            elif egress.get("hosts"):
                hosts = ", ".join(f"{h['host']} ({h['data']})" for h in egress["hosts"])
                lines.append(f"        egress: {hosts}")
            else:
                lines.append("        egress: none declared (egress not enforced yet)")
            if row.get("search_providers"):
                lines.append(f"        search providers: {', '.join(row['search_providers'])}")
        else:
            lines.append(f"  [{flag}] {row['name']:<20} NOT FOUND: {row.get('error')}")
    graph = capability_graph(tree["plugins"])
    if graph:
        lines.append("capabilities (declared by enabled plugins):")
        for name, roles in graph.items():
            parts = [
                f"{label} {', '.join(roles[role]) or '-'}"
                for role, label in (
                    ("provides", "provided by"),
                    ("uses", "used by"),
                    ("requires", "required by"),
                )
            ]
            lines.append(f"  {name:<24} " + "  ".join(parts))
    if profile["intercept_order"]:
        lines.append("intercept order override: " + ", ".join(profile["intercept_order"]))
    others = [p for p in tree["available_profiles"] if p != profile["name"]]
    if others:
        lines.append("other shipped profiles: " + ", ".join(others))
    return "\n".join(lines)


def render_json(tree: dict[str, Any]) -> str:
    return json.dumps(tree, indent=2, sort_keys=True)


__all__ = ["capability_graph", "dump_config", "render_json", "render_text"]
