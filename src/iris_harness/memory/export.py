"""Export the memory graph as a markdown vault — one way, on request, never read back.

The memory graph (``memory/graph.py``) is a view over the stores. This writes a
*snapshot* of that view as one markdown note per node, linked with ``[[wikilinks]]``,
which is what Obsidian reads natively and what a notebook tool will take as sources.

Four rules, each load-bearing:

- **One way.** ``sync_from`` is not implemented and is not planned. A vault edit that
  flowed back would undo the confirmation gate — the whole point of which is that
  nothing becomes a belief without the owner saying so — and two-way sync is exactly
  how the wiki ended up with 2,858 vectors against 2,505 pages.
- **Confirmed by default.** An unreviewed fact in a vault looks as authoritative as any
  other note. ``include_unconfirmed=True`` exports them, clearly marked.
- **A snapshot, not a store.** The agent never reads this back. Re-export to refresh;
  the export directory can be deleted at any time with nothing lost.
- **Egress is the caller's decision.** These notes carry the owner's bank, employer,
  family and email. A local vault stays on the machine; uploading the folder to a cloud
  notebook does not. The CLI says so at the point of export rather than in a doc.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_harness.foundation.paths import iris_home

logger = logging.getLogger(__name__)

# Folder per kind. Obsidian does not care about the layout — this is for humans
# opening the folder in Finder.
_DIRS = {
    "entity": "entities",
    "fact": "facts",
    "session": "sessions",
    "lesson": "lessons",
    "pattern": "patterns",
}
_EXPORT_CAP = 5_000  # a vault, not a prompt: effectively "everything"


@dataclass
class ExportResult:
    out_dir: Path
    notes: int = 0
    links: int = 0
    included_unconfirmed: bool = False
    skipped_unconfirmed: int = 0
    removed_stale: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "notes": self.notes,
            "links": self.links,
            "included_unconfirmed": self.included_unconfirmed,
            "skipped_unconfirmed": self.skipped_unconfirmed,
            "removed_stale": self.removed_stale,
            "by_kind": dict(self.by_kind),
        }


def _note_name(label: str) -> str:
    """A filename-safe, link-safe note name. Obsidian links by name, not path."""
    cleaned = re.sub(r"[\\/:*?\"<>|#\[\]^]", " ", label or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return (cleaned[:80] or "untitled").strip()


def _frontmatter(fields: dict[str, Any]) -> str:
    import yaml

    return "---\n" + yaml.dump(fields, sort_keys=True, allow_unicode=True) + "---\n\n"


def export_memory_vault(
    store: Any,
    out_dir: Path,
    *,
    include_unconfirmed: bool = False,
) -> ExportResult:
    """Write the graph as a linked markdown vault. Returns what was written."""
    from iris_harness.memory.graph import build_memory_graph

    graph = build_memory_graph(store, confirmed_only=not include_unconfirmed, node_cap=_EXPORT_CAP)
    nodes = {n["id"]: n for n in graph["nodes"] if n["kind"] != "more"}
    edges = [e for e in graph["edges"] if e["source"] in nodes and e["target"] in nodes]

    result = ExportResult(out_dir=out_dir, included_unconfirmed=include_unconfirmed)
    if not include_unconfirmed:
        full = build_memory_graph(store, node_cap=_EXPORT_CAP)
        result.skipped_unconfirmed = sum(1 for n in full["nodes"] if not n["confirmed"])

    names = {nid: _note_name(node["label"]) for nid, node in nodes.items()}
    neighbours: dict[str, list[tuple[str, str]]] = {nid: [] for nid in nodes}
    for edge in edges:
        neighbours[edge["source"]].append((edge["target"], edge["label"]))
        neighbours[edge["target"]].append((edge["source"], f"{edge['label']} (from)"))

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).date().isoformat()
    written: set[Path] = set()

    for nid, node in nodes.items():
        kind = str(node["kind"])
        folder = out_dir / _DIRS.get(kind, "") if kind in _DIRS else out_dir
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{names[nid]}.md"

        meta = {
            "title": node["label"],
            "kind": kind,
            "confirmed": bool(node["confirmed"]),
            "exported": stamp,
            "source": "IRIS memory (snapshot — edits here do not flow back)",
            "tags": [f"iris/{kind}"],
        }
        for key, value in (node.get("meta") or {}).items():
            if isinstance(value, (str, int, float, bool)):
                meta[str(key)] = value

        lines = [_frontmatter(meta), f"# {node['label']}\n"]
        if not node["confirmed"]:
            lines.append(
                "> [!warning] Not confirmed\n"
                "> IRIS inferred this and the owner has not reviewed it. "
                "It is not used in prompts.\n"
            )
        summary = str((node.get("meta") or {}).get("summary") or "")
        if summary:
            lines.append(f"{summary}\n")
        links = neighbours.get(nid, [])
        if links:
            lines.append("## Links\n")
            seen: set[str] = set()
            for target, label in links:
                key = f"{target}:{label}"
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {label} → [[{names[target]}]]")
                result.links += 1
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        written.add(path.resolve())
        result.notes += 1
        result.by_kind[kind] = result.by_kind.get(kind, 0) + 1

    _write_index(out_dir, nodes, names, result, stamp)
    result.removed_stale = _prune_stale(out_dir, written)
    logger.info("memory vault exported: %s", result.as_dict())
    return result


_OWNED_MARKER = "source: IRIS memory (snapshot"


def _prune_stale(out_dir: Path, written: set[Path]) -> int:
    """Remove notes a previous export wrote that this one did not.

    A snapshot that keeps a note for a fact you forgot is worse than no snapshot. Only
    files carrying the export's own ``source:`` line are touched — notes the owner wrote
    in the same vault are never deleted.
    """
    removed = 0
    for path in out_dir.rglob("*.md"):
        resolved = path.resolve()
        if resolved in written or path.name == "index.md":
            continue
        try:
            head = path.read_text(encoding="utf-8")[:400]
        except OSError:
            continue
        if _OWNED_MARKER not in head:
            continue  # someone else's note; leave it alone
        try:
            path.unlink()
            removed += 1
        except OSError:
            logger.debug("could not remove stale note %s", path, exc_info=True)
    return removed


def _write_index(
    out_dir: Path,
    nodes: dict[str, dict[str, Any]],
    names: dict[str, str],
    result: ExportResult,
    stamp: str,
) -> None:
    lines = [
        _frontmatter(
            {
                "title": "IRIS memory",
                "exported": stamp,
                "source": "IRIS memory (snapshot — edits here do not flow back)",
                "tags": ["iris/index"],
            }
        ),
        "# IRIS memory\n",
        f"A snapshot of what IRIS remembers, exported {stamp}. " f"{result.notes} note(s).\n",
        "Edits made here are **not** read back — IRIS only believes what you confirm "
        "inside it (`iris facts review`). Re-export to refresh.\n",
    ]
    if result.skipped_unconfirmed:
        lines.append(
            f"{result.skipped_unconfirmed} unconfirmed item(s) were left out. "
            "Export with `--include-unconfirmed` to see what is waiting for review.\n"
        )
    for kind in ("entity", "fact", "session", "lesson", "pattern", "you"):
        of_kind = [n for n in nodes.values() if n["kind"] == kind]
        if not of_kind:
            continue
        lines.append(f"## {kind.title()} ({len(of_kind)})\n")
        for node in sorted(of_kind, key=lambda n: str(n["label"]).lower()):
            mark = "" if node["confirmed"] else "  _(not confirmed)_"
            lines.append(f"- [[{names[node['id']]}]]{mark}")
        lines.append("")
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


_NAME_MAX = 64
_EXPORT_DIR_ENV = "IRIS_EXPORT_DIR"


class ExportTargetError(ValueError):
    """A requested export folder name that cannot be used."""


def export_root() -> Path:
    """The one folder the web API exports under: ``$IRIS_EXPORT_DIR`` else ``<IRIS_HOME>/exports``."""
    configured = os.environ.get(_EXPORT_DIR_ENV, "").strip()
    return Path(configured).expanduser() if configured else iris_home() / "exports"


def resolve_export_dir(name: str) -> Path:
    """The folder ``name`` stands for under :func:`export_root`, created owner-only if the root
    is new. ``name`` comes from a request, so it is one plain path component: no separator, no
    ``..``, not absolute, at most 64 characters. The result, with symlinks resolved, must sit
    directly inside the resolved root, and must not be an existing non-directory."""
    if not name or len(name) > _NAME_MAX:
        raise ExportTargetError(f"export folder name must be 1-{_NAME_MAX} characters")
    if name in {".", ".."} or any(c in name for c in ("/", "\\", "\x00")) or os.path.isabs(name):
        raise ExportTargetError("export folder name must be a single plain folder name")
    root = export_root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    real_root = os.path.realpath(root)
    target = os.path.realpath(os.path.join(real_root, name))
    if not target.startswith(real_root + os.sep) or os.path.dirname(target) != real_root:
        raise ExportTargetError("export folder resolves outside the export folder")
    if os.path.exists(target) and not os.path.isdir(target):
        raise ExportTargetError("export folder name is an existing file")
    return Path(target)


__all__ = [
    "ExportResult",
    "ExportTargetError",
    "export_memory_vault",
    "export_root",
    "resolve_export_dir",
]
