"""Graphiti export → memris source records (memris plan PR 9).

The reader knows Graphiti's data model; what each record becomes is ``mappings.yaml``'s
business, and the writing is ``memris.interchange.apply_mappings``.

**ASSUMPTIONS about the export — verify against a real Graphiti dump.** Graphiti has no
single export file; this reads a JSON object with four lists, named after its models:

* ``nodes``      — EntityNode:   ``uuid``, ``name``, ``labels``, ``summary``, ``created_at``, ``group_id``
* ``edges``      — EntityEdge:   ``uuid``, ``source_node_uuid``, ``target_node_uuid``, ``name``,
  ``fact``, ``episodes``, ``created_at``, ``expired_at``, ``valid_at``, ``invalid_at``
* ``episodes``   — EpisodicNode: ``uuid``, ``name``, ``content``, ``source``, ``valid_at``, ``created_at``
* ``episodic_edges`` — MENTIONS: ``uuid``, ``source_node_uuid`` (episode), ``target_node_uuid``
  (entity), ``created_at``

Times are ISO-8601. How Graphiti's time model lands on memris's two axes:

* ``created_at`` → ``recorded_at`` (when the other system learned it — record time)
* ``valid_at``   → ``valid_from``, ``invalid_at`` → ``valid_to`` (valid time, same meaning)
* ``expired_at`` is Graphiti's record-time END: the moment the edge stopped being believed,
  usually because a newer edge contradicted it. memris keeps a record-time end only for a
  retraction ("never true"), which would be wrong for a fact that held and then changed. So
  with ``invalid_at`` present, ``expired_at`` is noted and not stored; without it, it is the
  best available end of validity and becomes ``valid_to`` — noted as an approximation.
* ``episodes`` → ``source_episode`` (the first; more than one is noted) and ``fact`` →
  ``evidence``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from memris.interchange import EntityRef, SourceRecord

HERE = Path(__file__).resolve().parent
EDGE = "graphiti_edge"
MENTION = "graphiti_mention"
EXTRACTOR = "graphiti"


@dataclass
class ReadResult:
    records: list[SourceRecord] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # what the reader itself refused


def load_config(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or HERE / "graphiti.yaml").read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


def _time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _class_of(labels: Any, classes: dict[str, str]) -> str | None:
    """The first label the config maps, preferring a specific type over plain ``Entity``."""
    names = [str(label) for label in (labels or [])]
    specific = [classes[n] for n in names if n in classes and n != "Entity"]
    if specific:
        return specific[0]
    return classes.get("Entity") if "Entity" in names else None


def read_export(document: dict[str, Any], config: dict[str, Any] | None = None) -> ReadResult:
    config = config if config is not None else load_config()
    classes = {str(k): str(v) for k, v in (config.get("classes") or {}).items()}
    result = ReadResult()

    nodes: dict[str, EntityRef] = {}
    for node in document.get("nodes") or []:
        uuid = str(node.get("uuid") or "")
        if not uuid or not node.get("name"):
            result.skipped.append((uuid or "?", "entity node without a uuid or a name"))
            continue
        nodes[uuid] = EntityRef(str(node["name"]), _class_of(node.get("labels"), classes), uuid)
    used: set[str] = set()

    for edge in document.get("edges") or []:
        uuid = str(edge.get("uuid") or "?")
        source = nodes.get(str(edge.get("source_node_uuid")))
        target = nodes.get(str(edge.get("target_node_uuid")))
        if source is None or target is None:
            result.skipped.append((uuid, "edge endpoint is not a node in the export"))
            continue
        try:
            valid_to = _time(edge.get("invalid_at"))
            expired = _time(edge.get("expired_at"))
            times = (_time(edge.get("created_at")), _time(edge.get("valid_at")))
        except ValueError as exc:
            result.skipped.append((uuid, f"unreadable timestamp: {exc}"))
            continue
        notes: list[str] = []
        if expired is not None and valid_to is None:
            valid_to = expired
            notes.append(f"expired_at {expired.isoformat()} used as valid_to (no invalid_at)")
        elif expired is not None:
            notes.append(
                f"expired_at {expired.isoformat()} is a record-time end memris does not keep"
            )
        episodes = [str(e) for e in (edge.get("episodes") or [])]
        if len(episodes) > 1:
            notes.append(f"{len(episodes)} episodes; the first is kept as source_episode")
        used.update({source.source_id or "", target.source_id or ""})
        result.records.append(
            SourceRecord(
                source_type=EDGE,
                key=str(edge.get("name") or "") or None,
                source_id=uuid,
                subject=source,
                object=target,
                recorded_at=times[0],
                valid_from=times[1],
                valid_to=valid_to,
                episode=episodes[0] if episodes else None,
                evidence=(str(edge["fact"])[:500] if edge.get("fact") else None),
                extractor=EXTRACTOR,
                statement_id=f"st_graphiti_{uuid}",
                notes=tuple(notes),
            )
        )

    episode_class = config.get("episode_class")
    episodes_by_id = {str(e.get("uuid")): e for e in document.get("episodes") or []}
    for mention in document.get("episodic_edges") or []:
        uuid = str(mention.get("uuid") or "?")
        episode = episodes_by_id.get(str(mention.get("source_node_uuid")))
        target = nodes.get(str(mention.get("target_node_uuid")))
        if episode is None or target is None:
            result.skipped.append((uuid, "mention endpoint is not in the export"))
            continue
        try:
            recorded = _time(mention.get("created_at")) or _time(episode.get("created_at"))
        except ValueError as exc:
            result.skipped.append((uuid, f"unreadable timestamp: {exc}"))
            continue
        used.add(target.source_id or "")
        episode_name = str(episode.get("name") or episode.get("uuid"))
        result.records.append(
            SourceRecord(
                source_type=MENTION,
                key=None,
                source_id=uuid,
                subject=EntityRef(episode_name, str(episode_class) if episode_class else None),
                object=target,
                recorded_at=recorded,
                episode=str(episode.get("uuid")),
                extractor=EXTRACTOR,
                statement_id=f"st_graphiti_{uuid}",
            )
        )

    for uuid, ref in nodes.items():
        if uuid not in used:
            # memris keeps entities through what is said about them; a lone node has nothing
            result.skipped.append((uuid, f"node {ref.name!r} takes part in no edge"))
    return result


__all__ = ["EDGE", "EXTRACTOR", "MENTION", "ReadResult", "load_config", "read_export"]
