"""The owner's audited allow-list for the external-content floor (issue #139).

The floor redacts instruction-like text in a third party's words. A text that merely quotes
an attack phrase (an article about prompt injection, the documentation of an LLM library) is
a false positive, and until now the owner could not say "this source is fine". This file is
that statement, narrowly:

- An entry names a pattern id AND a ``source``, and may narrow it to one ``tool``. The source
  is always in a namespace only the harness stamps: ``plugin:<name>``, ``skill:<name>``,
  ``mcp:<server>`` or ``core:<tool>``. A tool name alone is never a scope: the registry
  refuses a duplicate name only among plugins, not against the core's own tools, so another
  plugin's tool of the same name must not borrow the exemption; and a plugin name cannot hold
  a ``:``, so no plugin can claim another namespace's source. A wildcard anywhere, or a missing
  source, is refused when the file loads: there is no entry that means "everything".
- Only the floor's phrase patterns can be allowed. The hidden-character patterns
  (``bidi_override``, ``invisible_run``, ``tag_characters``) never are, and nothing here is
  read by secret detection, egress, the owner-PII guards or the model guard: this cannot
  weaken the secret floor, only decide that one named pattern need not redact one named
  source's text. The envelope still wraps allowed text: allowed means "not redacted", not
  "trusted".
- It is the owner's: the file is operator-edited YAML (``config/governance/external-content.yaml``)
  or written by ``iris governance allow``, every change is a ledger row, and every use is
  recorded on the floor's row (``allowed``). The SDK exposes no way to write it.
- An entry may carry ``until`` (an ISO date); once it has passed the entry is ignored.

A file that does not parse, or has a refused entry, allows nothing at all (and says why in
the log): the failure is more redaction, never less.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ALLOW_FILE = "external-content.yaml"
_WILDCARD_CHARS = frozenset("*?[]")
#: A source is a namespaced name the harness stamps; a bare name is refused.
_SOURCE_RE = re.compile(r"^(?:plugin|skill|mcp|core):[^\s*?\[\]]+$")
_SYSTEM = "system"


def scope_source(stamped: str | None, tool: str) -> str:
    """The namespaced scope the floor judges an entry against, from what the harness stamped.

    ``stamped`` is the call's ``tool_plugin``: ``skill:<name>`` and ``mcp:<server>`` already
    carry their namespace, a plugin's bare name becomes ``plugin:<name>``, and a core tool
    (stamped ``system`` or nothing) becomes ``core:<tool>``.
    """
    if not stamped or stamped == _SYSTEM:
        return f"core:{tool}"
    if stamped.startswith(("skill:", "mcp:")):
        return stamped
    return f"plugin:{stamped}"


def scope_for_label(label: str) -> str:
    """The scope of a tripwire-only caller's own label (a brief slot's ``skill:<name>``, a
    lesson): a harness literal, namespaced the same way, never a plugin's choice."""
    return label if label.startswith(("skill:", "mcp:")) else f"core:{label}"


class AllowListError(ValueError):
    """The allow-list file, or an entry for it, is refused."""


@dataclass(frozen=True)
class AllowEntry:
    """One allowed (pattern, scope) pair."""

    pattern: str
    source: str | None
    tool: str | None
    reason: str
    added: str
    until: str | None = None

    def scope(self) -> str:
        return " ".join(
            f"{k}={v}" for k, v in (("source", self.source), ("tool", self.tool)) if v is not None
        )

    def expired(self, today: date) -> bool:
        return self.until is not None and date.fromisoformat(self.until) < today

    def covers(self, source: str, tool: str | None) -> bool:
        """``source`` is a scope from :func:`scope_source` / :func:`scope_for_label`."""
        if self.source != source:
            return False
        return self.tool is None or self.tool == tool


def allowlistable_ids() -> frozenset[str]:
    """The pattern ids an entry may name: the floor's phrase patterns, never the hidden ones."""
    from iris_harness.kernel.governance.external_content import _HIDDEN_IDS, PATTERNS

    return frozenset(p.id for p in PATTERNS if p.id not in _HIDDEN_IDS)


def validate_entry(raw: Any) -> AllowEntry:
    """``raw`` as an :class:`AllowEntry`, or :class:`AllowListError` saying what is wrong."""
    if not isinstance(raw, dict):
        raise AllowListError(f"an entry must be a mapping, not {type(raw).__name__}")
    unknown = set(raw) - {"pattern", "source", "tool", "reason", "added", "until"}
    if unknown:
        raise AllowListError(f"unknown key(s) {sorted(unknown)}")
    pattern = raw.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise AllowListError("'pattern' is required: the id of one floor pattern")
    if any(c in pattern for c in _WILDCARD_CHARS):
        raise AllowListError(f"pattern {pattern!r}: wildcards are not allowed")
    if pattern not in allowlistable_ids():
        from iris_harness.kernel.governance.external_content import _HIDDEN_IDS

        why = (
            "a hidden-character pattern can never be allowed"
            if pattern in _HIDDEN_IDS
            else ("not a floor pattern id")
        )
        raise AllowListError(f"pattern {pattern!r}: {why}")
    scope: dict[str, str | None] = {}
    for key in ("source", "tool"):
        value = raw.get(key)
        if value is None:
            scope[key] = None
            continue
        if not isinstance(value, str) or not value.strip():
            raise AllowListError(f"{key!r} must be a non-empty string")
        if any(c in value for c in _WILDCARD_CHARS):
            raise AllowListError(f"{key} {value!r}: wildcards are not allowed")
        scope[key] = value.strip()
    if scope["source"] is None:
        raise AllowListError(
            "an entry needs a 'source' (plugin:<name>, skill:<name>, mcp:<server> or "
            "core:<tool>); a tool name alone is not a scope"
        )
    if not _SOURCE_RE.match(scope["source"]):
        raise AllowListError(
            f"source {scope['source']!r} must name its namespace: plugin:<name>, "
            "skill:<name>, mcp:<server> or core:<tool>"
        )
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise AllowListError("'reason' is required: why this source is fine")
    until = raw.get("until")
    until_text: str | None = None
    if until is not None:
        try:
            until_text = (
                until if isinstance(until, date) else date.fromisoformat(str(until))
            ).isoformat()
        except ValueError as exc:
            raise AllowListError(f"'until' must be an ISO date (YYYY-MM-DD): {exc}") from exc
    added = raw.get("added")
    return AllowEntry(
        pattern=pattern,
        source=scope["source"],
        tool=scope["tool"],
        reason=reason.strip(),
        added=str(added) if added is not None else datetime.now(UTC).date().isoformat(),
        until=until_text,
    )


def parse_allow(document: Any) -> tuple[AllowEntry, ...]:
    """Every entry of a parsed file, or :class:`AllowListError` naming the first refusal."""
    if document is None:
        return ()
    if not isinstance(document, dict) or set(document) - {"allow"}:
        raise AllowListError("the file holds one key, 'allow', a list of entries")
    items = document.get("allow") or []
    if not isinstance(items, list):
        raise AllowListError("'allow' must be a list")
    entries: list[AllowEntry] = []
    for index, raw in enumerate(items):
        try:
            entries.append(validate_entry(raw))
        except AllowListError as exc:
            raise AllowListError(f"allow[{index}]: {exc}") from exc
    return tuple(entries)


def allow_path() -> Path:
    from iris_harness.foundation.paths import config_path

    return config_path("governance", ALLOW_FILE)


def load_allow(path: Path | None = None) -> tuple[AllowEntry, ...]:
    """The entries in ``path`` (the default file), ``()`` when it does not exist.

    Raises :class:`AllowListError` when it does not parse or an entry is refused.
    """
    target = path or allow_path()
    if not target.exists():
        return ()
    try:
        document = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AllowListError(f"{target.name} could not be read: {exc}") from exc
    return parse_allow(document)


_lock = threading.Lock()
# path -> (mtime_ns, entries): re-read when the file changes, so an edit needs no restart.
_cache: dict[Path, tuple[int, tuple[AllowEntry, ...]]] = {}
_warned: set[tuple[Path, int]] = set()


def current_allow() -> tuple[AllowEntry, ...]:
    """The live entries: expired ones dropped. An unreadable or refused file allows nothing."""
    path = allow_path()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return ()
    with _lock:
        hit = _cache.get(path)
        if hit is None or hit[0] != mtime:
            try:
                entries = load_allow(path)
            except AllowListError as exc:
                if (path, mtime) not in _warned:
                    _warned.add((path, mtime))
                    logger.warning(
                        "external-content allow-list ignored (nothing is allowed): %s", exc
                    )
                entries = ()
            hit = _cache[path] = (mtime, entries)
    today = datetime.now(UTC).date()
    return tuple(e for e in hit[1] if not e.expired(today))


def allowed_ids(source: str, tool: str | None) -> frozenset[str]:
    """The floor pattern ids the owner allowed for text from ``source`` / ``tool``."""
    return frozenset(e.pattern for e in current_allow() if e.covers(source, tool))


def allow_status() -> tuple[int, str | None]:
    """``(live entries, problem)`` for ``/governance/state``: the problem is why the file is
    being ignored (then nothing is allowed), or None."""
    try:
        entries = load_allow()
    except AllowListError as exc:
        return 0, str(exc)
    today = datetime.now(UTC).date()
    return sum(1 for e in entries if not e.expired(today)), None


# ---------------------------------------------------------------- the owner's edits (CLI only)
_HEADER = (
    "# The owner's allow-list for the external-content floor (issue #139).\n"
    "# Written by `iris governance allow`; every change is a ledger row. An entry names ONE\n"
    "# floor pattern id and a source and/or a tool: there is no wildcard. See\n"
    "# docs/architecture/unified-governance-layer.md.\n"
)


def _entry_dict(entry: AllowEntry) -> dict[str, str]:
    out = {"pattern": entry.pattern}
    if entry.source is not None:
        out["source"] = entry.source
    if entry.tool is not None:
        out["tool"] = entry.tool
    out["reason"] = entry.reason
    out["added"] = entry.added
    if entry.until is not None:
        out["until"] = entry.until
    return out


def _write(entries: tuple[AllowEntry, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"allow": [_entry_dict(e) for e in entries]}
    path.write_text(_HEADER + yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _record(action: str, entry: AllowEntry, actor: str) -> None:
    """One ledger row for an edit: who, what, why. Never carries any text."""
    from iris_harness.foundation.paths import audit_db_path
    from iris_harness.kernel.governance.audit.log import AuditLog

    AuditLog(db_path=audit_db_path()).record(
        run_id=f"allow:{actor}",
        step_id=None,
        agent_type="owner",
        hook_point="owner_action",
        plugin="external_content_allow",
        decision=action,
        severity="warn",
        reason=f"external_content_allow: {action} {entry.pattern} for {entry.scope()}",
        payload={**_entry_dict(entry), "actor": actor},
    )


def add_entry(raw: dict[str, Any], *, actor: str, path: Path | None = None) -> AllowEntry:
    """Validate ``raw``, add it to the file and record it. Raises :class:`AllowListError`
    when it is refused (nothing is written), or when the file already holds a refused entry."""
    target = path or allow_path()
    entry = validate_entry(raw)
    existing = load_allow(target)
    if any(
        (e.pattern, e.source, e.tool) == (entry.pattern, entry.source, entry.tool) for e in existing
    ):
        raise AllowListError(f"{entry.pattern} for {entry.scope()} is already allowed")
    _write((*existing, entry), target)
    _record("add", entry, actor)
    return entry


def remove_entry(
    pattern: str, *, source: str | None, tool: str | None, actor: str, path: Path | None = None
) -> AllowEntry:
    """Remove the entry for exactly (pattern, source, tool) and record it."""
    target = path or allow_path()
    existing = load_allow(target)
    kept = tuple(e for e in existing if (e.pattern, e.source, e.tool) != (pattern, source, tool))
    if len(kept) == len(existing):
        raise AllowListError("no such entry (the pattern and scope must match exactly)")
    gone = next(e for e in existing if (e.pattern, e.source, e.tool) == (pattern, source, tool))
    _write(kept, target)
    _record("remove", gone, actor)
    return gone


__all__ = [
    "ALLOW_FILE",
    "add_entry",
    "allow_status",
    "remove_entry",
    "AllowEntry",
    "AllowListError",
    "allow_path",
    "allowed_ids",
    "allowlistable_ids",
    "scope_for_label",
    "scope_source",
    "current_allow",
    "load_allow",
    "parse_allow",
    "validate_entry",
]
