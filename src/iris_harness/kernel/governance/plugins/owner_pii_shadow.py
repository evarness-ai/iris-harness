"""Owner-PII shadow mode (ADR-0125 PR 4, amendment 7): what each guard WOULD do.

``IRIS_GOVERNANCE_OWNER_PII=shadow`` registers one hook, :class:`OwnerPiiShadowHook`, at
the four points where the owner-PII guards act. For every call it runs
``owner_pii.decide`` over the text each guard reads and records, on its one audit row,
what the ``identity.yaml`` table says that guard would do with every occurrence. It never
changes anything: it returns ``allow`` with no transform, no label and no tier, and it
runs FIRST at each point (priority 1), so the decision and context the kernel hands back
are the ones the real guards produced, byte for byte. Two callers read that final
decision's metadata (the curator reads ``dump_flag``; the tool runner the rewritten
``args``), which is why the hook cannot run last.

Where each column is checked:

- ``egress`` -- ``PRE_TOOL_USE``, a network tool as ``network_egress`` defines one (the
  same pattern set, so the calls audited are the calls that guard inspects), over the
  tool's arguments; ``log_only_destinations`` relax a deny to ``log`` at ``github_*`` and
  ``mcp_*``, and the observation says so.
  A tool declaring ``sends_to: external_service`` is checked here too, by declaration.
- ``web_search`` -- ``PRE_TOOL_USE``, a tool that declares ``sends_to: search_engine`` in
  its manifest (the runner stamps it, ``tool_payload.TOOL_SENDS_TO``). By declaration,
  never by a list of tool names: the research plugin may import only the SDK, so its own
  guard cannot write a kernel row, and the kernel learns what research is from what
  research says about itself.
- ``egress`` also -- ``PRE_EGRESS``, over the strings a plugin's governed HTTP request
  carries (address and parameters), the destination being the host (issue #103).
- ``tier3`` -- ``PRE_LLM_CALL`` when the target tier the kernel was given is ``tier_3``
  (the egress gate's reading of "leaves the machine"), over the prompt. A local tier is
  never observed. How that tier is derived is the caller's; this hook only reads it.
- ``answer_owner`` / ``answer_other`` -- ``PRE_RESPONSE``, over the answer, by its
  ``audience`` (``hooks/response_payload.audience_of``).

The row (``audit_metadata["owner_pii_shadow"]``) carries the columns checked and, per
guard x kind x action, the count and span offsets (``[leaf, start, end]``; ``leaf`` is the
ordinal of the string in a deterministic walk of the arguments). Never a literal, never
text around it. When the audit key is already resolved it adds keyed digests of the
distinct literals (``audit/digest.py``) so the read-back can tell one literal seen a
hundred times from a hundred literals. It never resolves the key itself (a Keychain
prompt must not come from an observer), and without one it records kinds, counts and
spans only: a shadow observation never refuses or breaks a call.

:func:`owner_pii_shadow_summary` is the read-back: counts by hook point x guard x kind x
action over a window. ``iris governance pii-shadow`` and ``GET /governance/pii-shadow``
render it.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal

from iris_harness.foundation.paths import audit_db_path
from iris_harness.kernel.governance.audit import AuditLog, AuditRow
from iris_harness.kernel.governance.audit.digest import (
    AuditDigester,
    audit_digester,
    audit_key_status,
)
from iris_harness.kernel.governance.hooks.response_payload import RESPONSE, audience_of
from iris_harness.kernel.governance.hooks.tool_payload import (
    ARGS,
    sends_to_external_service,
    sends_to_search_engine,
    tool_name_of,
)
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.identity_config import GuardTable, guard_table
from iris_harness.kernel.governance.identity_redaction import owner_identity
from iris_harness.kernel.governance.owner_identity import OwnerIdentity
from iris_harness.kernel.governance.owner_matchers import Match, canonical, is_first_name_alone
from iris_harness.kernel.governance.owner_pii import Guard, column_for, decide
from iris_harness.kernel.governance.plugins.network_egress import (
    DEFAULT_NETWORK_TOOLS,
    is_network_tool,
)

logger = logging.getLogger(__name__)

# -- the flag --------------------------------------------------------------------------

ENV_FLAG: Final = "IRIS_GOVERNANCE_OWNER_PII"
OwnerPiiMode = Literal["off", "shadow"]
#: The values the flag accepts as written. ``enforce`` arrives with PR 5.
ACCEPTED_MODES: tuple[OwnerPiiMode, ...] = ("off", "shadow")
_OFF_VALUES: Final = frozenset({"", "off", "0", "false", "no"})


@dataclass(frozen=True)
class ModeSetting:
    """The flag as read: the mode it runs, what was written, and why they differ."""

    mode: OwnerPiiMode
    raw: str
    problem: str | None = None


def parse_owner_pii_mode(raw: str | None) -> ModeSetting:
    """``off`` (and the usual falsy spellings, and unset) or ``shadow``.

    Any other value -- ``enforce`` included -- is an operator asking for MORE than off,
    so it runs ``shadow``, the most this build has, with a ``problem`` saying so. Not
    ``off``: that would quietly give less than was asked. Not enforce: nothing here can
    enforce. Shadow changes no decision, so a mistyped value cannot break a call.
    """
    value = (raw or "").strip().lower()
    if value in _OFF_VALUES:
        return ModeSetting("off", value)
    if value == "shadow":
        return ModeSetting("shadow", value)
    accepted = ", ".join(ACCEPTED_MODES)
    if value == "enforce":
        why = "enforce is not built yet (ADR-0125 PR 5)"
    else:
        why = f"{value!r} is not a mode"
    return ModeSetting(
        "shadow",
        value,
        f"{ENV_FLAG}={value!r}: {why}; accepted values are {accepted}. Running shadow: "
        "owner PII is audited and NOTHING is masked or denied.",
    )


def owner_pii_mode_from_env() -> ModeSetting:
    """The flag from the environment (not logged: the kernel build logs it once)."""
    return parse_owner_pii_mode(os.getenv(ENV_FLAG))


# -- the hook --------------------------------------------------------------------------

HOOK_NAME: Final = "owner_pii_shadow"
#: The audit-metadata key the observation lives under.
SHADOW_KEY: Final = "owner_pii_shadow"
#: The hook points it is registered at.
SHADOW_POINTS: tuple[HookPoint, ...] = (
    HookPoint.PRE_TOOL_USE,
    HookPoint.PRE_LLM_CALL,
    HookPoint.PRE_RESPONSE,
    # What a plugin's governed HTTP request carries out (issue #103).
    HookPoint.PRE_EGRESS,
)
#: Span offsets kept per observation (the count is always complete).
MAX_SPANS: Final = 20
#: The tier whose target leaves the machine (``egress_gate``: max tier for personal data
#: is ``tier_2``; ``tier_3`` is the cloud).
CLOUD_TIER: Final = "tier_3"


def _leaves(value: Any) -> Iterator[str]:
    """The strings in a tool's arguments, in a deterministic order (insertion order).

    Values only: argument names are the tool's schema, not text that leaves the machine.
    Integers count (a phone number passed as a number); booleans do not.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, bool):
        return
    elif isinstance(value, int):
        yield str(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _leaves(item)


def _digester_if_ready() -> AuditDigester | None:
    """The audit digester when a governed call already resolved the key; never resolves it."""
    try:
        state, _ = audit_key_status()
        return audit_digester() if state == "ready" else None
    except Exception:  # noqa: BLE001 - an observer never fails on the key
        return None


@dataclass
class _Group:
    guard: str
    kind: str
    action: str
    first_name_alone: bool
    log_only_destination: bool
    count: int = 0
    spans: list[list[int]] = field(default_factory=list)
    digests: set[str] = field(default_factory=set)

    def row(self, *, with_digests: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "guard": self.guard,
            "kind": self.kind,
            "action": self.action,
            "count": self.count,
            "spans": self.spans,
        }
        if self.first_name_alone:
            out["first_name_alone"] = True
        if self.log_only_destination:
            out["log_only_destination"] = True
        if with_digests:
            out["digests"] = sorted(self.digests)
        return out


def observe_texts(
    texts: list[str],
    *,
    guard: Guard,
    identity: OwnerIdentity,
    table: GuardTable,
    audience: Literal["owner", "other"] = "owner",
    destination: str | None = None,
    digester: AuditDigester | None = None,
) -> list[dict[str, Any]]:
    """What ``guard`` would do with every owner literal in ``texts``, grouped, literal-free.

    One entry per guard column x kind x action (x first-name-alone x log-only), with the
    full count, up to :data:`MAX_SPANS` ``[leaf, start, end]`` offsets and, given a
    ``digester``, the keyed digests of the distinct literals.
    """
    column = column_for(guard, audience)
    groups: dict[tuple[str, str, str, bool, bool], _Group] = {}
    for leaf, text in enumerate(texts):
        decisions = decide(
            text,
            guard=guard,
            identity=identity,
            table=table,
            audience=audience,
            destination=destination,
        )
        for d in decisions:
            first = is_first_name_alone(Match(d.start, d.end, d.kind, d.literal))
            relaxed = (
                column == "egress"
                and d.action == "log"
                and table.action(d.kind, "egress", first_name=first) == "deny"
            )
            key = (column, d.kind, d.action, first, relaxed)
            group = groups.get(key)
            if group is None:
                group = groups[key] = _Group(column, d.kind, d.action, first, relaxed)
            group.count += 1
            if len(group.spans) < MAX_SPANS:
                group.spans.append([leaf, d.start, d.end])
            if digester is not None:
                group.digests.add(digester.digest([d.kind, canonical(d.kind, d.literal)]))
    ordered = sorted(groups.values(), key=lambda g: (g.guard, g.kind, g.action))
    return [g.row(with_digests=digester is not None) for g in ordered]


class OwnerPiiShadowHook:
    """Audits what every owner-PII guard would do; changes nothing (ADR-0125 PR 4).

    Registered at ``PRE_TOOL_USE``, ``PRE_LLM_CALL``, ``PRE_RESPONSE`` and ``PRE_EGRESS`` (one instance,
    ``kernel.register(hook, at=...)``). ``network_tools`` is the egress guard's pattern
    set, empty when that guard is not registered (nothing to shadow).
    """

    name: str = HOOK_NAME
    hook_point: HookPoint = HookPoint.PRE_TOOL_USE
    # First at every point, before any hook that transforms, denies or annotates: the
    # kernel's final decision stays the last real guard's (module docstring).
    priority: int = 1

    def __init__(self, *, network_tools: frozenset[str] = DEFAULT_NETWORK_TOOLS) -> None:
        self._network_tools = network_tools

    async def __call__(self, ctx: HookContext) -> HookDecision:
        try:
            report = self.observe(ctx)
        except Exception as exc:  # noqa: BLE001 - an observer never breaks a call
            # The class only: an exception's text could quote what it was reading.
            logger.warning("owner_pii_shadow: observation failed (%s)", type(exc).__name__)
            return HookDecision(
                outcome="allow",
                reason=f"owner_pii_shadow: observation failed ({type(exc).__name__}); "
                "nothing changed",
                audit_metadata={SHADOW_KEY: {"checked": [], "error": type(exc).__name__}},
            )
        seen = sum(int(o["count"]) for o in report.get("observations", ()))
        if report.get("note"):
            reason = f"owner_pii_shadow: not observed ({report['note']}); nothing changed"
        elif not report["checked"]:
            reason = "owner_pii_shadow: no owner-PII guard reads this call"
        else:
            reason = f"owner_pii_shadow: {seen} occurrence(s) observed; nothing changed"
        return HookDecision(outcome="allow", reason=reason, audit_metadata={SHADOW_KEY: report})

    def observe(self, ctx: HookContext) -> dict[str, Any]:
        """The row's ``owner_pii_shadow`` value for ``ctx`` (pure apart from the corpus)."""
        checks: list[tuple[Guard, list[str], dict[str, Any]]] = []
        payload = ctx.payload
        if ctx.hook_point == HookPoint.PRE_TOOL_USE:
            tool = tool_name_of(payload)
            args = list(_leaves(payload.get(ARGS)))
            if (tool and is_network_tool(tool, self._network_tools)) or sends_to_external_service(
                ctx.metadata
            ):
                # A network tool by name, or a tool that declares its arguments go to an
                # external service (issue #103): one egress check, whichever says so.
                checks.append(("egress", args, {"destination": tool}))
            if sends_to_search_engine(ctx.metadata):
                checks.append(("web_search", args, {}))
        elif ctx.hook_point == HookPoint.PRE_EGRESS:
            # The request a plugin makes through the governed client: the strings in its
            # address and parameters, against the egress column, destination the host.
            egress = ctx.payload.get("egress")
            host = egress.get("host") if isinstance(egress, dict) else None
            checks.append(
                (
                    "egress",
                    list(_leaves(ctx.payload.get("egress_content"))),
                    {"destination": str(host or "")},
                )
            )
        elif ctx.hook_point == HookPoint.PRE_LLM_CALL:
            prompt = payload.get("prompt")
            if ctx.tier == CLOUD_TIER and isinstance(prompt, str):
                checks.append(("tier3", [prompt], {}))
        elif ctx.hook_point == HookPoint.PRE_RESPONSE:
            response = payload.get(RESPONSE)
            if isinstance(response, str):
                checks.append(("answer", [response], {"audience": audience_of(payload)}))

        checked = [column_for(g, kw.get("audience", "owner")) for g, _, kw in checks]
        report: dict[str, Any] = {"checked": checked}
        if not checks:
            return report
        identity = owner_identity()
        if identity is None:
            report["note"] = "no_identity"
            return report
        try:
            table = guard_table()
        except Exception:  # noqa: BLE001 - a malformed table is reported, not guessed
            report["note"] = "table_unreadable"
            return report
        if table is None:
            report["note"] = "no_table"
            return report
        digester = _digester_if_ready()
        observations: list[dict[str, Any]] = []
        for guard, texts, kwargs in checks:
            observations.extend(
                observe_texts(
                    texts,
                    guard=guard,
                    identity=identity,
                    table=table,
                    digester=digester,
                    **kwargs,
                )
            )
        report["observations"] = observations
        if digester is not None:
            report["digest_alg"] = digester.alg
        return report


# -- the read-back ---------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowCell:
    """One hook point x guard x kind x action: how often, in how many calls."""

    hook_point: str
    guard: str
    kind: str
    action: str
    first_name_alone: bool
    log_only_destination: bool
    occurrences: int
    calls: int
    # Distinct literals, by keyed digest; None when any contributing row had no key.
    distinct: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "hook_point": self.hook_point,
            "guard": self.guard,
            "kind": self.kind,
            "action": self.action,
            "first_name_alone": self.first_name_alone,
            "log_only_destination": self.log_only_destination,
            "occurrences": self.occurrences,
            "calls": self.calls,
            "distinct": self.distinct,
        }


@dataclass(frozen=True)
class ShadowSummary:
    """Shadow observations over a window. Counts only: no row holds a literal."""

    since: str
    rows: int
    # Calls each guard column read (the denominator), by column.
    checked: dict[str, int]
    # Calls a guard would have read but could not: no corpus, no table, an error.
    unobserved: dict[str, int]
    cells: tuple[ShadowCell, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "since": self.since,
            "rows": self.rows,
            "checked": dict(self.checked),
            "unobserved": dict(self.unobserved),
            "cells": [c.as_dict() for c in self.cells],
        }


@dataclass
class _Tally:
    occurrences: int = 0
    calls: int = 0
    digests: set[tuple[str, str]] = field(default_factory=set)
    keyless: bool = False


def summarize_shadow_rows(rows: Iterable[AuditRow], *, since: str = "") -> ShadowSummary:
    """Aggregate ``owner_pii_shadow`` rows (any others are skipped)."""
    checked: dict[str, int] = {}
    unobserved: dict[str, int] = {}
    tallies: dict[tuple[str, str, str, str, bool, bool], _Tally] = {}
    count = 0
    for row in rows:
        if row.plugin != HOOK_NAME:
            continue
        try:
            report = json.loads(row.payload_json).get(SHADOW_KEY)
        except (ValueError, AttributeError):
            continue
        if not isinstance(report, dict):
            continue
        count += 1
        note = report.get("note") or report.get("error")
        if note:
            unobserved[str(note)] = unobserved.get(str(note), 0) + 1
            continue
        for column in report.get("checked", ()):
            checked[str(column)] = checked.get(str(column), 0) + 1
        alg = str(report.get("digest_alg") or "")
        for obs in report.get("observations", ()):
            key = (
                row.hook_point,
                str(obs.get("guard")),
                str(obs.get("kind")),
                str(obs.get("action")),
                bool(obs.get("first_name_alone")),
                bool(obs.get("log_only_destination")),
            )
            tally = tallies.setdefault(key, _Tally())
            tally.occurrences += int(obs.get("count", 0))
            tally.calls += 1
            digests = obs.get("digests")
            if isinstance(digests, list) and alg:
                tally.digests.update((alg, str(d)) for d in digests)
            else:
                tally.keyless = True
    cells = tuple(
        ShadowCell(
            hook_point=k[0],
            guard=k[1],
            kind=k[2],
            action=k[3],
            first_name_alone=k[4],
            log_only_destination=k[5],
            occurrences=t.occurrences,
            calls=t.calls,
            distinct=None if t.keyless else len(t.digests),
        )
        for k, t in sorted(tallies.items())
    )
    return ShadowSummary(
        since=since, rows=count, checked=checked, unobserved=unobserved, cells=cells
    )


def owner_pii_shadow_summary(
    audit_log: AuditLog | None = None,
    *,
    days: float = 7.0,
    now: datetime | None = None,
) -> ShadowSummary:
    """The shadow rows of the last ``days`` in ``audit_log`` (the default ledger), summed."""
    log = audit_log if audit_log is not None else AuditLog(db_path=audit_db_path())
    since = (now or datetime.now(UTC)) - timedelta(days=days)
    rows = log.query(plugin=HOOK_NAME, since=since)
    return summarize_shadow_rows(rows, since=since.isoformat())


__all__ = [
    "ACCEPTED_MODES",
    "CLOUD_TIER",
    "ENV_FLAG",
    "HOOK_NAME",
    "MAX_SPANS",
    "SHADOW_KEY",
    "SHADOW_POINTS",
    "ModeSetting",
    "OwnerPiiMode",
    "OwnerPiiShadowHook",
    "ShadowCell",
    "ShadowSummary",
    "observe_texts",
    "owner_pii_mode_from_env",
    "owner_pii_shadow_summary",
    "parse_owner_pii_mode",
    "summarize_shadow_rows",
]
