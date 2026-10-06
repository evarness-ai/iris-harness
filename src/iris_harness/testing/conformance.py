"""The governance conformance suite: what any plugin's own CI can assert about itself.

Stable tier (L4.5 hardening). Hand :func:`check_conformance` your plugin (as
:func:`~iris_harness.testing.plugin` wraps it) and one example call for every tool and
capability method it declares; it mounts the plugin in a governed :func:`harness`, runs
each example from code, as your plugin's own caller, and reads the audit ledger:

* **audit** -- every call has a ``pre_tool_use`` row and, when it ran, a
  ``post_tool_use`` row naming it;
* **caller** -- every one of those rows names the caller the harness stamped
  (``plugin:<name>``): the plugin has no way to name another;
* **approval** -- a call that changes the owner's world without asking first (an
  ``effect: destructive`` tool, a write that confirms) is held for the owner, never run
  unasked. Approved, it runs exactly once with the arguments it was queued with (the
  ``args_digest`` of the run matches the held call's). Rejected, it never runs;
* **coverage** -- every declared tool and every method of a provided capability has an
  example. One without is reported, not passed: an unexercised tool is unproven.

An approval check never passes on nothing: a held call with no queued approval, no
``args_digest`` on the rows to compare the queued and the run arguments by, or no outcome
row for the approval is a violation, not a skipped step.

Two harnesses are built, one that approves every held call and one that rejects it, so the
plugin's ``setup`` runs twice and must not depend on state the first run left behind (a
plugin that mounts the first time and not the second is a ``mount`` violation).

What the suite does not prove. Whether a tool must be held is read from the plugin's own
manifest (``effect``, ``confirm``), as governance reads it: a plugin that declares
``effect: read`` and writes anyway is not detected here. A provided capability's write
that confirms is refused from code and never queued (there is no approval to answer), so
for those the suite checks that the call was held, the provider's result never left, and
the ledger has the refusal; it does not run an approve / reject round as it does for tools.
The suite runs the plugin on a worker thread when the caller is inside a running event
loop, since the governed call path cannot run inside one.

The result is a list of :class:`Violation` (empty when the plugin conforms);
:func:`assert_conformant` raises :class:`ConformanceError` listing them, for a one-line
test::

    def test_my_plugin_conforms() -> None:
        assert_conformant(
            plugin(setup, manifest="manifest.yaml"),
            tools={"list_notes": {}, "remove_note": {"note_id": "n2"}},
        )
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from iris_harness.foundation.capabilities import (
    CapabilityDenied,
    capability_tool_name,
    published_capability,
)
from iris_harness.testing.harness import Harness, harness, plugin

if TYPE_CHECKING:
    from iris_harness.kernel.governance.audit.log import AuditRow
    from iris_harness.runtime.plugin_host.loader import InProcessPlugin
    from iris_harness.runtime.tool_service import ToolService
    from iris_harness.sdk import PluginAPI

PRE = "pre_tool_use"
POST = "post_tool_use"
# Where the approval queue records what became of an approved (or rejected) code call.
APPROVAL_HOOK = "approval_queue"
# The plugin the suite mounts to call a provided capability as a consumer would.
CONSUMER = "conformance-consumer"

ConformanceCheck = Literal["mount", "coverage", "audit", "caller", "approval", "example"]


@dataclass(frozen=True)
class Violation:
    """One way the plugin does not conform.

    ``check`` is the rule broken (``mount``, ``coverage``, ``audit``, ``caller``,
    ``approval``, or ``example`` -- the example call itself failed, so nothing was
    proven), ``subject`` the tool or ``capability:<name>.<method>`` it is about, and
    ``detail`` what was seen.
    """

    check: ConformanceCheck
    subject: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.check}] {self.subject}: {self.detail}"


class ConformanceError(AssertionError):
    """:func:`assert_conformant` found violations; ``violations`` lists them."""

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        lines = "\n".join(f"  {v}" for v in violations)
        super().__init__(f"{len(violations)} governance conformance violation(s):\n{lines}")


ToolExamples = Mapping[str, Mapping[str, Any]]
CapabilityExamples = Mapping[str, Mapping[str, Mapping[str, Any]]]


def check_conformance(
    plugin_under_test: InProcessPlugin,
    *,
    tools: ToolExamples | None = None,
    capabilities: CapabilityExamples | None = None,
    profile: str = "minimal",
) -> list[Violation]:
    """Run the conformance checks against ``plugin_under_test``; empty when it conforms.

    ``tools`` maps each declared tool to the arguments of one example call;
    ``capabilities`` maps each provided capability to ``{method: arguments}``. ``profile``
    is the shipped profile the plugin is mounted on (see :func:`harness`).

    Safe to call from async code: inside a running event loop the checks run on a worker
    thread, because the governed call path refuses to run inside a loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _check(plugin_under_test, tools, capabilities, profile)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_check, plugin_under_test, tools, capabilities, profile).result()


def _check(
    plugin_under_test: InProcessPlugin,
    tools: ToolExamples | None,
    capabilities: CapabilityExamples | None,
    profile: str,
) -> list[Violation]:
    tools = dict(tools or {})
    capabilities = {name: dict(methods) for name, methods in (capabilities or {}).items()}
    manifest = plugin_under_test.manifest
    violations = _coverage(plugin_under_test, tools, capabilities)
    caller = f"plugin:{manifest.name}"

    consumer: list[PluginAPI] = []
    mounted = _mounted(plugin_under_test, capabilities, consumer)
    with harness(profile=profile, plugins=mounted) as h:
        if not h.plugin_loaded(manifest.name):
            status, error = h.plugins().get(manifest.name, ("absent", None))
            return [*violations, Violation("mount", manifest.name, error or status)]
        declared = {n: a for n, a in tools.items() if n in manifest.tools}
        held_tools = {n: a for n, a in declared.items() if _expects_hold(h, n)}
        for name, args in declared.items():
            violations += _tool_call(h, caller, name, args, approve=True)
        provided = set(manifest.capabilities.provides)
        violations += _capability_calls(
            h, {c: m for c, m in capabilities.items() if c in provided}, consumer
        )

    if held_tools:
        with harness(profile=profile, plugins=[plugin_under_test]) as h:
            if not h.plugin_loaded(manifest.name):
                status, error = h.plugins().get(manifest.name, ("absent", None))
                detail = f"mounted the first time, not the rejecting run's: {error or status}"
                return [*violations, Violation("mount", manifest.name, detail)]
            for name, args in held_tools.items():
                violations += _tool_call(h, caller, name, args, approve=False)
    return violations


def assert_conformant(
    plugin_under_test: InProcessPlugin,
    *,
    tools: ToolExamples | None = None,
    capabilities: CapabilityExamples | None = None,
    profile: str = "minimal",
) -> None:
    """:func:`check_conformance`, raising :class:`ConformanceError` on any violation."""
    violations = check_conformance(
        plugin_under_test, tools=tools, capabilities=capabilities, profile=profile
    )
    if violations:
        raise ConformanceError(violations)


# ------------------------------------------------------------------------- coverage
def _coverage(
    under_test: InProcessPlugin, tools: ToolExamples, capabilities: CapabilityExamples
) -> list[Violation]:
    manifest = under_test.manifest
    out: list[Violation] = []
    for name in sorted(set(manifest.tools) - set(tools)):
        out.append(Violation("coverage", name, "declared but no example call was given"))
    for name in sorted(set(tools) - set(manifest.tools)):
        out.append(
            Violation("coverage", name, "an example for a tool the manifest does not declare")
        )
    provided = set(manifest.capabilities.provides)
    for cap in sorted(provided | set(capabilities)):
        spec = published_capability(cap)
        if spec is None:
            out.append(Violation("coverage", cap, "not a capability this SDK publishes"))
            continue
        if cap not in provided:
            out.append(Violation("coverage", cap, "an example for a capability not provided"))
            continue
        given = capabilities.get(cap, {})
        for method in sorted(set(spec.members()) - set(given)):
            subject = capability_tool_name(cap, method)
            out.append(Violation("coverage", subject, "declared but no example call was given"))
        for method in sorted(set(given) - set(spec.members())):
            subject = capability_tool_name(cap, method)
            out.append(Violation("coverage", subject, "an example for a method it does not have"))
    return out


def _mounted(
    under_test: InProcessPlugin, capabilities: CapabilityExamples, consumer: list[PluginAPI]
) -> list[Any]:
    """The plugin, then (when it provides capabilities) a consumer that uses them; the
    consumer's ``PluginAPI`` lands in ``consumer`` when the harness mounts it."""
    used = sorted(c for c in capabilities if c in under_test.manifest.capabilities.provides)
    if not used:
        return [under_test]
    return [
        under_test,
        plugin(consumer.append, manifest={"name": CONSUMER, "capabilities": {"uses": used}}),
    ]


# ----------------------------------------------------------------------------- tools
def _tool_service(h: Harness) -> ToolService:
    service = h._runtime.tool_service
    if service is None:
        raise RuntimeError("the harness built no tool service; nothing can be checked")
    return service


def _expects_hold(h: Harness, name: str) -> bool:
    """Whether a code call of ``name`` must wait for the owner: a destructive tool, or a
    write that confirms (``once`` from code has nobody to ask; ``approval`` is pinned)."""
    info = next(iter(_tool_service(h).describe(name)), None)
    if info is None:
        return False
    return info.effect == "destructive" or (info.effect == "write" and info.confirm != "never")


def _tool_call(
    h: Harness, caller: str, name: str, args: Mapping[str, Any], *, approve: bool
) -> list[Violation]:
    tools = _tool_service(h).for_caller(caller)
    expects_hold = _expects_hold(h, name)
    mark, failures = h._audit_high_water(), _failures(h, caller)
    result = tools.call(name, dict(args))
    rows = _tool_rows(h, name, after=mark)
    out = _audited(name, rows, caller, ran=not result.held)

    if expects_hold and not result.held:
        out.append(Violation("approval", name, "ran from code without the owner's approval"))
        return out
    if not result.held:
        if not result.ok or _failures(h, caller) > failures:
            out.append(Violation("example", name, f"the example call failed: {result.text}"))
        return out
    if not expects_hold:
        out.append(
            Violation("approval", name, f"held, but its declaration says it runs: {result.text}")
        )
        return out
    if result.approval_id is None:
        out.append(Violation("approval", name, f"held without a queued approval: {result.text}"))
        return out

    queued = {d for d in (_digest(r) for r in rows) if d is not None}
    mark, failures = h._audit_high_water(), _failures(h, caller)
    h.respond_to_approval(result.approval_id, approve=approve)
    after = _tool_rows(h, name, after=mark)
    runs = _executions(after)
    closed = _approval_status(h, result.approval_id, after=mark)
    if closed is None:
        out.append(Violation("audit", name, "the approval's outcome has no audit row"))
    if not approve:
        if runs or closed not in (None, "rejected"):
            out.append(Violation("approval", name, "ran after the owner rejected it"))
        return out
    if closed == "failed" or _failures(h, caller) > failures:
        out.append(Violation("example", name, "approved, the example call failed when it ran"))
    elif closed not in (None, "ran"):
        out.append(Violation("approval", name, f"approved, but it did not run ({closed})"))
    out += _audited(name, after, caller, ran=True)
    if runs != 1:
        out.append(Violation("approval", name, f"approved, it ran {runs} time(s), not once"))
    ran_with = {d for d in (_digest(r) for r in after if r.hook_point == PRE) if d is not None}
    if not queued or not ran_with:
        # Nothing to compare is not a match: the pinned arguments went unchecked.
        side = "queued" if not queued else "run"
        out.append(
            Violation(
                "approval",
                name,
                f"the {side} call's rows carry no args_digest; its arguments "
                "cannot be shown to be the ones the owner approved",
            )
        )
    elif ran_with != queued:
        out.append(
            Violation("approval", name, "approved, it ran with arguments other than those queued")
        )
    return out


# ---------------------------------------------------------------------- capabilities
def _capability_calls(
    h: Harness, capabilities: CapabilityExamples, consumer: list[PluginAPI]
) -> list[Violation]:
    """Call each provided capability's methods as a consumer would; the examples are
    already narrowed to capabilities the plugin provides."""
    if not capabilities:
        return []
    if not consumer:
        return [Violation("mount", CONSUMER, "the suite's consumer plugin did not mount")]
    api = consumer[-1]
    caller = f"plugin:{CONSUMER}"
    out: list[Violation] = []
    for cap, methods in capabilities.items():
        spec = published_capability(cap)
        if spec is None:
            continue
        impl = api.capability(cap)
        if impl is None:
            out.append(Violation("mount", cap, "provided, but a consumer that uses it got None"))
            continue
        for method, args in methods.items():
            if method not in spec.members():
                continue
            subject = capability_tool_name(cap, method)
            mark = h._audit_high_water()
            held = False
            try:
                _invoke(getattr(impl, method), dict(args))
            except CapabilityDenied as exc:
                held = exc.outcome == "require_approval"
                if not held:
                    out.append(Violation("example", subject, f"denied by governance: {exc.reason}"))
            except Exception as exc:  # noqa: BLE001 -- the provider's failure, reported
                out.append(Violation("example", subject, f"the example call failed: {exc!r}"))
            rows = _tool_rows(h, subject, after=mark)
            out += _audited(subject, rows, caller, ran=not held)
            if held:
                out += _held_capability(subject, rows)
            if held != (spec.methods[method].confirm_mode == "once"):
                state = "held" if held else "ran"
                out.append(Violation("approval", subject, f"{state}, against its declared effect"))
    return out


def _held_capability(subject: str, rows: list[AuditRow]) -> list[Violation]:
    """A capability call held for approval is refused from code (nothing is queued, so
    there is no approval to answer): the provider's result must not have left, and a hook
    must have recorded the hold."""
    out: list[Violation] = []
    if any(row.hook_point == POST for row in rows):
        out.append(Violation("approval", subject, f"held, yet it has a {POST} row: it ran"))
    if not any(row.hook_point == PRE and row.decision == "require_approval" for row in rows):
        out.append(Violation("approval", subject, f"held, but no {PRE} row records the hold"))
    return out


def _executions(rows: list[AuditRow]) -> int:
    """How many times the tool ran: each hook writes one ``post_tool_use`` row per run, so
    the busiest hook's row count is the number of runs. Distinct ``run_id`` is no count:
    a call run twice by the harness keeps its run id."""
    per_hook = Counter(row.plugin for row in rows if row.hook_point == POST)
    return max(per_hook.values(), default=0)


def _invoke(method: Any, args: dict[str, Any]) -> Any:
    result = method(**args)
    if inspect.isawaitable(result):
        return asyncio.run(_awaited(result))
    if inspect.isasyncgen(result) or hasattr(result, "__aiter__"):
        return asyncio.run(_drained(result))
    if inspect.isgenerator(result):
        return list(result)
    return result


async def _awaited(awaitable: Any) -> Any:
    return await awaitable


async def _drained(stream: Any) -> list[Any]:
    return [item async for item in stream]


# --------------------------------------------------------------------------- the ledger
def _tool_rows(h: Harness, subject: str, *, after: int) -> list[AuditRow]:
    from iris_harness.kernel.governance.audit.log import AuditLog

    return [
        row
        for row in AuditLog(db_path=h.audit_db).query(since=h._started)
        if row.id > after
        and row.hook_point in (PRE, POST)
        and _payload(row).get("tool_name") == subject
    ]


def _failures(h: Harness, caller: str) -> int:
    """How many failures the plugin's fault boundary has caught so far. A tool that raises
    is caught there and answered with an "unavailable" message, so the call's own result
    does not say it failed; the registry does."""
    name = caller.removeprefix("plugin:")
    record = next((r for r in h._runtime.plugin_registry.plugins() if r.name == name), None)
    return int(record.failure_count) if record is not None else 0


def _approval_status(h: Harness, approval_id: str, *, after: int) -> str | None:
    """What the approval queue recorded for ``approval_id``: ``ran``, ``failed``, ...."""
    from iris_harness.kernel.governance.audit.log import AuditLog

    for row in AuditLog(db_path=h.audit_db).query(since=h._started):
        payload = _payload(row)
        if (
            row.id > after
            and row.hook_point == APPROVAL_HOOK
            and row.decision.startswith("call_")
            and payload.get("approval_id") == approval_id
        ):
            return row.decision.removeprefix("call_")
    return None


def _audited(subject: str, rows: list[AuditRow], caller: str, *, ran: bool) -> list[Violation]:
    out: list[Violation] = []
    hooks = {row.hook_point for row in rows}
    if PRE not in hooks:
        out.append(Violation("audit", subject, f"no {PRE} row"))
    if ran and POST not in hooks:
        out.append(Violation("audit", subject, f"ran with no {POST} row"))
    stamped = {str(_payload(row).get("caller")) for row in rows}
    if rows and stamped != {caller}:
        out.append(
            Violation("caller", subject, f"rows name {sorted(stamped)}, not the stamp {caller!r}")
        )
    return out


def _payload(row: AuditRow) -> dict[str, Any]:
    try:
        payload = json.loads(row.payload_json)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _digest(row: AuditRow) -> str | None:
    value = _payload(row).get("args_digest")
    return str(value) if value is not None else None


__all__ = [
    "ConformanceCheck",
    "ConformanceError",
    "Violation",
    "assert_conformant",
    "check_conformance",
]
