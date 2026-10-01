"""Capability results carry the owner's identity as pseudonyms (ADR-0125, PR 3).

A consumer of a capability receives ``[owner:<kind>#<n>]`` for each of the owner's names,
emails, phones, addresses and handles, unless its own manifest grants the kind for that
capability. Secrets stay ``[redacted: identity]`` and no grant unmasks them. The consumer
is the caller the harness stamps, so one plugin cannot use another's grant. A result that
held the owner's identity is marked ``personal``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import pytest

from iris_harness.foundation import capabilities as catalogue
from iris_harness.foundation.capabilities import CapabilitySpec, MethodSpec
from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
)
from iris_harness.kernel.governance.caller_policy import register_caller_policy
from iris_harness.kernel.governance.identity_config import guard_table
from iris_harness.kernel.governance.owner_pii import (
    MASK,
    pseudonym,
    redact_capability_text,
    reset_pseudonyms,
)
from iris_harness.kernel.governance.plugins.caller_policy import CallerPolicyHook
from iris_harness.kernel.governance.plugins.capability_redaction import CapabilityRedactionHook
from iris_harness.kernel.governance.plugins.tool_policy import ToolPolicyHook
from iris_harness.kernel.governance.unmask_grants import register_unmask_policy, unmask_grants
from iris_harness.runtime.plugin_host.manifest import PluginCapabilities, PluginManifest
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.runtime.tool_access import compile_caller_policy, compile_unmask_policy

SECRET = "CANARY_SOUL_SECRET_DIRECTIVE_d41f8a27"
NAME = "Robin Example"
EMAIL = "owner.canary@example.com"
PHONE = "+1 555 0100 0199"
HOME = "1 Example Street, Springfield"
OFFICE = "9 Sample Road, Shelbyville"
HANDLE = "robin-gh"


@dataclasses.dataclass(frozen=True)
class Msg:
    subject: str
    body: str


class Inbox(Protocol):
    def search(self, query: str) -> list[Msg]: ...


INBOX = CapabilitySpec(
    name="test.inbox",
    protocol=Inbox,
    methods={"search": MethodSpec(fields=("[].subject", "[].body"))},
)
BODY = (
    f"{NAME} ({EMAIL}, {PHONE}) lives at {HOME}, works at {OFFICE}, "
    f"is @{HANDLE}; the key is {SECRET}"
)
MAILBOX = [Msg("For Robin", BODY)]


class Provider:
    def search(self, query: str) -> list[Msg]:
        return list(MAILBOX)


class Spy:
    name = "spy"
    hook_point = HookPoint.POST_TOOL_USE
    priority = 99

    def __init__(self) -> None:
        self.seen: list[HookContext] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.seen.append(ctx)
        return HookDecision(outcome="allow", reason="spy")


@pytest.fixture(autouse=True)
def _isolation(monkeypatch: pytest.MonkeyPatch, owner_identity_seam: Any) -> Iterator[None]:
    monkeypatch.setattr(catalogue, "CAPABILITIES", MappingProxyType({"test.inbox": INBOX}))
    owner_identity_seam.register_identity_text_provider(lambda: [f"my api key: {SECRET}"])
    owner_identity_seam.register_owner_identity_source(
        "profile",
        lambda: {
            "name": [NAME, "Robin"],
            "email": [EMAIL],
            "phone": [PHONE],
            "address": [HOME, OFFICE],
            "handle": [HANDLE],
        },
    )
    reset_pseudonyms()
    yield
    register_caller_policy(None)
    register_unmask_policy(None)
    reset_pseudonyms()


def _registry(
    tmp_path: Path, kernel: GovernanceKernel, consumers: dict[str, Any]
) -> PluginRegistry:
    registry = PluginRegistry()
    plugins = {"mail": {"provides": ["test.inbox"]}, **consumers}
    for name, caps in plugins.items():
        registry.add_plugin(
            PluginRecord(
                name=name,
                source="t",
                status=PluginStatus.LOADED,
                manifest=PluginManifest.model_validate({"name": name, "capabilities": caps}),
            )
        )
    assert registry.provide_capability("mail", "test.inbox", Provider())
    registry.bind_kernel(lambda: kernel)
    register_caller_policy(compile_caller_policy(registry, config_dir=tmp_path))
    register_unmask_policy(compile_unmask_policy(registry))
    return registry


def _kernel(*extra: Any) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=None)
    for hook in (CallerPolicyHook(), ToolPolicyHook(), CapabilityRedactionHook(), *extra):
        kernel.register(hook)
    kernel.init_lock()
    return kernel


PLANNER = {"uses": [{"test.inbox": {"unmask": ["name", "address"]}}]}
FINANCE = {"uses": ["test.inbox"]}


def _body(registry: PluginRegistry, consumer: str) -> str:
    out = registry.resolve_capability(consumer, "test.inbox").search("q")
    return str(out[0].body)


# -- what a consumer receives --------------------------------------------------------------


def test_a_consumer_without_grants_sees_pseudonyms_and_a_masked_secret(tmp_path: Path) -> None:
    registry = _registry(tmp_path, _kernel(), {"fin": FINANCE})
    body = _body(registry, "fin")
    for literal in (NAME, EMAIL, PHONE, HOME, OFFICE, HANDLE, SECRET):
        assert literal not in body
    assert body == (
        "[owner:name#1] ([owner:email#1], [owner:phone#1]) lives at [owner:address#1], "
        f"works at [owner:address#2], is [owner:handle#1]; the key is {MASK}"
    )


def test_a_grant_unmasks_exactly_its_kinds(tmp_path: Path) -> None:
    registry = _registry(tmp_path, _kernel(), {"planner": PLANNER})
    body = _body(registry, "planner")
    assert NAME in body and HOME in body and OFFICE in body
    assert EMAIL not in body and PHONE not in body and HANDLE not in body
    assert SECRET not in body and MASK in body


def test_a_consumer_cannot_use_another_consumers_grant(tmp_path: Path) -> None:
    """The grant is read for the caller the harness stamps: fin never gets planner's."""
    registry = _registry(tmp_path, _kernel(), {"planner": PLANNER, "fin": FINANCE})
    assert NAME in _body(registry, "planner")
    fin = _body(registry, "fin")
    assert NAME not in fin and HOME not in fin
    assert unmask_grants("plugin:fin", "test.inbox") == frozenset()
    assert unmask_grants("plugin:planner", "test.inbox") == {"name", "address"}
    assert unmask_grants("core:anything", "test.inbox") == frozenset()


def test_a_grant_is_per_capability(tmp_path: Path) -> None:
    _registry(tmp_path, _kernel(), {"planner": PLANNER})
    assert unmask_grants("plugin:planner", "test.other") == frozenset()


def test_no_grant_unmasks_a_secret() -> None:
    table = guard_table()
    corpus_text = f"key {SECRET}"
    from iris_harness.kernel.governance.identity_redaction import owner_identity

    identity = owner_identity()
    assert identity is not None and table is not None
    out = redact_capability_text(
        corpus_text, identity=identity, table=table, grants={"secret", "name", "link"}
    )
    assert out.text == f"key {MASK}"
    with pytest.raises(ValueError):
        PluginCapabilities.model_validate({"uses": [{"test.inbox": {"unmask": ["secret"]}}]})


def test_the_hook_ignores_a_grant_for_a_kind_the_table_does_not_pseudonymise(
    tmp_path: Path,
) -> None:
    """A policy that hands out ``secret`` still unmasks nothing it cannot."""
    registry = _registry(tmp_path, _kernel(), {"fin": FINANCE})
    register_unmask_policy(lambda _c, _cap: frozenset({"secret"}))
    assert SECRET not in _body(registry, "fin")


def test_with_no_policy_registered_nothing_is_granted(tmp_path: Path) -> None:
    registry = _registry(tmp_path, _kernel(), {"planner": PLANNER})
    register_unmask_policy(None)
    assert NAME not in _body(registry, "planner")


# -- pseudonyms ------------------------------------------------------------------------------


def test_pseudonyms_are_stable_per_literal_and_distinct_across_literals(tmp_path: Path) -> None:
    registry = _registry(tmp_path, _kernel(), {"fin": FINANCE})
    first = _body(registry, "fin")
    assert _body(registry, "fin") == first
    assert pseudonym("address", HOME) == "[owner:address#1]"
    assert pseudonym("address", OFFICE) == "[owner:address#2]"
    # Another spelling of the same literal is the same pseudonym.
    assert pseudonym("address", "1 example street springfield") == "[owner:address#1]"
    assert pseudonym("phone", "+1 (555) 0100-0199") == pseudonym("phone", PHONE)
    assert pseudonym("email", EMAIL.upper()) == pseudonym("email", EMAIL)


def test_numbering_is_per_kind() -> None:
    assert pseudonym("email", "a@example.com") == "[owner:email#1]"
    assert pseudonym("phone", "+1 555 0100 0100") == "[owner:phone#1]"
    assert pseudonym("email", "b@example.com") == "[owner:email#2]"


def test_a_first_name_alone_reaches_the_consumer_unchanged(tmp_path: Path) -> None:
    """Masked only in web-search arguments (ADR-0125 decision 4)."""
    registry = _registry(tmp_path, _kernel(), {"fin": FINANCE})
    MAILBOX[:] = [Msg("hi", "Robin says hi")]
    try:
        assert _body(registry, "fin") == "Robin says hi"
    finally:
        MAILBOX[:] = [Msg("For Robin", BODY)]


# -- classification and audit ------------------------------------------------------------


def test_a_result_with_owner_identity_is_marked_personal(tmp_path: Path) -> None:
    spy = Spy()
    registry = _registry(tmp_path, _kernel(spy), {"planner": PLANNER})
    _body(registry, "planner")
    (ctx,) = spy.seen
    assert ctx.classification == "personal"


def test_later_hooks_see_the_redacted_result(tmp_path: Path) -> None:
    spy = Spy()
    registry = _registry(tmp_path, _kernel(spy), {"fin": FINANCE})
    _body(registry, "fin")
    (ctx,) = spy.seen
    assert "[owner:name#1]" in ctx.payload["result"]


async def test_the_hook_reports_kinds_and_paths() -> None:
    hook = CapabilityRedactionHook()
    ctx = HookContext(
        hook_point=HookPoint.POST_TOOL_USE,
        run_id="r",
        agent_type="chat",
        payload={
            "tool_name": "capability:test.inbox.search",
            "caller": "plugin:fin",
            "capability": "test.inbox",
            "fields": {"[0].body": f"to {EMAIL}", "[0].subject": "hello"},
            "result": f"to {EMAIL}\nhello",
        },
    )
    decision = await hook(ctx)
    assert decision.outcome == "transform"
    assert decision.set_classification == "personal"
    assert decision.audit_metadata == {
        "masked_fields": ["[0].body"],
        "identity_kinds": ["email"],
    }
    assert EMAIL not in str(decision.audit_metadata)


async def test_a_result_without_identity_is_left_alone() -> None:
    ctx = HookContext(
        hook_point=HookPoint.POST_TOOL_USE,
        run_id="r",
        agent_type="chat",
        payload={
            "tool_name": "capability:test.inbox.search",
            "caller": "plugin:fin",
            "capability": "test.inbox",
            "fields": {"[0].body": "nothing here"},
            "result": "nothing here",
        },
    )
    decision = await CapabilityRedactionHook()(ctx)
    assert decision.outcome == "allow" and decision.set_classification is None


# -- the manifest ------------------------------------------------------------------------


def test_the_manifest_reads_grants_from_uses_and_requires_entries() -> None:
    caps = PluginCapabilities.model_validate(
        {
            "uses": ["a.read", {"b.read": {"unmask": ["name", "address", "name"]}}],
            "requires": [{"c.read": {"unmask": ["email"]}}, {"d.read": None}],
        }
    )
    assert caps.uses == ("a.read", "b.read")
    assert caps.requires == ("c.read", "d.read")
    assert caps.unmask == {"b.read": ("name", "address"), "c.read": ("email",)}
    # A dumped manifest loads back the same.
    assert PluginCapabilities.model_validate(caps.model_dump()) == caps


@pytest.mark.parametrize(
    "raw",
    [
        {"uses": [{"b.read": {"unmask": ["secret"]}}]},
        {"uses": [{"b.read": {"unmask": ["link"]}}]},
        {"uses": [{"b.read": {"unmask": ["name"], "also": 1}}]},
        {"uses": [{"b.read": {}, "c.read": {}}]},
        {"uses": ["b.read"], "unmask": {"c.read": ["name"]}},  # not consumed
        {"provides": ["b.read"], "unmask": {"b.read": ["name"]}},  # its own
    ],
)
def test_the_manifest_refuses_a_bad_grant(raw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PluginCapabilities.model_validate(raw)


def test_a_bare_name_grants_nothing() -> None:
    assert PluginCapabilities.model_validate({"uses": ["b.read"]}).unmask == {}


def test_the_grantable_kinds_are_the_owners_pii_kinds() -> None:
    from iris_harness.kernel.governance.owner_identity import OWNER_PII_KINDS
    from iris_harness.runtime.plugin_host.manifest import GrantableKind

    assert set(GrantableKind.__args__) == set(OWNER_PII_KINDS)  # type: ignore[attr-defined]
    table = guard_table()
    assert table is not None and table.grantable() == set(OWNER_PII_KINDS)
