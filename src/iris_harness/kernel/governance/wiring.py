"""Wiring helpers — construct a governance kernel ready for use by callers.

The kernel is normally a single instance per process. This module owns
the canonical "construct + register all v1 plugins + init_lock" flow so
runtime code (AgenticCore, src/iris_harness/server/governor, future evaluators) calls
one function and gets a ready-to-fire kernel.

Phase 1 registrations:

- ``DataClassifierHook`` at ``PreClassify`` (priority 10)
- ``EgressGate`` at ``PreLLMCall`` (priority 30)
- ``ToolPolicyHook`` at ``PreToolUse`` (priority 20)
- ``DestructiveApprovalHook`` at ``PreToolUse`` (priority 50, ADR-0118)

Subsequent commits register the redaction filter (priority 20 at
``PreLLMCall``), the persona surface plugin, the credential broker,
the evaluator hooks, etc. — all in this module.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from iris_harness.foundation.paths import config_path as _resolved_config_path
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.cost import CostStore
from iris_harness.kernel.governance.evaluator import EvaluatorHook, EvaluatorRegistry
from iris_harness.kernel.governance.evaluator.embeddings import Embedder
from iris_harness.kernel.governance.evaluator.flagged_runs import FlaggedRunThoughtWriter
from iris_harness.kernel.governance.evaluator.signals import (
    ActionRepeatSignal,
    ClassificationViolationSignal,
    CostBudgetSignal,
    GoalDriftSignal,
    LoopDetectSignal,
    StepCapSignal,
    ToolFailureStreakSignal,
)
from iris_harness.kernel.governance.hooks.types import DataClassification, Hook, HookPoint
from iris_harness.kernel.governance.kernel import GovernanceKernel
from iris_harness.kernel.governance.plugins import (
    CommandSandbox,
    CostLimiter,
    CredentialBroker,
    DataClassifier,
    DataClassifierHook,
    DestructiveApprovalHook,
    EgressGate,
    FSJail,
    MCPAllowlistHook,
    MCPServerGovernance,
    NetworkEgress,
    OutputClassifierHook,
    PersonaPolicyDocument,
    PersonaSurface,
    PostToolUseLedgerHook,
    PreToolUseLedgerHook,
    RedactionFilterHook,
    ToolPolicyHook,
)
from iris_harness.kernel.governance.plugins.caller_policy import CallerPolicyHook
from iris_harness.kernel.governance.plugins.capability_redaction import CapabilityRedactionHook
from iris_harness.kernel.governance.plugins.mcp_client_egress import McpClientEgressHook
from iris_harness.kernel.governance.plugins.network_egress import DEFAULT_NETWORK_TOOLS
from iris_harness.kernel.governance.plugins.owner_pii_shadow import (
    SHADOW_POINTS,
    OwnerPiiMode,
    OwnerPiiShadowHook,
    owner_pii_mode_from_env,
)
from iris_harness.kernel.governance.plugins.response_safety import ResponseSafetyHook
from iris_harness.kernel.governance.side_effects import (
    SideEffectLedger,
    shared_side_effect_ledger,
)
from iris_harness.kernel.governance.vault import VaultStore

if TYPE_CHECKING:
    # Imported lazily at runtime by `_approvals_from_env`, so a caller that switches
    # approvals off never pays for the approvals package.
    from iris_harness.kernel.governance.approvals import ApprovalQueue
    from iris_harness.kernel.governance.approvals.router import ChannelRouter
    from iris_harness.kernel.governance.evaluator.drift_exemptions import DriftExemptionStore

logger = logging.getLogger(__name__)

ENV_FLAG: Final[str] = "IRIS_GOVERNANCE_ENABLED"
_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
# Explicit opt-out values (exp-007 GAP-13: governance is secure-by-default; it is
# only disabled when the operator sets one of these on purpose).
_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def _load_default_persona_policy() -> PersonaPolicyDocument | None:
    """Best-effort load of the packaged persona-policy.yaml.

    Returns ``None`` (degraded mode) when the file is missing — e.g. for
    chat-only installs that don't ship the coding resource bundle, or for
    tests that import the governance package without the coding package.
    A misconfigured *present* file raises ``ValueError`` (loud failure on
    actual misconfiguration; silent degradation only when absent).
    """
    try:
        # absent from the public tree: mypy must pass there as well as here
        from iris_code.resource_paths import (  # type: ignore[import-not-found,unused-ignore]
            default_config_path,
        )
    except ImportError:
        logger.debug(
            "governance: iris_code not importable; PersonaSurface will run " "in degraded mode"
        )
        return None
    policy_path = default_config_path("persona_policy")
    if not policy_path.exists():
        logger.debug(
            "governance: persona-policy.yaml missing at %s; "
            "PersonaSurface will run in degraded mode",
            policy_path,
        )
        return None
    return PersonaPolicyDocument.from_yaml(policy_path)


def build_default_kernel(
    *,
    classifier: DataClassifier | None = None,
    trusted_cloud_for: frozenset[DataClassification] = frozenset(),
    allowed_tools: frozenset[str] | None = None,
    blocked_tools: frozenset[str] = frozenset(),
    confirm_once_tools: frozenset[str] = frozenset(),
    redaction_secrets: VaultStore | None = None,
    evaluator_registry: EvaluatorRegistry | None = None,
    audit_log: AuditLog | None = None,
    loop_detect_embedder: Embedder | None = None,
    flagged_run_writer: FlaggedRunThoughtWriter | None = None,
    cost_store: CostStore | None = None,
    daily_cost_cap_usd: float | None = None,
    cost_user_id: str = "local",
    cost_enforce: bool = True,
    persona_policy: PersonaPolicyDocument | None = None,
    command_sandbox_enabled: bool = True,
    fs_jail_enabled: bool = True,
    network_egress_enabled: bool = True,
    mcp_allowlist_enabled: bool = True,
    mcp_governance_map: dict[str, MCPServerGovernance] | None = None,
    side_effect_ledger: SideEffectLedger | None = None,
    side_effect_ledger_enabled: bool = True,
    side_effect_ledger_db_path: Path | None = None,
    prompt_guard_inbound: Hook | None = None,
    prompt_guard_retrieved: Hook | None = None,
    input_safety: Hook | None = None,
    approval_queue: ApprovalQueue | None = None,
    channel_router: ChannelRouter | None = None,
    owner_pii_mode: OwnerPiiMode = "off",
) -> GovernanceKernel:
    """Construct a kernel with the default plugin set + init_lock().

    The returned kernel is ready to ``fire()`` / ``fire_sync()``.
    Construct once per process; the kernel is not safe to mutate after
    ``init_lock()`` (design §5.3 trust boundary).

    ``evaluator_registry`` is the Phase 3 evaluator. When ``None``, an
    empty registry is created (the evaluator hook returns ``allow``
    until signals are registered); 12.gov-3.2 + later stories register
    the signals.

    ``audit_log`` is the Phase 3 hot-tier sink. When ``None`` the
    kernel writes to the default path (``~/.local/share/iris/audit.db``).
    Pass an explicit instance from tests to point at ``tmp_path``.

    ``loop_detect_embedder`` opts the embedding-based ``loop_detect``
    signal into the default registry (12.gov-3.6). Left ``None`` the
    signal is not registered — keeps the import surface dependency-light
    for tests and embedded callers that don't want the ONNX model load.
    Ignored if ``evaluator_registry`` is supplied (the caller owns the
    registry contents in that case).

    ``cost_store`` + ``daily_cost_cap_usd`` opt the Phase 3 cost limiter
    (12.gov-3.7) in. Both must be set to register ``CostLimiter`` at
    ``PreLLMCall`` and ``cost_budget`` at ``PostStep``. ``cost_user_id``
    defaults to ``"local"`` for the single-user assumption.

    ``side_effect_ledger`` is the side-effect ledger. Passed, it is the pre-existing
    opt-in: every non-read call is recorded after it runs, and a high-risk call (a
    destructive tool, or a write declared ``approval: pinned``) is written before it runs.
    Left ``None`` with ``side_effect_ledger_enabled`` true (the default), the ledger covers
    the high-risk class only, and only opens (the process's shared ``DeferredSideEffectLedger``
    for that database, ``shared_side_effect_ledger``: one handle however many kernels are
    built) when such a call first runs: a plain write or a read leaves no row and touches no
    file, exactly as before issue #73. A kernel with no ledger (``side_effect_ledger_enabled=False``) denies every
    high-risk call rather than run it with no durable record first (``PreToolUseLedgerHook``,
    ``register_side_effect_ledger``).

    ``owner_pii_mode="shadow"`` registers ``OwnerPiiShadowHook`` at ``PreToolUse``,
    ``PreLLMCall`` and ``PreResponse`` (priority 1, first at each): it audits what each
    owner-PII guard would do and changes nothing (ADR-0125 PR 4). ``"off"`` registers
    nothing, so the kernel is exactly what it was before the mode existed.
    """
    kernel = GovernanceKernel(audit_log=audit_log or AuditLog())
    high_risk_only = False
    if side_effect_ledger is None and side_effect_ledger_enabled:
        side_effect_ledger = shared_side_effect_ledger(side_effect_ledger_db_path)
        high_risk_only = True
    # PromptGuardInboundHook (priority 5) runs before DataClassifierHook (10):
    # Phase 6 G1, opt-in, shadow-first. Off unless explicitly built/passed.
    if prompt_guard_inbound is not None:
        kernel.register(prompt_guard_inbound)
    data_classifier = DataClassifierHook(classifier=classifier)
    kernel.register(data_classifier)
    # The same inbound screens, once per turn on the raw user message, so a turn a
    # deterministic handler answers without any model call is screened too
    # (docs/architecture/deterministic-path-parity.md, step a).
    if prompt_guard_inbound is not None:
        kernel.register(prompt_guard_inbound, at=HookPoint.PRE_TURN)
    kernel.register(data_classifier, at=HookPoint.PRE_TURN)
    # Opt-in hazard screen on the user turn (step b2); None unless the operator enabled it.
    if input_safety is not None:
        kernel.register(input_safety)
    # The model-free response check every answer passes, at PRE_RESPONSE — the point had
    # no caller until the curator fired it (deterministic-path parity, step b).
    kernel.register(ResponseSafetyHook())
    # Regex-pack redaction runs always; vault-handle substitution layers on
    # top when a vault is available. Without this, an inline raw key in a
    # prompt could still reach the upstream provider.
    kernel.register(RedactionFilterHook(secrets=redaction_secrets))
    kernel.register(EgressGate(trusted_cloud_for=trusted_cloud_for))
    kernel.register(CredentialBroker(vault=redaction_secrets))
    # PersonaSurface (priority 15) runs before ToolPolicyHook (20) — see
    # design §8.1. Default behavior: load packaged persona-policy.yaml.
    # Missing file → degraded no-op so chat-only installs and tests that
    # don't ship coding resources still construct cleanly.
    effective_persona_policy = (
        persona_policy if persona_policy is not None else _load_default_persona_policy()
    )
    kernel.register(PersonaSurface(policy=effective_persona_policy))
    # The permission contract (plugin-capabilities §4): a plugin calls only its own tools
    # and the ones its manifest's `uses: tools` lists. Always registered; the policy is
    # config the runtime compiles once plugins mount, and plugin calls fail closed without.
    kernel.register(CallerPolicyHook())
    kernel.register(
        ToolPolicyHook(
            allowed_tools=allowed_tools,
            blocked_tools=blocked_tools,
            confirm_once_tools=confirm_once_tools,
        )
    )
    # CommandSandbox runs at priority 25 — after PersonaSurface (15) and
    # ToolPolicyHook (20). Auto-on with PersonaSurface; degraded to no-op
    # when the persona policy is absent. Disable via
    # ``IRIS_GOVERNANCE_COMMAND_SANDBOX=0`` if a caller needs to opt out.
    if command_sandbox_enabled:
        kernel.register(CommandSandbox(policy=effective_persona_policy))
    # FSJail runs at priority 30 — after PersonaSurface (15),
    # ToolPolicyHook (20), and CommandSandbox (25). It is auto-on with the
    # coding persona policy and degrades to no-op when the policy is absent.
    if fs_jail_enabled:
        kernel.register(FSJail(policy=effective_persona_policy))
    # NetworkEgress runs at priority 35 — after FSJail (30). Enforces the
    # per-persona network_egress_domains allowlist for outbound HTTP tools.
    # Default-deny when domains are unset. Disable via
    # ``IRIS_GOVERNANCE_NETWORK_EGRESS=0`` if a caller opts out.
    if network_egress_enabled:
        kernel.register(NetworkEgress(policy=effective_persona_policy))
    # OwnerPiiShadowHook (priority 1) observes before every guard at its three points and
    # returns a bare allow, so every real decision is unchanged (ADR-0125 PR 4). Its egress
    # column shadows NetworkEgress, so it reads the same tools, or none when that is off.
    if owner_pii_mode == "shadow":
        shadow = OwnerPiiShadowHook(
            network_tools=DEFAULT_NETWORK_TOOLS if network_egress_enabled else frozenset()
        )
        for point in SHADOW_POINTS:
            kernel.register(shadow, at=point)
    # MCPAllowlistHook runs at priority 18 — between PersonaSurface (15) and
    # ToolPolicyHook (20). Enforces (persona, server, tool) tuples from the
    # governance blocks in mcp-servers.yaml. Ungoverned servers allow + warn.
    # Disable via ``IRIS_GOVERNANCE_MCP_ALLOWLIST=0``.
    if mcp_allowlist_enabled:
        kernel.register(MCPAllowlistHook(governance_map=mcp_governance_map))
    if cost_store is not None and daily_cost_cap_usd is not None:
        kernel.register(
            CostLimiter(
                store=cost_store,
                daily_cap_usd=daily_cost_cap_usd,
                user_id=cost_user_id,
                enforce=cost_enforce,
            )
        )
    # OutputClassifierHook (priority 10) runs first at PostToolUse: it re-derives the
    # run's data classification from the tool result, so the ledger and the injection
    # guard below — and every later PreLLMCall egress decision — see the label the
    # prompt actually earns rather than the one the question came in with (design §6.5).
    kernel.register(OutputClassifierHook())
    # CapabilityRedactionHook (priority 5, before the classifier): a capability result
    # (plugin-capabilities §4) has the owner's identity literals masked out of its declared
    # text fields before any consumer sees it. Only `capability:` calls; always registered.
    kernel.register(CapabilityRedactionHook())
    # The side-effect ledger: PostToolUseLedgerHook (priority 40) records every non-read
    # call's effect (only the high-risk class's, when the ledger is the default one) so the
    # resume flow can verify it with a probe before re-executing, and PreToolUseLedgerHook
    # (priority 100, last at PreToolUse) writes a high-risk call's row before it runs
    # (issue #73). With no ledger, high-risk calls are denied.
    register_side_effect_ledger(kernel, side_effect_ledger, high_risk_only=high_risk_only)
    # PromptGuardRetrievedHook (priority 45) runs after the ledger (40): Phase 6
    # G2 indirect-injection guard over tool/RAG results. Opt-in, shadow-first.
    if prompt_guard_retrieved is not None:
        kernel.register(prompt_guard_retrieved)
    # McpClientEgressHook (priority 42, after the ledger): a result served to an
    # MCP client (`iris mcp serve`) is withheld when it is secret, or personal and the
    # owner has not declared the client local. A no-op for every other caller.
    kernel.register(McpClientEgressHook())

    # A `require_approval` verdict has to become something a human can actually answer.
    # Both hook sites below accepted these two arguments since the approval queue
    # shipped and neither was ever passed one, so `self._approval_queue` was always
    # None: the enqueue branch never ran, no channel was ever notified, and
    # `approvals.db` was never even created. The halt said "paused for approval" and
    # there was no approval. Defaulted on, with an env kill switch.
    if approval_queue is None and channel_router is None:
        approval_queue, channel_router = _approvals_from_env()
    # ADR-0118: a destructive tool call becomes an itemised approval on the same queue
    # and router, at PreToolUse priority 50 (after every other deny). Always registered:
    # with approvals disabled it has no queue and denies, which is the safe answer.
    kernel.register(
        DestructiveApprovalHook(approval_queue=approval_queue, channel_router=channel_router)
    )
    remote_backend = _remote_evaluator_backend_from_env()
    if remote_backend is not None:
        kernel.register(
            EvaluatorHook(
                backend=remote_backend,
                approval_queue=approval_queue,
                channel_router=channel_router,
            )
        )
        logger.info(
            "governance: evaluator mode=remote (%d remote signals visible)",
            remote_backend.signal_count(),
        )
    else:
        registry = evaluator_registry or _build_default_evaluator_registry(
            loop_detect_embedder=loop_detect_embedder,
            flagged_run_writer=flagged_run_writer,
            cost_store=cost_store,
            daily_cost_cap_usd=daily_cost_cap_usd,
            cost_user_id=cost_user_id,
        )
        if not registry.is_locked:
            registry.init_lock()
        kernel.register(
            EvaluatorHook(
                registry=registry,
                approval_queue=approval_queue,
                channel_router=channel_router,
            )
        )

    kernel.init_lock()
    logger.info(
        "governance: built default kernel; evaluator signals=%d",
        registry.signal_count(),
    )
    return kernel


def kernel_from_env() -> GovernanceKernel | None:
    """Return a governance kernel unless it is **explicitly disabled**.

    Secure-by-default (exp-007 GAP-13): the embedded governance kernel — the
    "mandatory-passage" classifier / egress gate / command-fs-network jails —
    is ON unless the operator opts out by setting ``IRIS_GOVERNANCE_ENABLED``
    to a falsy value (``0``/``false``/``no``/``off``). Unset or truthy → enabled.
    Disabling is logged at WARNING since it turns the safety layer off.
    """
    raw = os.getenv(ENV_FLAG, "").strip().lower()
    if raw in _FALSY:
        logger.warning(
            "governance: %s=%r — DISABLED by operator; egress gate, data "
            "classifier, and command/fs/network jails are OFF.",
            ENV_FLAG,
            raw,
        )
        return None
    if raw and raw not in _TRUTHY:
        logger.warning(
            "governance: %s=%r is not a recognised boolean; treating as ENABLED "
            "(secure-by-default).",
            ENV_FLAG,
            raw,
        )

    trusted = _parse_trusted_cloud_for_from_env()
    allowed_tools = _parse_csv_env("IRIS_GOVERNANCE_ALLOWED_TOOLS")
    blocked_tools = _parse_csv_env("IRIS_GOVERNANCE_BLOCKED_TOOLS") or frozenset()
    confirm_once_tools = _confirm_once_tools_from_env()
    redaction_secrets = _redaction_store_from_env()
    audit_log = _audit_log_from_env()
    loop_detect_embedder = _loop_detect_embedder_from_env()
    flagged_run_writer = (
        _flagged_run_writer_from_env() if loop_detect_embedder is not None else None
    )
    cost_store, daily_cost_cap_usd, cost_user_id, cost_enforce = _cost_limiter_from_env()
    command_sandbox_enabled = _command_sandbox_from_env()
    fs_jail_enabled = _fs_jail_from_env()
    network_egress_enabled = _network_egress_from_env()
    mcp_allowlist_enabled, mcp_governance_map = _mcp_allowlist_from_env()
    side_effect_ledger, side_effect_ledger_enabled = _side_effect_ledger_from_env()
    prompt_guard_inbound, prompt_guard_retrieved = _prompt_guards_from_env()
    input_safety = _input_safety_from_env()
    owner_pii = owner_pii_mode_from_env()
    if owner_pii.problem is not None:
        logger.warning("governance: %s", owner_pii.problem)
    logger.info(
        "governance: %s enabled; trusted_cloud_for=%s allowed_tools=%s "
        "blocked_tools=%s redaction=%s audit=%s loop_detect=%s cost_limiter=%s "
        "command_sandbox=%s fs_jail=%s network_egress=%s mcp_allowlist=%s "
        "side_effect_ledger=%s input_safety=%s owner_pii=%s",
        ENV_FLAG,
        sorted(trusted) if trusted else "{}",
        sorted(allowed_tools) if allowed_tools else "*",
        sorted(blocked_tools) if blocked_tools else "{}",
        "on" if redaction_secrets is not None else "off",
        "on" if audit_log is not None else "default",
        "on" if loop_detect_embedder is not None else "off",
        (
            f"on (cap=${daily_cost_cap_usd:.2f}, "
            f"{'enforcing' if cost_enforce else 'recording only'})"
            if cost_store is not None
            else "off"
        ),
        "on" if command_sandbox_enabled else "off",
        "on" if fs_jail_enabled else "off",
        "on" if network_egress_enabled else "off",
        f"on ({len(mcp_governance_map)} servers)" if mcp_governance_map else "on (degraded)",
        (
            "on (all non-read)"
            if side_effect_ledger is not None
            else "on (high-risk only)" if side_effect_ledger_enabled else "off"
        ),
        "on" if input_safety is not None else "off",
        owner_pii.mode,
    )
    return build_default_kernel(
        trusted_cloud_for=trusted,
        allowed_tools=allowed_tools,
        blocked_tools=blocked_tools,
        confirm_once_tools=confirm_once_tools,
        redaction_secrets=redaction_secrets,
        audit_log=audit_log,
        loop_detect_embedder=loop_detect_embedder,
        flagged_run_writer=flagged_run_writer,
        cost_store=cost_store,
        daily_cost_cap_usd=daily_cost_cap_usd,
        cost_user_id=cost_user_id,
        cost_enforce=cost_enforce,
        command_sandbox_enabled=command_sandbox_enabled,
        fs_jail_enabled=fs_jail_enabled,
        network_egress_enabled=network_egress_enabled,
        mcp_allowlist_enabled=mcp_allowlist_enabled,
        mcp_governance_map=mcp_governance_map,
        side_effect_ledger=side_effect_ledger,
        # Resolved from the environment above: unset is the default (high-risk class only),
        # truthy a ledger for every non-read call, falsy the operator's opt-out.
        side_effect_ledger_enabled=side_effect_ledger_enabled,
        side_effect_ledger_db_path=_side_effect_ledger_db_path_from_env(),
        prompt_guard_inbound=prompt_guard_inbound,
        prompt_guard_retrieved=prompt_guard_retrieved,
        input_safety=input_safety,
        owner_pii_mode=owner_pii.mode,
    )


def _parse_trusted_cloud_for_from_env() -> frozenset[DataClassification]:
    """Parse comma-separated ``IRIS_GOVERNANCE_TRUSTED_CLOUD_FOR``.

    Only ``personal`` and ``internal`` are accepted (``secret`` is
    permanently rejected by EgressGate; ``public`` is meaningless to
    list). Unknown values are dropped with a warning.
    """
    raw = os.getenv("IRIS_GOVERNANCE_TRUSTED_CLOUD_FOR", "").strip()
    if not raw:
        return frozenset()

    allowed: set[DataClassification] = set()
    valid_values: tuple[DataClassification, ...] = ("personal", "internal")
    for token in raw.split(","):
        cleaned = token.strip().lower()
        if cleaned in valid_values:
            allowed.add(cleaned)
        elif cleaned:
            logger.warning("governance: ignoring unknown trusted_cloud_for value %r", cleaned)
    return frozenset(allowed)


def _confirm_once_tools_from_env() -> frozenset[str]:
    """``IRIS_GOVERNANCE_CONFIRM_ONCE_TOOLS`` (CSV): an operator's override by tool name.

    ADR-0110: which tools ask first is each tool's own declaration (``confirm: once``
    in the plugin manifest, or the core ``ToolSpec``), carried to the policy by the
    loop. This set only ADDS names on top; unset or ``none`` adds nothing, and the
    core's default names no plugin tool.
    """
    raw = os.getenv("IRIS_GOVERNANCE_CONFIRM_ONCE_TOOLS", "").strip()
    if not raw or raw.lower() == "none":
        return frozenset()
    return _parse_csv_env("IRIS_GOVERNANCE_CONFIRM_ONCE_TOOLS") or frozenset()


def _parse_csv_env(name: str) -> frozenset[str] | None:
    """Parse a comma-separated string env var into an exact-name frozenset."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    values = frozenset(token.strip() for token in raw.split(",") if token.strip())
    return values or None


def _build_default_evaluator_registry(
    *,
    loop_detect_embedder: Embedder | None = None,
    flagged_run_writer: FlaggedRunThoughtWriter | None = None,
    cost_store: CostStore | None = None,
    daily_cost_cap_usd: float | None = None,
    cost_user_id: str = "local",
) -> EvaluatorRegistry:
    """Register the four Phase 3 cheap signals + optional add-ons.

    Optional add-ons:

    - ``loop_detect`` + ``goal_drift`` (12.gov-3.6) register when
      ``loop_detect_embedder`` is supplied — both reuse the same
      embedder, so one opt-in covers both semantic signals.
    - ``cost_budget`` (12.gov-3.7) registers when both ``cost_store`` and
      ``daily_cost_cap_usd`` are supplied — keeps the cost ledger DB
      open only for callers that opt in.
    """
    registry = EvaluatorRegistry()
    registry.register(ClassificationViolationSignal())
    registry.register(StepCapSignal())
    registry.register(ActionRepeatSignal())
    registry.register(ToolFailureStreakSignal())
    if loop_detect_embedder is not None:
        registry.register(
            LoopDetectSignal(
                embedder=loop_detect_embedder,
                flagged_writer=flagged_run_writer,
            )
        )
        # Same embedder powers ``goal_drift`` — no extra opt-in flag needed.
        registry.register(
            GoalDriftSignal(
                embedder=loop_detect_embedder,
                flagged_writer=flagged_run_writer,
                exemptions=_drift_exemption_store(),
                **_goal_drift_overrides(),
            )
        )
    if cost_store is not None and daily_cost_cap_usd is not None:
        registry.register(
            CostBudgetSignal(
                store=cost_store,
                daily_cap_usd=daily_cost_cap_usd,
                user_id=cost_user_id,
            )
        )
    return registry


def _audit_log_from_env() -> AuditLog | None:
    """Construct an AuditLog from ``IRIS_GOVERNANCE_AUDIT_DB_PATH`` if set.

    Returns ``None`` to let ``build_default_kernel`` use the default
    location. Construction failures degrade to ``None`` with a WARN —
    the kernel still works, just without persistence.
    """
    raw = os.getenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", "").strip()
    if not raw:
        return None
    try:
        return AuditLog(db_path=Path(raw))
    except Exception as exc:  # noqa: BLE001
        logger.warning("governance: audit log disabled because init failed: %s", exc)
        return None


def _approvals_from_env() -> tuple[ApprovalQueue | None, ChannelRouter | None]:
    """The approval queue + delivery router, unless switched off.

    On by default, because the alternative is what shipped: a `require_approval`
    verdict that enqueued nothing and told nobody.
    ``IRIS_GOVERNANCE_APPROVALS_ENABLED=0`` restores that, for a caller that wants a
    halt to be purely advisory.

    ``force_interactive=False`` is the important argument here. The router would
    otherwise pick ``CLIChannel`` whenever stdin is a TTY, and ``CLIChannel.notify``
    blocks on ``input()`` — inside the PostStep hook, mid-turn. There is nothing for
    that prompt to unblock: the evaluator halt is not a gate the loop waits on, it has
    already broken out and the turn is about to answer. So a Y/n there would hang the
    owner's REPL to collect an answer arriving too late to matter. CLI delivery is the
    Approval ID now printed in the halt message, plus ``iris approvals list``.
    """
    raw = os.getenv("IRIS_GOVERNANCE_APPROVALS_ENABLED", "1").strip().lower()
    if raw not in _TRUTHY:
        logger.info("governance: approval queue disabled by env; halts are advisory only")
        return (None, None)
    try:
        from iris_harness.kernel.governance.approvals import (
            ApprovalQueue as _Queue,
        )
        from iris_harness.kernel.governance.approvals.router import (
            ChannelRouter as _Router,
        )

        queue = _Queue(audit_log=_audit_log_from_env() or AuditLog())
        return (queue, _Router(queue=queue, force_interactive=False))
    except Exception as exc:  # noqa: BLE001 — a kernel without approvals beats no kernel
        logger.warning("governance: could not build the approval queue (%s); halts advisory", exc)
        return (None, None)


def _drift_exemption_store() -> DriftExemptionStore | None:
    """The store of thought shapes the person has approved, or None if it cannot open.

    A signal that cannot read its exemptions still works — it just stops remembering,
    which fails towards asking rather than towards allowing.
    """
    try:
        from iris_harness.kernel.governance.evaluator.drift_exemptions import (
            DriftExemptionStore,
        )

        return DriftExemptionStore()
    except Exception:  # governance must build even without the store
        logger.warning("governance: goal_drift exemption store unavailable", exc_info=True)
        return None


def _goal_drift_overrides() -> dict[str, float]:
    """``max_distance`` from ``IRIS_GOVERNANCE_GOAL_DRIFT_MAX_DISTANCE``, if set.

    The threshold was a constructor default no caller passed, so the only way to
    retune a signal that halts real turns was to edit the signal. An unparseable or
    out-of-range value is ignored with a warning rather than failing the kernel
    build: a bad tuning env var must not be able to take the harness down.
    """
    raw = os.getenv("IRIS_GOVERNANCE_GOAL_DRIFT_MAX_DISTANCE", "").strip()
    if not raw:
        return {}
    try:
        value = float(raw)
    except ValueError:
        logger.warning("governance: ignoring unparseable goal_drift max_distance %r", raw)
        return {}
    if not 0.0 <= value <= 2.0:
        logger.warning("governance: ignoring out-of-range goal_drift max_distance %r", raw)
        return {}
    return {"max_distance": value}


def _loop_detect_embedder_from_env() -> Embedder | None:
    """Lazy-construct the default embedder iff ``IRIS_GOVERNANCE_LOOP_DETECT_ENABLED`` is truthy.

    Returns ``None`` (signal not registered) on:
    - the flag being absent or falsy, or
    - ChromaDB import failure (caller gets a working kernel without the signal).
    """
    raw = os.getenv("IRIS_GOVERNANCE_LOOP_DETECT_ENABLED", "0").strip().lower()
    if raw not in _TRUTHY:
        return None
    try:
        from iris_harness.kernel.governance.evaluator.embeddings import (
            DefaultEmbedder,
        )

        return DefaultEmbedder()
    except Exception as exc:  # noqa: BLE001
        logger.warning("governance: loop_detect disabled because embedder init failed: %s", exc)
        return None


def _flagged_run_writer_from_env() -> FlaggedRunThoughtWriter | None:
    """Construct the optional flagged-run writer when semantic signals are on."""
    try:
        from iris_harness.kernel.governance.evaluator.flagged_runs import (
            ChromaFlaggedRunThoughtWriter,
        )

        writer = ChromaFlaggedRunThoughtWriter()
        return writer if writer.is_ready else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("governance: flagged-run persistence disabled because init failed: %s", exc)
        return None


def _remote_evaluator_backend_from_env() -> Any | None:
    """Return a RemoteEvaluatorClient iff IRIS_GOVERNANCE_EVALUATOR_MODE=remote.

    URL comes from ``IRIS_GOVERNANCE_EVALUATOR_URL`` (default
    ``http://127.0.0.1:8090``); timeout from
    ``IRIS_GOVERNANCE_EVALUATOR_TIMEOUT_S`` (default 5s). Returns
    ``None`` on construction failure so the kernel falls back to the
    in-process registry rather than crashing at startup.
    """
    mode = os.getenv("IRIS_GOVERNANCE_EVALUATOR_MODE", "local").strip().lower()
    if mode != "remote":
        return None

    base_url = os.getenv("IRIS_GOVERNANCE_EVALUATOR_URL", "http://127.0.0.1:8090").strip()
    timeout_raw = os.getenv("IRIS_GOVERNANCE_EVALUATOR_TIMEOUT_S", "5.0").strip()
    try:
        timeout_s = float(timeout_raw)
    except ValueError:
        timeout_s = 5.0
    if timeout_s <= 0:
        timeout_s = 5.0

    try:
        from iris_harness.kernel.governance.evaluator.client import (
            RemoteEvaluatorClient,
        )

        return RemoteEvaluatorClient(base_url=base_url, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "governance: remote evaluator client init failed (%s); falling back to local",
            exc,
        )
        return None


def _cost_limiter_from_env() -> tuple[CostStore | None, float | None, str, bool]:
    """Build the cost ledger + cap + user_id + enforce quad from env vars.

    Two independent switches, because recording and enforcing are different
    decisions:

    * ``IRIS_GOVERNANCE_COST_LIMITER_ENABLED=1`` starts the ledger. Nothing
      else writes it, so with this off every spend figure is a truthful 0.00
      that means "nobody is counting".
    * ``IRIS_GOVERNANCE_COST_ENFORCE`` decides whether passing the cap refuses
      the call. It defaults to ``1``, so turning recording on does not quietly
      remove a guard an existing install already had; set it to ``0`` to watch
      spend for a while before choosing a cap.

    Cap defaults to ``DEFAULT_DAILY_CAP_USD`` ($5.00); override via
    ``IRIS_GOVERNANCE_DAILY_COST_CAP_USD``. The single-user assumption means
    ``IRIS_USER_ID`` defaults to ``"local"``.
    """
    user_id = (os.getenv("IRIS_USER_ID", "") or "local").strip() or "local"
    enforce = os.getenv("IRIS_GOVERNANCE_COST_ENFORCE", "1").strip().lower() in _TRUTHY

    raw = os.getenv("IRIS_GOVERNANCE_COST_LIMITER_ENABLED", "0").strip().lower()
    if raw not in _TRUTHY:
        return None, None, user_id, enforce

    from iris_harness.kernel.governance.cost import DEFAULT_DAILY_CAP_USD

    cap_raw = os.getenv("IRIS_GOVERNANCE_DAILY_COST_CAP_USD", "").strip()
    try:
        cap = float(cap_raw) if cap_raw else DEFAULT_DAILY_CAP_USD
    except ValueError:
        logger.warning(
            "governance: IRIS_GOVERNANCE_DAILY_COST_CAP_USD=%r not a float; using $%.2f",
            cap_raw,
            DEFAULT_DAILY_CAP_USD,
        )
        cap = DEFAULT_DAILY_CAP_USD

    db_raw = os.getenv("IRIS_GOVERNANCE_COST_LEDGER_DB_PATH", "").strip()
    db_path = Path(db_raw) if db_raw else None
    try:
        store = CostStore(db_path=db_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("governance: cost_limiter disabled because store init failed: %s", exc)
        return None, None, user_id, enforce
    return store, cap, user_id, enforce


def _prompt_guards_from_env() -> tuple[Hook | None, Hook | None]:
    """Build the Phase 6 prompt-guard hooks (G1 inbound + G2 retrieved) when opted in.

    Returns ``(inbound, retrieved)``. OFF by default (heavy ML model + still
    tuning). Set ``IRIS_GOVERNANCE_PROMPT_GUARD`` truthy to enable. Both share a
    single detector (one lazy model load) and honor the packaged
    ``config/governance/threat-detection.yaml`` ``mode`` + per-surface
    ``on_detect``. Any build failure degrades to ``(None, None)`` with a warning; a per-surface
    ``enabled: false`` drops just that hook.
    """
    raw = os.getenv("IRIS_GOVERNANCE_PROMPT_GUARD", "").strip().lower()
    if raw not in _TRUTHY:
        return None, None
    try:
        from iris_harness.kernel.governance.plugins.prompt_guard import (
            PromptGuardInboundHook,
            PromptGuardRetrievedHook,
        )
        from iris_harness.kernel.governance.threat import (
            ThreatDetectionConfig,
            build_threat_detector,
        )

        config_path = _resolved_config_path("governance", "threat-detection.yaml")
        override = os.getenv("IRIS_THREAT_DETECTION_CONFIG")
        config = ThreatDetectionConfig.from_yaml(Path(override) if override else config_path)
        if not config.enabled:
            return None, None
        detector = build_threat_detector(config)
        shadow = config.mode == "shadow"
        inbound: Hook | None = None
        retrieved: Hook | None = None
        if config.inbound.enabled:
            inbound = PromptGuardInboundHook(
                classifier=detector.prompt_guard,
                on_detect=config.inbound.on_detect,
                shadow=shadow,
            )
        if config.retrieved.enabled:
            retrieved = PromptGuardRetrievedHook(
                classifier=detector.prompt_guard,
                on_detect=config.retrieved.on_detect,
                shadow=shadow,
            )
        return inbound, retrieved
    except Exception:  # opt-in guard; never break kernel construction
        # The operator asked for the guards, so losing them is not a detail: an override
        # config that no longer loads (a key a release removed, like ``scan_tools``) must
        # not turn both off quietly.
        logger.warning(
            "governance: prompt guards requested (IRIS_GOVERNANCE_PROMPT_GUARD) but failed "
            "to build; BOTH are OFF",
            exc_info=True,
        )
        return None, None


def _input_safety_from_env() -> Hook | None:
    """Build the opt-in input safety screen (``PRE_TURN``) when opted in.

    OFF by default. Set ``IRIS_GOVERNANCE_INPUT_SAFETY`` truthy to install it; the
    ``input_safety`` section of ``threat-detection.yaml`` must also be enabled. It runs
    the output guard's model on the user's message with the output guard's categories,
    and follows the config's ``mode`` (shadow logs, enforce refuses). A build failure
    leaves it off and says so at WARNING: the operator asked for this screen, so its
    absence must not look like it is working (deterministic-path parity, step b2).
    """
    raw = os.getenv("IRIS_GOVERNANCE_INPUT_SAFETY", "").strip().lower()
    if raw not in _TRUTHY:
        return None
    try:
        from iris_harness.kernel.governance.plugins.input_safety import InputSafetyHook
        from iris_harness.kernel.governance.threat import (
            ThreatDetectionConfig,
            build_threat_detector,
        )

        config_path = _resolved_config_path("governance", "threat-detection.yaml")
        override = os.getenv("IRIS_THREAT_DETECTION_CONFIG")
        config = ThreatDetectionConfig.from_yaml(Path(override) if override else config_path)
        if not (config.enabled and config.input_safety.enabled):
            logger.warning(
                "governance: IRIS_GOVERNANCE_INPUT_SAFETY is on but threat-detection.yaml "
                "disables it (enabled / input_safety.enabled); input safety screen is OFF"
            )
            return None
        detector = build_threat_detector(config)
        enforce, log_only = config.input_safety_categories()
        return InputSafetyHook(
            classifier=detector.output_guard,
            enforce=enforce,
            log_only=log_only,
            shadow=config.mode == "shadow",
        )
    except Exception:  # opt-in guard; never break kernel construction
        logger.warning(
            "governance: input safety screen requested but failed to build; it is OFF",
            exc_info=True,
        )
        return None


def _command_sandbox_from_env() -> bool:
    """Resolve the CommandSandbox enable flag from env.

    Auto-on (mirrors PersonaSurface registration). Set
    ``IRIS_GOVERNANCE_COMMAND_SANDBOX=0`` (or ``false``/``no``/``off``)
    to disable without dropping the rest of the kernel.
    """
    raw = os.getenv("IRIS_GOVERNANCE_COMMAND_SANDBOX", "").strip().lower()
    if not raw:
        return True
    if raw in _TRUTHY:
        return True
    return False


def _fs_jail_from_env() -> bool:
    """Resolve the FSJail enable flag from env.

    Auto-on (mirrors PersonaSurface registration). Set
    ``IRIS_GOVERNANCE_FS_JAIL=0`` (or ``false``/``no``/``off``) to
    disable without dropping the rest of the kernel.
    """
    raw = os.getenv("IRIS_GOVERNANCE_FS_JAIL", "").strip().lower()
    if not raw:
        return True
    if raw in _TRUTHY:
        return True
    return False


def _network_egress_from_env() -> bool:
    """Resolve the NetworkEgress enable flag from env.

    Auto-on (mirrors PersonaSurface registration). Set
    ``IRIS_GOVERNANCE_NETWORK_EGRESS=0`` (or ``false``/``no``/``off``)
    to disable without dropping the rest of the kernel.
    """
    raw = os.getenv("IRIS_GOVERNANCE_NETWORK_EGRESS", "").strip().lower()
    if not raw:
        return True
    if raw in _TRUTHY:
        return True
    return False


def _mcp_allowlist_from_env() -> tuple[bool, dict[str, MCPServerGovernance] | None]:
    """Resolve the MCPAllowlistHook enable flag and governance map from env.

    ``IRIS_GOVERNANCE_MCP_ALLOWLIST=0`` disables the hook entirely.
    ``IRIS_MCP_SERVERS_CONFIG_PATH`` points to the mcp-servers.yaml;
    when absent, defaults to ``config/coding-agent/mcp-servers.yaml``
    relative to cwd.

    Returns ``(enabled, governance_map)``. ``governance_map`` is ``None``
    when the config file is missing or fails to parse — the hook runs in
    degraded mode (allow + warn).
    """
    raw = os.getenv("IRIS_GOVERNANCE_MCP_ALLOWLIST", "").strip().lower()
    enabled = (not raw) or raw in _TRUTHY
    if not enabled:
        return False, None

    config_path_raw = os.getenv("IRIS_MCP_SERVERS_CONFIG_PATH", "").strip()
    config_path = (
        Path(config_path_raw)
        if config_path_raw
        else _resolved_config_path("coding-agent", "mcp-servers.yaml")
    )
    if not config_path.exists():
        logger.debug(
            "governance: mcp-servers.yaml not found at %s; "
            "MCPAllowlistHook will run in degraded mode",
            config_path,
        )
        return True, None

    try:
        import yaml

        raw_doc = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        governance_map = _parse_mcp_governance_map(raw_doc)
        return True, governance_map
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "governance: failed to parse mcp governance map from %s: %s; "
            "MCPAllowlistHook will run in degraded mode",
            config_path,
            exc,
        )
        return True, None


def _parse_mcp_governance_map(
    raw_doc: dict[str, Any],
) -> dict[str, MCPServerGovernance] | None:
    """Extract ``{server_name: MCPServerGovernance}`` from a parsed YAML doc."""
    servers_raw = raw_doc.get("servers")
    if not isinstance(servers_raw, list):
        return None
    result: dict[str, MCPServerGovernance] = {}
    for entry in servers_raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        gov_raw = entry.get("governance")
        if not isinstance(gov_raw, dict):
            continue
        try:
            result[name] = MCPServerGovernance.model_validate(gov_raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "governance: invalid mcp governance block for server %r: %s",
                name,
                exc,
            )
    return result or None


def _redaction_store_from_env() -> VaultStore | None:
    """Build a vault store for redaction if enabled and available."""
    raw = os.getenv("IRIS_GOVERNANCE_REDACTION_ENABLED", "0").strip().lower()
    if raw not in _TRUTHY:
        return None

    db_raw = os.getenv("IRIS_GOVERNANCE_VAULT_DB_PATH", "").strip()
    db_path = Path(db_raw) if db_raw else None
    try:
        return VaultStore(db_path=db_path)
    except Exception as exc:  # noqa: BLE001 — no vault means redaction off; logged
        logger.warning("governance: redaction disabled because vault init failed: %s", exc)
        return None


def register_side_effect_ledger(
    kernel: GovernanceKernel, ledger: SideEffectLedger | None, *, high_risk_only: bool = False
) -> None:
    """Register the side-effect ledger's hooks on ``kernel`` (before ``init_lock``).

    ``PreToolUseLedgerHook`` is always registered: it writes a high-risk call's row before
    the call runs, and with ``ledger=None`` it denies that call (fail closed) -- as the
    runner does a high-risk call no hook recorded. ``PostToolUseLedgerHook`` records (or
    settles) rows when there is a ledger to write: every non-read call's, or with
    ``high_risk_only`` only the high-risk class's. ``build_default_kernel`` wires the
    ledger through here; a kernel assembled by hand that runs high-risk tools calls it too.
    """
    if ledger is not None:
        kernel.register(PostToolUseLedgerHook(ledger=ledger, high_risk_only=high_risk_only))
    kernel.register(PreToolUseLedgerHook(ledger=ledger))


def _open_side_effect_ledger(db_path: Path | None = None) -> SideEffectLedger | None:
    """The ledger at ``db_path`` (default: ``<governance data dir>/side_effects.db``), or
    ``None`` (logged) when it will not open: high-risk calls then fail closed, never run
    unrecorded."""
    try:
        # The process's one ledger for this database: opened here, once, not per kernel.
        ledger = shared_side_effect_ledger(db_path)
        ledger.open()
        return ledger
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "governance: side_effect_ledger could not open (%s); high-risk tool calls "
            "will be denied until it can",
            type(exc).__name__,
        )
        return None


def _side_effect_ledger_from_env() -> tuple[SideEffectLedger | None, bool]:
    """``(ledger, enabled)`` from ``IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER``.

    * unset: ``(None, True)`` -- the default: ``build_default_kernel`` builds a deferred
      ledger for the high-risk class only; plain writes and reads are not recorded.
    * truthy: the ledger, opened now, recording every non-read call (the original meaning
      of the flag). DB path from ``IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH`` (also the
      deferred ledger's), else ``<governance data dir>/side_effects.db``. If it will not
      open: ``(None, False)``.
    * falsy (``0``/``false``/``no``/``off``): ``(None, False)``. The kernel then denies every
      high-risk call (a destructive tool, or a pinned write) instead of running it with no
      durable record: the ledger is what makes those calls safe to attempt, so opting out
      also turns them off.
    """
    raw = os.getenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", "").strip().lower()
    if raw in _FALSY:
        logger.warning(
            "governance: IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=%r -- the side-effect ledger is "
            "OFF; destructive tools and pinned writes will be denied.",
            raw,
        )
        return None, False
    if raw not in _TRUTHY:
        if raw:
            logger.warning(
                "governance: IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=%r is not recognised (on: 1/true/"
                "yes/on, off: 0/false/no/off); applying the default, the side-effect ledger for "
                "high-risk calls only.",
                raw,
            )
        return None, True
    ledger = _open_side_effect_ledger(_side_effect_ledger_db_path_from_env())
    return ledger, ledger is not None


def _side_effect_ledger_db_path_from_env() -> Path | None:
    """``IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH``, or ``None`` for the default path."""
    db_raw = os.getenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", "").strip()
    return Path(db_raw).expanduser() if db_raw else None
