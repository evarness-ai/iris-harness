"""CI gate: hook plugins cannot be added after kernel ``init_lock()``.

Design §5.3 (hook plugin trust boundary): hooks observe every prompt,
every tool call, and every credential handle resolution — registering
one is equivalent to root in the governance plane. The kernel locks
its registry at init time; any later attempt to ``register()`` must
raise ``HookRegistrationLockedError``. Skills, MCP servers, extensions,
and runtime code cannot smuggle a hook into a live process.

There is a unit-level version of this assertion in
``tests/unit/test_governance/test_kernel_basics.py``. This file is the
*security-tier* sibling: it pins the same invariant against the
production wiring (``build_default_kernel``) so a regression in the
default plugin set or a refactor of ``GovernanceKernel`` fails CI
even if the bare-kernel unit test is deleted or relaxed.
"""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
    HookRegistrationLockedError,
    build_default_kernel,
)


class _NoOpHook:
    name: str = "test_noop"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 9999

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="noop")


def test_bare_kernel_rejects_registration_after_init_lock() -> None:
    """A raw ``GovernanceKernel`` must refuse ``register()`` after lock."""
    kernel = GovernanceKernel()
    kernel.init_lock()
    with pytest.raises(HookRegistrationLockedError):
        kernel.register(_NoOpHook())


def test_default_wiring_rejects_runtime_registration() -> None:
    """The production wiring path must also refuse runtime registration.

    ``build_default_kernel`` registers the full v1 plugin set and calls
    ``init_lock()`` before returning. Any path that re-acquires the
    kernel and tries to add a hook (a skill, an extension, a
    well-meaning startup hook) must fail.
    """
    kernel = build_default_kernel()
    assert kernel.is_locked, "build_default_kernel must lock the registry"
    with pytest.raises(HookRegistrationLockedError):
        kernel.register(_NoOpHook())


def test_lock_is_idempotent_and_does_not_unlock() -> None:
    """Calling ``init_lock()`` a second time must not re-open the gate."""
    kernel = GovernanceKernel()
    kernel.init_lock()
    kernel.init_lock()  # second call is a no-op, not an unlock
    with pytest.raises(HookRegistrationLockedError):
        kernel.register(_NoOpHook())
