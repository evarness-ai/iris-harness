"""Pending actions: blockers the owner can clear, raised by a plugin's provider.

The mechanism is the harness's; the policy is yours. Implement
`PendingActionProvider` (a ``source_kind``, the `DesiredAction` set your state calls
for right now, and ``invoke`` for a safe action; `ChoiceActionProvider` adds
``invoke_choice`` when you raise choice cards), `register_provider` it from
``setup()``, and `reconcile` raises
the new ones as tasks and completes the ones whose blocker is gone -- idempotent, so
run it as often as your state changes. `provider_for(source_kind)` finds a registered
provider; `PendingActionsSummary` is what one reconcile did.
"""

from __future__ import annotations

from iris_harness.services.tasks.pending_actions import (
    ChoiceActionProvider,
    DesiredAction,
    PendingActionProvider,
    PendingActionsSummary,
    provider_for,
    reconcile,
    register_provider,
)

__all__ = [
    "ChoiceActionProvider",
    "DesiredAction",
    "PendingActionProvider",
    "PendingActionsSummary",
    "provider_for",
    "reconcile",
    "register_provider",
]
