"""Which owner-identity kinds a capability consumer may see unmasked, supplied from above.

A capability result reaches its consumer with the owner's personal identifiers as
pseudonyms (ADR-0125, PR 3). A consumer's manifest may grant itself kinds back, per
capability it uses (``capabilities: uses: - mail.read: {unmask: [name, address]}``). The
plugin host, which sits above the kernel, is what knows the manifests, so the kernel asks:
the runtime registers a policy once plugins have mounted, and ``CapabilityRedactionHook``
reads it on every capability result -- for the caller the harness stamped on the call, so a
consumer can never claim another's grant.

Fail closed: with no policy registered, nothing is granted and every pseudonym stays.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from iris_harness.foundation.process_state import track_globals

#: ``policy(caller, capability) -> kinds`` the caller may see unmasked in its results.
UnmaskPolicy = Callable[[str, str], frozenset[str]]

_lock = threading.Lock()
_policy: UnmaskPolicy | None = None


def register_unmask_policy(policy: UnmaskPolicy | None) -> None:
    """Install (or, with ``None``, remove) the policy capability masking reads."""
    global _policy
    with _lock:
        _policy = policy


def unmask_grants(caller: str, capability: str) -> frozenset[str]:
    """The kinds ``caller`` may see unmasked in ``capability``'s results; none by default."""
    with _lock:
        policy = _policy
    if policy is None or not caller or not capability:
        return frozenset()
    return frozenset(policy(caller, capability))


__all__ = ["UnmaskPolicy", "register_unmask_policy", "unmask_grants"]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_policy")
