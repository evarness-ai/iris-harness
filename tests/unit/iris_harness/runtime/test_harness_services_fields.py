"""``HarnessServices`` field freeze.

``HarnessServices`` is the typed object a plugin receives at mount time. It has grown
field by field, each addition locally reasonable, into a service locator: a plugin
that wants one thing ends up holding a reference to half of the runtime.

The rule: no new field without proving it cannot be expressed through an existing
``PluginAPI``/SDK abstraction (a tool, a capability, an event) -- stated in the PR
description. This test is the forcing function that makes the rule visible in review
instead of assumed, mirroring how ``tests/unit/test_stable_tier.py`` pins every stable
SDK name. Adding a field means touching ``_EXPECTED_FIELDS`` in the same PR.
"""

from __future__ import annotations

import dataclasses

from iris_harness.runtime.harness_services import HarnessServices

# Declaration order, name -> "required" (no default) or "optional" (has a default).
# A field moving between the two is also a change this test should catch.
_EXPECTED_FIELDS: dict[str, str] = {
    "config_dir": "required",
    "data_dir": "required",
    "tier_router": "required",
    "agent_executor": "required",
    "heartbeats": "required",
    "channels": "required",
    "deterministic_reply": "required",
    "events": "optional",
    "submit_activity": "optional",
    "react_handler": "optional",
    "react_stream_handler": "optional",
    "current_query": "optional",
    "current_session_id": "optional",
    "deliver_in_chat": "optional",
    "embed": "optional",
    "lessons": "optional",
    "continuations": "optional",
    "classify_intent": "optional",
    "skill_registry": "optional",
    "conversation_in_flight": "optional",
    "default_channel": "optional",
    "heartbeat_diagnostics": "optional",
    "tools": "optional",
}


def _has_default(f: dataclasses.Field) -> bool:  # type: ignore[type-arg]
    return f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING


def test_harness_services_field_set_is_pinned() -> None:
    fields = dataclasses.fields(HarnessServices)
    actual = {f.name: ("optional" if _has_default(f) else "required") for f in fields}
    assert actual == _EXPECTED_FIELDS, (
        "HarnessServices's field set changed. Per ADR-0129 (internal): no new field "
        "without proving it cannot be expressed through an existing PluginAPI/SDK "
        "abstraction (a tool, a capability, an event) -- say what was tried first in "
        "the PR description, then update _EXPECTED_FIELDS above.\n"
        f"expected: {sorted(_EXPECTED_FIELDS)}\n"
        f"actual:   {sorted(actual)}"
    )


def test_harness_services_field_count_has_not_grown_silently() -> None:
    # A second, blunter assertion: even a same-name swap that changes the count would
    # be caught above, but this makes the growth itself the headline on failure.
    assert len(dataclasses.fields(HarnessServices)) == len(_EXPECTED_FIELDS)
