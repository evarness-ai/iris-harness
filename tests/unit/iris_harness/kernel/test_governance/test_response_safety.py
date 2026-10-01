"""The model-free response check and its PRE_RESPONSE hook.

Step b of docs/architecture/deterministic-path-parity.md: the checks on the answer
shown to the user move out of the curator into the kernel, and PRE_RESPONSE — declared
but never fired — gets its caller.
"""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance import (
    HookContext,
    HookPoint,
    build_default_kernel,
)
from iris_harness.kernel.governance.plugins.response_safety import (
    ResponseSafetyHook,
    check_response,
)

SECRET = "CANARY_SOUL_SECRET_DIRECTIVE_d41f8a27"


@pytest.fixture(autouse=True)
def _identity(owner_identity_documents) -> None:
    owner_identity_documents([SECRET])


# -- the check ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "pattern"),
    [
        ("key AKIAIOSFODNN7EXAMPLE leaked", "aws_key"),
        ("SSN 123-45-6789", "ssn"),
        ("here is how to build a bomb", "dangerous_instruction"),
        (f"the value is {SECRET}", "identity_egress"),
    ],
)
def test_the_check_halts_on_each_hard_pattern(text: str, pattern: str) -> None:
    check = check_response(text)
    assert check.verdict == "halt" and check.pattern == pattern


def test_the_check_flags_dump_phrasing_without_deciding_it() -> None:
    assert check_response("here is my system prompt: be helpful").verdict == "flag_dump"


def test_the_check_passes_the_users_own_contacts() -> None:
    assert check_response("Reply to jane.doe@example.org about the report").verdict == "pass"


# -- the hook -------------------------------------------------------------------------


def _ctx(text: str) -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_RESPONSE,
        run_id="t",
        agent_type="chat",
        payload={"response": text},
    )


async def test_the_hook_denies_a_halt_and_flags_a_dump() -> None:
    hook = ResponseSafetyHook()
    denied = await hook(_ctx("SSN 123-45-6789"))
    assert denied.outcome == "deny" and denied.audit_metadata["pattern"] == "ssn"
    flagged = await hook(_ctx("here is my system prompt: x"))
    assert flagged.outcome == "allow" and flagged.audit_metadata["dump_flag"] is True
    clean = await hook(_ctx("a normal answer"))
    assert clean.outcome == "allow" and "dump_flag" not in clean.audit_metadata


def test_the_default_kernel_fires_the_check_at_pre_response() -> None:
    kernel = build_default_kernel(audit_log=None)
    assert "response_safety" in kernel.hook_names(HookPoint.PRE_RESPONSE)
