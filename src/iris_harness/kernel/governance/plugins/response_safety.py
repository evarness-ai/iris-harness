"""The model-free response check every answer passes, and its ``PreResponse`` hook.

Every answer IRIS gives — generated or deterministic — must pass the same
model-free guards (docs/architecture/deterministic-path-parity.md, step b). These
checks used to live inside ``ResponseCurator._judge_safety``, which only generated
answers reach, and the kernel's ``PRE_RESPONSE`` hook point had no caller. The
checks are a governance concern and need nothing above ``foundation``, so they
live here: ``check_response`` is the one implementation, ``ResponseSafetyHook``
fires it at ``PRE_RESPONSE`` (so the kernel writes the audit row), and the curator
calls ``check_response`` directly only when governance is disabled, so response
safety never weakens with it.

What is checked, all deterministic:

- hard patterns: an AWS access key, an SSN, a short list of dangerous instructions;
- identity egress: a secret-shaped literal from SOUL / USER / AGENTS (credentials,
  canaries), read through the kernel's identity-text seam;
- internal-architecture disclosure (``disclosure.is_architecture_disclosure``).

Dump phrasing ("here is my system prompt: ...") is only *flagged* here. Telling a
dump from a self-description is an intent question the curator's semantic leak
judge answers, failing closed without one; this check does not call a model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from iris_harness.kernel.governance.disclosure import is_architecture_disclosure
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.identity_redaction import (
    owner_identity,
    reset_owner_identity,
)

logger = logging.getLogger(__name__)

# Hard, deterministic halts on genuine secrets / harmful instructions in the response
# shown to the user. Deliberately NOT a generic DLP set: an email address is the
# user's OWN data here — the email, calendar, planner and finance agents surface
# contacts' addresses as their core function, so a bare email match must not halt.
# Exfiltration to external services is governed at the egress layer, not by hiding
# the user's own data from the user.
SAFETY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "dangerous_instruction",
        re.compile(
            r"\b(build a bomb|make meth|weaponize|bypass malware detection)\b",
            re.IGNORECASE,
        ),
    ),
)

# SOFT signal: the act of *presenting* the system prompt / identity docs, not a bare
# mention. A match is a flag for the leak judge, never a verdict on its own.
DUMP_PATTERN: re.Pattern[str] = re.compile(
    r"(here(?:'s| is)\s+(?:my|the)\s+(?:full\s+|complete\s+|entire\s+)?"
    r"(?:system prompt|developer message|soul\.md)"
    r"|(?:system prompt|soul\.md|developer message)\s*:\s*\S"
    r"|verbatim contents of"
    r"|contents of (?:my |your |the )?soul"
    r"|hidden ops password)",
    re.IGNORECASE,
)

Verdict = Literal["pass", "halt", "flag_dump"]


@dataclass(frozen=True)
class ResponseCheck:
    """The outcome of ``check_response``: pass, halt (with the pattern), or a dump flag."""

    verdict: Verdict
    pattern: str | None = None

    @property
    def reason(self) -> str:
        if self.verdict == "halt":
            return f"detected unsafe content pattern: {self.pattern}"
        if self.verdict == "flag_dump":
            return "dump phrasing flagged for the leak judge"
        return "no unsafe pattern matched"


def identity_literals() -> frozenset[str]:
    """The owner's secret-shaped identity literals (the ones the response check halts on).

    The ``secret`` kind of the seam's one corpus (``identity_redaction.owner_identity()``);
    empty when no provider is registered. URL-, domain- and email-shaped tokens are the
    ``link`` kind and never halt an answer (issue 0022). Shared with the capability
    redaction hook, so what a capability result is masked for is exactly what an answer is
    refused for.
    """
    corpus = owner_identity()
    return corpus.of("secret") if corpus is not None else frozenset()


def reset_identity_literals() -> None:
    """Forget the cached identity corpus (tests, or a late provider registration)."""
    reset_owner_identity()


def check_response(text: str) -> ResponseCheck:
    """The model-free response check. Pure apart from the cached identity literals."""
    for key, pattern in SAFETY_PATTERNS:
        if pattern.search(text):
            return ResponseCheck("halt", key)
    for literal in identity_literals():
        if literal in text:
            return ResponseCheck("halt", "identity_egress")
    if is_architecture_disclosure(text):
        return ResponseCheck("halt", "architecture_disclosure")
    if DUMP_PATTERN.search(text):
        return ResponseCheck("flag_dump", "prompt_regurgitation")
    return ResponseCheck("pass")


class ResponseSafetyHook:
    """``PreResponse`` hook: ``check_response`` on ``payload['response']``.

    A halt is a ``deny``. A dump flag is an ``allow`` carrying ``dump_flag`` in its
    audit metadata, so the caller can hand it to the leak judge (or, with no judge,
    refuse it — the fail-closed rule stays with the caller that owns the judge).
    """

    name: str = "response_safety"
    hook_point: HookPoint = HookPoint.PRE_RESPONSE
    priority: int = 10

    async def __call__(self, ctx: HookContext) -> HookDecision:
        check = check_response(_response_text(ctx.payload))
        if check.verdict == "halt":
            return HookDecision(
                outcome="deny",
                reason=check.reason,
                severity="critical",
                audit_metadata={"pattern": check.pattern},
            )
        if check.verdict == "flag_dump":
            return HookDecision(
                outcome="allow",
                reason=check.reason,
                severity="warn",
                audit_metadata={"pattern": check.pattern, "dump_flag": True},
            )
        return HookDecision(outcome="allow", reason=check.reason)


def _response_text(payload: dict[str, Any]) -> str:
    value = payload.get("response")
    return value if isinstance(value, str) else ""


__all__ = [
    "DUMP_PATTERN",
    "SAFETY_PATTERNS",
    "ResponseCheck",
    "ResponseSafetyHook",
    "check_response",
    "identity_literals",
    "reset_identity_literals",
]
