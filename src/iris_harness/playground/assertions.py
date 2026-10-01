"""Pure expectation evaluation — no runtime, no I/O, fully unit-testable.

Given an expectation and the observed turn outcome, produce one
``AssertionResult`` per checked field. Kept separate from the runner so the
assertion logic can be tested without building a runtime.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .models import AssertionResult, ScenarioExpectation

# Matches a bare email address; used by the no_pii_leak guardrail.
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
# Raw ReAct/prompt scaffolding markers that must never reach the user.
_PROMPT_MARKERS = ("Thought:", "Action:", "Observation:", "Final Answer:", "<|")


class ObservedTurn:
    """The facts about a completed turn that assertions read from."""

    def __init__(
        self,
        *,
        response: str,
        intent: str | None,
        agent_type: str | None,
        handler: str | None,
        sources: Sequence[str],
        metadata: dict[str, object],
        has_errors: bool,
    ) -> None:
        self.response = response
        self.intent = intent
        self.agent_type = agent_type
        self.handler = handler
        self.sources = tuple(sources)
        self.metadata = dict(metadata)
        self.has_errors = has_errors


def evaluate(expect: ScenarioExpectation, turn: ObservedTurn) -> list[AssertionResult]:
    """Return one AssertionResult per expectation field that was set."""
    out: list[AssertionResult] = []

    if expect.intent is not None:
        out.append(_eq("intent", expect.intent, turn.intent))
    if expect.agent_type is not None:
        out.append(_eq("agent_type", expect.agent_type, turn.agent_type))
    if expect.handler is not None:
        # "" means "no intercept fired" — the turn reached the agent loop.
        want = expect.handler or None
        out.append(_eq("handler", want, turn.handler))
    if expect.has_errors is not None:
        out.append(_eq("has_errors", expect.has_errors, turn.has_errors))

    for tool in expect.sources_include:
        out.append(
            AssertionResult(
                field="sources_include",
                ok=tool in turn.sources,
                expected=tool,
                actual=turn.sources,
                detail=f"expected source {tool!r} present",
            )
        )
    for tool in expect.sources_exclude:
        out.append(
            AssertionResult(
                field="sources_exclude",
                ok=tool not in turn.sources,
                expected=f"absent:{tool}",
                actual=turn.sources,
                detail=f"expected source {tool!r} absent",
            )
        )

    low = turn.response.lower()
    for needle in expect.response_contains:
        out.append(
            AssertionResult(
                field="response_contains",
                ok=needle.lower() in low,
                expected=needle,
                actual=_preview(turn.response),
                detail=f"response must contain {needle!r}",
            )
        )
    for needle in expect.response_not_contains:
        out.append(
            AssertionResult(
                field="response_not_contains",
                ok=needle.lower() not in low,
                expected=f"absent:{needle}",
                actual=_preview(turn.response),
                detail=f"response must not contain {needle!r}",
            )
        )
    if expect.response_regex is not None:
        matched = re.search(expect.response_regex, turn.response, re.MULTILINE) is not None
        out.append(
            AssertionResult(
                field="response_regex",
                ok=matched,
                expected=expect.response_regex,
                actual=_preview(turn.response),
                detail="response must match regex",
            )
        )

    for key, want in expect.metadata.items():
        got = turn.metadata.get(key)
        out.append(
            AssertionResult(
                field=f"metadata.{key}",
                ok=got == want,
                expected=want,
                actual=got,
                detail=f"metadata[{key!r}] must equal expected",
            )
        )

    if expect.no_pii_leak:
        out.append(_no_pii_leak(turn.response))

    return out


def _eq(field: str, expected: object, actual: object) -> AssertionResult:
    return AssertionResult(
        field=field,
        ok=expected == actual,
        expected=expected,
        actual=actual,
        detail=f"{field} must equal expected",
    )


def _no_pii_leak(response: str) -> AssertionResult:
    email = _EMAIL_RE.search(response)
    marker = next((m for m in _PROMPT_MARKERS if m in response), None)
    leak = email.group(0) if email else marker
    return AssertionResult(
        field="no_pii_leak",
        ok=leak is None,
        expected="no email / prompt marker",
        actual=_preview(response),
        detail=f"leaked {leak!r}" if leak else "clean",
    )


def _preview(text: str, limit: int = 160) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"
