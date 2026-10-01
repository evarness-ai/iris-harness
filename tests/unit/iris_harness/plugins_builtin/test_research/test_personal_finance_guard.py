"""The research guard that refuses to web-search the user's own finances (ADR-0102).

The live failure: "Any insurance dues?" routed the finance ReAct agent to the web
`research` tool, which leaked the user's email and fabricated an insurance provider.

The guard travels with the tool it guards: `research` became a reference plugin at
M4.7, so the refusal lives in `plugins_builtin/research/guard.py`. What stays CORE
is the dues VOCABULARY it reads (`data.dues_vocabulary`), shared with
``configure_brief``'s dues-section narrowing, which answers with no plugin mounted.
The plugin-side detector that picks the dues digest is tested with
``finance_workflows``.
"""

from __future__ import annotations

from iris_harness.plugins_builtin.research.guard import is_personal_finance_web_query


def test_personal_finance_web_query_blocks_possessive_finance() -> None:
    # The research guard refuses possessive personal-finance asks before web egress.
    assert is_personal_finance_web_query("what are my insurance payments, dues") is True
    assert is_personal_finance_web_query("my credit card balance") is True


def test_personal_finance_web_query_allows_general_research() -> None:
    # Genuinely informational/market queries must still reach the web.
    assert is_personal_finance_web_query("best health insurance plans in India") is False
    assert is_personal_finance_web_query("latest Nifty 50 news") is False
    assert is_personal_finance_web_query("how does an EMI work") is False
