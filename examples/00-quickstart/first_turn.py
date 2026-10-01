"""Your first governed turn, from Python: ``python examples/00-quickstart/first_turn.py``.

``harness()`` builds the same IRIS the ``iris`` command runs -- routing, the governed
loop, the curator, the audit ledger -- in a throwaway home, on a scripted model, with
the network refused. One question goes through the whole pipeline; the script prints
the answer and the audit rows the turn wrote.
"""

from __future__ import annotations

from typing import Any

from iris_harness.testing import TurnAuditRow, TurnResult, harness

QUESTION = "What can you help me with?"
ANSWER = "I can sort your email, keep your to-dos and answer from your own documents."

# The scripted model: the router's call gets an intent, every other call the answer.
SCRIPT: dict[str, Any] = {
    "rules": [
        {
            "name": "route",
            "match": {"system": "request router"},
            "reply": {"json": {"intent": "general"}},
        }
    ],
    "default": {"content": ANSWER},
}


def first_turn() -> tuple[TurnResult, list[TurnAuditRow], list[str]]:
    """One turn; returns its result, its audit rows and what ``audit_gaps`` found."""
    with harness(fake_model=SCRIPT) as h:
        result = h.chat(QUESTION)
        rows = h.audit_rows(session_id=result.session_id)
        return result, rows, h.audit_gaps()


def main() -> None:
    result, rows, gaps = first_turn()
    print(f"You:  {QUESTION}")
    print(f"IRIS: {result.text}")
    print(f"      (intent {result.intent}, agent {result.agent})")
    print(f"\n{len(rows)} audit rows for this turn:")
    for row in rows:
        print(f"  {row.hook_point:<14} {row.plugin:<22} {row.decision}")
    print("\nEvery model call and every answer audited:", "yes" if not gaps else gaps)


if __name__ == "__main__":
    main()
