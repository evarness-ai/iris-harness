"""The public face of a row's identity fields (issue #134, stage 2).

``public_payload`` is a closed allowlist with exact scalar types: identifiers and a count,
never what was said or passed to a tool.
"""

from __future__ import annotations

import json

from iris_harness.foundation.observability.audit_view import public_payload


def _view(**payload: object) -> dict[str, object]:
    return public_payload(json.dumps(payload))


def test_the_identity_fields_pass_with_their_types() -> None:
    view = _view(
        call_id="C",
        turn_id="T",
        parent_call_id="P",
        attempt=2,
        replay_of="H",
        resumed_from_run="R",
    )
    assert view == {
        "call_id": "C",
        "turn_id": "T",
        "parent_call_id": "P",
        "attempt": 2,
        "replay_of": "H",
        "resumed_from_run": "R",
    }


def test_attempt_is_the_one_int_and_a_bool_or_a_string_never_passes_as_one() -> None:
    assert _view(attempt=1) == {"attempt": 1}
    assert _view(attempt=True) == {}  # bool is an int subclass: excluded explicitly
    assert _view(attempt="2") == {}
    assert _view(turn_id=7, replay_of=["x"], resumed_from_run=True) == {}


def test_nothing_outside_the_allowlist_and_no_empty_value_comes_through() -> None:
    view = _view(turn_id="", args={"path": "secret.txt"}, prompt="my passport", attempt=1)
    assert view == {"attempt": 1}
