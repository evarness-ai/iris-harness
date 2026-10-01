"""The opening-hours handler, run in a real governed IRIS (``iris_harness.testing``).

Offline and model-free: the handler answers without a model, and a message it does not
claim reaches the scripted fake model instead.
"""

from __future__ import annotations

from pathlib import Path

from opening_hours import HANDLER, load_hours, render, setup

from iris_harness.testing import harness, plugin

MANIFEST = Path(__file__).with_name("manifest.yaml")
SCRIPT = {"default": {"content": "I only know about the library."}}


def test_the_handler_answers_without_a_model_and_is_audited() -> None:
    with harness(plugins=[plugin(setup, manifest=MANIFEST)], fake_model=SCRIPT) as h:
        assert h.plugin_loaded("opening_hours")

        result = h.chat("When is the library open on Saturday?")

        assert result.text == render(load_hours())
        assert "Saturday: 10:00 to 14:00" in result.text
        assert h.model_calls() == ()  # no model was asked

        # The answer passed the model-free response check, and its audit row says a
        # deterministic handler gave it -- and which one (R15).
        answer_rows = h.audit_rows(hook_point="pre_response", session_id=result.session_id)
        marked = [row for row in answer_rows if row.deterministic]
        assert [(row.handler, row.decision) for row in marked] == [(HANDLER, "allow")]
        assert h.audit_gaps() == []


def test_a_message_it_does_not_claim_goes_to_the_model() -> None:
    with harness(plugins=[plugin(setup, manifest=MANIFEST)], fake_model=SCRIPT) as h:
        result = h.chat_stream("Tell me a joke about libraries.")

        assert result.text == "I only know about the library."
        assert h.model_calls(), "the model answered this one"
        answer_rows = h.audit_rows(hook_point="pre_response", session_id=result.session_id)
        assert answer_rows and not any(row.deterministic for row in answer_rows)
        assert h.audit_gaps() == []
