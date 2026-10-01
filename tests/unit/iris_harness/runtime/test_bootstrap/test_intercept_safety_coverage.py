"""Coverage-policy invariant: intercepts bypass the curator's safety judges, so
every intercept must be safe by construction.

The 22-step intercept chain answers *before* the ResponseCurator, so the
faithfulness / grounding / output-safety / escalation judges never see an
intercept-answered turn (2026-07-06 governance red-team, finding 2 / open item).
That is acceptable ONLY because:

  * the deterministic intercepts emit templated / local-data text, and
  * the intercepts that invoke a GENERATIVE model constrain it to structured
    extraction (a title, a routine field, a behavior summary, image category
    tags) and template the user-facing response from parsed fields — raw model
    prose is never echoed.

This test pins the set of generative-model-using intercepts. Adding a new
intercept that runs a model must trip this test, forcing the author to confirm
its output is templated/structured (or route it through the curator) rather
than silently bypassing the safety judges.
"""

from __future__ import annotations

from pathlib import Path

from iris_harness.runtime.intercepts import load_intercept_chain

REPO_ROOT = Path(__file__).resolve().parents[5]
_CHAIN = REPO_ROOT / "config" / "intercepts.yaml"

# Intercepts whose handler invokes a generative model (VLM or Tier-2 LLM).
# Reviewed set — each constrains the model to structured extraction and
# templates its response (verified 2026-07-06):
#   image_categorize_request -> on-device VLM tags -> grouped, templated
#   meeting_creation         -> Tier-2 generates the event TITLE only
#   routine_authoring        -> Tier-2 parses the request into routine fields
#   standing_instruction     -> Tier-2 extracts a structured behavior rule
#   move_request             -> Tier-2 LLM fallback extracts a STRUCTURED file-op
#                               (operation/type/source/dest JSON, IRIS_FILEOP_LLM_PARSE);
#                               the deterministic executor acts, response templated
#   account_statement        -> the finance tier reads the institution's unread mail into
#                               a STRUCTURED reading (kind + figures JSON), every figure
#                               checked against the email text before it is stored; the
#                               reply is templated from the stored records (ADR-0121)
_MODEL_USING_INTERCEPTS = frozenset(
    {
        "image_categorize_request",
        "meeting_creation",
        "routine_authoring",
        "standing_instruction",
        "move_request",
        "account_statement",
    }
)

# Every intercept in the chain, each reviewed as safe-by-construction (its
# user-facing response is templated / local-data, or a model output constrained
# to structured extraction). Pinned so ANY added or removed intercept trips this
# test, forcing the author to classify the new one here — the single reviewed
# place an intercept's model use is declared.
_DETERMINISTIC_INTERCEPTS = frozenset(
    {
        "confirmation",
        "cleanup_selection",
        "categorize_selection",
        "organize_confirmation",
        "filemanager_skill_confirmation",
        "folder_files",
        "organize_request",
        "cleanup_request",
        # Moves a judged email to the bucket the owner named: templated from the
        # judgments store + judge.yaml replies; no model.
        "email_rebucket",
        # Done / Snooze / Cancel / list: templated from the reminder store; no model.
        "reminder_action",
        "reminder_creation",
        "time_date",
        "brief_request",
        "brief_config",
        "portfolio_request",
        "bill_paid",
        "dues_request",
        # Templated from the user's own derived dues; no model in the path.
        "bill_amount_request",
        "statement_email_details",
        "account_confirmation",
        "accounts_request",
        "routine_management",
    }
)
_ALL_REVIEWED_INTERCEPTS = _DETERMINISTIC_INTERCEPTS | _MODEL_USING_INTERCEPTS


def test_model_using_and_deterministic_sets_are_disjoint() -> None:
    assert not (_DETERMINISTIC_INTERCEPTS & _MODEL_USING_INTERCEPTS)


def test_every_intercept_is_classified_for_model_use() -> None:
    """The live chain must exactly match the reviewed classification. A new
    intercept (model-using or not) fails this test until it's added to the
    deterministic set or the model-using allowlist — so no intercept can
    silently bypass the safety judges with unreviewed model-generated content.
    """
    names = {spec.name for spec in load_intercept_chain(_CHAIN)}

    unclassified = names - _ALL_REVIEWED_INTERCEPTS
    assert not unclassified, (
        f"unclassified intercept(s): {sorted(unclassified)} — intercepts answer "
        "before the curator, so classify each in test_intercept_safety_coverage.py "
        "(deterministic vs model-using) and confirm its response is templated."
    )
    dropped = _ALL_REVIEWED_INTERCEPTS - names
    assert not dropped, f"reviewed intercept(s) no longer in the chain: {sorted(dropped)}"
