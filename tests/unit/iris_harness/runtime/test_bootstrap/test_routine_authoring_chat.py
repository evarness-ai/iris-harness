"""Tests for conversational routine authoring in IrisRuntime.chat."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.runtime import build_runtime
from iris_harness.services.routines import (
    RoutineApprovalRequestStatus,
    RoutineApprovalStatus,
    create_routine_spec,
    parse_routine_authoring,
)

# No model server in tests (tests/conftest.py network guard): runtime turns here
# reach the LLM on their degrade paths, so the model is a stubbed dead server.
pytestmark = pytest.mark.usefixtures("offline_llm")

REPO_ROOT = Path(__file__).resolve().parents[5]


def _runtime(tmp_path: Path):
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    return build_runtime(
        config_dir=config_dir,
        data_dir=data_dir,
        use_background_scheduler=False,
    )


def _runtime_with_real_skills(tmp_path: Path):
    """Runtime backed by the real ``config/`` directory so skill
    discovery finds ``web-fetch``, ``iris-tasks``, etc. — only
    the data dir is isolated. Used by integration tests that need a
    populated capability candidate set.

    ``build_runtime`` does not auto-discover skills; production calls
    ``runtime.startup()`` which does. We invoke ``discover()`` directly
    so the test doesn't pay startup's heartbeat / poller / warmup cost.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    runtime = build_runtime(
        config_dir=REPO_ROOT / "config",
        data_dir=data_dir,
        use_background_scheduler=False,
    )
    runtime.skill_registry.discover()
    return runtime


@pytest.mark.minilm
def test_chat_drafts_and_approves_routine(tmp_path: Path) -> None:
    """Full draft -> refine -> approve lifecycle on the morning-briefing
    skill. Rewritten 2026-05-20 against the post-anchor-commit handlers
    (was previously asserting the deleted ``daily_repo_brief`` /
    ``Repository source`` hardcoded strings)."""

    runtime = _runtime_with_real_skills(tmp_path)
    session_id = "routine-session"

    drafted = runtime.chat(
        "send me a morning briefing everyday at 9 am with reminders, "
        "active items, and top 10 git repositories",
        session_id=session_id,
    )

    assert drafted.intent == "routine_authoring"
    assert drafted.metadata["routine_action"] == "drafted"
    assert "Drafted routine" in drafted.response
    assert "`approve`" in drafted.response
    assert "`sample`" in drafted.response
    pending_request = runtime.routine_store.get_pending_approval_request(session_id)
    assert pending_request is not None
    draft_routine = runtime.routine_store.load(pending_request.routine_id)
    assert draft_routine is not None
    assert draft_routine.template == "morning-briefing"
    assert draft_routine.schedule == "daily:09:00"
    assert "reminders" in draft_routine.metadata.get("briefing_sections", [])
    assert "top_repos" in draft_routine.metadata.get("briefing_sections", [])

    refined = runtime.chat(
        "deliver to telegram, content should be 3 lines per news",
        session_id=session_id,
    )

    assert refined.intent == "routine_authoring"
    assert refined.metadata["routine_action"] == "updated"
    assert refined.metadata["delivery_channel"] == "telegram"
    assert refined.metadata["content_style"] == "3 lines per news"
    refined_spec = runtime.routine_store.load(pending_request.routine_id)
    assert refined_spec is not None
    assert refined_spec.delivery_channel == "telegram"
    assert refined_spec.metadata["content_lines_per_item"] == 3

    approved = runtime.chat("approve it", session_id=session_id)
    finalized = runtime.routine_store.load(pending_request.routine_id)

    assert approved.intent == "routine_authoring"
    assert approved.metadata["routine_action"] == "approved"
    assert "Scheduled routine" in approved.response
    assert "Delivery: telegram" in approved.response
    assert finalized is not None
    assert str(finalized.approval_status) == "scheduled"
    assert finalized.delivery_channel == "telegram"
    approval_request = runtime.routine_store.load_approval_request(pending_request.id)
    assert approval_request is not None
    assert approval_request.status == RoutineApprovalRequestStatus.APPROVED


@pytest.mark.minilm
def test_chat_telegram_origin_defaults_delivery_to_telegram(tmp_path: Path) -> None:
    """Phase 1 regression: a routine authored from the Telegram gateway
    (``channel="telegram"``) must default ``delivery_channel`` to telegram,
    not console. Previously every routine landed on ``console`` because the
    gateway never forwarded its channel, so scheduled briefs were delivered
    to the API process stdout and never reached the user."""

    runtime = _runtime_with_real_skills(tmp_path)
    session_id = "telegram:123456789"

    drafted = runtime.chat(
        "send me a morning briefing everyday at 9 am with reminders and " "top 10 git repositories",
        session_id=session_id,
        channel="telegram",
    )

    assert drafted.intent == "routine_authoring"
    assert drafted.metadata["routine_action"] == "drafted"
    pending_request = runtime.routine_store.get_pending_approval_request(session_id)
    assert pending_request is not None
    draft_routine = runtime.routine_store.load(pending_request.routine_id)
    assert draft_routine is not None
    assert draft_routine.delivery_channel == "telegram"


@pytest.mark.minilm
def test_chat_sample_previews_draft_without_persisting(
    tmp_path: Path, offline_web_fetch
) -> None:  # type: ignore[no-untyped-def]
    """Phase 4: replying ``sample`` to a pending draft renders the routine's
    real output inline and leaves the draft a draft — no new row, no status
    change, no run counters. Previews can never leak into the store."""

    runtime = _runtime_with_real_skills(tmp_path)
    offline_web_fetch(runtime)  # the preview renders web-fetch slots (GitHub, Yahoo)
    session_id = "sample-session"

    drafted = runtime.chat(
        "send me a morning briefing everyday at 9 am with reminders and " "top 10 git repositories",
        session_id=session_id,
    )
    assert drafted.metadata["routine_action"] == "drafted"
    assert "Reply `sample`" in drafted.response
    pending = runtime.routine_store.get_pending_approval_request(session_id)
    assert pending is not None
    before = runtime.routine_store.load(pending.routine_id)
    assert before is not None
    routine_count_before = len(runtime.routine_store.list_all())

    sampled = runtime.chat("sample", session_id=session_id)

    assert sampled.metadata["routine_action"] == "sample"
    # still a draft, same row, same run counters, no extra routines
    after = runtime.routine_store.load(pending.routine_id)
    assert after is not None
    assert str(after.approval_status) == str(before.approval_status)
    assert after.run_count == before.run_count
    assert len(runtime.routine_store.list_all()) == routine_count_before


def test_chat_clarifies_when_capability_cannot_be_matched(tmp_path: Path) -> None:
    """Post-anchor-commit semantics: clarification surfaces when the
    semantic router can't resolve the bound capability from the
    message. ``Every morning at 8 run my thing`` has a valid schedule
    + signal + verb but no matchable skill, so the chat handler must
    reach the no-draft clarify branch. Rewritten 2026-05-20 from the
    legacy ``test_chat_clarifies_generic_morning_briefing_before_drafting``
    which asserted strings from the deleted clarify-sections path."""

    runtime = _runtime_with_real_skills(tmp_path)
    session_id = "clarify-session"

    clarification = runtime.chat(
        "Every morning at 8 run my thing",
        session_id=session_id,
    )

    assert clarification.intent == "routine_authoring"
    assert clarification.metadata["routine_action"] == "clarify"
    # Missing template — semantic router didn't pick anything.
    assert "template" in clarification.metadata.get("missing_slots", [])
    # No CLARIFY routine persisted when template is missing (the
    # handler routes through the inline-reason branch).
    pending = runtime.routine_store.list_by_status(RoutineApprovalStatus.CLARIFY)
    assert pending == []


def test_routine_parser_defers_gmail_morning_brief_setup_question() -> None:
    parsed = parse_routine_authoring(
        "how to connect to my gmail and extract emails everyday for a morning brief?"
    )

    assert parsed.action == "none"


def test_chat_routes_gmail_morning_brief_setup_to_email_not_routine(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    result = runtime.chat(
        "how to connect to my gmail and extract emails everyday for a morning brief?",
        session_id="gmail-brief-session",
    )

    assert result.intent == "communication"
    assert result.agent_type == "email"
    assert result.sources == ("email",)
    # Email turns run on the shared governed loop (2026-09-21, ADR-0118 step 5: only it
    # can pause on a destructive call's approval); the email plugin's agent is its
    # degrade path. Either way a gmail-setup question must NOT become a persisted
    # routine. (The exact phrasing is not asserted — the loop's answer varies.)
    assert result.metadata.get("agentic_core") is True
    assert result.response.strip()
    assert not runtime.routine_store.list_all()


@pytest.mark.minilm
def test_chat_approval_survives_runtime_rebuild(tmp_path: Path) -> None:
    """A draft + pending approval written by one IrisRuntime must be
    approvable by a freshly-constructed runtime pointing at the same
    data dir. Rewritten 2026-05-20 to use real skills (so the initial
    draft chat finds a capability) and to align with the current
    post-anchor-commit handler shape."""

    runtime = _runtime_with_real_skills(tmp_path)

    drafted = runtime.chat(
        "Every morning at 8 AM send me the top 10 repos from GitHub Trending",
        session_id="durable-routine-session",
    )
    assert drafted.metadata["routine_action"] == "drafted"
    request = runtime.routine_store.get_pending_approval_request("durable-routine-session")
    assert request is not None

    rebuilt = _runtime_with_real_skills(tmp_path)
    approved = rebuilt.chat("approve it", session_id="durable-routine-session")
    updated_request = rebuilt.routine_store.load_approval_request(request.id)
    updated_routine = rebuilt.routine_store.load(request.routine_id)

    assert approved.intent == "routine_authoring"
    assert approved.metadata["routine_action"] == "approved"
    assert "Scheduled routine" in approved.response
    assert updated_request is not None
    assert updated_request.status == RoutineApprovalRequestStatus.APPROVED
    assert updated_routine is not None
    assert str(updated_routine.approval_status) == "scheduled"


@pytest.mark.minilm
def test_chat_stream_returns_routine_authoring_done_event(tmp_path: Path) -> None:
    """Streaming variant of the routine-authoring flow yields a final
    ``done`` event whose ``result`` carries the routine_authoring intent
    + drafted metadata. Rewritten 2026-05-20 to use real skills and to
    assert on stable fields (intent + schedule + presence of sections)
    instead of the exact section ordering which depends on the brief
    skill's slot list."""

    runtime = _runtime_with_real_skills(tmp_path)

    # NB: deliberately avoid the word "routines" in the section list —
    # the routine-management handler runs before routine-authoring and
    # would catch the message via keyword.
    events = list(
        runtime.chat_stream(
            "Every weekday at 8:30 send me a morning briefing with reminders " "and active items",
            session_id="stream-routine-session",
        )
    )

    assert events[0].kind == "trace"
    assert events[-1].kind == "done"
    assert events[-1].result is not None
    result = events[-1].result
    assert result.intent == "routine_authoring"
    assert result.metadata["routine_action"] == "drafted"
    saved = runtime.routine_store.list_all()
    assert len(saved) == 1
    assert saved[0].schedule == "cron:30 8 * * 1-5"
    assert saved[0].template == "morning-briefing"
    sections = saved[0].metadata.get("briefing_sections") or []
    assert "reminders" in sections
    assert "active_items" in sections


def test_chat_lists_updates_and_deletes_existing_routines(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    spec = runtime.routine_store.save(
        create_routine_spec(
            title="Daily repo brief",
            goal="Send trending repos",
            schedule="daily:09:00",
            template="daily_repo_brief",
            approval_status=RoutineApprovalStatus.SCHEDULED,
            metadata={"repo_source": "github_trending", "repo_limit": 10},
        )
    )

    listed = runtime.chat("list routines", session_id="manage-routines")

    assert listed.intent == "routine_authoring"
    assert listed.metadata["routine_action"] == "list"
    assert spec.id in listed.response
    assert listed.metadata["routine_ids"] == [spec.id]

    updated = runtime.chat(
        f"update routine {spec.id} delivery is telegram, content should be 2 lines per item",
        session_id="manage-routines",
    )
    saved = runtime.routine_store.load(spec.id)

    assert updated.metadata["routine_action"] == "updated"
    assert updated.metadata["routine_id"] == spec.id
    assert "Delivery: telegram" in updated.response
    assert "Content style: 2 lines per item" in updated.response
    assert saved is not None
    assert saved.delivery_channel == "telegram"
    assert saved.metadata["content_style"] == "2 lines per item"

    deleted = runtime.chat(f"delete routine {spec.id}", session_id="manage-routines")

    assert deleted.metadata["routine_action"] == "deleted"
    assert deleted.metadata["routine_id"] == spec.id
    assert spec.id in deleted.response
    assert runtime.routine_store.load(spec.id) is None


def test_chat_routine_update_without_id_shows_full_ids(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    spec = runtime.routine_store.save(
        create_routine_spec(
            title="Morning briefing",
            goal="Send a briefing",
            schedule="daily:08:00",
            template="morning_briefing",
        )
    )

    result = runtime.chat("update routine delivery is telegram", session_id="manage-routines")

    assert result.metadata["routine_action"] == "missing_update_id"
    assert spec.id in result.response
    assert result.metadata["routine_ids"] == [spec.id]


def test_chat_stream_bulk_updates_all_routine_delivery(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    first = runtime.routine_store.save(
        create_routine_spec(
            title="Morning briefing",
            goal="Send a briefing",
            schedule="daily:08:00",
            template="morning_briefing",
        )
    )
    second = runtime.routine_store.save(
        create_routine_spec(
            title="Daily repo brief",
            goal="Send trending repos",
            schedule="daily:09:00",
            template="daily_repo_brief",
            approval_status=RoutineApprovalStatus.SCHEDULED,
        )
    )

    events = list(
        runtime.chat_stream(
            "update delivery to Telegram channel for all routines",
            session_id="manage-routines-stream",
        )
    )
    result = events[-1].result

    assert events[-1].kind == "done"
    assert result is not None
    assert result.metadata["routine_action"] == "updated_many"
    assert result.metadata["routine_ids"] == [first.id, second.id]
    assert "Updated delivery to telegram for 2 routines" in result.response
    assert "routine draft in progress" not in result.response
    updated_first = runtime.routine_store.load(first.id)
    updated_second = runtime.routine_store.load(second.id)
    assert updated_first is not None
    assert updated_second is not None
    assert updated_first.delivery_channel == "telegram"
    assert updated_second.delivery_channel == "telegram"


# ---------------------------------------------------------------------------
# Phase A-2 / Step 5: end-to-end tool-args clarification flow.
# These use the real ``config/`` so skill discovery finds web-fetch.
# ---------------------------------------------------------------------------


@pytest.mark.minilm
def test_chat_clarifies_tool_args_then_drafts_on_complete_reply(
    tmp_path: Path,
) -> None:
    """End-to-end: routine for the multi-arg ``fetch_web_content``
    tool first asks for the missing values, then drafts on a complete
    reply. With no Ollama in tests, ``parse_tool_arg_reply`` falls back
    to deterministic ``key=value`` parsing — the flow still works."""

    runtime = _runtime_with_real_skills(tmp_path)

    clarification = runtime.chat(
        "every day at 9 am fetch web content",
        session_id="tool-args-session",
    )

    assert clarification.intent == "routine_authoring"
    assert clarification.metadata["routine_action"] == "clarify"
    assert "tool_args" in clarification.metadata.get("missing_slots", [])
    # The rich clarification reason enumerates the choices.
    assert "**type**" in clarification.response
    assert "**category**" in clarification.response
    assert "`git`" in clarification.response

    pending = runtime.routine_store.list_by_status(RoutineApprovalStatus.CLARIFY)
    assert len(pending) == 1
    pending_meta = pending[0].metadata.get("pending_tool_args")
    assert isinstance(pending_meta, list) and len(pending_meta) == 2

    drafted = runtime.chat(
        "type=git, category=git-repositories",
        session_id="tool-args-session",
    )

    assert drafted.metadata["routine_action"] == "drafted"
    resolved = drafted.metadata.get("resolved_tool_args") or {}
    assert resolved.get("type") == "git"
    assert resolved.get("category") == "git-repositories"


@pytest.mark.minilm
def test_chat_re_asks_only_missing_tool_args_after_partial_reply(
    tmp_path: Path,
) -> None:
    """When the user resolves only some args, the next clarification
    re-asks ONLY the missing ones — already-answered args persist.

    A live Tier-2 LLM might infer ``category`` from ``type=git`` (only
    one category exists for that type in the manifest), which would
    skip past clarify and draft immediately. Stubbing the caller to
    return just ``type`` keeps the test deterministic regardless of
    whether Ollama is running locally."""

    runtime = _runtime_with_real_skills(tmp_path)
    runtime.routines.routine_authoring_llm_caller = lambda: (
        lambda _: '{"type": "git", "missing": ["category"]}'
    )

    runtime.chat(
        "every day at 9 am fetch web content",
        session_id="partial-args-session",
    )

    partial = runtime.chat(
        "type=git",
        session_id="partial-args-session",
    )

    assert partial.metadata["routine_action"] == "clarify"
    assert partial.metadata.get("missing_slots") == ["tool_args"]
    resolved = partial.metadata.get("resolved_tool_args") or {}
    assert resolved.get("type") == "git"
    pending_names = partial.metadata.get("pending_tool_args") or []
    assert pending_names == ["category"]
    # Re-ask renders the still-missing arg's options only.
    assert "**category**" in partial.response
    assert "**type**" not in partial.response


@pytest.mark.minilm
def test_chat_zero_arg_non_brief_tool_drafts_without_clarification(
    tmp_path: Path,
) -> None:
    """Routines bound to a tool with no required args (``iris-tasks``, a core skill
    that ships in every tree) skip the clarification step entirely — no
    ``tool_args`` ask."""

    runtime = _runtime_with_real_skills(tmp_path)

    result = runtime.chat(
        "every morning at 8 send me my overdue tasks",
        session_id="zero-arg-session",
    )

    assert result.metadata["routine_action"] == "drafted"
    assert "tool_args" not in result.metadata.get("missing_slots", [])


# ---------------------------------------------------------------------------
# Phase B: post-approval refinement updates the approved routine in place.
# ---------------------------------------------------------------------------


@pytest.mark.minilm
def test_chat_post_approval_refinement_updates_single_scheduled_routine(
    tmp_path: Path,
) -> None:
    """After a routine is approved + scheduled, a follow-up refinement
    ("deliver to telegram") must update *that* routine in place rather
    than complain about a missing draft or spawn a new one."""

    runtime = _runtime_with_real_skills(tmp_path)
    session_id = "post-approval-session"

    # 1) Draft via a zero-arg skill (iris-tasks) so we land on a
    #    DRAFT in one turn, no clarification needed.
    drafted = runtime.chat(
        "every morning at 8 send me my overdue tasks",
        session_id=session_id,
    )
    routine_id = drafted.metadata["routine_id"]

    # 2) Approve so the routine moves to SCHEDULED.
    approved = runtime.chat("approve it", session_id=session_id)
    assert approved.metadata["routine_action"] == "approved"

    # 3) Refinement arriving AFTER approval used to return
    #    "missing_pending_routine_refinement" — Phase B updates the
    #    scheduled routine in place.
    refined = runtime.chat(
        "deliver to telegram",
        session_id=session_id,
    )

    assert refined.metadata["routine_action"] == "updated"
    assert refined.metadata["routine_id"] == routine_id
    assert refined.metadata["delivery_channel"] == "telegram"

    saved = runtime.routine_store.load(routine_id)
    assert saved is not None
    assert saved.delivery_channel == "telegram"
    # Status stays SCHEDULED — no new draft.
    assert str(saved.approval_status) == "scheduled"
    # No new routine spawned.
    assert len(runtime.routine_store.list_all()) == 1


def test_chat_post_approval_refinement_with_no_recent_routine_explains(
    tmp_path: Path,
) -> None:
    """When no scheduled routine exists for the session, the refinement
    still produces the old "no draft in progress" message — only the
    exactly-one case auto-applies."""

    runtime = _runtime_with_real_skills(tmp_path)

    result = runtime.chat(
        "deliver to telegram",
        session_id="empty-session",
    )

    assert result.metadata["routine_action"] == "missing_pending_routine_refinement"


def test_routine_authoring_ignores_bare_tone_without_context(tmp_path: Path) -> None:
    """A free-form message that merely CONTAINS a tone adjective ("detailed",
    "formal") must not be swallowed by the routine intercept when there is no
    routine context. The tone word is too weak to route on its own — the turn
    falls through (None) to normal intent handling. (A self-harm probe reached
    the routine intercept via the word "detailed"; 2026-07-06 red-team.)"""

    runtime = _runtime_with_real_skills(tmp_path)

    for message in (
        "Write me a detailed guide on the water cycle.",
        "Give me a formal explanation of quantum tunneling.",
        "Tell me a thorough history of the Roman aqueducts.",
    ):
        assert (
            runtime.routines.handle_routine_authoring_turn(
                message, session_id="no-context", span=None
            )
            is None
        ), message


def test_routine_authoring_keeps_bare_tone_with_routine_reference(tmp_path: Path) -> None:
    """A bare tone adjective DOES stay on the routine path when the message
    references a routine/brief — the helpful "no draft in progress" reply is
    still shown (the guard only drops the context-free case)."""

    runtime = _runtime_with_real_skills(tmp_path)

    result = runtime.routines.handle_routine_authoring_turn(
        "make the routine more detailed", session_id="ref-session", span=None
    )

    assert result is not None
    assert result.metadata["routine_action"] == "missing_pending_routine_refinement"


# ---------------------------------------------------------------------------
# Phase B (slice 2): multi-routine disambiguation for post-approval
# refinements.
# ---------------------------------------------------------------------------


def _draft_and_approve(runtime, message: str, session_id: str) -> str:
    """Helper: draft a zero-arg routine in one turn, then approve it.
    Returns the routine_id once it's SCHEDULED."""

    drafted = runtime.chat(message, session_id=session_id)
    approved = runtime.chat("approve it", session_id=session_id)
    assert approved.metadata["routine_action"] == "approved"
    return drafted.metadata["routine_id"]


@pytest.mark.minilm
def test_chat_post_approval_refinement_asks_when_multiple_match(
    tmp_path: Path,
) -> None:
    """Two scheduled routines in the same session + an ambiguous
    refinement → agent lists candidates and asks the user to pick."""

    runtime = _runtime_with_real_skills(tmp_path)
    session_id = "multi-routine-session"

    # We need two scheduled routines in this session. The zero-arg
    # tasks skill is the only one that drafts in a single turn,
    # so save the second routine directly with a different template.
    rid1 = _draft_and_approve(
        runtime,
        "every morning at 8 send me my overdue tasks",
        session_id,
    )
    second = runtime.routine_store.save(
        create_routine_spec(
            title="Daily repo brief",
            goal="Send trending repos",
            schedule="daily:09:00",
            template="daily-repo-brief",
            approval_status=RoutineApprovalStatus.SCHEDULED,
            metadata={"session_id": session_id},
        )
    )

    refinement = runtime.chat("deliver to telegram", session_id=session_id)

    assert refinement.metadata["routine_action"] == "clarify"
    assert refinement.metadata["missing_slots"] == ["refinement_target"]
    candidate_ids = refinement.metadata["candidate_routine_ids"]
    assert set(candidate_ids) == {rid1, second.id}
    assert "Refine which routine" in refinement.response
    # Both ids appear in the rendered list.
    assert rid1 in refinement.response
    assert second.id in refinement.response


@pytest.mark.minilm
def test_chat_post_approval_refinement_pick_by_position_applies(
    tmp_path: Path,
) -> None:
    """After the disambiguation prompt, a position reply ("2") resolves
    to the second candidate and applies the original refinement to it."""

    runtime = _runtime_with_real_skills(tmp_path)
    session_id = "multi-pick-session"

    _draft_and_approve(
        runtime,
        "every morning at 8 send me my overdue tasks",
        session_id,
    )
    second = runtime.routine_store.save(
        create_routine_spec(
            title="Daily repo brief",
            goal="Send trending repos",
            schedule="daily:09:00",
            template="daily-repo-brief",
            approval_status=RoutineApprovalStatus.SCHEDULED,
            metadata={"session_id": session_id},
        )
    )

    # Trigger disambiguation
    ambiguous = runtime.chat("deliver to telegram", session_id=session_id)
    candidate_ids = ambiguous.metadata["candidate_routine_ids"]

    # Pick the position matching `second` in the rendered order
    second_index = candidate_ids.index(second.id) + 1
    applied = runtime.chat(str(second_index), session_id=session_id)

    assert applied.metadata["routine_action"] == "updated"
    assert applied.metadata["routine_id"] == second.id
    assert applied.metadata["delivery_channel"] == "telegram"

    saved = runtime.routine_store.load(second.id)
    assert saved is not None
    assert saved.delivery_channel == "telegram"


@pytest.mark.minilm
def test_chat_post_approval_refinement_cancel_clears_pending_pick(
    tmp_path: Path,
) -> None:
    """``cancel`` after the disambiguation prompt drops the pending
    pick — a follow-up refinement starts fresh."""

    runtime = _runtime_with_real_skills(tmp_path)
    session_id = "multi-cancel-session"

    _draft_and_approve(
        runtime,
        "every morning at 8 send me my overdue tasks",
        session_id,
    )
    runtime.routine_store.save(
        create_routine_spec(
            title="Daily repo brief",
            goal="g",
            schedule="daily:09:00",
            template="daily-repo-brief",
            approval_status=RoutineApprovalStatus.SCHEDULED,
            metadata={"session_id": session_id},
        )
    )

    runtime.chat("deliver to telegram", session_id=session_id)
    cancelled = runtime.chat("cancel", session_id=session_id)

    assert cancelled.metadata["routine_action"] == "refinement_cancelled"
    assert runtime.routines._pending_refinement_picks.get(session_id) is None


# ---------------------------------------------------------------------------
# Phase B slice 3: capability swap requires explicit confirmation.
# ---------------------------------------------------------------------------


@pytest.mark.minilm
def test_chat_capability_swap_proposal_asks_for_confirmation(
    tmp_path: Path,
) -> None:
    """A swap intent against the one scheduled routine surfaces a
    confirmation prompt — the swap is NOT applied yet."""

    runtime = _runtime_with_real_skills(tmp_path)
    session_id = "swap-propose-session"

    routine_id = _draft_and_approve(
        runtime,
        "every morning at 8 send me my overdue tasks",
        session_id,
    )

    proposed = runtime.chat(
        "switch to morning briefing instead",
        session_id=session_id,
    )

    assert proposed.metadata["routine_action"] == "clarify"
    assert proposed.metadata["missing_slots"] == ["capability_swap_confirmation"]
    assert proposed.metadata["routine_id"] == routine_id
    assert proposed.metadata["proposed_template"] == "morning-briefing"
    # Routine isn't rebound yet.
    saved = runtime.routine_store.load(routine_id)
    assert saved is not None
    assert saved.template == "iris-tasks"


@pytest.mark.minilm
def test_chat_capability_swap_confirmation_applies(tmp_path: Path) -> None:
    """``yes`` after a swap proposal rebinds the routine."""

    runtime = _runtime_with_real_skills(tmp_path)
    session_id = "swap-confirm-session"

    routine_id = _draft_and_approve(
        runtime,
        "every morning at 8 send me my overdue tasks",
        session_id,
    )
    runtime.chat("switch to morning briefing instead", session_id=session_id)

    applied = runtime.chat("yes", session_id=session_id)

    assert applied.metadata["routine_action"] == "updated"
    assert applied.metadata["routine_id"] == routine_id
    saved = runtime.routine_store.load(routine_id)
    assert saved is not None
    assert saved.template == "morning-briefing"
    # Pending state cleared.
    assert runtime.routines._pending_capability_swaps.get(session_id) is None


@pytest.mark.minilm
def test_chat_capability_swap_cancel_keeps_original(tmp_path: Path) -> None:
    """``cancel`` (or any non-approve reply) drops the proposed swap."""

    runtime = _runtime_with_real_skills(tmp_path)
    session_id = "swap-cancel-session"

    routine_id = _draft_and_approve(
        runtime,
        "every morning at 8 send me my overdue tasks",
        session_id,
    )
    runtime.chat("switch to morning briefing instead", session_id=session_id)

    cancelled = runtime.chat("cancel", session_id=session_id)

    assert cancelled.metadata["routine_action"] == "refinement_cancelled"
    saved = runtime.routine_store.load(routine_id)
    assert saved is not None
    assert saved.template == "iris-tasks"
    assert runtime.routines._pending_capability_swaps.get(session_id) is None


# ---------------------------------------------------------------------------
# Phase B slice 4: 5-minute undo window for post-approval refinements.
# ---------------------------------------------------------------------------


@pytest.mark.minilm
def test_chat_undo_restores_pre_refinement_state(tmp_path: Path) -> None:
    """After a successful refinement, ``undo`` rolls the routine back."""

    runtime = _runtime_with_real_skills(tmp_path)
    session_id = "undo-session"

    routine_id = _draft_and_approve(
        runtime,
        "every morning at 8 send me my overdue tasks",
        session_id,
    )
    pre = runtime.routine_store.load(routine_id)
    assert pre is not None
    pre_channel = pre.delivery_channel

    runtime.chat("deliver to telegram", session_id=session_id)
    after_refine = runtime.routine_store.load(routine_id)
    assert after_refine is not None
    assert after_refine.delivery_channel == "telegram"

    undone = runtime.chat("undo", session_id=session_id)

    assert undone.metadata["routine_action"] == "undone"
    assert undone.metadata["routine_id"] == routine_id
    restored = runtime.routine_store.load(routine_id)
    assert restored is not None
    assert restored.delivery_channel == pre_channel
    # Snapshot consumed — a second undo finds nothing.
    assert runtime.routines._recent_refinement_snapshots.get(session_id) is None


@pytest.mark.minilm
def test_chat_undo_after_window_expires_explains(tmp_path: Path) -> None:
    """An ``undo`` issued after the 5-minute window degrades to a
    no-op explainer instead of restoring stale state."""

    from datetime import UTC, datetime, timedelta

    runtime = _runtime_with_real_skills(tmp_path)
    session_id = "undo-expired-session"

    routine_id = _draft_and_approve(
        runtime,
        "every morning at 8 send me my overdue tasks",
        session_id,
    )
    runtime.chat("deliver to telegram", session_id=session_id)

    # Backdate the snapshot expiry so the window has elapsed.
    runtime.routines._recent_refinement_snapshots[session_id]["expires_at"] = datetime.now(
        UTC
    ) - timedelta(minutes=1)

    expired = runtime.chat("undo", session_id=session_id)

    assert expired.metadata["routine_action"] == "undo_unavailable"
    saved = runtime.routine_store.load(routine_id)
    assert saved is not None
    # Refinement stays applied.
    assert saved.delivery_channel == "telegram"


def test_undo_in_a_session_that_refined_no_routine_is_not_claimed(tmp_path: Path) -> None:
    """Owner decision 2026-09-22: "undo that" after an email trash reached this handler,
    which answered "Nothing to undo in this chat" and the trash was never undone. With
    no routine refinement in the session, the turn goes on to the rest of the pipeline.
    (An expired refinement in the same session still explains itself: see above.)"""
    runtime = _runtime_with_real_skills(tmp_path)

    for text in ("undo", "undo that", "revert it"):
        assert (
            runtime.routines.handle_routine_authoring_turn(text, session_id="cold-session") is None
        )


# ---------------------------------------------------------------------------
# Phase C: gateway origin plumbing — channel kwarg defaults delivery.
# ---------------------------------------------------------------------------


@pytest.mark.minilm
def test_chat_telegram_channel_defaults_routine_delivery_to_telegram(
    tmp_path: Path,
) -> None:
    """A user authoring a routine from the Telegram gateway gets the
    routine's delivery_channel defaulted to ``telegram`` without
    having to say so explicitly."""

    runtime = _runtime_with_real_skills(tmp_path)

    drafted = runtime.chat(
        "every morning at 8 send me my overdue tasks",
        session_id="tg-session",
        channel="telegram",
    )

    assert drafted.metadata["routine_action"] == "drafted"
    assert drafted.metadata["delivery_channel"] == "telegram"


@pytest.mark.minilm
def test_chat_explicit_delivery_overrides_channel_origin(
    tmp_path: Path,
) -> None:
    """``deliver to console`` in the message wins over the gateway
    origin — explicit user intent always trumps the default."""

    runtime = _runtime_with_real_skills(tmp_path)

    drafted = runtime.chat(
        "every morning at 8 send me the tasks that are overdue, deliver to console",
        session_id="tg-explicit-session",
        channel="telegram",
    )

    assert drafted.metadata["routine_action"] == "drafted"
    assert drafted.metadata["delivery_channel"] == "console"


@pytest.mark.minilm
def test_chat_default_channel_is_console(tmp_path: Path) -> None:
    """Callers that don't specify the channel get the console default —
    the existing CLI flow doesn't have to change to keep working."""

    runtime = _runtime_with_real_skills(tmp_path)

    drafted = runtime.chat(
        "every morning at 8 send me my overdue tasks",
        session_id="cli-default-session",
    )

    assert drafted.metadata["delivery_channel"] == "console"


def test_chat_set_up_phrasing_is_creation_not_update(tmp_path: Path) -> None:
    """Phase 3 multiturn scenario regression: "set up a routine ..." matched
    the update verb regex (bare "set") and dead-ended in missing_update_id
    instead of reaching the authoring flow."""
    runtime = _runtime_with_real_skills(tmp_path)

    result = runtime.chat(
        "set up a routine for me every morning at 7",
        session_id="setup-phrasing-session",
    )

    assert result.metadata.get("routine_action") != "missing_update_id"
    # Authoring should either clarify the capability or draft directly.
    assert result.metadata.get("routine_action") in ("clarify", "drafted") or (
        "routine" in result.response.lower()
    )


@pytest.mark.minilm
def test_chat_capability_clarify_is_stateful(tmp_path: Path) -> None:
    """Phase 3 follow-up: a missing-capability clarification remembers the
    request, so naming the capability on the next turn drafts the routine
    without restating the schedule."""
    runtime = _runtime_with_real_skills(tmp_path)
    session = "capclar-session"

    first = runtime.chat(
        "set up a routine for me every morning at 7",
        session_id=session,
    )
    assert first.metadata.get("routine_action") == "clarify"
    assert first.metadata.get("awaiting") == "capability"

    second = runtime.chat("the morning briefing", session_id=session)
    assert second.metadata.get("routine_action") == "drafted"
    pending = runtime.routine_store.get_pending_approval_request(session)
    assert pending is not None
    assert runtime.routine_store.load(pending.routine_id).template == "morning-briefing"
    # The remembered schedule survived the combine.
    assert runtime.routine_store.load(pending.routine_id).schedule == "daily:07:00"
