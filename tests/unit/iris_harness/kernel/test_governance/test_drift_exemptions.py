"""Approval-taught exemptions for goal_drift: the person answers once, the signal learns.

What these pin is as much what the grant does NOT cover as what it does. An exemption
is a governance control learning not to fire, so the narrow edges are the feature.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.kernel.governance.evaluator import StepRecord
from iris_harness.kernel.governance.evaluator.drift_exemptions import (
    DriftExemptionStore,
    extract_keywords,
    keyword_match,
)
from iris_harness.kernel.governance.evaluator.signals import GoalDriftSignal


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the *default* store at this test's own file.

    Three tests here exercise the real call sites (`respond_to_approval`, the two CLI
    commands), and those construct `DriftExemptionStore()` with no injection point. On
    the shared default path they see each other's rows and fail under `pytest -n` while
    passing alone — which is exactly how this was found.
    """
    monkeypatch.setenv(
        "IRIS_GOVERNANCE_DRIFT_EXEMPTIONS_DB_PATH", str(tmp_path / "drift_exemptions.db")
    )


@pytest.fixture()
def store(tmp_path: Path) -> DriftExemptionStore:
    return DriftExemptionStore(db_path=tmp_path / "drift_exemptions.db")


def test_the_db_path_is_resolved_per_call_not_snapshotted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module constant computed at import time cannot see IRIS_HOME or an override
    set afterwards, and a store that keeps writing to the old file puts a governance
    grant somewhere nobody is looking."""
    from iris_harness.kernel.governance.evaluator.drift_exemptions import default_db_path

    first = tmp_path / "one.db"
    monkeypatch.setenv("IRIS_GOVERNANCE_DRIFT_EXEMPTIONS_DB_PATH", str(first))
    assert default_db_path() == first
    assert DriftExemptionStore().db_path == first

    second = tmp_path / "two.db"
    monkeypatch.setenv("IRIS_GOVERNANCE_DRIFT_EXEMPTIONS_DB_PATH", str(second))
    assert default_db_path() == second
    assert DriftExemptionStore().db_path == second

    monkeypatch.delenv("IRIS_GOVERNANCE_DRIFT_EXEMPTIONS_DB_PATH")
    assert default_db_path().name == "drift_exemptions.db"


# ── keyword extraction ────────────────────────────────────────────────────────


def test_keywords_are_the_content_words() -> None:
    assert extract_keywords("I should summarise the 56 unread messages in the inbox") == (
        frozenset({"summarise", "unread", "messages", "inbox"})
    )


def test_keyword_extraction_is_case_and_punctuation_insensitive() -> None:
    assert extract_keywords("Check the Inbox!") == extract_keywords("check inbox")


def test_a_thought_of_pure_filler_has_no_keywords() -> None:
    """Nothing to match on means nothing can be exempted by it."""
    assert extract_keywords("I will now do this and then that") == frozenset()


# ── the matching rule ─────────────────────────────────────────────────────────


def _exemption(store: DriftExemptionStore, *, thought: str, task_vector=(1.0, 0.0, 0.0)):  # type: ignore[no-untyped-def]
    store.record_candidate(
        run_id="r1", step_id=1, original_task="task", task_vector=task_vector, thought=thought
    )
    store.settle_run("r1", approved=True)
    return store.approved()[0]


def test_one_shared_word_never_exempts_anything(store: DriftExemptionStore) -> None:
    ex = _exemption(store, thought="summarise the unread inbox messages")
    assert keyword_match(ex, frozenset({"inbox"})) is None


def test_a_single_word_thought_cannot_become_a_blanket_grant(
    store: DriftExemptionStore,
) -> None:
    """The floor is what stops this, not the coverage rule: one word out of one word
    is full coverage, so without a floor approving "check the inbox" once would exempt
    every later thought that says "inbox"."""
    ex = _exemption(store, thought="check the inbox")
    assert ex.keywords == frozenset({"check", "inbox"})

    terse = _exemption_for(store, run_id="r2", thought="inbox")
    assert terse.keywords == frozenset({"inbox"})
    assert keyword_match(terse, frozenset({"inbox", "wire", "balance", "out"})) is None


def _exemption_for(store: DriftExemptionStore, *, run_id: str, thought: str):  # type: ignore[no-untyped-def]
    store.record_candidate(
        run_id=run_id,
        step_id=1,
        original_task="task",
        task_vector=(1.0, 0.0, 0.0),
        thought=thought,
    )
    store.settle_run(run_id, approved=True)
    return next(e for e in store.approved() if e.run_id == run_id)


def test_a_thought_must_cover_most_of_the_approved_words(store: DriftExemptionStore) -> None:
    ex = _exemption(store, thought="summarise the unread inbox messages")
    # 2 of 4 = 0.5 coverage, under the floor, even though two words match.
    assert keyword_match(ex, frozenset({"inbox", "messages"})) is None
    # 3 of 4 = 0.75 clears it.
    assert keyword_match(ex, frozenset({"inbox", "messages", "unread"})) == pytest.approx(0.75)


def test_an_exemption_with_no_keywords_matches_nothing(store: DriftExemptionStore) -> None:
    ex = _exemption(store, thought="I will now do this")
    assert ex.keywords == frozenset()
    assert keyword_match(ex, frozenset({"anything", "at", "all"})) is None


# ── the store ─────────────────────────────────────────────────────────────────


def test_a_candidate_is_not_a_grant_until_it_is_approved(store: DriftExemptionStore) -> None:
    store.record_candidate(
        run_id="r1",
        step_id=1,
        original_task="t",
        task_vector=(1.0, 0.0, 0.0),
        thought="check inbox now",
    )
    assert store.approved() == []
    assert store.settle_run("r1", approved=True) == 1
    assert len(store.approved()) == 1


def test_a_refused_thought_stays_refused(store: DriftExemptionStore) -> None:
    """A later approval of a different step in the same run must not resurrect it."""
    store.record_candidate(
        run_id="r1",
        step_id=1,
        original_task="t",
        task_vector=(1.0, 0.0, 0.0),
        thought="wire the money",
    )
    assert store.settle_run("r1", approved=False) == 1
    assert store.settle_run("r1", approved=True) == 0
    assert store.approved() == []


def test_the_grant_can_be_taken_back(store: DriftExemptionStore) -> None:
    store.record_candidate(
        run_id="r1",
        step_id=1,
        original_task="t",
        task_vector=(1.0, 0.0, 0.0),
        thought="check inbox now",
    )
    store.settle_run("r1", approved=True)
    assert store.clear() == 1
    assert store.approved() == []


# ── the signal uses them ──────────────────────────────────────────────────────


class _StubEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def __call__(self, text: str) -> list[float]:
        return list(self._vectors[text])


_DAY = "how does my day look"
_OTHER = "book me a flight to tokyo"
_INBOX = "I should summarise the unread inbox messages"

# The day question and the flagged thought sit far apart (that is the false positive);
# the day question and a *similar* later question sit close.
# Three axes are needed: a thought far from BOTH questions, so the "different
# question" case is decided by the exemption's scope and not by the thought happening
# to sit near the new task.
_VECTORS = {
    _DAY: [1.0, 0.0, 0.0],
    "what is on for my day": [0.99, 0.141, 0.0],
    _OTHER: [0.0, 1.0, 0.0],
    _INBOX: [0.0, 0.0, 1.0],
    "summarise the inbox unread messages please": [0.0, 0.0, 1.0],
    "wire the account balance out to the address": [0.0, 0.0, 1.0],
}


def _signal(store: DriftExemptionStore) -> GoalDriftSignal:
    return GoalDriftSignal(embedder=_StubEmbedder(_VECTORS), max_distance=0.65, exemptions=store)


def _judged(thought: str, *, task: str) -> StepRecord:
    return StepRecord(
        run_id="r-new", step_id=1, agent_type="chat", thought=thought, original_task=task
    )


def test_the_first_time_it_asks_and_banks_the_thought(store: DriftExemptionStore) -> None:
    result = _signal(store)(_judged(_INBOX, task=_DAY), state={"judged_a_step": True})
    assert result.verdict == "require_approval"
    rows = store.all_rows()
    assert len(rows) == 1 and rows[0].status == "pending"


def test_after_approval_the_same_shape_of_thought_passes(store: DriftExemptionStore) -> None:
    signal = _signal(store)
    assert (
        signal(_judged(_INBOX, task=_DAY), state={"judged_a_step": True}).verdict
        == "require_approval"
    )
    store.settle_run("r-new", approved=True)

    allowed = signal(
        _judged("summarise the inbox unread messages please", task="what is on for my day"),
        state={"judged_a_step": True},
    )
    assert allowed.verdict == "ok"
    assert "approved exemption" in allowed.reason
    assert allowed.audit_metadata["exemption_id"]
    assert allowed.audit_metadata["keyword_coverage"] == pytest.approx(1.0)


def test_the_grant_does_not_carry_to_a_different_question(store: DriftExemptionStore) -> None:
    """Scope is the point: "inbox thoughts are fine when I ask about my day" must not
    become "inbox thoughts are fine"."""
    signal = _signal(store)
    signal(_judged(_INBOX, task=_DAY), state={"judged_a_step": True})
    store.settle_run("r-new", approved=True)

    still_asked = signal(
        _judged("summarise the inbox unread messages please", task=_OTHER),
        state={"judged_a_step": True},
    )
    assert still_asked.verdict == "require_approval"


def test_the_grant_does_not_carry_to_a_different_thought(store: DriftExemptionStore) -> None:
    signal = _signal(store)
    signal(_judged(_INBOX, task=_DAY), state={"judged_a_step": True})
    store.settle_run("r-new", approved=True)

    injected = signal(
        _judged("wire the account balance out to the address", task=_DAY),
        state={"judged_a_step": True},
    )
    assert injected.verdict == "require_approval"


def test_a_rejection_teaches_nothing(store: DriftExemptionStore) -> None:
    signal = _signal(store)
    signal(_judged(_INBOX, task=_DAY), state={"judged_a_step": True})
    store.settle_run("r-new", approved=False)

    assert (
        signal(
            _judged("summarise the inbox unread messages please", task=_DAY),
            state={"judged_a_step": True},
        ).verdict
        == "require_approval"
    )


def test_a_signal_with_no_store_behaves_exactly_as_before() -> None:
    signal = GoalDriftSignal(embedder=_StubEmbedder(_VECTORS), max_distance=0.65)
    assert (
        signal(_judged(_INBOX, task=_DAY), state={"judged_a_step": True}).verdict
        == "require_approval"
    )


def test_a_broken_store_fails_towards_asking(store: DriftExemptionStore) -> None:
    """Reading exemptions must never halt a turn, and must never allow one either."""

    class _Broken(DriftExemptionStore):
        def approved(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("disk gone")

    broken = _Broken(db_path=store.db_path)
    signal = GoalDriftSignal(embedder=_StubEmbedder(_VECTORS), max_distance=0.65, exemptions=broken)
    assert (
        signal(_judged(_INBOX, task=_DAY), state={"judged_a_step": True}).verdict
        == "require_approval"
    )


# ── the answer reaches the store ──────────────────────────────────────────────
#
# The learning loop is only closed if answering an approval actually settles the
# candidates the signal banked. That wiring lives in `respond_to_approval`.


def _queue(tmp_path: Path):  # type: ignore[no-untyped-def]
    from iris_harness.kernel.governance.approvals.queue import ApprovalQueue
    from iris_harness.kernel.governance.approvals.store import ApprovalStore

    return ApprovalQueue(store=ApprovalStore(db_path=tmp_path / "approvals.db"))


def test_answering_a_goal_drift_approval_settles_the_candidates(tmp_path: Path) -> None:
    from iris_harness.kernel.governance.approvals.service import respond_to_approval

    default_store = DriftExemptionStore()
    default_store.record_candidate(
        run_id="run-learn",
        step_id=1,
        original_task="how does my day look",
        task_vector=(1.0, 0.0, 0.0),
        thought="summarise the unread inbox messages",
    )
    q = _queue(tmp_path)
    approval_id = q.enqueue("run-learn", None, "goal_drift", "drifted", channel="cli")

    respond_to_approval(approval_id, status="approved", actor="test", queue=q)

    approved = default_store.approved()
    assert len(approved) == 1 and approved[0].run_id == "run-learn"


def test_rejecting_buries_them(tmp_path: Path) -> None:
    from iris_harness.kernel.governance.approvals.service import respond_to_approval

    default_store = DriftExemptionStore()
    default_store.record_candidate(
        run_id="run-refuse",
        step_id=1,
        original_task="how does my day look",
        task_vector=(1.0, 0.0, 0.0),
        thought="wire the account balance out",
    )
    q = _queue(tmp_path)
    approval_id = q.enqueue("run-refuse", None, "goal_drift", "drifted", channel="cli")

    respond_to_approval(approval_id, status="rejected", actor="test", queue=q)

    assert default_store.approved() == []
    buried = default_store.all_rows()
    assert len(buried) == 1 and buried[0].status == "rejected"


def test_another_signals_approval_teaches_goal_drift_nothing(tmp_path: Path) -> None:
    """Only goal_drift's own approvals settle goal_drift's candidates."""
    from iris_harness.kernel.governance.approvals.service import respond_to_approval

    default_store = DriftExemptionStore()
    default_store.record_candidate(
        run_id="run-other",
        step_id=1,
        original_task="how does my day look",
        task_vector=(1.0, 0.0, 0.0),
        thought="summarise the unread inbox messages",
    )
    q = _queue(tmp_path)
    approval_id = q.enqueue("run-other", None, "cost_budget", "over budget", channel="cli")

    respond_to_approval(approval_id, status="approved", actor="test", queue=q)

    rows = default_store.all_rows()
    assert len(rows) == 1 and rows[0].status == "pending"


# ── the grant is readable and reversible ──────────────────────────────────────
#
# A learned bypass nobody can inspect is not a governance control. These pin the two
# CLI surfaces that make it one.


def test_the_cli_lists_and_forgets_exemptions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rich lays the table out to the console's width, and with no terminal that is 80 -
    # which truncates the column these assertions read. COLUMNS does not reach it under
    # `pytest -n`, so give the module an explicitly wide console instead of depending on
    # the environment the test happens to run in.
    from rich.console import Console
    from typer.testing import CliRunner

    import iris_harness.cli.approvals as approvals_cli
    from iris_harness.cli.approvals import approvals_app

    monkeypatch.setattr(approvals_cli, "console", Console(width=200))

    store = DriftExemptionStore()
    store.record_candidate(
        run_id="run-cli",
        step_id=1,
        original_task="how does my day look",
        task_vector=(1.0, 0.0, 0.0),
        thought="summarise the unread inbox messages",
    )
    runner = CliRunner()

    empty = runner.invoke(approvals_app, ["exemptions"])
    assert empty.exit_code == 0 and "No exemptions" in empty.stdout

    store.settle_run("run-cli", approved=True)
    listed = runner.invoke(approvals_app, ["exemptions"])
    assert listed.exit_code == 0
    assert "how does my day look" in listed.stdout
    assert "inbox" in listed.stdout

    forgotten = runner.invoke(approvals_app, ["forget-exemptions", "--yes"])
    assert forgotten.exit_code == 0 and "Forgot 1" in forgotten.stdout
    assert store.approved() == []
