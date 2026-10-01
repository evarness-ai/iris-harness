"""Email setup, the onboarding state machine (OSS plan R4), on the demo mailbox, offline.

Pinned:

* the steps run in order and each finished step's result is kept: setup resumes where
  it stopped, never re-running a finished step, and ``restart`` forgets the state only;
* a process that dies after any step is saved, or inside any step before it is, resumes
  to the run's true counts with one ledger row per step and one approval, and no email
  stranded between storage, the judge's queue and its release;
* nothing touches the mailbox before step 6: the judge runs without its label step, the
  demo provider refuses a write with no grant, and ``--yes``/``assume_defaults`` never
  approves -- it stops at step 6 with a pending approval-queue row;
* only an approved approval-queue row lets setup grant mailbox writes, and the grant
  names that row (R14); an approval answered in the Action Center counts on resume;
  a decline leaves the mailbox untouched and setup finishes with 0 labels;
* the vault-key check: a missing key waits for the owner's decision, defaults create
  one through ``sdk.vault.fix_master_key``, and an ``export`` line is shown once and
  never stored;
* the API and the CLI drive the same machine and see the same state;
* the scheduled sweep waits for setup (owner decision 2026-09-30): an account whose
  setup has not reached ``enable_sweep`` is not swept, that step turns it on, and an
  account setup never touched -- or one already being swept when setup (or a restart
  of it) began -- keeps being swept.

No network, no model: the judge and the namer are scripted, the embedder is a hash.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import typer
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from iris_harness.sdk import vault as sdk_vault
from iris_harness.sdk.approvals import ApprovalStore, respond_to_approval
from iris_harness.sdk.audit import AuditLog
from iris_harness.sdk.llm import JsonReply
from iris_harness.testing import no_network
from iris_personal.email import providers as providers_module
from iris_personal.email.store import EmailStore
from iris_personal.email.write_approvals import WriteApprovalStore
from iris_personal.plugins.email_workflows import cli_setup
from iris_personal.plugins.email_workflows.demo.provider import DEMO_ACCOUNT, DemoMailProvider
from iris_personal.plugins.email_workflows.judgments import JudgmentStore
from iris_personal.plugins.email_workflows.onboarding import (
    BLOCKED,
    COMPLETE,
    DECISION,
    STEPS,
    Inputs,
    Onboarding,
    OnboardingConfig,
    OnboardingDeps,
)
from iris_personal.plugins.email_workflows.onboarding_api import build_router

KEY = "x" * 43 + "="  # the fixture sets a real Fernet key; this is only a shape

HEARTBEATS = """
heartbeats:
  - name: email_sweep
    handler: email_sweep
    schedule: "15 6 * * *"
    enabled: true
  - name: email_judge
    handler: email_judge
    schedule: "30 6 * * *"
    enabled: true
"""


def _embed(texts: list[str], model: str) -> np.ndarray[Any, Any]:
    """Vectors by sender domain (so each sender clusters), plus a little per-text noise."""
    del model
    out = []
    for text in texts:
        domain = text.split("(", 1)[1].split(")", 1)[0] if "(" in text else text[:20]
        seed = int(hashlib.sha256(domain.encode()).hexdigest()[:8], 16)
        base = np.random.default_rng(seed).normal(size=32)
        noise = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
        vec = base + 0.05 * noise.normal(size=32)
        out.append(vec / np.linalg.norm(vec))
    return np.asarray(out, dtype=np.float32)


@dataclass
class Namer:
    calls: int = 0

    def complete_json(self, system: str, user: str) -> str:
        del system, user
        self.calls += 1
        return json.dumps(
            {"root": "personal", "branch": "mail", "leaf": f"group-{self.calls}", "rationale": ""}
        )


@dataclass
class Judge:
    calls: int = 0

    def __call__(self, system: str, user: str, schema: Mapping[str, Any]) -> JsonReply:
        del system, schema
        self.calls += 1
        text = user.lower()
        bucket = "bill" if ("statement" in text or "due" in text) else "fyi"
        return JsonReply(
            data={"bucket": bucket, "confidence": 0.9, "reason": "scripted"},
            latency_ms=1,
            model="fake",
        )


@dataclass
class CountingProvider:
    """The demo provider, counting fetches (a finished step must not fetch again)."""

    inner: DemoMailProvider
    fetches: int = 0
    name: str = "demo"

    def fetch_new(self, account_id: str, **kw: Any) -> Any:
        self.fetches += 1
        return self.inner.fetch_new(account_id, **kw)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.inner, attr)


@dataclass
class World:
    home: Path
    deps: OnboardingDeps
    provider: CountingProvider
    judge: Judge
    namer: Namer
    config: OnboardingConfig
    writes: WriteApprovalStore
    approvals: ApprovalStore
    events: list[tuple[str, Any]] = field(default_factory=list)

    def machine(self) -> Onboarding:
        return Onboarding(self.deps, self.config)

    def demo_labels(self) -> dict[str, list[str]]:
        path = self.home / "data" / "demo_mailbox.json"
        if not path.exists():
            return {}
        return dict(json.loads(path.read_text()).get("labels", {}))


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    from cryptography.fernet import Fernet

    home = tmp_path / "demo-home"
    data = home / "data"
    data.mkdir(parents=True)
    monkeypatch.setenv("IRIS_HOME", str(home))
    monkeypatch.setenv("IRIS_DEMO_HOME", str(home))
    monkeypatch.setenv("IRIS_DATA_DIR", str(data))
    monkeypatch.setenv("IRIS_VAULT_MASTER_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("IRIS_EMAIL_JUDGE", "1")
    monkeypatch.setenv("IRIS_EMAIL_JUDGE_LABELS", "1")
    heartbeats = tmp_path / "heartbeats.yaml"
    heartbeats.write_text(HEARTBEATS)
    # The gate's one file, in the data dir (IRIS_DATA_DIR), as every provider reads it.
    writes = WriteApprovalStore()
    approvals = ApprovalStore(db_path=tmp_path / "approvals.db")
    provider = CountingProvider(
        DemoMailProvider(state_path=data / "demo_mailbox.json", write_approvals=writes)
    )
    judge, namer = Judge(), Namer()
    events: list[tuple[str, Any]] = []
    deps = OnboardingDeps(
        data_dir=data,
        workspace_dir=home / "workspace",
        provider_for=lambda account: provider if account.startswith("demo") else None,
        judge_llm=lambda: judge,
        naming_client=lambda: namer,
        narrate=lambda: None,
        embedder=_embed,
        emit=lambda topic, payload: events.append((topic, payload)),
        approvals_db=tmp_path / "approvals.db",
        audit_db=tmp_path / "audit.db",
        heartbeats_path=heartbeats,
    )
    config = replace(OnboardingConfig.load(), judge_limit=500)
    return World(home, deps, provider, judge, namer, config, writes, approvals, events)


def _stop_at_approval(world: World) -> Any:
    with no_network():
        state = world.machine().run(DEMO_ACCOUNT, Inputs(assume_defaults=True))
    assert state.step == "label_approval", state.waiting_for
    return state


# -- the flow ------------------------------------------------------------------------


def test_defaults_run_to_the_label_approval_and_stop_there(world: World) -> None:
    state = _stop_at_approval(world)

    assert state.status == "waiting" and state.waiting_kind == DECISION
    assert set(state.results) == set(STEPS[: STEPS.index("label_approval")])
    fetch = state.results["fetch"]
    assert fetch["fetched"] == 200 and fetch["waiting_for_judge"] > 0
    assert state.results["discover"]["proposals"], "the hash embedder makes clusters"
    assert state.results["review_categories"]["accepted"], "defaults accept every category"
    classify = state.results["classify"]
    assert classify["judged"] == fetch["waiting_for_judge"] and world.judge.calls
    assert classify["knn_classified"] + classify["knn_to_review"] > 0  # kNN ran on accepted
    # Released mail and what kNN filed reach the process bus (the runtime's readers).
    topics = Counter(topic for topic, _ in world.events)
    assert topics["email.new_arrived"] > 0
    assert topics["email.classified"] == classify["knn_classified"]

    # Nothing touched the mailbox: no grant, no label, every judged email's label is due.
    assert world.writes.get(DEMO_ACCOUNT) is None
    assert world.demo_labels() == {}
    assert JudgmentStore(db_path=world.deps.email_db).labels_due(DEMO_ACCOUNT)

    # The approval is a governed, pending row with the preview on its card.
    row = world.approvals.get(state.approval_id)
    assert row is not None and row.status == "pending"
    assert row.run_id == state.run_id and row.signal == "email.mailbox_writes"
    assert row.card is not None and row.card.effect == "write"
    assert any("IRIS/" in line for line in row.card.lines)

    preview = world.machine().label_preview(DEMO_ACCOUNT)
    assert preview.total == classify["judged"] and preview.approval_id == state.approval_id
    assert all(g.samples for g in preview.groups)


def test_the_provider_refuses_a_write_before_the_grant(world: World) -> None:
    _stop_at_approval(world)
    with pytest.raises(PermissionError):
        world.provider.modify_labels(DEMO_ACCOUNT, ["demo-0001"], ["Label_1"], [])


def test_approving_grants_from_the_approved_row_and_writes_the_labels(world: World) -> None:
    waiting = _stop_at_approval(world)
    with no_network():
        state = world.machine().run(DEMO_ACCOUNT, Inputs(approve_writes=True, actor="cli:owner"))

    assert state.step == COMPLETE and state.status == "done"
    row = world.approvals.get(waiting.approval_id)
    assert row is not None and row.status == "approved" and row.response_actor == "cli:owner"
    grant = world.writes.get(DEMO_ACCOUNT)
    assert grant is not None and grant.approval_ref.startswith(f"{waiting.approval_id} [audit #")
    approval = state.results["label_approval"]
    assert approval["writes_approved"] and approval["labels_written"] > 0
    assert len(world.demo_labels()) == approval["labels_written"]
    assert not JudgmentStore(db_path=world.deps.email_db).labels_due(DEMO_ACCOUNT)

    summary = state.results["summary"]
    assert summary["labels_written"] == approval["labels_written"]
    assert summary["sweep"] == {"enabled": True, "schedule": "daily at 06:15"}
    hooks = summary["audit_by_hook"]
    assert hooks["approval_queue"] >= 2  # enqueued, then answered
    assert hooks["mailbox_writes"] == 1  # the grant, recorded as `iris email writes approve`
    grant_row = next(
        r
        for r in AuditLog(db_path=world.deps.audit_db).query(run_id=state.run_id)
        if r.hook_point == "mailbox_writes"
    )
    assert grant_row.plugin == "email_write_approvals" and grant_row.decision == "allow"
    assert json.loads(grant_row.payload_json) == {
        "account": DEMO_ACCOUNT,
        "actor": "cli:owner",
        "ref": waiting.approval_id,
    }
    assert (
        grant is not None and grant.approval_ref == f"{waiting.approval_id} [audit #{grant_row.id}]"
    )
    assert hooks["email_onboarding"] == len(STEPS) - 1  # every step but the summary itself
    assert state.results["first_digest"]["text"]


def test_no_audit_row_no_grant(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """R14: the grant waits for its ledger entry; a failed audit write grants nothing."""
    waiting = _stop_at_approval(world)

    class BrokenLedger:
        def record(self, **kw: Any) -> int:
            if kw.get("hook_point") == "mailbox_writes":
                raise sqlite3.OperationalError("disk I/O error")
            return AuditLog(db_path=world.deps.audit_db).record(**kw)

        def query(self, **kw: Any) -> Any:
            return AuditLog(db_path=world.deps.audit_db).query(**kw)

    monkeypatch.setattr(OnboardingDeps, "audit_log", lambda self: BrokenLedger())
    state = world.machine().advance(DEMO_ACCOUNT, Inputs(approve_writes=True))
    assert state.step == "label_approval" and state.waiting_kind == BLOCKED
    assert world.writes.get(DEMO_ACCOUNT) is None and world.demo_labels() == {}

    monkeypatch.undo()
    with no_network():
        state = world.machine().advance(DEMO_ACCOUNT)  # the approved row still counts
    grant = world.writes.get(DEMO_ACCOUNT)
    assert grant is not None and grant.approval_ref.startswith(waiting.approval_id)
    assert state.step == "first_digest"


def test_declining_keeps_the_mailbox_untouched(world: World) -> None:
    waiting = _stop_at_approval(world)
    with no_network():
        state = world.machine().run(DEMO_ACCOUNT, Inputs(approve_writes=False))

    assert state.step == COMPLETE
    row = world.approvals.get(waiting.approval_id)
    assert row is not None and row.status == "rejected"
    assert world.writes.get(DEMO_ACCOUNT) is None
    assert world.demo_labels() == {}
    assert state.results["summary"]["labels_written"] == 0
    assert not state.results["summary"]["writes_approved"]


def test_defaults_never_approve_on_resume(world: World) -> None:
    _stop_at_approval(world)
    state = world.machine().run(DEMO_ACCOUNT, Inputs(assume_defaults=True))
    assert state.step == "label_approval" and state.waiting_kind == DECISION
    assert world.writes.get(DEMO_ACCOUNT) is None


def test_an_approval_answered_in_the_action_center_counts(world: World) -> None:
    waiting = _stop_at_approval(world)
    respond_to_approval(
        waiting.approval_id,
        status="approved",
        actor="web",
        queue=world.deps.approval_queue(),
    )
    with no_network():
        state = world.machine().advance(DEMO_ACCOUNT)
    assert state.step == "first_digest"
    grant = world.writes.get(DEMO_ACCOUNT)
    assert grant is not None and grant.approval_ref.startswith(f"{waiting.approval_id} [audit #")


def test_resume_never_reruns_a_finished_step(world: World) -> None:
    waiting = _stop_at_approval(world)
    fetches, judged = world.provider.fetches, world.judge.calls
    again = world.machine().run(DEMO_ACCOUNT, Inputs(assume_defaults=True))
    assert again.results == waiting.results and again.run_id == waiting.run_id
    assert again.approval_id == waiting.approval_id  # the same pending row, not a second
    assert (world.provider.fetches, world.judge.calls) == (fetches, judged)


class _ProcessDied(BaseException):
    """The process going away: a ``BaseException``, so no ``except Exception`` in the
    machine can turn it into a waiting step -- what was saved is all that survives."""


def _setup_to_the_end(world: World) -> Any:
    """What the owner does: run with the defaults, approve at step 6, run to the end."""
    machine = world.machine()
    state = machine.run(DEMO_ACCOUNT, Inputs(assume_defaults=True))
    if state.step == "label_approval" and state.waiting_kind == DECISION:
        state = machine.run(DEMO_ACCOUNT, Inputs(approve_writes=True, actor="cli:owner"))
    return state


@pytest.mark.parametrize("stop_after", STEPS)
def test_setup_resumes_after_a_process_exit_at_every_step(
    world: World, monkeypatch: pytest.MonkeyPatch, stop_after: str
) -> None:
    """Launch issue #11: kill the process the moment ``stop_after`` is saved as finished;
    a fresh machine picks up at the next step, runs no finished step again, and
    completes."""
    from iris_personal.plugins.email_workflows.onboarding import OnboardingStore

    calls: Counter[str] = Counter()
    for step in STEPS:
        original = getattr(Onboarding, f"_{step}")

        def counted(self: Onboarding, state: Any, inputs: Inputs, *, _f=original, _s=step) -> Any:
            calls[_s] += 1
            return _f(self, state, inputs)

        monkeypatch.setattr(Onboarding, f"_{step}", counted)

    real_save = OnboardingStore.save
    armed = [True]

    def save_then_die(self: OnboardingStore, state: Any) -> None:
        real_save(self, state)
        if armed[0] and stop_after in state.results:
            armed[0] = False
            raise _ProcessDied(stop_after)

    monkeypatch.setattr(OnboardingStore, "save", save_then_die)

    with no_network(), pytest.raises(_ProcessDied):
        _setup_to_the_end(world)
    saved = world.machine().state(DEMO_ACCOUNT)
    assert saved is not None
    assert set(saved.results) == set(STEPS[: STEPS.index(stop_after) + 1])
    finished = dict(saved.results)
    before = Counter(calls)
    side_effects = (world.provider.fetches, world.judge.calls, world.namer.calls)

    with no_network():
        state = _setup_to_the_end(world)

    assert state.step == COMPLETE and state.status == "done"
    assert state.run_id == saved.run_id
    for step, result in finished.items():
        assert calls[step] == before[step], f"{step} ran again after the resume"
        if step != "summary":  # the summary is rendered from the run, not re-run
            assert state.results[step] == result, f"{step}'s result changed on resume"
    assert set(state.results) == set(STEPS)
    if "classify" in finished:  # the model work is behind it: none of it again
        assert (world.provider.fetches, world.judge.calls, world.namer.calls) == side_effects
    assert world.provider.fetches == 1

    # One ledger row per step for the whole run: a re-run step would write a second.
    rows = AuditLog(db_path=world.deps.audit_db).query(run_id=state.run_id)
    done = Counter(r.step_id for r in rows if r.hook_point == "email_onboarding")
    assert done == Counter(range(len(STEPS)))
    # One approval-queue row for the run, answered: step 6 never asked twice.
    asked = [
        r
        for status in ("pending", "approved", "rejected", "expired")
        for r in world.approvals.list_by_status(status)
        if r.run_id == state.run_id
    ]
    assert [(r.approval_id, r.status) for r in asked] == [(state.approval_id, "approved")]
    assert world.writes.get(DEMO_ACCOUNT) is not None
    assert len(world.demo_labels()) == state.results["label_approval"]["labels_written"]


def _die_before_saving(monkeypatch: pytest.MonkeyPatch, when: Callable[[Any], bool]) -> None:
    """Kill the process at the first save ``when`` matches, before it is written: the
    step's side effects are done, its result is lost."""
    from iris_personal.plugins.email_workflows.onboarding import OnboardingStore

    real_save = OnboardingStore.save
    armed = [True]

    def die_or_save(self: OnboardingStore, state: Any) -> None:
        if armed[0] and when(state):
            armed[0] = False
            raise _ProcessDied(state.step)
        real_save(self, state)

    monkeypatch.setattr(OnboardingStore, "save", die_or_save)


def _classified_by_knn(world: World) -> int:
    with sqlite3.connect(world.deps.email_db) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM emails WHERE account_id = ? AND triage_state = 'classified'",
            (DEMO_ACCOUNT,),
        ).fetchone()
    return int(row[0])


def _assert_setup_is_whole(world: World, state: Any) -> None:
    """What a finished setup must say, however many times a step ran: the run's true
    counts, one ledger row per step, one approval, every fetched email accounted for."""
    assert state.step == COMPLETE and state.status == "done"
    r = state.results
    fetch, classify, approval = r["fetch"], r["classify"], r["label_approval"]
    assert fetch["fetched"] == 200 == fetch["stored"]
    assert fetch["waiting_for_judge"] + fetch["released"] == fetch["fetched"]
    # Every email reached the readers exactly once: released at the fetch, or after
    # the judge read it. None stranded between storage and the judge's queue.
    arrived = [
        mid
        for topic, p in world.events
        if topic == "email.new_arrived"
        for mid in p.new_message_ids
    ]
    assert len(arrived) == len(set(arrived)) == 200
    assert classify["judged"] == fetch["waiting_for_judge"] == sum(classify["buckets"].values())
    assert world.judge.calls == classify["judged"]  # no email judged twice
    assert classify["knn_classified"] == _classified_by_knn(world) > 0
    categories = r["review_categories"]
    assert categories["inserted"] == len(categories["accepted"]) > 0
    assert approval["writes_approved"] and not approval["already_approved"]
    assert approval["approval_id"] == state.approval_id
    assert approval["labels_written"] == len(world.demo_labels()) > 0
    assert approval["labels_previewed"] == approval["labels_written"] + approval["labels_failed"]
    summary = r["summary"]
    assert (summary["fetched"], summary["judged"], summary["labels_written"]) == (
        200,
        classify["judged"],
        approval["labels_written"],
    )
    rows = AuditLog(db_path=world.deps.audit_db).query(run_id=state.run_id)
    done = Counter(row.step_id for row in rows if row.hook_point == "email_onboarding")
    assert done == Counter(range(len(STEPS)))
    asked = [
        row
        for status in ("pending", "approved", "rejected", "expired")
        for row in world.approvals.list_by_status(status)
        if row.run_id == state.run_id
    ]
    assert [(row.approval_id, row.status) for row in asked] == [(state.approval_id, "approved")]


@pytest.mark.parametrize("dies_in", STEPS)
def test_setup_resumes_after_a_process_exit_inside_every_step(
    world: World, monkeypatch: pytest.MonkeyPatch, dies_in: str
) -> None:
    """The counterpart of the test above: the process dies after ``dies_in`` did its
    work but before its result was saved. The step runs again on resume, finishes what
    the dead attempt left, and reports the run's true counts -- not the re-run's."""
    _die_before_saving(monkeypatch, lambda state: dies_in in state.results)
    with no_network(), pytest.raises(_ProcessDied):
        _setup_to_the_end(world)
    saved = world.machine().state(DEMO_ACCOUNT)
    assert saved is not None and saved.step == dies_in and dies_in not in saved.results

    with no_network():
        state = _setup_to_the_end(world)

    _assert_setup_is_whole(world, state)
    # Only the step that died re-ran its model work; discovery's naming is the one
    # step that simply repeats (it writes nothing outside its proposals file).
    expected_fetches = 2 if dies_in == "fetch" else 1
    assert world.provider.fetches == expected_fetches


@pytest.mark.parametrize(
    "seam",
    [
        # The fetch stored the mail, then died before the judge's queue admitted it.
        "admit_swept_mail",
        # The judge recorded its verdicts, then died before releasing the mail.
        "release",
    ],
)
def test_mail_a_dead_attempt_left_half_handled_is_handed_on(
    world: World, monkeypatch: pytest.MonkeyPatch, seam: str
) -> None:
    """Mail must not strand between two of a step's effects: the re-run finds what the
    dead attempt did not hand on (from the run's effects) and hands it on."""
    from iris_personal.plugins.email_workflows import judge_wiring

    real = getattr(judge_wiring, seam)
    armed = [True]

    def die_once(*args: Any, **kwargs: Any) -> Any:
        if armed[0]:
            armed[0] = False
            raise _ProcessDied(seam)
        return real(*args, **kwargs)

    monkeypatch.setattr(judge_wiring, seam, die_once)
    with no_network(), pytest.raises(_ProcessDied):
        _setup_to_the_end(world)
    with no_network():
        state = _setup_to_the_end(world)
    _assert_setup_is_whole(world, state)


def test_an_approval_asked_before_a_crash_is_not_asked_again(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 6 enqueued its approval row, then died before saving that it waits on it:
    the resume finds the row this run asked for instead of asking a second time."""
    _die_before_saving(
        monkeypatch, lambda state: state.step == "label_approval" and state.status == "waiting"
    )
    with no_network(), pytest.raises(_ProcessDied):
        _setup_to_the_end(world)
    saved = world.machine().state(DEMO_ACCOUNT)
    assert saved is not None and saved.approval_id is None

    with no_network():
        state = _setup_to_the_end(world)
    _assert_setup_is_whole(world, state)


def test_restart_forgets_the_state_only(world: World) -> None:
    waiting = _stop_at_approval(world)
    world.machine().run(DEMO_ACCOUNT, Inputs(approve_writes=True))
    stored = EmailStore(db_path=world.deps.email_db).count(DEMO_ACCOUNT, include_held=True)

    assert world.machine().restart(DEMO_ACCOUNT) is True
    assert world.machine().state(DEMO_ACCOUNT) is None
    assert EmailStore(db_path=world.deps.email_db).count(DEMO_ACCOUNT, include_held=True) == stored
    assert world.writes.get(DEMO_ACCOUNT) is not None  # the approval is the owner's; kept

    with no_network():
        fresh = world.machine().run(DEMO_ACCOUNT, Inputs(assume_defaults=True))
    assert fresh.run_id != waiting.run_id
    # Writes were approved before: step 6 says so instead of asking again.
    assert fresh.step == COMPLETE
    assert fresh.results["label_approval"]["already_approved"] is True


def test_without_defaults_the_category_review_waits_for_the_owner(world: World) -> None:
    with no_network():
        state = world.machine().run(DEMO_ACCOUNT, Inputs())
    assert state.step == "review_categories" and state.waiting_kind == DECISION
    first = state.results["discover"]["proposals"][0]["cluster_id"]
    with no_network():
        state = world.machine().advance(DEMO_ACCOUNT, Inputs(accept_categories=(first,)))
    assert len(state.results["review_categories"]["accepted"]) == 1
    assert state.step == "classify"


# -- connect ------------------------------------------------------------------------


def test_the_demo_mailbox_is_refused_outside_a_demo_home(
    world: World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "owner-home"))
    state = world.machine().advance(DEMO_ACCOUNT, Inputs(assume_defaults=True))
    assert state.step == "connect" and state.waiting_kind == BLOCKED
    assert "demo home" in state.waiting_for


def test_an_unconnected_account_names_its_login_command(world: World) -> None:
    state = world.machine().advance("imap:you@example.com", Inputs(assume_defaults=True))
    assert state.step == "connect" and state.waiting_kind == BLOCKED
    assert "iris auth imap login --user you@example.com" in state.waiting_for


@dataclass
class KeyScript:
    status: str
    fix_outcome: str = "stored"
    fixes: int = 0

    def status_of(self, *, read_keyring: bool) -> sdk_vault.MasterKeyStatus:
        del read_keyring
        return sdk_vault.MasterKeyStatus(self.status, f"scripted {self.status}")  # type: ignore[arg-type]

    def fix(self) -> sdk_vault.KeyFix:
        self.fixes += 1
        line = f"export IRIS_VAULT_MASTER_KEY={KEY}" if self.fix_outcome == "export" else None
        return sdk_vault.KeyFix(self.fix_outcome, "scripted", export_line=line)  # type: ignore[arg-type]


def _script_key(monkeypatch: pytest.MonkeyPatch, script: KeyScript) -> None:
    monkeypatch.setattr(sdk_vault, "master_key_status", script.status_of)
    monkeypatch.setattr(sdk_vault, "fix_master_key", script.fix)


def test_a_missing_key_waits_for_the_owner_and_defaults_create_it(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = KeyScript("absent")
    _script_key(monkeypatch, script)
    state = world.machine().advance(DEMO_ACCOUNT, Inputs())
    assert state.step == "connect" and state.waiting_kind == DECISION
    assert "master key" in state.waiting_for and script.fixes == 0

    state = world.machine().advance(DEMO_ACCOUNT, Inputs(assume_defaults=True))
    assert script.fixes == 1
    assert state.step == "fetch" and state.results["connect"]["key"] == "stored"


def test_an_export_line_is_shown_once_and_never_stored(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script_key(monkeypatch, KeyScript("no_keyring", fix_outcome="export"))
    state = world.machine().advance(DEMO_ACCOUNT, Inputs(create_master_key=True))
    assert state.waiting_kind == BLOCKED and KEY in state.notice
    with sqlite3.connect(world.deps.email_db) as conn:
        dumped = "".join(str(r) for r in conn.execute("SELECT * FROM email_onboarding"))
    assert KEY not in dumped
    stored = world.machine().state(DEMO_ACCOUNT)
    assert stored is not None and stored.notice == ""


# -- the API and the CLI ------------------------------------------------------------


def _client(world: World) -> TestClient:
    app = FastAPI()
    app.include_router(build_router(lambda: world.deps))
    return TestClient(app)


def test_the_api_drives_the_same_machine(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "iris_personal.plugins.email_workflows.onboarding_api.Onboarding",
        lambda deps: Onboarding(deps, world.config),
    )
    client = _client(world)
    base = f"/api/v1/email/onboarding/{DEMO_ACCOUNT}"
    assert client.get(base).status_code == 404

    with no_network():
        body = client.post(
            base + "/advance", json={"accept_defaults": True, "until_waiting": True}
        ).json()
    assert body["step"] == "label_approval" and body["waiting_kind"] == DECISION
    assert [s["step"] for s in body["steps"]] == list(STEPS)
    assert body["sweep"]["swept"] is False and body["sweep"]["state"] == "held"
    assert world.writes.get(DEMO_ACCOUNT) is None

    preview = client.get(base + "/label-preview").json()
    assert preview["total"] > 0 and preview["approval_id"] == body["approval_id"]
    assert preview["already_approved"] is False
    # The Setup screen draws the groups and these; the card's lines are the same words.
    assert preview["status"] == ""
    assert preview["notes"] and preview["lines"][-len(preview["notes"]) :] == preview["notes"]
    assert f"--account {DEMO_ACCOUNT}" in preview["notes"][-1]

    # The CLI sees exactly the state the API left.
    assert world.machine().state(DEMO_ACCOUNT).approval_id == body["approval_id"]  # type: ignore[union-attr]

    with no_network():
        done = client.post(base + "/approve-writes", json={"approve": True, "actor": "web:me"})
    assert done.status_code == 200 and done.json()["step"] == "first_digest"
    assert done.json()["sweep"]["swept"] is False  # enable_sweep is still ahead
    grant = world.writes.get(DEMO_ACCOUNT)
    assert grant is not None and grant.approval_ref.startswith(body["approval_id"])

    # approve-writes answers step 6 only.
    assert client.post(base + "/approve-writes", json={"approve": True}).status_code == 409
    overview = client.get("/api/v1/email/onboarding").json()
    assert [s["account_id"] for s in overview["setups"]] == [DEMO_ACCOUNT]
    assert client.post(base + "/restart").json() == {"restarted": True}
    assert client.get(base).status_code == 404


def test_the_api_rejects_a_malformed_account(world: World) -> None:
    response = _client(world).post("/api/v1/email/onboarding/nobody/advance", json={})
    assert response.status_code == 422


# -- what the web Setup screen renders (R4 + R17): the server's words, never its own ------


def test_a_blocked_connect_carries_its_login_command_as_a_field(world: World) -> None:
    account = "imap:you@example.com"
    body = (
        _client(world)
        .post(f"/api/v1/email/onboarding/{account}/advance", json={"until_waiting": True})
        .json()
    )
    assert body["step"] == "connect" and body["waiting_kind"] == BLOCKED
    # The screen copies this; the same command the waiting text names.
    assert body["connect_command"] == "iris auth imap login --user you@example.com"
    assert body["connect_command"] in body["waiting_for"]
    assert body["rendered"] == {}


def test_the_overview_offers_each_providers_login_and_the_demo_in_a_demo_home(
    world: World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    overview = _client(world).get("/api/v1/email/onboarding").json()
    hints = {h["provider"]: h["command"] for h in overview["connect_hints"]}
    # The CLI's no-mailbox hint, from onboarding.yaml; the demo has no login command.
    assert hints == {
        "gmail": "iris auth gmail login --user you@example.com",
        "imap": "iris auth imap login --user you@example.com",
    }
    assert overview["demo_account"] == DEMO_ACCOUNT  # the fixture runs in a demo home

    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "owner-home"))
    assert _client(world).get("/api/v1/email/onboarding").json()["demo_account"] is None


def test_each_finished_step_is_rendered_in_the_clis_words(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_personal.plugins.email_workflows.onboarding import render_step

    monkeypatch.setattr(
        "iris_personal.plugins.email_workflows.onboarding_api.Onboarding",
        lambda deps: Onboarding(deps, world.config),
    )
    with no_network():
        body = (
            _client(world)
            .post(
                f"/api/v1/email/onboarding/{DEMO_ACCOUNT}/advance",
                json={"accept_defaults": True, "until_waiting": True},
            )
            .json()
        )
    assert set(body["rendered"]) == set(body["results"])
    for step, text in body["rendered"].items():
        assert render_step(world.config, step, body["results"][step]).endswith("\n" + text)


def test_setup_posts_stay_behind_the_consoles_write_gate() -> None:
    """R17: Setup's confirmed action is the approval-queue row; the routes that answer
    it here are gated exactly as the Action Center's answer to it is. No read-only
    device and not the service secret."""
    from iris_harness.foundation.auth import SERVICE_PRINCIPAL, Principal
    from iris_harness.server.iris_api.main import (
        _device_may_act_on_reminder,
        _is_gated_write,
        _service_may_write,
    )

    read_only = Principal(kind="device", scope="read", device_id="d1")
    base = f"/api/v1/email/onboarding/{DEMO_ACCOUNT}"
    for path in (base + "/advance", base + "/approve-writes", base + "/restart"):
        assert _is_gated_write("POST", path), path
        assert not _device_may_act_on_reminder(read_only, "POST", path), path
        assert not _service_may_write(SERVICE_PRINCIPAL, "POST", path), path
    assert _is_gated_write("POST", "/governance/approvals/a1/respond")
    assert not _is_gated_write("GET", "/api/v1/email/onboarding")


def _cli(world: World, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Any]:
    monkeypatch.setattr(cli_setup, "cli_deps", lambda *, interactive: world.deps)
    monkeypatch.setattr(
        "iris_personal.plugins.email_workflows.onboarding.OnboardingConfig.load",
        classmethod(lambda cls, config_dir=None: world.config),
    )
    monkeypatch.setattr(providers_module, "mount_cli_mail_providers", lambda: ())
    app = typer.Typer()
    app.command()(cli_setup.cmd_email_setup)
    runner = CliRunner()

    def invoke(*args: str, input: str | None = None) -> Any:
        with no_network():
            return runner.invoke(app, list(args), input=input)

    return invoke


def test_cli_yes_without_approve_writes_stops_at_step_6(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoke = _cli(world, monkeypatch)
    result = invoke("--provider", "demo", "--yes")
    assert result.exit_code == cli_setup.EXIT_WAITING, result.output
    assert "Label preview and approval" in result.output
    assert world.writes.get(DEMO_ACCOUNT) is None and world.demo_labels() == {}

    status = invoke("--account", DEMO_ACCOUNT, "--status")
    assert status.exit_code == 0 and "[>] Label preview and approval" in status.output
    assert "Scheduled sweep: waiting for email setup" in status.output

    done = invoke("--account", DEMO_ACCOUNT, "--yes", "--approve-writes")
    assert done.exit_code == 0, done.output
    assert "What IRIS just did" in done.output
    assert world.writes.get(DEMO_ACCOUNT) is not None and world.demo_labels()
    status = invoke("--account", DEMO_ACCOUNT, "--status")
    assert "Scheduled sweep: on (email setup (onboard-" in status.output


def test_a_setup_started_in_the_cli_resumes_in_the_web(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Public issue #19: start `iris email setup` in a terminal, finish it on the Setup
    screen. The web console's API reads the run the CLI left, at the step it stopped,
    and the owner's answer there resumes that same run -- nothing restarts."""
    monkeypatch.setattr(
        "iris_personal.plugins.email_workflows.onboarding_api.Onboarding",
        lambda deps: Onboarding(deps, world.config),
    )
    invoke = _cli(world, monkeypatch)
    started = invoke("--provider", "demo", "--yes")
    assert started.exit_code == cli_setup.EXIT_WAITING, started.output
    cli_state = world.machine().state(DEMO_ACCOUNT)
    assert cli_state is not None

    client = _client(world)
    base = f"/api/v1/email/onboarding/{DEMO_ACCOUNT}"
    seen = client.get(base).json()
    assert seen["run_id"] == cli_state.run_id
    assert seen["step"] == "label_approval" and seen["waiting_kind"] == DECISION
    assert [s["step"] for s in seen["steps"] if s["done"]] == list(
        STEPS[: STEPS.index("label_approval")]
    )
    overview = client.get("/api/v1/email/onboarding").json()
    assert [s["account_id"] for s in overview["setups"]] == [DEMO_ACCOUNT]

    with no_network():
        done = client.post(base + "/approve-writes", json={"approve": True, "actor": "web:me"})
    assert done.status_code == 200, done.text
    assert done.json()["run_id"] == cli_state.run_id
    assert done.json()["step"] == "first_digest"
    assert world.writes.get(DEMO_ACCOUNT) is not None


def test_cli_interactive_step_6_defaults_to_no(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoke = _cli(world, monkeypatch)
    # Enter at every prompt: accept all categories (yes), then step 6 (default no).
    result = invoke("--provider", "demo", input="\n\n\n")
    assert result.exit_code == 0, result.output
    assert world.writes.get(DEMO_ACCOUNT) is None
    assert "Mailbox writes not approved" in result.output


def test_cli_rejects_contradictory_write_flags(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _cli(world, monkeypatch)("--provider", "demo", "--approve-writes", "--decline-writes")
    assert result.exit_code == 2


# -- the scheduled sweep waits for setup ------------------------------------------------


def _sweep(world: World) -> tuple[Any, list[str]]:
    """One scheduled sweep tick over the world's stores; returns the run and the
    accounts it fetched."""
    from iris_harness.sdk.types import HeartbeatDefinition
    from iris_personal.email.providers import FetchResult
    from iris_personal.email.sweep import EmailSweepHandler

    calls: list[str] = []

    def fetch(account_id: str, *, store: Any, max_messages: int) -> FetchResult:
        del store, max_messages
        calls.append(account_id)
        return FetchResult(account_id, 0, (), "x", False)

    providers_module.register_mail_provider(world.provider)  # type: ignore[arg-type]
    try:
        run = EmailSweepHandler(
            bus=None,
            accounts_store=world.deps.accounts(),
            email_store=EmailStore(db_path=world.deps.email_db),
            fetcher=fetch,
        )(HeartbeatDefinition(name="email_sweep", handler="email_sweep", schedule="interval:900"))
    finally:
        providers_module.clear_mail_providers()
    return run, calls


def test_an_unfinished_setup_is_not_swept(world: World) -> None:
    _stop_at_approval(world)
    run, calls = _sweep(world)
    assert calls == []
    assert f"waiting for email setup: {DEMO_ACCOUNT}" in run.output
    status = world.machine().sweep_status(DEMO_ACCOUNT)
    assert status["swept"] is False and status["state"] == "held"


def test_enable_sweep_turns_the_sweep_on(world: World) -> None:
    _stop_at_approval(world)
    with no_network():
        state = world.machine().run(DEMO_ACCOUNT, Inputs(approve_writes=True))
    assert state.step == COMPLETE
    assert state.results["enable_sweep"]["account_swept"] is True
    _run, calls = _sweep(world)
    assert calls == [DEMO_ACCOUNT]


def test_an_account_setup_never_touched_is_swept(world: World) -> None:
    """The owner's accounts connected before setup existed have no setup row."""
    world.deps.accounts().add(provider="demo", address=DEMO_ACCOUNT.split(":", 1)[1])
    _run, calls = _sweep(world)
    assert calls == [DEMO_ACCOUNT]
    assert world.machine().sweep_status(DEMO_ACCOUNT)["state"] == "default"


def test_an_account_already_swept_keeps_its_sweep_when_setup_begins(world: World) -> None:
    """Setting up an account the sweep already fetches never stops its mail."""
    world.deps.accounts().add(provider="demo", address=DEMO_ACCOUNT.split(":", 1)[1])
    store = EmailStore(db_path=world.deps.email_db)
    store.ensure_schema()
    world.provider.fetch_new(DEMO_ACCOUNT, store=store, max_messages=10)  # a past sweep
    _stop_at_approval(world)
    _run, calls = _sweep(world)
    assert calls == [DEMO_ACCOUNT]
    assert world.machine().sweep_status(DEMO_ACCOUNT)["reason"] == (
        "already swept when email setup began"
    )


def test_a_restart_keeps_the_sweep_as_it_was(world: World) -> None:
    """A finished setup restarted: still swept through the new run. An unfinished one
    restarted: still held (restart drops setup's state, never the gate)."""
    _stop_at_approval(world)
    world.machine().restart(DEMO_ACCOUNT)
    assert _sweep(world)[1] == []  # held: its setup never turned the sweep on
    with no_network():
        world.machine().run(DEMO_ACCOUNT, Inputs(assume_defaults=True, approve_writes=True))
    assert _sweep(world)[1] == [DEMO_ACCOUNT]

    world.machine().restart(DEMO_ACCOUNT)
    with no_network():
        state = world.machine().advance(DEMO_ACCOUNT, Inputs(assume_defaults=True))
    assert state.step == "fetch"  # the new run is at its start ...
    assert _sweep(world)[1] == [DEMO_ACCOUNT]  # ... and the sweep goes on


def test_a_held_account_has_a_health_row(world: World) -> None:
    from iris_personal.plugins.email_workflows.sweep_health import TARGET, sweep_wait_checks

    _stop_at_approval(world)
    address = DEMO_ACCOUNT.split(":", 1)[1]
    rows = sweep_wait_checks(
        accounts=lambda: [(DEMO_ACCOUNT, address), ("demo:other@example.com", "other")],
        provider_for=lambda account: object(),
    )
    (row,) = rows
    assert (row.target, row.subject) == (TARGET, address)
    assert row.action == f"iris email setup --account {DEMO_ACCOUNT}"


# -- the CLI seam's providers ----------------------------------------------------------


def test_offered_providers_mount_only_when_a_command_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers_module, "_cli_factories", {})
    monkeypatch.setattr(providers_module, "_providers", {})
    built: list[str] = []

    def good() -> Any:
        built.append("fake")
        return type("P", (), {"name": "fake"})()

    def broken() -> Any:
        raise ImportError("extra not installed")

    providers_module.offer_cli_mail_provider("fake", good)
    providers_module.offer_cli_mail_provider("broken", broken)
    assert built == [] and providers_module.mail_provider_for("fake:x@example.com") is None

    assert providers_module.mount_cli_mail_providers() == ("fake",)
    assert providers_module.mail_provider_for("fake:x@example.com") is not None
    assert providers_module.mount_cli_mail_providers() == ()  # already mounted: not rebuilt
    assert built == ["fake"]


def test_audit_rows_carry_counts_never_content(world: World) -> None:
    state = _stop_at_approval(world)
    rows = AuditLog(db_path=world.deps.audit_db).query(run_id=state.run_id)
    setup_rows = [r for r in rows if r.hook_point == "email_onboarding"]
    assert len(setup_rows) == STEPS.index("label_approval")
    emails = EmailStore(db_path=world.deps.email_db)
    subjects = {m.subject for m in emails.list_recent(DEMO_ACCOUNT, limit=50, include_held=True)}
    for row in setup_rows:
        assert not any(s and s in row.payload_json for s in subjects)


def test_no_setup_ledger_row_names_the_address_in_its_reason(world: World) -> None:
    """Every row setup writes -- each step, the grant -- names the account's provider in
    its reason; the account id is in the payload, which the proof bundle pseudonymises."""
    _stop_at_approval(world)
    with no_network():
        world.machine().run(DEMO_ACCOUNT, Inputs(approve_writes=True, actor="cli:owner"))
    address = DEMO_ACCOUNT.split(":", 1)[1]
    rows = AuditLog(db_path=world.deps.audit_db).query()
    assert {r.hook_point for r in rows} >= {"email_onboarding", "mailbox_writes"}
    for row in rows:
        assert address not in row.reason, (row.hook_point, row.reason)
    grant = next(r for r in rows if r.hook_point == "mailbox_writes")
    assert grant.reason == "mailbox writes approved for a demo account"


def test_the_governed_namer_asks_with_the_naming_schema() -> None:
    from iris_personal.plugins.email_workflows.discovery import (
        NAMING_SCHEMA,
        GovernedNamingClient,
    )

    seen: list[Mapping[str, Any]] = []

    def ask(system: str, user: str, schema: Mapping[str, Any]) -> JsonReply:
        seen.append(schema)
        return JsonReply(data={"root": "work", "branch": "b", "leaf": "l"}, latency_ms=1, model="f")

    raw = GovernedNamingClient(ask).complete_json("sys", "user")
    assert json.loads(raw) == {"root": "work", "branch": "b", "leaf": "l"}
    assert seen == [NAMING_SCHEMA]
