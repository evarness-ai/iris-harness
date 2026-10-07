"""Email setup: the resumable onboarding state machine (OSS plan R4).

One flow, two surfaces: ``iris email setup`` (``cli_setup.py``) and
``/api/v1/email/onboarding/*`` (``onboarding_api.py``) both call :class:`Onboarding`
and only render what it returns. The steps, in order (``STEPS``)::

    connect -> fetch -> discover -> review_categories -> classify -> label_approval
            -> first_digest -> review_queue -> enable_sweep -> summary

**State** is one row per account in ``email_onboarding`` (``email.db``, beside the mail,
under ``sdk.persistence.data_dir()``): the run id, the current step, each finished step's
result, and what the flow is waiting for. A step is done when its result is saved, and
only then gets its one ledger row (``_audit_finished`` writes any a crash left out). A
finished step never runs again, so setup resumes where it stopped.

**A step that dies before it is saved runs again**, so each is safe to re-run and reports
the run's true counts, not the last attempt's. What a step does is recorded as it
happens, one row per item in ``email_onboarding_effects`` (``fetch``: each stored and
each released email; ``review_categories``: each category it adds; ``classify``: each
email kNN filed and each judged one released; ``label_approval``: each label written),
and the judge's verdicts carry the setup's run id. The re-run finishes what the dead
attempt left half done (mail stored but never queued for the judge, judged mail never
released, a granted approval whose labels were never written) and asks no approval
twice. Discovery is the exception that just repeats: it writes nothing outside its
proposals file, so a re-run costs its naming calls again and nothing else. Restart
drops the row only: fetched mail, judgments, accepted categories and any mailbox-write
approval stay.

**Waiting.** A step that cannot finish says why and what it needs. ``decision`` means
the owner decides (create the vault key, accept categories, approve mailbox writes);
``blocked`` means something outside setup must happen first (log in, start the model);
``activity`` means a long step (``fetch``, ``classify``) is running in the background
on the harness's Activity spine -- call ``advance`` again to poll it, no answer needed.
``assume_defaults`` (``--yes``) takes the default of every decision except one:

**Nothing touches the mailbox before ``label_approval``.** Setup judges without the label
step (``judge_and_release(labels=False)``); the preview shows the labels IRIS would
write (count per label, sample subjects). The approval is governed: setup enqueues an
approval-queue row (it shows in the Action Center and ``iris approvals``) and only the
owner's explicit yes -- ``approve_writes=True``, never implied by defaults -- answers
it, through the same ``respond_to_approval`` every surface uses, so the answer is on the
ledger. Only an *approved* row lets setup grant writes, recorded as ``iris email writes
approve`` records them: an audit row (plugin ``email_write_approvals``), then
``approve_mailbox_writes(account, "<approval id> [audit #<row>]")``; no audit row, no
grant. Every provider checks that grant before any write (``email.write_approvals``).
A yes given in the Action Center counts too: setup reads the row when it resumes. A no
leaves the mailbox untouched and setup goes on.

**The scheduled sweep waits for setup** (owner decision 2026-09-30). When setup starts
for an account the sweep has never fetched, it holds the account out of the sweep
(``email.sweep_gate``); ``enable_sweep`` releases it. An account already fetched when
its setup starts -- one connected before setup existed, or a restart of a finished
setup -- keeps being swept, and so does any account setup never touched.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from iris_harness.foundation.logsafe import log_safe
from iris_harness.sdk.persistence import data_path, ensure_columns, sqlite_conn

logger = logging.getLogger(__name__)

SHIPPED_CONFIG = Path(__file__).with_name("onboarding.yaml")
OVERRIDE_RELATIVE_PATH = Path("email") / "onboarding.yaml"

STEPS: tuple[str, ...] = (
    "connect",
    "fetch",
    "discover",
    "review_categories",
    "classify",
    "label_approval",
    "first_digest",
    "review_queue",
    "enable_sweep",
    "summary",
)
COMPLETE = "complete"

# Row status.
IN_PROGRESS = "in_progress"
WAITING = "waiting"
DONE = "done"

# What a waiting step needs.
DECISION = "decision"
BLOCKED = "blocked"
# A long step (fetch, classify) running in the background on the Activity spine;
# the caller just polls (calls ``advance`` again) rather than answering anything.
ACTIVITY = "activity"

PLUGIN = "email_onboarding"
HOOK = "email_onboarding"

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat()


# -- configuration ------------------------------------------------------------------


@dataclass(frozen=True)
class OnboardingConfig:
    """``onboarding.yaml``: the step titles, how each provider connects, the numbers."""

    titles: Mapping[str, str]
    connect_commands: Mapping[str, str]
    fetch_max_messages: int = 500
    fetch_cold_start_days: int = 90
    discover_min_corpus: int = 40
    discover_samples: int = 3
    judge_limit: int | None = None
    knn_limit: int = 500
    preview_samples: int = 3
    approval_signal: str = "email.mailbox_writes"
    approval_timeout_minutes: int = 10080
    approval_title: str = "Let IRIS label mail in {account}"
    approval_never: str = ""
    approval_revoke: str = ""
    review_samples: int = 5

    @classmethod
    def load(cls, config_dir: Path | None = None) -> OnboardingConfig:
        raw: dict[str, Any] = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8")) or {}
        if config_dir is not None:
            override = Path(config_dir) / OVERRIDE_RELATIVE_PATH
            if override.is_file():
                raw.update(yaml.safe_load(override.read_text(encoding="utf-8")) or {})
        titles = {str(k): str(v) for k, v in (raw.get("steps") or {}).items()}
        missing = [s for s in STEPS if s not in titles]
        if missing:
            raise ValueError(f"onboarding.yaml: steps must name every step; missing {missing}")
        providers = raw.get("providers") or {}
        fetch = raw.get("fetch") or {}
        discover = raw.get("discover") or {}
        classify = raw.get("classify") or {}
        approval = raw.get("label_approval") or {}
        review = raw.get("review_queue") or {}
        judge_limit = classify.get("judge_limit")
        return cls(
            titles=titles,
            connect_commands={
                str(k): str((v or {}).get("connect") or "") for k, v in providers.items()
            },
            fetch_max_messages=int(fetch.get("max_messages", 500)),
            fetch_cold_start_days=int(fetch.get("cold_start_days", 90)),
            discover_min_corpus=int(discover.get("min_corpus", 40)),
            discover_samples=int(discover.get("samples", 3)),
            judge_limit=int(judge_limit) if judge_limit is not None else None,
            knn_limit=int(classify.get("knn_limit", 500)),
            preview_samples=int(approval.get("samples", 3)),
            approval_signal=str(approval.get("signal", "email.mailbox_writes")),
            approval_timeout_minutes=int(approval.get("timeout_minutes", 10080)),
            approval_title=str(approval.get("title", cls.approval_title)),
            approval_never=str(approval.get("never", "")),
            approval_revoke=str(approval.get("revoke", "")),
            review_samples=int(review.get("samples", 5)),
        )


# -- state --------------------------------------------------------------------------


@dataclass(frozen=True)
class OnboardingState:
    """One account's setup. ``step`` is the first step not done (``complete`` at the end);
    ``results`` holds each finished step's result by step name."""

    account_id: str
    run_id: str
    provider: str
    step: str = STEPS[0]
    status: str = IN_PROGRESS
    waiting_kind: str = ""
    waiting_for: str = ""
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    approval_id: str | None = None
    # The Activity id a ``waiting_kind == ACTIVITY`` step is polling. Set only while
    # that wait holds for that same activity; cleared the moment it resolves (done,
    # blocked, or a decision) so the step starts a fresh one if it is asked to run again.
    activity_id: str | None = None
    started_at: str = ""
    updated_at: str = ""
    completed_at: str | None = None
    # Shown once, never stored: an ``export`` line holding a new vault key.
    notice: str = ""

    @property
    def complete(self) -> bool:
        return self.step == COMPLETE

    def as_dict(self, config: OnboardingConfig | None = None) -> dict[str, Any]:
        steps = [
            {
                "step": name,
                "title": config.titles.get(name, name) if config else name,
                "done": name in self.results,
                "current": name == self.step,
            }
            for name in STEPS
        ]
        return {
            "account_id": self.account_id,
            "run_id": self.run_id,
            "provider": self.provider,
            "step": self.step,
            "status": self.status,
            "waiting_kind": self.waiting_kind,
            "waiting_for": self.waiting_for,
            "approval_id": self.approval_id,
            "activity_id": self.activity_id,
            "steps": steps,
            "results": self.results,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "notice": self.notice,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS email_onboarding (
    account_id   TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    provider     TEXT NOT NULL,
    step         TEXT NOT NULL,
    status       TEXT NOT NULL,
    waiting_kind TEXT NOT NULL DEFAULT '',
    waiting_for  TEXT NOT NULL DEFAULT '',
    results      TEXT NOT NULL DEFAULT '{}',
    approval_id  TEXT,
    activity_id  TEXT,
    started_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    completed_at TEXT
);

-- What a run's steps did, recorded as it happens (one row per item, so recording it
-- again is a no-op): a step re-run after a crash reports the run's true counts.
CREATE TABLE IF NOT EXISTS email_onboarding_effects (
    run_id TEXT NOT NULL,
    step   TEXT NOT NULL,
    kind   TEXT NOT NULL,
    item   TEXT NOT NULL,
    PRIMARY KEY (run_id, step, kind, item)
);
"""


@dataclass
class OnboardingStore:
    """The ``email_onboarding`` table in ``email.db``."""

    db_path: Path = field(default_factory=lambda: data_path("email.db"))

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite_conn(self.db_path, row_factory=sqlite3.Row) as conn:
            conn.executescript(_SCHEMA)
        # On a connection of its own under BEGIN IMMEDIATE (#201): a read of ``table_info`` then an
        # ``ALTER`` raised "duplicate column name" for the process that lost a race to open an
        # older email.db.
        ensure_columns(self.db_path, "email_onboarding", {"activity_id": "TEXT"})
        with sqlite_conn(self.db_path, row_factory=sqlite3.Row) as conn:
            yield conn

    def get(self, account_id: str) -> OnboardingState | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM email_onboarding WHERE account_id = ?", (account_id,)
            ).fetchone()
        return _from_row(row) if row is not None else None

    def list(self) -> list[OnboardingState]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM email_onboarding ORDER BY started_at").fetchall()
        return [_from_row(r) for r in rows]

    def save(self, state: OnboardingState) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO email_onboarding (account_id, run_id, provider, step, status, "
                "waiting_kind, waiting_for, results, approval_id, activity_id, started_at, "
                "updated_at, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(account_id) DO UPDATE SET run_id = excluded.run_id, "
                "provider = excluded.provider, step = excluded.step, "
                "status = excluded.status, waiting_kind = excluded.waiting_kind, "
                "waiting_for = excluded.waiting_for, results = excluded.results, "
                "approval_id = excluded.approval_id, activity_id = excluded.activity_id, "
                "started_at = excluded.started_at, "
                "updated_at = excluded.updated_at, completed_at = excluded.completed_at",
                (
                    state.account_id,
                    state.run_id,
                    state.provider,
                    state.step,
                    state.status,
                    state.waiting_kind,
                    state.waiting_for,
                    json.dumps(state.results, sort_keys=True, default=str),
                    state.approval_id,
                    state.activity_id,
                    state.started_at,
                    state.updated_at,
                    state.completed_at,
                ),
            )

    def delete(self, account_id: str) -> bool:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM email_onboarding_effects WHERE run_id IN "
                "(SELECT run_id FROM email_onboarding WHERE account_id = ?)",
                (account_id,),
            )
            cur = conn.execute("DELETE FROM email_onboarding WHERE account_id = ?", (account_id,))
        return cur.rowcount > 0

    def record_effects(self, run_id: str, step: str, kind: str, items: Iterable[str]) -> None:
        """Record that ``step`` of ``run_id`` did ``kind`` to each of ``items``."""
        rows = [(run_id, step, kind, str(item)) for item in items]
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO email_onboarding_effects (run_id, step, kind, item) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def effect_items(self, run_id: str, step: str, kind: str) -> set[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT item FROM email_onboarding_effects WHERE run_id = ? AND step = ? "
                "AND kind = ?",
                (run_id, step, kind),
            ).fetchall()
        return {str(r[0]) for r in rows}

    def effect_counts(self, run_id: str, step: str) -> Counter[str]:
        """Items per kind that ``step`` of ``run_id`` recorded."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT kind, COUNT(*) FROM email_onboarding_effects WHERE run_id = ? "
                "AND step = ? GROUP BY kind",
                (run_id, step),
            ).fetchall()
        return Counter({str(kind): int(n) for kind, n in rows})


def _from_row(row: sqlite3.Row) -> OnboardingState:
    return OnboardingState(
        account_id=row["account_id"],
        run_id=row["run_id"],
        provider=row["provider"],
        step=row["step"],
        status=row["status"],
        waiting_kind=row["waiting_kind"],
        waiting_for=row["waiting_for"],
        results=json.loads(row["results"] or "{}"),
        approval_id=row["approval_id"],
        activity_id=row["activity_id"],
        started_at=row["started_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


# -- what the owner said, and what setup reaches for --------------------------------


@dataclass(frozen=True)
class Inputs:
    """The owner's answers for this advance. ``None`` means "not said"."""

    # Take the default of every decision except mailbox writes (``--yes``).
    assume_defaults: bool = False
    create_master_key: bool | None = None
    # Cluster ids to accept; ``None`` = not said (the default is every acceptable one).
    accept_categories: tuple[int, ...] | None = None
    # The step-6 answer. Only an explicit True approves; defaults never do.
    approve_writes: bool | None = None
    actor: str = "cli"
    channel: str = "cli"


def _default_provider_for(account_id: str) -> Any:
    from iris_personal.email.providers import mail_provider_for

    return mail_provider_for(account_id)


def _default_emit(topic: str, payload: Any) -> None:
    from iris_harness.sdk.events import get_default_bus

    get_default_bus().emit_sync(topic, payload)


def _none() -> Any:
    return None


@dataclass
class ActivityJobs:
    """What a long step (fetch, classify) needs from the harness's Activity spine:
    submit work in the background, and check how a submitted job is doing. Duck-typed
    on purpose -- ``submit`` matches ``ActivityRunner.submit``'s keywords and ``get``
    matches ``ActivityStore.get``, but this module never imports either class."""

    submit: Callable[..., str]
    get: Callable[[str], Any]


@dataclass
class OnboardingDeps:
    """Where setup reads and writes, and the calls it makes. Production defaults; a test
    passes its own paths, stubs the model calls and the embedder."""

    data_dir: Path = field(default_factory=lambda: data_path("email.db").parent)
    config_dir: Path | None = None
    workspace_dir: Path | None = None
    provider_for: Callable[[str], Any] = _default_provider_for
    # The judge's governed call (``judge.llm_from_router``), or None: no model tier.
    judge_llm: Callable[[], Any] = _none
    # The category namer (``discovery.GovernedNamingClient``), or None: unnamed proposals.
    naming_client: Callable[[], Any] = _none
    # The digest's narrative call (``make_narrative_llm_call``), or None: deterministic.
    narrate: Callable[[], Any] = _none
    # ``embed_corpus``'s shape; None = the local MiniLM embedder.
    embedder: Any = None
    emit: Callable[[str, Any], None] = _default_emit
    # An :class:`ActivityJobs`, or None: fetch/classify then run inline, exactly as
    # they did before the Activity spine existed (a test, or a host without one).
    activities: Callable[[], ActivityJobs | None] = _none
    # Read the OS keyring for the vault key (a terminal); False for a server or script.
    read_keyring: bool = False
    approvals_db: Path | None = None
    audit_db: Path | None = None
    accounts_db: Path | None = None
    heartbeats_path: Path | None = None

    @property
    def email_db(self) -> Path:
        return self.data_dir / "email.db"

    @property
    def iris_db(self) -> Path:
        return self.accounts_db or self.data_dir / "iris.db"

    def audit_log(self) -> Any:
        from iris_harness.sdk.audit import AuditLog, audit_db_path

        return AuditLog(db_path=self.audit_db or audit_db_path())

    def approval_queue(self) -> Any:
        from iris_harness.sdk.approvals import ApprovalQueue, ApprovalStore

        return ApprovalQueue(
            store=ApprovalStore(db_path=self.approvals_db), audit_log=self.audit_log()
        )

    def write_approvals(self) -> Any:
        from iris_personal.email.write_approvals import WriteApprovalStore

        # The one gate every provider and the label sync read (the data dir's file).
        return WriteApprovalStore()

    def sweep_gate(self) -> Any:
        from iris_personal.email.sweep_gate import SweepGate

        # Beside the mail and the sweep's cursors: the file the sweep itself reads.
        return SweepGate(db_path=self.email_db)

    def accounts(self) -> Any:
        from iris_personal.email.accounts import EmailAccountStore

        store = EmailAccountStore(db_path=self.iris_db)
        store.ensure_schema()
        return store


@dataclass
class _RecordingSyncStore:
    """The mail store a provider fetches into, telling ``record`` the ids of every batch
    it stored. Providers store a batch before they move their cursor, so a fetch that
    dies after storing re-fetches whatever was stored but not recorded."""

    inner: Any
    record: Callable[[list[str]], None]

    def upsert_many(self, messages: Iterable[Any]) -> int:
        batch = list(messages)
        stored = int(self.inner.upsert_many(batch))
        self.record([m.id for m in batch])
        return stored

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


# -- a step's answer ----------------------------------------------------------------


@dataclass(frozen=True)
class StepOutcome:
    """``done`` with its ``result``, or waiting (``kind`` + ``waiting_for``)."""

    done: bool
    result: dict[str, Any] = field(default_factory=dict)
    kind: str = ""
    waiting_for: str = ""
    approval_id: str | None = None
    activity_id: str | None = None
    notice: str = ""

    @classmethod
    def finished(cls, **result: Any) -> StepOutcome:
        return cls(done=True, result=result)

    @classmethod
    def decide(cls, why: str, **kw: Any) -> StepOutcome:
        return cls(done=False, kind=DECISION, waiting_for=why, **kw)

    @classmethod
    def blocked(cls, why: str, **kw: Any) -> StepOutcome:
        return cls(done=False, kind=BLOCKED, waiting_for=why, **kw)

    @classmethod
    def running(cls, activity_id: str, *, why: str) -> StepOutcome:
        return cls(done=False, kind=ACTIVITY, waiting_for=why, activity_id=activity_id)


# -- the label preview --------------------------------------------------------------


@dataclass(frozen=True)
class LabelGroup:
    bucket: str
    label: str
    count: int
    samples: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "label": self.label,
            "count": self.count,
            "samples": list(self.samples),
        }


@dataclass(frozen=True)
class LabelPreview:
    """The labels IRIS would write to ``account_id`` once approved."""

    account_id: str
    groups: tuple[LabelGroup, ...] = ()
    # Emails whose IRIS label would come off (the owner called them promo).
    removals: int = 0
    # The provider can label at all; the setting IRIS_EMAIL_JUDGE_LABELS is on.
    labelling: bool = True
    labels_enabled: bool = True
    approval_id: str | None = None
    already_approved: bool = False

    @property
    def total(self) -> int:
        return sum(g.count for g in self.groups)

    def status(self) -> str:
        """Why nothing would be labelled, or "" when labels are due."""
        if not self.labels_enabled:
            return "Labels are off (IRIS_EMAIL_JUDGE_LABELS=0): no label would be written."
        if not self.labelling:
            return "This mailbox cannot take labels; nothing would be labelled."
        if not self.groups and not self.removals:
            return "No labels are due yet."
        return ""

    def notes(self, config: OnboardingConfig) -> list[str]:
        """What IRIS never does, and how to take the approval back."""
        out: list[str] = []
        if config.approval_never:
            out.append(config.approval_never)
        if config.approval_revoke:
            out.append(config.approval_revoke.format(account=self.account_id))
        return out

    def lines(self, config: OnboardingConfig) -> list[str]:
        out: list[str] = [self.status()] if self.status() else []
        for group in self.groups:
            sample = "; ".join(group.samples)
            out.append(
                f"{group.label}: {group.count} email(s)" + (f" (e.g. {sample})" if sample else "")
            )
        if self.removals:
            out.append(f"IRIS labels removed from {self.removals} email(s) marked promo.")
        return out + self.notes(config)

    def as_dict(self, config: OnboardingConfig) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "groups": [g.as_dict() for g in self.groups],
            "removals": self.removals,
            "total": self.total,
            "labelling": self.labelling,
            "labels_enabled": self.labels_enabled,
            "approval_id": self.approval_id,
            "already_approved": self.already_approved,
            "status": self.status(),
            "notes": self.notes(config),
            "lines": self.lines(config),
        }


# -- the machine ----------------------------------------------------------------------


class OnboardingError(ValueError):
    """A request setup cannot take (no such account, a step out of order)."""


class Onboarding:
    """The state machine. Every surface calls this; none decides anything itself."""

    def __init__(self, deps: OnboardingDeps, config: OnboardingConfig | None = None) -> None:
        self.deps = deps
        self.config = config or OnboardingConfig.load(deps.config_dir)
        self.store = OnboardingStore(db_path=deps.email_db)
        self._steps: dict[str, Callable[[OnboardingState, Inputs], StepOutcome]] = {
            "connect": self._connect,
            "fetch": self._fetch,
            "discover": self._discover,
            "review_categories": self._review_categories,
            "classify": self._classify,
            "label_approval": self._label_approval,
            "first_digest": self._first_digest,
            "review_queue": self._review_queue,
            "enable_sweep": self._enable_sweep,
            "summary": self._summary,
        }

    # -- lifecycle --------------------------------------------------------------------

    def state(self, account_id: str) -> OnboardingState | None:
        return self.store.get(account_id)

    def states(self) -> list[OnboardingState]:
        return self.store.list()

    def start(self, account_id: str) -> OnboardingState:
        """The account's setup, created (at ``connect``) when it has none."""
        account_id = account_id.strip()
        if ":" not in account_id:
            raise OnboardingError(
                f"not an account id: {account_id!r} (expected provider:address, "
                "e.g. imap:you@example.com)"
            )
        existing = self.store.get(account_id)
        if existing is not None:
            return existing
        now = _now()
        state = OnboardingState(
            account_id=account_id,
            run_id=f"onboard-{uuid.uuid4().hex[:12]}",
            provider=account_id.split(":", 1)[0],
            started_at=now,
            updated_at=now,
        )
        self._gate_sweep(state)
        self.store.save(state)
        return state

    def _gate_sweep(self, state: OnboardingState) -> None:
        """A new setup holds the scheduled sweep off an account it has never fetched,
        until ``enable_sweep``. An account with a gate row already (a restart) keeps it;
        one the sweep already fetched keeps being swept -- setup never stops the mail of
        an account that was being kept current."""
        from iris_personal.email.store import EmailStore

        gate = self.deps.sweep_gate()
        if gate.get(state.account_id) is not None:
            return
        emails = EmailStore(db_path=self.deps.email_db)
        emails.ensure_schema()
        if emails.has_synced(state.account_id):
            gate.release(state.account_id, "already swept when email setup began")
        else:
            gate.hold(
                state.account_id,
                f"email setup ({state.run_id}) has not reached "
                f"'{self.config.titles.get('enable_sweep', 'enable_sweep')}'",
            )

    def restart(self, account_id: str) -> bool:
        """Forget the account's setup (its state only: mail, judgments, categories, any
        mailbox-write approval and the sweep's gate stay -- an account that was being
        swept keeps being swept). True when there was one."""
        return self.store.delete(account_id)

    def sweep_status(self, account_id: str) -> dict[str, Any]:
        """Whether the scheduled sweep takes ``account_id``, and why."""
        entry = self.deps.sweep_gate().get(account_id)
        if entry is None:
            return {
                "swept": True,
                "state": "default",
                "reason": "email setup never held this account",
                "since": None,
            }
        return {
            "swept": not entry.held,
            "state": entry.state,
            "reason": entry.reason,
            "since": entry.updated_at,
        }

    def advance(self, account_id: str, inputs: Inputs | None = None) -> OnboardingState:
        """Run the current step once. Done: record its result, move on. Otherwise: record
        what it waits for. Returns the state after."""
        inputs = inputs or Inputs()
        with _lock:
            state = self.start(account_id)
            # A process that died between saving a step and its ledger row: write it now.
            self._audit_finished(state)
            if state.complete:
                return state
            step = state.step
            try:
                outcome = self._steps[step](state, inputs)
            except OnboardingError:
                raise
            except Exception as exc:  # a step's failure waits; it never loses the state
                logger.exception("email setup: %s failed for %s", step, log_safe(account_id))
                outcome = StepOutcome.blocked(f"{step} failed: {type(exc).__name__}: {exc}")
            state = self._apply(state, step, outcome)
            # The result first, then its ledger row: a step is done when it is saved, and
            # a step that is not saved re-runs on resume -- it must not be on the ledger.
            self.store.save(state)
            self._audit_finished(state)
        return replace(state, notice=outcome.notice)

    def run(self, account_id: str, inputs: Inputs | None = None) -> OnboardingState:
        """Advance until setup waits or completes."""
        state = self.advance(account_id, inputs)
        notice = state.notice
        while state.status == IN_PROGRESS and not state.complete:
            state = self.advance(account_id, inputs)
            notice = notice or state.notice
        return replace(state, notice=notice)

    def _apply(self, state: OnboardingState, step: str, outcome: StepOutcome) -> OnboardingState:
        now = _now()
        approval_id = outcome.approval_id or state.approval_id
        if not outcome.done:
            # An activity id is only ever carried forward while still waiting on THAT
            # activity. A decision or a block -- even one reached mid-poll, like the
            # activity having failed -- clears it: asked to run this step again, it
            # must submit a fresh one rather than keep polling a resolved row.
            carried_activity = outcome.activity_id if outcome.kind == ACTIVITY else None
            return replace(
                state,
                status=WAITING,
                waiting_kind=outcome.kind,
                waiting_for=outcome.waiting_for,
                approval_id=approval_id,
                activity_id=carried_activity,
                updated_at=now,
            )
        results = {**state.results, step: outcome.result}
        index = STEPS.index(step)
        following = STEPS[index + 1] if index + 1 < len(STEPS) else COMPLETE
        return replace(
            state,
            step=following,
            status=DONE if following == COMPLETE else IN_PROGRESS,
            activity_id=None,
            waiting_kind="",
            waiting_for="",
            results=results,
            approval_id=approval_id,
            updated_at=now,
            completed_at=now if following == COMPLETE else None,
        )

    def _audit_finished(self, state: OnboardingState) -> None:
        """One ledger row per saved step, written once: the steps of ``state`` that are
        finished and have no row yet get theirs. A ledger hiccup leaves the row for the
        next advance; it never strands the owner mid-setup."""
        if not state.results:
            return
        try:
            recorded = {
                r.step_id
                for r in self.deps.audit_log().query(run_id=state.run_id, plugin=PLUGIN)
                if r.hook_point == HOOK
            }
        except Exception:  # noqa: BLE001 — retried on the next advance
            logger.warning("email setup: could not read the ledger for %s", state.run_id)
            return
        for index, step in enumerate(STEPS):
            if step in state.results and index not in recorded:
                self._audit(state, step, state.results[step])

    def _audit(self, state: OnboardingState, step: str, result: Mapping[str, Any]) -> None:
        """One ledger row per finished step: counts only, never content."""
        from iris_personal.email.accounts import ledger_account_label

        counts = {k: v for k, v in result.items() if isinstance(v, bool | int | float)}
        try:
            self.deps.audit_log().record(
                run_id=state.run_id,
                step_id=STEPS.index(step),
                agent_type="email_setup",
                hook_point=HOOK,
                plugin=PLUGIN,
                decision="allow",
                severity="info",
                # The account id is in the payload; the reason names only its provider.
                reason=f"email setup: {step} done for {ledger_account_label(state.account_id)}",
                payload={"account_id": state.account_id, "step": step, **counts},
            )
        except Exception:  # a ledger hiccup must not strand the owner mid-setup
            logger.warning("email setup: audit write failed for %s", step, exc_info=True)

    # -- 1. connect ---------------------------------------------------------------------

    def _connect(self, state: OnboardingState, inputs: Inputs) -> StepOutcome:
        key = self._key_check(inputs)
        if not key.done:
            return key
        provider = state.provider
        address = state.account_id.split(":", 1)[1]
        accounts = self.deps.accounts()
        if provider == "demo":
            from .demo.home import HOME_ENV, in_demo_home
            from .demo.provider import DEMO_ACCOUNT, DemoMailProvider

            if not in_demo_home():
                return StepOutcome.blocked(
                    "the demo mailbox is set up only inside a demo home, never your own "
                    "profile: run `iris email demo` first, then set IRIS_HOME and "
                    f"{HOME_ENV} to that home"
                )
            if state.account_id != DEMO_ACCOUNT:
                raise OnboardingError(f"the demo mailbox's account is {DEMO_ACCOUNT}")
            if accounts.get(DEMO_ACCOUNT) is None:
                accounts.add(provider="demo", address=address)
            if self.deps.provider_for(DEMO_ACCOUNT) is None:
                from iris_personal.email.providers import register_mail_provider

                register_mail_provider(DemoMailProvider())
        account = accounts.get(state.account_id)
        command = self.connect_command(state.account_id)
        if account is None or not account.active:
            how = f"run `{command}`" if command else f"connect it with the {provider} plugin"
            return StepOutcome.blocked(
                f"{state.account_id} is not connected yet: {how}, then run setup again"
            )
        if self.deps.provider_for(state.account_id) is None:
            return StepOutcome.blocked(
                f"no {provider!r} mail provider is mounted: enable its plugin in the profile"
            )
        return StepOutcome.finished(
            provider=provider, address=account.address, key=key.result.get("key", "")
        )

    def connect_command(self, account_id: str) -> str:
        """The command that connects ``account_id`` (its provider plugin's login), or ""
        when its provider names none (the demo mailbox, a provider without a login)."""
        provider, _, address = account_id.partition(":")
        return self.config.connect_commands.get(provider, "").format(address=address)

    def connect_hints(self, address: str = "you@example.com") -> list[dict[str, str]]:
        """How each provider is connected, for an owner with no mailbox yet."""
        return [
            {"provider": name, "command": command.format(address=address)}
            for name, command in self.config.connect_commands.items()
            if command
        ]

    def demo_account(self) -> str | None:
        """The synthetic mailbox's account id when setup may connect it here (inside a
        demo home only, never the owner's profile), else None."""
        from .demo.home import in_demo_home
        from .demo.provider import DEMO_ACCOUNT

        return DEMO_ACCOUNT if in_demo_home() else None

    def _key_check(self, inputs: Inputs) -> StepOutcome:
        """The vault master key every governed call needs (#741)."""
        from iris_harness.sdk.vault import fix_master_key, master_key_status

        status = master_key_status(read_keyring=self.deps.read_keyring)
        if status.present:
            return StepOutcome.finished(key=status.source)
        if status.source == "not_read":
            # A server or script never opens a Keychain dialog; the key is resolved by
            # the first governed call, and `iris doctor` checks it interactively.
            return StepOutcome.finished(key="not_read")
        if status.source == "invalid":
            return StepOutcome.blocked(
                f"{status.detail}: unset it or set the valid key your vault uses"
            )
        create = inputs.create_master_key
        if create is None and inputs.assume_defaults:
            create = True
        if not create:
            return StepOutcome.decide(
                f"no vault master key ({status.detail}): IRIS refuses every governed call "
                "without one. Create one now (the same as `iris doctor --fix`)?"
            )
        fixed = fix_master_key()
        if fixed.outcome in ("kept", "stored"):
            return StepOutcome.finished(key=fixed.outcome)
        if fixed.outcome == "export" and fixed.export_line:
            return StepOutcome.blocked(
                "a new master key was made, but this host has no keyring to keep it: add "
                "the export line shown to your shell profile and IRIS's environment, then "
                "run setup again",
                notice=f"{fixed.export_line}\n{fixed.detail}",
            )
        return StepOutcome.blocked(fixed.detail)

    # -- long steps: run inline, or backgrounded on the Activity spine --------------

    def _run_long_step(
        self,
        state: OnboardingState,
        *,
        kind: str,
        title: str,
        waiting_for: str,
        work: Callable[[Callable[[float, str], None] | None], dict[str, Any]],
    ) -> StepOutcome:
        """Run a step whose work can take minutes: backgrounded on the harness's
        Activity spine when one is wired (submit once, poll after, the webui and the
        CLI both just call ``advance`` again); inline, exactly as before, when it
        isn't (a test, or a host with no Activities).

        ``work`` raises on a real failure either way: inline, the exception becomes a
        blocked outcome directly; backgrounded, ``ActivityRunner`` catches it and
        fails the row, and the next poll turns that into the same blocked outcome."""
        from iris_harness.sdk.activities import ActivityOutcome

        activities = self.deps.activities()
        if activities is None:
            try:
                return StepOutcome.finished(**work(None))
            except Exception as exc:  # noqa: BLE001 — a step's failure waits, not crashes
                return StepOutcome.blocked(str(exc))

        if state.activity_id:
            activity = activities.get(state.activity_id)
            if activity is not None and activity.status == "completed":
                return StepOutcome.finished(**activity.metadata)
            if activity is not None and activity.status == "failed":
                return StepOutcome.blocked(activity.error or f"{title} failed")
            if activity is not None and activity.status in ("queued", "running"):
                return StepOutcome.running(
                    state.activity_id, why=activity.progress_message or waiting_for
                )
            # None (the row is gone), or cancelled (nothing cancels one today, but
            # nothing should get stuck if it ever does): submit a fresh one below.

        def run_work(progress: Callable[[float, str], None]) -> ActivityOutcome:
            return ActivityOutcome(metadata=work(progress))

        activity_id = activities.submit(
            kind=kind, title=title, origin="email_onboarding", work=run_work
        )
        return StepOutcome.running(activity_id, why=waiting_for)

    # -- 2. fetch -------------------------------------------------------------------

    def _do_fetch(
        self, state: OnboardingState, progress: Callable[[float, str], None] | None
    ) -> dict[str, Any]:
        from iris_harness.sdk.events import EventBus
        from iris_personal.email.events import (
            EMAIL_NEW_ARRIVED,
            EMAIL_SWEPT,
            EmailNewArrivedPayload,
        )
        from iris_personal.email.store import EmailStore

        from .judge_wiring import build_queue_handler
        from .judgments import JudgmentStore

        provider = self.deps.provider_for(state.account_id)
        if provider is None:
            raise RuntimeError(f"no mail provider is mounted for {state.account_id}")
        store = EmailStore(db_path=self.deps.email_db)
        store.ensure_schema()
        judgments = JudgmentStore(db_path=self.deps.email_db)
        judgments.ensure_schema()
        run, effects = state.run_id, self.store

        def forward(topic: str, payload: Any) -> None:
            self.deps.emit(topic, payload)
            if topic == EMAIL_NEW_ARRIVED:
                # After the readers have it: a crash in between re-releases, never loses.
                effects.record_effects(run, "fetch", "released", payload.new_message_ids)

        # A private bus, as the sweep's: the judge's queue holds what it will judge and
        # releases the rest to the process bus (where the runtime's readers listen).
        bus = EventBus()
        bus.on(
            EMAIL_SWEPT,
            build_queue_handler(
                db_path=self.deps.email_db, emit=forward, config_dir=self.deps.config_dir
            ),
        )
        try:
            result = provider.fetch_new(
                state.account_id,
                store=_RecordingSyncStore(
                    store, lambda ids: effects.record_effects(run, "fetch", "fetched", ids)
                ),
                max_messages=self.config.fetch_max_messages,
                cold_start_days=self.config.fetch_cold_start_days,
                progress=progress,
            )
        except Exception as exc:  # a login or network failure: fix it, resume
            raise RuntimeError(f"could not fetch mail: {exc}") from exc
        # Everything this run fetched and has not handed to the judge's queue yet: this
        # fetch's mail, and, on a resume, what an earlier attempt stored before it died.
        # (Queued mail has a judgment row; released mail is recorded above.)
        fetched = effects.effect_items(run, "fetch", "fetched")
        handled = effects.effect_items(run, "fetch", "released")
        unadmitted = judgments.missing(sorted(fetched - handled))
        if unadmitted:
            bus.emit_sync(
                EMAIL_SWEPT,
                EmailNewArrivedPayload(
                    account_id=state.account_id,
                    new_message_ids=tuple(unadmitted),
                    count=len(unadmitted),
                    fell_back_to_cold_start=result.fell_back_to_cold_start,
                ),
            )
        waiting = sum(1 for j in judgments.waiting() if j.account_id == state.account_id)
        done = effects.effect_counts(run, "fetch")
        return {
            "fetched": done["fetched"],
            "stored": store.count(state.account_id, include_held=True),
            "waiting_for_judge": waiting,
            "released": done["released"],
        }

    def _fetch(self, state: OnboardingState, inputs: Inputs) -> StepOutcome:
        del inputs
        return self._run_long_step(
            state,
            kind="email.fetch",
            title=f"Fetch mail for {state.account_id}",
            waiting_for="fetching mail…",
            work=lambda progress: self._do_fetch(state, progress),
        )

    # -- 3. discover ----------------------------------------------------------------

    def _discover(self, state: OnboardingState, inputs: Inputs) -> StepOutcome:
        from .discovery import bootstrap_categories, category_fields, load_corpus
        from .triage import _proposal_path

        del inputs
        corpus = len(load_corpus(self.deps.email_db, state.account_id, include_held=True))
        if corpus < self.config.discover_min_corpus:
            return StepOutcome.finished(
                corpus=corpus,
                proposals=[],
                skipped=(
                    f"{corpus} email(s) is too few to find categories (needs "
                    f"{self.config.discover_min_corpus}); run `iris email "
                    "bootstrap-categories` once more mail has arrived"
                ),
            )
        proposals = bootstrap_categories(
            state.account_id,
            db_path=self.deps.email_db,
            min_corpus=self.config.discover_min_corpus,
            fetch_if_low=False,
            naming_client=self.deps.naming_client(),
            embedder=self.deps.embedder,
            include_held=True,
        )
        # Where the kNN classifier reads each accepted category's members from.
        path = _proposal_path(self._workspace(), state.account_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(p.model_dump_json() + "\n" for p in proposals), encoding="utf-8")
        shown = []
        for p in proposals:
            error, fields = category_fields(p.model_dump(), state.account_id)
            shown.append(
                {
                    "cluster_id": p.cluster_id,
                    "size": p.size,
                    "cohesion": round(float(p.cohesion), 3),
                    "path": fields.get("path"),
                    "acceptable": error is None,
                    "why_not": error or "",
                    "top_domain": p.top_domains[0][0] if p.top_domains else "",
                    "samples": [r.subject for r in p.representatives][
                        : self.config.discover_samples
                    ],
                }
            )
        return StepOutcome.finished(corpus=corpus, proposals=shown, file=str(path))

    def _workspace(self) -> Path:
        if self.deps.workspace_dir is not None:
            return self.deps.workspace_dir
        from iris_harness.sdk.config import workspace_dir

        return workspace_dir()

    # -- 4. review categories -----------------------------------------------------------

    def _review_categories(self, state: OnboardingState, inputs: Inputs) -> StepOutcome:
        from iris_personal.email.category_store import Category, CategoryStore

        from .discovery import category_fields
        from .triage import _proposal_path

        proposals = state.results.get("discover", {}).get("proposals", [])
        acceptable = [p for p in proposals if p.get("acceptable")]
        if not acceptable:
            return StepOutcome.finished(accepted=[], inserted=0, unchanged=0, skipped=[])
        wanted = inputs.accept_categories
        if wanted is None:
            if not inputs.assume_defaults:
                return StepOutcome.decide(
                    f"review the {len(acceptable)} proposed categor"
                    f"{'y' if len(acceptable) == 1 else 'ies'}: accept all, some or none"
                )
            wanted = tuple(int(p["cluster_id"]) for p in acceptable)
        known = {int(p["cluster_id"]) for p in acceptable}
        unknown = sorted(set(wanted) - known)
        if unknown:
            raise OnboardingError(f"no acceptable proposed category with id {unknown}")
        raw_rows = {
            int(r["cluster_id"]): r
            for r in (
                json.loads(line)
                for line in _proposal_path(self._workspace(), state.account_id)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
        }
        store = CategoryStore(db_path=self.deps.iris_db)
        store.ensure_schema()
        accepted: list[str] = []
        skipped: list[str] = []
        for cluster_id in sorted(set(wanted)):
            error, fields = category_fields(raw_rows[cluster_id], state.account_id)
            if error is not None:
                skipped.append(error)
                continue
            if fields["path"] in accepted:
                # Two clusters named alike: the first keeps the name (ADR-0020 Path D).
                skipped.append(f"cluster {cluster_id}: {fields['path']} is already accepted")
                continue
            # Whether this run adds it is decided before the write and recorded first, so
            # a re-run after a crash still counts the category this run inserted.
            before = store.get(fields["path"])
            if before is None or not before.active:
                self.store.record_effects(
                    state.run_id, "review_categories", "inserted", [fields["path"]]
                )
            store.upsert_if_new(Category(**fields))
            accepted.append(fields["path"])
        added = self.store.effect_items(state.run_id, "review_categories", "inserted")
        inserted = sum(1 for path in accepted if path in added)
        return StepOutcome.finished(
            accepted=accepted,
            inserted=inserted,
            unchanged=len(accepted) - inserted,
            skipped=skipped,
        )

    # -- 5. classify ----------------------------------------------------------------

    def _do_classify(
        self, state: OnboardingState, progress: Callable[[float, str], None] | None
    ) -> dict[str, Any]:
        """The judge over what the fetch queued, then kNN over what it released.

        The judge runs first because mail it has not read is held: no reader, the kNN
        classifier included, may file or publish it before then. The kNN classifier is
        the first classifier for categories (pure kNN, no model; ADR-0022) -- what it
        is unsure of waits for `iris email triage-batch`. Labels are not written here.
        """
        from iris_personal.email.events import EMAIL_NEW_ARRIVED

        from .judge import UNSURE
        from .judge_config import JudgeConfig
        from .judge_wiring import judge_and_release, release
        from .judgments import JudgmentStore

        run, effects = state.run_id, self.store

        def forward(topic: str, payload: Any) -> None:
            self.deps.emit(topic, payload)
            if topic == EMAIL_NEW_ARRIVED:
                effects.record_effects(run, "classify", "released", payload.new_message_ids)

        llm = self.deps.judge_llm()
        report, _note = judge_and_release(
            llm=llm,
            config_dir=self.deps.config_dir,
            db_path=self.deps.email_db,
            emit=forward,
            labels=False,
            limit=self.config.judge_limit,
            # Each verdict carries the setup's run id: the record of what this run
            # judged, written with the verdict, so a resumed step counts it too.
            run_id=run,
            progress=progress,
        )
        if report.enabled and report.no_model:
            raise RuntimeError(
                "no local email_judge model tier is configured (llm_tiers.yaml): the judge "
                "cannot read the mail. Configure it, then run setup again"
            )
        if report.unreachable and not report.judged:
            raise RuntimeError(
                f"the email_judge model is unreachable ({report.unreachable_error[:120]}): "
                "start it, then run setup again"
            )
        rows = JudgmentStore(db_path=self.deps.email_db).judged_in_run(run)
        # Judged before a crash but never released to the readers: release it now.
        released = effects.effect_items(run, "classify", "released")
        pending: dict[str, list[str]] = {}
        for row in rows:
            if row.message_id not in released:
                pending.setdefault(row.account_id, []).append(row.message_id)
        release(forward, pending)
        verdicts = [r for r in rows if not r.error.startswith("skipped:")]
        counts = Counter(str(r.bucket) for r in verdicts)
        names = JudgeConfig.load(self.deps.config_dir).name
        knn = self._knn(state)
        return {
            "judge_enabled": report.enabled,
            "judged": len(verdicts),
            "buckets": {names(k): v for k, v in sorted(counts.items())},
            "unsure": counts.get(UNSURE, 0),
            "judge_errors": sum(1 for r in verdicts if r.error),
            "still_waiting": report.waiting,
            **knn,
        }

    def _classify(self, state: OnboardingState, inputs: Inputs) -> StepOutcome:
        del inputs
        return self._run_long_step(
            state,
            kind="email.classify",
            title=f"Judge mail for {state.account_id}",
            waiting_for="judging mail…",
            work=lambda progress: self._do_classify(state, progress),
        )

    def _knn(self, state: OnboardingState) -> dict[str, Any]:
        accepted = state.results.get("review_categories", {}).get("accepted") or []
        if not accepted:
            return {"knn_classified": 0, "knn_to_review": 0, "knn_errors": 0}
        from iris_harness.sdk.events import EventBus
        from iris_personal.email.events import EMAIL_CLASSIFIED

        from .triage import EmailTriageClassifier

        run, effects = state.run_id, self.store

        def filed(payload: Any) -> None:
            # What it files reaches the process bus (the wiki's translator) through
            # ``emit``, and is counted for this run once it has.
            self.deps.emit(EMAIL_CLASSIFIED, payload)
            effects.record_effects(run, "classify", "knn_classified", [payload.id])

        bus = EventBus()
        bus.on(EMAIL_CLASSIFIED, filed)
        classifier = EmailTriageClassifier(
            workspace_dir=self._workspace(),
            db_path=self.deps.iris_db,
            email_db_path=self.deps.email_db,
            bus=bus,
            embedder=self.deps.embedder,
        )
        from iris_personal.email.store import EmailStore

        results = classifier.classify_unclassified(state.account_id, limit=self.config.knn_limit)
        emails = EmailStore(db_path=self.deps.email_db)
        return {
            "knn_classified": effects.effect_counts(run, "classify")["knn_classified"],
            # What waits for review is in the mail store; this attempt's errors are not.
            "knn_to_review": len(emails.list_pending_review(state.account_id, limit=10000)),
            "knn_errors": sum(1 for r in results if r.error is not None and not r.queued),
        }

    # -- 6. label preview + approval ----------------------------------------------------

    def label_preview(self, account_id: str) -> LabelPreview:
        """The labels IRIS would write to ``account_id`` now (read-only)."""
        from iris_personal.email.store import EmailStore

        from .judge_config import JudgeConfig, labels_enabled
        from .judge_labels import _labelling
        from .judgments import PROMO, JudgmentStore

        config = JudgeConfig.load(self.deps.config_dir)
        judgments = JudgmentStore(db_path=self.deps.email_db)
        judgments.ensure_schema()
        emails = EmailStore(db_path=self.deps.email_db)
        emails.ensure_schema()
        due = judgments.labels_due(account_id)
        by_bucket: dict[str, list[str]] = {}
        removals = 0
        for row in due:
            bucket = row.effective_bucket
            if bucket == PROMO:
                removals += 1
            elif bucket is not None:
                by_bucket.setdefault(bucket, []).append(row.message_id)
        groups = []
        for key in config.keys:
            ids = by_bucket.get(key, [])
            if not ids:
                continue
            samples = []
            for mid in ids[: self.config.preview_samples]:
                message = emails.get(mid, include_held=True)
                if message is not None:
                    samples.append(message.subject or "(no subject)")
            groups.append(LabelGroup(key, config.labels[key], len(ids), tuple(samples)))
        state = self.store.get(account_id)
        grant = self.deps.write_approvals().get(account_id)
        return LabelPreview(
            account_id=account_id,
            groups=tuple(groups),
            removals=removals,
            labelling=_labelling(self.deps.provider_for(account_id)),
            labels_enabled=labels_enabled(),
            approval_id=state.approval_id if state else None,
            already_approved=grant is not None,
        )

    def _label_approval(self, state: OnboardingState, inputs: Inputs) -> StepOutcome:
        preview = self.label_preview(state.account_id)
        grant = self.deps.write_approvals().get(state.account_id)
        queue = self.deps.approval_queue()
        if grant is not None:
            asked = queue.get(grant.approval_ref.split(" ", 1)[0])
            if asked is not None and asked.run_id == state.run_id:
                # This run granted it, then died before step 6 was saved: finish it.
                granted = self.write_labels(
                    state.account_id,
                    preview,
                    grant.approval_ref,
                    already=False,
                    run_id=state.run_id,
                )
                granted.result["approval_id"] = asked.approval_id
                return replace(granted, approval_id=asked.approval_id)
            # Approved before (an earlier setup, or the one-time writes command).
            return self.write_labels(
                state.account_id, preview, grant.approval_ref, already=True, run_id=state.run_id
            )
        row = queue.get(state.approval_id) if state.approval_id else None
        if row is None or row.status in ("timed_out", "expired"):
            # A row this run asked for before it died, unsaved, is still the question.
            row = queue.pending_for_run(state.run_id) or queue.get(
                self._request_approval(queue, state, preview, inputs)
            )
        assert row is not None
        answer = inputs.approve_writes
        if answer is None and row.status in ("approved", "rejected"):
            answer = row.status == "approved"  # answered in the Action Center, or earlier
        if answer is None:
            return StepOutcome.decide(
                f"approve the label preview (approval {row.approval_id}) to let IRIS change "
                f"{state.account_id}, or decline to keep it read-only",
                approval_id=row.approval_id,
            )
        from iris_harness.sdk.approvals import respond_to_approval

        if row.status == "pending":
            outcome = respond_to_approval(
                row.approval_id,
                status="approved" if answer else "rejected",
                actor=inputs.actor,
                queue=queue,
            )
            row = outcome.row
        elif answer and row.status != "approved":
            # Declined earlier; the owner now says yes: a fresh row records this answer
            # (the one a crash left pending, if any).
            waiting = queue.pending_for_run(state.run_id)
            fresh = (
                waiting.approval_id
                if waiting is not None
                else self._request_approval(queue, state, preview, inputs)
            )
            row = respond_to_approval(fresh, status="approved", actor=inputs.actor, queue=queue).row
        if row.status != "approved":
            return StepOutcome(
                done=True,
                result={
                    "writes_approved": False,
                    "approval_id": row.approval_id,
                    "labels_previewed": preview.total,
                    "labels_written": 0,
                },
                approval_id=row.approval_id,
            )
        ref = self._grant(state, row.approval_id, inputs.actor)
        if ref is None:
            return StepOutcome.blocked(
                "the approval is recorded, but its audit row could not be written, so "
                "nothing was granted: run setup again",
                approval_id=row.approval_id,
            )
        granted = self.write_labels(
            state.account_id, preview, ref, already=False, run_id=state.run_id
        )
        granted.result["approval_id"] = row.approval_id
        return replace(granted, approval_id=row.approval_id)

    def _request_approval(
        self, queue: Any, state: OnboardingState, preview: LabelPreview, inputs: Inputs
    ) -> str:
        from iris_harness.sdk.approvals import ApprovalCard

        title = self.config.approval_title.format(account=state.account_id)
        card = ApprovalCard(
            title=title,
            lines=tuple(preview.lines(self.config)),
            asked=f"email setup for {state.account_id}",
            effect="write",
        )
        return str(
            queue.enqueue(
                state.run_id,
                None,
                self.config.approval_signal,
                f"{title}: {preview.total} label(s) previewed",
                channel=inputs.channel,
                timeout_minutes=self.config.approval_timeout_minutes,
                card=card,
            )
        )

    def write_labels(
        self,
        account_id: str,
        preview: LabelPreview,
        ref: str,
        *,
        already: bool,
        run_id: str | None = None,
    ) -> StepOutcome:
        """Writes are approved: write the labels now due (when labels are on). Step 6's
        second half; ``iris email demo`` runs it for its own account too.

        With ``run_id`` (setup), each written group is recorded in the run's effects as
        the mailbox takes it, and the counts are the run's: a step 6 that resumes after a
        crash reports the labels the first attempt wrote, not 0."""
        from .judge_config import JudgeConfig, labels_enabled
        from .judge_labels import sync_labels
        from .judgments import JudgmentStore

        written = removed = failed = 0
        error = ""
        previewed = preview.total
        provider = self.deps.provider_for(account_id)
        record: Callable[[str, str | None, list[str]], None] | None = None
        if run_id is not None:
            effects, run = self.store, run_id

            def _record(_account: str, bucket: str | None, ids: list[str]) -> None:
                kind = "label_written" if bucket is not None else "label_removed"
                effects.record_effects(run, "label_approval", kind, ids)

            record = _record

        if labels_enabled() and provider is not None:
            sync = sync_labels(
                JudgmentStore(db_path=self.deps.email_db),
                JudgeConfig.load(self.deps.config_dir),
                {account_id: provider},
                on_written=record,
            )
            written, removed, failed = sync.written, sync.removed, sync.failed
            error = "; ".join(sync.errors)
        if run_id is not None:
            done = self.store.effect_counts(run_id, "label_approval")
            written, removed = done["label_written"], done["label_removed"]
            # What the owner was shown: written by this run, plus what is still due.
            previewed = written + self.label_preview(account_id).total
        return StepOutcome.finished(
            writes_approved=True,
            already_approved=already,
            approval_ref=ref,
            labels_previewed=previewed,
            labels_written=written,
            labels_removed=removed,
            labels_failed=failed,
            label_error=error,
        )

    def _grant(self, state: OnboardingState, approval_id: str, actor: str) -> str | None:
        """Record the owner's grant: ``<approval id> [audit #<row>]``, so it points at
        both the approved queue row and its ledger entry (R14)."""
        return self.grant_writes(
            state.account_id,
            approval_id,
            actor=actor,
            run_id=state.run_id,
            step_id=STEPS.index("label_approval"),
            agent_type="email_setup",
        )

    def grant_writes(
        self,
        account_id: str,
        ref: str,
        *,
        actor: str,
        run_id: str,
        agent_type: str,
        step_id: int | None = None,
    ) -> str | None:
        """Record a mailbox-write grant the way ``iris email writes approve`` does
        (``write_approvals.grant_mailbox_writes``: the audit row first, then the approval
        naming it). No audit row, no grant: returns None. Returns the reference."""
        from iris_personal.email.write_approvals import grant_mailbox_writes

        try:
            grant, _row = grant_mailbox_writes(
                account_id,
                ref,
                actor=actor,
                run_id=run_id,
                agent_type=agent_type,
                step_id=step_id,
                audit_log=self.deps.audit_log(),
                store=self.deps.write_approvals(),
            )
        except Exception:  # no audit trail, no approval
            logger.warning("email setup: the grant's audit row failed", exc_info=True)
            return None
        return grant.approval_ref

    # -- 7. first digest ------------------------------------------------------------

    def _first_digest(self, state: OnboardingState, inputs: Inputs) -> StepOutcome:
        from .first_digest import build_first_digest

        del state, inputs
        digest = build_first_digest(
            self.deps.data_dir, narrate=self.deps.narrate(), config_dir=self.deps.config_dir
        )
        return StepOutcome.finished(text=digest.text, sections=list(digest.sections))

    # -- 8. review queue ------------------------------------------------------------

    def _review_queue(self, state: OnboardingState, inputs: Inputs) -> StepOutcome:
        from iris_personal.email.store import EmailStore

        from .judge_cards import refresh_judge_cards
        from .judgments import JudgmentStore

        del inputs
        judgments = JudgmentStore(db_path=self.deps.email_db)
        emails = EmailStore(db_path=self.deps.email_db)
        unsure = [
            j
            for j in judgments.recent(bucket="unsure", limit=1000)
            if j.account_id == state.account_id
        ]
        samples = []
        for j in unsure[: self.config.review_samples]:
            message = emails.get(j.message_id)
            if message is not None:
                samples.append(f"{message.subject or '(no subject)'} ({message.from_address})")
        to_review = len(emails.list_pending_review(state.account_id, limit=10000))
        # The Unsure cards are how the owner answers them (Action Center, chat, web).
        refresh_judge_cards(self.deps.data_dir, config_dir=self.deps.config_dir)
        return StepOutcome.finished(
            unsure=len(unsure), unsure_samples=samples, knn_to_review=to_review
        )

    # -- 9. keep it current -----------------------------------------------------------

    def _enable_sweep(self, state: OnboardingState, inputs: Inputs) -> StepOutcome:
        from iris_harness.sdk.heartbeat import describe_schedule, load_heartbeats

        del inputs
        accounts = self.deps.accounts()
        account = accounts.get(state.account_id)
        if account is not None and not account.active:
            accounts.activate(state.account_id)
        path = self.deps.heartbeats_path
        if path is None:
            from iris_harness.sdk.config import config_path

            path = config_path("heartbeats.yaml")
        # The scheduled sweep takes this account from now on (it waited for setup).
        self.deps.sweep_gate().release(
            state.account_id, f"email setup ({state.run_id}) turned the sweep on"
        )
        jobs = {d.name: d for d in load_heartbeats(path)}
        out: dict[str, Any] = {"account_active": True, "account_swept": True}
        for name in ("email_sweep", "email_judge"):
            job = jobs.get(name)
            out[name] = (
                {"enabled": job.enabled, "schedule": describe_schedule(job.schedule)}
                if job is not None
                else {"enabled": False, "schedule": ""}
            )
        return StepOutcome.finished(**out)

    # -- 10. summary ----------------------------------------------------------------

    def _summary(self, state: OnboardingState, inputs: Inputs) -> StepOutcome:
        del inputs
        rows = self.deps.audit_log().query(since=state.started_at)
        by_hook = Counter(r.hook_point for r in rows)
        mine = [r for r in rows if r.run_id == state.run_id]
        results = state.results
        approval = results.get("label_approval", {})
        return StepOutcome.finished(
            fetched=results.get("fetch", {}).get("fetched", 0),
            categories_proposed=len(results.get("discover", {}).get("proposals", [])),
            categories_accepted=len(results.get("review_categories", {}).get("accepted", [])),
            judged=results.get("classify", {}).get("judged", 0),
            knn_classified=results.get("classify", {}).get("knn_classified", 0),
            writes_approved=bool(approval.get("writes_approved")),
            labels_written=int(approval.get("labels_written", 0)),
            unsure=results.get("review_queue", {}).get("unsure", 0),
            audit_rows=len(rows),
            audit_rows_this_setup=len(mine),
            audit_by_hook=dict(sorted(by_hook.items())),
            sweep=results.get("enable_sweep", {}).get("email_sweep", {}),
            judge=results.get("enable_sweep", {}).get("email_judge", {}),
        )


# -- rendering (the CLI prints these; the API returns the dicts) ----------------------


def render_step(config: OnboardingConfig, step: str, result: Mapping[str, Any]) -> str:
    """One finished step, in a few plain lines."""
    return f"## {config.titles.get(step, step)}\n{step_body(step, result)}"


def step_body(step: str, result: Mapping[str, Any]) -> str:
    """What a finished step did, in plain lines (the CLI and the web Setup screen)."""
    r = result
    if step == "connect":
        body = f"{r.get('provider')} account {r.get('address')} (vault key: {r.get('key')})"
    elif step == "fetch":
        body = (
            f"fetched {r.get('fetched', 0)} email(s); {r.get('waiting_for_judge', 0)} wait "
            f"for the judge, {r.get('released', 0)} released at once (promotions, social, "
            "your own sent mail)"
        )
    elif step == "discover":
        proposals = r.get("proposals") or []
        lines = [f"{len(proposals)} categor{'y' if len(proposals) == 1 else 'ies'} proposed"]
        if r.get("skipped"):
            lines.append(str(r["skipped"]))
        for p in proposals:
            mark = p.get("path") or f"(not acceptable: {p.get('why_not')})"
            lines.append(
                f"  [{p.get('cluster_id')}] {mark}  size {p.get('size')}, "
                f"cohesion {p.get('cohesion')}, e.g. {'; '.join(p.get('samples') or [])}"
            )
        body = "\n".join(lines)
    elif step == "review_categories":
        body = f"accepted {len(r.get('accepted') or [])}: " + (
            ", ".join(r.get("accepted") or []) or "none"
        )
        for why in r.get("skipped") or []:
            body += f"\n  skipped: {why}"
    elif step == "classify":
        buckets = ", ".join(f"{n} {k}" for k, n in (r.get("buckets") or {}).items()) or "none"
        body = (
            f"judged {r.get('judged', 0)} ({buckets}); {r.get('still_waiting', 0)} still wait "
            f"for the next judge run. kNN filed {r.get('knn_classified', 0)}, "
            f"{r.get('knn_to_review', 0)} left for review. No label written."
        )
        if not r.get("judge_enabled", True):
            body = "the judge is off (IRIS_EMAIL_JUDGE=0): " + body
    elif step == "label_approval":
        if r.get("writes_approved"):
            body = (
                f"mailbox writes approved ({r.get('approval_ref')}); "
                f"{r.get('labels_written', 0)} label(s) written, "
                f"{r.get('labels_failed', 0)} failed"
            )
            if r.get("label_error"):
                body += f" ({r['label_error']})"
        else:
            body = (
                f"declined (approval {r.get('approval_id')}): IRIS will not change the "
                "mailbox; 0 labels written"
            )
    elif step == "first_digest":
        body = str(r.get("text", ""))
    elif step == "review_queue":
        body = f"{r.get('unsure', 0)} email(s) the judge was unsure of wait in the Action Center"
        for s in r.get("unsure_samples") or []:
            body += f"\n  - {s}"
        if r.get("knn_to_review"):
            body += f"\n{r['knn_to_review']} wait for `iris email triage-batch`"
    elif step == "enable_sweep":
        body = "\n".join(
            f"{name}: " + (f"on, {job.get('schedule')}" if job.get("enabled") else "OFF")
            for name, job in ((n, r.get(n) or {}) for n in ("email_sweep", "email_judge"))
        )
    elif step == "summary":
        body = render_summary(r)
    else:
        body = json.dumps(r, default=str)
    return body


def render_summary(r: Mapping[str, Any]) -> str:
    """The "what IRIS just did" block."""
    hooks = ", ".join(f"{n} {h}" for h, n in (r.get("audit_by_hook") or {}).items()) or "none"
    sweep = r.get("sweep") or {}
    judge = r.get("judge") or {}
    lines = [
        f"- Fetched {r.get('fetched', 0)} email(s).",
        f"- Proposed {r.get('categories_proposed', 0)} categories; you accepted "
        f"{r.get('categories_accepted', 0)}.",
        f"- Judged {r.get('judged', 0)}; kNN filed {r.get('knn_classified', 0)}; "
        f"{r.get('unsure', 0)} wait for you as Unsure.",
        (
            f"- Mailbox writes approved: {r.get('labels_written', 0)} label(s) written."
            if r.get("writes_approved")
            else "- Mailbox writes not approved: 0 labels written, nothing changed in the mailbox."
        ),
        f"- Governance audit rows since setup began: {r.get('audit_rows', 0)} ({hooks}); "
        f"{r.get('audit_rows_this_setup', 0)} from setup itself.",
        "- Next: the sweep "
        + (f"runs {sweep.get('schedule')}" if sweep.get("enabled") else "is OFF")
        + ", the judge "
        + (f"{judge.get('schedule')}." if judge.get("enabled") else "is OFF."),
    ]
    return "\n".join(lines)


def render_sweep(status: Mapping[str, Any], config: OnboardingConfig) -> str:
    """One line: whether the scheduled sweep takes the account, and why."""
    if status.get("swept"):
        return f"Scheduled sweep: on ({status.get('reason')})"
    title = config.titles.get("enable_sweep", "enable_sweep")
    return (
        f"Scheduled sweep: waiting for email setup ({status.get('reason')}); it starts when "
        f"setup reaches '{title}' -- run `iris email setup` to go on"
    )


def render_waiting(state: OnboardingState, config: OnboardingConfig) -> str:
    title = config.titles.get(state.step, state.step)
    verb = "Needs your decision" if state.waiting_kind == DECISION else "Waiting"
    return f"## {title}\n{verb}: {state.waiting_for}"


def candidate_accounts(deps: OnboardingDeps) -> list[str]:
    """Connected mailbox accounts (active), for picking one to set up."""
    from iris_personal.email.accounts import is_non_mailbox_provider

    return [
        a.id
        for a in deps.accounts().list(active_only=True)
        if not is_non_mailbox_provider(a.provider)
    ]


def parse_ids(text: str) -> tuple[int, ...]:
    """``"1, 3 4"`` -> ``(1, 3, 4)``; ``""``/``"none"`` -> ``()``."""
    cleaned = text.replace(",", " ").strip().lower()
    if cleaned in ("", "none"):
        return ()
    try:
        return tuple(int(part) for part in cleaned.split())
    except ValueError as exc:
        raise OnboardingError(f"not a list of category ids: {text!r}") from exc


def acceptable_ids(state: OnboardingState) -> list[int]:
    proposals: Sequence[Mapping[str, Any]] = state.results.get("discover", {}).get("proposals", [])
    return [int(p["cluster_id"]) for p in proposals if p.get("acceptable")]


__all__ = [
    "ACTIVITY",
    "BLOCKED",
    "COMPLETE",
    "DECISION",
    "STEPS",
    "ActivityJobs",
    "Inputs",
    "LabelGroup",
    "LabelPreview",
    "Onboarding",
    "OnboardingConfig",
    "OnboardingDeps",
    "OnboardingError",
    "OnboardingState",
    "OnboardingStore",
    "StepOutcome",
    "acceptable_ids",
    "candidate_accounts",
    "parse_ids",
    "render_step",
    "render_summary",
    "render_sweep",
    "render_waiting",
    "step_body",
]
