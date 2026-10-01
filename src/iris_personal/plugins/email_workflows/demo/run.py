"""One demo run, inside the demo home: fetch -> judge -> first digest -> what IRIS did.

``iris email demo`` starts this module as a child process with the environment
``home.demo_environment`` builds, so every store it opens is the demo home's. It uses
the same pieces a mounted email plugin runs, not look-alikes:

1. **fetch** -- the email sweep (``EmailSweepHandler``) over the demo mailbox provider,
   which stores the mail and emits ``email.swept``;
2. **classify** -- the judge's queue holds each email it will judge, and the judge job
   (``judge_and_release``) asks the ``email_judge`` tier for one governed JSON verdict
   per email, then releases what it judged. The tier is on the scripted fake, so every
   call still passes the governance kernel and leaves its audit rows. (The kNN
   classifier needs categories the owner accepted first -- ``iris email setup``'s
   discovery step -- so a fresh demo home classifies with the judge alone.)
3. **labels** -- email setup's step 6, run for the demo's own account: the label
   preview setup shows the owner (``Onboarding.label_preview``: count per IRIS/* label,
   sample subjects), then the approval recorded the audited way (``grant_writes``: an
   audit row, then the approval naming it) and the labels written (``write_labels``).
   On a real mailbox only the owner's explicit yes records that approval; the demo
   records it for its synthetic account alone, and only inside a demo home
   (:func:`approve_demo_writes`). Approved once: a re-run records nothing new;
4. **first digest** -- the email agent's inbox digest (narrated on the governed tier),
   the morning digest's Needs reply section and the bill emails the judge found;
5. **what IRIS just did** -- counts from the run and from the audit ledger.

Network access is refused for the whole run (socket connect raises), so "no network" is
enforced, not assumed. Re-running is idempotent: the provider's cursor delivers nothing
new and nothing is judged twice.
"""

from __future__ import annotations

import os
import socket
import sys
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .home import HOME_ENV, MARKER
from .provider import DEMO_ACCOUNT, DEMO_PROVIDER, DemoMailProvider

if TYPE_CHECKING:
    from ..onboarding import Onboarding

# The demo sweep takes the whole mailbox in one tick (the corpus is 200 messages).
_SWEEP_MAX = 1000


class DemoEnvironmentError(RuntimeError):
    """The run is not inside a demo home's environment."""


@dataclass
class DemoReport:
    """What one run did, and the digest it produced."""

    home: Path
    fetched: int = 0
    stored: int = 0
    released_unjudged: int = 0
    judged: int = 0
    buckets: dict[str, int] = field(default_factory=dict)
    unsure: int = 0
    errors: int = 0
    llm_calls: int = 0
    audit_rows: int = 0
    audit_by_hook: dict[str, int] = field(default_factory=dict)
    providers: tuple[str, ...] = ()
    network_attempts: int = 0
    digest: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    # Step 6 for the demo's account: the preview, the approval, the labels written.
    label_preview: list[str] = field(default_factory=list)
    labels_previewed: int = 0
    labels_written: int = 0
    labels_removed: int = 0
    labels_failed: int = 0
    label_error: str = ""
    writes_approval_ref: str = ""
    writes_approved_now: bool = False


def require_demo_environment() -> Path:
    """The demo home, when this process runs in its environment; raises otherwise.

    The run writes stores; this makes sure they can only be the demo home's."""
    from iris_harness.sdk.llm import FAKE_PROVIDER, FORCED_PROVIDER_ENV

    raw = os.environ.get(HOME_ENV, "").strip()
    home = Path(raw) if raw else None
    fake = os.environ.get(FORCED_PROVIDER_ENV, "") == FAKE_PROVIDER
    if (
        home is None
        or os.environ.get("IRIS_HOME") != str(home)
        or os.environ.get("IRIS_DATA_DIR") != str(home / "data")
        or not (home / MARKER).is_file()
        or not fake
    ):
        raise DemoEnvironmentError(
            "the demo runs only inside its own home; start it with `iris email demo`"
        )
    return home


@contextmanager
def _offline() -> Iterator[list[Any]]:
    """Refuse every non-Unix socket connection for the block (the demo is offline)."""
    attempts: list[Any] = []
    real = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> None:
        if self.family == getattr(socket, "AF_UNIX", None):
            real(self, address)
            return
        attempts.append(address)
        raise OSError(f"the email demo runs offline; refused a connection to {address!r}")

    socket.socket.connect = connect  # type: ignore[method-assign,assignment]
    try:
        yield attempts
    finally:
        socket.socket.connect = real  # type: ignore[method-assign]


def _ensure_account() -> None:
    from iris_personal.email.accounts import EmailAccountStore

    from .corpus import OWNER_ADDRESS

    accounts = EmailAccountStore()
    accounts.ensure_schema()
    if accounts.get(DEMO_ACCOUNT) is None:
        accounts.add(provider=DEMO_PROVIDER, address=OWNER_ADDRESS)


def _fetch(report: DemoReport, bus: Any) -> None:
    from iris_harness.sdk.types import HeartbeatDefinition
    from iris_personal.email.store import EmailStore
    from iris_personal.email.sweep import EmailSweepHandler

    handler = EmailSweepHandler(bus=bus)
    handler(
        HeartbeatDefinition(
            name="email_sweep",
            handler="email_sweep",
            schedule="interval:900",
            params={"max_messages": _SWEEP_MAX},
        )
    )
    store = EmailStore()
    store.ensure_schema()
    report.stored = store.count(DEMO_ACCOUNT, include_held=True)


def _classify(report: DemoReport, router: Any, emit: Any) -> None:
    from iris_harness.sdk.config import config_dir

    from ..judge import llm_from_router
    from ..judge_config import JudgeConfig
    from ..judge_wiring import judge_and_release

    llm = llm_from_router(router)
    if llm is None:
        raise DemoEnvironmentError("no email_judge tier in llm_tiers.yaml; cannot judge")
    # As setup's classify step: no label is written before the step-6 approval.
    judge_report, _note = judge_and_release(
        llm=llm, config_dir=config_dir(), emit=emit, labels=False, limit=_SWEEP_MAX
    )
    report.judged = judge_report.judged
    # By the names judge.yaml gives the buckets ("Needs reply", not "needs_reply").
    names = JudgeConfig.load(config_dir()).name
    report.buckets = {names(k): v for k, v in sorted(judge_report.counts.items())}
    report.unsure = judge_report.unsure
    report.errors = judge_report.errors


def _setup_machine() -> Onboarding:
    """Email setup's state machine over the demo home's stores (its defaults resolve
    through ``IRIS_DATA_DIR`` and the audit path, both the demo home's here)."""
    from iris_harness.sdk.config import config_dir

    from ..onboarding import Onboarding, OnboardingDeps

    return Onboarding(OnboardingDeps(config_dir=config_dir()))


def _inside(path: Path, home: Path) -> bool:
    return path.resolve().is_relative_to(home.resolve())


def approve_demo_writes(machine: Onboarding, home: Path) -> tuple[str, bool]:
    """Record the mailbox-write approval for the demo's own synthetic account, the way
    setup's step 6 records the owner's: an audit row (plugin ``email_write_approvals``,
    hook ``mailbox_writes``, actor ``demo``), then the approval naming it. Returns the
    approval reference and whether it was recorded now (an existing one is reused, so a
    re-run records nothing).

    Refused -- :class:`DemoEnvironmentError`, nothing written -- unless this process is
    in the demo home's environment (``IRIS_HOME`` = ``IRIS_DEMO_HOME``, the ``.iris-demo``
    marker) and both the approvals file and the audit ledger are inside that home."""
    from iris_harness.sdk.audit import audit_db_path

    if require_demo_environment().resolve() != home.resolve():
        raise DemoEnvironmentError("the demo approves writes only in its own home")
    writes = machine.deps.write_approvals()
    if not _inside(writes.db_path, home) or not _inside(audit_db_path(), home):
        raise DemoEnvironmentError(
            "the demo's approval would land outside its home; nothing was approved"
        )
    existing = writes.get(DEMO_ACCOUNT)
    if existing is not None:
        return str(existing.approval_ref), False
    ref = machine.grant_writes(
        DEMO_ACCOUNT,
        "iris email demo",
        actor="demo",
        run_id=f"email-demo-{uuid.uuid4().hex[:12]}",
        agent_type="email_demo",
    )
    if ref is None:
        raise DemoEnvironmentError("the approval's audit row could not be written")
    return ref, True


def _labels(report: DemoReport) -> None:
    """Setup's step 6 for the demo account: preview, approve (demo only), label."""
    machine = _setup_machine()
    preview = machine.label_preview(DEMO_ACCOUNT)
    report.label_preview = preview.lines(machine.config)
    report.labels_previewed = preview.total
    ref, now = approve_demo_writes(machine, report.home)
    result = machine.write_labels(DEMO_ACCOUNT, preview, ref, already=not now).result
    report.writes_approval_ref = ref
    report.writes_approved_now = now
    report.labels_written = int(result.get("labels_written", 0))
    report.labels_removed = int(result.get("labels_removed", 0))
    report.labels_failed = int(result.get("labels_failed", 0))
    report.label_error = str(result.get("label_error", ""))


def _digest(report: DemoReport, router: Any) -> None:
    from iris_harness.sdk.config import config_dir
    from iris_harness.sdk.llm import make_narrative_llm_call
    from iris_harness.sdk.persistence import data_path

    from ..first_digest import build_first_digest

    digest = build_first_digest(
        data_path("email.db").parent,
        narrate=make_narrative_llm_call(router),
        config_dir=config_dir(),
    )
    report.sections = dict(digest.sections)
    report.digest = digest.text


def _audit(report: DemoReport, since: datetime) -> None:
    import json

    from iris_harness.sdk.audit import AuditLog, audit_db_path

    rows = AuditLog(db_path=audit_db_path()).query(since=since)
    report.audit_rows = len(rows)
    report.audit_by_hook = dict(sorted(Counter(r.hook_point for r in rows).items()))
    llm_runs: set[str] = set()
    providers: set[str] = set()
    for row in rows:
        if row.hook_point != "pre_llm_call":
            continue
        llm_runs.add(row.run_id)
        try:
            payload = json.loads(row.payload_json)
        except ValueError:
            payload = {}
        if isinstance(payload, dict) and payload.get("provider"):
            providers.add(str(payload["provider"]))
    report.llm_calls = len(llm_runs)
    report.providers = tuple(sorted(providers))


def run_demo() -> DemoReport:
    """Run the demo in this process's demo home and return what it did."""
    from iris_harness.sdk.config import config_path
    from iris_harness.sdk.events import EventBus
    from iris_harness.sdk.llm import TierRouter
    from iris_personal.email.events import EMAIL_NEW_ARRIVED, EMAIL_SWEPT
    from iris_personal.email.providers import register_mail_provider

    from ..judge_wiring import build_queue_handler

    home = require_demo_environment()
    report = DemoReport(home=home)
    started = datetime.now(UTC)
    with _offline() as attempts:
        register_mail_provider(DemoMailProvider())
        _ensure_account()
        # A private bus: the sweep -> queue -> judge -> release chain, and nothing else.
        bus = EventBus()
        released: list[int] = []
        bus.on(EMAIL_SWEPT, lambda p: setattr(report, "fetched", report.fetched + p.count))
        bus.on(EMAIL_SWEPT, build_queue_handler(None, emit=bus.emit_sync))
        bus.on(EMAIL_NEW_ARRIVED, lambda p: released.append(int(p.count)))
        _fetch(report, bus)
        released_unjudged = sum(released)
        router = TierRouter.load_from_yaml(config_path("llm_tiers.yaml"))
        _classify(report, router, bus.emit_sync)
        report.released_unjudged = released_unjudged
        _labels(report)
        _digest(report, router)
    report.network_attempts = len(attempts)
    _audit(report, started)
    return report


def render_summary(report: DemoReport) -> str:
    """The "what IRIS just did" block."""
    buckets = ", ".join(f"{n} {name}" for name, n in report.buckets.items()) or "none"
    hooks = ", ".join(f"{n} {name}" for name, n in report.audit_by_hook.items()) or "none"
    store = report.home / "data" / "email.db"
    if report.fetched:
        lines = [
            "## What IRIS just did",
            f"- Fetched {report.fetched} new email(s) from the demo mailbox "
            f"({report.stored} stored in {store}).",
            f"- Released {report.released_unjudged} at once (promotions and social tabs, "
            "your own sent mail): nothing to judge there.",
            f"- Judged {report.judged} with the email judge: {buckets}.",
        ]
    else:
        lines = [
            "## What IRIS just did",
            f"- Nothing new in the demo mailbox: its {report.stored} emails were fetched and "
            f"judged on the first run ({store}). The digest above is rebuilt from them.",
        ]
        if report.judged:
            lines.append(f"- Judged {report.judged} still waiting: {buckets}.")
    if report.unsure:
        lines.append(f"- {report.unsure} were too thin to call: they wait for you as Unsure.")
    if report.errors:
        lines.append(f"- {report.errors} could not be judged (see the judgments list).")
    lines += _label_lines(report)
    lines += [
        f"- Model calls: {report.llm_calls}, all on "
        f"{', '.join(report.providers) or 'no provider'} (the scripted demo model, "
        "runs local). No cloud call, no credentials.",
        f"- Governance audit rows written: {report.audit_rows} ({hooks}), in "
        f"{report.home / 'governance' / 'audit.db'}.",
        f"- Network connections attempted: {report.network_attempts} (the demo refuses "
        "them all).",
        "",
        f"Everything is in {report.home}. Run it again: nothing is fetched or judged twice.",
        "Connect your own mailbox with `iris email setup` when you are ready.",
    ]
    return "\n".join(lines)


def _label_lines(report: DemoReport) -> list[str]:
    owner = (
        "  On your own mailbox IRIS writes nothing until you approve it yourself: "
        "`iris email setup` step 6 (Label preview and approval), or "
        "`iris email writes approve --account <id>`."
    )
    if report.writes_approved_now:
        approval = (
            f"approved mailbox writes for the demo's own synthetic account ({DEMO_ACCOUNT}; "
            f"{report.writes_approval_ref}), as the demo, never for any other account"
        )
    else:
        approval = f"used the demo's approval from its first run ({report.writes_approval_ref})"
    labelled = (
        f"labelled {report.labels_written} email(s) in the demo mailbox"
        if report.labels_written
        else "no new label was due"
    )
    lines = [f"- Labels (setup step 6): {labelled}; {approval}.", owner]
    if report.labels_failed:
        lines.append(
            f"- {report.labels_failed} label write(s) failed: {report.label_error or 'see log'}"
        )
    return lines


def render_label_preview(report: DemoReport) -> str:
    """Step 6's preview, as ``iris email setup`` shows it before asking."""
    return "\n".join(
        [
            "## Label preview (email setup step 6)",
            *(f"- {line}" for line in report.label_preview),
        ]
    )


def main() -> int:
    from iris_harness.sdk.cli import console, print_error

    try:
        report = run_demo()
    except DemoEnvironmentError as exc:
        print_error(str(exc))
        return 2
    console.print(report.digest, markup=False, highlight=False)
    console.print("")
    console.print(render_label_preview(report), markup=False, highlight=False)
    console.print("")
    console.print(render_summary(report), markup=False, highlight=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
