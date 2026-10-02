"""``iris email setup`` -- the onboarding flow at a terminal (OSS plan R4).

A thin renderer over :class:`onboarding.Onboarding`: it picks the account, asks the
owner what a waiting step needs (unless ``--yes``), and prints what each step did. The
API (``onboarding_api.py``) drives the same machine, so both surfaces see one state.

The CLI runs with no runtime built, so it mounts the mail providers the provider plugins
offer on the CLI seam (``email.providers.mount_cli_mail_providers``) and reads the model
tiers from ``llm_tiers.yaml`` itself.

Exit codes: 0 done (or nothing to do), 2 bad input, 3 stopped: setup is waiting (on the
owner's decision, or on something outside setup); run it again to resume.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Annotated, Any

import typer

from iris_harness.sdk.cli import console, print_error

if TYPE_CHECKING:
    from .onboarding import Onboarding, OnboardingDeps, OnboardingState

EXIT_WAITING = 3


def cli_deps(*, interactive: bool) -> OnboardingDeps:
    """Setup's calls in a CLI process: the config dir's tiers for the judge, the namer
    and the digest. The OS keyring is read only at an interactive terminal.

    There is no running :class:`IrisRuntime` here to hand fetch/classify the harness's
    shared Activity spine, so this builds its own small one -- same ``activities.db``,
    same reconciliation, a worker of its own -- lazily, and only once per process: the
    CLI's own poll loop (``cmd_email_setup``) calls ``advance`` many times a second, and
    a fresh :class:`ActivityRunner` (its own thread pool) on every one of those would
    leak threads for no reason."""
    from iris_harness.sdk.config import config_dir, config_path
    from iris_harness.sdk.llm import TierRouter, make_narrative_llm_call

    from .discovery import GovernedNamingClient
    from .judge import llm_from_router
    from .onboarding import ActivityJobs, OnboardingDeps

    tiers = config_path("llm_tiers.yaml")
    try:
        router: Any = TierRouter.load_from_yaml(tiers)
    except Exception:  # noqa: BLE001 — no tiers file: the steps that need a model say so
        router = None

    def judge() -> Any:
        return llm_from_router(router)

    def namer() -> Any:
        call = llm_from_router(router)
        return GovernedNamingClient(call) if call is not None else None

    def narrate() -> Any:
        return make_narrative_llm_call(router) if router is not None else None

    jobs: list[ActivityJobs] = []

    def activities() -> ActivityJobs:
        if not jobs:
            from iris_harness.sdk.activities import ActivityRunner, ActivityStore
            from iris_harness.sdk.persistence import data_path

            store = ActivityStore(db_path=data_path("activities.db"))
            store.ensure_schema()
            store.reconcile_orphaned()
            runner = ActivityRunner(store=store, max_workers=1)
            jobs.append(ActivityJobs(submit=runner.submit, get=store.get))
        return jobs[0]

    return OnboardingDeps(
        config_dir=config_dir(),
        judge_llm=judge,
        naming_client=namer,
        narrate=narrate,
        activities=activities,
        read_keyring=interactive,
    )


def _pick_account(
    machine: Onboarding, account: str | None, provider: str | None, yes: bool
) -> str | None:
    from .demo.provider import DEMO_ACCOUNT
    from .onboarding import candidate_accounts

    if account:
        return account.strip()
    if provider == "demo":
        return DEMO_ACCOUNT
    unfinished = [s.account_id for s in machine.states() if not s.complete]
    if len(unfinished) == 1:
        return unfinished[0]
    candidates = [a for a in candidate_accounts(machine.deps) if a not in unfinished]
    choices = unfinished + candidates
    if len(choices) == 1:
        return choices[0]
    if not choices:
        console.print("No mailbox is connected yet. Connect one, then run setup again:")
        for hint in machine.connect_hints():
            console.print(f"  {hint['provider']}: {hint['command']}")
        console.print("  or try the synthetic mailbox: iris email demo")
        return None
    if yes:
        print_error("more than one account: pick one with --account " + ", ".join(choices))
        raise typer.Exit(2)
    for i, choice in enumerate(choices, 1):
        console.print(f"  {i}. {choice}")
    picked = int(typer.prompt("Set up which account", type=int, default=1))
    if not 1 <= picked <= len(choices):
        print_error("no such account")
        raise typer.Exit(2)
    return choices[picked - 1]


def _print(text: str) -> None:
    console.print(text, markup=False, highlight=False)


def _ask(machine: Onboarding, state: OnboardingState, interactive: bool) -> dict[str, Any] | None:
    """What the owner says to a waiting decision, or None to stop here."""
    from .onboarding import DECISION, acceptable_ids, parse_ids

    if state.waiting_kind != DECISION or not interactive:
        return None
    if state.step == "connect":
        return {"create_master_key": typer.confirm("Create a vault master key now?", True)}
    if state.step == "review_categories":
        ids = acceptable_ids(state)
        if typer.confirm(f"Accept all {len(ids)} proposed categories?", True):
            return {"accept_categories": tuple(ids)}
        raw = typer.prompt("Category ids to accept (comma separated, or none)", default="none")
        return {"accept_categories": parse_ids(raw)}
    if state.step == "label_approval":
        preview = machine.label_preview(state.account_id)
        _print("\n".join(f"  {line}" for line in preview.lines(machine.config)))
        # Default no: the owner's explicit yes is the only thing that lets IRIS write.
        return {"approve_writes": typer.confirm(f"Let IRIS change {state.account_id}?", False)}
    return None


def cmd_email_setup(
    account: Annotated[
        str | None,
        typer.Option("--account", help="The account to set up (provider:address)."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="With no --account: 'demo' sets up the synthetic mailbox (inside a demo "
            "home only).",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Take every default without asking, except mailbox writes "
            "(those need --approve-writes).",
        ),
    ] = False,
    approve_writes: Annotated[
        bool,
        typer.Option(
            "--approve-writes",
            help="Approve the label preview: let IRIS label mail in this account.",
        ),
    ] = False,
    decline_writes: Annotated[
        bool,
        typer.Option("--decline-writes", help="Decline it: keep the mailbox read-only."),
    ] = False,
    restart: Annotated[
        bool,
        typer.Option(
            "--restart",
            help="Forget this account's setup and start over (fetched mail, judgments, "
            "categories and approvals stay).",
        ),
    ] = False,
    status: Annotated[
        bool, typer.Option("--status", help="Show where setup stands; change nothing.")
    ] = False,
) -> None:
    """Set up email, step by step: connect, fetch, discover categories, classify, preview
    and approve labels, first digest, review queue, keep it current. Resumable."""
    from iris_personal.email.providers import mount_cli_mail_providers

    from .onboarding import (
        ACTIVITY,
        COMPLETE,
        IN_PROGRESS,
        STEPS,
        Inputs,
        Onboarding,
        OnboardingError,
        render_step,
        render_sweep,
        render_waiting,
    )

    if approve_writes and decline_writes:
        print_error("--approve-writes and --decline-writes contradict each other")
        raise typer.Exit(2)
    interactive = not yes
    mount_cli_mail_providers()
    machine = Onboarding(cli_deps(interactive=interactive))
    picked = _pick_account(machine, account, provider, yes)
    if picked is None:
        return
    if restart:
        forgot = machine.restart(picked)
        console.print(f"[dim]{'forgot' if forgot else 'no'} setup state for {picked}[/dim]")
    if status:
        state = machine.state(picked)
        sweep = render_sweep(machine.sweep_status(picked), machine.config)
        if state is None:
            console.print(f"No setup for {picked} yet.")
            _print(sweep)
            return
        for step in STEPS:
            mark = "x" if step in state.results else (">" if step == state.step else " ")
            console.print(f"  [{mark}] {machine.config.titles[step]}", markup=False)
        if state.waiting_for:
            _print(render_waiting(state, machine.config))
        _print(sweep)
        return

    answers: dict[str, Any] = {
        "assume_defaults": yes,
        "approve_writes": True if approve_writes else (False if decline_writes else None),
    }
    console.print(f"[dim]email setup for {picked}[/dim]")
    shown: set[str] = set()
    asked: set[str] = set()
    last_activity_line = ""
    before = machine.state(picked)
    if before is not None:
        shown = set(before.results)
    while True:
        try:
            state = machine.advance(picked, Inputs(**answers))
        except OnboardingError as exc:
            print_error(str(exc))
            raise typer.Exit(2) from exc
        if state.notice:
            _print(state.notice)
        for step in STEPS:
            if step in state.results and step not in shown:
                shown.add(step)
                _print(render_step(machine.config, step, state.results[step]))
        if state.step == COMPLETE:
            return
        if state.status == IN_PROGRESS:
            continue
        if state.waiting_kind == ACTIVITY:
            # A background job (fetch, classify), not a question: show it's moving
            # and poll again, rather than asking anything or giving up the terminal.
            if state.waiting_for != last_activity_line:
                console.print(f"[dim]… {state.waiting_for}[/dim]", markup=False)
                last_activity_line = state.waiting_for
            time.sleep(0.5)
            continue
        # Asked once per step: an answer that leaves the step waiting (no key wanted)
        # stops here rather than asking again.
        said = None if state.step in asked else _ask(machine, state, interactive)
        asked.add(state.step)
        if said is None:
            _print(render_waiting(state, machine.config))
            console.print("[dim]Run `iris email setup` again to resume.[/dim]")
            raise typer.Exit(EXIT_WAITING)
        answers.update(said)


__all__ = ["EXIT_WAITING", "cli_deps", "cmd_email_setup"]
