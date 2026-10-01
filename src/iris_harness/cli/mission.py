"""``iris mission`` CLI — inspect, approve (run), and reject auto-created missions.

Auto-created missions (``IRIS_MISSION_AUTOCREATE``) land ``PENDING`` and surface in the
Action Center as an approval task whose action is ``iris mission run <id>`` — so
approval is a deliberate, channel-agnostic step. ``run`` executes the mission through
the governed agent; ``reject`` cancels it. Both resolve the Action-Center task.
"""

from __future__ import annotations

from typing import Annotated

import typer

from iris_harness.cli.render import console, print_error
from iris_harness.foundation.paths import data_dir
from iris_harness.services.missions.models import MissionStatus
from iris_harness.services.missions.store import MissionStore

mission_app = typer.Typer(
    name="mission",
    help="Missions — multi-step tracked tasks (auto-created, propose-not-act).",
    no_args_is_help=True,
)


def _store() -> MissionStore:
    return MissionStore(db_path=data_dir() / "missions.db")


def _resolve_task(mission_id: str) -> None:
    """Close the Action-Center approval task once the mission is run/rejected."""
    try:
        from iris_harness.services.tasks import TaskStore

        ts = TaskStore(db_path=data_dir() / "tasks.db")
        task = ts.get_by_dedup_key(f"mission-approval:{mission_id}")
        if task is not None:
            ts.complete(task.id)
    except Exception:  # noqa: BLE001, S110 — best-effort task cleanup
        pass


@mission_app.command("list")
def cmd_list(
    show_all: Annotated[bool, typer.Option("--all", help="Include finished missions.")] = False,
) -> None:
    """List active (or all) missions."""
    missions = _store().list_all() if show_all else _store().list_active()
    if not missions:
        console.print("[dim]No missions.[/dim]")
        return
    for m in missions:
        src = m.metadata.get("source", "?")
        console.print(f"[bold]{m.id}[/bold]  {m.status}  [{src}]  {m.name}  ({len(m.steps)} steps)")


@mission_app.command("run")
def cmd_run(
    mission_id: Annotated[str, typer.Argument(help="Mission id (from `iris mission list`).")],
) -> None:
    """Approve & run a mission through the governed agent (one step per query)."""
    from iris_harness.runtime import build_runtime

    runtime = build_runtime()
    mission = runtime.mission_engine.store.load(mission_id)
    if mission is None:
        print_error(f"mission {mission_id} not found")
        raise typer.Exit(1)
    console.print(f"Running mission [bold]{mission.name}[/bold] ({len(mission.steps)} step(s))…")
    done = runtime.mission_engine.run(mission)
    _resolve_task(mission_id)
    for i, step in enumerate(done.steps, 1):
        console.print(f"  {i}. [{step.status}] {step.name}")
        if step.output:
            console.print(f"     {step.output[:300]}")
        if step.error:
            console.print(f"     [red]{step.error}[/red]")
    console.print(f"Mission [bold]{done.status}[/bold].")


@mission_app.command("reject")
def cmd_reject(
    mission_id: Annotated[str, typer.Argument(help="Mission id.")],
) -> None:
    """Reject (cancel) a proposed mission and dismiss its Action-Center task."""
    store = _store()
    mission = store.load(mission_id)
    if mission is None:
        print_error(f"mission {mission_id} not found")
        raise typer.Exit(1)
    mission.status = MissionStatus.CANCELLED
    store.save(mission)
    _resolve_task(mission_id)
    console.print(f"Mission {mission_id} rejected.")
