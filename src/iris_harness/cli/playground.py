"""``iris playground`` — run, list and diff YAML harness scenarios.

The playground is the harness's test bench: encode a behavior as a YAML
scenario (input + flags + expectations) and run it against the real runtime to
see how it routes, which intercept answers, what tools fire, and whether the
guardrails hold. The suite doubles as the regression net for refactors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from iris_harness.cli.render import console, print_error
from iris_harness.playground.baseline import diff as diff_snapshots
from iris_harness.playground.baseline import read_baseline, snapshot, write_baseline
from iris_harness.playground.loader import discover_suites, load_suite
from iris_harness.playground.models import ScenarioResult, SuiteResult
from iris_harness.playground.service import run_suite

playground_app = typer.Typer(
    name="playground",
    help="Run/list/diff YAML harness scenarios (the test bench + regression net).",
    no_args_is_help=True,
)

ScenarioDir = Annotated[
    Path | None,
    typer.Option("--dir", help="Scenario directory (default: config/playground)."),
]


@playground_app.command("list")
def cmd_list(scenario_dir: ScenarioDir = None) -> None:
    """List discoverable scenario suites and their cases."""
    suites = discover_suites(scenario_dir)
    if not suites:
        console.print(
            "  [yellow]no suites found[/yellow] "
            "[dim](looked under config/playground; override with --dir)[/dim]"
        )
        return
    for path in suites:
        suite = load_suite(path)
        console.print(f"  [bold cyan]{suite.name}[/bold cyan]  [dim]{path}[/dim]")
        if suite.description:
            console.print(f"    [dim]{suite.description}[/dim]")
        for s in suite.scenarios:
            tags = f"  [dim]{', '.join(s.tags)}[/dim]" if s.tags else ""
            console.print(f"    · {s.name}{tags}")


@playground_app.command("run")
def cmd_run(
    suite_ref: Annotated[
        str | None,
        typer.Argument(help="Suite name or path to a YAML file. Omit to run all."),
    ] = None,
    scenario_dir: ScenarioDir = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show every assertion, not just failures.")
    ] = False,
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", "-b", help="Write the run's outcomes to this baseline JSON."),
    ] = None,
) -> None:
    """Run one suite (by name or path) or all discoverable suites.

    Builds an in-process runtime with warmup disabled, so deterministic
    intercept/routing scenarios need no live Ollama. Exits non-zero on any
    failure, so it drops straight into CI.
    """
    paths = _resolve_suite_paths(suite_ref, scenario_dir)
    if not paths:
        print_error(f"no suite matched {suite_ref!r} (looked under config/playground; use --dir)")
        raise typer.Exit(2)

    any_failed = False
    for path in paths:
        try:
            suite = load_suite(path)
        except ValueError as exc:
            print_error(str(exc))
            any_failed = True
            continue
        result = run_suite(suite)
        _render_suite(result, verbose=verbose)
        if baseline is not None:
            write_baseline(baseline, result)
            console.print(f"    [dim]baseline written → {baseline}[/dim]")
        any_failed = any_failed or not result.ok

    raise typer.Exit(1 if any_failed else 0)


@playground_app.command("diff")
def cmd_diff(
    suite_ref: Annotated[str, typer.Argument(help="Suite name or path to a YAML file.")],
    baseline: Annotated[
        Path, typer.Option("--baseline", "-b", help="Baseline JSON to compare against.")
    ],
    scenario_dir: ScenarioDir = None,
) -> None:
    """Run a suite and diff its outcomes against a saved baseline.

    The regression workflow: `run <suite> --baseline before.json` on the old
    code, then `diff <suite> --baseline before.json` after a refactor. Exits
    non-zero if any scenario's behavior moved.
    """
    paths = _resolve_suite_paths(suite_ref, scenario_dir)
    if not paths:
        print_error(f"no suite matched {suite_ref!r}")
        raise typer.Exit(2)
    if not baseline.is_file():
        print_error(f"baseline not found: {baseline} (create it with `run --baseline`)")
        raise typer.Exit(2)

    current = run_suite(load_suite(paths[0]))
    deltas = diff_snapshots(read_baseline(baseline), snapshot(current))
    if not deltas:
        console.print(f"  [green]no behavior change[/green]  [dim]{current.suite_name}[/dim]")
        raise typer.Exit(0)

    console.print(f"  [yellow]{len(deltas)} scenario(s) changed[/yellow]")
    for d in deltas:
        console.print(f"    [bold]{d.name}[/bold]  [magenta]{d.status}[/magenta]")
        for field, before, after in d.changes:
            console.print(f"      {field}: [red]{before!r}[/red] → [green]{after!r}[/green]")
    raise typer.Exit(1)


def _resolve_suite_paths(suite_ref: str | None, scenario_dir: Path | None) -> list[Path]:
    if suite_ref is None:
        return discover_suites(scenario_dir)
    as_path = Path(suite_ref)
    if as_path.is_file():
        return [as_path]
    # Match by suite name (file stem) among discovered suites.
    return [p for p in discover_suites(scenario_dir) if p.stem == suite_ref]


def _render_suite(result: SuiteResult, *, verbose: bool) -> None:
    mark = "[green]PASS[/green]" if result.ok else "[red]FAIL[/red]"
    console.print(
        f"\n  {mark}  [bold]{result.suite_name}[/bold]  "
        f"[dim]{result.passed}/{result.total} passed[/dim]"
    )
    for r in result.results:
        _render_scenario(r, verbose=verbose)


def _render_scenario(r: ScenarioResult, *, verbose: bool) -> None:
    dot = "[green]·[/green]" if r.passed else "[red]x[/red]"
    handler = r.handler or "agent-loop"
    console.print(
        f"    {dot} {r.scenario_name}  "
        f"[dim]intent={r.intent} handler={handler} {r.duration_ms:.0f}ms[/dim]"
    )
    if r.error:
        console.print(f"      [red]error:[/red] {r.error}")
    shown = r.assertions if verbose else r.failed_assertions
    for a in shown:
        tick = "[green]✓[/green]" if a.ok else "[red]✗[/red]"
        console.print(f"      {tick} {a.field}  [dim]want={a.expected!r} got={a.actual!r}[/dim]")
