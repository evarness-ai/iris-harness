"""The one terminal console every ``iris`` command prints to.

It lives at the bottom layer so the core CLI (`cli.render`) and plugin CLIs (through
`iris_harness.sdk.cli`) share one ``rich`` console object rather than each building
their own.
"""

from __future__ import annotations

from rich.console import Console

console = Console(highlight=False)


def print_error(msg: str) -> None:
    """Print ``error: <msg>`` in the CLI's error style, then a blank line."""
    console.print(f"[bold red]error:[/bold red] {msg}")
    console.print()


__all__ = ["console", "print_error"]
