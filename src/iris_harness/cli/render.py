"""Rich-based rendering layer for the IRIS CLI."""

from __future__ import annotations

import re
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

# `console` is the one console object, shared with plugin CLIs through iris_harness.sdk.cli.
from iris_harness.foundation.console import console as console

if TYPE_CHECKING:
    from .session import Session

# ---------------------------------------------------------------------------
# Context window sizes (tokens) for known model families.
# Matched by substring so "gpt-4o-mini-2024" hits "gpt-4o".
# ---------------------------------------------------------------------------
_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI / GitHub Models
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o3": 200_000,
    "o4": 200_000,
    # Anthropic Claude 4.x
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
    # Anthropic Claude 3.x
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    # Meta / Ollama
    "llama3.2": 128_000,
    "llama3.1": 128_000,
    "llama3": 8_192,
    "llama2": 4_096,
    # Mistral
    "mistral": 32_768,
    "mixtral": 32_768,
    # Other
    "nemotron": 4_096,
    "phi3": 128_000,
    "phi-3": 128_000,
    "gemma": 8_192,
    "deepseek": 128_000,
    "qwen": 128_000,
}

_BAR_FILLED = "█"
_BAR_EMPTY = "░"
_BAR_WIDTH = 12

IRIS_VERSION = "0.1.0"

err_console = Console(stderr=True)

_LOGO = """\
[#4169E1]      ·   ˚    ✦     ·    ˚    ✦    ·   ˚[/#4169E1]

[bold #4169E1]   ██╗██████╗ ██╗███████╗[/bold #4169E1]
[bold #2176CC]   ██║██╔══██╗██║██╔════╝[/bold #2176CC]
[bold #0D98BA]   ██║██████╔╝██║███████╗[/bold #0D98BA]
[bold #00B4D8]   ██║██╔══██╗██║╚════██║[/bold #00B4D8]
[bold #00CED1]   ██║██║  ██║██║███████║[/bold #00CED1]
[bold cyan]   ╚═╝╚═╝  ╚═╝╚═╝╚══════╝[/bold cyan]

[cyan]   ──────────────── ◎ ────────────────[/cyan]
   Intelligent Reasoning & Integration System

[cyan]      ˚    ·   ✦    ˚    ·    ✦    ˚    ·[/cyan]\
"""


def print_banner(
    session_id: str | None = None,
    api_url: str = "http://localhost:8003",
    provider: str = "",
    model: str = "",
) -> None:
    console.print(_LOGO)
    console.print()
    meta: list[str] = [f"[dim]v{IRIS_VERSION}[/dim]"]
    if session_id:
        meta.append(f"[dim]session[/dim] [cyan]{session_id[:8]}[/cyan]")
    if provider and model:
        meta.append(f"[dim]via[/dim] [cyan]{provider}[/cyan] [dim]·[/dim] [cyan]{model}[/cyan]")
    elif provider:
        meta.append(f"[dim]via[/dim] [cyan]{provider}[/cyan]")
    meta.append(f"[dim]{api_url}[/dim]")
    console.print("   " + "  [dim]·[/dim]  ".join(meta))
    console.print()


def print_response(
    text: str,
    *,
    intent: str = "",
    agent: str = "",
    has_errors: bool = False,
) -> None:
    if text:
        console.print(Markdown(text))

    if has_errors:
        tag = "[bold red]✗[/bold red]"
        meta: list[str] = []
        if intent:
            meta.append(f"intent={intent}")
        if agent:
            meta.append(f"agent={agent}")
        console.print(f"  {tag}  [dim]{' · '.join(meta)}[/dim]")
        console.print()


def print_error(msg: str) -> None:
    console.print(f"[bold red]error:[/bold red] {msg}")
    console.print()


def print_sessions(sessions: list[Session]) -> None:
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(
        box=box.ROUNDED,
        border_style="dim",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
        pad_edge=True,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Started", width=17)
    table.add_column("Msgs", justify="right", width=5)
    table.add_column("Directory", style="dim")

    for i, s in enumerate(sessions, 1):
        table.add_row(
            str(i),
            s.id[:8],
            s.created_at.strftime("%Y-%m-%d %H:%M"),
            str(s.message_count),
            s.cwd,
        )

    console.print(table)
    console.print()


def print_help_commands(commands: dict[str, str]) -> None:
    table = Table(
        box=None,
        show_header=False,
        padding=(0, 2),
        show_edge=False,
    )
    table.add_column("command", style="cyan", width=14)
    table.add_column("description", style="dim")
    for name, desc in commands.items():
        table.add_row(name, desc)
    console.print(table)
    console.print()


_STRIP_COMMANDS = ["/help", "/provider", "/skills", "/compact", "/clear", "/sessions", "/exit"]


def format_command_strip() -> str:
    """Return a dim ANSI string of the footer command strip."""
    items = "  ".join(_STRIP_COMMANDS)
    return f"  \033[2m{items}\033[0m"


def render_divider() -> None:
    """Full-width separator — used between header, body, and prompt."""
    console.rule(style="bright_black")


@contextmanager
def spinner(label: str = "thinking…") -> Generator[None, None, None]:
    if console.is_terminal:
        with console.status(f"[dim]{label}[/dim]", spinner="dots"):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Footer metrics
# ---------------------------------------------------------------------------


@dataclass
class FooterMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    router_model: str = ""
    context_window: int = 0
    latency_ms: float = 0.0
    cpu_pct: float = field(default=-1.0)
    mem_used_gb: float = field(default=-1.0)
    mem_total_gb: float = field(default=-1.0)
    gpu_pct: float = field(default=-1.0)
    gpu_mem_used_gb: float = field(default=-1.0)
    gpu_mem_total_gb: float = field(default=-1.0)


def sample_system_metrics() -> dict[str, float]:
    """Collect CPU, memory, and GPU metrics. Returns {} keys on failure."""
    import logging

    _log = logging.getLogger(__name__)

    result: dict[str, float] = {}
    try:
        import psutil

        result["cpu_pct"] = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        result["mem_used_gb"] = vm.used / 1024**3
        result["mem_total_gb"] = vm.total / 1024**3
    except Exception:  # noqa: BLE001
        _log.debug("psutil metrics unavailable", exc_info=True)

    # GPU — try nvidia-smi, then skip gracefully (no nvidia-smi on non-NVIDIA hosts)
    try:
        import subprocess

        # fixed argv, no shell, no user input — not an injection vector
        proc = subprocess.run(
            [  # noqa: S607
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if proc.returncode == 0:
            parts = [p.strip() for p in proc.stdout.strip().split(",")]
            if len(parts) >= 3:
                result["gpu_pct"] = float(parts[0])
                result["gpu_mem_used_gb"] = float(parts[1]) / 1024
                result["gpu_mem_total_gb"] = float(parts[2]) / 1024
    except Exception:  # noqa: BLE001
        _log.debug("nvidia-smi unavailable, GPU metrics skipped")

    return result


def _context_window_for(model: str) -> int | None:
    """Return the context window size (tokens) for a model name, or None if unknown."""
    lower = model.lower()
    for key, size in _CONTEXT_WINDOWS.items():
        if key in lower:
            return size
    return None


def _ctx_bar(used: int, total: int) -> str:
    ratio = min(used / total, 1.0)
    filled = round(ratio * _BAR_WIDTH)
    bar = _BAR_FILLED * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)
    color = "green" if ratio < 0.6 else "yellow" if ratio < 0.85 else "red"
    return f"[{color}]{bar}[/{color}]"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000:
        return f"{n / 1000:.1f}k"
    return str(n)


def render_footer(metrics: FooterMetrics) -> None:
    """Render the fixed status bar below the prompt divider."""
    parts: list[str] = []

    if metrics.prompt_tokens or metrics.completion_tokens:
        parts.append(
            f"[dim]↑[/dim][cyan]{_fmt_tokens(metrics.prompt_tokens)}[/cyan]"
            f"  [dim]↓[/dim][cyan]{_fmt_tokens(metrics.completion_tokens)}[/cyan]"
        )

    ctx_total = metrics.context_window or _context_window_for(metrics.model)
    if ctx_total and metrics.prompt_tokens:
        bar = _ctx_bar(metrics.prompt_tokens, ctx_total)
        parts.append(
            f"[dim]ctx[/dim] {bar}"
            f" [dim]{_fmt_tokens(metrics.prompt_tokens)}/{_fmt_tokens(ctx_total)}[/dim]"
        )

    if metrics.model:
        short = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", metrics.model)
        parts.append(f"[dim cyan]{short}[/dim cyan]")

    if metrics.latency_ms:
        parts.append(f"[dim]{metrics.latency_ms:.0f}ms[/dim]")

    if metrics.cpu_pct >= 0:
        cpu_color = "green" if metrics.cpu_pct < 60 else "yellow" if metrics.cpu_pct < 85 else "red"
        parts.append(f"[dim]cpu[/dim] [{cpu_color}]{metrics.cpu_pct:.0f}%[/{cpu_color}]")

    if metrics.mem_used_gb >= 0:
        parts.append(
            f"[dim]mem[/dim] [dim]{metrics.mem_used_gb:.1f}/{metrics.mem_total_gb:.0f}GB[/dim]"
        )

    if metrics.gpu_pct >= 0:
        gpu_color = "green" if metrics.gpu_pct < 60 else "yellow" if metrics.gpu_pct < 85 else "red"
        parts.append(
            f"[dim]gpu[/dim] [{gpu_color}]{metrics.gpu_pct:.0f}%[/{gpu_color}]"
            f" [dim]{metrics.gpu_mem_used_gb:.1f}/{metrics.gpu_mem_total_gb:.0f}GB[/dim]"
        )

    if not parts:
        return

    sep = "  [dim]·[/dim]  "
    console.print(f"  {sep.join(parts)}")
