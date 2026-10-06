"""Slash command registry and handlers for the IRIS interactive REPL.

Commands are registered at module import time. Add new commands by calling
register() with a SlashCommand instance at the bottom of this file.
"""

from __future__ import annotations

import json
import shlex
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from iris_harness.cli.api_client import harness_urlopen
from iris_harness.foundation.auth import auth_headers

if TYPE_CHECKING:
    from iris_harness.llm.providers import ProviderManager, ProviderProfile

    from .render import FooterMetrics
    from .session import Session, SessionManager


# ── Context ────────────────────────────────────────────────────────────────────


@dataclass
class REPLContext:
    """Mutable bag of state shared across the REPL loop and all command handlers."""

    session: Session
    session_manager: SessionManager
    api_url: str
    provider_manager: ProviderManager
    footer: FooterMetrics | None = None
    footer_lock: threading.Lock = field(default_factory=threading.Lock)
    redraw_banner: Callable[[], None] | None = field(default=None, repr=False)
    # Raw trace (cmd, stdout, stderr, JSON tool calls) buffered for /trace
    # so users can expand the most recent turn's tool activity on demand.
    last_trace: str = ""
    # Live status text shown in the bottom toolbar while a chat turn is in flight.
    # Empty string means idle (toolbar shows the metrics line instead).
    # Updated by the REPL loop under ``footer_lock``.
    thinking: str = ""
    # When true, chat requests ask the runtime to run strict pre-response checks.
    strict_mode: bool = False


@dataclass(frozen=True)
class SkillListEntry:
    """One row in the `/skills list` display."""

    name: str
    source: str
    status: str
    tools: str
    agents: str
    location: str


# ── Registry ───────────────────────────────────────────────────────────────────


class CompleterSession(Protocol):
    """The one field of a session a completer reads."""

    @property
    def cwd(self) -> str: ...


class CompleterContext(Protocol):
    """The slice of the REPL context an argument completer may read.

        A completer runs in two places: the REPL, where it gets a real ``REPLContext``,
        and the API's ``/slash/commands`` endpoint, where there is no REPL and the
        server builds a stand-in. Typing the parameter as the full ``REPLContext``
        forced that stand-in to be a lie; naming the two fields any completer actually
        reads makes both callers honest, and ``REPLContext`` satisfies it structurally.

    It names the two things completers read, and no more: the provider profiles, and
        the session's working directory. Widening it to everything ``REPLContext``
        carries would make the stand-in fail for fields nobody uses; narrowing it past
        these two breaks `/queue`, which resolves proposals against `session.cwd`.

        Read-only properties, not bare attributes: a protocol with mutable attributes
        is invariant, and a narrower implementation then fails to match.
    """

    @property
    def session(self) -> CompleterSession: ...

    @property
    def provider_manager(self) -> ProviderManager: ...


ArgumentCompleter = Callable[[CompleterContext, str], list[tuple[str, str]]]


@dataclass
class SlashCommand:
    name: str
    handler: Callable[[REPLContext, str], bool]
    args: str = ""
    description: str = ""
    hidden: bool = False
    argument_completer: ArgumentCompleter | None = None


_REGISTRY: dict[str, SlashCommand] = {}
_ORDER: list[str] = []


def register(cmd: SlashCommand) -> SlashCommand:
    _REGISTRY[cmd.name] = cmd
    if cmd.name not in _ORDER:
        _ORDER.append(cmd.name)
    return cmd


def dispatch(ctx: REPLContext, raw: str) -> bool | None:
    """Dispatch a raw slash-command string.

    Returns False to exit the REPL, True to continue, None if not found.
    """
    parts = raw.split(maxsplit=1)
    name = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""
    cmd = _REGISTRY.get(name)
    if cmd is None:
        return None
    return cmd.handler(ctx, args)


def strip_commands() -> list[SlashCommand]:
    """Return visible commands in registration order for the footer strip."""
    return [_REGISTRY[n] for n in _ORDER if n in _REGISTRY and not _REGISTRY[n].hidden]


def all_names() -> list[str]:
    """Return all registered command names (including hidden) for tab completion."""
    return list(_REGISTRY.keys())


def get(name: str) -> SlashCommand | None:
    """Return the registered command for ``name`` (case-insensitive), or None."""
    return _REGISTRY.get(name.lower())


def visible_commands() -> list[SlashCommand]:
    """Return non-hidden commands in registration order, for the popup menu."""
    return [_REGISTRY[n] for n in _ORDER if n in _REGISTRY and not _REGISTRY[n].hidden]


# ── Warmup helper ──────────────────────────────────────────────────────────────


def _warmup_url(ctx: REPLContext) -> str:
    """The API to warm against, or "" when this caller does not want warming.

    The REPL wants it: a 5-30s Ollama load is better paid now, with a spinner,
    than on the user's next message. The API running these handlers for a web
    client does not — there the call would re-enter this same process and hold a
    worker for those seconds, to pre-load a model the next chat turn loads anyway.
    """
    return ctx.api_url if getattr(ctx, "warm_models", True) else ""


def _warm_model(
    api_url: str,
    *,
    role: str,
    model: str = "",
    provider_profile: str = "",
    label: str = "",
) -> None:
    """Trigger a model warm-up on the API and print clear progress to the user.

    Shows a spinner during the call so the user knows what they're waiting for —
    Ollama can take 5-30s to load a small model into memory on first use, and we
    don't want that latency to surprise them on the next chat turn.

    No ``api_url`` means there is nothing to warm against, so this does nothing.
    That is the case when the API itself runs these handlers for a web client
    (``cli/web_commands.py``): warming there would be a blocking call from the
    process into itself, holding a worker for those same 5-30s to pre-load a
    model the next chat turn loads anyway.
    """
    if not api_url:
        return

    from .render import console, spinner

    display = label or model or role
    payload: dict[str, str] = {"role": role}
    if model:
        payload["model"] = model
    if provider_profile:
        payload["provider_profile"] = provider_profile

    req = urllib.request.Request(  # noqa: S310
        f"{api_url}/warmup",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **auth_headers()},
        method="POST",
    )
    try:
        with spinner(f"Loading {display} — first load can take 5–30s…"):
            with harness_urlopen(req, purpose="warmup", timeout=120) as resp:
                body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        console.print(
            f"  [yellow]⚠[/yellow]  Warm-up call failed (HTTP {exc.code}). "
            "[dim]Next chat turn will load the model on demand.[/dim]"
        )
        return
    except OSError as exc:
        console.print(
            f"  [yellow]⚠[/yellow]  Cannot reach IRIS API for warm-up: {exc}. "
            "[dim]Is the server running?[/dim]"
        )
        return
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]⚠[/yellow]  Warm-up failed: {exc}")
        return

    if body.get("ok"):
        secs = float(body.get("latency_ms", 0)) / 1000
        resolved = str(body.get("model") or display)
        console.print(
            f"  [bold green]✓[/bold green]  [cyan]{resolved}[/cyan] ready  "
            f"[dim]({secs:.1f}s)[/dim]"
        )
    else:
        err = body.get("error") or "unknown error"
        console.print(
            f"  [yellow]⚠[/yellow]  Warm-up failed for [cyan]{display}[/cyan]: " f"[dim]{err}[/dim]"
        )


def _api_json(
    api_url: str,
    *,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Call one JSON API endpoint and return the decoded mapping."""

    data = json.dumps(payload or {}).encode() if payload is not None else None
    req = urllib.request.Request(  # noqa: S310
        f"{api_url.rstrip('/')}/{path.lstrip('/')}",
        data=data,
        headers={"Content-Type": "application/json", **auth_headers()},
        method=method,
    )
    with harness_urlopen(req, purpose="repl-command", timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    decoded = json.loads(body) if body else {}
    if not isinstance(decoded, dict):
        raise RuntimeError(f"API returned {type(decoded).__name__}, expected object")
    return decoded


# ── Handlers ───────────────────────────────────────────────────────────────────


def _cmd_help(ctx: REPLContext, _args: str) -> bool:
    from .render import print_help_commands

    visible: dict[str, str] = {}
    for n in _ORDER:
        cmd = _REGISTRY.get(n)
        if cmd and not cmd.hidden:
            key = f"{cmd.name} {cmd.args}".strip()
            visible[key] = cmd.description
    print_help_commands(visible)
    return True


def _cmd_exit(ctx: REPLContext, _args: str) -> bool:
    from .render import console

    console.print("[dim]bye.[/dim]")
    return False


def _cmd_clear(ctx: REPLContext, _args: str) -> bool:
    from .render import console

    console.clear()
    if ctx.redraw_banner:
        ctx.redraw_banner()
    return True


def _cmd_trace(ctx: REPLContext, _args: str) -> bool:
    """Print the trace from the most recent turn."""
    from .render import console

    blocks: list[str] = []
    if ctx.last_trace:
        blocks.extend(block for block in ctx.last_trace.split("\n\n") if block.strip())

    try:
        _path, events = _load_session_events(ctx.session.id)
    except OSError:
        events = []
    if events:
        turns = _session_turns(events)
        if turns:
            blocks.append("session timeline")
            blocks.extend(_trace_event_detail(event) for event in turns[-1])

    if not blocks:
        console.print(
            "  [dim]No trace from the last turn. Send a message first, then run /trace.[/dim]"
        )
        console.print()
        return True
    console.print("  [dim]── trace from last turn ──[/dim]")
    for block in blocks:
        console.print(f"  [dim]{block}[/dim]")
        console.print()
    return True


def _cmd_session(ctx: REPLContext, args: str) -> bool:
    """Show the unified session log path and tail the last N events.

    ``/session`` shows path + last 10 events; ``/session N`` tails N events.
    Each line summarises one event (user_message/llm_call/tool_run/error/
    agent_response) so the user can audit what happened in this conversation.
    """
    from iris_harness.foundation.observability.session_log import session_log_path

    from .render import console

    path = session_log_path(ctx.session.id)
    console.print(f"  [dim]session log[/dim]  [cyan]{path}[/cyan]")

    if not path.exists():
        console.print("  [dim](no events yet — send a message first)[/dim]")
        console.print()
        return True

    raw = args.strip()
    try:
        limit = int(raw) if raw else 10
    except ValueError:
        console.print(f"  [red]invalid count: {raw!r}[/red]")
        console.print()
        return True
    limit = max(1, limit)

    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as exc:
        console.print(f"  [red]could not read log: {exc}[/red]")
        console.print()
        return True

    tail = lines[-limit:]
    console.print(f"  [dim]── last {len(tail)} event(s) ──[/dim]")
    for line in tail:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            console.print(f"  [dim]{line.rstrip()}[/dim]")
            continue
        kind = str(event.get("kind", "?"))
        ts = str(event.get("ts", ""))[11:19]  # HH:MM:SS slice
        agent = str(event.get("agent_type") or "")
        iteration = event.get("iteration")
        ctx_label = agent
        if iteration is not None:
            ctx_label = f"{agent}#{iteration}" if agent else f"iter#{iteration}"
        prefix = f"[dim]{ts}[/dim] [cyan]{kind}[/cyan]"
        if ctx_label:
            prefix += f" [magenta]({ctx_label})[/magenta]"
        if kind == "user_message":
            preview = _short(event.get("text", ""))
            console.print(f"  {prefix} [white]{preview}[/white]")
        elif kind == "agent_response":
            preview = _short(event.get("response", ""))
            console.print(f"  {prefix} [white]{preview}[/white]")
        elif kind == "llm_call":
            output = (event.get("output") or {}).get("text", "")
            tokens = event.get("tokens") or {}
            tok = tokens.get("total_tokens")
            tail_label = f" [dim]({tok} tok)[/dim]" if tok else ""
            console.print(f"  {prefix}{tail_label} [white]{_short(output)}[/white]")
        elif kind == "tool_run":
            cmd_text = _short(event.get("cmd", ""), width=80)
            exit_code = event.get("exit_code")
            color = "green" if exit_code == 0 else "red"
            console.print(f"  {prefix} [{color}]exit={exit_code}[/{color}] [dim]$[/dim] {cmd_text}")
        elif kind == "error":
            err_type = event.get("type", "Error")
            msg = _short(event.get("message", ""))
            console.print(f"  {prefix} [red]{err_type}: {msg}[/red]")
        else:
            console.print(f"  {prefix} [dim]{_short(json.dumps(event))}[/dim]")
    console.print()
    return True


def _load_session_events(session_id: str) -> tuple[Path, list[dict[str, Any]]]:
    from iris_harness.foundation.observability.session_log import session_log_path

    path = session_log_path(session_id)
    events: list[dict[str, Any]] = []
    if not path.exists():
        return path, events
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return path, events


def _session_turns(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    from iris_harness.foundation.observability.session_log import TURN_OPENING_KINDS

    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") in TURN_OPENING_KINDS and current:
            turns.append(current)
            current = []
        current.append(event)
    if current:
        turns.append(current)
    return turns


def _replay_text(text: object, *, width: int, full: bool) -> str:
    if full:
        return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
    return _short(text, width=width)


def _replay_event_summary(event: dict[str, Any], *, full: bool = False) -> tuple[str, str, str]:
    kind = str(event.get("kind", "?"))
    ts = str(event.get("ts", ""))[11:19]
    agent = str(event.get("agent_type") or "")
    iteration = event.get("iteration")
    if iteration is not None:
        agent = f"{agent}#{iteration}" if agent else f"iter#{iteration}"

    if kind == "user_message":
        return ts, "user", _replay_text(event.get("text", ""), width=240, full=full)
    if kind == "turn_open":
        label = _replay_text(event.get("label") or event.get("opener"), width=240, full=full)
        return ts, "open", f"{label} (opened by IRIS, no user message)"
    if kind == "turn.start":
        raw_payload = event.get("payload")
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        return ts, "turn", f"start · {payload.get('message_chars', '?')} chars"
    if kind == "pipeline.phase":
        return ts, "phase", str(event.get("phase") or "pipeline")
    if kind == "intent_router.end":
        raw_payload = event.get("payload")
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        intent = payload.get("intent", "?")
        agent_type = payload.get("agent_type", "?")
        confidence = payload.get("confidence")
        conf = f" · confidence {confidence}" if confidence is not None else ""
        return ts, "router", f"{intent} -> {agent_type}{conf}"
    if kind == "memory.context":
        raw_payload = event.get("payload")
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        parts = [
            f"profile={bool(payload.get('has_user_profile'))}",
            f"active={bool(payload.get('has_active_context'))}",
            f"recent={payload.get('recent_turns', 0)}",
            f"episodic={payload.get('episodic_patterns', 0)}",
        ]
        behavior = payload.get("behavior_name")
        if behavior:
            parts.append(f"behavior={behavior}")
        return ts, "memory", " · ".join(parts)
    if kind == "planner.end":
        raw_payload = event.get("payload")
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        return (
            ts,
            "plan",
            f"{payload.get('plan_size', '?')} task(s) · first={payload.get('first_agent', '?')}",
        )
    if kind in {"tool.invoke.start", "tool.invoke.end"}:
        raw_payload = event.get("payload")
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        action = "start" if kind.endswith("start") else "end"
        tool = payload.get("tool", "?")
        ok = payload.get("ok")
        suffix = f" · ok={ok}" if ok is not None else ""
        return ts, "tool", f"{action} · {tool}{suffix}"
    if kind in {"response_curator.start", "response_curator.end"}:
        raw_payload = event.get("payload")
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        if kind.endswith("start"):
            return ts, "curate", f"start · {payload.get('result_count', '?')} result(s)"
        return (
            ts,
            "curate",
            f"end · {payload.get('response_chars', '?')} chars · errors={payload.get('has_errors', '?')}",
        )
    if kind == "agent.trace":
        raw_payload = event.get("payload")
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        name = str(payload.get("event") or payload.get("name") or event.get("phase") or "trace")
        if name == "agent.start":
            return (
                ts,
                "agent",
                f"start · {payload.get('agent_type', '?')} · task={payload.get('task_id', '?')}",
            )
        if name == "llm.start":
            model = payload.get("model", "?")
            provider = payload.get("provider", "?")
            ctx = payload.get("num_ctx") or payload.get("context_window")
            governor = payload.get("governor_mode")
            details = f"{provider}/{model}"
            if ctx:
                details += f" · ctx {ctx}"
            if governor:
                details += f" · governor {governor}"
            return ts, "llm", f"start · {details}"
        if name == "llm.end":
            return ts, "llm", "end"
        if name in {"llm.invoke.start", "llm.invoke.end"}:
            action = "start" if name.endswith("start") else "end"
            provider = payload.get("provider") or payload.get("tier_provider") or "?"
            model = payload.get("model") or payload.get("tier_model") or "?"
            strategy = payload.get("strategy")
            details = f"{action} · {provider}/{model}"
            if strategy:
                details += f" · {strategy}"
            if payload.get("total_tokens"):
                details += f" · {payload.get('total_tokens')} tok"
            return ts, "llm", details
        if name == "agent.result":
            provider = payload.get("provider") or payload.get("tier_provider") or "?"
            model = payload.get("model") or payload.get("tier_model") or "?"
            tokens = payload.get("total_tokens")
            ctx = payload.get("num_ctx") or payload.get("context_window")
            details = f"{payload.get('agent_type', '?')} · {provider}/{model}"
            if ctx:
                details += f" · ctx {ctx}"
            if tokens:
                details += f" · {tokens} tok"
            return ts, "agent", details
        return ts, "trace", _replay_text(event.get("text", ""), width=240, full=full)
    if kind == "llm_call":
        model = event.get("model", "?")
        provider = event.get("provider", "?")
        raw_tokens = event.get("tokens")
        tokens = raw_tokens if isinstance(raw_tokens, dict) else {}
        tok = tokens.get("total_tokens")
        raw_output = event.get("output")
        output = raw_output if isinstance(raw_output, dict) else {}
        detail = f"{provider}/{model}"
        if tok:
            detail += f" · {tok} tok"
        tool_calls = output.get("tool_calls")
        if tool_calls:
            detail += f" · tool_calls={len(tool_calls)}"
        if agent:
            detail += f" · {agent}"
        text = _replay_text(output.get("text", ""), width=240, full=full)
        return ts, "llm", f"{detail} · {text}" if text else detail
    if kind == "tool_run":
        exit_code = event.get("exit_code", "?")
        cmd_text = _replay_text(event.get("cmd", ""), width=220, full=full)
        return ts, "tool", f"exit={exit_code} · $ {cmd_text}"
    if kind == "agent_response":
        response = _replay_text(event.get("response", ""), width=600, full=full)
        return ts, "final", response
    if kind == "error":
        message = _replay_text(event.get("message", ""), width=400, full=full)
        return ts, "error", f"{event.get('type', 'Error')}: {message}"
    return ts, kind, _replay_text(json.dumps(event, default=str), width=400, full=full)


def _trace_event_detail(event: dict[str, Any]) -> str:
    ts, label, summary = _replay_event_summary(event, full=True)
    kind = str(event.get("kind", "?"))
    lines = [f"{ts} {label}: {summary}"]
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else None
    if kind == "llm_call":
        inputs = (
            event.get("input_messages") if isinstance(event.get("input_messages"), list) else []
        )
        raw_output = event.get("output")
        output = raw_output if isinstance(raw_output, dict) else {}
        tokens = event.get("tokens") if isinstance(event.get("tokens"), dict) else {}
        lines.append(f"  model: {event.get('provider', '?')}/{event.get('model', '?')}")
        if tokens:
            lines.append(f"  tokens: {json.dumps(tokens, sort_keys=True)}")
        if event.get("duration_ms") is not None:
            lines.append(f"  duration_ms: {float(event.get('duration_ms') or 0):.0f}")
        if inputs:
            rendered_inputs = []
            for message in inputs:
                if not isinstance(message, dict):
                    continue
                content = _short(message.get("content", ""), width=220)
                rendered_inputs.append(f"{message.get('role', '?')}: {content}")
            if rendered_inputs:
                lines.append("  input:")
                lines.extend(f"    {item}" for item in rendered_inputs)
        tool_calls = output.get("tool_calls")
        if tool_calls:
            lines.append(f"  tool_calls: {json.dumps(tool_calls, default=str, sort_keys=True)}")
        text = str(output.get("text") or "").strip()
        if text:
            lines.append(f"  output: {_short(text, width=500)}")
    elif kind == "tool_run":
        if event.get("stdout"):
            lines.append(f"  stdout: {_short(event.get('stdout'), width=500)}")
        if event.get("stderr"):
            lines.append(f"  stderr: {_short(event.get('stderr'), width=500)}")
        if event.get("artifacts"):
            lines.append(f"  artifacts: {json.dumps(event.get('artifacts'), default=str)}")
    elif (
        kind
        in {
            "tool.invoke.start",
            "tool.invoke.end",
            "response_curator.start",
            "response_curator.end",
            "agent.trace",
            "intent_router.end",
            "memory.context",
            "planner.end",
        }
        and payload
    ):
        lines.append(json.dumps(payload, indent=2, sort_keys=True, default=str))
    elif kind == "agent_response":
        response = str(event.get("response") or "")
        lines.append(f"  response: {response}")
    return "\n".join(lines)


def _parse_replay_args(raw: str) -> tuple[int, bool, str | None]:
    try:
        parts = shlex.split(raw)
    except ValueError as exc:
        return 1, False, str(exc)
    limit = 1
    full = False
    for part in parts:
        if part in {"--full", "-f", "full"}:
            full = True
            continue
        if part.isdigit():
            limit = int(part)
            continue
        return limit, full, f"unknown replay option: {part!r}"
    return max(1, limit), full, None


def _cmd_replay(ctx: REPLContext, args: str) -> bool:
    """Replay the current session's JSONL timeline as a readable turn flow."""
    from .render import console

    limit, full, error = _parse_replay_args(args.strip())
    if error:
        console.print(f"  [red]{error}[/red]")
        console.print("  [dim]usage: /replay [turns] [--full][/dim]")
        console.print()
        return True

    try:
        path, events = _load_session_events(ctx.session.id)
    except OSError as exc:
        console.print(f"  [red]could not read replay log: {exc}[/red]")
        console.print()
        return True

    console.print(f"  [dim]replay log[/dim]  [cyan]{path}[/cyan]")
    if not events:
        console.print("  [dim](no events yet — send a message first)[/dim]")
        console.print()
        return True

    turns = _session_turns(events)[-limit:]
    mode = "full" if full else "summary"
    console.print(f"  [dim]── replaying last {len(turns)} turn(s) · {mode} ──[/dim]")
    for turn_index, turn in enumerate(turns, start=1):
        console.print(f"  [bold cyan]turn {turn_index}[/bold cyan]")
        for event in turn:
            ts, label, detail = _replay_event_summary(event, full=full)
            color = (
                "red"
                if label == "error"
                else "magenta" if label in {"llm", "tool", "agent"} else "cyan"
            )
            console.print(f"    [dim]{ts}[/dim] [{color}]{label:>6}[/{color}]  {detail}")
        console.print()
    return True


def _short(text: object, *, width: int = 100) -> str:
    """Render a value as a single-line preview no longer than ``width`` chars."""
    s = str(text or "").replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s if len(s) <= width else s[: width - 1] + "…"


def _cmd_info(ctx: REPLContext, _args: str) -> bool:
    from .render import console

    active = ctx.provider_manager.get_active()
    model_display = ctx.session.preferred_model or active.model
    avail = "[green]✓[/green]" if active.is_available() else "[red]✗ key missing[/red]"
    console.print(
        f"  [dim]session[/dim]  [cyan]{ctx.session.id}[/cyan]\n"
        f"  [dim]api[/dim]      [dim]{ctx.api_url}[/dim]\n"
        f"  [dim]messages[/dim] {ctx.session.message_count}\n"
        f"  [dim]cwd[/dim]      [dim]{ctx.session.cwd}[/dim]\n"
        f"  [dim]provider[/dim] [cyan]{active.display_name}[/cyan]  "
        f"[dim]{model_display}[/dim]  {avail}"
    )
    console.print()
    return True


def _cmd_sessions(ctx: REPLContext, _args: str) -> bool:
    from .render import print_sessions

    print_sessions(ctx.session_manager.list())
    return True


def _cmd_model(ctx: REPLContext, args: str) -> bool:
    from .render import console

    if not args:
        active = ctx.provider_manager.get_active()
        current = ctx.session.preferred_model or active.model
        override_note = "  [dim](override active)[/dim]" if ctx.session.preferred_model else ""
        console.print(
            f"  [dim]model[/dim]  [cyan]{current}[/cyan]{override_note}  "
            f"[dim]via[/dim] [cyan]{active.display_name}[/cyan]\n"
            f"  [dim]default:[/dim] [dim cyan]{active.model}[/dim cyan]\n"
            "  [dim]Set with[/dim] [cyan]/model <name>[/cyan]  "
            "[dim]list models with[/dim] [cyan]/model list[/cyan]  "
            "[dim]reset with[/dim] [cyan]/model reset[/cyan]  "
            "[dim]switch provider with[/dim] [cyan]/provider[/cyan]"
        )
    elif args.strip().lower() == "list":
        from iris_harness.llm.model_metadata import get_metadata

        active = ctx.provider_manager.get_active()
        console.print(
            f"  [dim]Fetching models from[/dim] [cyan]{active.display_name}[/cyan][dim]...[/dim]"
        )
        listing = active.fetch_models()
        if not listing.usable and not listing.hidden:
            reason = f"  [dim]{listing.error}[/dim]\n" if listing.error else ""
            console.print(
                "  [yellow]Could not retrieve model list.[/yellow]\n"
                f"{reason}"
                "  [dim]Check credentials or try /info to verify setup.[/dim]\n"
                f"  [dim]Provider default model:[/dim] [cyan]{active.model}[/cyan]"
            )
        else:
            from rich import box
            from rich.table import Table

            current_model = ctx.session.preferred_model or active.model
            table = Table(box=box.SIMPLE, show_header=False, pad_edge=False, border_style="dim")
            table.add_column("model", style="cyan")
            table.add_column("capabilities", style="dim")
            table.add_column("status", style="dim")
            for m in listing.usable:
                meta = get_metadata(m)
                caps = meta.display_str() if meta else ""
                status = "← active" if m == current_model else ""
                table.add_row(m, caps, status)
            console.print(table)
            if listing.from_fallback:
                console.print(
                    f"  [yellow]Live catalog unavailable[/yellow] [dim]({listing.error}); "
                    "showing known models for this provider.[/dim]"
                )
            console.print(
                "  [dim]Use[/dim] [cyan]/model <name>[/cyan]  "
                "[dim]to switch, or[/dim] [cyan]/model reset[/cyan] [dim]to clear override[/dim]"
            )
            if listing.hidden:
                console.print(
                    f"  [dim]Hidden ({len(listing.hidden)}): unusable for chat completions —[/dim]"
                )
                for mid, reason in listing.hidden:
                    console.print(f"    [dim]{mid}[/dim]  [dim]({reason})[/dim]")
    elif args.strip().lower() == "reset":
        ctx.session.preferred_model = ""
        ctx.session_manager.save(ctx.session)
        active = ctx.provider_manager.get_active()
        console.print(
            f"  [bold green]✓[/bold green]  Model reset to provider default  "
            f"[dim cyan]{active.model}[/dim cyan]"
        )
    else:
        active = ctx.provider_manager.get_active()
        new_model = args.strip()
        ctx.session.preferred_model = new_model
        ctx.session_manager.save(ctx.session)
        console.print(
            f"  [bold green]✓[/bold green]  Model override set to [cyan]{new_model}[/cyan]  "
            f"[dim](provider default:[/dim] [dim cyan]{active.model}[/dim cyan][dim])[/dim]"
        )
        _warm_model(
            _warmup_url(ctx),
            role="executor",
            model=new_model,
            provider_profile=active.name,
            label=new_model,
        )
    console.print()
    return True


_ROUTER_DEFAULT_LABEL = "llama3.2:3b (tier-1 default)"


def _cmd_router(ctx: REPLContext, args: str) -> bool:
    """Show or set the model used by the LLM intent router.

    The router is the small fast model that classifies each user message into
    an intent (coding / search / system / etc.) before the executor LLM runs.
    Defaults to the Tier-1 model in ``config/llm_tiers.yaml`` (llama3.2:3b).
    """
    from .render import console

    if not args:
        current = ctx.session.router_model or _ROUTER_DEFAULT_LABEL
        override_note = "  [dim](session override)[/dim]" if ctx.session.router_model else ""
        console.print(
            f"  [dim]router[/dim]  [cyan]{current}[/cyan]{override_note}\n"
            "  [dim]Set with[/dim] [cyan]/router <model>[/cyan]  "
            "[dim]reset with[/dim] [cyan]/router reset[/cyan]\n"
            "  [dim]Provider is auto-detected from the model name "
            "(claude/gpt/gemini/ollama).[/dim]"
        )
    elif args.strip().lower() == "reset":
        ctx.session.router_model = ""
        ctx.session_manager.save(ctx.session)
        console.print(
            "  [bold green]✓[/bold green]  Router reset to "
            f"[dim cyan]{_ROUTER_DEFAULT_LABEL}[/dim cyan]"
        )
    else:
        new_model = args.strip()
        ctx.session.router_model = new_model
        ctx.session_manager.save(ctx.session)
        console.print(
            f"  [bold green]✓[/bold green]  Router model set to " f"[cyan]{new_model}[/cyan]"
        )
        _warm_model(_warmup_url(ctx), role="router", model=new_model, label=new_model)
    console.print()
    return True


def _warn_readiness(console: object, profile: object) -> None:
    """Print setup warnings if the selected provider has unmet requirements."""
    from .render import console as _console

    issues = profile.readiness_issues()  # type: ignore[attr-defined]
    if issues:
        _console.print(
            "  [bold yellow]⚠[/bold yellow]  "
            "[yellow]This provider needs setup before it will work:[/yellow]"
        )
        for issue in issues:
            _console.print(f"    [dim]•[/dim] [yellow]{issue}[/yellow]")
        _console.print("  [dim]Restart the IRIS API after making changes.[/dim]")


_LOCAL_PROVIDER_TYPES = frozenset({"lmstudio", "ollama"})


def _autopick_local_model_if_default_missing(ctx: REPLContext, profile: ProviderProfile) -> None:
    """For local providers, override the session model when the profile default isn't loaded.

    `/provider lmstudio` would otherwise hand the profile's default model id to LM Studio
    even if the user has only loaded a different one (e.g. ``google/gemma-4-e4b``).
    We query the provider's ``/v1/models`` and:

    * if the default model is in the listing → no change,
    * if some other models are loaded → set ``session.preferred_model`` to the first,
    * if the endpoint is unreachable / returns nothing → leave the default in place
      (the user can diagnose with ``/model list``).
    """
    if profile.provider_type not in _LOCAL_PROVIDER_TYPES:
        return
    from .render import console

    listing = profile.fetch_models()
    if not listing.usable:
        return
    if profile.model in listing.usable:
        return
    new_model = listing.usable[0]
    ctx.session.preferred_model = new_model
    ctx.session_manager.save(ctx.session)
    console.print(
        f"  [dim]Default model[/dim] [yellow]{profile.model}[/yellow] "
        f"[dim]not loaded in[/dim] [cyan]{profile.display_name}[/cyan][dim]; "
        f"using[/dim] [cyan]{new_model}[/cyan]"
    )


def _cmd_provider(ctx: REPLContext, args: str) -> bool:
    from .interactive import pick_from_list
    from .render import console, print_error

    mgr = ctx.provider_manager
    profiles = mgr.list_profiles()
    current = mgr.active_name()

    if args:
        name = args.strip().lower()
        if name not in {p.name for p in profiles}:
            print_error(
                f"Unknown provider: {name!r}  " "(try /provider without args to open the picker)"
            )
            return True
        mgr.set_active(name)
        active = mgr.get_active()
        console.print(
            f"  [bold green]✓[/bold green]  Provider switched to "
            f"[cyan]{active.display_name}[/cyan]  [dim]{active.model}[/dim]"
        )
        _warn_readiness(console, active)
        if active.is_available():
            _autopick_local_model_if_default_missing(ctx, active)
            effective_model = ctx.session.preferred_model or active.model
            _warm_model(
                ctx.api_url,
                role="executor",
                provider_profile=active.name,
                model=effective_model if effective_model != active.model else "",
                label=f"{active.display_name} · {effective_model}",
            )
        console.print()
        return True

    options = [
        (
            p.name,
            f"{'★ ' if p.name == current else '  '}"
            f"{p.display_name}  [{p.model}]"
            f"  {'[green]●[/green]' if p.is_available() else '[red]○[/red]'}",
        )
        for p in profiles
    ]
    selected = pick_from_list("Select Provider", options, current=current)
    if selected and selected != current:
        mgr.set_active(selected)
        active = mgr.get_active()
        console.print(
            f"\n  [bold green]✓[/bold green]  Provider switched to "
            f"[cyan]{active.display_name}[/cyan]  [dim]{active.model}[/dim]"
        )
        _warn_readiness(console, active)
        if active.is_available():
            _autopick_local_model_if_default_missing(ctx, active)
            effective_model = ctx.session.preferred_model or active.model
            _warm_model(
                ctx.api_url,
                role="executor",
                provider_profile=active.name,
                model=effective_model if effective_model != active.model else "",
                label=f"{active.display_name} · {effective_model}",
            )
    elif selected == current:
        console.print(f"\n  [dim]Provider unchanged:[/dim] [cyan]{current}[/cyan]")
    console.print()
    return True


def _repo_relative_display(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _list_skill_entries(repo_root: Path) -> tuple[SkillListEntry, ...]:
    from iris_harness.tools.skills.registry import SkillRegistry

    registry = SkillRegistry(repo_root=repo_root)
    packages = registry.discover()
    entries: list[SkillListEntry] = []
    for package in packages:
        manifest = package.manifest
        status = (
            "loadable"
            if package.is_loadable
            else "blocked: " + ", ".join(package.missing_prerequisites or ("no tools",))
        )
        entries.append(
            SkillListEntry(
                name=manifest.name,
                source="config/skills",
                status=status,
                tools=", ".join(tool.name for tool in manifest.tools) or "-",
                agents=", ".join(manifest.requires.agents) or "all",
                location=_repo_relative_display(repo_root, package.skill_dir),
            )
        )
    for skill_dir, reason in registry.load_failures.items():
        entries.append(
            SkillListEntry(
                name=skill_dir.name,
                source="config/skills",
                status=f"failed: {reason}",
                tools="-",
                agents="-",
                location=_repo_relative_display(repo_root, skill_dir),
            )
        )
    return tuple(sorted(entries, key=lambda entry: (entry.source, entry.name)))


def _cmd_skills(ctx: REPLContext, args: str) -> bool:
    from pathlib import Path

    from rich import box
    from rich.table import Table

    from .render import console

    subcommand = args.split(maxsplit=1)[0].lower() if args.strip() else "list"
    if subcommand not in {"list", "ls"}:
        console.print("  [yellow]Usage:[/yellow] /skills [list]")
        console.print()
        return True

    repo_root = Path(ctx.session.cwd)
    entries = list(_list_skill_entries(repo_root))

    skill_dirs: list[Path] = [
        Path.home() / ".iris" / "skills",
        Path.home() / ".iris" / "extensions",
        repo_root / ".iris" / "extensions",
    ]
    for d in skill_dirs:
        if d.exists() and d.is_dir():
            for item in sorted(d.iterdir()):
                if item.is_dir() and not item.name.startswith("."):
                    entries.append(
                        SkillListEntry(
                            name=item.name,
                            source="legacy",
                            status="installed",
                            tools="-",
                            agents="-",
                            location=_repo_relative_display(repo_root, item),
                        )
                    )

    table = Table(
        box=box.ROUNDED,
        border_style="dim",
        show_header=True,
        header_style="bold cyan",
        pad_edge=True,
    )
    table.add_column("Skill", style="cyan")
    table.add_column("Source")
    table.add_column("Status")
    table.add_column("Tools")
    table.add_column("Agents")
    table.add_column("Location", style="dim")

    if not entries:
        console.print("  [dim]No skills installed.[/dim]")
        console.print("  [dim]Runtime skill directory:[/dim]")
        console.print(f"    [dim]{repo_root / 'config' / 'skills'}[/dim]")
    else:
        for entry in entries:
            table.add_row(
                entry.name,
                entry.source,
                entry.status,
                entry.tools,
                entry.agents,
                entry.location,
            )
        console.print(table)
    console.print()
    return True


_REMINDERS_USAGE = "Usage: /reminders [list | due | missed | all | tick]"

#: The one reminder store's read API (loop-proof D14; the calendar plugin serves it).
_REMINDERS_PATH = "/api/v1/reminders"


def _print_reminder_rows(rows: list[dict[str, object]], *, title: str) -> None:
    from rich import box
    from rich.table import Table

    from .render import console

    if not rows:
        console.print(f"  [dim]No {title.lower()}.[/dim]")
        console.print()
        return

    table = Table(box=box.ROUNDED, border_style="dim", header_style="bold cyan")
    table.add_column("ID", style="cyan")
    table.add_column("Reminder")
    table.add_column("When", style="dim")
    table.add_column("Status")
    for item in rows:
        when = str(item.get("remind_at_local", "") or item.get("remind_at", ""))
        repeats = str(item.get("recurrence_label", "") or "")
        table.add_row(
            str(item.get("id", "")),
            str(item.get("text", "")),
            f"{when} ({repeats})" if repeats else when,
            str(item.get("status", "")),
        )
    console.print(f"  [bold]{title}[/bold]")
    console.print(table)
    console.print()


def _cmd_reminders(ctx: REPLContext, args: str) -> bool:
    """Read the one reminder store through the IRIS API (``GET /api/v1/reminders``).

    Reminders are created in chat ("remind me …"); Done and Snooze arrive with PR 3b.
    ``tick`` fires the delivery heartbeat now.
    """

    from .render import console, print_error

    try:
        parts = shlex.split(args)
    except ValueError as exc:
        print_error(f"Could not parse reminder command: {exc}")
        return True

    sub = parts[0].lower() if parts else "list"
    queries = {
        "list": ("?status=open", "Reminders"),
        "ls": ("?status=open", "Reminders"),
        "": ("?status=open", "Reminders"),
        "due": ("?status=open&due=true", "Due reminders"),
        "missed": ("?status=failed", "Missed reminders"),
        "all": ("?status=all", "All reminders"),
    }

    try:
        if sub in queries:
            query, title = queries[sub]
            data = _api_json(ctx.api_url, method="GET", path=f"{_REMINDERS_PATH}{query}")
            rows = data.get("reminders", [])
            _print_reminder_rows(rows if isinstance(rows, list) else [], title=title)
            zone = data.get("timezone")
            if zone:
                console.print(f"  [dim]times in {zone}[/dim]\n")
            return True

        if sub == "tick":
            data = _api_json(
                ctx.api_url,
                method="POST",
                path="/heartbeat/trigger/notification_reminder_tick",
            )
            status = data.get("status", "unknown")
            color = "green" if status == "success" else "yellow"
            console.print(f"  [bold {color}]{status}[/bold {color}] {data.get('output', '')}")
            if data.get("error"):
                console.print(f"  [red]{data['error']}[/red]")
            console.print()
            return True
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print_error(f"Reminder API failed (HTTP {exc.code}): {detail}")
        return True
    except OSError as exc:
        print_error(f"Cannot reach IRIS API for reminders: {exc}")
        return True
    except Exception as exc:  # noqa: BLE001
        print_error(f"Reminder command failed: {exc}")
        return True

    print_error(_REMINDERS_USAGE)
    return True


_ROUTINES_USAGE = (
    "Usage: /routines [list | due | tick | run <id> | preview <id> | "
    "add <schedule> <template> <title> | set-schedule <id> <schedule> | "
    "approve <id> | schedule <id> | pause <id> | retire <id> | delete <id> | clear]"
)


def _print_routine_rows(rows: list[dict[str, object]], *, title: str) -> None:
    from rich import box
    from rich.table import Table

    from .render import console

    if not rows:
        console.print(f"  [dim]No {title.lower()}.[/dim]")
        console.print()
        return

    table = Table(box=box.ROUNDED, border_style="dim", header_style="bold cyan", expand=False)
    table.add_column("ID", style="cyan", no_wrap=True, overflow="ignore")
    table.add_column("Title", overflow="fold", max_width=30)
    table.add_column("Schedule", style="dim", no_wrap=True, overflow="ignore")
    table.add_column("Template", overflow="fold", max_width=22)
    table.add_column("Status", no_wrap=True)
    table.add_column("Runs", justify="right")
    for item in rows:
        runs = f"{item.get('success_count', 0)}/{item.get('run_count', 0)}"
        table.add_row(
            str(item.get("id", "")),
            str(item.get("title", "")),
            str(item.get("schedule", "")),
            str(item.get("template", "")),
            str(item.get("approval_status", "")),
            runs,
        )
    console.print(f"  [bold]{title}[/bold]")
    console.print(table)
    console.print("  [dim]Full routine IDs:[/dim]")
    for item in rows:
        console.print(f"  [cyan]{item.get('id', '')}[/cyan]")
    console.print()


def _cmd_routines(ctx: REPLContext, args: str) -> bool:
    """Manage approved routines through the local IRIS API."""

    from .render import console, print_error

    try:
        parts = shlex.split(args)
    except ValueError as exc:
        print_error(f"Could not parse routine command: {exc}")
        return True

    sub = parts[0].lower() if parts else "list"
    rest = parts[1:]

    try:
        if sub in {"list", "ls", ""}:
            data = _api_json(ctx.api_url, method="GET", path="/routines")
            rows = data.get("routines", [])
            _print_routine_rows(rows if isinstance(rows, list) else [], title="Routines")
            store_path = data.get("routine_store")
            if store_path:
                console.print(f"  [dim]store:[/dim] [cyan]{store_path}[/cyan]\n")
            return True

        if sub == "due":
            data = _api_json(ctx.api_url, method="GET", path="/routines/due")
            rows = data.get("routines", [])
            _print_routine_rows(rows if isinstance(rows, list) else [], title="Due routines")
            return True

        if sub == "add":
            if len(rest) < 3:
                print_error(_ROUTINES_USAGE)
                return True
            payload: dict[str, object] = {
                "schedule": rest[0],
                "template": rest[1],
                "title": " ".join(rest[2:]),
            }
            data = _api_json(ctx.api_url, method="POST", path="/routines", payload=payload)
            raw_routine = data.get("routine")
            routine = (
                {str(key): value for key, value in raw_routine.items()}
                if isinstance(raw_routine, dict)
                else {}
            )
            console.print(
                "  [bold green]✓[/bold green] routine added "
                f"[cyan]{routine.get('id', '')}[/cyan]"
            )
            console.print(f"  [dim]schedule:[/dim] {routine.get('schedule', '')}")
            console.print(f"  [dim]template:[/dim] {routine.get('template', '')}")
            console.print(f"  [dim]status:[/dim] {routine.get('approval_status', '')}\n")
            return True

        if sub in {"set-schedule", "reschedule"}:
            if len(rest) != 2:
                print_error(_ROUTINES_USAGE)
                return True
            routine_id, new_schedule = rest[0], rest[1]
            data = _api_json(
                ctx.api_url,
                method="PATCH",
                path=f"/routines/{routine_id}",
                payload={"schedule": new_schedule},
            )
            raw_routine = data.get("routine")
            routine = (
                {str(key): value for key, value in raw_routine.items()}
                if isinstance(raw_routine, dict)
                else {}
            )
            console.print(
                f"  [bold green]✓[/bold green] routine "
                f"[cyan]{routine.get('id', routine_id)}[/cyan] schedule updated"
            )
            console.print(f"  [dim]schedule:[/dim] {routine.get('schedule', new_schedule)}\n")
            return True

        status_commands = {
            "approve": "approved",
            "schedule": "scheduled",
            "pause": "paused",
            "retire": "retired",
            "draft": "draft",
        }
        if sub in status_commands:
            if len(rest) != 1:
                print_error(_ROUTINES_USAGE)
                return True
            data = _api_json(
                ctx.api_url,
                method="PATCH",
                path=f"/routines/{rest[0]}",
                payload={"approval_status": status_commands[sub]},
            )
            routine = data.get("routine", {})
            routine_id = routine.get("id", rest[0]) if isinstance(routine, dict) else rest[0]
            console.print(
                f"  [bold green]✓[/bold green] routine [cyan]{routine_id}[/cyan] "
                f"is now [bold]{status_commands[sub]}[/bold]\n"
            )
            return True

        if sub in {"delete", "rm"}:
            if len(rest) != 1:
                print_error(_ROUTINES_USAGE)
                return True
            data = _api_json(ctx.api_url, method="DELETE", path=f"/routines/{rest[0]}")
            routine = data.get("deleted_routine", {})
            routine_id = routine.get("id", rest[0]) if isinstance(routine, dict) else rest[0]
            console.print(
                f"  [bold green]✓[/bold green] routine deleted [cyan]{routine_id}[/cyan]\n"
            )
            return True

        if sub in {"clear", "delete-all", "rm-all"}:
            if rest:
                print_error(_ROUTINES_USAGE)
                return True
            data = _api_json(ctx.api_url, method="DELETE", path="/routines")
            deleted_count = data.get("deleted_count", data.get("count", 0))
            console.print(
                f"  [bold green]✓[/bold green] deleted [cyan]{deleted_count}[/cyan] routines"
            )
            deleted_routines = data.get("deleted_routines", [])
            if isinstance(deleted_routines, list) and deleted_routines:
                console.print("  [dim]Deleted routine IDs:[/dim]")
                for item in deleted_routines:
                    if isinstance(item, dict):
                        console.print(f"  [cyan]{item.get('id', '')}[/cyan]")
            console.print()
            return True

        if sub == "tick":
            data = _api_json(ctx.api_url, method="POST", path="/routines/tick")
            status = data.get("status", "unknown")
            color = "green" if status == "success" else "yellow"
            console.print(f"  [bold {color}]{status}[/bold {color}] {data.get('output', '')}")
            if data.get("error"):
                console.print(f"  [red]{data['error']}[/red]")
            console.print()
            return True

        if sub == "run":
            if len(rest) != 1:
                print_error(_ROUTINES_USAGE)
                return True
            data = _api_json(ctx.api_url, method="POST", path=f"/routines/{rest[0]}/run")
            status = str(data.get("status", "unknown"))
            color = "green" if status == "success" else "red"
            console.print(
                f"  [bold {color}]{status}[/bold {color}] "
                f"[cyan]{data.get('title', rest[0])}[/cyan] "
                f"-> delivered ({data.get('template', '')})"
            )
            if data.get("detail"):
                console.print(f"  [dim]{data['detail']}[/dim]")
            console.print()
            return True

        if sub == "preview":
            if len(rest) != 1:
                print_error(_ROUTINES_USAGE)
                return True
            data = _api_json(ctx.api_url, method="POST", path=f"/routines/{rest[0]}/preview")
            console.print(f"  [bold]Preview:[/bold] [cyan]{data.get('title', rest[0])}[/cyan]\n")
            console.print(str(data.get("body", "")))
            console.print()
            return True
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print_error(f"Routine API failed (HTTP {exc.code}): {detail}")
        return True
    except OSError as exc:
        print_error(f"Cannot reach IRIS API for routines: {exc}")
        return True
    except Exception as exc:  # noqa: BLE001
        print_error(f"Routine command failed: {exc}")
        return True

    print_error(_ROUTINES_USAGE)
    return True


_HEARTBEATS_USAGE = (
    "Usage: /heartbeats [list | runs [name] [N] | trigger <name> | "
    "set <name> (every 10m | daily 08:30 | cron <5 fields> | on | off) | reset <name>]"
)


def heartbeat_edit_payload(words: list[str]) -> dict[str, object]:
    """The PATCH body for ``/heartbeats set <name> <words>`` (ADR-0120).

    ``on`` / ``off`` flip the heartbeat; anything else is a schedule in words. Raises
    ``ValueError`` with the accepted forms, before anything reaches the API.
    """
    from iris_harness.services.heartbeat.schedule_text import (
        parse_schedule_words,
    )

    if len(words) == 1 and words[0].lower() in {"on", "off"}:
        return {"enabled": words[0].lower() == "on"}
    return {"schedule": parse_schedule_words(words)}


def _print_heartbeat_list(rows: list[dict[str, object]]) -> None:
    from rich import box
    from rich.table import Table

    from .render import console

    if not rows:
        console.print("  [dim]No heartbeats registered.[/dim]\n")
        return
    table = Table(box=box.ROUNDED, border_style="dim", header_style="bold cyan")
    table.add_column("Name", style="cyan")
    table.add_column("Schedule")
    table.add_column("Enabled")
    table.add_column("Description", style="dim")
    for item in rows:
        enabled = bool(item.get("enabled", True))
        schedule = str(item.get("schedule_text") or item.get("schedule", ""))
        if item.get("overridden"):
            schedule += " [yellow](changed)[/yellow]"
        table.add_row(
            str(item.get("name", "")),
            schedule,
            "[green]yes[/green]" if enabled else "[red]no[/red]",
            str(item.get("description", "")),
        )
    console.print("  [bold]Heartbeats[/bold]")
    console.print(table)
    console.print()


def _print_heartbeat_runs(rows: list[dict[str, object]]) -> None:
    from rich import box
    from rich.table import Table

    from .render import console

    if not rows:
        console.print("  [dim]No heartbeat runs recorded yet.[/dim]\n")
        return
    table = Table(box=box.ROUNDED, border_style="dim", header_style="bold cyan")
    table.add_column("Finished", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Status")
    table.add_column("Output")
    status_colors = {
        "success": "green",
        "failed": "red",
        "skipped": "yellow",
        "pending": "dim",
        "running": "cyan",
    }
    for item in rows:
        status = str(item.get("status", ""))
        color = status_colors.get(status, "white")
        finished = str(item.get("finished_at") or item.get("started_at") or "")
        detail = str(item.get("output") or item.get("error") or "")
        table.add_row(
            finished,
            str(item.get("name", "")),
            f"[{color}]{status}[/{color}]",
            detail,
        )
    console.print("  [bold]Recent heartbeat runs[/bold]")
    console.print(table)
    console.print()


def _cmd_heartbeats(ctx: REPLContext, args: str) -> bool:
    """Inspect heartbeat definitions, recent runs, and fire one manually."""

    from .render import console, print_error

    parts = args.split()
    sub = parts[0].lower() if parts else "list"
    rest = parts[1:]

    try:
        if sub in {"list", "ls", ""}:
            data = _api_json(ctx.api_url, method="GET", path="/heartbeat")
            rows = data.get("heartbeats", [])
            _print_heartbeat_list(rows if isinstance(rows, list) else [])
            return True

        if sub == "runs":
            name: str | None = None
            limit = 20
            for token in rest:
                if token.isdigit():
                    limit = int(token)
                else:
                    name = token
            query = f"?limit={limit}" + (f"&name={name}" if name else "")
            data = _api_json(ctx.api_url, method="GET", path=f"/heartbeat/runs{query}")
            runs = data.get("runs", [])
            _print_heartbeat_runs(runs if isinstance(runs, list) else [])
            total = data.get("total", 0)
            shown = data.get("count", 0)
            if isinstance(total, int) and isinstance(shown, int) and total > shown:
                console.print(f"  [dim]showing {shown} of {total} runs[/dim]\n")
            return True

        if sub == "trigger":
            if len(rest) != 1:
                print_error(_HEARTBEATS_USAGE)
                return True
            data = _api_json(
                ctx.api_url,
                method="POST",
                path=f"/heartbeat/trigger/{rest[0]}",
            )
            status = str(data.get("status", "unknown"))
            color = "green" if status == "success" else "yellow" if status == "skipped" else "red"
            console.print(
                f"  [bold {color}]{status}[/bold {color}] "
                f"{data.get('output', '') or data.get('error', '')}"
            )
            console.print()
            return True

        if sub in {"set", "reset"}:
            if not rest or (sub == "set" and len(rest) < 2) or (sub == "reset" and len(rest) != 1):
                print_error(_HEARTBEATS_USAGE)
                return True
            name = rest[0]
            if sub == "set":
                try:
                    payload = heartbeat_edit_payload(rest[1:])
                except ValueError as exc:
                    print_error(str(exc))
                    return True
                data = _api_json(
                    ctx.api_url, method="PATCH", path=f"/heartbeat/{name}", payload=payload
                )
            else:
                data = _api_json(ctx.api_url, method="DELETE", path=f"/heartbeat/{name}/override")
            state = "on" if data.get("enabled") else "off"
            console.print(
                f"  [bold green]{data.get('name', name)}[/bold green] "
                f"{data.get('schedule_text', data.get('schedule', ''))}, {state}"
                + (f", next run {data['next_run_at']}" if data.get("next_run_at") else "")
            )
            console.print()
            return True
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print_error(f"Heartbeat API failed (HTTP {exc.code}): {detail}")
        return True
    except OSError as exc:
        print_error(f"Cannot reach IRIS API for /heartbeats: {exc}")
        return True
    except Exception as exc:  # noqa: BLE001
        print_error(f"/heartbeats command failed: {exc}")
        return True

    print_error(_HEARTBEATS_USAGE)
    return True


def _complete_heartbeats(_ctx: CompleterContext, _partial: str) -> list[tuple[str, str]]:
    return [
        ("list", "Show registered heartbeats and their schedules"),
        ("runs", "Recent run history (filter: /heartbeats runs <name> <N>)"),
        ("set", "Change one: set <name> every 10m | daily 08:30 | cron <5 fields> | on | off"),
        ("reset", "Back to the shipped schedule: reset <name>"),
        ("trigger notification_reminder_tick", "Fire the reminder delivery tick now"),
        ("trigger routine_tick", "Fire routine_tick now"),
        ("trigger morning_briefing", "Fire morning_briefing now"),
        ("trigger pressure_tick", "Fire pressure_tick now"),
        ("trigger learning_tick", "Fire learning_tick now"),
        ("trigger wiki_lint", "Fire wiki_lint now"),
    ]


_LLM_USAGE = "Usage: /llm [status | pressure | mode <active|idle|thermal> | unpin]"
_LLM_VALID_MODES = {"active", "idle", "thermal"}


def _print_llm_state(data: dict[str, Any]) -> None:
    from .render import console

    mode = data.get("mode", "?")
    auto_mode = data.get("auto_mode", mode)
    pin = data.get("pin")
    adaptive = data.get("adaptive", False)
    pin_label = f"[bold]{pin}[/bold]" if pin else "[dim]none[/dim]"
    adaptive_label = "[green]on[/green]" if adaptive else "[dim]off[/dim]"
    console.print(f"  [bold]mode:[/bold] [cyan]{mode}[/cyan] (auto={auto_mode})")
    console.print(f"  [bold]pin:[/bold] {pin_label}")
    console.print(f"  [bold]adaptive:[/bold] {adaptive_label}")
    snap = data.get("snapshot")
    if isinstance(snap, dict):
        console.print(
            f"  [dim]ram_free={snap.get('ram_free_gb', 0):.1f}GB "
            f"cpu={snap.get('cpu_percent', 0):.0f}% "
            f"speed_limit={snap.get('cpu_speed_limit', 100)} "
            f"throttled={snap.get('thermal_throttled', False)}[/dim]"
        )
    else:
        console.print("  [dim]no pressure snapshot yet (heartbeat hasn't fired)[/dim]")
    console.print()


def _cmd_llm(ctx: REPLContext, args: str) -> bool:
    """Inspect and pin the LLM resource governor mode."""

    from .render import print_error

    parts = args.split()
    sub = parts[0].lower() if parts else "status"
    rest = parts[1:]

    try:
        if sub in {"status", ""}:
            data = _api_json(ctx.api_url, method="GET", path="/llm/mode")
            _print_llm_state(data)
            return True

        if sub == "pressure":
            data = _api_json(ctx.api_url, method="GET", path="/llm/pressure")
            _print_llm_state(data)
            return True

        if sub == "mode":
            if len(rest) != 1 or rest[0].lower() not in _LLM_VALID_MODES:
                print_error(_LLM_USAGE)
                return True
            data = _api_json(
                ctx.api_url,
                method="POST",
                path="/llm/mode/pin",
                payload={"mode": rest[0].lower()},
            )
            _print_llm_state(data)
            return True

        if sub == "unpin":
            data = _api_json(ctx.api_url, method="DELETE", path="/llm/mode/pin")
            _print_llm_state(data)
            return True
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print_error(f"LLM API failed (HTTP {exc.code}): {detail}")
        return True
    except OSError as exc:
        print_error(f"Cannot reach IRIS API for /llm: {exc}")
        return True
    except Exception as exc:  # noqa: BLE001
        print_error(f"/llm command failed: {exc}")
        return True

    print_error(_LLM_USAGE)
    return True


def _complete_llm(_ctx: CompleterContext, _partial: str) -> list[tuple[str, str]]:
    return [
        ("status", "Show current mode + last pressure snapshot"),
        ("pressure", "Sample pressure now and show snapshot"),
        ("mode active", "Pin to active (no constraints)"),
        ("mode idle", "Pin to idle (prefer small tier)"),
        ("mode thermal", "Pin to thermal (small tier, evict large)"),
        ("unpin", "Release pin and resume automatic transitions"),
    ]


def _complete_queue(ctx: CompleterContext, _partial: str) -> list[tuple[str, str]]:
    """Argument completer for /queue subcommands and proposal slugs."""
    from pathlib import Path

    from . import queue as queue_module

    items: list[tuple[str, str]] = [
        ("list", "Show ready / proposed drafts"),
        ("show", "Show one proposal in detail"),
        ("promote", "Hand a proposal to the coding agent"),
        ("dismiss", "Mark a proposal as dismissed"),
        ("stats", "Counts by status"),
    ]
    try:
        repo_root = Path(ctx.session.cwd)
        for summary in queue_module.list_proposals(repo_root, status_filter=None):
            items.append((summary.slug, f"[{summary.status}] {summary.skill_name}"))
    except Exception:  # noqa: BLE001, S110 — completer must never raise into the REPL
        pass
    return items


def _cmd_queue(ctx: REPLContext, args: str) -> bool:
    from pathlib import Path

    from rich import box
    from rich.table import Table

    from . import queue as queue_module
    from .render import console

    repo_root = Path(ctx.session.cwd)
    parts = args.split(maxsplit=1)
    sub = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("list", "ls", ""):
        status_filter: tuple[str, ...] | None = queue_module.DEFAULT_VISIBLE_STATUSES
        if rest in ("--all", "all"):
            status_filter = None
        summaries = queue_module.list_proposals(repo_root, status_filter=status_filter)
        if not summaries:
            console.print("  [dim]No proposals in the queue.[/dim]")
            console.print()
            return True
        table = Table(box=box.ROUNDED, border_style="dim", header_style="bold cyan")
        table.add_column("Slug", style="cyan")
        table.add_column("Status")
        table.add_column("Origin", style="dim")
        table.add_column("Runs", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Last run", style="dim")
        for s in summaries:
            last = s.last_run_at.isoformat(timespec="seconds") if s.last_run_at else "—"
            table.add_row(
                s.slug,
                s.status,
                s.source_kind,
                str(s.run_count),
                str(s.promotion_threshold),
                last,
            )
        console.print(table)
        console.print()
        return True

    if sub == "stats":
        counts = queue_module.queue_stats(repo_root)
        if counts.get("total", 0) == 0:
            console.print("  [dim]No proposals in the queue.[/dim]")
        else:
            for status, count in sorted(counts.items()):
                console.print(f"  [cyan]{status}[/cyan]: {count}")
        console.print()
        return True

    if sub == "show":
        if not rest:
            console.print("  [yellow]Usage:[/yellow] /queue show <slug>")
            console.print()
            return True
        try:
            detail = queue_module.show_proposal(repo_root, rest)
        except FileNotFoundError:
            console.print(f"  [red]No proposal found for slug:[/red] {rest}")
            console.print()
            return True
        p = detail.proposal
        console.print(f"  [bold cyan]{p.skill_name}[/bold cyan]  ([cyan]{p.skill_slug}[/cyan])")
        console.print(
            f"  status: [bold]{p.status}[/bold]   origin: {p.source_kind}   "
            f"runs: {p.run_count}/{p.promotion_threshold}"
        )
        console.print(f"  proposal_dir: [dim]{p.proposal_dir}[/dim]")
        if detail.narrative:
            console.print()
            console.print("  [bold]Narrative[/bold]")
            for line in detail.narrative.rstrip().splitlines():
                console.print(f"    {line}")
        if detail.recent_runs:
            console.print()
            console.print("  [bold]Recent runs[/bold]")
            for entry in detail.recent_runs:
                console.print(
                    f"    [dim]{entry.get('ts', '?')}[/dim]  "
                    f"exit={entry.get('exit_code', '?')}  "
                    f"duration_ms={entry.get('duration_ms', '—')}"
                )
        console.print()
        return True

    if sub == "dismiss":
        if not rest:
            console.print("  [yellow]Usage:[/yellow] /queue dismiss <slug>")
            console.print()
            return True
        try:
            updated = queue_module.dismiss_proposal(repo_root, rest)
        except FileNotFoundError:
            console.print(f"  [red]No proposal found for slug:[/red] {rest}")
            console.print()
            return True
        console.print(f"  [bold green]✓[/bold green] dismissed [cyan]{updated.skill_slug}[/cyan]")
        console.print()
        return True

    if sub == "promote":
        tokens = rest.split()
        use_preflight = "--preflight" in tokens
        slug = next((t for t in tokens if not t.startswith("-")), "")
        if not slug:
            console.print("  [yellow]Usage:[/yellow] /queue promote <slug> [--preflight]")
            console.print()
            return True
        preflight = None
        if use_preflight:
            from iris_harness.services.learning.preflight import (
                build_proposal_preflight,
            )

            console.print("  [dim]Running replay-eval pre-flight (needs a model)…[/dim]")
            preflight = build_proposal_preflight(repo_root)
        try:
            updated = queue_module.promote_proposal(repo_root, slug, preflight=preflight)
        except FileNotFoundError:
            console.print(f"  [red]No proposal found for slug:[/red] {slug}")
            console.print()
            return True
        except ValueError as exc:
            console.print(f"  [red]{exc}[/red]")
            console.print()
            return True
        except Exception as exc:  # noqa: BLE001 — PreflightError + any preflight glue failure
            from iris_harness.services.learning.preflight import PreflightError

            if isinstance(exc, PreflightError):
                console.print(
                    f"  [red]Pre-flight failed — not promoted:[/red] {exc.verdict.reason}"
                )
            else:
                console.print(f"  [red]Pre-flight error — not promoted:[/red] {exc}")
            console.print()
            return True

        try:
            # Staging hands the proposal to the coding agent, which is not part of the
            # harness release (OSS plan decision 2). Without it the flip still happens;
            # only the hand-off is skipped, and the user is told so.
            # absent from the public tree: mypy must pass there as well as here
            from iris_code.promotion import (  # type: ignore[import-not-found,unused-ignore]
                stage_promotion_task,
            )
        except ImportError:
            console.print(
                f"  [bold green]✓[/bold green] [cyan]{updated.skill_slug}[/cyan] "
                f"flipped to [bold]promoting[/bold]; the coding agent is not installed, "
                f"so no implementation task was staged."
            )
            console.print()
            return True

        try:
            task_id = stage_promotion_task(repo_root, repo_root / updated.proposal_dir, updated)
        except Exception as exc:  # noqa: BLE001 — surface staging failures to user
            console.print(
                f"  [bold green]✓[/bold green] [cyan]{updated.skill_slug}[/cyan] "
                f"flipped to [bold]promoting[/bold], but staging failed:"
            )
            console.print(f"  [red]{exc}[/red]")
            console.print()
            return True

        console.print(
            f"  [bold green]✓[/bold green] [cyan]{updated.skill_slug}[/cyan] "
            f"flipped to [bold]promoting[/bold]."
        )
        console.print(f"  Coding task staged: [cyan]{task_id}[/cyan]")
        console.print(
            f"  [dim]Next:[/dim] run [cyan]iris-code resume --task-id {task_id}[/cyan] "
            f"in another terminal to land the change as a PR."
        )
        console.print()
        return True

    console.print(f"  [yellow]Unknown subcommand:[/yellow] {sub}")
    console.print("  Try: /queue [list|show|promote|dismiss|stats] <slug>")
    console.print()
    return True


def _cmd_compact(ctx: REPLContext, _args: str) -> bool:
    from .interactive import confirm
    from .render import console

    ok = confirm("Compact context? This resets the conversation window (new session).")
    if ok:
        new_session = ctx.session_manager.create(ctx.session.cwd)
        ctx.session = new_session
        console.print(
            f"\n  [bold green]✓[/bold green]  Context compacted — "
            f"new session [cyan]{new_session.id}[/cyan] started."
        )
    console.print()
    return True


def _cmd_reset(ctx: REPLContext, _args: str) -> bool:
    from .interactive import confirm
    from .render import console

    ok = confirm("Start a fresh session? Current context will be lost.")
    if ok:
        new_session = ctx.session_manager.create(ctx.session.cwd)
        ctx.session = new_session
        console.print(
            f"\n  [bold green]✓[/bold green]  New session "
            f"[cyan]{new_session.id}[/cyan] started."
        )
    console.print()
    return True


def _cmd_export(ctx: REPLContext, args: str) -> bool:
    from datetime import UTC, datetime
    from pathlib import Path

    from .render import console, print_error

    active = ctx.provider_manager.get_active()
    now = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    default_name = f"iris_session_{ctx.session.id[:8]}_{now}.md"
    out_path = Path(args.strip()) if args.strip() else Path(ctx.session.cwd) / default_name

    lines = [
        "# IRIS Session Export\n\n",
        f"- **Session ID:** `{ctx.session.id}`\n",
        f"- **Started:** {ctx.session.created_at.isoformat()}\n",
        f"- **Messages:** {ctx.session.message_count}\n",
        f"- **Directory:** `{ctx.session.cwd}`\n",
        f"- **Provider:** {active.display_name} (`{active.model}`)\n",
        f"- **API:** {ctx.api_url}\n",
    ]
    try:
        out_path.write_text("".join(lines), encoding="utf-8")
        console.print(f"  [bold green]✓[/bold green]  Exported to [cyan]{out_path}[/cyan]")
    except OSError as exc:
        print_error(f"Export failed: {exc}")
    console.print()
    return True


_ACTIVE_USAGE = "Usage: /active [list | add <text> | done <N>]"


def _cmd_active(_ctx: REPLContext, args: str) -> bool:
    """Manage in-flight items in ``~/.iris/memory/active.md``."""
    from iris_harness.memory.identity import (
        active_md_path,
        add_active_item,
        list_active_items,
        mark_active_done,
    )

    from .render import console, print_error

    parts = args.split(maxsplit=1)
    sub = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub in {"list", "ls", ""}:
        items = list_active_items()
        if not items:
            console.print(
                "  [dim]No active items.  Add one with[/dim]  " "[cyan]/active add <text>[/cyan]"
            )
            console.print(f"  [dim]File:[/dim] {active_md_path()}")
            console.print()
            return True
        console.print("  [bold]Active context[/bold]  " f"[dim]({active_md_path()})[/dim]")
        for item in items:
            tag = "[green]✓[/green]" if item.done else "[yellow]○[/yellow]"
            style = "dim" if item.done else ""
            text = f"[{style}]{item.text}[/{style}]" if style else item.text
            console.print(f"  {item.index:>2}. {tag}  {text}")
        console.print()
        return True

    if sub == "add":
        if not rest:
            print_error(_ACTIVE_USAGE)
            return True
        try:
            item = add_active_item(rest)
        except (OSError, ValueError) as exc:
            print_error(f"Could not add item: {exc}")
            return True
        console.print(
            f"  [bold green]✓[/bold green]  Added [cyan]#{item.index}[/cyan]  " f"{item.text}"
        )
        console.print()
        return True

    if sub == "done":
        try:
            idx = int(rest)
        except ValueError:
            print_error(_ACTIVE_USAGE)
            return True
        done_item = mark_active_done(idx)
        if done_item is None:
            print_error(f"No in-flight item at #{idx}.")
            return True
        console.print(
            f"  [bold green]✓[/bold green]  Marked [cyan]#{done_item.index}[/cyan] done  "
            f"[dim]{done_item.text}[/dim]"
        )
        console.print()
        return True

    print_error(_ACTIVE_USAGE)
    return True


# ── Argument completers (used by the slash popup menu) ────────────────────────


def _complete_provider(ctx: CompleterContext, _prefix: str) -> list[tuple[str, str]]:
    """Yield configured provider profiles for `/provider <name>`."""
    try:
        return [
            (
                p.name,
                f"{p.display_name}  [{p.model}]"
                + ("  ✓" if p.is_available() else "  ✗ key missing"),
            )
            for p in ctx.provider_manager.list_profiles()
        ]
    except Exception:  # noqa: BLE001
        return []


def _complete_model(_ctx: CompleterContext, _prefix: str) -> list[tuple[str, str]]:
    """Static suggestions for `/model <name>`. Network fetches go through `/model list`."""
    # Curated picker of models we verified locally plus catalog-known cloud aliases.
    return [
        ("qwen3-coder:30b", "verified Ollama · coding/default"),
        ("qwen3.6:27b", "verified Ollama · reasoning/complex"),
        ("qwen3.5:latest", "verified Ollama · reasoning"),
        ("llama3.2:3b", "verified Ollama · fast/router"),
        ("llama3.2:latest", "verified Ollama · fast"),
        ("phi4-mini:latest", "verified Ollama · compact"),
        ("gemma2:9b", "verified Ollama · general"),
        ("~anthropic/claude-sonnet-latest", "OpenRouter catalog · requires valid key"),
        ("~openai/gpt-mini-latest", "OpenRouter catalog · requires valid key"),
        ("qwen/qwen3.6-flash", "OpenRouter catalog · requires valid key"),
        ("list", "Show all models from the active provider (network)"),
        ("reset", "Clear override; use the provider's default"),
    ]


def _complete_active(_ctx: CompleterContext, _prefix: str) -> list[tuple[str, str]]:
    """Static subcommand suggestions for `/active`."""
    return [
        ("list", "Show in-flight items"),
        ("add", "Add a new item: /active add <text>"),
        ("done", "Mark an item done: /active done <N>"),
    ]


def _complete_reminders(_ctx: CompleterContext, _prefix: str) -> list[tuple[str, str]]:
    """Static subcommand suggestions for `/reminders`."""

    return [
        ("list", "Show open reminders"),
        ("due", "Show open reminders whose time has come"),
        ("missed", "Show reminders that could not be delivered"),
        ("all", "Show every reminder, closed and expired included"),
        ("tick", "Fire the reminder delivery tick now"),
    ]


def _complete_routines(_ctx: CompleterContext, _prefix: str) -> list[tuple[str, str]]:
    """Static subcommand suggestions for `/routines`."""

    return [
        ("list", "Show all durable routines"),
        ("due", "Show approved routines due now"),
        ("tick", "Trigger routine_tick heartbeat now"),
        ("add daily:08:00 daily_repo_brief <title>", "Create a draft repo-brief routine"),
        ("set-schedule <id> <schedule>", "Update a routine's schedule (e.g. daily:08:00)"),
        ("approve <id>", "Approve a draft routine for execution"),
        ("schedule <id>", "Mark an approved routine as scheduled"),
        ("pause <id>", "Pause a routine"),
        ("delete <id>", "Delete a routine"),
        ("clear", "Delete all routines"),
    ]


def _complete_router(_ctx: CompleterContext, _prefix: str) -> list[tuple[str, str]]:
    """Curated short list of router-friendly small models for `/router <name>`."""
    return [
        ("llama3.2:3b", "verified Ollama · Tier-1 default"),
        ("llama3.2:latest", "verified Ollama · fast"),
        ("phi4-mini:latest", "verified Ollama · compact"),
        ("gemma2:9b", "verified Ollama · general"),
        ("qwen3-coder:30b", "verified Ollama · stronger but slower"),
        ("reset", "Restore Tier-1 default"),
    ]


# ── Registration ───────────────────────────────────────────────────────────────

register(SlashCommand("/help", _cmd_help, description="Show available slash commands"))
register(
    SlashCommand(
        "/provider",
        _cmd_provider,
        args="[name]",
        description="Switch LLM provider",
        argument_completer=_complete_provider,
    )
)
register(SlashCommand("/skills", _cmd_skills, description="List installed skills"))
register(
    SlashCommand(
        "/queue",
        _cmd_queue,
        args="[list|show|promote|dismiss|stats] <slug>",
        description="Inspect and promote sandbox skill proposals",
        argument_completer=_complete_queue,
    )
)
register(SlashCommand("/compact", _cmd_compact, description="Compact context window"))
register(SlashCommand("/clear", _cmd_clear, description="Clear screen and redraw"))
register(SlashCommand("/sessions", _cmd_sessions, description="List recent sessions"))
register(SlashCommand("/reset", _cmd_reset, description="Start a fresh session"))
register(
    SlashCommand(
        "/model",
        _cmd_model,
        args="[name]",
        description="Show or set response model",
        argument_completer=_complete_model,
    )
)
register(
    SlashCommand(
        "/router",
        _cmd_router,
        args="[name]",
        description="Show or set intent-router model",
        argument_completer=_complete_router,
    )
)
register(
    SlashCommand(
        "/active",
        _cmd_active,
        args="[list|add <text>|done <N>]",
        description="Manage in-flight items in active.md",
        argument_completer=_complete_active,
    )
)
register(
    SlashCommand(
        "/reminders",
        _cmd_reminders,
        args="[list|due|missed|all|tick]",
        description="Show reminders (the one store); create them in chat",
        argument_completer=_complete_reminders,
    )
)


def _cmd_portfolio(ctx: REPLContext, args: str) -> bool:
    """Show the live-valued stock portfolio (holdings x live prices)."""

    from rich import box
    from rich.table import Table

    from .render import console, print_error

    parts = args.split()
    if parts and parts[0].lower() in {"import-holdings", "import"}:
        if len(parts) < 2:
            print_error("Usage: /portfolio import-holdings <path-to-holdings.csv>")
            return True
        path = " ".join(parts[1:])
        try:
            data = _api_json(
                ctx.api_url,
                method="POST",
                path="/portfolio/import-holdings",
                payload={"path": path},
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print_error(f"Import failed (HTTP {exc.code}): {detail}")
            return True
        console.print(
            f"  [bold green]✓[/bold green] imported broker cost basis: "
            f"{data.get('written')} written, {data.get('mapped')}/{data.get('parsed')} mapped"
        )
        unmapped = data.get("unmapped_symbols") or []
        if unmapped:
            console.print(f"  [dim]no ISIN mapping for: {', '.join(unmapped)}[/dim]")
        console.print("  [dim]run /portfolio to see updated P&L[/dim]\n")
        return True

    try:
        data = _api_json(ctx.api_url, method="GET", path="/portfolio")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print_error(f"Portfolio API failed (HTTP {exc.code}): {detail}")
        return True

    positions = data.get("positions", [])
    if not positions:
        console.print("  [dim]No holdings found. Ingest a broker/CAS statement first.[/dim]\n")
        return True

    only_priced = args.strip().lower() in {"live", "priced"}
    rows = [p for p in positions if p.get("priced")] if only_priced else positions

    table = Table(box=box.ROUNDED, border_style="dim", header_style="bold cyan", expand=False)
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Name", overflow="fold", max_width=28)
    table.add_column("Qty", justify="right")
    table.add_column("Live", justify="right")
    table.add_column("Day", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("P&L", justify="right")
    for p in rows:
        live = p.get("live_price")
        day = p.get("day_change_pct")
        day_str = "" if day is None else f"{day:+.2f}%"
        pnl = p.get("pnl")
        table.add_row(
            str(p.get("symbol") or p.get("isin") or ""),
            str(p.get("name") or ""),
            str(p.get("quantity") or ""),
            "" if live is None else f"{live:,.2f}",
            day_str,
            str(p.get("market_value") or ""),
            "" if pnl is None else str(pnl),
        )
    console.print("  [bold]Portfolio[/bold]")
    console.print(table)
    for t in data.get("totals", []):
        pnl = t.get("pnl")
        pnl_str = "" if pnl is None else f" | P&L {pnl} ({t.get('pnl_pct')}%)"
        console.print(
            f"  [bold]{t.get('currency')}[/bold] total: {t.get('market_value')}"
            f" across {t.get('position_count')} positions{pnl_str}"
        )
    as_of = data.get("as_of_dates") or []
    console.print(
        f"  [dim]{data.get('priced_count', 0)}/{data.get('count', 0)} priced live"
        + (f" · values as of {', '.join(as_of)}" if as_of else "")
        + "[/dim]\n"
    )
    return True


register(
    SlashCommand(
        "/portfolio",
        _cmd_portfolio,
        args="[live | import-holdings <csv>]",
        description="Show stock portfolio with live prices + P&L; import broker cost basis",
    )
)
register(
    SlashCommand(
        "/routines",
        _cmd_routines,
        args="[list|due|tick|add <schedule> <template> <title>|approve <id>|clear]",
        description="Manage and trigger approved routines",
        argument_completer=_complete_routines,
    )
)
register(
    SlashCommand(
        "/routine",
        _cmd_routines,
        args="[list|due|tick|add <schedule> <template> <title>|approve <id>|clear]",
        description="Alias for /routines",
        hidden=True,
        argument_completer=_complete_routines,
    )
)
register(
    SlashCommand(
        "/heartbeats",
        _cmd_heartbeats,
        args="[list|runs [name] [N]|trigger <name>]",
        description="Inspect heartbeats, run history, or fire one manually",
        argument_completer=_complete_heartbeats,
    )
)
register(
    SlashCommand(
        "/heartbeat",
        _cmd_heartbeats,
        args="[list|runs [name] [N]|trigger <name>]",
        description="Alias for /heartbeats",
        hidden=True,
        argument_completer=_complete_heartbeats,
    )
)
register(
    SlashCommand(
        "/llm",
        _cmd_llm,
        args="[status|pressure|mode <active|idle|thermal>|unpin]",
        description="Inspect or pin the local-LLM resource governor mode",
        argument_completer=_complete_llm,
    )
)
register(SlashCommand("/info", _cmd_info, description="Show session info"))
register(SlashCommand("/trace", _cmd_trace, description="Show full tool trace from last turn"))
register(
    SlashCommand(
        "/session",
        _cmd_session,
        args="[N]",
        description="Show session log path + tail last N events",
    )
)
register(
    SlashCommand(
        "/replay",
        _cmd_replay,
        args="[turns]",
        description="Replay the current session timeline",
    )
)
register(SlashCommand("/export", _cmd_export, args="[file]", description="Export session to file"))
register(SlashCommand("/exit", _cmd_exit, description="Exit IRIS"))
register(SlashCommand("/quit", _cmd_exit, hidden=True))
register(SlashCommand("/bye", _cmd_exit, hidden=True))
