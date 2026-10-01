"""Interactive REPL for IRIS.

Uses prompt_toolkit for a fixed bottom prompt + two-line footer layout:
  - LLM responses and spinner print into the scrolling area above the prompt
  - The `you ›` input line is pinned to the bottom
  - Line 1 of the footer toolbar shows metrics (tokens, model, latency, resources)
  - Line 2 of the footer toolbar shows the slash-command strip
  - Tab-completion is available for all registered slash commands
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

from iris_harness.cli.api_client import harness_urlopen
from iris_harness.foundation.auth import auth_headers
from iris_harness.llm.providers import ProviderManager

from . import commands as _cmds
from .commands import REPLContext
from .render import (
    _BAR_EMPTY,
    _BAR_FILLED,
    _BAR_WIDTH,
    FooterMetrics,
    _context_window_for,
    _fmt_tokens,
    console,
    format_command_strip,
    print_banner,
    print_error,
    print_response,
    sample_system_metrics,
)
from .session import Session, SessionManager
from .welcome import fetch_welcome

_HISTORY_FILE = Path.home() / ".iris" / "history"
_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


class _SlashCompleter(Completer):
    """Auto-popup menu that fires on '/' and offers per-command argument suggestions.

    Two modes:
      • ``/<prefix>``               → list visible slash commands with descriptions
      • ``/<command> <arg-prefix>`` → call the command's ``argument_completer``
        for dynamic suggestions (provider profiles, router models, etc.)
    """

    def __init__(self, ctx_getter: Callable[[], REPLContext]) -> None:
        self._ctx_getter = ctx_getter

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Generator[Completion, None, None]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        space_idx = text.find(" ")
        if space_idx < 0:
            # ── Command-name completion ──────────────────────────────────
            for cmd in _cmds.visible_commands():
                if cmd.name.startswith(text):
                    label = f"{cmd.name} {cmd.args}".strip() if cmd.args else cmd.name
                    yield Completion(
                        cmd.name,
                        start_position=-len(text),
                        display=label,
                        display_meta=cmd.description,
                    )
            return

        # ── Argument completion ──────────────────────────────────────────
        cmd_name = text[:space_idx]
        arg_prefix = text[space_idx + 1 :]
        target = _cmds.get(cmd_name)
        if target is None or target.argument_completer is None:
            return
        try:
            options = target.argument_completer(self._ctx_getter(), arg_prefix)
        except Exception:  # noqa: BLE001 — completer must never crash the prompt
            return
        for value, desc in options:
            if not value.lower().startswith(arg_prefix.lower()):
                continue
            yield Completion(
                value,
                start_position=-len(arg_prefix),
                display=value,
                display_meta=desc,
            )


def _make_token_printer(
    status: object,
    state: dict[str, object],
    live_writer: Callable[[str], None] | None = None,
) -> Callable[[str], None]:
    """Build a token collector that stops the spinner on first token.

    ``status`` only needs a ``stop()`` method (Rich's ``console.status`` return
    value satisfies this; tests pass a small stub).
    """

    def on_token(text: str) -> None:
        if not state["started"]:
            stop = getattr(status, "stop", None)
            if callable(stop):
                stop()
            else:
                # Compatibility for older tests/callers that passed REPLContext.
                lock = getattr(status, "footer_lock", None)
                if hasattr(lock, "__enter__") and hasattr(lock, "__exit__"):
                    with cast(Any, lock):
                        status.thinking = ""  # type: ignore[attr-defined]
            state["started"] = True
        buffer = state.setdefault("buffer", [])
        if isinstance(buffer, list):
            buffer.append(text)
        if live_writer is not None:
            live_writer(text)
            state["live_rendered"] = True

    return on_token


def _flush_streamed_response(state: dict[str, object]) -> None:
    """Render buffered streaming output with the normal Markdown renderer."""
    if state.get("live_rendered"):
        return
    buffer = state.get("buffer", [])
    if not isinstance(buffer, list) or not buffer:
        return
    text = "".join(str(chunk) for chunk in buffer)
    if text.strip():
        print_response(text)


def _flush_final_response(body: dict[str, object]) -> bool:
    """Render a final done-event response when no token stream was emitted."""

    response = body.get("response")
    if not isinstance(response, str) or not response.strip():
        return False
    print_response(response)
    return True


def _post_stream(
    message: str,
    session_id: str,
    api_url: str,
    *,
    preferred_model: str = "",
    provider_profile: str = "",
    router_model: str = "",
    strict: bool = False,
    on_token: Callable[[str], None],
    on_activity: Callable[[str], None] | None = None,
    on_trace: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """POST to ``/chat/stream`` and dispatch each event kind.

    ``on_activity`` receives short ephemeral status strings (live spinner
    replacement). ``on_trace`` receives raw tool-call detail for opt-in
    inspection (/trace). Returns the final ``done`` event payload.
    """
    payload: dict[str, object] = {"message": message, "session_id": session_id, "strict": strict}
    if preferred_model:
        payload["preferred_model"] = preferred_model
    if provider_profile:
        payload["provider_profile"] = provider_profile
    if router_model:
        payload["router_model"] = router_model
    encoded = json.dumps(payload).encode()
    req = urllib.request.Request(  # noqa: S310
        f"{api_url}/chat/stream",
        data=encoded,
        headers={"Content-Type": "application/json", **auth_headers()},
        method="POST",
    )
    final: dict[str, object] = {}
    with harness_urlopen(req, purpose="chat-stream", timeout=600) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            evt = json.loads(line)
            kind = evt.get("event")
            if kind == "token":
                text = str(evt.get("text") or "")
                if text:
                    on_token(text)
            elif kind == "activity":
                if on_activity:
                    on_activity(str(evt.get("text") or ""))
            elif kind == "trace":
                if on_trace:
                    trace_text = str(evt.get("text") or "")
                    payload = evt.get("payload")
                    if isinstance(payload, dict):
                        payload_text = json.dumps(payload, indent=2, sort_keys=True, default=str)
                        trace_text = f"{trace_text}\n{payload_text}" if trace_text else payload_text
                    on_trace(trace_text)
            elif kind == "done":
                final = {k: v for k, v in evt.items() if k != "event"}
            elif kind == "error":
                raise RuntimeError(str(evt.get("error") or "stream error"))
    return final


def _build_footer(body: dict[str, object], sys_metrics: dict[str, float]) -> FooterMetrics:
    meta: dict[str, object] = body.get("metadata", {}) or {}  # type: ignore[assignment]
    return FooterMetrics(
        prompt_tokens=int(str(meta.get("prompt_tokens") or 0)),
        completion_tokens=int(str(meta.get("completion_tokens") or 0)),
        model=str(meta.get("model", "")),
        router_model=str(meta.get("router_model", "")),
        latency_ms=float(meta.get("total_latency_ms", 0) or 0),  # type: ignore[arg-type]
        cpu_pct=sys_metrics.get("cpu_pct", -1.0),
        mem_used_gb=sys_metrics.get("mem_used_gb", -1.0),
        mem_total_gb=sys_metrics.get("mem_total_gb", -1.0),
        gpu_pct=sys_metrics.get("gpu_pct", -1.0),
        gpu_mem_used_gb=sys_metrics.get("gpu_mem_used_gb", -1.0),
        gpu_mem_total_gb=sys_metrics.get("gpu_mem_total_gb", -1.0),
        context_window=int(str(meta.get("context_window") or meta.get("num_ctx") or 0)),
    )


def _footer_metrics_text(metrics: FooterMetrics | None) -> str:
    """Build the plain-text metrics line for the toolbar (line 1)."""
    import re

    if metrics is None:
        return "  IRIS  ·  ready"

    parts: list[str] = []

    if metrics.prompt_tokens or metrics.completion_tokens:
        parts.append(
            f"↑{_fmt_tokens(metrics.prompt_tokens)}  ↓{_fmt_tokens(metrics.completion_tokens)}"
        )

    ctx_total = metrics.context_window or _context_window_for(metrics.model)
    if ctx_total and metrics.prompt_tokens:
        ratio = min(metrics.prompt_tokens / ctx_total, 1.0)
        filled = round(ratio * _BAR_WIDTH)
        bar = _BAR_FILLED * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)
        parts.append(f"ctx {bar} {_fmt_tokens(metrics.prompt_tokens)}/{_fmt_tokens(ctx_total)}")

    if metrics.router_model:
        router_clean = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", metrics.router_model)
        parts.append(f"router {router_clean}")

    if metrics.model:
        model_clean = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", metrics.model)
        parts.append(f"model {model_clean}")

    if metrics.latency_ms:
        parts.append(f"{metrics.latency_ms:.0f}ms")

    if metrics.cpu_pct >= 0:
        parts.append(f"cpu {metrics.cpu_pct:.0f}%")

    if metrics.mem_used_gb >= 0:
        parts.append(f"mem {metrics.mem_used_gb:.1f}/{metrics.mem_total_gb:.0f}GB")

    if metrics.gpu_pct >= 0:
        parts.append(
            f"gpu {metrics.gpu_pct:.0f}%  "
            f"{metrics.gpu_mem_used_gb:.1f}/{metrics.gpu_mem_total_gb:.0f}GB"
        )

    return "  " + "  ·  ".join(parts) if parts else "  IRIS  ·  ready"


def run_repl(
    *,
    session: Session,
    session_manager: SessionManager,
    api_url: str,
    provider_manager: ProviderManager | None = None,
    strict: bool = False,
) -> int:
    mgr = provider_manager or ProviderManager()

    ctx = REPLContext(
        session=session,
        session_manager=session_manager,
        api_url=api_url,
        provider_manager=mgr,
        strict_mode=strict,
    )

    def redraw_banner() -> None:
        active = ctx.provider_manager.get_active()
        print_banner(
            session_id=ctx.session.id,
            api_url=api_url,
            provider=active.display_name,
            model=active.model,
        )
        console.print(
            "  [dim]Type a message and press Enter. "
            "Tab for slash commands. Type[/dim] [cyan]/help[/cyan] [dim]for the full list.[/dim]\n"
        )

    ctx.redraw_banner = redraw_banner

    def get_toolbar() -> ANSI:
        width = shutil.get_terminal_size().columns
        divider = "\033[2m" + "─" * width + "\033[0m"
        with ctx.footer_lock:
            thinking = ctx.thinking
            metrics_line = _footer_metrics_text(ctx.footer)
        status_line = f"  {thinking}" if thinking else metrics_line
        strip = format_command_strip()
        return ANSI(f"{divider}\n\n{status_line}\n\n{strip}")

    _toolbar_style = Style.from_dict(
        {
            "bottom-toolbar": "bg:default fg:default noreverse",
        }
    )

    pt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(_HISTORY_FILE)),
        bottom_toolbar=get_toolbar,
        refresh_interval=1.0,
        style=_toolbar_style,
        completer=_SlashCompleter(lambda: ctx),
        complete_while_typing=True,
    )

    redraw_banner()
    # The first chat on this install opens with IRIS's welcome (ADR-0127); the harness
    # decides whether it is due, and says so only once.
    welcome = fetch_welcome(api_url, channel="console")
    if welcome:
        print_response(welcome)

    with patch_stdout(raw=True):
        while True:
            try:
                raw = pt_session.prompt(
                    ANSI("\033[1;36myou\033[0m \033[2m›\033[0m "),
                ).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye.[/dim]")
                break

            if not raw:
                continue

            # ── Slash commands ────────────────────────────────────────────────
            if raw.startswith("/"):
                result = _cmds.dispatch(ctx, raw)
                if result is None:
                    cmd = raw.split()[0].lower()
                    print_error(f"Unknown command: {cmd}  (try /help)")
                elif result is False:
                    break
                continue

            # ── Send to API (streaming) ───────────────────────────────────────
            active = ctx.provider_manager.get_active()
            stream_state: dict[str, object] = {"started": False, "buffer": []}
            trace_buffer: list[str] = []
            # Start spinner manually (not via context manager) so that
            # _flush_streamed_response and footer update run with a clean
            # console state — the context manager's __exit__ can interfere
            # with subsequent console.print() calls on some terminals.
            status_obj = console.status("[dim]thinking…[/dim]", spinner="dots")
            status_obj.start()

            def write_live_token(text: str) -> None:
                console.print(text, end="", markup=False, highlight=False, soft_wrap=True)

            on_token = _make_token_printer(status_obj, stream_state, write_live_token)

            # Activity events update the spinner text live (replaces "thinking…").
            # Trace events accumulate per-turn for the /trace slash command.
            def on_activity(
                text: str,
                _state: dict[str, object] = stream_state,
                _status: object = status_obj,
            ) -> None:
                if not text or _state["started"]:
                    return
                try:
                    _status.update(f"[dim]{text}[/dim]")  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001, S110 — spinner update is best-effort
                    pass

            def on_trace(
                text: str,
                _buf: list[str] = trace_buffer,
            ) -> None:
                if text:
                    _buf.append(text)

            body: dict[str, object] = {}
            try:
                body = _post_stream(
                    raw,
                    ctx.session.id,
                    api_url,
                    preferred_model=ctx.session.preferred_model,
                    provider_profile=active.name,
                    router_model=ctx.session.router_model,
                    strict=ctx.strict_mode,
                    on_token=on_token,
                    on_activity=on_activity,
                    on_trace=on_trace,
                )
            except OSError as exc:
                if not stream_state["started"]:
                    status_obj.stop()
                print_error(f"Cannot reach IRIS API at {api_url} — is the server running?\n  {exc}")
                continue
            except urllib.error.HTTPError as exc:
                if not stream_state["started"]:
                    status_obj.stop()
                print_error(f"HTTP {exc.code}: {exc.read().decode()[:200]}")
                continue
            except Exception as exc:  # noqa: BLE001
                if not stream_state["started"]:
                    status_obj.stop()
                print_error(str(exc))
                continue
            finally:
                # Guard: ensure spinner is always stopped even on unexpected paths.
                if not stream_state["started"]:
                    status_obj.stop()

            if stream_state["started"]:
                _flush_streamed_response(stream_state)
                console.print()
            elif _flush_final_response(body):
                console.print()

            ctx.last_trace = "\n\n".join(trace_buffer)

            if bool(body.get("has_errors", False)):
                meta_parts: list[str] = []
                intent_label = str(body.get("intent", ""))
                agent_label = str(body.get("agent_type", ""))
                if intent_label:
                    meta_parts.append(f"intent={intent_label}")
                if agent_label:
                    meta_parts.append(f"agent={agent_label}")
                console.print(f"  [bold red]✗[/bold red]  [dim]{' · '.join(meta_parts)}[/dim]")
                console.print()

            session_manager.touch(ctx.session)
            sys_metrics = sample_system_metrics()

            with ctx.footer_lock:
                ctx.footer = _build_footer(body, sys_metrics)

    return 0
