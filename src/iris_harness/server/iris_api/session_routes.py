"""Sessions and traces for the console: the chat list and messages, chat status, Call Trace, slash commands, and answer feedback.

    GET    /api/traces
    GET    /api/sessions
    GET    /api/sessions/{session_id}/messages
    GET    /api/chat-status
    GET    /api/slash-commands
    POST   /api/slash-dispatch
    POST   /api/feedback
    GET    /api/feedback
    POST   /surface-feedback
    GET    /api/traces/{trace_id}

Moved out of ``create_app`` unchanged (review item: split the god function); the route
table and OpenAPI schema are identical before and after. The write guard in ``main``
still gates the mutating routes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from iris_harness.foundation.logsafe import log_safe
from iris_harness.foundation.paths import repo_root
from iris_harness.runtime import IrisRuntime

REPO_ROOT = repo_root()
logger = logging.getLogger(__name__)


def _as_preformatted(output: str) -> str:
    """Fence captured terminal output so a markdown renderer keeps its columns.

    The web chat renders a command's output as markdown, which collapses runs of
    whitespace — that would turn rich's aligned tables (`/skills`, `/queue`) into
    a jumble. A one-line confirmation has no alignment to protect and reads better
    as plain text, so only multi-line output is fenced.
    """
    text = output.strip()
    if "\n" not in text:
        return text
    return f"```text\n{text}\n```"


def _mask_governance_reasons(trace: dict[str, Any]) -> dict[str, Any]:
    """Mask email addresses in a trace's governance reasons, at the display boundary.

    A reason is text a hook wrote into the ledger, and some name an account; the ledger
    keeps them, a screen does not show them (``kernel/governance/display_mask.py``, the
    same mask ``GET /governance/audit`` applies).
    """
    from iris_harness.kernel.governance.display_mask import mask_text

    for event in trace.get("governance") or ():
        event["reason"] = mask_text(str(event.get("reason") or ""))
    for node in trace.get("nodes") or ():
        gov = node.get("governance")
        if isinstance(gov, dict) and gov.get("reason"):
            gov["reason"] = mask_text(str(gov["reason"]))
    return trace


def _self_api_url(http_request: Request) -> str:
    """This API's own base URL, for the slash commands that are its HTTP clients.

    ``/llm``, ``/reminders``, ``/routines``, ``/heartbeats`` and ``/portfolio``
    reach the runtime over HTTP — that is how the REPL uses them too. Taken from
    the live request rather than a constant so it stays right whatever host and
    port the server was actually bound to.
    """
    return str(http_request.base_url).rstrip("/")


class FeedbackRequest(BaseModel):
    """Body for ``POST /api/feedback`` — one user reaction to an answer (ADR-0072)."""

    sentiment: str  # "up" | "down"
    session_id: str | None = None
    turn_id: str | None = None
    trace_id: str | None = None
    rating: int | None = None  # optional 1-5
    note: str | None = None
    intent: str | None = None
    agent_type: str | None = None


class SurfaceFeedbackRequest(BaseModel):
    """Body for ``POST /surface-feedback`` — was a proactively-surfaced item useful?

    Generic across subsystems (issue 0028). Either supply ``ref`` (a self-describing
    ``fb:...`` surface token) or the explicit ``(subsystem, surface_kind, dims)``.
    ``verdict`` is ``not_useful`` (suppress similar) or ``useful`` (lift suppression).
    """

    verdict: str = "not_useful"
    ref: str | None = None
    subsystem: str | None = None
    surface_kind: str | None = None
    dims: dict[str, str] | None = None
    session_id: str | None = None


class SlashDispatchRequest(BaseModel):
    """Request body for ``POST /api/slash-dispatch`` from web chat."""

    command: str = Field(..., min_length=1, max_length=256)
    session_id: str = Field(default="default", max_length=128)


def install_session_routes(app: FastAPI, runtime: Callable[[], Any]) -> None:
    """Register these routes. ``runtime`` returns the live runtime or raises 503."""

    @app.get("/api/traces")
    def list_traces(limit: int = 50) -> list[dict[str, Any]]:
        """Newest-first summaries of recent turns, for the Call Trace screen."""
        from iris_harness.foundation.observability.trace_builder import list_traces as _list

        return _list(limit=max(1, min(limit, 200)))

    @app.get("/api/sessions")
    def list_sessions(limit: int = 50, include_runs: bool = False) -> list[dict[str, Any]]:
        """Recent conversation sessions with their turns — the chat history and Sessions.

        Playground, eval and test runs (retention.yaml's ephemeral prefixes) are the
        harness exercising itself, not the owner's conversations: they are left out
        unless ``include_runs``. They still log, so Call Trace can show them.
        """
        from iris_harness.foundation.observability.trace_builder import list_sessions as _list
        from iris_harness.memory.retention import is_ephemeral_session

        # A session the owner removed (ADR-0119) is out of the chat list, runs or not.
        removed = _removed_session_ids()

        def skip(session_id: str) -> bool:
            return session_id in removed or (not include_runs and is_ephemeral_session(session_id))

        return _list(limit=max(1, min(limit, 200)), skip=skip)

    def _removed_session_ids() -> set[str]:
        """Sessions the owner removed (ADR-0119); 503 when the ledger is unreadable.

        Fails closed, like the retriever and the Map: without the ledger the list
        cannot tell which sessions the owner removed, and listing them all would bring
        a removed conversation back. A 503 says "try again", which an empty list
        ("you have no conversations") would not.
        """
        store = getattr(app.state.runtime, "memory_store", None) if app.state.runtime else None
        try:
            return set(store.removed_session_ids()) if store is not None else set()
        except AttributeError:  # a store without the ledger removes nothing
            return set()
        except Exception as exc:  # refuse the list; logged
            logger.warning(
                "sessions: removed-session ledger unreadable (%s); refusing to list sessions",
                type(exc).__name__,
                exc_info=True,
            )
            raise HTTPException(
                status_code=503,
                detail="the list of removed conversations cannot be read; try again shortly",
            ) from None

    @app.get("/api/sessions/{session_id}/messages")
    def session_messages(session_id: str) -> list[dict[str, Any]]:
        """Chronological user/assistant messages for one session — to resume it in chat."""
        from iris_harness.foundation.observability.trace_builder import session_messages as _msgs

        return _msgs(session_id)

    @app.get("/api/chat-status")
    def chat_status(session_id: str = "default") -> dict[str, Any]:
        """REPL-style status for the web chat header/footer.

        Includes IRIS version, active provider/model, and current context-window
        utilization for the selected session.
        """
        from iris_harness.llm.providers import ProviderManager
        from iris_harness.services.system.inventory import runtime_inventory

        inv = runtime_inventory()
        profile = ProviderManager().get_active()

        rt = getattr(app.state, "runtime", None)
        window: dict[str, Any] | None = None
        if rt is not None:
            health = rt.sessions.context_health(session_id)
            raw_window = health.get("window")
            if isinstance(raw_window, dict):
                window = {
                    "budget_tokens": raw_window.get("budget_tokens"),
                    "current_tokens": raw_window.get("current_tokens"),
                    "fill_pct": raw_window.get("fill_pct"),
                }

        return {
            "session_id": session_id,
            # A turn still running for this session — one the app lost the stream of
            # when it went to the background. The app polls the session until it ends.
            "turn_in_progress": app.state.turns.running(session_id) is not None,
            "iris_version": inv.iris_version,
            "provider": profile.name,
            "provider_label": profile.display_name,
            "model": profile.model,
            "window": window,
        }

    @app.get("/api/slash-commands")
    def slash_commands() -> dict[str, Any]:
        """Slash commands the web can actually run, for the chat composer.

        Suggestions are filtered to what `/api/slash-dispatch` will execute. They
        were not, which is the whole of the reported bug: the composer offered all
        22 visible commands and typing `/model qwen3.5:latest` came back
        "unsupported slash command for web execution". Both ends now read the same
        table in `cli/web_commands.py`.
        """
        from iris_harness.cli.commands import visible_commands
        from iris_harness.cli.web_commands import web_supported_names
        from iris_harness.llm.providers import ProviderManager

        offered = {
            "/help",
            "/provider",
            "/info",
            "/sessions",
            "/clear",
            "/reset",
        } | web_supported_names()

        class _SessionStub:
            def __init__(self, cwd: str) -> None:
                self.cwd = cwd

        class _ContextStub:
            def __init__(self, cwd: str) -> None:
                self.session = _SessionStub(cwd)
                self.provider_manager = ProviderManager()

        ctx = _ContextStub(str(REPO_ROOT))
        commands: list[dict[str, Any]] = []
        seen: set[str] = set()

        for cmd in visible_commands():
            if cmd.name not in offered:
                continue
            key = cmd.name
            if key not in seen:
                commands.append(
                    {
                        "name": cmd.name,
                        "args": cmd.args,
                        "description": cmd.description,
                    }
                )
                seen.add(key)

            if cmd.argument_completer is None:
                continue
            try:
                suggestions = cmd.argument_completer(ctx, "")
            except Exception:
                logger.debug("slash completer failed for %s", cmd.name, exc_info=True)
                continue
            for suggestion, description in suggestions:
                full = f"{cmd.name} {suggestion}".strip()
                if full in seen:
                    continue
                commands.append(
                    {
                        "name": full,
                        "args": "",
                        "description": description,
                    }
                )
                seen.add(full)
        return {"count": len(commands), "commands": commands}

    @app.post("/api/slash-dispatch")
    def slash_dispatch(request: SlashDispatchRequest, http_request: Request) -> dict[str, Any]:
        """Execute web-supported slash commands.

        The web composer exposes the same slash command suggestions as the REPL,
        but execution is scoped to commands that are safe/meaningful for HTTP
        clients and don't depend on terminal-only rendering.
        """
        from iris_harness.cli.commands import visible_commands
        from iris_harness.llm.providers import ProviderManager
        from iris_harness.services.system.inventory import runtime_inventory

        raw = request.command.strip()
        if not raw.startswith("/"):
            raise HTTPException(status_code=422, detail="slash command must start with '/'")

        parts = raw.split(maxsplit=1)
        name = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        from iris_harness.cli.web_commands import (
            WebCommandError,
            run_web_command,
            web_supported_names,
        )

        # Commands answered here rather than by running the REPL handler: each
        # needs an HTTP-shaped reply (an action for the client, or JSON fields
        # beside the text) instead of captured terminal output.
        api_handled = {"/help", "/provider", "/info", "/sessions", "/clear", "/reset"}
        runnable = web_supported_names()
        web_supported = api_handled | runnable

        if name not in web_supported:
            raise HTTPException(
                status_code=400,
                detail=(
                    "unsupported slash command for web execution; "
                    "currently supported: " + ", ".join(sorted(web_supported))
                ),
            )

        if name == "/help":
            rows = []
            for cmd in visible_commands():
                marker = "*" if cmd.name in web_supported else " "
                tail = f" {cmd.args}" if cmd.args else ""
                desc = f" - {cmd.description}" if cmd.description else ""
                rows.append(f"{marker} {cmd.name}{tail}{desc}")
            output = (
                "Slash commands (* executable in web):\n"
                + "\n".join(rows)
                + "\n\nTip: use /provider <name> to switch model provider."
            )
            return {
                "ok": True,
                "executed": True,
                "command": name,
                "output": output,
            }

        if name == "/clear":
            return {
                "ok": True,
                "executed": True,
                "command": name,
                "action": "clear_messages",
                "output": "Cleared chat messages for this session view.",
            }

        if name == "/reset":
            return {
                "ok": True,
                "executed": True,
                "command": name,
                "action": "new_session",
                "output": "Started a fresh chat session.",
            }

        if name == "/sessions":
            from iris_harness.foundation.observability.trace_builder import list_sessions as _list

            sessions = _list(limit=10)
            if not sessions:
                output = "No recent sessions found."
            else:
                lines = ["Recent sessions:"]
                for row in sessions:
                    sid = str(row.get("session_id") or "unknown")
                    turns = int(row.get("turn_count") or 0)
                    when = str(row.get("last_at") or row.get("started_at") or "")
                    title = str(row.get("title") or "").strip()
                    summary = f"- {sid} · {turns} turn(s)"
                    if when:
                        summary += f" · {when}"
                    if title:
                        summary += f"\n  {title}"
                    lines.append(summary)
                output = "\n".join(lines)
            return {
                "ok": True,
                "executed": True,
                "command": name,
                "output": output,
            }

        if name == "/info":
            inv = runtime_inventory()
            profile = ProviderManager().get_active()
            rt = getattr(app.state, "runtime", None)
            window_text = "context window: unavailable"
            if rt is not None:
                health = rt.sessions.context_health(request.session_id)
                raw_window = health.get("window")
                if isinstance(raw_window, dict):
                    cur = raw_window.get("current_tokens")
                    budget = raw_window.get("budget_tokens")
                    fill = raw_window.get("fill_pct")
                    if isinstance(fill, (float, int)):
                        window_text = (
                            f"context window: {cur}/{budget} " f"({float(fill) * 100:.1f}% used)"
                        )
                    else:
                        window_text = f"context window: {cur}/{budget}"
            output = "\n".join(
                [
                    f"IRIS v{inv.iris_version}",
                    f"session: {request.session_id}",
                    f"provider: {profile.display_name} ({profile.name})",
                    f"model: {profile.model}",
                    window_text,
                ]
            )
            return {
                "ok": True,
                "executed": True,
                "command": name,
                "provider": profile.name,
                "provider_label": profile.display_name,
                "model": profile.model,
                "output": output,
            }

        if name in runnable:
            # Run the real REPL handler and return what it printed, so the web
            # and the terminal cannot drift into two different answers.
            try:
                output = run_web_command(
                    name,
                    args,
                    session_id=request.session_id,
                    overrides=app.state.model_overrides,
                    provider_manager=ProviderManager(),
                    cwd=str(REPO_ROOT),
                    api_url=_self_api_url(http_request),
                )
            except WebCommandError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("slash command %s failed", log_safe(name))
                raise HTTPException(
                    status_code=500,
                    detail=f"{name} failed ({type(exc).__name__}); see the server log",
                ) from exc
            model, router_model = app.state.model_overrides.get(request.session_id)
            return {
                "ok": True,
                "executed": True,
                "command": name,
                "output": _as_preformatted(output),
                "model": model or None,
                "router_model": router_model or None,
            }

        mgr = ProviderManager()
        profiles = mgr.list_profiles()
        by_name = {p.name: p for p in profiles}

        if not args:
            current = mgr.get_active()
            return {
                "ok": True,
                "executed": True,
                "command": name,
                "provider": current.name,
                "provider_label": current.display_name,
                "model": current.model,
                "output": (
                    "Active provider: "
                    f"{current.display_name} ({current.name}) · model {current.model}. "
                    "Choose one via `/provider <name>`: " + ", ".join(sorted(by_name.keys()))
                ),
            }

        target = args.lower()
        if target not in by_name:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"unknown provider: {target!r}. "
                    "Available: " + ", ".join(sorted(by_name.keys()))
                ),
            )

        mgr.set_active(target)
        active = mgr.get_active()
        return {
            "ok": True,
            "executed": True,
            "command": name,
            "provider": active.name,
            "provider_label": active.display_name,
            "model": active.model,
            "output": (
                "Provider switched to "
                f"{active.display_name} ({active.name}) · model {active.model}."
            ),
        }

    @app.post("/api/feedback")
    def submit_feedback(request: FeedbackRequest) -> dict[str, Any]:
        """Record one user reaction to an answer (ADR-0072 slice 1).

        Local learning telemetry only — no external effect — so it is intentionally
        not behind the write gate; a read-only user can still rate answers.
        """
        rt: IrisRuntime | None = app.state.runtime
        collector = getattr(rt, "signal_collector", None) if rt is not None else None
        if collector is None:
            raise HTTPException(status_code=503, detail="runtime unavailable")
        sentiment = request.sentiment.strip().lower()
        if sentiment not in {"up", "down"}:
            raise HTTPException(status_code=422, detail="sentiment must be 'up' or 'down'")
        collector.record_feedback(
            sentiment=sentiment,
            session_id=request.session_id,
            turn_id=request.turn_id,
            trace_id=request.trace_id,
            rating=request.rating,
            note=request.note,
            intent=request.intent,
            agent_type=request.agent_type,
        )
        return {"ok": True}

    @app.get("/api/feedback")
    def list_feedback(limit: int = 100) -> list[dict[str, Any]]:
        """Recent user-feedback signals, newest first (read-back for the learning UI)."""
        rt: IrisRuntime | None = app.state.runtime
        store = getattr(rt, "learning_store", None) if rt is not None else None
        if store is None:
            return []
        rows = store.recent_signals(metric_name="user_feedback", limit=max(1, min(limit, 500)))
        return [
            {
                "ts": r.ts.isoformat(),
                "sentiment": r.metadata.get("sentiment"),
                "rating": r.metadata.get("rating"),
                "note": r.metadata.get("note"),
                "intent": r.metadata.get("intent"),
                "agent_type": r.metadata.get("agent_type"),
                "session_id": r.session_id,
                "trace_id": r.trace_id,
            }
            for r in rows
        ]

    @app.post("/surface-feedback")
    def submit_surface_feedback(request: SurfaceFeedbackRequest) -> dict[str, Any]:
        """Record whether a proactively-surfaced item was useful (issue 0028).

        Local learning telemetry only (suppression ledger in ``learning.db``) — no
        external effect — so it is not behind the write gate; a read-only user can
        still suppress noise. Mirrors the CLI ``iris feedback`` and chat surfaces.
        """
        from iris_harness.services.learning.suppression import (
            VERDICTS,
            SurfaceFeedbackStore,
            decode_ref,
        )

        verdict = request.verdict.strip().lower()
        if verdict not in VERDICTS:
            raise HTTPException(status_code=422, detail=f"verdict must be one of {list(VERDICTS)}")
        if request.ref:
            try:
                subsystem, surface_kind, dims = decode_ref(request.ref)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        elif request.subsystem and request.surface_kind and request.dims is not None:
            subsystem, surface_kind, dims = (
                request.subsystem,
                request.surface_kind,
                request.dims,
            )
        else:
            raise HTTPException(
                status_code=422,
                detail="supply either 'ref' or all of (subsystem, surface_kind, dims)",
            )
        store = SurfaceFeedbackStore()
        store.ensure_schema()
        store.record(subsystem, surface_kind, dims, verdict, session_id=request.session_id or "")
        return {"ok": True, "subsystem": subsystem, "surface_kind": surface_kind}

    @app.get("/api/traces/{trace_id}")
    def get_trace(trace_id: str) -> dict[str, Any]:
        """Full node/edge graph for a single turn (``<session_id>~<turn>``)."""
        from iris_harness.foundation.observability.trace_builder import get_trace as _get

        trace = _get(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return _mask_governance_reasons(trace)
