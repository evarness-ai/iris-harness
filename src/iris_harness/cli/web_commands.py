"""Run REPL slash commands for HTTP clients, and remember the models they pick.

The web composer offers the same slash commands as the REPL, so the two must
agree on which of them actually run. They did not: the composer suggested all 22
visible commands while ``/api/slash-dispatch`` executed 6, and typing
``/model qwen3.5:latest`` in chat came back as "unsupported slash command for
web execution".

Rather than reimplement each command's output next to the API route — a second
renderer to keep in step with the REPL's, against the thin-renderer rule — this
runs the real handler and captures what it printed. One implementation, two
surfaces. What the web *cannot* safely run is declared here as data
(:data:`WEB_READONLY`), so the suggestion list and the executor read the same
table and cannot drift apart again.

Read-only is enforced per sub-command, not per command: ``/reminders list`` is a
query, ``/reminders tick`` fires a heartbeat, and only the first is offered.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from iris_harness.cli.commands import REPLContext

__all__ = [
    "ModelOverrides",
    "WEB_READONLY",
    "WEB_STATEFUL",
    "WebCommandError",
    "is_web_supported",
    "run_web_command",
    "web_supported_names",
]


class WebCommandError(Exception):
    """A command exists but this surface will not run it as asked."""


# Commands whose output is a read of existing state. The value is the set of
# accepted first words (``""`` means "no argument"); ``None`` means the command
# takes a free-form argument that cannot change anything (a count, a slug).
#
# Anything absent is refused. Adding a key here is a deliberate act: check the
# handler for writes, network egress and terminal-only rendering first.
WEB_READONLY: dict[str, frozenset[str] | None] = {
    "/skills": frozenset({""}),
    "/trace": frozenset({""}),
    "/session": None,  # [N] — tail length
    "/replay": None,  # [turns]
    "/llm": frozenset({"", "status", "pressure"}),
    "/active": frozenset({"", "list"}),
    "/reminders": frozenset({"", "list", "due", "missed", "all"}),
    "/routines": frozenset({"", "list", "due"}),
    "/heartbeats": frozenset({"", "list", "runs"}),
    "/queue": frozenset({"", "list", "show", "stats"}),
    "/portfolio": frozenset({"", "live"}),
}

# Commands that change something this surface *does* own: the model this session
# talks to. They write through the session stub into :class:`ModelOverrides`.
WEB_STATEFUL: frozenset[str] = frozenset({"/model", "/router"})

# Deliberately never offered over HTTP:
#   /exit     — terminal lifecycle, meaningless to a browser
#   /export   — writes a file to the server's disk, not the caller's
#   /compact  — mutates the context window; a write, not a read
#   /clear /reset /help /info /provider /sessions — the API answers these itself,
#     because each needs an HTTP-shaped reply (an action for the client to take,
#     or JSON fields beside the text) rather than captured terminal output.

# Rich renders to the console's width; 80 (the default when stdout is not a tty)
# ellipsizes table cells into uselessness. Wide enough for the real tables.
_CAPTURE_WIDTH = 160

# ``console`` is a module-level singleton shared by every handler, and capturing
# swaps its output buffer. Serialize so two concurrent requests cannot interleave
# into each other's transcript.
_capture_lock = threading.Lock()


@dataclass
class ModelOverrides:
    """Per-session model choices made through the web, kept for later turns.

    The REPL keeps these on its ``Session`` and writes them to the session store.
    A browser has no such object, so a ``/model`` that only echoed a confirmation
    would be forgotten by the next message. This is that memory, and the chat
    routes read it when a request does not name a model itself.

    Process-lifetime only: a reload keeps the choice, a server restart drops back
    to the configured default, which is also when ``config/llm_tiers.yaml`` is
    re-read.
    """

    #: Bound so a long-lived server cannot accumulate one entry per session seen.
    max_sessions: int = 512
    _models: dict[str, str] = field(default_factory=dict)
    _routers: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, session_id: str) -> tuple[str, str]:
        """Return ``(preferred_model, router_model)`` for ``session_id``."""
        with self._lock:
            return self._models.get(session_id, ""), self._routers.get(session_id, "")

    def set(self, session_id: str, *, model: str | None = None, router: str | None = None) -> None:
        if not session_id:
            return
        with self._lock:
            for store, value in ((self._models, model), (self._routers, router)):
                if value is None:
                    continue
                if value:
                    store[session_id] = value
                else:
                    store.pop(session_id, None)
                while len(store) > self.max_sessions:
                    store.pop(next(iter(store)))


class _SessionStub:
    """Stands in for the REPL ``Session`` that the handlers read and write.

    Carries every attribute the runnable handlers touch: ``id`` (``/trace``,
    ``/session`` and ``/replay`` read the session log by it, so these report on
    the caller's real conversation), ``cwd``, the two model overrides, and the
    counters ``/info``-style output prints.
    """

    def __init__(
        self,
        session_id: str,
        cwd: str,
        preferred_model: str,
        router_model: str,
    ) -> None:
        self.id = session_id
        self.cwd = cwd
        self.preferred_model = preferred_model
        self.router_model = router_model
        self.message_count = 0
        self.created_at = ""


class _SessionManagerStub:
    """``save`` is where the handlers persist; route it at the override store."""

    def __init__(self, overrides: ModelOverrides, session_id: str) -> None:
        self._overrides = overrides
        self._session_id = session_id

    def save(self, session: Any) -> None:
        self._overrides.set(
            self._session_id,
            model=getattr(session, "preferred_model", ""),
            router=getattr(session, "router_model", ""),
        )

    def list(self) -> list[Any]:
        return []


class _WebContext:
    """A ``REPLContext`` shaped well enough for the handlers this module runs.

    Duck-typed rather than a real ``REPLContext``: that dataclass carries a
    terminal footer, a redraw callback and a lock that mean nothing here, and
    requiring them would tie the API to REPL internals.
    """

    def __init__(
        self,
        *,
        session_id: str,
        cwd: str,
        overrides: ModelOverrides,
        last_trace: str,
        provider_manager: Any,
        api_url: str,
    ) -> None:
        model, router = overrides.get(session_id)
        self.session = _SessionStub(session_id, cwd, model, router)
        self.session_manager = _SessionManagerStub(overrides, session_id)
        self.provider_manager = provider_manager
        #: Several handlers (``/llm``, ``/reminders``, ``/routines``,
        #: ``/heartbeats``, ``/portfolio``) are HTTP clients for the API by
        #: construction — the REPL reaches the runtime the same way. When the API
        #: runs them it passes its own base URL, so they call back into this
        #: process. These are short read-only GETs; the caller is a person typing
        #: a slash command, so the extra in-flight request is bounded.
        self.api_url = api_url
        self.last_trace = last_trace
        self.footer = None
        self.thinking = ""
        self.strict_mode = False
        #: Read at the ``/model`` and ``/router`` warm-up call sites. A warm-up is
        #: the one self-call worth refusing: it blocks for the 5-30s an Ollama load
        #: takes, to pre-load a model the next chat turn loads anyway.
        self.warm_models = False


def web_supported_names() -> frozenset[str]:
    """Every command name this module can run."""
    return frozenset(WEB_READONLY) | WEB_STATEFUL


def is_web_supported(name: str, args: str = "") -> bool:
    """True when ``name`` with ``args`` is safe to run for an HTTP caller."""
    name = name.lower()
    if name in WEB_STATEFUL:
        return True
    if name not in WEB_READONLY:
        return False
    allowed = WEB_READONLY[name]
    if allowed is None:
        return True
    first = args.strip().split(maxsplit=1)[0].lower() if args.strip() else ""
    return first in allowed


def run_web_command(
    name: str,
    args: str,
    *,
    session_id: str,
    overrides: ModelOverrides,
    provider_manager: Any,
    cwd: str = ".",
    last_trace: str = "",
    api_url: str = "",
) -> str:
    """Run one slash command and return what it printed.

    Raises :class:`WebCommandError` when the command is unknown here, or is known
    but was asked to do something this surface does not allow (a write through a
    read-only command's sub-argument).
    """
    from iris_harness.cli.commands import visible_commands
    from iris_harness.cli.render import console

    name = name.lower()
    if name not in web_supported_names():
        raise WebCommandError(f"{name} cannot run here")
    if not is_web_supported(name, args):
        allowed = WEB_READONLY.get(name)
        offered = ", ".join(sorted(x for x in (allowed or set()) if x)) or "no arguments"
        raise WebCommandError(
            f"{name} is read-only here; this surface accepts: {offered}. "
            "Use the CLI for anything that writes."
        )

    handler = next((c.handler for c in visible_commands() if c.name == name), None)
    if handler is None:
        raise WebCommandError(f"{name} is not a known command")

    ctx = _WebContext(
        session_id=session_id,
        cwd=cwd,
        overrides=overrides,
        last_trace=last_trace,
        provider_manager=provider_manager,
        api_url=api_url,
    )

    with _capture_lock:
        previous_width = console.width
        console.width = _CAPTURE_WIDTH
        try:
            with console.capture() as captured:
                # `_WebContext` satisfies the attributes these handlers read, but
                # `REPLContext` is a dataclass of terminal concerns (footer, redraw
                # callback, lock) that have no meaning here. Structural stand-in
                # against a nominal type — the runnable set is pinned by tests.
                handler(cast("REPLContext", ctx), args)
            output = captured.get()
        finally:
            console.width = previous_width

    # A handler that writes through ``session_manager.save`` has already updated
    # the store; one that mutates the session in place without saving (``/router``
    # does this on some paths) is flushed here so neither route can lose a choice.
    overrides.set(
        session_id,
        model=ctx.session.preferred_model,
        router=ctx.session.router_model,
    )
    return output.strip() or f"{name} produced no output."
