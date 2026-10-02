"""The first-chat welcome (ADR-0127): one turn IRIS opens itself, once per IRIS_HOME.

The first time anyone opens a chat on an install — the web console's Chat, the CLI, a
Telegram chat — the surface asks the harness for the welcome (``POST /chat/welcome``).
The harness decides whether it is due: it is, exactly once per ``IRIS_HOME``, recorded
in ``$IRIS_HOME/welcome.json``. When it is, the welcome runs as a real turn through the
governed pipeline (``IrisRuntime.open_turn`` → ``OPENER_STAGES``): a session log, a
call trace and the response check's audit row, like any turn a deterministic handler
answers, with no model. When it is not, the same call returns the welcome that already
ran, so the call is safe to repeat and every surface can make it.

The text says what this install can do, from the plugins that are mounted (each plugin's
manifest ``summary``), invites a first question and points at Call trace. Its wording is
``config/welcome.yaml``; the opener is declared in ``config/intercepts.yaml``
``openers:``. Held as ``runtime.welcome``; ``compose`` is the opener's handler.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import yaml

from iris_harness.foundation.observability.trace_builder import ID_SEP
from iris_harness.foundation.paths import iris_home
from iris_harness.runtime.types import ChatResult

if TYPE_CHECKING:
    from iris_harness.kernel.governance.hooks.response_payload import Audience
    from iris_harness.runtime.plugin_host.registry import PluginRegistry

logger = logging.getLogger(__name__)

# The opener's name in config/intercepts.yaml ``openers:``.
WELCOME_OPENER = "welcome"

# Used when config/welcome.yaml is missing or unreadable, so a broken config never
# leaves a first chat without its welcome.
DEFAULT_WORDING: dict[str, str] = {
    "intro": "Hi, I'm IRIS. This is the first chat on this install. Here is what I can do:",
    "capability": "- {summary}",
    "no_capabilities": (
        "No plugins are mounted yet, so for now I can talk with you and remember what "
        "you tell me."
    ),
    "invitation": "Ask me something to get started.",
    "trace_note": (
        "No model wrote this reply: a deterministic handler did. Open Call trace to see "
        "this turn's path and the check it passed, and Governance for its audit rows."
    ),
}


def welcome_marker_path() -> Path:
    """Where the welcome is recorded as done: ``$IRIS_HOME/welcome.json``."""
    return iris_home() / "welcome.json"


@dataclass(frozen=True)
class WelcomeOutcome:
    """The welcome turn: the one that just ran (``created``) or the one that ran before.

    ``skipped`` is a home that already had conversations when the welcome was first asked
    for: no turn ran, and none ever will (``skip_reason`` says why). ``session_id`` and
    ``response`` are empty then.
    """

    session_id: str
    created: bool
    response: str
    at: str
    skipped: bool = False
    skip_reason: str = ""

    @property
    def trace_id(self) -> str:
        """The Call trace id of the welcome turn: the first turn of its own session."""
        return f"{self.session_id}{ID_SEP}0" if self.session_id else ""


class WelcomeHost(Protocol):
    """The runtime members the welcome reads."""

    config_dir: Path
    plugin_registry: PluginRegistry

    def open_turn(
        self,
        opener: str,
        *,
        session_id: str,
        channel: str = "console",
        audience: Audience = "owner",
    ) -> ChatResult: ...


def capability_summaries(registry: PluginRegistry) -> list[str]:
    """One line per mounted plugin that says what it lets IRIS do, in mount order.

    A plugin that failed to load is not mounted, so it is not offered. A plugin with no
    ``summary`` (a delivery surface, an import tool) has nothing to say to the owner.
    """
    from iris_harness.runtime.plugin_host.registry import PluginStatus

    lines: list[str] = []
    for record in registry.plugins():
        if record.status not in (PluginStatus.LOADED, PluginStatus.DEGRADED):
            continue
        summary = (record.manifest.summary if record.manifest else "").strip()
        if summary:
            lines.append(summary)
    return lines


def load_wording(config_dir: Path) -> dict[str, str]:
    """``config/welcome.yaml`` over the built-in wording, key by key."""
    wording = dict(DEFAULT_WORDING)
    path = config_dir / "welcome.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return wording
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("welcome.yaml unreadable (%s); using the built-in wording", exc)
        return wording
    if not isinstance(raw, dict):
        logger.warning("welcome.yaml is not a mapping; using the built-in wording")
        return wording
    for key in DEFAULT_WORDING:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            wording[key] = value.strip()
    return wording


def welcome_text(summaries: list[str], wording: dict[str, str]) -> str:
    """The welcome, from what is mounted and the configured wording."""
    if summaries:
        body = "\n".join(wording["capability"].format(summary=s) for s in summaries)
        parts = [wording["intro"], body]
    else:
        parts = [wording["no_capabilities"]]
    parts += [wording["invitation"], wording["trace_note"]]
    return "\n\n".join(parts)


class FirstChatWelcome:
    """Decides whether the welcome is due, runs it once, and composes it. See the module."""

    def __init__(self, host: WelcomeHost) -> None:
        self._host = host
        # One welcome per home, however many surfaces ask at the same moment.
        self._lock = threading.Lock()

    def compose(self, *, session_id: str, span: Any = None) -> ChatResult:
        """The opener's handler: the welcome text for this install, as a turn's result."""
        summaries = capability_summaries(self._host.plugin_registry)
        text = welcome_text(summaries, load_wording(self._host.config_dir))
        if span is not None:
            try:
                span.set_attribute("output.value", text)
            except Exception:  # noqa: BLE001, S110 — tracing never fails a turn
                pass
        return ChatResult(
            response=text,
            intent="welcome",
            agent_type="system",
            sources=("welcome",),
            has_errors=False,
            error_summary=None,
            metadata={
                "session_id": session_id,
                "handler": WELCOME_OPENER,
                "capabilities": len(summaries),
                "model": "deterministic",
                "provider": "local",
                "total_latency_ms": 0.0,
            },
        )

    def ensure(self, *, channel: str = "console", audience: Audience = "owner") -> WelcomeOutcome:
        """Run the welcome if it is due; otherwise return the one that already ran.

        Due means ``$IRIS_HOME/welcome.json`` does not exist and the home has no
        conversation yet. A home that already has one (an install from before the welcome)
        is recorded as skipped, with the reason, and no turn runs: the welcome is for a
        fresh home only. The record of a run is written only after the turn completed, so
        a turn that failed is offered again next time.
        """
        with self._lock:
            recorded = self.recorded()
            if recorded is not None:
                return recorded
            existing = existing_conversations()
            if existing:
                outcome = WelcomeOutcome(
                    session_id="",
                    created=False,
                    response="",
                    at=datetime.now(UTC).isoformat(),
                    skipped=True,
                    skip_reason=(
                        f"this home already had {existing} conversation"
                        f"{'s' if existing != 1 else ''} when the welcome was first asked for"
                    ),
                )
                _write_marker(outcome, channel=channel)
                return outcome
            session_id = uuid.uuid4().hex[:12]
            result = self._host.open_turn(
                WELCOME_OPENER, session_id=session_id, channel=channel, audience=audience
            )
            outcome = WelcomeOutcome(
                session_id=session_id,
                created=True,
                response=result.response,
                at=datetime.now(UTC).isoformat(),
            )
            _write_marker(outcome, channel=channel)
            return outcome

    def new_text_for(self, channel: str) -> Callable[[Audience], str | None]:
        """An in-process surface's opener: the welcome's text when the call ran it, else None.

        What the runtime's own Telegram poller is handed; a surface that reaches the
        harness over HTTP gets the same answer from ``POST /chat/welcome``.
        """

        def opener(audience: Audience = "owner") -> str | None:
            outcome = self.ensure(channel=channel, audience=audience)
            return outcome.response if outcome.created and outcome.response else None

        return opener

    def recorded(self) -> WelcomeOutcome | None:
        """The welcome that already ran on this home, or None when it never has.

        An unreadable record still means it ran: the welcome is never offered twice.
        """
        path = welcome_marker_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("welcome record %s unreadable (%s); treating it as done", path, exc)
            data = {}
        if not isinstance(data, dict):
            data = {}
        return WelcomeOutcome(
            session_id=str(data.get("session_id") or ""),
            created=False,
            response=str(data.get("response") or ""),
            at=str(data.get("at") or ""),
            skipped=data.get("skipped") is True,
            skip_reason=str(data.get("reason") or ""),
        )


def mark_shown_by_setup() -> None:
    """Record the welcome as already delivered by ``iris setup``'s closing screen.

    A no-op once a marker already exists. Keeps ``FirstChatWelcome.ensure`` from
    showing a second "you're all set" message on the first real chat after the
    wizard's own closing screen just showed one.
    """
    if welcome_marker_path().exists():
        return
    outcome = WelcomeOutcome(
        session_id="",
        created=False,
        response="",
        at=datetime.now(UTC).isoformat(),
        skipped=True,
        skip_reason="shown by `iris setup`'s closing screen",
    )
    _write_marker(outcome, channel="cli-setup")


def existing_conversations() -> int:
    """How many conversations this home already has, as the Sessions list counts them.

    The same reading the console's Sessions screen and chat history get from
    ``GET /api/sessions``: the session logs under IRIS_HOME, playground / eval / test runs
    left out. The removed-session ledger is not consulted: a conversation the owner
    removed still means the home was used.
    """
    from iris_harness.foundation.observability.trace_builder import list_sessions
    from iris_harness.memory.retention import is_ephemeral_session

    return len(list_sessions(limit=1, skip=is_ephemeral_session))


def _write_marker(outcome: WelcomeOutcome, *, channel: str) -> None:
    """Record the welcome as done, atomically (a reader never sees half a file)."""
    path = welcome_marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    record: dict[str, object] = {
        "session_id": outcome.session_id,
        "at": outcome.at,
        "channel": channel,
        "response": outcome.response,
    }
    if outcome.skipped:
        record["skipped"] = True
        record["reason"] = outcome.skip_reason
    tmp.write_text(
        json.dumps(
            record,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


__all__ = [
    "WELCOME_OPENER",
    "FirstChatWelcome",
    "WelcomeOutcome",
    "capability_summaries",
    "existing_conversations",
    "load_wording",
    "mark_shown_by_setup",
    "welcome_marker_path",
    "welcome_text",
]
