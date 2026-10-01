"""``HarnessServices`` -- the capabilities the harness offers a plugin.

Defined here, in the runtime, because the composition root is what fills it: every
field is a handle the runtime owns or a callable it binds during ``build_runtime``.
A plugin author never imports this module -- ``iris_harness.sdk`` re-exports the class, and that is the name to import. It lives below the
SDK so the composition root can construct one without importing the layer above it
(OSS plan M6.2 layer 7).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from iris_harness.llm.tier_router import TierRouterService

if TYPE_CHECKING:
    from iris_harness.agent.agent_executor import AgentTask, HandlerResult, StreamChunk
    from iris_harness.memory.state.continuations import Continuation
    from iris_harness.runtime.tool_service import ToolCatalogue, ToolService
    from iris_harness.services.channels.models import ChannelMessage, DeliveryReceipt
    from iris_harness.services.heartbeat.models import HeartbeatDefinition, HeartbeatRun
    from iris_harness.services.learning.lesson_capture import Lesson
    from iris_harness.tools.skills.models import SkillPackage


# -- what each handle promises ------------------------------------------------------
#
# These were ``Any`` until the core/SDK boundary plan (PR 1). Each Protocol lists only
# the methods a plugin calls on the handle -- itself, or through the ``PluginAPI`` verb
# or core helper it hands the handle to -- so a third-party author can see what they
# may rely on, and a fake in their tests only has to supply those methods. A method
# the harness object has but no plugin calls is deliberately left out: listing it here
# would make it contract. ``iris_harness.sdk.services`` re-exports them; they are
# defined here, below the SDK, because the composition root types its fields with them
# (``TierRouterService`` lives with the router, where the helpers that take it are).


class AgentExecutorService(Protocol):
    """Where intent handlers live. ``PluginAPI.register_intent_handler`` calls these."""

    def register(self, agent_type: str, handler: Callable[[AgentTask], HandlerResult]) -> None:
        """Answer ``agent_type`` with ``handler``."""
        ...

    def register_stream(
        self, agent_type: str, handler: Callable[[AgentTask], Iterator[StreamChunk]]
    ) -> None:
        """The streaming variant of an ``agent_type`` handler."""
        ...


class HeartbeatService(Protocol):
    """The heartbeat scheduler."""

    def register_handler(self, name: str, handler: Callable[..., Any]) -> None:
        """Back heartbeat ``name`` with ``handler`` (``PluginAPI.register_heartbeat``)."""
        ...

    def register(self, definition: HeartbeatDefinition) -> bool:
        """Schedule ``definition``; False when it ends up disabled or unbound."""
        ...

    def trigger_by_name(self, name: str) -> HeartbeatRun | None:
        """Run ``name`` now -- the health watch's re-run repair (system plugin)."""
        ...


class ChannelService(Protocol):
    """The delivery gateway every channel connector registers on."""

    def register(self, connector: Any) -> None:
        """Add a connector (``PluginAPI.register_channel``)."""
        ...

    def channels(self) -> list[str]:
        """The registered channel names."""
        ...

    def broadcast(
        self, message: ChannelMessage, *, channels: list[str] | None = None
    ) -> list[DeliveryReceipt]:
        """Send ``message`` to ``channels`` (every registered one when None)."""
        ...


class EventBusService(Protocol):
    """The runtime's private event bus (``PluginAPI.subscribe`` / ``publish``)."""

    def on(self, event: str, handler: Callable[[Any], Any]) -> Any:
        """Run ``handler(payload)`` whenever ``event`` is emitted."""
        ...

    def emit_sync(self, event: str, payload: Any = None) -> None:
        """Emit ``event`` from synchronous code."""
        ...


class LessonService(Protocol):
    """Prior attempts at a similar task, and where a run's outcome is recorded."""

    def find_similar(self, query: str, k: int | None = None) -> list[Lesson]:
        """Up to ``k`` past lessons whose query overlaps ``query``."""
        ...

    def render_prior_lessons(self, lessons: list[Lesson]) -> str:
        """Those lessons, formatted for a prompt."""
        ...

    def handle(
        self,
        *,
        query: str,
        answer: str,
        artifacts: list[str] | tuple[str, ...] = (),
        session_id: str | None = None,
        iterations: int = 0,
        all_succeeded: bool = True,
    ) -> tuple[Lesson | None, str]:
        """Record a finished run; returns the lesson (if kept) and the cleaned answer."""
        ...


class ContinuationService(Protocol):
    """The session-owned continuation registry (ADR-0106 D1)."""

    def ask(
        self,
        session_id: str,
        owner: str,
        *,
        question: str = "",
        kind: str = "approval",
        intent: str = "",
        executor_kind: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Continuation:
        """Record that ``owner`` asked this session something and waits for the answer."""
        ...

    def pending(self, session_id: str) -> Continuation | None:
        """The session's open question, if any."""
        ...

    def answered(self, continuation_id: str) -> None:
        """Close a question: it was answered (or its asker moved on)."""
        ...


class SkillRegistryService(Protocol):
    """The loaded skill packages (briefs among them)."""

    def list_packages(
        self, *, agent_name: str | None = None, only_loadable: bool = False
    ) -> Sequence[SkillPackage]:
        """Discovered packages, optionally only one agent's or only the loadable ones."""
        ...


@dataclass
class HarnessServices:
    """The harness capabilities a plugin may use. Deliberately small.

    ``deterministic_reply`` builds a governed, signal-recorded ``ChatResult`` for a
    templated answer (what every deterministic intercept returns). The three
    subsystem sinks are the same objects the core registers into.
    """

    config_dir: Path
    data_dir: Path
    tier_router: TierRouterService
    agent_executor: AgentExecutorService
    heartbeats: HeartbeatService
    channels: ChannelService
    deterministic_reply: Callable[..., Any]
    # The runtime's private EventBus, as a guarded view (``GuardedEventBus``): topics the
    # harness owns (``approval.call_completed``) are refused on every verb. Subscribe
    # through ``PluginAPI.subscribe`` so the handler is inside the fault boundary.
    events: EventBusService | None = None
    # ``submit_activity(kind=, title=, work=, origin=, metadata=) -> Activity``:
    # hand a long job to the background Activity spine and return now. ``work`` is
    # ``work(progress) -> ActivityOutcome``, where ``progress(frac, message)``
    # streams onto the Activity row. Completion notices (in-chat + channel) are the
    # harness's job, not yours.
    submit_activity: Callable[..., Any] | None = None
    # The harness's own governed ReAct loop, for a plugin that wants to mount a
    # PERSONA rather than supply a handler: pass these straight to
    # ``register_intent_handler`` and the agent type answers on the shared loop,
    # biased by intent, with its tools coming from its skill packs. ``None`` when
    # the loop is off (``IRIS_AGENTIC_CORE`` not on/shadow) — check before using.
    react_handler: Callable[..., Any] | None = None
    react_stream_handler: Callable[..., Any] | None = None
    # The message the current turn is answering, or "" outside a turn. A plugin tool
    # is registered once at setup and cannot close over a per-turn query the way a
    # core-built tool does, so this is how it reads the question it is answering
    # (OSS plan M4.2). Backed by a ContextVar — safe under concurrent requests.
    current_query: Callable[[], str] = field(default_factory=lambda: (lambda: ""))
    # The session the current turn belongs to, or "" outside a turn — read beside
    # ``current_query`` by a plugin tool that asks the user something and must record
    # the question against this conversation (``continuations.ask``). ContextVar-backed.
    current_session_id: Callable[[], str] = field(default_factory=lambda: (lambda: ""))
    # ``deliver_in_chat(session_id, text)``: append a proactive notice to a chat
    # session's history — the in-chat leg a finished Activity already takes. It is
    # what a CHAT-SURFACE channel plugin delivers into (OSS plan M4.5); None when
    # no runtime is wired (tests), and a surface plugin must check before using it.
    deliver_in_chat: Callable[[str, str], None] | None = None
    # The harness's shared text embedder (MiniLM — the same vectors memory recall
    # uses), for a plugin that does semantic work of its own: the research tool's
    # blended rerank is the first caller. None when no semantic index is wired, and
    # callers must degrade rather than fail (rerank falls back to lexical).
    embed: Callable[[list[str]], list[list[float]]] | None = None
    # The harness's lesson store (``iris_harness.services.learning.lesson_capture``): prior
    # attempts at a similar task, and where a run's outcome is recorded. An agent
    # plugin primes itself from it and writes back through it; the code-exec agent
    # is the first caller. ``None`` when lesson capture is off
    # (``IRIS_LESSON_CAPTURE_ENABLED=0``), and callers must run without it.
    lessons: LessonService | None = None
    # The session-owned continuation registry (ADR-0106 D1). A plugin that parks a
    # proposal and asks for approval records the question here — ``ask(session_id,
    # <its intercept name>, question=, intent=)`` — and closes it with ``answered``
    # or ``drop`` when the proposal is decided or withdrawn. The harness then shields
    # every OTHER confirmation intercept from the answer, which is the guarantee the
    # file organizer lacked when it claimed a "yes" the planner was waiting on.
    # Opening one is not a privilege the core keeps to itself: agents and plugins
    # use the same object. None when no runtime is wired (tests) — check first.
    continuations: ContinuationService | None = None
    # The shared intent router's classifier: ``classify_intent(message)`` returns the
    # router's result (``.intent``, ``.source``) or raises. A plugin whose intercept
    # has a semantic gate reads the ONE router the harness routes with, instead of
    # shipping a classifier of its own — the calendar plugin's meeting_creation gate
    # is the caller (OSS plan M5.7 track A). None when no runtime is wired (tests).
    classify_intent: Callable[[str], Any] | None = None
    # The skill registry: the loaded skill packages (briefs among them). A plugin that
    # composes over skills — the planner renders and configures the brief skill — reads
    # the one registry the harness discovered with. None when no runtime is wired.
    skill_registry: SkillRegistryService | None = None
    # ``conversation_in_flight(session_id) -> bool``: True while a harness-owned
    # multi-turn conversation (routine authoring today) is mid-flight in the session, so
    # an on-demand intercept can step aside and let the continuation turn reach it.
    conversation_in_flight: Callable[[str], bool] | None = None
    # ``default_channel()``: the validated default delivery channel name, read at call
    # time because channel plugins mount after the heartbeats register and the harness
    # pins the default only once every channel is known. None when no runtime is wired.
    default_channel: Callable[[], str] | None = None
    heartbeat_diagnostics: Callable[[], list[Any]] = field(default_factory=lambda: (lambda: []))
    # Registered tools for code, run through the same governed runner as the model's
    # (plugin-capabilities step 2). The core asks ``tools.for_caller("core:<workflow>")``;
    # a plugin uses ``api.tools``, already bound to its own name. None when no runtime.
    # The core's services hold the ToolService; a plugin's view holds its ToolCatalogue
    # (describe only) -- a plugin calls tools through the bound ``api.tools``.
    tools: ToolService | ToolCatalogue | None = None


__all__ = [
    "AgentExecutorService",
    "ChannelService",
    "ContinuationService",
    "EventBusService",
    "HarnessServices",
    "HeartbeatService",
    "LessonService",
    "SkillRegistryService",
    "TierRouterService",
]
