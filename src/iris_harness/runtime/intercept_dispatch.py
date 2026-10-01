"""Intercept dispatch — the deterministic short-circuits that end a turn before any model.

The chain is declared in ``config/intercepts.yaml`` (``host.intercept_chain``) and joined
by plugin-registered intercepts; ``effective_chain`` resolves it, ``dispatch`` runs it
(first match wins, with the ADR-0106 shield), and ``activity_hint`` asks it for the
status text to stream before a slow intercept. Both ``chat()`` and ``chat_stream()``
reach it through the turn pipeline's intercept stage, so the chain, its order and its
observability cannot drift between the two paths.

Carved out of ``IrisRuntime`` at OSS plan M5.7 track C slice 15 as
``InterceptDispatch(host)``, held as ``runtime.intercepts``. Stateless.
:class:`InterceptDispatchHost` declares the runtime members read; the host is read
**at call time**, not captured. The host object itself is also what a core ``handler:``
row resolves against (``confirmations.handle_confirmation_turn`` walks the runtime), so a
row names where the handler lives on the runtime, never on this collaborator.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from iris_harness.foundation.observability.session_log import log_timeline_event
from iris_harness.runtime.intercepts import InterceptHit, InterceptSpec, resolve_runtime_handler

if TYPE_CHECKING:
    from iris_harness.runtime.continuations import ContinuationRegistry
    from iris_harness.runtime.plugin_host.registry import PluginRegistry

logger = logging.getLogger(__name__)


class InterceptDispatchHost(Protocol):
    """The five runtime members intercept dispatch reaches."""

    continuations: ContinuationRegistry
    intercept_chain: tuple[InterceptSpec, ...]
    # The ``openers:`` rows of config/intercepts.yaml, by name (ADR-0127).
    openers: dict[str, InterceptSpec]
    plugin_registry: PluginRegistry
    # iris_harness.runtime.plugin_host.EffectiveProfile once build_runtime mounts plugins, else None;
    # only ``intercept_order`` is read, through getattr, as it always was.
    profile: Any


class InterceptDispatch:
    """Resolves and runs the intercept chain for one runtime. See the module docstring."""

    def __init__(self, host: InterceptDispatchHost) -> None:
        self._host = host

    def dispatch(
        self,
        message: str,
        *,
        session_id: str,
        channel: str,
        span: Any,
    ) -> InterceptHit | None:
        """Run the declared intercept chain; return the first match or None.

        Single source of truth for the deterministic short-circuits — both
        chat() and chat_stream() call this so the chain, its order, and the
        `<name>.end` observability can never drift between the two paths. The
        chain itself is declared in config/intercepts.yaml (host.intercept_chain);
        plugin-registered intercepts join it via ``effective_chain``.
        """
        pending = self._host.continuations.pending(session_id)
        for spec, handler in self.effective_chain():
            # ADR-0106 shield. A confirmation-resolving intercept speaks the same
            # vocabulary as every other pending decision — a bare "yes" — so when
            # this session's open question belongs to someone else, it must not get
            # to answer it. This is the incident, generalised: the organize
            # intercept claimed a "yes" the planner was waiting on, because nothing
            # in the chain could see who had asked. An intercept that owns the
            # pending continuation still runs; so does everything non-confirming.
            if spec.resolves_confirmation and pending is not None and pending.owner != spec.name:
                logger.debug(
                    "intercept %r shielded: session %s owes an answer to %r",
                    spec.name,
                    session_id,
                    pending.owner,
                )
                continue
            kwargs: dict[str, Any] = {"session_id": session_id, "span": span}
            if spec.passes_channel:
                kwargs["channel"] = channel
            result = handler(message, **kwargs)
            if result is not None:
                log_timeline_event(
                    "pipeline.phase",
                    phase=f"{spec.name}.end",
                    payload={"matched": True},
                )
                return InterceptHit(spec=spec, result=result)
        return None

    def opener(self, name: str) -> InterceptSpec | None:
        """The declared opener ``name`` (ADR-0127), or None when none is declared."""
        return self._host.openers.get(name)

    def open(self, name: str, *, session_id: str, channel: str, span: Any) -> InterceptHit | None:
        """Run opener ``name`` for a turn the system opened; None when it gives no answer.

        An opener takes no message — there is none — so it is called with the session and
        the span (and the channel, when its row says ``passes_channel``). Core rows only:
        a ``plugin:`` opener has no registration seam yet, so it resolves to nothing.
        """
        spec = self.opener(name)
        if spec is None or spec.handler.startswith("plugin:"):
            return None
        handler = resolve_runtime_handler(self._host, spec.handler)
        if handler is None:
            logger.warning("opener %r → unknown handler %r", name, spec.handler)
            return None
        kwargs: dict[str, Any] = {"session_id": session_id, "span": span}
        if spec.passes_channel:
            kwargs["channel"] = channel
        result = handler(**kwargs)
        if result is None:
            return None
        return InterceptHit(spec=spec, result=result)

    def effective_chain(self) -> tuple[tuple[InterceptSpec, Callable[..., Any]], ...]:
        """The ordered (spec, handler) chain: declared YAML rows + plugin intercepts.

        Resolution per declared row, in YAML order:
          * a plugin registered an intercept of that name → the plugin's handler
            (its trace metadata wins; the row only pins position + enablement);
          * ``handler: plugin:<name>`` with nothing registered → declared-only,
            skipped with a warning (the drift panel shows it);
          * otherwise the runtime method named by ``handler`` — a dotted name reaches a
            collaborator's method (``routines.handle_routine_authoring_turn``).
        Plugin intercepts not declared in YAML run after the declared ones, in
        registration order. A profile ``intercept_order`` lists names to move to
        the front, in that order (OSS plan decision 4).
        """
        resolved: list[tuple[InterceptSpec, Callable[..., Any]]] = []
        seen: set[str] = set()
        for spec in self._host.intercept_chain:
            registered = self._host.plugin_registry.intercept(spec.name)
            if registered is not None:
                # The plugin's spec wins for trace metadata, but the safety flags are
                # properties the core also gets to assert. `resolves_confirmation`
                # (ADR-0106) decides whether dispatch may let this handler answer a
                # "yes" that belongs to someone else; `guard_output` (parity decision B)
                # whether the output guard checks its answer. Either side saying yes is
                # enough — a safeguard must not be droppable by a plugin that forgot to
                # declare it.
                merged = replace(
                    registered.spec,
                    resolves_confirmation=(
                        registered.spec.resolves_confirmation or spec.resolves_confirmation
                    ),
                    guard_output=registered.spec.guard_output or spec.guard_output,
                )
                resolved.append((merged, registered.handler))
                seen.add(spec.name)
                continue
            if spec.handler.startswith("plugin:"):
                logger.warning(
                    "intercept %r is declared for %s but no plugin registered it; skipping",
                    spec.name,
                    spec.handler,
                )
                continue
            # Core rows name a handler on the runtime, not on this collaborator.
            handler = resolve_runtime_handler(self._host, spec.handler)
            if handler is None:
                logger.warning(
                    "intercept %r → unknown handler %r; skipping", spec.name, spec.handler
                )
                continue
            resolved.append((spec, handler))
            seen.add(spec.name)
        for registered in self._host.plugin_registry.intercepts():
            if registered.spec.name not in seen:
                resolved.append((registered.spec, registered.handler))
                seen.add(registered.spec.name)
        order = list(getattr(self._host.profile, "intercept_order", None) or ())
        if order:
            rank = {name: i for i, name in enumerate(order)}
            resolved.sort(key=lambda item: rank.get(item[0].name, len(rank)))
        return tuple(resolved)

    def activity_hint(self, message: str) -> str | None:
        """Status text to stream before a deterministic intercept that may take seconds.

        An intercept that walks a folder and embeds its files runs synchronously, and
        without a hint the stream is silent for that long. The hint belongs to the
        intercept that is about to be slow, so it is registered alongside it
        (``api.register_intercept(..., activity_hint=...)``) rather than as a kind of
        its own. Asked in chain order; first answer wins. The core registers none of
        its own — the FileManager hints moved out with the plugin (OSS plan M2.4).
        """
        for spec, _handler in self.effective_chain():
            registered = self._host.plugin_registry.intercept(spec.name)
            if registered is None or registered.activity_hint is None:
                continue
            hint = registered.activity_hint(message)
            if hint:
                return hint
        return None
