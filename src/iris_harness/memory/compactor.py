"""Conversation compaction with LLM-based summarization and hierarchical archival.

Two triggers, whichever fires first (Claude-Code-style auto-compaction):

  - **count** — more than ``compaction_threshold`` turns have accumulated.
  - **window** — the running history's estimated tokens reach ``compaction_ratio``
    (default 0.8) of ``token_budget``, the model's window-derived prompt budget. A
    few large turns can fill the window long before the count threshold is hit, so
    the token trigger is the load-bearing one for real conversations; the count
    threshold is a floor for tiny-turn chats where no window is wired.

On compaction the OLDEST turns are summarized (LLM, governance-wrapped, with an
extractive fallback) and the most-recent turns are kept verbatim — bounded by a
token sub-budget when the window is known, else by ``keep_recent`` count. Empty /
whitespace-only turns are dropped from the summary input so garbage is compressed
out, not preserved ("keep only the last *good* turns").
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from iris_harness.foundation.process_state import track_globals
from iris_harness.kernel.governance import HookContext, HookPoint, kernel_from_env
from iris_harness.kernel.governance.reentry import reenter_many, reenter_text
from iris_harness.kernel.governance.turn_label import apply_turn_floor
from iris_harness.llm.budget import estimate_tokens, trim_text
from iris_harness.llm.client import GovernedPromptCall

logger = logging.getLogger(__name__)


_SUMMARY_CONFIG_CACHE: dict[str, Any] | None = None

_SUMMARY_FALLBACK: dict[str, Any] = {
    "tier_intent": "memory_compaction",
    "max_tokens": 320,
    "sections": [
        {"label": "Goal", "hint": "what the user is trying to get done in this session"},
        {"label": "Decisions & outcomes", "hint": "what was agreed, found, or completed"},
        {"label": "Open items", "hint": "questions not answered and actions not done"},
        {"label": "Referenced", "hint": "people, organisations, files and amounts mentioned"},
    ],
    "prompt": (
        "Rewrite the running summary of this conversation so it covers the earlier "
        "summary AND the new exchanges. Keep specifics; invent nothing.\n\n"
        "Write exactly these sections, each on its own line:\n{sections}\n\n"
        'Write "none" after a label that has nothing. Stay under {max_words} words.\n\n'
        "--- EARLIER SUMMARY ---\n{previous}\n\n"
        "--- NEW EXCHANGES ---\n{exchanges}\n\n--- UPDATED SUMMARY ---\n"
    ),
}


def summary_config() -> dict[str, Any]:
    """Read ``config/memory/summary.yaml`` (cached), falling back to the built-in shape."""
    global _SUMMARY_CONFIG_CACHE
    if _SUMMARY_CONFIG_CACHE is not None:
        return _SUMMARY_CONFIG_CACHE
    from iris_harness.foundation.paths import config_path, default_config_dir

    path = config_path("memory", "summary.yaml")
    if not path.exists():
        path = default_config_dir() / "memory" / "summary.yaml"
    config = dict(_SUMMARY_FALLBACK)
    if path.exists():
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config.update(loaded)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s: %s — using the built-in summary shape", path, exc)
    _SUMMARY_CONFIG_CACHE = config
    return config


def reset_summary_config_cache() -> None:
    """Drop the cached summary config (tests, and a reload after editing the YAML)."""
    global _SUMMARY_CONFIG_CACHE
    _SUMMARY_CONFIG_CACHE = None


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str
    # Where an assistant turn came from (#145 step two): ``"external"`` when the run read
    # third-party text before answering, None when unknown. The owner's own turns have none.
    origin: str | None = None


def summary_flag(
    previous_flag: bool | None,
    *,
    had_summary: bool,
    folded: Sequence[ConversationTurn],
) -> bool | None:
    """The provenance flag of a summary rolled forward over ``folded`` turns (#145).

    ``True`` once any folded turn is external-origin or the summary already was: sticky, because
    a model-written paraphrase of third-party text cannot be told apart from the model's own
    words, so it is never un-marked (over-marking is the safe direction). ``False`` only when
    nothing earlier is in doubt (no previous summary, or one known ``False``) and every folded
    turn is known not to be external: the owner's own turns, and assistant turns the loop
    recorded as ``internal``. Anything else is ``None``, unknown, which is not enveloped.
    """
    if previous_flag is True or any(t.origin == "external" for t in folded):
        return True
    previous_known = (not had_summary) or previous_flag is False
    folded_known = all(t.role == "user" or t.origin == "internal" for t in folded)
    return False if previous_known and folded_known else None


@dataclass(frozen=True)
class CompactedHistory:
    """Result of compacting a conversation."""

    summary: str
    recent_turns: tuple[ConversationTurn, ...]
    archived_count: int
    # The turns that were summarized away — exposed so a caller can mine them for durable
    # learning signal before they leave the working window ("learn before you forget").
    archived_turns: tuple[ConversationTurn, ...] = ()
    # Telemetry (defaulted for backward compatibility with callers that ignore it).
    trigger: str = "none"  # "none" | "count" | "tokens"
    tokens_before: int = 0
    tokens_after: int = 0


def _dedupe_list_sections(summary: str) -> str:
    """Collapse repeats in the comma-separated sections (deterministic, not a re-prompt).

    A small local model asked for "people, organisations, files mentioned" returns
    "Petra Sutton, landlord, Petra Sutton, landlord, Petra Sutton, ..." often enough
    that it is worth fixing here rather than hoping the next prompt lands.
    """
    labels = {
        str(section.get("label", "")).strip().lower()
        for section in (summary_config().get("sections") or [])
        if isinstance(section, dict) and "," in str(section.get("hint", ""))
    }
    if not labels:
        return summary
    out: list[str] = []
    for line in summary.splitlines():
        label, sep, body = line.partition(":")
        if sep and label.strip().lower() in labels:
            seen: set[str] = set()
            items: list[str] = []
            for raw in body.split(","):
                item = raw.strip()
                if item and item.lower() not in seen:
                    seen.add(item.lower())
                    items.append(item)
            out.append(f"{label}: {', '.join(items)}" if items else line)
        else:
            out.append(line)
    return "\n".join(out)


class ConversationCompactor:
    """Compact conversation history via hierarchical truncation + optional LLM summarization.

    ``token_budget`` is the model's window-derived prompt-token budget (see
    ``iris_harness.llm.budget.budget_for``); when set, compaction fires once the running history
    reaches ``compaction_ratio`` of it, and the kept-verbatim window is sized by tokens
    rather than a fixed turn count. Left ``None`` (e.g. in unit tests), the controller
    falls back to the pure count-based behaviour and is byte-identical to before.
    """

    def __init__(
        self,
        compaction_threshold: int = 20,
        keep_recent: int = 10,
        llm_call: Callable[[str], str] | None = None,
        *,
        token_budget: int | None = None,
        compaction_ratio: float = 0.8,
    ) -> None:
        self.compaction_threshold = compaction_threshold
        self.keep_recent = keep_recent
        self._llm = llm_call
        self.token_budget = token_budget if token_budget and token_budget > 0 else None
        self.compaction_ratio = compaction_ratio if 0 < compaction_ratio < 1 else 0.8
        self._kernel = kernel_from_env()

    @staticmethod
    def _turn_tokens(turn: ConversationTurn) -> int:
        return estimate_tokens(f"{turn.role}: {turn.content}")

    def _history_tokens(self, turns: list[ConversationTurn]) -> int:
        return sum(self._turn_tokens(t) for t in turns)

    def _trigger(self, turns: list[ConversationTurn]) -> str:
        """Return which threshold (if any) the history has crossed.

        With a window wired, the TOKEN budget is the only trigger: ``_split`` sizes the
        kept window by tokens, so a count-triggered compaction archived nothing and
        summarized nothing — it just logged "compaction fired: archived=0" on every
        turn past the count. The count threshold stays as the floor for the no-window
        case (unit tests, embedders with no declared context).
        """
        if self.token_budget is not None:
            if self._history_tokens(turns) >= int(self.compaction_ratio * self.token_budget):
                return "tokens"
            return "none"
        if len(turns) > self.compaction_threshold:
            return "count"
        return "none"

    def needs_compaction(self, turns: list[ConversationTurn]) -> bool:
        return self._trigger(turns) != "none"

    def _split(
        self, turns: list[ConversationTurn]
    ) -> tuple[list[ConversationTurn], list[ConversationTurn]]:
        """Partition into (archive-to-summarize, recent-to-keep-verbatim).

        Token-aware when a window is wired: keep the most-recent turns whose cumulative
        estimate stays within half the compaction trigger, so post-compaction we land
        well under threshold even when single turns are huge (at least one turn is always
        kept). Else fall back to the fixed ``keep_recent`` tail.
        """
        if self.token_budget is not None:
            keep_budget = max(1, int(self.token_budget * self.compaction_ratio * 0.5))
            kept_rev: list[ConversationTurn] = []
            used = 0
            for turn in reversed(turns):
                tok = self._turn_tokens(turn)
                if kept_rev and used + tok > keep_budget:
                    break
                kept_rev.append(turn)
                used += tok
            recent = list(reversed(kept_rev))
        else:
            recent = turns[-self.keep_recent :]
        archive = turns[: len(turns) - len(recent)]
        return archive, recent

    def compact(
        self,
        turns: list[ConversationTurn],
        *,
        force: bool = False,
        previous_summary: str = "",
    ) -> CompactedHistory:
        """Compact turns: roll the summary forward, keep recent ones verbatim.

        ``previous_summary`` is the session's current summary. It is an INPUT now:
        each compaction used to overwrite it without reading it, so everything older
        than the last compaction was lost.

        ``force`` compacts even below the trigger (the on-demand "compact now" path) — but
        only if there is actually an older span to summarize; a short conversation with
        nothing to archive is still a no-op.
        """
        tokens_before = self._history_tokens(turns)
        trigger = self._trigger(turns)
        if trigger == "none" and not force:
            return CompactedHistory(
                summary="",
                recent_turns=tuple(turns),
                archived_count=0,
                trigger="none",
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )
        archive, recent = self._split(turns)
        if not archive:
            # Nothing old enough to summarize: swapping the window here would rewrite
            # state and log a compaction that compacted nothing.
            return CompactedHistory(
                summary=previous_summary,
                recent_turns=tuple(turns),
                archived_count=0,
                trigger="none",
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )
        if trigger == "none" and force:
            if not archive:
                # Nothing old enough to summarize — forcing wouldn't change anything.
                return CompactedHistory(
                    summary="",
                    recent_turns=tuple(turns),
                    archived_count=0,
                    trigger="none",
                    tokens_before=tokens_before,
                    tokens_after=tokens_before,
                )
            trigger = "manual"
        summary = (
            self._summarize(archive, previous_summary=previous_summary)
            if archive
            else previous_summary
        )
        recent_t = tuple(recent)
        tokens_after = (estimate_tokens(summary) if summary else 0) + sum(
            self._turn_tokens(t) for t in recent_t
        )
        return CompactedHistory(
            summary=summary,
            recent_turns=recent_t,
            archived_count=len(archive),
            archived_turns=tuple(archive),
            trigger=trigger,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

    def summarize_all(self, turns: list[ConversationTurn], *, previous_summary: str = "") -> str:
        """Summarize a WHOLE conversation — the closing roll for an idle session.

        ``compact`` only summarizes the span older than the kept window, so a short
        conversation (most of them) never got a summary and could never be cooled.
        Returns ``previous_summary`` unchanged when no summarizer is wired or it fails.
        """
        return self._summarize(turns, previous_summary=previous_summary)

    def _summarize(self, turns: list[ConversationTurn], *, previous_summary: str = "") -> str:
        """Roll ``previous_summary`` forward over ``turns``.

        With no LLM wired, or on any failure, the PREVIOUS summary is returned
        unchanged. The old extractive fallback (first 3 turns cut to 80 chars) ran in
        production for months and produced summaries like
        "assistant: 1. **Business News | Today's Inter…; user: what's happening in
        India today?" — worse than nothing, because it looked like a summary.
        """
        # Drop empty / whitespace-only turns — never spend summary budget on garbage.
        turns = [t for t in turns if t.content.strip()]
        if not turns:
            return previous_summary
        if self._llm is None:
            logger.info(
                "conversation summary skipped: no summarizer wired; keeping the previous summary"
            )
            return previous_summary
        try:
            prompt = self._summary_prompt(turns, previous_summary)
            summary = self._invoke_with_governance(prompt).strip()
        except Exception:
            logger.exception("conversation summary failed; keeping the previous summary")
            return previous_summary
        if not summary:
            return previous_summary
        summary = _dedupe_list_sections(summary)
        return trim_text(summary, max_chars=int(summary_config().get("max_tokens", 320)) * 4)

    @staticmethod
    def _summary_prompt(turns: list[ConversationTurn], previous_summary: str) -> str:
        config = summary_config()
        sections = config.get("sections") or []
        rendered = "\n".join(
            f"{s.get('label', '')}: <{s.get('hint', '')}>"
            for s in sections
            if isinstance(s, dict) and s.get("label")
        )
        max_tokens = int(config.get("max_tokens", 320))
        template = str(config.get("prompt") or _SUMMARY_FALLBACK["prompt"])
        # The summary the model writes is STORED, and a phrase scan at read time cannot match
        # a reworded instruction, so the text goes through the re-entry scan on the way IN
        # (issue #161): an assistant turn and the previous summary are scanned, the owner's own
        # turns are verbatim. Scanned, not enveloped: the model would copy the tags into the
        # stored summary, and its provenance is already tracked (``summary_flag``).
        scanned = reenter_many(
            [(t.role, t.content) for t in turns],
            reader="compactor",
            origin="transcript",
        )
        previous = previous_summary.strip()
        if previous:
            previous = reenter_text(
                previous, reader="compactor", origin="summary", role="assistant"
            ).text
        return template.format(
            sections=rendered,
            max_words=int(max_tokens * 0.75),
            previous=previous or "(none yet — this is the first roll)",
            exchanges="\n".join(f"{t.role}: {r.text}" for t, r in zip(turns, scanned, strict=True)),
        )

    # Legacy dict-based API retained for backward compatibility
    def compact_conversation(
        self, conversation_turns: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        typed = [
            ConversationTurn(role=str(t.get("role", "")), content=str(t.get("content", "")))
            for t in conversation_turns
        ]
        result = self.compact(typed)
        output: list[dict[str, object]] = []
        if result.summary:
            output.append(
                {
                    "type": "summary",
                    "content": result.summary,
                    "metadata": {"turn_count": result.archived_count},
                }
            )
        output.extend({"role": t.role, "content": t.content} for t in result.recent_turns)
        return output

    def summarize_turns(self, turns: list[dict[str, object]]) -> dict[str, object]:
        typed = [
            ConversationTurn(role=str(t.get("role", "")), content=str(t.get("content", "")))
            for t in turns
        ]
        summary = self._summarize(typed, previous_summary="")
        return {
            "type": "summary",
            "content": summary,
            "metadata": {"turn_count": len(turns), "original_turns": turns},
        }

    def validate_compaction(self, compacted_turns: list[dict[str, object]]) -> bool:
        if not compacted_turns:
            return False
        first = compacted_turns[0]
        return first.get("type") == "summary" and "metadata" in first and "turn_count" in first.get("metadata", {})  # type: ignore[operator]

    def _invoke_with_governance(self, prompt: str) -> str:
        assert self._llm is not None
        # A GovernedPromptCall governs itself, at the tier it goes to (llm/client.py).
        if self._kernel is None or isinstance(self._llm, GovernedPromptCall):
            return self._llm(prompt)

        run_id = str(uuid.uuid4())
        classify_ctx = HookContext(
            hook_point=HookPoint.PRE_CLASSIFY,
            run_id=run_id,
            agent_type="memory_compactor",
            payload={"prompt": prompt},
        )
        classify_decision, classified_ctx = self._kernel.fire_sync(
            HookPoint.PRE_CLASSIFY, classify_ctx
        )
        if classify_decision.outcome in ("deny", "require_approval"):
            raise RuntimeError(
                f"governance blocked conversation compaction call: {classify_decision.reason}"
            )

        llm_ctx = HookContext(
            hook_point=HookPoint.PRE_LLM_CALL,
            run_id=run_id,
            agent_type="memory_compactor",
            # Floored at the turn's label: this prompt may look tamer than the data the
            # turn holds (kernel/governance/turn_label.apply_turn_floor).
            classification=apply_turn_floor(classified_ctx.classification),
            # An opaque callable: where it sends the prompt is unknown, so the call is
            # governed as leaving the machine (fail closed), not guessed to be local.
            tier="tier_3",
            # The callable is opaque: the model behind it is not known here, and the row
            # says so rather than leaving the identity off. (A ``GovernedPromptCall``
            # governs itself and names its model.)
            payload={"prompt": prompt, "model": "unknown", "provider": "unknown"},
        )
        decision, _ = self._kernel.fire_sync(HookPoint.PRE_LLM_CALL, llm_ctx)
        if decision.outcome in ("deny", "require_approval"):
            raise RuntimeError(
                f"governance blocked conversation compaction call: {decision.reason}"
            )

        return self._llm(prompt)


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_SUMMARY_CONFIG_CACHE")
