"""Checks on a loop's final answer before the user sees it.

Two ways a small model's "final answer" is not an answer, both seen in the
2026-09-27 tool eval (qwen3.5:4b, thinking off):

* **An unbacked write claim.** The model's own thought said "I need to call
  complete_task", then it answered "Marked the expense report task as done." with no
  action: nothing was written. A final answer that claims a change while no tool that
  changes anything ran on this request is turned back once with that fact; if the next
  answer still claims it (and does not place it on an earlier request) it is replaced
  by an honest one. The claim phrases are ``config/write_claims.yaml``.
* **A scaffold echo.** The model returned the loop's own prompt text ("Your steps so
  far on this request …") or a raw ReAct line ("Thought: …") as its answer. That is
  checked against the loop's own markers, not vocabulary: turned back once, like a
  claim.

The loops (``AgenticCore._loop`` and ``run_stream``) call :func:`check_final_answer`
at their terminal branch and act on its verdict; the run's executed effects are the
evidence (``effects_executed`` records a non-read tool before it runs).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from iris_harness.foundation.paths import config_path

logger = logging.getLogger(__name__)

WRITE_CLAIMS_FILENAME = "write_claims.yaml"

UNBACKED_CLAIM_NOTE = (
    "Your answer says a change was made, but no tool that changes anything ran on this "
    "request, so nothing was saved. If it was made on an EARLIER request, say that "
    "plainly. Otherwise tell the user it was not done, or make exactly the change this "
    "request asks for — nothing from earlier requests or from memory."
)
# The request is quoted with the note: turned back without it, qwen3.5 completed a task
# it recalled from an earlier turn instead of the one asked for (2026-09-27 rerun).
_REQUEST_LINE = ' This request: "{request}".'
UNBACKED_CLAIM_ANSWER = (
    "I haven't made that change — nothing was saved. Ask me again and I'll do it."
)
SCAFFOLD_ECHO_NOTE = (
    "Your answer repeated your own working notes instead of answering. Reply with "
    "'Final Answer: <the answer for the user>', or call a tool."
)
# A final answer that starts with one of the loop's own line labels is working, not
# an answer (the loop's format, not vocabulary).
_REACT_LABEL_RE = re.compile(r"^\s*(?:thought|action|action input|observation)\s*:", re.I)
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


@dataclass(frozen=True)
class ClaimVocabulary:
    claim: re.Pattern[str] | None
    negation: re.Pattern[str] | None
    prior: re.Pattern[str] | None
    whole: frozenset[str] = frozenset()
    promise: re.Pattern[str] | None = None
    conditional: re.Pattern[str] | None = None


def _alternation(phrases: Iterable[object]) -> re.Pattern[str] | None:
    parts: list[str] = []
    for phrase in phrases:
        pieces = [
            r"\s+".join(re.escape(w) for w in piece.split())
            for piece in str(phrase).split("...")
            if piece.strip()
        ]
        if pieces:
            parts.append(r"\b" + r"\b.*?\b".join(pieces) + r"(?!\w)")
    if not parts:
        return None
    return re.compile("|".join(parts), re.IGNORECASE)


def load_claim_vocabulary(path: str | Path) -> ClaimVocabulary:
    """The YAML's phrase lists, compiled; empty patterns when absent or malformed."""
    try:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — missing/malformed file → no claims checked
        return ClaimVocabulary(None, None, None)
    if not isinstance(raw, dict):
        return ClaimVocabulary(None, None, None)

    def listed(key: str) -> list[object]:
        value = raw.get(key)
        return value if isinstance(value, list) else []

    return ClaimVocabulary(
        _alternation(listed("claims")),
        _alternation(listed("negations")),
        _alternation(listed("prior_markers")),
        frozenset(_bare(w) for w in listed("whole_answer_claims") if _bare(w)),
        _alternation(listed("promises")),
        _alternation(listed("conditionals")),
    )


@lru_cache(maxsize=1)
def _vocabulary() -> ClaimVocabulary:
    return load_claim_vocabulary(config_path(WRITE_CLAIMS_FILENAME))


def _bare(text: object) -> str:
    """Lower-cased, with surrounding punctuation and runs of space collapsed."""
    return " ".join(str(text).lower().split()).strip(" .!✅")


def claims_a_write(
    text: str, vocab: ClaimVocabulary | None = None, *, whole_answers: bool = True
) -> bool:
    """True when *text* says a change was made or is being made: a whole answer like
    "Done.", a sentence with a claim phrase, or — unless the answer ends on a question —
    a promise. A negated, conditional or question sentence is never a claim. ``whole_answers``
    off: a bare "Done." is not read as a claim (it can close a run of reads)."""
    vocab = vocab or _vocabulary()
    if whole_answers and _bare(text) in vocab.whole:
        return True
    asks = (text or "").rstrip().endswith("?")
    patterns = [p for p in (vocab.claim, None if asks else vocab.promise) if p is not None]
    for sentence in _SENTENCE_RE.findall(text or ""):
        if sentence.rstrip().endswith("?"):
            continue
        if vocab.conditional and vocab.conditional.search(sentence):
            continue  # what would happen, not what did
        if any(p.search(sentence) for p in patterns) and not (
            vocab.negation and vocab.negation.search(sentence)
        ):
            return True
    return False


def places_it_earlier(text: str, vocab: ClaimVocabulary | None = None) -> bool:
    vocab = vocab or _vocabulary()
    return bool(vocab.prior and vocab.prior.search(text or ""))


def echoes_scaffold(text: str, markers: Sequence[str]) -> bool:
    """True when *text* is the loop's own working: a ReAct label or a prompt marker."""
    stripped = (text or "").strip()
    if _REACT_LABEL_RE.match(stripped):
        return True
    return any(marker and marker in stripped for marker in markers)


@dataclass(frozen=True)
class Verdict:
    """``ok`` → use the answer; ``retry`` → turn it back with ``note``;
    ``replace`` → answer with ``text`` instead; ``fail`` → end the run unsuccessfully
    (a read-first turn that answered twice without reading)."""

    kind: str
    note: str = ""
    text: str = ""
    reason: str = ""


OK = Verdict("ok")


def check_final_answer(
    answer: str,
    *,
    effects_executed: Sequence[str],
    retries_used: int,
    markers: Sequence[str],
    fallback: str = "",
    any_tool_ran: bool = False,
    request: str = "",
) -> Verdict:
    """The verdict on a final answer. One turn-back per run (``retries_used``).

    A bare "Done." counts as a claim only when the run called no tool at all: after
    reads it may just close the turn."""
    if echoes_scaffold(answer, markers):
        if retries_used < 1:
            return Verdict("retry", note=SCAFFOLD_ECHO_NOTE, reason="scaffold_echo")
        return Verdict("replace", text=fallback, reason="scaffold_echo")
    if effects_executed or not claims_a_write(answer, whole_answers=not any_tool_ran):
        return OK
    if retries_used < 1:
        note = UNBACKED_CLAIM_NOTE
        if request.strip():
            note += _REQUEST_LINE.format(request=" ".join(request.split())[:300])
        return Verdict("retry", note=note, reason="unbacked_write_claim")
    if places_it_earlier(answer):
        return OK
    return Verdict("replace", text=UNBACKED_CLAIM_ANSWER, reason="unbacked_write_claim")


__all__ = [
    "SCAFFOLD_ECHO_NOTE",
    "UNBACKED_CLAIM_ANSWER",
    "UNBACKED_CLAIM_NOTE",
    "Verdict",
    "check_final_answer",
    "claims_a_write",
    "echoes_scaffold",
    "load_claim_vocabulary",
    "places_it_earlier",
]
