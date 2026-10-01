"""ContinuationRegistry — the harness's answer to "who is this turn's reply for?".

ADR-0106 M5.C2. The store (``governance.checkpoint.continuations``) keeps rows
honest; this is the policy over it, and the object agents and plugins are handed.

The incident it exists to prevent: a research plan's *"Would you like to proceed
with this plan?"* was answered **"yes go head with the plan"**, and the file
organizer's approval intercept claimed the turn because it could see that *an*
approvable thing existed somewhere. Nothing in the harness could see *who had
actually asked*. A continuation is that missing fact.

Two rules carry most of the weight:

- **One pending continuation per session** (ADR-0106 decision 5). This is doing
  more work than it looks: with at most one open question per conversation, "who
  owns this answer" is unambiguous *by construction* rather than by arbitration.
  :meth:`ask` supersedes before opening, which is the composition the store
  deliberately refuses to do on its own.
- **Never across sessions.** A continuation the user did not see in *this*
  conversation is not one they can be answering, however few are open globally.
  The store enforces it; this layer never widens the query.

Deliberately not here: *when* a pending continuation should claim a turn. That
depends on the intercept chain's ordering and belongs with the intercept in M5.C3.
This module answers only "does this message read as an answer to this question?",
and only for ``approval`` continuations, where a yes/no is the whole vocabulary.
Free-text ``question`` continuations need the chain to reason about whether the
user answered or changed the subject, which is a C3 problem with C3's tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from iris_harness.memory.state.continuations import (
    Continuation,
    ContinuationStore,
)
from iris_harness.runtime.nlu_parsing import _parse_confirmation_decision

__all__ = [
    "ContinuationRegistry",
    "choice_request",
    "ends_on_a_question",
    "offered_choices",
    "offers_to_proceed",
    "question_request",
    "reads_as_answer",
    "reads_as_choice",
]


# An agent that ends its answer by offering to do the next thing is waiting for a
# reply, whether or not it knows it. Deliberately narrow: it must END on a question
# AND that question must offer to act. A merely inquisitive answer ("what else is on
# your mind?") opens nothing, because an open continuation shields the confirmation
# intercepts for the next turn, and shielding on a rhetorical flourish would be worse
# than the problem. Widen this only with a failing case in hand.
_PROCEED_OFFER_RE = re.compile(
    r"\b(?:"
    r"would you like (?:me )?to|would you like|"
    r"shall i|shall we|should i|"
    r"do you want (?:me )?to|want me to|"
    r"proceed with|go ahead with|"
    r"ready (?:for me )?to (?:start|proceed|begin)"
    r")\b",
    re.IGNORECASE,
)
# Only the closing stretch is considered — an offer made in paragraph two of a long
# answer is context, not the question the answer leaves hanging.
_OFFER_TAIL_CHARS = 240


def ends_on_a_question(answer: str) -> bool:
    """True when an answer ends on a question — the only phrasing-free signal there is.

    This is what decides whether a continuation opens at all. :func:`offers_to_proceed`
    then decides only what KIND it is.

    It used to be the other way round, and that was the defect. Opening required the
    answer to match a list of offer phrasings, so each session that phrased a question
    a new way went unowned and its follow-ups were classified from scratch. Measured on
    2026-09-16, three of one thread's four closing questions opened nothing:

        "Do you want full articles, more sources, or a short briefing on any one story?"
        "Tell me which and I'll fetch them."
        "Reply with answers 1-5."

    The third reply in that thread reached the inbox. A natural language has unbounded
    ways to ask a question and exactly one way to punctuate it, so the punctuation is
    the rule and the phrasing is not.
    """
    return (answer or "").strip().endswith("?")


def offers_to_proceed(answer: str) -> bool:
    """True when an answer ends by offering to take the next step.

    Deterministic and model-free, like every other pre- and post-loop decision in
    the harness. No longer decides WHETHER a continuation opens (see
    :func:`ends_on_a_question`) — only whether the one that opens is a yes/no
    ``approval`` rather than the free-text ``question`` that is now the default.
    """
    text = (answer or "").strip()
    if not ends_on_a_question(text):
        return False
    return bool(_PROCEED_OFFER_RE.search(text[-_OFFER_TAIL_CHARS:]))


# An offer that asks the user to pick among what was just listed, as opposed to one that
# asks whether to go ahead with it. The difference decides the continuation's kind: a
# research plan's numbered steps ending "Would you like to proceed with this plan?" is
# an approval (the ADR-0106 incident, answered "yes go head with the plan"), while five
# headlines ending "Want me to open any of these?" is a choice (the 2026-09-16 session,
# answered "4"). Read from the closing question only, so an item that happens to say
# "select" never turns a plan into a menu. Widen only with a failing case in hand.
_SELECTION_OFFER_RE = re.compile(
    r"\b(?:any|one|each|either) of (?:these|them|those|the above)\b|"
    r"\bwhich (?:one|of|item|article|story|result)s?\b|"
    r"\b(?:pick|choose|select)\b",
    re.IGNORECASE,
)
_LIST_ITEM_RE = re.compile(r"^\s*(?:\*\*)?(\d{1,2})[.)](?:\*\*)?\s+(\S.*?)\s*$")
_OPTION_TEXT_MAX_CHARS = 300


def offered_choices(answer: str) -> tuple[str, ...]:
    """The numbered items an answer asks the user to pick from, in the order shown.

    Empty unless the answer ends by offering to act (:func:`offers_to_proceed`), that
    closing question asks for a pick, and a list numbered 1..N (N >= 2) precedes it.
    Detail lines between items are allowed; a break in the numbering starts over, so
    the options are the last complete list — the one the question is about.

    This records what the agent put on screen, when it put it there. It is not a
    re-reading of the conversation at pick time: the next reply resolves against the
    options stored now, exactly as a tool's shortlist does.
    """
    text = (answer or "").strip()
    if not ends_on_a_question(text):
        return ()
    question = text.splitlines()[-1]
    if _LIST_ITEM_RE.match(question) or not _SELECTION_OFFER_RE.search(question):
        return ()
    items: list[str] = []
    for line in text.splitlines():
        match = _LIST_ITEM_RE.match(line)
        if match is None:
            continue
        position, item = int(match.group(1)), match.group(2)[:_OPTION_TEXT_MAX_CHARS]
        if position == 1:
            items = [item]
        elif items and position == len(items) + 1:
            items.append(item)
        else:
            items = []
    return tuple(items) if len(items) >= 2 else ()


def choice_request(continuation: Continuation, option: dict[str, Any], reply: str) -> str:
    """The task an owner receives for a pick from a list its answer showed.

    An option recorded from an answer carries the item's text, not an id the owner can
    act on directly — so the owner is told which item, from which offer, in words it
    can act on. The reply alone ("4") is not a task: the loop asked for "4" invents
    which item was meant, and the goal-drift guard, measuring every thought against
    "4", halts the run. Options a tool recorded with ids of its own carry no ``text``
    and reach their owner untouched on ``AgentTask.selected_choice``.
    """
    item = str(option.get("text") or "").strip()
    if not item:
        return reply
    offer = (continuation.question or "").strip()
    lines = [f"Item {option.get('position')} from the numbered list in your last answer: {item}"]
    if offer:
        lines.append(f"You offered: {offer}")
    lines.append(f"The user replied: {reply}")
    return "\n".join(lines)


def question_request(continuation: Continuation, reply: str) -> str:
    """The task an owner receives for a reply to a free-text question it asked.

    The same shape as :func:`choice_request` and for the same reason: the reply alone
    ("Key quotes") is not a task. The loop asked for "Key quotes" invents what they are
    quotes OF, and the goal-drift guard, measuring every thought against two words,
    halts the run. The question carries the subject; the reply only selects within it.
    """
    question = (continuation.question or "").strip()
    if not question:
        return reply
    return f"You asked: {question}\nThe user replied: {reply}"


def reads_as_answer(continuation: Continuation, message: str) -> str | None:
    """``"approve"`` / ``"reject"`` if ``message`` answers ``continuation``, else None.

    Deterministic and model-free, like every other pre-loop decision in the harness.
    Reuses the core's existing confirmation parser rather than adding a third
    affirmation vocabulary next to it and the file organizer's — the parser anchors
    at the start of the message, so "yes go head with the plan" reads as approve
    while "the plan looks yes-ish to me" does not.

    Returns None for a ``question`` continuation: a free-text reply cannot be
    classified this way, and deciding whether such a turn is an answer or a change
    of subject is the intercept's call (M5.C3), not this function's.
    """
    if continuation.kind != "approval":
        return None
    return _parse_confirmation_decision(message)


# A reply that picks one numbered option. The whole message must reduce to exactly one
# ordinal plus words that only frame a pick ("just go with the 1 st one", "#2", "the
# second email") — anything else stays unclaimed, so "show my top 2 holdings" asked
# while a shortlist is open is a new question, not option 2. Widen the filler set only
# with a failing case in hand: every word added is one more way to mistake a new
# question for an answer.
_ORDINAL_WORDS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
_PICK_FILLER = frozenset(
    {
        "a", "an", "the", "one", "that", "this", "it", "is",
        "just", "go", "going", "with", "for", "take", "pick", "choose", "select", "use",
        "read", "open", "show", "me", "please", "pls", "i", "want", "need", "like", "d",
        "ll", "lets", "let", "s", "us", "ok", "okay", "yes", "yeah", "sure", "then",
        "number", "no", "num", "option", "choice", "item",
        "email", "mail", "message", "result", "entry",
        # A pick said as a sentence about the list (2026-09-16: "Just tell me more about
        # the 4th news from your news list"), and the nouns a listed item goes by.
        "tell", "more", "about", "details", "explain", "from", "your", "list", "of",
        "those", "these", "news", "article", "story", "link",
        # The rest of the ways the same pick is said: the prepositions the phrasing hangs
        # on, the question framing, and the verbs that only ask the owner to go deeper on
        # an item already on screen. A miss here is not benign — the reply stops reading
        # as a pick, the choice is withdrawn, and the message is planned as a fresh
        # question, which is the 2026-09-16 failure ("3" routed away from the news list).
        "on", "into", "to", "can", "you", "what", "do", "know", "give",
        "dig", "expand", "elaborate", "summarize", "summarise", "summary", "detail",
        "headline",
        # Plural nouns stay OUT on purpose: "give me 3 headlines" is a quantity, and with
        # "headlines" as filler it would read as headline 3. Only the singular a picked
        # item goes by is safe.
    }
)  # fmt: skip
# "more" after a number counts rather than picks: "show me 5 more" asks for five more,
# not for item 5. Only matters because "more" is filler for "tell me more about the 4th".
_QUANTITY_AFTER = frozenset({"more"})
_PICK_TOKEN_RE = re.compile(r"#?\d+(?:st|nd|rd|th)?|[a-z]+")
_DIGIT_SUFFIX_GAP_RE = re.compile(r"(\d)\s+(st|nd|rd|th)\b")


def _ordinal_of(word: str, count: int) -> int | None:
    digits = re.fullmatch(r"#?(\d+)(?:st|nd|rd|th)?", word)
    if digits:
        return int(digits.group(1))
    return count if word == "last" else _ORDINAL_WORDS.get(word)


def reads_as_choice(continuation: Continuation, message: str) -> int | None:
    """The 0-based index ``message`` picks from a ``choice`` continuation, else None.

    Deterministic and model-free, like :func:`reads_as_answer`. Returns None for any
    other kind, for a reply naming no option or more than one, for an ordinal outside
    the options shown, and for a reply carrying words that do more than frame a pick.
    """
    options = continuation.choices
    if not options:
        return None
    text = _DIGIT_SUFFIX_GAP_RE.sub(r"\1\2", (message or "").lower())
    picked: int | None = None
    words = _PICK_TOKEN_RE.findall(text)
    for at, word in enumerate(words):
        ordinal = _ordinal_of(word, len(options))
        if ordinal is not None:
            if picked is not None:
                return None  # "1 and 3" is not one choice
            if at + 1 < len(words) and words[at + 1] in _QUANTITY_AFTER:
                return None  # "5 more" is a quantity
            picked = ordinal
        elif word not in _PICK_FILLER:
            return None
    if picked is None or not 1 <= picked <= len(options):
        return None
    return picked - 1


@dataclass
class ContinuationRegistry:
    """Session-scoped record of the question this conversation owes an answer to."""

    store: ContinuationStore = field(default_factory=ContinuationStore)

    def ask(
        self,
        session_id: str,
        owner: str,
        *,
        question: str = "",
        kind: str = "approval",
        intent: str = "",
        run_id: str | None = None,
        step_id: int | None = None,
        executor_kind: str | None = None,
        payload: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> Continuation:
        """Record that ``owner`` has asked ``session_id`` something and is waiting.

        ``executor_kind`` + ``payload`` make the question *executable*: the action to run
        if the answer is yes, and the registered ConfirmationExecutor that knows how. The
        payload is durable and opaque — this layer owns who is waiting and answers
        outliving the process, never what the action means.

        Supersedes whatever this session was already asked — the one-pending rule is
        a partial unique index in the store, so opening without superseding raises
        rather than quietly creating two owners for one "yes". Asking again is the
        normal case (an agent re-prompts, a plugin re-proposes), so it must not be
        the caller's job to remember.
        """
        self.store.supersede(session_id, now=now)
        return self.store.open(
            session_id=session_id,
            owner=owner,
            kind=kind,
            question=question,
            intent=intent,
            run_id=run_id,
            step_id=step_id,
            executor_kind=executor_kind,
            payload=payload,
            now=now,
        )

    def pending(self, session_id: str, *, now: datetime | None = None) -> Continuation | None:
        """This session's open question, or None. Past-TTL rows expire on read."""
        return self.store.pending_for(session_id, now=now)

    def answered(self, continuation_id: str, *, now: datetime | None = None) -> None:
        """Close a continuation because the user answered it."""
        self.store.set_status(continuation_id, "answered", now=now)

    def drop(self, session_id: str, *, now: datetime | None = None) -> int:
        """Withdraw this session's pending question without an answer.

        For an owner that no longer needs one — the plan was executed by another
        surface, the agent resolved it alone. Marked superseded, never deleted: a
        question the user was actually asked stays on the conversation's record.
        """
        return self.store.supersede(session_id, now=now)

    def opened_since(
        self, session_id: str, since: datetime, *, now: datetime | None = None
    ) -> Continuation | None:
        """This session's pending continuation if it was opened at or after ``since``."""
        pending = self.pending(session_id, now=now)
        if pending is None:
            return None
        try:
            created = datetime.fromisoformat(pending.created_at)
        except ValueError:
            return None
        return pending if created >= since else None

    def drop_stale_inferred(
        self, session_id: str, since: datetime, *, now: datetime | None = None
    ) -> bool:
        """Withdraw an inferred question opened before ``since`` — the reply did not answer it.

        Covers both kinds the harness *infers* from an answer: a ``choice`` and a
        non-resumable ``question``. Each is answered by the next reply or not at all (see
        the classify stage). A turn an intercept answers never reaches classify, so the
        intercept stage calls this to keep that rule on every path — without it an
        inferred question would survive the intercepted turn and claim the turn after it,
        which is the two-turn shield the one-turn rule exists to prevent.

        Left alone: anything opened during this turn, an ``approval`` (its yes/no
        vocabulary is narrow enough to stay open across an unrelated turn) and a
        resumable Tier-B question (it owns a halted run, not a guess).
        """
        pending = self.pending(session_id, now=now)
        if pending is None:
            return False
        inferred = pending.kind == "choice" or (
            pending.kind == "question" and not pending.is_resumable_run
        )
        if not inferred:
            return False
        if self.opened_since(session_id, since, now=now) is not None:
            return False
        self.drop(session_id, now=now)
        return True

    def history(self, session_id: str) -> tuple[Continuation, ...]:
        """Every continuation this session has had, oldest first (for inspection)."""
        return self.store.history_for(session_id)
