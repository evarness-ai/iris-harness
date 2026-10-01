"""Semantic intent classifier + YAML anchor loader (model-free via a fake embedder)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from iris_harness.agent.intent_router import (
    IntentResult,
    SemanticIntentClassifier,
    load_intent_anchors,
)

# One-hot 3-D vectors keyed by a coarse domain cue → deterministic cosine.
_EMAIL_VEC = [1.0, 0.0, 0.0]
_CAL_VEC = [0.0, 1.0, 0.0]
_OTHER_VEC = [0.0, 0.0, 1.0]


def _fake_embedder(text: str) -> Sequence[float]:
    low = text.lower()
    if "email" in low or "inbox" in low:
        return _EMAIL_VEC
    if "calendar" in low or "meeting" in low or "schedule" in low:
        return _CAL_VEC
    return _OTHER_VEC


_ANCHORS = {
    "communication": ["check my email", "summarize my inbox"],
    "calendar": ["what's on my calendar", "schedule a meeting"],
}


class _Fallback:
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, query: str, *, context: str | None = None) -> IntentResult:
        self.calls += 1
        return IntentResult(
            intent="general",
            agent_type="system",
            confidence=0.5,
            raw_query=query,
            source="fallback",
        )


def _clf(fb: _Fallback, embedder=_fake_embedder):  # type: ignore[no-untyped-def]
    return SemanticIntentClassifier(embedder, _ANCHORS, fallback=fb, threshold=0.42, margin=0.05)


def test_confident_match_decides_and_skips_fallback() -> None:
    fb = _Fallback()
    r = _clf(fb).classify("any new email in my inbox?")
    assert r.intent == "communication"
    assert r.agent_type == "email"  # derived from _INTENT_TO_AGENT
    assert r.source == "semantic"
    assert fb.calls == 0


def test_ambiguous_defers_to_fallback() -> None:
    fb = _Fallback()
    # "other" query matches neither intent's anchors → below threshold → fallback.
    r = _clf(fb).classify("tell me a joke about cats")
    assert r.source == "fallback"
    assert fb.calls == 1


def test_embedder_failure_defers_to_fallback() -> None:
    fb = _Fallback()

    def boom(_t: str) -> Sequence[float]:
        raise RuntimeError("model down")

    r = _clf(fb, boom).classify("any new email?")
    assert r.source == "fallback"
    assert fb.calls == 1


def test_empty_query_defers_to_fallback() -> None:
    fb = _Fallback()
    assert _clf(fb).classify("   ").source == "fallback"
    assert fb.calls == 1


def test_multi_step_flag_derived_from_regex() -> None:
    fb = _Fallback()
    r = _clf(fb).classify("check my email and then summarize my inbox")
    assert r.source == "semantic" and r.intent == "communication"
    assert r.is_multi_step is True


# ── YAML anchor loader ───────────────────────────────────────────────────────


def test_calendar_create_vs_read_distinction() -> None:
    # The unified router distinguishes calendar *create* from calendar *read* so the
    # create short-circuit can be driven by the one router classifier.
    def cal_embedder(text: str) -> Sequence[float]:
        low = text.lower()
        # read cues FIRST so "show me my schedule" maps to read, not create (real
        # MiniLM separates them; this fake just mimics the intent split).
        if any(w in low for w in ("what", "show", "when", "do i have")):
            return [0.0, 1.0, 0.0]
        if any(w in low for w in ("schedule", "book", "set up", "add an event", "put a")):
            return [1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0]

    anchors = {
        "calendar": ["what's on my calendar", "show me my schedule"],
        "calendar_create": ["schedule a meeting", "book an appointment"],
    }
    clf = SemanticIntentClassifier(
        cal_embedder, anchors, fallback=_Fallback(), threshold=0.42, margin=0.05
    )
    assert clf.classify("book a dentist appointment friday at 9am").intent == "calendar_create"
    assert clf.classify("what's on my calendar today").intent == "calendar"


def test_load_intent_anchors_from_shipped_yaml() -> None:
    anchors = load_intent_anchors(Path("config/intent_anchors.yaml"))
    for intent in ("communication", "calendar", "calendar_create", "finance", "planner", "weather"):
        assert intent in anchors, intent
        assert anchors[intent] and all(isinstance(p, str) for p in anchors[intent])


def test_load_intent_anchors_missing_or_malformed(tmp_path: Path) -> None:
    assert load_intent_anchors(tmp_path / "nope.yaml") == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("not a mapping\n", encoding="utf-8")
    assert load_intent_anchors(bad) == {}


# ── defer anchors (messages that need no domain) ─────────────────────────────


def _chat_embedder(text: str) -> Sequence[float]:
    # "email" and "poem" both present → halfway between the email and chat axes;
    # the chat axis stands in for MiniLM seeing a creative request.
    low = text.lower()
    email = 1.0 if ("email" in low or "inbox" in low) else 0.0
    chat = 1.0 if ("poem" in low or "joke" in low) else 0.0
    if not email and not chat:
        return _OTHER_VEC
    return [email, 0.0, chat]


def test_defer_anchor_closest_hands_to_fallback() -> None:
    fb = _Fallback()
    clf = SemanticIntentClassifier(
        _chat_embedder, _ANCHORS, fallback=fb, defer_anchors=["write a poem", "tell me a joke"]
    )
    r = clf.classify("tell me a joke")
    assert r.source == "fallback"
    assert fb.calls == 1


def test_defer_anchor_within_margin_hands_to_fallback() -> None:
    # "a poem about my email": email and chat tie, so the email intent doesn't beat
    # the defer anchors by the margin and the fallback (LLM) decides.
    fb = _Fallback()
    clf = SemanticIntentClassifier(
        _chat_embedder, _ANCHORS, fallback=fb, defer_anchors=["write a poem"]
    )
    assert clf.classify("write a poem about my email").source == "fallback"


def test_defer_anchors_leave_clear_domain_turns_alone() -> None:
    fb = _Fallback()
    clf = SemanticIntentClassifier(
        _chat_embedder, _ANCHORS, fallback=fb, defer_anchors=["write a poem"]
    )
    r = clf.classify("check my email")
    assert r.intent == "communication" and r.source == "semantic"
    assert fb.calls == 0


def test_load_defer_anchors_from_shipped_yaml(tmp_path: Path) -> None:
    from iris_harness.agent.intent_router import load_intent_defer_anchors

    shipped = load_intent_defer_anchors(Path("config/intent_anchors.yaml"))
    assert "tell me a joke" in shipped
    # `defer` is not an intent: the intent loader must not pick it up.
    assert "defer" not in load_intent_anchors(Path("config/intent_anchors.yaml"))
    assert load_intent_defer_anchors(tmp_path / "nope.yaml") == []
    no_list = tmp_path / "no_list.yaml"
    no_list.write_text("defer: nope\n", encoding="utf-8")
    assert load_intent_defer_anchors(no_list) == []
