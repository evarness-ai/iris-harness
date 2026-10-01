"""Tests for the behavior-pattern miner engine (pure, model-driven)."""

from __future__ import annotations

from iris_harness.services.learning.behavior_miner import (
    BehaviorPattern,
    dedupe_semantically,
    mine_behavior_patterns,
    parse_behavior_patterns,
    pattern_id_for,
)


def _bp(text: str, conf: str = "high") -> BehaviorPattern:
    return BehaviorPattern(pattern_id=pattern_id_for(text), text=text, confidence=conf, evidence=())


# A toy embedder: vector keyed by topic so paraphrases collide, distinct topics don't.
_VECTORS = {
    "weather a": [1.0, 0.0, 0.0],
    "weather b": [0.97, 0.05, 0.0],  # near-dup of weather a (cosine ~0.998)
    "reminders": [0.0, 1.0, 0.0],
    "calendar": [0.0, 0.0, 1.0],
}


def _topic(text: str) -> str:
    t = text.lower()
    if "weather" in t and "london" in t:
        return "weather a" if "daily" in t else "weather b"
    if "remind" in t:
        return "reminders"
    return "calendar"


def _stub_embed(texts: list[str]) -> list[list[float]]:
    return [_VECTORS[_topic(t)] for t in texts]


_GOOD = (
    '[{"pattern":"Asks for an inbox summary most mornings","confidence":"high",'
    '"evidence":["mon: summarize my inbox","tue: what is new in my inbox"]},'
    '{"pattern":"Prefers short answers","confidence":"medium","evidence":["be concise"]}]'
)


def _turns(n: int) -> list[tuple[str, str]]:
    return [("user", f"message {i}") for i in range(n)]


def test_parse_valid_array() -> None:
    out = parse_behavior_patterns(_GOOD)
    assert [p.text for p in out] == [
        "Asks for an inbox summary most mornings",
        "Prefers short answers",
    ]
    assert out[0].confidence == "high"
    assert out[0].evidence[0].startswith("mon:")


def test_parse_strips_code_fence_and_prose() -> None:
    raw = 'Sure!\n```json\n[{"pattern":"Reviews calendar before scheduling"}]\n```'
    out = parse_behavior_patterns(raw)
    assert len(out) == 1 and out[0].confidence == "low"  # default when omitted


def test_parse_garbage_and_non_list_return_empty() -> None:
    assert parse_behavior_patterns("not json") == []
    assert parse_behavior_patterns('{"pattern":"x"}') == []  # object, not array
    assert parse_behavior_patterns("") == []


def test_parse_dedups_same_pattern() -> None:
    raw = '[{"pattern":"Works late"},{"pattern":"works   LATE"}]'  # same normalized
    out = parse_behavior_patterns(raw)
    assert len(out) == 1


def test_pattern_id_is_stable_and_normalized() -> None:
    assert pattern_id_for("Works late") == pattern_id_for("  works   LATE  ")


def test_mine_requires_minimum_turns() -> None:
    called = {"n": 0}

    def invoke(_s: str, _u: str) -> str:
        called["n"] += 1
        return _GOOD

    assert mine_behavior_patterns(_turns(3), invoke=invoke, min_turns=6) == []
    assert called["n"] == 0  # never even calls the model on thin history


def test_mine_returns_patterns_from_model() -> None:
    out = mine_behavior_patterns(_turns(10), invoke=lambda _s, _u: _GOOD)
    assert all(isinstance(p, BehaviorPattern) for p in out)
    assert len(out) == 2


def test_mine_swallows_llm_failure() -> None:
    def boom(_s: str, _u: str) -> str:
        raise RuntimeError("model down")

    assert mine_behavior_patterns(_turns(10), invoke=boom) == []


def test_semantic_dedup_drops_within_batch_paraphrase() -> None:
    cands = [
        _bp("Asks for daily weather updates in London"),
        _bp("Checks weather in London multiple times a day"),  # paraphrase → dropped
        _bp("Prefers to be reminded of important tasks"),
    ]
    kept = dedupe_semantically(cands, embed=_stub_embed)
    texts = [p.text for p in kept]
    assert "Asks for daily weather updates in London" in texts  # first wins
    assert "Checks weather in London multiple times a day" not in texts
    assert "Prefers to be reminded of important tasks" in texts  # distinct topic kept


def test_semantic_dedup_drops_against_existing() -> None:
    cands = [_bp("Checks weather in London multiple times a day")]
    existing = ["Asks for daily weather updates in London"]  # already approved/queued
    assert dedupe_semantically(cands, existing, embed=_stub_embed) == []


def test_semantic_dedup_noop_when_embeddings_unavailable() -> None:
    cands = [_bp("Asks for daily weather updates in London"), _bp("Checks weather in London")]
    # embed returns nothing usable → pass through unchanged (exact-id dedup still applies)
    assert len(dedupe_semantically(cands, embed=lambda _t: [])) == 2


def test_semantic_dedup_keeps_distinct_habits() -> None:
    cands = [_bp("Prefers to be reminded of tasks"), _bp("Reviews calendar before scheduling")]
    assert len(dedupe_semantically(cands, embed=_stub_embed)) == 2
