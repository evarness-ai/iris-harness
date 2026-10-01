"""Tests for the memory garbage-in / garbage-out audit harness."""

from __future__ import annotations

from iris_harness.memory.gate_audit import (
    FactCandidate,
    audit_capture_gates,
    audit_recall_filter,
    run_full_audit,
    sweep_recall_threshold,
)
from iris_harness.memory.gate_audit_corpus import DEFAULT_CORPUS


def test_capture_gate_confusion_matrix() -> None:
    corpus = [
        FactCandidate("name", "Anita Rao", "my name is anita rao", 0.9, "keep"),  # admit
        FactCandidate(
            "hobby", "i love hiking on weekends", "i love hiking on weekends", 0.6, "junk"
        ),  # plausibility
        FactCandidate("interest", "hello", "i said hello", 0.6, "junk"),  # durability
        FactCandidate("email", "apple", "what is my weather", 0.5, "junk"),  # grounding
        FactCandidate("topic", "chips", "i grabbed some chips", 0.2, "junk"),  # key_allowlist
        FactCandidate(
            "hobby", "chips", "i grabbed some chips", 0.2, "junk"
        ),  # passes every gate (low conf) — only the recall filter can catch it
    ]
    g = audit_capture_gates(corpus)

    assert g.keep == 1 and g.junk == 5
    assert g.true_admit == 1
    assert g.false_admit == 1  # the low-conf 'chips' under an allowed key slips the gates
    assert g.true_reject == 4
    assert g.false_reject == 0
    assert g.by_gate == {
        "plausibility": 1,
        "durability": 1,
        "grounding": 1,
        "key_allowlist": 1,
    }
    assert g.recall == 1.0


def test_grounding_overblocks_a_normalized_keep() -> None:
    # A real fact normalized away from the user's wording (nyc -> New York City).
    corpus = [FactCandidate("city", "New York City", "i moved to nyc last year", 0.8, "keep")]
    g = audit_capture_gates(corpus)
    assert g.false_reject == 1
    assert g.false_reject_by_gate == {"grounding": 1}


def test_recall_filter_false_recall_and_drop() -> None:
    corpus = [
        FactCandidate("a", "x", "x", 0.42, "junk"),  # >= 0.35 -> reaches prompt = false_recall
        FactCandidate("b", "y", "y", 0.2, "junk"),  # < 0.35 -> dropped (correct)
        FactCandidate("c", "z", "z", 0.3, "keep"),  # < 0.35 -> dropped = false_drop
        FactCandidate("d", "w", "w", 0.9, "keep"),  # clean
    ]
    r = audit_recall_filter(corpus, min_fact_confidence=0.35, uncertain_below=0.6)
    assert r.false_recall == 1
    assert r.false_drop == 1
    assert r.dropped == 2
    assert r.reaching_prompt == 2


def test_run_full_audit_recall_runs_on_admitted_only() -> None:
    # A high-confidence gate-junk (blocked by grounding) must NOT count as a recall leak —
    # it never reaches the store.
    corpus = [
        FactCandidate("email", "apple", "what is my weather", 0.9, "junk"),  # gate-blocked
        FactCandidate("hobby", "lamp", "i bought a lamp", 0.42, "junk"),  # slips both
    ]
    a = run_full_audit(corpus, min_fact_confidence=0.35, uncertain_below=0.6)
    assert a.gates.false_admit == 1  # only 'lamp' slipped the gates
    assert a.recall.false_recall == 1  # and only 'lamp' is the recall leak (apple excluded)


def test_sweep_suggests_separating_threshold() -> None:
    corpus = [
        FactCandidate("k1", "a", "a", 0.8, "keep"),
        FactCandidate("k2", "b", "b", 0.7, "keep"),
        FactCandidate("j1", "c", "c", 0.42, "junk"),
        FactCandidate("j2", "d", "d", 0.2, "junk"),
    ]
    s = sweep_recall_threshold(corpus, current_min_confidence=0.35, uncertain_below=0.6)
    # A threshold in (0.42, 0.7] drops both junk and keeps both keep.
    assert 0.42 < s.suggested_min_confidence <= 0.7
    assert s.rows  # the full sweep is exposed


def test_default_corpus_is_well_formed_and_exercises_both_layers() -> None:
    a = run_full_audit(DEFAULT_CORPUS, min_fact_confidence=0.35, uncertain_below=0.6)
    # Both garbage paths are represented: junk slips the gates AND junk slips recall.
    assert a.gates.false_admit > 0
    assert a.gates.false_reject_by_gate.get("grounding", 0) >= 1  # the known over-block
    assert a.recall.false_recall >= 1  # the known recall leak
    assert a.sweep.suggested_min_confidence >= 0.0
    # The two filters ahead of the validation gates do most of the work now: the
    # corpus carries the real junk keys from the store (topic, source, number, ...)
    # and third-party messages, and those never reach the validation gates at all.
    assert a.gates.by_gate.get("key_allowlist", 0) >= 5
    assert a.gates.by_gate.get("first_person", 0) >= 3
