from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from iris_harness.memory.store import MemoryStore, UserFact


def test_user_fact_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_path)

    fact = UserFact(
        key="timezone",
        value="America/Los_Angeles",
        confidence=0.95,
        source="explicit",
        first_seen=datetime.now(UTC),
        last_confirmed=datetime.now(UTC),
        times_confirmed=2,
    )

    store.upsert_user_fact(fact)
    loaded = store.fetch_user_fact("timezone")

    assert loaded is not None
    assert loaded.key == fact.key
    assert loaded.value == fact.value
    assert loaded.confidence == fact.confidence


def test_validate_user_fact_rejects_out_of_range_confidence() -> None:
    store = MemoryStore(db_path=Path("data/test-memory.db"))
    invalid = UserFact(
        key="response_style",
        value="concise",
        confidence=1.2,
        source="inferred",
        first_seen=datetime.now(UTC),
        last_confirmed=datetime.now(UTC),
        times_confirmed=1,
    )

    assert store.validate_user_fact(invalid) is False


# ── Confidence-guarded upsert (issue 0021) ─────────────────────────────────


def test_low_confidence_does_not_clobber_higher(tmp_path) -> None:
    import datetime as _dt

    from iris_harness.memory.store import MemoryStore, UserFact

    ms = MemoryStore(db_path=tmp_path / "m.db")
    ms.ensure_schema()
    now = _dt.datetime.now(_dt.timezone.utc)

    def _fact(v, c):
        return UserFact(
            key="blog",
            value=v,
            confidence=c,
            source="x",
            first_seen=now,
            last_confirmed=now,
            times_confirmed=1,
        )

    ms.upsert_user_fact(_fact("www.web3notes.example", 0.9))
    ms.upsert_user_fact(_fact("site", 0.3))  # mis-extraction must NOT clobber
    stored = ms.fetch_user_fact("blog")
    assert stored is not None
    assert (stored.value, stored.confidence) == ("www.web3notes.example", 0.9)
    # a higher-confidence correction DOES replace it
    ms.upsert_user_fact(_fact("blog.web3notes.example", 0.95))
    replaced = ms.fetch_user_fact("blog")
    assert replaced is not None and replaced.value == "blog.web3notes.example"
