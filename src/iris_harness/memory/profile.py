"""DB-backed user profile with confidence tracking."""

from __future__ import annotations

from datetime import UTC, datetime

from .store import MemoryStore, UserFact


class UserProfile:
    """Persistent user profile backed by MemoryStore."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        self._store = store or MemoryStore()

    def upsert(
        self, key: str, value: str, *, source: str = "user", confidence: float = 0.9
    ) -> UserFact:
        """Add or update a user fact with confidence tracking."""
        now = datetime.now(UTC)
        existing = self._store.fetch_user_fact(key)
        if existing is not None:
            fact = UserFact(
                key=key,
                value=value,
                confidence=min(1.0, existing.confidence + 0.05),
                source=source,
                first_seen=existing.first_seen,
                last_confirmed=now,
                times_confirmed=existing.times_confirmed + 1,
            )
        else:
            fact = UserFact(
                key=key,
                value=value,
                confidence=confidence,
                source=source,
                first_seen=now,
                last_confirmed=now,
            )
        self._store.upsert_user_fact(fact)
        return fact

    def get(self, key: str) -> str | None:
        fact = self._store.fetch_user_fact(key)
        return fact.value if fact is not None else None

    def all_facts(self) -> list[UserFact]:
        return self._store.fetch_all_user_facts()

    def remove(self, key: str) -> bool:
        existing = self._store.fetch_user_fact(key)
        if existing is None:
            return False
        self._store.delete_user_fact(key)
        return True

    def to_summary_string(self) -> str:
        facts = self.all_facts()
        if not facts:
            return ""
        return "; ".join(f"{f.key}={f.value}" for f in facts[:10])
