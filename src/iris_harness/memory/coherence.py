"""Coherence checking + repair across the three user-fact homes.

See ``coordinator.py`` for the three-home model (SQLite truth / ChromaDB index /
``USER.md`` projection). This module is the read-side auditor + repairer that
pairs with the write-side coordinator: it diagnoses drift between the truth and
the two derived homes, and can repair the derived homes back to the truth.
``iris memory doctor`` is the CLI front end.

Direction matters: the SQLite store is authoritative. Repair only ever rewrites
the derived homes to match it — it never mutates the store, so it is safe to run
at any time and can never lose a fact or its history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from iris_harness.memory.identity import (
    curated_fact_keys,
    read_auto_fact_keys,
    reconcile_user_facts_md,
)
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.memory.store import MemoryStore


@dataclass
class CoherenceReport:
    """Drift between the SQLite truth and the two derived fact-homes."""

    store_count: int
    index_count: int
    md_count: int
    # Index drift (against the store).
    index_missing: set[str] = field(default_factory=set)  # in store, absent from index
    index_orphans: set[str] = field(default_factory=set)  # in index, absent from store
    # USER.md projection drift (against the store).
    md_missing: set[str] = field(default_factory=set)  # in store, not projected (and not curated)
    md_orphans: set[str] = field(default_factory=set)  # stale auto-bullet, gone from store
    md_value_mismatches: list[tuple[str, str, str]] = field(default_factory=list)  # key, md, store
    checked_md: bool = True

    @property
    def is_coherent(self) -> bool:
        return not (
            self.index_missing
            or self.index_orphans
            or self.md_missing
            or self.md_orphans
            or self.md_value_mismatches
        )

    @property
    def drift_total(self) -> int:
        return (
            len(self.index_missing)
            + len(self.index_orphans)
            + len(self.md_missing)
            + len(self.md_orphans)
            + len(self.md_value_mismatches)
        )


def diagnose(
    store: MemoryStore,
    index: SemanticIndex | None,
    *,
    include_md: bool = True,
) -> CoherenceReport:
    """Compare the SQLite truth against the index and ``USER.md`` projection.

    ``include_md=False`` skips the markdown leg (e.g. headless contexts with no
    identity workspace). A fact a human curated in ``USER.md``'s head section is
    not counted as ``md_missing`` — it lives outside the auto block by design.
    """
    # One entry per key, as the derived homes hold them: a key with several values
    # (two cards) is compared as the list of them all, not whichever came last.
    store_facts = store.fetch_fact_projections()
    store_map = {f.key: f for f in store_facts}
    store_keys = set(store_map)

    index_keys = index.fact_keys() if index is not None else set()
    report = CoherenceReport(
        store_count=len(store_keys),
        index_count=len(index_keys),
        md_count=0,
        checked_md=include_md,
    )
    if index is not None:
        report.index_missing = store_keys - index_keys
        report.index_orphans = index_keys - store_keys

    if include_md:
        md_map = read_auto_fact_keys()
        md_keys = set(md_map)
        curated = curated_fact_keys()
        report.md_count = len(md_keys)
        report.md_missing = store_keys - md_keys - curated
        report.md_orphans = md_keys - store_keys
        for key in md_keys & store_keys:
            md_value, _ = md_map[key]
            store_value = store_map[key].value.strip()
            if md_value != store_value:
                report.md_value_mismatches.append((key, md_value, store_value))

    return report


def repair(
    store: MemoryStore,
    index: SemanticIndex | None,
    *,
    reproject_md: bool = True,
) -> dict[str, int]:
    """Rewrite the derived homes to match the SQLite truth. Never mutates the store.

    Returns action counts: ``index_indexed``/``index_dropped`` and, when
    ``reproject_md``, ``md_added``/``md_updated``/``md_removed``.
    """
    actions: dict[str, int] = {}
    if index is not None:
        index_actions = index.reconcile_facts(store)
        actions["index_indexed"] = index_actions["indexed"]
        actions["index_dropped"] = index_actions["dropped"]
    if reproject_md:
        md_actions = reconcile_user_facts_md(store)
        actions["md_added"] = md_actions["added"]
        actions["md_updated"] = md_actions["updated"]
        actions["md_removed"] = md_actions["removed"]
    return actions
