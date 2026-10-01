"""Obsidian vault connector — one-way export of the memory graph.

The stub here said "Phase 4" for months while the wiki it was meant to sync filled up
with 2,505 regex-built pages nothing read. What is worth exporting now is the memory
graph: confirmed facts, session summaries, lessons and patterns, linked.

``sync_from`` stays unimplemented, and that is the design, not a gap. A vault edit
flowing back would undo the confirmation gate — nothing becomes a belief without the
owner — and two-way sync between a store and its copy is exactly how the wiki ended up
with 2,858 vectors against 2,505 pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import SyncResult


class ObsidianConnector:
    """Export IRIS memory into an Obsidian vault. Export only, by design."""

    def __init__(self, vault_path: Path) -> None:
        self.vault_path = vault_path

    def export(self, store: Any, *, include_unconfirmed: bool = False) -> SyncResult:
        """Write the memory graph into the vault as linked markdown notes."""
        from iris_harness.memory.export import export_memory_vault

        result = export_memory_vault(
            store, self.vault_path, include_unconfirmed=include_unconfirmed
        )
        return SyncResult(pages_exported=result.notes)

    def sync_to(self, wiki_dir: Path) -> SyncResult:
        """Legacy wiki-directory sync. Superseded by :meth:`export` (ADR-0114)."""
        raise NotImplementedError(
            "wiki→vault sync is retired with automatic wiki ingest (ADR-0114); "
            "use ObsidianConnector.export(store) to export the memory graph instead"
        )

    def sync_from(self, vault_dir: Path) -> SyncResult:
        """Not implemented, and not planned — see the module docstring."""
        raise NotImplementedError(
            "importing from a vault is deliberately unsupported: IRIS believes what the "
            "owner confirms inside it, and a two-way copy is how stores drift apart"
        )
