"""IWikiConnector protocol for external vault sync."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class SyncResult:
    pages_exported: int = 0
    pages_imported: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors


@runtime_checkable
class IWikiConnector(Protocol):
    """Protocol for wiki sync connectors (Obsidian, Notion, etc.)."""

    def sync_to(self, wiki_dir: Path) -> SyncResult: ...
    def sync_from(self, vault_dir: Path) -> SyncResult: ...
