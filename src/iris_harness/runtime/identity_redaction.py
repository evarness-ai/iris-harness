"""The composition root's answer to the kernel's identity-redaction seam.

The guards need the owner's identity to know which literals they must refuse to send or
show. They cannot read it themselves -- the kernel sits below the layer that owns it -- so
they ask, and this registers the answers (ADR-0125):

- ``documents``: SOUL.md, USER.md and AGENTS.md as free text (secrets, links, and the
  email and phone shapes).
- ``user_md``: the ``identity:`` block of the USER.md frontmatter -- names, emails,
  phones, addresses, handles, and ``never_match``. The only source of postal addresses.
- ``facts``: the owner's CONFIRMED facts whose ontology attribute
  ``config/governance/identity.yaml`` maps to a kind. Confirmed, never "confident": the
  store once held ``name=ollama`` at 1.0, mined from content the owner was discussing.

Each has a cheap fingerprint the seam checks at most every few seconds, so a write from
another process (the CLI, the API) is seen; this process's own writers invalidate the seam
directly. Plugin sources are registered by ``PluginAPI``, bound to the plugin's manifest.

Imported by `runtime.bootstrap` for that side effect, so every process that can
build a runtime can also redact. The same shape as `runtime/eval_runtime.py`.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from iris_harness.foundation.persistence import data_path
from iris_harness.kernel.governance.identity_config import load_identity_config
from iris_harness.kernel.governance.identity_redaction import (
    invalidate_owner_identity,
    register_identity_text_provider,
    register_owner_identity_source,
)
from iris_harness.kernel.governance.owner_identity import NEVER_MATCH

if TYPE_CHECKING:
    from iris_harness.memory.fact_statements import FactStatements

logger = logging.getLogger(__name__)

USER_MD_SOURCE = "user_md"
FACTS_SOURCE = "facts"


def _stat(path: Path) -> tuple[str, int, int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_ino, st.st_mtime_ns, st.st_size)


def identity_documents() -> Sequence[str]:
    """SOUL.md, USER.md and AGENTS.md as text, skipping whatever is unreadable.

    Best-effort per document on purpose: a missing AGENTS.md must not cost the
    guard the secrets in USER.md. What it must never do is fail as a whole and
    leave the guard with nothing while looking fine -- that distinction is the
    registry's `None`-vs-`[]`.
    """
    from iris_harness.memory.identity import loader as identity_loader

    texts: list[str] = []
    for name in ("load_soul", "load_user_md", "load_agents_md"):
        fn = getattr(identity_loader, name, None)
        if fn is None:
            continue
        try:
            body = fn()
        except Exception:  # one unreadable document, not all of them
            logger.debug("identity redaction: %s failed", name, exc_info=True)
            continue
        if body:
            texts.append(str(body))
    return texts


def identity_documents_fingerprint() -> Hashable:
    """Each identity file's inode, mtime and size (legacy paths too): a few ``stat`` calls."""
    from iris_harness.memory.identity import loader as identity_loader

    return tuple(_stat(p) for p in identity_loader.identity_document_paths())


def user_md_identity() -> Mapping[str, Iterable[str]]:
    """The USER.md ``identity:`` block, by kind (an invalid block reads as empty)."""
    from iris_harness.memory.identity import loader as identity_loader

    block = identity_loader.load_user_identity()
    return {
        "name": block.names,
        "email": block.emails,
        "phone": block.phones,
        "address": block.addresses,
        "handle": block.handles,
        NEVER_MATCH: block.never_match,
    }


def user_md_fingerprint() -> Hashable:
    """USER.md's inode, mtime and size, workspace and legacy copies."""
    from iris_harness.memory.identity import loader as identity_loader

    return tuple(_stat(p) for p in identity_loader.user_md_paths())


class OwnerFacts:
    """The ``facts`` source: the owner's confirmed facts, by kind.

    ``db_path`` is the memory store's database; ``None`` resolves ``$IRIS_DATA_DIR`` at
    each read, so a process that never builds a runtime still reads the right file.
    ``build_runtime`` points it at the runtime's own store (:func:`use_memory_db`).

    The fingerprint is SQLite's own change counter, ``PRAGMA data_version``, on a
    connection this source keeps open and never writes on: it moves whenever ANY other
    connection commits to the database, and reading it touches no table. Row probes
    (a statement count, the latest ``recorded_at``) are not correct here -- memris
    confirms and retracts by rewriting a row in place, which changes neither. The
    counter also moves for writes that are not facts (conversation turns share the
    file), which costs one re-read of the facts at most every few seconds; never a
    missed change. The file's inode is part of it, so a replaced database is noticed.
    """

    def __init__(self) -> None:
        self.db_path: Path | None = None
        self._lock = threading.Lock()
        self._probe: tuple[Path, int, sqlite3.Connection] | None = None
        self._facts: tuple[Path, FactStatements] | None = None
        self._warned: set[str] = set()

    def path(self) -> Path:
        return self.db_path or data_path("memory.db")

    def fingerprint(self) -> Hashable:
        path = self.path()
        try:
            inode = os.stat(path).st_ino
        except OSError:
            self._close_probe()
            return (str(path), None)
        with self._lock:
            if self._probe is None or self._probe[0] != path or self._probe[1] != inode:
                # A new connection's counter is not comparable with the old one's; the
                # path and inode in the fingerprint already say that it changed.
                self._close_probe_locked()
                self._probe = (path, inode, sqlite3.connect(path, check_same_thread=False))
            version = self._probe[2].execute("PRAGMA data_version").fetchone()[0]
        return (str(path), inode, version)

    def _close_probe(self) -> None:
        with self._lock:
            self._close_probe_locked()

    def _close_probe_locked(self) -> None:
        if self._probe is not None:
            self._probe[2].close()
            self._probe = None

    def _statements(self, path: Path) -> FactStatements:
        if self._facts is None or self._facts[0] != path:
            from iris_harness.memory.store import MemoryStore

            self._facts = (path, MemoryStore(db_path=path).fact_statements())
        return self._facts[1]

    def __call__(self) -> Mapping[str, Iterable[str]]:
        path = self.path()
        if not path.exists():
            return {}  # no memory yet; reading must not create the database
        configured = load_identity_config().ontology_kinds
        if not configured:
            return {}
        from iris_harness.memory.fact_statements import FactKeyError

        facts = self._statements(path)
        # identity.yaml names attributes as ontology.yaml does: a bare name is in the
        # ontology's default prefix (``name`` is ``mem:name``).
        kinds = {facts.ontology.qualify(attr): kind for attr, kind in configured.items()}
        unknown = sorted(set(kinds) - set(facts.ontology.attributes) - self._warned)
        if unknown:
            self._warned.update(unknown)
            logger.warning(
                "identity.yaml ontology_kinds names no ontology attribute: %s", ", ".join(unknown)
            )
        out: dict[str, list[str]] = {}
        # ``all`` reads the owner entity's statements only (a contact's name is never the
        # owner's), and ``confirmed_only`` only what the owner confirmed.
        for fact in facts.all(confirmed_only=True):
            try:
                predicate = facts.rule(fact.key).predicate
            except FactKeyError:  # a learned term since rejected: no mapping to a kind
                continue
            kind = kinds.get(predicate)
            if kind is not None:
                out.setdefault(kind, []).append(fact.value)
        return out


_FACTS = OwnerFacts()


def use_memory_db(path: Path) -> None:
    """Read the owner's facts from ``path`` (the runtime's memory store)."""
    _FACTS.db_path = path
    invalidate_owner_identity()


register_identity_text_provider(identity_documents, fingerprint=identity_documents_fingerprint)
register_owner_identity_source(USER_MD_SOURCE, user_md_identity, fingerprint=user_md_fingerprint)
register_owner_identity_source(FACTS_SOURCE, _FACTS, fingerprint=_FACTS.fingerprint)

__all__ = [
    "FACTS_SOURCE",
    "USER_MD_SOURCE",
    "OwnerFacts",
    "identity_documents",
    "identity_documents_fingerprint",
    "use_memory_db",
    "user_md_fingerprint",
    "user_md_identity",
]
