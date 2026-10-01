"""The memory vocabulary, for plugins (memris PR 10).

A plugin that reads or writes the memory graph (an importer, a connector) needs the
ontology IRIS runs with: the core ``config/memory`` YAML plus every installed plugin's
fragment (``fin:`` from the finance plugin). Compiling the core directory alone would
miss those, and a stored bank would look like a class nobody declared.

A plugin that knows which names are noise (a mail plugin knows which senders are shops)
registers a provider with ``register_map_exclusions`` so the memory Map leaves them out
(ADR-0119).

A plugin that feeds the knowledge wiki emits a ``WikiIngestEvent`` on
``WIKI_INGEST_REQUESTED`` (the wiki consumes it). A command that runs with no runtime
built (a backfill) builds a ``WikiEngine`` and wires it with
``subscribe_wiki_ingest_consumer``.
"""

from __future__ import annotations

from iris_harness.memory.knowledge.event_subscribers import (
    WIKI_INGEST_REQUESTED,
    subscribe_wiki_ingest_consumer,
)
from iris_harness.memory.knowledge.models import WikiIngestEvent
from iris_harness.memory.knowledge.wiki_engine import WikiEngine
from iris_harness.memory.map_exclusions import register_map_exclusions
from iris_harness.memory.ontology import memory_config_dir, memory_ontology

__all__ = [
    "WIKI_INGEST_REQUESTED",
    "WikiEngine",
    "WikiIngestEvent",
    "memory_config_dir",
    "memory_ontology",
    "register_map_exclusions",
    "subscribe_wiki_ingest_consumer",
]
