"""IRIS Knowledge Wiki — file-backed compiled knowledge base."""

from .entity_extractor import EntityExtractor
from .models import Entity, LintReport, LintResult, WikiIngestEvent, WikiPage
from .page_manager import PageManager, slugify
from .wiki_engine import KnowledgeWikiEngine, WikiEngine
from .wiki_index import WikiIndex

__all__ = [
    "Entity",
    "EntityExtractor",
    "KnowledgeWikiEngine",
    "LintReport",
    "LintResult",
    "PageManager",
    "WikiEngine",
    "WikiIngestEvent",
    "WikiIndex",
    "WikiPage",
    "slugify",
]
