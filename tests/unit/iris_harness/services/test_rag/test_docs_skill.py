"""The docs-search skill package loads and matches its manifest (RAG R0)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.tools.skills.loader import load_skill_package

REPO_ROOT = Path(__file__).resolve().parents[5]
SKILL_DIR = REPO_ROOT / "config" / "skills" / "rag" / "docs-search"


def test_docs_search_skill_loads() -> None:
    pkg = load_skill_package(REPO_ROOT, SKILL_DIR)
    assert pkg.manifest.name == "docs-search"
    assert pkg.manifest.default_enabled is True
    assert {t.name for t in pkg.manifest.tools} == {"search_documents"}
    impl = {c.model_fields["name"].default for c in pkg.tool_classes}
    assert impl == {"search_documents"}
    assert pkg.manifest.tools[0].governor_route == "rag/read"
    assert pkg.missing_prerequisites == ()
