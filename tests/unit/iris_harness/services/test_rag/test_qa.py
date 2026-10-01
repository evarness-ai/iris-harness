"""Tests for grounded cited Q&A + tag-scoped retrieval (RAG R3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.rag.ingest import ingest_path
from iris_harness.services.rag.qa import answer_question
from iris_harness.services.rag.retrieve import search_documents
from iris_harness.services.rag.store import DocumentStore


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    s = DocumentStore(db_path=tmp_path / "rag.db")
    s.ensure_schema()
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "finance.md").write_text(
        "---\ntitle: Budget\ntags: [finance]\n---\n# Budget\n\nThe travel budget for Q3 is 5000 USD."
    )
    (vault / "garden.md").write_text(
        "---\ntitle: Garden\ntags: [home]\n---\n# Garden\n\nTomatoes need full sun."
    )
    ingest_path(vault, store=s, index=None, kind="obsidian")
    return s


def test_answer_without_llm_returns_cited_passages(store: DocumentStore) -> None:
    ans = answer_question("What is the travel budget?", store=store, index=None)
    assert ans.grounded is False
    assert "5000" in ans.answer
    assert ans.citations and ans.citations[0].source_path.endswith("finance.md")
    assert "Sources:" in ans.render() and "[1]" in ans.render()


def test_answer_with_llm_synthesises_and_cites(store: DocumentStore) -> None:
    captured: dict[str, str] = {}

    def _fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "The Q3 travel budget is 5000 USD [1]."

    ans = answer_question("travel budget", store=store, index=None, llm_call=_fake_llm)
    assert ans.grounded is True
    assert ans.answer == "The Q3 travel budget is 5000 USD [1]."
    # The LLM only sees retrieved evidence (grounded), not free rein.
    assert "5000 USD" in captured["prompt"] and "ONLY the numbered sources" in captured["prompt"]


def test_answer_falls_back_when_llm_raises(store: DocumentStore) -> None:
    def _boom(_: str) -> str:
        raise RuntimeError("model down")

    ans = answer_question("travel budget", store=store, index=None, llm_call=_boom)
    assert ans.grounded is False and "5000" in ans.answer


def test_no_hits_returns_no_documents(store: DocumentStore) -> None:
    ans = answer_question("quantum chromodynamics", store=store, index=None)
    assert ans.citations == () and "No relevant documents" in ans.answer


def test_tag_scope_restricts_retrieval(store: DocumentStore) -> None:
    # 'sun' appears only in the home-tagged garden note; scoping to finance hides it.
    assert search_documents("sun", store=store, index=None, tags=("home",))
    assert search_documents("sun", store=store, index=None, tags=("finance",)) == []


def test_ask_scoped_by_tag(store: DocumentStore) -> None:
    ans = answer_question("budget", store=store, index=None, tags=("finance",))
    assert ans.citations and all(c.source_path.endswith("finance.md") for c in ans.citations)
