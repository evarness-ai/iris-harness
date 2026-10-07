"""Tests for the code_exec lesson capture service."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.memory.store import MemoryStore
from iris_harness.services.learning.lesson_capture import (
    LESSON_SIGNAL_TYPE,
    Lesson,
    LessonCapture,
    extract_from_answer,
)


def test_extract_from_answer_parses_fenced_block() -> None:
    answer = (
        "Done. Wrote /workspace/news.pdf with 12 articles.\n\n"
        "```lesson\n"
        '{"category": "news-pdf", "summary": "Used gnews + reportlab",\n'
        ' "tools": ["run_shell"], "sources": ["news.google.com"],\n'
        ' "scripts": ["fetch_news.py"]}\n'
        "```\n"
    )
    lesson, cleaned = extract_from_answer(answer)

    assert lesson is not None
    assert lesson.category == "news-pdf"
    assert lesson.summary.startswith("Used gnews")
    assert "run_shell" in lesson.tools
    assert lesson.sources == ("news.google.com",)
    assert "fetch_news.py" in lesson.scripts
    assert "```lesson" not in cleaned
    assert "Done. Wrote" in cleaned


def test_extract_from_answer_returns_none_when_no_block() -> None:
    answer = "Plain prose answer with no fenced lesson."
    lesson, cleaned = extract_from_answer(answer)
    assert lesson is None
    assert cleaned == answer


def test_extract_from_answer_handles_invalid_json() -> None:
    answer = "Done.\n\n```lesson\nnot valid json {{{\n```\n"
    lesson, cleaned = extract_from_answer(answer)
    assert lesson is None
    assert "```lesson" not in cleaned


def test_handle_persists_signal_and_redacts(tmp_path: Path) -> None:
    store = MemoryStore(db_path=tmp_path / "mem.db")
    store.ensure_schema()
    capture = LessonCapture(memory_store=store, wiki=None)

    answer = (
        "Wrote /Users/alice/secret/file.pdf using TOKEN=sk_live_abc123def456ghi789jkl012mno345pqr.\n"
        "```lesson\n"
        '{"category": "pdf-gen", "summary": "reportlab works",\n'
        ' "tools": ["run_shell"]}\n'
        "```"
    )
    lesson, cleaned = capture.handle(
        query="make a PDF for /Users/alice/foo",
        answer=answer,
        artifacts=("/Users/alice/secret/file.pdf",),
        session_id="sess-1",
        iterations=2,
        all_succeeded=True,
    )
    assert lesson is not None
    assert "```lesson" not in cleaned

    signals = store.fetch_learning_signals({"signal_type": LESSON_SIGNAL_TYPE})
    assert len(signals) == 1
    sig = signals[0]
    assert sig.domain == "pdf-gen"
    assert "/Users/alice" not in sig.query
    assert "/<home>" in sig.query
    assert "sk_live_abc123def456ghi789jkl012mno345pqr" not in sig.context


def test_handle_skips_persistence_on_failure(tmp_path: Path) -> None:
    store = MemoryStore(db_path=tmp_path / "mem.db")
    store.ensure_schema()
    capture = LessonCapture(memory_store=store, wiki=None)

    answer = 'Done.\n```lesson\n{"category":"x","summary":"ok"}\n```'
    capture.handle(
        query="q",
        answer=answer,
        artifacts=(),
        iterations=2,
        all_succeeded=False,
    )
    assert store.fetch_learning_signals({"signal_type": LESSON_SIGNAL_TYPE}) == []


def test_find_similar_returns_top_k(tmp_path: Path) -> None:
    store = MemoryStore(db_path=tmp_path / "mem.db")
    store.ensure_schema()
    capture = LessonCapture(memory_store=store, wiki=None)

    # Seed three lessons of varying overlap with the query.
    seed_answers = [
        (
            '"category":"news","summary":"fetch news with gnews and write PDF",'
            '"tools":["run_shell"],"sources":["news.example"]'
        ),
        ('"category":"data","summary":"read csv with pandas",' '"tools":["run_shell"]'),
        ('"category":"news","summary":"fetch ai news rss feed daily",' '"tools":["run_shell"]'),
    ]
    queries = [
        "fetch ai news and create a pdf",
        "load csv data and graph",
        "get the latest ai news",
    ]
    for q, body in zip(queries, seed_answers, strict=True):
        capture.handle(
            query=q,
            answer=f"ok\n```lesson\n{{{body}}}\n```",
            artifacts=("/x",),
            iterations=1,
            all_succeeded=True,
        )

    matches = capture.find_similar("fetch ai news pdf", k=2)
    assert len(matches) <= 2
    assert any("news" in m.summary.lower() for m in matches)


def test_render_prior_lessons_formats_block(tmp_path: Path) -> None:
    capture = LessonCapture(memory_store=MemoryStore(db_path=tmp_path / "m.db"), wiki=None)
    rendered = capture.render_prior_lessons(
        [
            Lesson(
                category="news",
                summary="fetched RSS and rendered PDF",
                tools=("run_shell",),
                sources=("https://news.example",),
            )
        ]
    )
    assert "PRIOR LESSONS" in rendered
    assert "[news]" in rendered
    assert "run_shell" in rendered


def test_render_prior_lessons_empty_returns_empty_string(tmp_path: Path) -> None:
    capture = LessonCapture(memory_store=MemoryStore(db_path=tmp_path / "m.db"), wiki=None)
    assert capture.render_prior_lessons([]) == ""


@pytest.mark.parametrize(
    ("text", "should_redact"),
    [
        ("/Users/bob/data", True),
        ("/home/carol/foo", True),
        ("~/Downloads/file.txt", True),
        ("API_KEY=sk-abcdef123", True),
        ("normal sentence", False),
    ],
)
def test_redaction_patterns(text: str, should_redact: bool, tmp_path: Path) -> None:
    from iris_harness.services.learning.lesson_capture import _redact

    out = _redact(text)
    if should_redact:
        assert out != text
    else:
        assert out == text


def _seed_two_domains(tmp_path: Path) -> LessonCapture:
    store = MemoryStore(db_path=tmp_path / "mem.db")
    store.ensure_schema()
    capture = LessonCapture(memory_store=store, wiki=None)
    seeds = [
        ("news", "fetch news pdf with gnews"),
        ("finance", "extract pdf statement with pdfplumber"),
        ("news", "rss feed daily news report"),
    ]
    for category, summary in seeds:
        body = (
            f'"category":"{category}","summary":"{summary}",'
            '"tools":["run_shell"],"sources":["src.example"]'
        )
        capture.handle(
            query=summary,
            answer=f"ok\n```lesson\n{{{body}}}\n```",
            artifacts=(),
            iterations=1,
            all_succeeded=True,
        )
    return capture


def test_find_similar_uses_top_k_default(tmp_path: Path) -> None:
    capture = _seed_two_domains(tmp_path)
    capture.top_k = 1
    matches = capture.find_similar("fetch news pdf rss")
    assert len(matches) == 1


def test_find_similar_top_k_param_overrides_default(tmp_path: Path) -> None:
    capture = _seed_two_domains(tmp_path)
    capture.top_k = 1
    matches = capture.find_similar("fetch news pdf rss", k=5)
    assert len(matches) >= 2


def test_find_similar_domain_allow_filter(tmp_path: Path) -> None:
    capture = _seed_two_domains(tmp_path)
    capture.domain_allow = frozenset({"finance"})
    matches = capture.find_similar("pdf statement extract")
    assert matches
    assert all(m.category == "finance" for m in matches)


def test_find_similar_domain_block_filter(tmp_path: Path) -> None:
    capture = _seed_two_domains(tmp_path)
    capture.domain_block = frozenset({"news"})
    matches = capture.find_similar("fetch news pdf rss")
    assert all(m.category != "news" for m in matches)


def test_find_similar_zero_top_k_returns_empty(tmp_path: Path) -> None:
    capture = _seed_two_domains(tmp_path)
    capture.top_k = 0
    assert capture.find_similar("fetch news pdf") == []
