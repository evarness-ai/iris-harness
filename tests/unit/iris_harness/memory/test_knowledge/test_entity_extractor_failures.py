"""A failed LLM entity pass keeps the rule-based entities, and now logs it (review 2026-09-26)."""

from __future__ import annotations

import logging
from typing import Any, NoReturn

import pytest

from iris_harness.memory.knowledge.entity_extractor import EntityExtractor

_LOGGER = "iris_harness.memory.knowledge.entity_extractor"
_TEXT = "Met Alice Johnson at Northwind Bank today to talk about the new savings account plan."


def _boom(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise TimeoutError("model did not answer")


def test_a_failed_llm_pass_keeps_the_regex_entities_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    baseline = {e.name for e in EntityExtractor().extract(_TEXT)}
    extractor = EntityExtractor(llm_call=lambda _prompt: "")
    monkeypatch.setattr(extractor, "_invoke_with_governance", _boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        found = {e.name for e in extractor.extract(_TEXT)}

    assert found == baseline
    [record] = [r for r in caplog.records if r.name == _LOGGER]
    assert record.levelno == logging.WARNING
    assert "LLM pass failed (TimeoutError)" in record.getMessage()
    assert "Alice" not in record.getMessage()
