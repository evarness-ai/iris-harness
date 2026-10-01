"""Tests for ``iris facts gate-eval`` — the garbage-in/out audit CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from iris_harness.cli.facts import facts_app

runner = CliRunner()


def test_gate_eval_default_corpus_renders() -> None:
    result = runner.invoke(facts_app, ["gate-eval"])
    assert result.exit_code == 0
    out = result.stdout
    assert "capture gates" in out and "recall filter" in out
    assert "false-admit" in out and "false-recall" in out
    assert "threshold sweep" in out


def test_gate_eval_custom_corpus(tmp_path: Path) -> None:
    corpus = [
        {
            "key": "name",
            "value": "Anita Rao",
            "message": "my name is anita rao",
            "confidence": 0.9,
            "label": "keep",
        },
        {
            "key": "greeting",
            "value": "hello",
            "message": "hello there",
            "confidence": 0.6,
            "label": "junk",
        },
    ]
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(corpus), encoding="utf-8")

    result = runner.invoke(facts_app, ["gate-eval", "--corpus", str(p)])
    assert result.exit_code == 0
    assert "2 candidates" in result.stdout


def test_gate_eval_bad_corpus_path() -> None:
    result = runner.invoke(facts_app, ["gate-eval", "--corpus", "/nonexistent/x.json"])
    assert result.exit_code == 1
