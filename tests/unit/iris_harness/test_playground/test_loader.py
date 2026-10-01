"""Loader tests — YAML parsing, discovery, and error surfacing."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.playground.loader import discover_suites, load_suite

_SUITE_YAML = """
name: sample
description: a couple of cases
env:
  IRIS_DUES_INTERCEPT: "1"
scenarios:
  - name: dues
    message: "any insurance dues this week?"
    tags: [finance, intercept]
    expect:
      handler: dues_request
      intent: finance
      sources_exclude: [research]
      no_pii_leak: true
  - name: greeting
    message: "hello"
    expect:
      handler: ""
"""


def test_load_suite_parses_scenarios(tmp_path: Path) -> None:
    path = tmp_path / "sample.yaml"
    path.write_text(_SUITE_YAML, encoding="utf-8")
    suite = load_suite(path)
    assert suite.name == "sample"
    assert suite.env == {"IRIS_DUES_INTERCEPT": "1"}
    assert len(suite.scenarios) == 2
    dues = suite.scenarios[0]
    assert dues.expect.handler == "dues_request"
    assert dues.expect.sources_exclude == ("research",)
    assert dues.expect.no_pii_leak is True
    assert suite.scenarios[1].expect.handler == ""


def test_load_suite_defaults_name_to_stem(tmp_path: Path) -> None:
    path = tmp_path / "my_suite.yaml"
    path.write_text("scenarios:\n  - name: a\n    message: hi\n", encoding="utf-8")
    assert load_suite(path).name == "my_suite"


def test_load_suite_rejects_bad_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.yaml"):
        load_suite(path)


def test_load_suite_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "typo.yaml"
    path.write_text(
        "scenarios:\n  - name: a\n    message: hi\n    expect:\n      bogus_field: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="typo.yaml"):
        load_suite(path)


def test_discover_suites_finds_yaml(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("scenarios: []\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.yaml").write_text("scenarios: []\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignore me", encoding="utf-8")
    found = discover_suites(tmp_path)
    assert [p.name for p in found] == ["a.yaml", "b.yaml"]


def test_discover_suites_empty_when_missing(tmp_path: Path) -> None:
    assert discover_suites(tmp_path / "nope") == []
