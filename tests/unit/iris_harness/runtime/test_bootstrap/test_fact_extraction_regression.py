"""Regression tests for fact-extraction safety + eval/sandbox isolation.

Incident this pins: exp-006 memory/identity prompts were parsed by the greedy
regex extractor into ``name='working on a project called Aur'`` and
``location='Berlin and I prefer very concise answers'`` and written to the real
``~/.iris`` profile + ``memory.db`` — because the extractor was greedy, there
was no validation gate, and evals drove the *live* instance with no isolation.

The three fixes, each pinned below:
  1. a plausibility gate at the persistence choke point (no fragment ever writes)
  2. a strict regex fallback (captures clean atomic facts, never a sentence)
  3. ``$IRIS_HOME`` / ``$IRIS_DATA_DIR`` to fully isolate a sandbox instance
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from iris_harness.memory.identity import loader
from iris_harness.runtime import build_runtime

CORRUPTION = (
    "I'm working on a project called Aur. " "I live in Berlin and I prefer very concise answers."
)


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A throwaway runtime whose data + profile writes land under ``tmp_path``."""
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    # Keep durable profile writes off the real ~/.iris/workspace/USER.md.
    monkeypatch.setattr(loader, "USER_MD_PATH", data_dir / "USER.md")
    return build_runtime(config_dir=config_dir, data_dir=data_dir, use_background_scheduler=False)


def test_validation_gate_blocks_corrupt_fact_from_any_extractor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even if an extractor emits a verb-phrase 'name', the gate stops the write."""
    runtime = _build(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runtime.capture,
        "_extract_facts_via_llm",
        lambda msg: [("name", "working on a project called Aur", 0.9, "test:fake")],
    )
    runtime.capture.extract_and_store_facts("tell me about my project")
    assert runtime.memory_store.fetch_user_fact("name") is None


def test_durability_gate_blocks_ephemeral_facts_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact pollution found in a real memory.db audit must never persist —
    greeting/day/task/reminder_time are ephemeral, not durable facts (issue 002x)."""
    runtime = _build(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runtime.capture,
        "_extract_facts_via_llm",
        lambda msg: [
            ("greeting", "Hello", 0.6, "test:fake"),
            ("day", "tomorrow", 0.6, "test:fake"),
            ("task", "plan", 0.3, "test:fake"),
            ("reminder_time", "6 pm tomorrow", 0.3, "test:fake"),
            ("name", "Robin", 0.95, "test:fake"),  # the one real fact survives
        ],
    )
    # Declarative message so the (stubbed) LLM path runs; the durability gate fires
    # BEFORE grounding, so the ephemeral entries are rejected as not-durable.
    runtime.capture.extract_and_store_facts("I'm Robin")
    proposed = {p.key: p.value for p in runtime.memory_store.fetch_fact_proposals()}
    for ephemeral in ("greeting", "day", "task", "reminder_time"):
        assert ephemeral not in proposed
        assert runtime.memory_store.fetch_user_fact(ephemeral) is None
    # The surviving fact is PROPOSED, not believed: extraction no longer decides what
    # IRIS believes about the user — the owner approves it (`iris facts approve`).
    # (Since memris PR 2c a proposal is a proposed statement: fetchable, never confirmed.)
    assert proposed == {"name": "Robin"}
    pending = runtime.memory_store.fetch_user_fact("name")
    assert pending is None or not pending.confirmed
    assert runtime.memory_store.fetch_all_user_facts(confirmed_only=True) == []


def test_hardened_regex_extracts_clean_facts_not_fragments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regex fallback yields clean atomic facts (location='Berlin'), never the
    full sentence, and never a bogus name from 'I'm working ...'."""
    monkeypatch.setenv("IRIS_FACT_EXTRACTION_MODE", "regex")
    runtime = _build(tmp_path, monkeypatch)
    runtime.capture.extract_and_store_facts(CORRUPTION)

    assert runtime.memory_store.fetch_user_fact("name") is None
    for fact in runtime.memory_store.fetch_all_user_facts():
        assert "prefer" not in fact.value.lower()
        assert "project called" not in fact.value.lower()
        assert len(fact.value.split()) <= 6


def test_iris_data_dir_env_relocates_memory_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``$IRIS_DATA_DIR`` moves the SQLite/Chroma stores off the real data dir."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    data_env = tmp_path / "sandbox-data"
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DATA_DIR", str(data_env))
    monkeypatch.setattr(loader, "USER_MD_PATH", tmp_path / "USER.md")
    build_runtime(config_dir=config_dir, use_background_scheduler=False)
    assert (data_env / "memory.db").exists()


def test_iris_home_env_isolates_profile_from_real_home() -> None:
    """``$IRIS_HOME`` (set by conftest before import) must relocate the profile
    paths off the developer's real ~/.iris — this is what keeps the suite and any
    in-process eval from ever corrupting USER.md / SOUL.md.
    """
    env_home = os.environ.get("IRIS_HOME")
    assert env_home, "conftest must set IRIS_HOME for test isolation"
    assert loader.iris_home() == Path(env_home)
    assert loader.user_md_path() == Path(env_home) / "workspace" / "USER.md"
    assert loader.soul_path() == Path(env_home) / "workspace" / "SOUL.md"
    # And critically: NOT the real home.
    assert loader.user_md_path() != Path.home() / ".iris" / "workspace" / "USER.md"
