"""A tier's ``think`` option turns a reasoning model's thinking off (or on).

qwen3.5:4b thinks before answering: accurate, but twice as slow in the model eval
(2026-09-27). ``think: false`` in llm_tiers.yaml (or an app edit) reaches Ollama as
ChatOllama(reasoning=False); unset leaves the model's default; other providers ignore it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.llm import tier_edits
from iris_harness.llm.client import CodingLLMClient, CodingLLMConfig, _default_model_factory
from iris_harness.llm.tier_router import TierRouter

YAML = """
tiers:
  fast:
    provider: ollama
    model: qwen3.5:4b-q4_K_M
    think: false
    use_for: [general]
  plain:
    provider: ollama
    model: qwen2.5:7b-instruct
    use_for: [calendar]
  bad:
    provider: ollama
    model: qwen3.5:4b-q4_K_M
    think: "no"
    use_for: [files]
  remote:
    provider: lmstudio
    model: qwen3.5:4b
    think: false
    use_for: [search]
"""


@pytest.fixture
def router(tmp_path: Path) -> TierRouter:
    path = tmp_path / "llm_tiers.yaml"
    path.write_text(YAML, encoding="utf-8")
    return TierRouter.load_from_yaml(
        path, settings=SettingsStore(db_path=tmp_path / SETTINGS_DB_NAME)
    )


def test_the_yaml_option_reaches_the_llm_config(router: TierRouter) -> None:
    assert router.get_llm_config("general").think is False
    assert router.get_llm_config("calendar").think is None  # unset: the model's default
    assert router.get_llm_config("files").think is None  # not a boolean: ignored


def test_think_false_reaches_chat_ollama_as_reasoning_false() -> None:
    model = _default_model_factory(
        provider="ollama", model="qwen3.5:4b-q4_K_M", base_url="http://127.0.0.1:11434", think=False
    )
    assert model.reasoning is False  # type: ignore[attr-defined]
    unset = _default_model_factory(
        provider="ollama", model="qwen3.5:4b-q4_K_M", base_url="http://x"
    )
    assert unset.reasoning is None  # type: ignore[attr-defined]


def test_only_ollama_tiers_send_it() -> None:
    ollama = CodingLLMClient(
        CodingLLMConfig(provider="ollama", model="m", base_url="http://x", think=False)
    )
    assert ollama.model_kwargs().get("think") is False
    remote = CodingLLMClient(
        CodingLLMConfig(provider="lmstudio", model="m", base_url="http://x/v1", think=False)
    )
    assert "think" not in remote.model_kwargs()


def test_the_app_can_turn_it_on_and_off(router: TierRouter, tmp_path: Path) -> None:
    store = SettingsStore(db_path=tmp_path / SETTINGS_DB_NAME)
    tier_edits.update_tier(router, store, "plain", {"think": False}, actor="d")
    assert router.get_llm_config("calendar").think is False
    tier_edits.update_tier(router, store, "plain", {"think": True}, actor="d")
    assert router.get_llm_config("calendar").think is True


@pytest.mark.parametrize("value", ["false", 0, "off", 1])
def test_the_app_refuses_anything_but_a_boolean(value: object) -> None:
    with pytest.raises(tier_edits.TierEditError):
        tier_edits.validate_fields({"think": value})
