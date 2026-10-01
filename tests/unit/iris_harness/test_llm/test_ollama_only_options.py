"""``num_ctx``, ``keep_alive`` and ``num_thread`` reach Ollama and nothing else.

They are Ollama options. ChatOpenAI puts unknown keywords into every request, and the
OpenAI client refuses them ("Completions.create() got an unexpected keyword argument
'num_ctx'"). In the FoundationModels bake-off (2026-09-28) the router tier was moved to
``lmstudio`` with its ``num_ctx`` kept: every call failed, and the router fell back to
keywords without saying so. An app edit that moves a tier across providers does the same.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.llm.client import CodingLLMClient, _default_model_factory
from iris_harness.llm.tier_router import TierRouter

YAML = """
tiers:
  local:
    provider: ollama
    model: llama3.2:3b
    num_ctx: 2048
    keep_alive: "10m"
    num_thread: 4
    use_for: [general]
  moved:
    provider: lmstudio
    model: apple-fm
    num_ctx: 2048
    keep_alive: "10m"
    num_thread: 4
    use_for: [intent_classification]
"""

OLLAMA_ONLY = ("num_ctx", "keep_alive", "num_thread")


@pytest.fixture
def router(tmp_path: Path) -> TierRouter:
    path = tmp_path / "llm_tiers.yaml"
    path.write_text(YAML, encoding="utf-8")
    return TierRouter.load_from_yaml(
        path, settings=SettingsStore(db_path=tmp_path / SETTINGS_DB_NAME)
    )


def _kwargs(router: TierRouter, intent: str) -> dict[str, object]:
    return CodingLLMClient(router.get_llm_config(intent)).model_kwargs()


def test_ollama_gets_its_options(router: TierRouter) -> None:
    kwargs = _kwargs(router, "general")
    assert (kwargs["num_ctx"], kwargs["keep_alive"], kwargs["num_thread"]) == (2048, "10m", 4)


def test_an_openai_style_provider_never_gets_them(router: TierRouter) -> None:
    kwargs = _kwargs(router, "intent_classification")
    assert not set(OLLAMA_ONLY) & set(kwargs)


def test_the_openai_request_carries_no_ollama_option(router: TierRouter) -> None:
    # The failure itself: ChatOpenAI moved num_ctx into model_kwargs, which it sends as
    # arguments to Completions.create().
    model = _default_model_factory(**_kwargs(router, "intent_classification"))
    assert not set(OLLAMA_ONLY) & set(getattr(model, "model_kwargs", {}) or {})
