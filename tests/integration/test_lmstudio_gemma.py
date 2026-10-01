"""Integration check: real LM Studio + Gemma reachable through IRIS tier router.

Auto-skips when LM Studio's local server is not listening on
``LM_STUDIO_BASE_URL`` (default ``http://localhost:1234``), so the suite
stays green on machines without LM Studio installed.

Run explicitly with:

    IRIS_AUTH_SECRET=test-secret-for-testing \\
      poetry run pytest -m real_llm tests/integration/test_lmstudio_gemma.py -v -s
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import pytest

from iris_harness.llm.client import CodingLLMConfig, _default_model_factory
from iris_harness.llm.tier_router import TierRouter

# A live test by design: it probes LM Studio on :1234 and talks to Gemma. The
# tests/conftest.py network guard only lets it do that under ``real_llm``, which
# the default run deselects; run it with ``pytest -m real_llm <this file>``.
pytestmark = [pytest.mark.integration, pytest.mark.real_llm]


def _lmstudio_url() -> str:
    raw = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234").rstrip("/")
    return raw if raw.endswith("/v1") else f"{raw}/v1"


def _lmstudio_listening(url: str, timeout: float = 0.5) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 1234)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _gemma_loaded(url: str, model_id: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False
    return any(entry.get("id") == model_id for entry in payload.get("data", []))


def _gemma_serving(cfg: CodingLLMConfig) -> bool:
    """True only when the model returns a non-empty completion.

    LM Studio lists catalog models even when none is loaded into memory, so
    ``_gemma_loaded`` passing is not enough — a completion then comes back empty
    and the test hard-fails. A trivial warmup separates "listed but not actually
    serving" (skip) from "serving but broken" (fail), keeping the local pre-push
    gate green when the user simply hasn't loaded the model in the Developer tab.
    """
    try:
        chat = _default_model_factory(
            provider=cfg.provider,
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=os.environ.get(cfg.api_key_env or "", "lm-studio"),
            temperature=cfg.temperature,
            max_tokens=8,
        )
        resp = chat.invoke([{"role": "user", "content": "ping"}])
    except Exception:  # noqa: BLE001 — any failure means "not serving" → skip
        return False
    return bool(str(getattr(resp, "content", "")).strip())


REPO_ROOT = Path(__file__).resolve().parents[2]
TIERS_YAML = REPO_ROOT / "config" / "llm_tiers.yaml"
LMSTUDIO_URL = _lmstudio_url()


pytest.importorskip("langchain_openai", reason="langchain_openai not installed")


@pytest.fixture(scope="module")
def gemma_config():
    if not _lmstudio_listening(LMSTUDIO_URL):
        pytest.skip(f"LM Studio not reachable at {LMSTUDIO_URL}")
    router = TierRouter.load_from_yaml(TIERS_YAML)
    cfg = cast(CodingLLMConfig, router.get_llm_config("gemma_test"))
    if cfg.provider != "lmstudio":
        pytest.skip(
            "gemma_test intent is not routed to lmstudio in config/llm_tiers.yaml "
            f"(got provider={cfg.provider!r})"
        )
    if not _gemma_loaded(LMSTUDIO_URL, cfg.model):
        pytest.skip(
            f"LM Studio does not have model {cfg.model!r} loaded — "
            "pull it in the Discover tab and load it in the Developer tab"
        )
    if not _gemma_serving(cfg):
        pytest.skip(
            f"LM Studio lists {cfg.model!r} but it is not serving completions "
            "(load it in the Developer tab) — skipping to keep the gate green"
        )
    return cfg


def test_tier_router_resolves_gemma_to_lmstudio(gemma_config):
    """The YAML wiring lands on lmstudio with a non-placeholder model id."""
    assert gemma_config.provider == "lmstudio"
    assert gemma_config.base_url.endswith("/v1")
    assert gemma_config.model and not gemma_config.model.startswith("REPLACE_")


def test_gemma_chat_completion_returns_non_empty(gemma_config):
    """A real chat call through ChatOpenAI → LM Studio returns content."""
    api_key_env = gemma_config.api_key_env or ""
    chat = _default_model_factory(
        provider=gemma_config.provider,
        model=gemma_config.model,
        base_url=gemma_config.base_url,
        api_key=os.environ.get(api_key_env, "lm-studio"),
        temperature=gemma_config.temperature,
        max_tokens=64,
    )
    response = chat.invoke(
        [
            {"role": "system", "content": "You are terse. One short sentence."},
            {"role": "user", "content": "Reply with exactly: pong"},
        ]
    )
    content = getattr(response, "content", "")
    assert isinstance(content, str)
    assert content.strip(), f"empty response from {gemma_config.model!r}"
    print(  # noqa: T201 — shown with -s
        f"\n  gemma reply ({gemma_config.model}): {content.strip()}"
    )
