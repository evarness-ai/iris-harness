"""Tests for ProviderProfile.coding_model and intent-aware profile config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris_harness.llm.providers import BUILTIN_PROFILES, ProviderManager, ProviderProfile


def test_coding_model_round_trips_through_dict() -> None:
    profile = ProviderProfile(
        name="lmstudio",
        display_name="LM Studio",
        provider_type="lmstudio",
        base_url="http://localhost:1234/v1",
        model="google/gemma-4-e4b",
        api_key_env="LM_STUDIO_API_KEY",
        coding_model="qwen2.5-coder-14b-instruct",
    )
    payload = profile.to_dict()
    assert payload["coding_model"] == "qwen2.5-coder-14b-instruct"
    rebuilt = ProviderProfile.from_dict(payload)
    assert rebuilt.coding_model == "qwen2.5-coder-14b-instruct"


def test_from_dict_defaults_coding_model_to_none_when_absent() -> None:
    payload: dict[str, object] = {
        "name": "x",
        "base_url": "http://example/v1",
        "model": "m",
        "api_key_env": None,
    }
    profile = ProviderProfile.from_dict(payload)
    assert profile.coding_model is None


def test_builtin_local_profiles_pin_coding_model_to_qwen_coder() -> None:
    # Local providers must default the coding intent to a coding-tuned model
    # so /provider doesn't silently demote coding to a generalist chat model.
    assert BUILTIN_PROFILES["ollama"].coding_model == "qwen2.5-coder:7b"
    assert BUILTIN_PROFILES["lmstudio"].coding_model == "qwen2.5-coder-14b-instruct"


def test_config_from_profile_swaps_in_coding_model_for_coding_intent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Point ProviderManager at a temp dir so the test never touches ~/.iris.
    monkeypatch.setenv("IRIS_PROVIDERS_CONFIG", str(tmp_path / "providers.json"))
    monkeypatch.setattr("iris_harness.llm.providers._CONFIG_PATH", tmp_path / "providers.json")

    # Persist an lmstudio profile whose chat default and coding_model differ.
    (tmp_path / "providers.json").write_text(
        json.dumps(
            {
                "active": "lmstudio",
                "profiles": {
                    "lmstudio": {
                        "name": "lmstudio",
                        "display_name": "LM Studio",
                        "provider_type": "lmstudio",
                        "base_url": "http://localhost:1234/v1",
                        "model": "google/gemma-4-e4b",
                        "coding_model": "qwen2.5-coder-14b-instruct",
                        "api_key_env": "LM_STUDIO_API_KEY",
                    }
                },
            }
        )
    )

    # Force ProviderManager to load from the temp path.
    mgr = ProviderManager(config_path=tmp_path / "providers.json")
    monkeypatch.setattr(
        "iris_harness.llm.providers.ProviderManager",
        lambda *args, **kwargs: mgr,
    )

    from iris_harness.runtime.client_config import config_from_profile

    chat_cfg = config_from_profile("lmstudio")
    coding_cfg = config_from_profile("lmstudio", intent="coding")
    general_cfg = config_from_profile("lmstudio", intent="general")

    assert chat_cfg.model == "google/gemma-4-e4b"
    assert coding_cfg.model == "qwen2.5-coder-14b-instruct"
    # Named in the session log by where it came from; not an llm_tiers.yaml key.
    assert chat_cfg.tier_name == "profile:lmstudio"
    # Non-coding intents stay on the chat default.
    assert general_cfg.model == "google/gemma-4-e4b"


def test_config_from_profile_falls_back_when_coding_model_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IRIS_PROVIDERS_CONFIG", str(tmp_path / "providers.json"))
    monkeypatch.setattr("iris_harness.llm.providers._CONFIG_PATH", tmp_path / "providers.json")
    (tmp_path / "providers.json").write_text(
        json.dumps(
            {
                "active": "custom",
                "profiles": {
                    "custom": {
                        "name": "custom",
                        "display_name": "Custom",
                        "provider_type": "ollama",
                        "base_url": "http://localhost:11434/v1",
                        "model": "phi3:mini",
                        "api_key_env": None,
                    }
                },
            }
        )
    )
    mgr = ProviderManager(config_path=tmp_path / "providers.json")
    monkeypatch.setattr(
        "iris_harness.llm.providers.ProviderManager",
        lambda *args, **kwargs: mgr,
    )

    from iris_harness.runtime.client_config import config_from_profile

    cfg = config_from_profile("custom", intent="coding")
    assert cfg.model == "phi3:mini"


# --- fetch_models: shape handling + graceful fallback (issue: /model list crash) ---


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *a: object) -> None:
        return None


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, fn: object) -> None:
    monkeypatch.setattr("urllib.request.urlopen", fn)


def test_fetch_models_handles_top_level_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare JSON list (GitHub Models catalog) must parse, not raise AttributeError."""
    payload = json.dumps([{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]).encode()
    _patch_urlopen(monkeypatch, lambda *a, **k: _FakeResp(payload))
    listing = BUILTIN_PROFILES["github"].fetch_models()
    assert listing.usable == ["gpt-4o", "gpt-4o-mini"]
    assert listing.error is None and listing.from_fallback is False


def test_fetch_models_handles_data_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"data": [{"id": "x/y"}, {"id": "a/b"}]}).encode()
    _patch_urlopen(monkeypatch, lambda *a, **k: _FakeResp(payload))
    listing = BUILTIN_PROFILES["openrouter"].fetch_models()
    assert listing.usable == ["a/b", "x/y"]  # sorted


def test_fetch_models_http_error_falls_back_with_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 401 from the catalog degrades to the static fallback + a clear reason, no raise."""
    import urllib.error

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    def _raise(*a: object, **k: object) -> None:
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)  # type: ignore[arg-type]

    _patch_urlopen(monkeypatch, _raise)
    listing = BUILTIN_PROFILES["anthropic"].fetch_models()
    assert listing.from_fallback is True
    assert listing.error is not None and "401" in listing.error
    assert "claude-opus-4-8" in listing.usable


def test_fetch_models_empty_catalog_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    _patch_urlopen(monkeypatch, lambda *a, **k: _FakeResp(json.dumps({"data": []}).encode()))
    listing = BUILTIN_PROFILES["anthropic"].fetch_models()
    assert listing.from_fallback is True and listing.usable


def test_clean_model_id_strips_azureml_path() -> None:
    p = BUILTIN_PROFILES["github"]
    assert (
        p._clean_model_id("azureml://registries/azure-openai/models/gpt-4o-mini/versions/1")
        == "gpt-4o-mini"
    )
    assert p._clean_model_id("gpt-4o") == "gpt-4o"  # plain ids unchanged


def test_copilot_filter_keeps_non_picker_models() -> None:
    """Models absent from the VS Code picker are still API-callable — must not be hidden."""
    p = BUILTIN_PROFILES["copilot"]
    # not in picker but supports chat → usable now (previously hidden)
    assert (
        p._copilot_filter_reason({"model_picker_enabled": False, "policy": {"state": "enabled"}})
        is None
    )
    # genuinely no chat endpoint → still hidden
    assert (
        p._copilot_filter_reason({"supported_endpoints": ["/embeddings"]}) == "no /chat/completions"
    )
    # policy disabled → hidden
    assert p._copilot_filter_reason({"policy": {"state": "disabled"}}) == "policy disabled"


def test_embedding_models_filtered_from_chat_list(monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    payload = _json.dumps(
        [{"id": "gpt-4o"}, {"id": "text-embedding-3-small"}, {"id": "claude-sonnet-4.5"}]
    ).encode()

    class _Resp:
        def read(self) -> bytes:
            return payload

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    listing = BUILTIN_PROFILES["github"].fetch_models()
    assert "gpt-4o" in listing.usable and "claude-sonnet-4.5" in listing.usable
    assert "text-embedding-3-small" not in listing.usable
    assert any(m == "text-embedding-3-small" for m, _ in listing.hidden)
