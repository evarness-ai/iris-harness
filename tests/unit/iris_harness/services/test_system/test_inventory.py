"""Runtime inventory (ADR-0069 fast-follow #5) — Ollama fetch is injected so the
parsing is hermetic; the rest is local reads (version, git, importlib)."""

from __future__ import annotations

import pytest

from iris_harness.services.system.inventory import (
    RuntimeInventory,
    _ollama_models,
    runtime_inventory,
)


def test_runtime_inventory_shape() -> None:
    inv = runtime_inventory(ollama_fetch=lambda _url: {"models": []})
    assert isinstance(inv, RuntimeInventory)
    assert inv.iris_version  # non-empty
    assert inv.python_version.count(".") >= 2
    assert "fastapi" in inv.packages  # key package surfaced
    assert inv.ollama_models == []


def test_ollama_models_parsed_from_tags() -> None:
    data = {
        "models": [
            {"name": "granite4:latest", "size": 123, "details": {"parameter_size": "7B"}},
            {"name": "llama3.2:3b", "size": 456, "details": {}},
            "garbage",  # non-dict entry is skipped, not crashed on
        ]
    }
    models = _ollama_models(fetch=lambda _url: data)
    assert [m["name"] for m in models] == ["granite4:latest", "llama3.2:3b"]
    assert models[0]["parameter_size"] == "7B"
    assert models[1]["parameter_size"] is None


def test_ollama_unreachable_yields_empty_not_error() -> None:
    def _boom(_url: str) -> dict[str, object]:
        raise OSError("connection refused")

    assert _ollama_models(fetch=_boom) == []


def test_ollama_models_wildcard_bind_host_queries_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_API_HOST", "0.0.0.0")  # noqa: S104
    seen: list[str] = []

    def _capture(url: str) -> dict[str, object]:
        seen.append(url)
        return {"models": []}

    _ollama_models(fetch=_capture)
    assert seen == ["http://127.0.0.1:11434/api/tags"]
