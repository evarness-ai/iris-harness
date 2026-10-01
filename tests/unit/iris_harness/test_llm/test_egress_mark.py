"""The failover proxy's data-free mark (llm/egress.py): only an opted-in call carries it."""

from __future__ import annotations

from iris_harness.llm.client import CodingLLMClient, CodingLLMConfig
from iris_harness.llm.egress import MAY_LEAVE_HEADER, data_free_call, egress_headers


def _client(provider: str = "lmstudio") -> CodingLLMClient:
    return CodingLLMClient(
        CodingLLMConfig(provider=provider, model="granite4:latest", base_url="http://p/v1")
    )


def test_a_call_is_unmarked_unless_it_opts_in() -> None:
    headers = _client().model_kwargs().get("default_headers") or {}
    assert MAY_LEAVE_HEADER not in headers  # private by default: the proxy keeps it local


def test_a_data_free_call_carries_the_mark_and_only_inside() -> None:
    with data_free_call():
        assert _client().model_kwargs()["default_headers"][MAY_LEAVE_HEADER] == "1"
    assert egress_headers() == {}
    assert MAY_LEAVE_HEADER not in (_client().model_kwargs().get("default_headers") or {})


def test_ollama_calls_take_no_headers_at_all() -> None:
    with data_free_call():
        assert "default_headers" not in _client("ollama").model_kwargs()


def test_no_call_site_opts_in_yet() -> None:
    """Every current prompt carries the profile, memory, conversation or tool results
    (router, agent, curator). An opt-in needs a test proving its prompt clean."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[4] / "src"
    opted_in = [
        f
        for f in src.rglob("*.py")
        if "data_free_call()" in f.read_text(encoding="utf-8") and f.name != "egress.py"
    ]
    assert opted_in == []
