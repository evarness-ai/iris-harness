"""The governed structured call, ``CodingLLMClient.invoke_json``.

It replaced ``llm/ollama_json.py``, a raw ``POST /api/chat`` that skipped governance: no
pre-LLM hooks, no audit row, no egress line. Pinned here: every attempt runs the governed
``invoke_turn`` (hooks, audit, egress, the shared breaker); the request asks Ollama for
the schema with thinking off; one retry on a non-JSON reply; and "the server is down"
stays distinct (``LLMUnreachable``), so a batch caller can stop at once.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.llm import client as client_module
from iris_harness.llm.arbiter import OllamaCircuitBreaker
from iris_harness.llm.client import (
    CodingLLMClient,
    CodingLLMConfig,
    CodingLLMInvocationError,
    LLMBadReply,
    LLMUnreachable,
    _default_model_factory,
)

SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
BASE_URL = "http://mac.example:11434/v1"


class _Reply:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None
        self.additional_kwargs: dict[str, Any] = {}
        self.usage_metadata = None
        self.response_metadata: dict[str, Any] = {}


class _Model:
    """A fake chat model: answers from a list, records the factory kwargs it was built with."""

    def __init__(self, answers: list[Any], built: list[dict[str, Any]]) -> None:
        self.answers = answers
        self.built = built

    def factory(self, **kwargs: Any) -> _Model:
        self.built.append(kwargs)
        return self

    def invoke(self, messages: Any, **_kw: Any) -> _Reply:
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _Reply(str(answer))


class _Hook:
    priority: int = 10

    def __init__(self, name: str, hook_point: HookPoint, outcome: str = "allow") -> None:
        self.name = name
        self.hook_point = hook_point
        self.outcome = outcome

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome=self.outcome, reason="test")  # type: ignore[arg-type]


def _kernel(tmp_path: Path, llm_outcome: str = "allow") -> tuple[GovernanceKernel, AuditLog]:
    log = AuditLog(db_path=tmp_path / "audit.db")
    kernel = GovernanceKernel(audit_log=log)
    kernel.register(_Hook("classify", HookPoint.PRE_CLASSIFY))
    kernel.register(_Hook("llm", HookPoint.PRE_LLM_CALL, llm_outcome))
    kernel.init_lock()
    return kernel, log


def _client(
    model: _Model,
    *,
    kernel: GovernanceKernel | None = None,
    breaker: OllamaCircuitBreaker | None = None,
    provider: str = "ollama",
) -> CodingLLMClient:
    config = CodingLLMConfig(
        provider=provider, model="judge:4b", tier_name="email_judge", base_url=BASE_URL
    )
    return CodingLLMClient(
        config,
        model_factory=model.factory,
        governance_kernel=kernel,
        governance_handled_upstream=kernel is None,
        governance_agent_type="chat",
        circuit_breaker=breaker or OllamaCircuitBreaker(),
    )


def _ask(client: CodingLLMClient) -> Any:
    return client.invoke_json(system_prompt="sys", user_prompt="usr", schema=SCHEMA)


def test_the_reply_is_the_json_object_and_the_model_is_asked_for_the_schema() -> None:
    built: list[dict[str, Any]] = []
    reply = _ask(_client(_Model(['{"label": "x"}'], built)))
    assert reply.data == {"label": "x"}
    assert reply.model == "judge:4b"
    assert reply.latency_ms >= 0
    assert built[0]["format"] == SCHEMA
    assert built[0]["think"] is False  # a reasoning model otherwise may answer empty


def test_every_attempt_is_governed_audited_and_egress_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    egress: list[dict[str, Any]] = []
    monkeypatch.setattr(client_module, "log_egress", lambda **kw: egress.append(kw))
    kernel, log = _kernel(tmp_path)
    built: list[dict[str, Any]] = []
    reply = _ask(_client(_Model(["not json", '{"label": "y"}'], built), kernel=kernel))
    assert reply.data == {"label": "y"}
    rows = log.query()
    assert [r.hook_point for r in rows] == ["pre_classify", "pre_llm_call"] * 2
    assert {r.agent_type for r in rows} == {"chat"}
    assert [e["kind"] for e in egress] == ["llm", "llm"]
    assert egress[0]["destination"] == "mac.example:11434"


def test_a_governance_refusal_sends_nothing(tmp_path: Path) -> None:
    kernel, _log = _kernel(tmp_path, llm_outcome="deny")
    built: list[dict[str, Any]] = []
    with pytest.raises(CodingLLMInvocationError, match="governance"):
        _ask(_client(_Model(['{"label": "x"}'], built), kernel=kernel))
    assert built == []  # the model was never built, let alone called


def test_two_replies_without_a_json_object_are_a_bad_reply() -> None:
    model = _Model(["nope", "[1, 2]"], [])
    with pytest.raises(LLMBadReply):
        _ask(_client(model))
    assert model.answers == []


@pytest.mark.parametrize(
    "error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")], ids=["refused", "timeout"]
)
def test_a_connection_failure_is_unreachable_and_not_retried(error: Exception) -> None:
    model = _Model([error, '{"label": "x"}'], [])
    with pytest.raises(LLMUnreachable):
        _ask(_client(model))
    assert len(model.answers) == 1  # never retried, never another endpoint


def test_an_open_breaker_is_unreachable_without_a_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_OLLAMA_BREAKER", raising=False)
    breaker = OllamaCircuitBreaker(failure_threshold=1, cooldown_seconds=60)
    model = _Model([httpx.ConnectError("refused"), '{"label": "x"}'], [])
    client = _client(model, breaker=breaker)
    with pytest.raises(LLMUnreachable):
        _ask(client)
    with pytest.raises(LLMUnreachable, match="(?i)circuit"):
        _ask(client)
    assert model.answers == ['{"label": "x"}']  # the second ask never reached the model
    assert breaker.is_open(BASE_URL)  # the chat clients' key: one down Mac trips them all


def test_another_provider_is_refused_before_anything_is_sent() -> None:
    built: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="ollama"):
        _ask(_client(_Model(['{"label": "x"}'], built), provider="lmstudio"))
    assert built == []


def test_the_ollama_factory_hands_format_and_think_to_chat_ollama() -> None:
    model = _default_model_factory(
        provider="ollama",
        model="judge:4b",
        base_url=BASE_URL,
        temperature=0.1,
        max_tokens=512,
        format=SCHEMA,
        think=False,
    )
    assert model.format == SCHEMA  # type: ignore[attr-defined]
    assert model.reasoning is False  # type: ignore[attr-defined]


# -- the wire: a real ChatOllama against a fake Ollama server --------------------------


class _FakeOllama(BaseHTTPRequestHandler):
    bodies: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # http.server calls this name
        length = int(self.headers.get("Content-Length") or 0)
        type(self).bodies.append(json.loads(self.rfile.read(length) or b"{}"))
        line = {
            "model": "judge:4b",
            "created_at": "2026-09-29T00:00:00Z",
            "message": {"role": "assistant", "content": '{"label": "wire"}'},
            "done": True,
            "done_reason": "stop",
        }
        payload = (json.dumps(line) + "\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args: Any) -> None:
        return


@pytest.fixture
def fake_ollama() -> Iterator[str]:
    _FakeOllama.bodies = []
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()


def test_the_request_on_the_wire_asks_for_the_schema_with_thinking_off(fake_ollama: str) -> None:
    config = CodingLLMConfig(
        provider="ollama", model="judge:4b", tier_name="email_judge", base_url=fake_ollama
    )
    client = CodingLLMClient(
        config, governance_handled_upstream=True, circuit_breaker=OllamaCircuitBreaker()
    )
    reply = client.invoke_json(system_prompt="sys", user_prompt="usr", schema=SCHEMA)
    assert reply.data == {"label": "wire"}
    [body] = _FakeOllama.bodies
    assert body["format"] == SCHEMA
    assert body["think"] is False
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
