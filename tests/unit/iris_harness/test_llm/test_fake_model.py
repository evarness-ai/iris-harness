"""The scripted fake model (``provider: fake``, llm/fake.py).

Pinned: it replaces only the transport -- a real ``CodingLLMClient`` built on it fires
the governance hooks and writes the audit rows exactly as on Ollama; it plays Ollama's
schema-constrained decoder for ``invoke_json``; it answers tool calls in the shape the
client normalizes; a call no rule answers is an error, never a silent default; and
``IRIS_LLM_PROVIDER`` puts every tier on a *declared* provider only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.llm import fake
from iris_harness.llm.client import (
    CodingLLMClient,
    CodingLLMConfig,
    LLMMessage,
    supports_json_schema,
)
from iris_harness.llm.fake import FakeModelError, Script
from iris_harness.llm.tier_router import FORCED_PROVIDER_ENV, TierRouter

SCHEMA = {
    "type": "object",
    "properties": {
        "bucket": {"type": "string"},
        "confidence": {"type": "number"},
        "due_date": {"type": ["string", "null"]},
    },
    "required": ["bucket", "confidence", "due_date"],
}

SCRIPT = {
    "rules": [
        {
            "name": "bill",
            "match": {"system": "sort ONE", "user": r"Amount due: \$(?P<amount>[\d.]+)"},
            "reply": {"json": {"bucket": "bill", "confidence": "{amount}"}},
        },
        {
            "name": "tool",
            "match": {"tool": "^search_inbox$"},
            "reply": {"tool_calls": [{"name": "search_inbox", "arguments": {"q": "{x}"}}]},
        },
        {
            "name": "hello",
            "match": {"user": "(?P<who>[A-Z][a-z]+) says hi"},
            "reply": {"content": "Hi {who}!"},
        },
    ]
}


@pytest.fixture(autouse=True)
def _script() -> Any:
    fake.install_script(Script.from_mapping(SCRIPT))
    fake.reset_transcript()
    yield
    fake.install_script(None)


@pytest.fixture(autouse=True)
def _audit_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "audit.db"))


def _client(tmp_path: Path) -> tuple[CodingLLMClient, AuditLog]:
    """A client on the fake with the production kernel (``kernel_from_env``)."""
    config = CodingLLMConfig(provider="fake", model="demo-model", base_url="fake://scripted")
    return CodingLLMClient(config), AuditLog(db_path=tmp_path / "audit.db")


def test_a_json_call_is_governed_and_audited_like_a_real_one(tmp_path: Path) -> None:
    client, log = _client(tmp_path)
    reply = client.invoke_json(
        system_prompt="You sort ONE email", user_prompt="Amount due: $0.93", schema=SCHEMA
    )
    # "{amount}" alone became a number; the decoder filled the nullable required key.
    assert reply.data == {"bucket": "bill", "confidence": 0.93, "due_date": None}
    rows = log.query()
    assert {r.hook_point for r in rows} == {"pre_classify", "pre_llm_call"}
    assert len({r.run_id for r in rows}) == 1  # one call, one governed run
    llm_rows = [r for r in rows if r.hook_point == "pre_llm_call"]
    assert all('"provider": "fake"' in r.payload_json for r in llm_rows)
    assert {r.tier for r in llm_rows} == {"tier_1"}  # declared runs: local
    assert [c.rule for c in fake.transcript()] == ["bill"]
    assert fake.transcript()[0].json_schema is True
    assert client.get_token_usage()["call_count"] == 1


def test_a_missing_required_non_nullable_key_is_an_error(tmp_path: Path) -> None:
    fake.install_script(
        Script.from_mapping({"rules": [{"match": {}, "reply": {"json": {"bucket": "x"}}}]})
    )
    client, _log = _client(tmp_path)
    with pytest.raises(FakeModelError, match="confidence"):
        client.invoke_json(system_prompt="s", user_prompt="u", schema=SCHEMA)


def test_tool_calls_come_back_in_the_clients_shape(tmp_path: Path) -> None:
    client, _log = _client(tmp_path)
    tools = [{"type": "function", "function": {"name": "search_inbox", "parameters": {}}}]
    result = client.invoke_turn(
        messages=[LLMMessage(role="user", content="find the invoice")], bound_tools=tools
    )
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("search_inbox", {"q": "{x}"})]


def test_text_replies_fill_named_groups_and_stream(tmp_path: Path) -> None:
    client, _log = _client(tmp_path)
    assert client.invoke(system_prompt="", user_prompt="Lena says hi") == "Hi Lena!"
    assert "".join(client.invoke_stream(system_prompt="", user_prompt="Omar says hi")) == "Hi Omar!"


def test_a_call_no_rule_answers_raises(tmp_path: Path) -> None:
    client, _log = _client(tmp_path)
    with pytest.raises(FakeModelError, match="no fake-model rule"):
        client.invoke(system_prompt="", user_prompt="something unscripted")


def test_a_script_is_validated_when_loaded() -> None:
    with pytest.raises(FakeModelError, match="unknown match keys"):
        Script.from_mapping({"rules": [{"match": {"sender": "x"}, "reply": {"content": "y"}}]})
    with pytest.raises(FakeModelError, match="needs content, json or tool_calls"):
        Script.from_mapping({"rules": [{"match": {}, "reply": {}}]})


def test_the_env_named_script_is_read_when_none_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake.install_script(None)
    path = tmp_path / "script.yaml"
    path.write_text("default:\n  content: from the file\n", encoding="utf-8")
    monkeypatch.setenv(fake.SCRIPT_ENV, str(path))
    client, _log = _client(tmp_path)
    assert client.invoke(system_prompt="", user_prompt="anything") == "from the file"


def test_the_fake_takes_a_json_schema_and_lmstudio_still_does_not() -> None:
    assert supports_json_schema("fake") and supports_json_schema("ollama")
    assert not supports_json_schema("lmstudio")


def _tiers(tmp_path: Path, providers: str) -> Path:
    path = tmp_path / "llm_tiers.yaml"
    path.write_text(
        f"providers:\n{providers}"
        "tiers:\n"
        "  tier1:\n    provider: ollama\n    model: small\n    use_for: [general]\n"
        "  email_judge:\n    provider: ollama\n    model: judge\n    use_for: [email_judge]\n",
        encoding="utf-8",
    )
    return path


def test_iris_llm_provider_puts_every_tier_on_a_declared_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FORCED_PROVIDER_ENV, "fake")
    router = TierRouter.load_from_yaml(
        _tiers(tmp_path, "  ollama:\n    runs: local\n  fake:\n    runs: local\n")
    )
    config = router.get_llm_config("email_judge")
    assert isinstance(config, CodingLLMConfig)
    assert (config.provider, config.model, config.governance_tier) == ("fake", "judge", "tier_1")
    assert {t.provider for t in router._tiers.values()} == {"fake"}


def test_an_undeclared_forced_provider_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FORCED_PROVIDER_ENV, "fake")
    router = TierRouter.load_from_yaml(_tiers(tmp_path, "  ollama:\n    runs: local\n"))
    assert {t.provider for t in router._tiers.values()} == {"ollama"}


def test_the_shipped_tiers_declare_the_fake_local() -> None:
    from iris_harness.llm.locality import declared_localities

    assert declared_localities(Path("config/llm_tiers.yaml"))["fake"] == "local"
