"""Owner-PII shadow mode (ADR-0125 PR 4): every guard's would-be action audited, nothing changed.

- ``off`` (the default) registers nothing: byte-identical kernel, no rows.
- ``shadow`` registers one hook first at PRE_TOOL_USE / PRE_LLM_CALL / PRE_RESPONSE. Over a
  matrix of calls every real decision and final context is the same with it as without it,
  and its rows say, per guard x kind, what the table would do -- never the literal.
- Any other flag value (``enforce`` included) runs shadow and says so at startup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.agent.tool_runner import GovernedToolRunner, ToolCall
from iris_harness.cli.governance import governance_app
from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governance import GovernanceKernel, build_default_kernel
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.audit import digest as audit_digest
from iris_harness.kernel.governance.hooks.response_payload import pre_response_payload
from iris_harness.kernel.governance.hooks.tool_payload import TOOL_SENDS_TO, pre_tool_payload
from iris_harness.kernel.governance.hooks.types import HookContext, HookPoint
from iris_harness.kernel.governance.plugins import owner_pii_shadow as shadow_mod
from iris_harness.kernel.governance.plugins.network_egress import NetworkEgress
from iris_harness.kernel.governance.plugins.owner_pii_shadow import (
    HOOK_NAME,
    SHADOW_KEY,
    SHADOW_POINTS,
    owner_pii_shadow_summary,
    parse_owner_pii_mode,
)
from iris_harness.kernel.governance.wiring import kernel_from_env

# Imported here, before any fixture runs: importing the server imports the runtime, whose
# identity module registers the composition root's owner-identity sources as it loads --
# inside a test that would replace the synthetic corpus the fixture registered.
from iris_harness.server.iris_api.main import create_app

# Synthetic, low-entropy owner identity (never real data).
SECRET = "CANARY_OWNER_TOKEN_0001"
BLOG = "https://blog.robin.example/2026"
PII = {
    "name": "Robin Example",
    "first_name": "Robin",
    "email": "owner.canary@example.com",
    "phone": "+1 555 0100 0199",
    "address": "1 Example Street, Springfield",
    "handle": "@robin-gh",
}
# Every spelling of an owner literal that must never reach a row.
LITERALS = [
    *PII.values(),
    "robin-gh",
    "5550100",
    "0100 0199",
    "owner.canary",
    "example street",
    SECRET,
    BLOG,
    "blog.robin",
]


@pytest.fixture(autouse=True)
def _corpus(owner_identity_seam: Any) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(lambda: [f"my key: {SECRET}"])
    seam.register_owner_identity_source(
        "facts",
        lambda: {
            "name": ["Robin Example", "Robin"],
            "email": [PII["email"]],
            "phone": [PII["phone"]],
            "address": [PII["address"]],
            "handle": ["robin-gh"],
            "link": [BLOG],
        },
    )
    NetworkEgress._reset_identity_cache()


# -- helpers ---------------------------------------------------------------------------


def _kernel(tmp_path: Path, mode: str, **kwargs: Any) -> tuple[GovernanceKernel, AuditLog]:
    audit = AuditLog(db_path=tmp_path / f"audit-{mode}-{len(list(tmp_path.iterdir()))}.db")
    return build_default_kernel(audit_log=audit, owner_pii_mode=mode, **kwargs), audit  # type: ignore[arg-type]


def _tool_ctx(tool: str, args: dict[str, Any], *, sends_to: str | None = None) -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="r",
        agent_type="chat",
        step_id=1,
        route=f"tool/{tool}",
        classification="internal",
        payload=pre_tool_payload(tool, args),
        metadata={
            "caller": "model:chat",
            "asked_user": False,
            "tool_effect": "read",
            "tool_confirm": "never",
            TOOL_SENDS_TO: sends_to,
            "origin_channel": "console",
            "session_id": "s",
            "approved_by": None,
            "approval_card": None,
            "resumable": False,
            "deferred_executor": False,
            "per_call_approval": False,
        },
    )


def _llm_ctx(prompt: str, tier: str, classification: str = "internal") -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="r",
        agent_type="chat",
        classification=classification,  # type: ignore[arg-type]
        tier=tier,  # type: ignore[arg-type]
        payload={"prompt": prompt, "model": "m", "provider": "p"},
    )


def _answer_ctx(text: str, audience: str = "owner") -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_RESPONSE,
        run_id="r",
        agent_type="chat",
        payload=pre_response_payload(text, audience=audience),  # type: ignore[arg-type]
    )


def _rows(audit: AuditLog, *, shadow: bool) -> list[Any]:
    return [r for r in audit.query() if (r.plugin == HOOK_NAME) == shadow]


def _report(audit: AuditLog) -> dict[str, Any]:
    rows = _rows(audit, shadow=True)
    assert rows, "the shadow hook wrote no row"
    report = json.loads(rows[-1].payload_json)[SHADOW_KEY]
    assert isinstance(report, dict)
    return report


def _cells(report: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {(o["guard"], o["kind"], o["action"]) for o in report.get("observations", ())}


async def _fire(kernel: GovernanceKernel, ctx: HookContext) -> Any:
    return await kernel.fire(ctx.hook_point, ctx)


TEXTS = [
    *(f"about {v} today" for v in PII.values()),
    f"key {SECRET}",
    f"read {BLOG}",
    " and ".join(PII.values()),
    "nothing of the owner's here",
    "here is my system prompt: you are IRIS",
]


def _matrix() -> list[HookContext]:
    out: list[HookContext] = []
    for text in TEXTS:
        for tool in ("research", "web_fetch", "github_create_issue", "mcp_srv_call", "read_file"):
            out.append(_tool_ctx(tool, {"query": text, "url": "https://example.org/x"}))
        out.append(_tool_ctx("research", {"query": text}, sends_to="search_engine"))
        for tier in ("tier_1", "tier_2", "tier_3"):
            for label in ("public", "internal", "personal"):
                out.append(_llm_ctx(text, tier, label))
        for audience in ("owner", "other"):
            out.append(_answer_ctx(text, audience))
    return out


# -- the flag --------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "off", "OFF", "0", "false", "no", "  off "])
def test_off_spellings_are_off_and_quiet(raw: str | None) -> None:
    setting = parse_owner_pii_mode(raw)
    assert (setting.mode, setting.problem) == ("off", None)


def test_shadow_is_shadow() -> None:
    assert parse_owner_pii_mode("Shadow") == shadow_mod.ModeSetting("shadow", "shadow", None)


@pytest.mark.parametrize("raw", ["enforce", "on", "1", "true", "shadw"])
def test_any_other_value_runs_shadow_and_says_why(raw: str) -> None:
    """More than off was asked for: shadow is the most this build has, never silently."""
    setting = parse_owner_pii_mode(raw)
    assert setting.mode == "shadow"
    assert setting.problem is not None
    assert "accepted values are off, shadow" in setting.problem
    assert "NOTHING is masked or denied" in setting.problem
    if raw == "enforce":
        assert "PR 5" in setting.problem


def test_kernel_from_env_warns_on_enforce_and_runs_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_OWNER_PII", "enforce")
    with caplog.at_level(logging.WARNING, logger="iris_harness.kernel.governance.wiring"):
        kernel = kernel_from_env()
    assert kernel is not None
    assert any("enforce is not built yet" in r.getMessage() for r in caplog.records)
    for point in SHADOW_POINTS:
        assert kernel.hook_names(point)[0] == HOOK_NAME


@pytest.mark.parametrize("raw", [None, "off"])
def test_off_registers_nothing(
    raw: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    if raw is None:
        monkeypatch.delenv("IRIS_GOVERNANCE_OWNER_PII", raising=False)
    else:
        monkeypatch.setenv("IRIS_GOVERNANCE_OWNER_PII", raw)
    kernel = kernel_from_env()
    assert kernel is not None
    for point in HookPoint:
        assert HOOK_NAME not in kernel.hook_names(point)


def test_shadow_runs_first_at_its_three_points_only(tmp_path: Path) -> None:
    kernel, _ = _kernel(tmp_path, "shadow")
    for point in HookPoint:
        names = kernel.hook_names(point)
        assert (names[:1] == (HOOK_NAME,)) is (point in SHADOW_POINTS), point
        assert names.count(HOOK_NAME) == (1 if point in SHADOW_POINTS else 0)


# -- off is byte-identical; shadow changes no decision ---------------------------------


async def test_off_writes_no_shadow_row(tmp_path: Path) -> None:
    kernel, audit = _kernel(tmp_path, "off")
    for ctx in _matrix():
        await _fire(kernel, ctx)
    assert _rows(audit, shadow=True) == []


async def test_shadow_changes_no_decision_over_the_matrix(tmp_path: Path) -> None:
    """Same decision, same final context, same rows from every real guard, on or off."""
    off, off_audit = _kernel(tmp_path, "off")
    on, on_audit = _kernel(tmp_path, "shadow")
    matrix = _matrix()
    for ctx in matrix:
        d_off, c_off = await _fire(off, ctx)
        d_on, c_on = await _fire(on, ctx)
        skip = {"approval_request_id"}
        assert d_on.model_dump(exclude=skip) == d_off.model_dump(exclude=skip), ctx
        assert c_on.model_dump() == c_off.model_dump(), ctx

    def real(audit: AuditLog) -> list[tuple[str, ...]]:
        return [
            (r.hook_point, r.plugin, r.decision, r.severity, r.reason, r.payload_json)
            for r in _rows(audit, shadow=False)
        ]

    assert real(on_audit) == real(off_audit)
    assert len(_rows(on_audit, shadow=True)) == len(matrix)


async def test_a_failing_observation_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError(PII["email"])  # an exception that quotes what it read

    monkeypatch.setattr(shadow_mod, "decide", boom)
    off, _ = _kernel(tmp_path, "off")
    on, audit = _kernel(tmp_path, "shadow")
    ctx = _tool_ctx("web_fetch", {"url": "https://example.org", "q": PII["email"]})
    d_off, _ = await _fire(off, ctx)
    d_on, _ = await _fire(on, ctx)
    assert d_on.model_dump() == d_off.model_dump()
    row = _rows(audit, shadow=True)[-1]
    assert row.decision == "allow"
    assert json.loads(row.payload_json)[SHADOW_KEY] == {"checked": [], "error": "RuntimeError"}
    assert PII["email"] not in row.payload_json + row.reason


# -- what the rows say -----------------------------------------------------------------


@pytest.mark.parametrize("tool", ["research", "web_fetch"])
async def test_egress_rows_per_kind(tmp_path: Path, tool: str) -> None:
    kernel, audit = _kernel(tmp_path, "shadow")
    await _fire(kernel, _tool_ctx(tool, {"q": " | ".join([*PII.values(), SECRET, BLOG])}))
    report = _report(audit)
    assert report["checked"] == ["egress"]
    assert _cells(report) == {
        ("egress", "email", "deny"),
        ("egress", "phone", "deny"),
        ("egress", "address", "deny"),
        ("egress", "name", "log"),
        ("egress", "handle", "log"),
        ("egress", "secret", "deny"),
        ("egress", "link", "deny"),
    }
    first = [o for o in report["observations"] if o.get("first_name_alone")]
    assert [(o["kind"], o["action"], o["count"]) for o in first] == [("name", "log", 1)]


@pytest.mark.parametrize("tool", ["github_create_issue", "mcp_srv_call"])
async def test_log_only_destinations_log_pii_and_still_deny_secrets(
    tmp_path: Path, tool: str
) -> None:
    kernel, audit = _kernel(tmp_path, "shadow")
    await _fire(kernel, _tool_ctx(tool, {"body": f"{PII['email']} {PII['phone']} {SECRET}"}))
    by_kind = {o["kind"]: o for o in _report(audit)["observations"]}
    assert by_kind["email"]["action"] == "log" and by_kind["email"]["log_only_destination"]
    assert by_kind["phone"]["action"] == "log" and by_kind["phone"]["log_only_destination"]
    assert by_kind["secret"]["action"] == "deny" and "log_only_destination" not in by_kind["secret"]


async def test_a_non_network_tool_is_not_an_egress(tmp_path: Path) -> None:
    kernel, audit = _kernel(tmp_path, "shadow")
    await _fire(kernel, _tool_ctx("read_file", {"path": PII["email"]}))
    assert _report(audit) == {"checked": []}


async def test_web_search_is_the_declared_search_engine_tool(tmp_path: Path) -> None:
    kernel, audit = _kernel(tmp_path, "shadow")
    query = f"{PII['name']} {PII['email']} {PII['handle']} and Robin"
    await _fire(kernel, _tool_ctx("research", {"query": query}, sends_to="search_engine"))
    report = _report(audit)
    assert report["checked"] == ["egress", "web_search"]
    web = {
        (o["kind"], o["action"], bool(o.get("first_name_alone")))
        for o in report["observations"]
        if o["guard"] == "web_search"
    }
    assert web == {
        ("name", "mask", False),
        ("name", "mask", True),
        ("email", "deny", False),
        ("handle", "mask", False),
    }


async def test_a_declared_external_service_is_checked_as_egress(tmp_path: Path) -> None:
    """Issue #103: a plugin tool's arguments (a home address) leaving to its declared host
    are read by the egress column by declaration, though no name in a list says "network"."""
    kernel, audit = _kernel(tmp_path, "shadow")
    await _fire(
        kernel,
        _tool_ctx("weather_forecast", {"place": PII["email"]}, sends_to="external_service"),
    )
    report = _report(audit)
    assert report["checked"] == ["egress"]
    assert {(o["kind"], o["guard"]) for o in report["observations"]} == {("email", "egress")}


async def test_an_undeclared_plugin_tool_is_not_an_egress_call(tmp_path: Path) -> None:
    kernel, audit = _kernel(tmp_path, "shadow")
    await _fire(kernel, _tool_ctx("weather_forecast", {"place": PII["email"]}))
    assert _report(audit) == {"checked": []}


async def test_web_search_off_without_the_declaration(tmp_path: Path) -> None:
    """``research`` by name alone is not a search engine: the declaration is what counts."""
    kernel, audit = _kernel(tmp_path, "shadow")
    await _fire(kernel, _tool_ctx("research", {"query": PII["email"]}))
    assert _report(audit)["checked"] == ["egress"]


async def test_egress_is_not_shadowed_when_the_egress_guard_is_off(tmp_path: Path) -> None:
    kernel, audit = _kernel(tmp_path, "shadow", network_egress_enabled=False)
    await _fire(kernel, _tool_ctx("research", {"query": PII["email"]}, sends_to="search_engine"))
    assert _report(audit)["checked"] == ["web_search"]


async def test_tier3_placeholders_only_when_the_target_leaves_the_machine(tmp_path: Path) -> None:
    kernel, audit = _kernel(tmp_path, "shadow")
    prompt = f"{PII['name']} {PII['email']} {PII['phone']} {PII['address']} {SECRET}"
    await _fire(kernel, _llm_ctx(prompt, "tier_3"))
    report = _report(audit)
    assert report["checked"] == ["tier3"]
    assert _cells(report) == {
        ("tier3", "name", "pass"),
        ("tier3", "email", "placeholder"),
        ("tier3", "phone", "placeholder"),
        ("tier3", "address", "placeholder"),
        ("tier3", "secret", "placeholder"),
    }
    for local in ("tier_1", "tier_2"):
        await _fire(kernel, _llm_ctx(prompt, local))
        assert _report(audit) == {"checked": []}, local


async def test_the_answer_column_follows_the_audience(tmp_path: Path) -> None:
    kernel, audit = _kernel(tmp_path, "shadow")
    text = f"{PII['name']}, {PII['email']}; hi Robin; {SECRET}"
    await _fire(kernel, _answer_ctx(text, "owner"))
    owner = _report(audit)
    await _fire(kernel, _answer_ctx(text, "other"))
    other = _report(audit)
    assert owner["checked"] == ["answer_owner"] and other["checked"] == ["answer_other"]
    assert _cells(owner) == {
        ("answer_owner", "name", "pass"),
        ("answer_owner", "email", "pass"),
        ("answer_owner", "secret", "halt"),
    }
    assert _cells(other) == {
        ("answer_other", "name", "mask"),
        ("answer_other", "name", "pass"),  # the first name alone never masks an answer
        ("answer_other", "email", "mask"),
        ("answer_other", "secret", "halt"),
    }
    rows = _rows(audit, shadow=True)
    assert [json.loads(r.payload_json)["audience"] for r in rows] == ["owner", "other"]


async def test_no_row_ever_holds_a_literal(tmp_path: Path) -> None:
    audit_digest.audit_digester()  # digests on too: they must not leak either
    kernel, audit = _kernel(tmp_path, "shadow")
    for ctx in _matrix():
        await _fire(kernel, ctx)
    rows = _rows(audit, shadow=True)
    assert any(json.loads(r.payload_json)[SHADOW_KEY].get("observations") for r in rows)
    blob = "\n".join(r.payload_json + r.reason for r in rows).lower()
    for literal in LITERALS:
        assert literal.lower() not in blob, literal


async def test_spans_point_at_the_occurrence(tmp_path: Path) -> None:
    kernel, audit = _kernel(tmp_path, "shadow")
    args = {"a": "nothing", "b": ["x", f"mail {PII['email']} now"]}
    await _fire(kernel, _tool_ctx("web_fetch", args))
    (obs,) = [o for o in _report(audit)["observations"] if o["kind"] == "email"]
    leaf, start, end = obs["spans"][0]
    leaves = ["nothing", "x", f"mail {PII['email']} now"]
    assert leaves[leaf][start:end] == PII["email"]


# -- the audit key ---------------------------------------------------------------------


async def test_digests_only_when_the_key_is_already_resolved(tmp_path: Path) -> None:
    kernel, audit = _kernel(tmp_path, "shadow")
    ctx = _tool_ctx("web_fetch", {"a": PII["email"], "b": PII["email"].upper()})
    await _fire(kernel, ctx)
    before = _report(audit)
    assert "digest_alg" not in before and "digests" not in before["observations"][0]
    # The observer never resolves the key itself (no Keychain prompt from an observer).
    assert audit_digest.audit_key_status()[0] == "unresolved"

    audit_digest.audit_digester()
    await _fire(kernel, ctx)
    after = _report(audit)
    (obs,) = after["observations"]
    assert obs["count"] == 2 and len(obs["digests"]) == 1  # one literal, two spellings
    assert after["digest_alg"].startswith("hmac-sha256/v1/")


@pytest.mark.usefixtures("no_vault_master_key")
async def test_no_master_key_still_observes(tmp_path: Path) -> None:
    with pytest.raises(audit_digest.AuditKeyUnavailable):
        audit_digest.audit_digester()
    off, _ = _kernel(tmp_path, "off")
    on, audit = _kernel(tmp_path, "shadow")
    for ctx in (_llm_ctx(PII["email"], "tier_3"), _answer_ctx(PII["email"], "other")):
        d_off, _ = await _fire(off, ctx)
        d_on, _ = await _fire(on, ctx)
        assert d_on.model_dump() == d_off.model_dump()
        report = _report(audit)
        assert report["observations"] and "digests" not in report["observations"][0]


def test_a_governed_tool_call_is_shadowed_on_the_real_runner(tmp_path: Path) -> None:
    """The runner resolves the key before PRE_TOOL_USE (#741), stamps ``sends_to``, and the
    shadow row carries digests; the tool still runs."""
    kernel, audit = _kernel(tmp_path, "shadow")
    runner = GovernedToolRunner(kernel=kernel, agent_type="chat")
    tool = ToolSpec(
        name="research",
        description="search",
        call=lambda args: "results",
        sends_to="search_engine",
    )
    outcome = runner.execute(tool, {"query": PII["email"]}, ToolCall(run_id="r1", step_id=1))
    rows = [r for r in _rows(audit, shadow=True) if r.hook_point == "pre_tool_use"]
    report = json.loads(rows[-1].payload_json)[SHADOW_KEY]
    assert report["checked"] == ["egress", "web_search"]
    assert _cells(report) == {("egress", "email", "deny"), ("web_search", "email", "deny")}
    assert all(o["digests"] for o in report["observations"])
    # Shadow only: the egress guard does not refuse an email yet, so the call ran.
    assert outcome.text == "results"


# -- degraded corpus / table -----------------------------------------------------------


async def test_no_identity_is_reported_not_guessed(
    tmp_path: Path, owner_identity_seam: Any
) -> None:
    owner_identity_seam.clear_identity_text_provider()
    for name in owner_identity_seam.owner_identity_sources():
        owner_identity_seam.unregister_owner_identity_source(name)
    kernel, audit = _kernel(tmp_path, "shadow")
    await _fire(kernel, _answer_ctx(PII["email"]))
    assert _report(audit) == {"checked": ["answer_owner"], "note": "no_identity"}


async def test_an_unreadable_table_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> Any:
        raise ValueError("bad table")

    monkeypatch.setattr(shadow_mod, "guard_table", broken)
    kernel, audit = _kernel(tmp_path, "shadow")
    await _fire(kernel, _answer_ctx(PII["email"]))
    assert _report(audit)["note"] == "table_unreadable"


# -- the read-back ---------------------------------------------------------------------


async def _seed(tmp_path: Path) -> AuditLog:
    kernel, audit = _kernel(tmp_path, "shadow")
    await _fire(kernel, _tool_ctx("web_fetch", {"q": f"{PII['email']} {PII['email']}"}))
    await _fire(kernel, _tool_ctx("github_create_issue", {"q": PII["email"]}))
    await _fire(kernel, _llm_ctx(PII["phone"], "tier_3"))
    await _fire(kernel, _llm_ctx(PII["phone"], "tier_1"))
    await _fire(kernel, _answer_ctx(f"hi Robin, {PII['email']}", "other"))
    return audit


async def test_the_summary_counts_guard_by_kind_by_action(tmp_path: Path) -> None:
    audit = await _seed(tmp_path)
    summary = owner_pii_shadow_summary(audit, days=1)
    cells = {(c.guard, c.kind, c.action, c.log_only_destination): c for c in summary.cells}
    email_deny = cells[("egress", "email", "deny", False)]
    assert (email_deny.occurrences, email_deny.calls, email_deny.distinct) == (2, 1, None)
    assert cells[("egress", "email", "log", True)].occurrences == 1
    assert cells[("tier3", "phone", "placeholder", False)].calls == 1
    assert cells[("answer_other", "email", "mask", False)].calls == 1
    first = [c for c in summary.cells if c.first_name_alone]
    assert [(c.guard, c.action) for c in first] == [("answer_other", "pass")]
    assert summary.checked == {"egress": 2, "tier3": 1, "answer_other": 1}
    assert summary.rows == 5
    blob = json.dumps(summary.as_dict()).lower()
    assert not any(literal.lower() in blob for literal in LITERALS)


async def test_the_cli_renders_the_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit = await _seed(tmp_path)
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(audit.db_path))
    monkeypatch.setenv("IRIS_GOVERNANCE_OWNER_PII", "shadow")
    result = CliRunner().invoke(governance_app, ["pii-shadow", "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["mode"] == "shadow" and body["rows"] == 5
    text = CliRunner().invoke(governance_app, ["pii-shadow"])
    assert text.exit_code == 0 and "tier3 1" in text.output
    assert not any(literal.lower() in text.output.lower() for literal in LITERALS)


async def test_the_api_renders_the_summary_and_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = await _seed(tmp_path)
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(audit.db_path))
    monkeypatch.setenv("IRIS_GOVERNANCE_OWNER_PII", "enforce")
    with TestClient(create_app(auto_start_runtime=False), headers=auth_headers()) as client:
        body = client.get("/governance/pii-shadow", params={"days": 1}).json()
        flags = {f["key"]: f for f in client.get("/governance/state").json()["flags"]}
        assert client.get("/governance/pii-shadow", params={"days": 0}).status_code == 422
    assert body["mode"] == "shadow" and body["rows"] == 5 and body["cells"], body
    assert flags["IRIS_GOVERNANCE_OWNER_PII"] == {
        "key": "IRIS_GOVERNANCE_OWNER_PII",
        "label": "Owner-PII guards",
        "on": True,
        "value": "shadow",
    }


def test_the_audience_key_is_audited() -> None:
    from iris_harness.kernel.governance.kernel import _AUDITED_PAYLOAD_KEYS

    assert "audience" in _AUDITED_PAYLOAD_KEYS


def test_the_research_manifest_declares_its_search_engine() -> None:
    from iris_harness.runtime.plugin_host.manifest import load_manifest

    root = Path(__file__).resolve().parents[5]
    manifest = load_manifest(root / "src/iris_harness/plugins_builtin/research/manifest.yaml")
    assert manifest.tools["research"].sends_to == "search_engine"
