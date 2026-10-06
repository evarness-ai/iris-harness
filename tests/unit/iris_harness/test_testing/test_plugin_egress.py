"""A plugin's outbound HTTP is declared, allowed and recorded (issue #103).

Driven through the real governed harness, reading the real ledger: a tool that calls
``api.http`` produces a ``pre_egress`` and a ``post_egress`` row naming the host, the
plugin, the tool and the caller on ``chat`` and ``chat_stream`` alike; an undeclared host
is denied before anything is sent. ``fake_http`` replaces only the transport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from iris_harness.foundation.ids import ULID_LENGTH, is_ulid
from iris_harness.kernel.governance.plugin_egress import (
    EgressScope,
    bind_egress_kernel,
    egress_scope,
)
from iris_harness.runtime.governed_http import GovernedHttp, current_http
from iris_harness.sdk import PluginAPI
from iris_harness.sdk.audit import AuditLog
from iris_harness.sdk.http import EgressDenied
from iris_harness.testing import fake_http, harness, no_network, plugin

QUERY_SECRET = "tok-plain-value-123"
PATH_SECRET = "path-plain-value-456"
FORECAST = f"https://api.open-meteo.com/v1/{PATH_SECRET}/forecast"


def _script(url: str) -> dict[str, Any]:
    return {
        "rules": [
            {
                "name": "answer from the tool",
                "match": {"user": r"(?s)Observation:.*?(status \d+|denied|network error)"},
                "reply": {"content": "Thought: Done.\nFinal Answer: Forecast handled."},
            },
            {
                "name": "call the tool",
                "match": {"user": r"User: What is the forecast"},
                "reply": {
                    "content": "Thought: Use the tool.\nAction: forecast\n"
                    f"Action Input: {json.dumps({'url': url})}"
                },
            },
        ]
    }


def _manifest(**extra: Any) -> dict[str, Any]:
    return {
        "name": "weather",
        "provides": ["tool"],
        "tools": {"forecast": {"effect": "read", "content": "external"}},
        **extra,
    }


_DECLARED = {
    "hosts": [
        "api.open-meteo.com",
        {"host": "geocoding-api.open-meteo.com", "data": "personal"},
    ]
}


def _weather(manifest: Any = None) -> Any:
    def setup(api: PluginAPI) -> None:
        http = api.http

        def forecast(args: dict[str, Any]) -> str:
            try:
                reply = http.get(args["url"], params={"token": QUERY_SECRET, "days": 3}, timeout=2)
            except EgressDenied as exc:
                return f"denied: {exc.host}"
            except httpx.HTTPError as exc:
                return f"network error: {type(exc).__name__}"
            return f"status {reply.status_code}: {reply.text}"

        api.register_tool("forecast", "Get a forecast. Args: {'url': str}.", forecast)

    return plugin(setup, manifest=manifest or _manifest(egress=_DECLARED))


def _rows(h: Any, point: str) -> list[Any]:
    return [r for r in h.audit_rows(hook_point=point) if r.egress is not None]


def _run(entry: str, h: Any) -> None:
    if entry == "chat":
        h.chat("What is the forecast for Berlin?")
    else:
        assert h.chat_stream("What is the forecast for Berlin?").answered


def _raw(h: Any) -> list[str]:
    return [r.payload_json for r in AuditLog(db_path=h.audit_db).query()]


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_declared_host_is_allowed_and_recorded_with_its_destination(entry: str) -> None:
    with fake_http({FORECAST: {"json": {"daily": [1, 2, 3]}}}) as sent:
        with harness(plugins=[_weather()], fake_model=_script_for(FORECAST)) as h:
            _run(entry, h)
            pre, post = _rows(h, "pre_egress"), _rows(h, "post_egress")
            tool_pre = [r for r in h.audit_rows(hook_point="pre_tool_use") if r.tool == "forecast"]
    assert len(sent) == 1 and sent[0].url.host == "api.open-meteo.com"
    assert len(pre) == 1 and len(post) == 1
    allowed, outcome = pre[0], post[0]
    assert (allowed.plugin, allowed.decision) == ("plugin_egress", "allow")
    assert allowed.egress is not None and outcome.egress is not None
    assert allowed.egress["host"] == "api.open-meteo.com"
    assert allowed.egress["method"] == "GET" and allowed.egress["port"] == 443
    assert allowed.egress["data"] == "internal"  # the shorthand's declared class
    assert outcome.egress["status"] == 200 and outcome.egress["bytes_in"] > 0
    assert outcome.egress["duration_ms"] >= 0 and "error" not in outcome.egress
    # Named like the tool's own rows, and joined to them by the turn's run id and session.
    for row in (allowed, outcome):
        assert (row.tool, row.tool_plugin, row.caller) == ("forecast", "weather", "model:system")
    assert allowed.run_id == tool_pre[0].run_id
    assert allowed.session_id == tool_pre[0].session_id


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_an_undeclared_host_is_denied_and_nothing_is_sent(entry: str) -> None:
    url = "https://evil.example/collect"
    with fake_http({url: {"text": "should never be reached"}}) as sent:
        with harness(plugins=[_weather()], fake_model=_script_for(url)) as h:
            _run(entry, h)
            pre, post = _rows(h, "pre_egress"), _rows(h, "post_egress")
            observed = [r for r in h.audit_rows(hook_point="post_tool_use") if r.tool == "forecast"]
    assert sent == []
    assert [(r.decision, r.egress["host"]) for r in pre] == [("deny", "evil.example")]
    assert "not in plugin 'weather'" in pre[0].reason
    assert post == []  # a denied request has no outcome: it never happened
    assert observed  # the tool still ran and told the model; the row records the refusal


@pytest.mark.parametrize("party", ["first-party", "trusted-third-party", "untrusted"])
def test_a_plugin_that_declares_no_egress_may_contact_nothing_whatever_its_party(
    party: str,
) -> None:
    with fake_http({FORECAST: {"json": {}}}) as sent:
        with harness(
            plugins=[_weather(_manifest(party=party))], fake_model=_script_for(FORECAST)
        ) as h:
            _run("chat", h)
            pre = _rows(h, "pre_egress")
    assert sent == []
    assert [r.decision for r in pre] == ["deny"]
    assert "declares no egress" in pre[0].reason


def test_the_ledger_never_holds_the_path_the_query_or_the_body() -> None:
    with fake_http({FORECAST: {"text": "ok"}}):
        with harness(plugins=[_weather()], fake_model=_script_for(FORECAST)) as h:
            _run("chat", h)
            raw = _raw(h)
    assert any('"egress"' in payload for payload in raw)
    for payload in raw:
        assert QUERY_SECRET not in payload and PATH_SECRET not in payload


def test_a_manifest_read_from_disk_declares_the_same_egress(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(_manifest(egress=_DECLARED)), encoding="utf-8")
    with fake_http({FORECAST: {"json": {}}}) as sent:
        with harness(plugins=[_weather(path)], fake_model=_script_for(FORECAST)) as h:
            _run("chat", h)
            assert [r.decision for r in _rows(h, "pre_egress")] == ["allow"]
    assert len(sent) == 1


def test_a_real_transport_under_no_network_fails_and_the_outcome_row_says_so() -> None:
    # (the harness refuses sockets itself; the outer guard says the same for a bare test)
    with no_network():
        with harness(plugins=[_weather()], fake_model=_script_for(FORECAST)) as h:
            _run("chat", h)
            post = _rows(h, "post_egress")
    assert [r.egress.get("error") for r in post if r.egress] == ["ConnectError"]


def test_a_redirect_is_returned_not_followed() -> None:
    hop = {"status": 302, "headers": {"location": "https://evil.example/x"}}
    with fake_http({FORECAST: hop}) as sent:
        with harness(plugins=[_weather()], fake_model=_script_for(FORECAST)) as h:
            _run("chat", h)
    assert [str(r.url.host) for r in sent] == ["api.open-meteo.com"]


def test_a_request_outside_a_tool_call_is_attributed_to_the_plugin() -> None:
    with fake_http({FORECAST: {"text": "ok"}}):
        with harness(plugins=[_weather()], fake_model={"default": {"content": "hi"}}) as h:
            GovernedHttp("weather").get(FORECAST)
            [row] = _rows(h, "pre_egress")
    assert (row.caller, row.tool_plugin, row.tool) == ("plugin:weather", "weather", None)


def test_the_row_names_the_client_s_plugin_not_the_scope_s_claim() -> None:
    """The plugin is the harness's stamp on the client; the surrounding call cannot rename it."""
    scope = EgressScope(run_id="r", agent_type="chat", tool="t", tool_plugin="someone-else")
    with fake_http({FORECAST: {"text": "ok"}}):
        with harness(plugins=[_weather()], fake_model={"default": {"content": "hi"}}) as h:
            with egress_scope(scope):
                GovernedHttp("weather").get(FORECAST)
            [row] = _rows(h, "pre_egress")
    assert (row.decision, row.tool_plugin, row.tool) == ("allow", "weather", "t")


async def test_the_async_client_is_governed_the_same_way() -> None:
    with fake_http({FORECAST: {"text": "ok"}}) as sent:
        with harness(plugins=[_weather()], fake_model={"default": {"content": "hi"}}) as h:
            reply = await GovernedHttp("weather").arequest("GET", FORECAST)
            with pytest.raises(EgressDenied):
                await GovernedHttp("weather").arequest("GET", "https://evil.example/")
            decisions = [r.decision for r in _rows(h, "pre_egress")]
            outcomes = _rows(h, "post_egress")
    assert reply.status_code == 200 and len(sent) == 1
    assert decisions == ["allow", "deny"] and len(outcomes) == 1


def test_without_a_kernel_the_client_fails_closed() -> None:
    bind_egress_kernel(None)
    with fake_http({FORECAST: {"text": "ok"}}) as sent:
        with pytest.raises(EgressDenied, match="no governance kernel"):
            GovernedHttp("weather").get(FORECAST)
    assert sent == []


def test_a_kernel_without_the_egress_hook_fails_closed() -> None:
    """An empty kernel allows everything ("no hooks registered"); the client must not."""
    from iris_harness.kernel.governance import GovernanceKernel

    kernel = GovernanceKernel()
    kernel.init_lock()
    bind_egress_kernel(lambda: kernel)
    try:
        with fake_http({FORECAST: {"text": "ok"}}) as sent:
            with pytest.raises(EgressDenied, match="no plugin_egress hook"):
                GovernedHttp("weather").get(FORECAST)
    finally:
        bind_egress_kernel(None)
    assert sent == []


def test_a_malformed_request_is_refused_and_still_leaves_a_row() -> None:
    """A refusal is a ledger row, never silence (#134)."""
    with fake_http({}) as sent:
        with harness(plugins=[_weather()], fake_model={"default": {"content": "hi"}}) as h:
            with pytest.raises(EgressDenied, match="credentials"):
                GovernedHttp("weather").get("https://user:pw@api.open-meteo.com/x")
            with pytest.raises(EgressDenied, match="host"):
                GovernedHttp("weather").get("/relative")
            rows = _rows(h, "pre_egress")
    assert sent == []
    assert [r.decision for r in rows] == ["deny", "deny"]
    assert "credentials" in rows[0].reason and "pw" not in rows[0].reason


def _twice_plugin() -> Any:
    def twice(api: PluginAPI) -> None:
        http = api.http

        def forecast(args: dict[str, Any]) -> str:
            http.get(FORECAST)
            http.get(FORECAST)
            return "status 200: ok"

        api.register_tool("forecast", "Forecast.", forecast)

    return plugin(twice, manifest=_manifest(egress=_DECLARED))


def _ledger(h: Any) -> list[tuple[Any, dict[str, Any]]]:
    return [(r, json.loads(r.payload_json)) for r in AuditLog(db_path=h.audit_db).query()]


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_each_egress_record_has_its_own_id_and_links_to_its_parent_call(entry: str) -> None:
    """#134 stage 1: a request's id is a ULID the kernel stamps from metadata (never the
    plugin's), shared by its PRE and POST rows; ``parent_call_id`` is the calling tool's
    runner-minted ``call_id`` (the id on that tool's own PRE/POST_TOOL_USE rows); a repeat
    inside one call is a replay of the first request."""
    with fake_http({FORECAST: {"text": "ok"}}):
        with harness(plugins=[_twice_plugin()], fake_model=_script_for(FORECAST)) as h:
            _run(entry, h)
            pre, post = _rows(h, "pre_egress"), _rows(h, "post_egress")
            ledger = _ledger(h)

    tool_ids = {
        p["call_id"]
        for r, p in ledger
        if r.hook_point in ("pre_tool_use", "post_tool_use") and p.get("tool_name") == "forecast"
    }
    assert len(tool_ids) == 1
    [tool_call_id] = tool_ids
    assert is_ulid(tool_call_id)

    first, second = (r.egress for r in pre)
    assert first is not None and second is not None
    assert first["call_id"] != second["call_id"]
    assert is_ulid(first["call_id"]) and is_ulid(second["call_id"])
    assert len(first["call_id"]) == ULID_LENGTH
    assert second["call_id"] > first["call_id"]  # time-sortable, strictly increasing
    assert [r.egress["call_id"] for r in post if r.egress] == [first["call_id"], second["call_id"]]
    assert (first["attempt"], first["replay_of"]) == (1, None)
    assert (second["attempt"], second["replay_of"]) == (2, first["call_id"])
    assert first["parent_call_id"] == second["parent_call_id"] == tool_call_id
    assert tool_call_id not in (first["call_id"], second["call_id"])

    # The ids sit on the ROW (kernel stamp), the same fields a tool row carries; the
    # nested egress record is not where they are written.
    egress_rows = [p for r, p in ledger if r.hook_point in ("pre_egress", "post_egress")]
    assert len(egress_rows) == 4
    for payload in egress_rows:
        assert is_ulid(payload["call_id"])
        assert payload["parent_call_id"] == tool_call_id
        assert "call_id" not in payload["egress"] and "parent_call_id" not in payload["egress"]
    # Every ledger row of the egress hooks, i.e. every hook that fired for the request.
    assert {p["call_id"] for _, p in ledger if p.get("egress")} == {
        first["call_id"],
        second["call_id"],
    }


def test_a_denied_request_still_leaves_rows_with_a_ulid_and_its_parent() -> None:
    """A refusal is a record carrying the same identity as an allowed request."""
    with fake_http({}):
        with harness(
            plugins=[_weather()], fake_model=_script_for("https://evil.example.org/x")
        ) as h:
            _run("chat", h)
            ledger = _ledger(h)
    tool_ids = {
        p["call_id"]
        for r, p in ledger
        if r.hook_point == "pre_tool_use" and p.get("tool_name") == "forecast"
    }
    assert len(tool_ids) == 1
    [tool_call_id] = tool_ids
    denied = [(r, p) for r, p in ledger if r.hook_point == "pre_egress" and r.decision == "deny"]
    assert denied
    for _, payload in denied:
        assert is_ulid(payload["call_id"]) and payload["call_id"] != tool_call_id
        assert payload["parent_call_id"] == tool_call_id


def test_a_failed_request_posts_a_row_with_the_same_id() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with fake_http(boom):
        with harness(plugins=[_weather()], fake_model=_script_for(FORECAST)) as h:
            _run("chat", h)
            ledger = _ledger(h)
    pre = [p for r, p in ledger if r.hook_point == "pre_egress" and r.decision == "allow"]
    post = [p for r, p in ledger if r.hook_point == "post_egress"]
    assert pre and post
    assert {p["call_id"] for p in pre} == {p["call_id"] for p in post}
    assert post[0]["egress"]["error"] == "ConnectError"


def test_a_request_outside_a_governed_call_has_an_id_and_no_parent() -> None:
    with fake_http({FORECAST: {"text": "ok"}}):
        with harness(plugins=[_weather()], fake_model={"default": {"content": "hi"}}) as h:
            GovernedHttp("weather").get(FORECAST)
            ledger = _ledger(h)
    rows = [p for r, p in ledger if r.hook_point in ("pre_egress", "post_egress")]
    assert len(rows) == 2
    assert all(is_ulid(p["call_id"]) for p in rows)
    assert len({p["call_id"] for p in rows}) == 1
    assert all("parent_call_id" not in p for p in rows)


def test_the_client_takes_no_caller_chosen_call_id() -> None:
    """The id is the harness's: there is no argument to name it, and a payload or
    metadata-less context cannot (see kernel/test_governance/test_call_id_stamp.py)."""
    with fake_http({FORECAST: {"text": "ok"}}):
        with harness(plugins=[_weather()], fake_model={"default": {"content": "hi"}}) as h:
            with pytest.raises(TypeError):
                GovernedHttp("weather").get(FORECAST, call_id="FORGED")  # type: ignore[call-arg]
            GovernedHttp("weather").get(FORECAST, headers={"call_id": "FORGED"})
            ledger = _ledger(h)
    assert all("FORGED" not in p.get("call_id", "") for _, p in ledger)


def test_current_http_is_the_plugin_whose_tool_is_running() -> None:
    with pytest.raises(EgressDenied, match="only available"):
        current_http()
    scope = EgressScope(run_id="r", agent_type="chat", tool="t", tool_plugin="weather")
    with egress_scope(scope):
        assert current_http().plugin == "weather"


def _script_for(url: str) -> dict[str, Any]:
    return _script(url)
