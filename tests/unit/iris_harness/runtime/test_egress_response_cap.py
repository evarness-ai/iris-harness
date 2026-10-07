"""``egress.max_response_bytes``: the response cap a plugin's manifest may set (issue #175).

Default 10 MiB. Lowering is always allowed. The harness ceiling is 64 MiB and a larger value is
refused when the manifest loads, so the plugin does not mount. A plugin whose cap is not the
default says so on every ``pre_egress`` row (``cap``) and in ``iris plugins``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governance import build_default_kernel
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugin_egress import (
    DEFAULT_RESPONSE_BYTES,
    MAX_RESPONSE_BYTES_CEILING,
    EgressScope,
    HostRule,
    PluginEgress,
    PluginEgressPolicy,
    bind_egress_kernel,
    egress_scope,
    register_egress_policy,
)
from iris_harness.runtime.governed_http import EgressDenied, GovernedHttp
from iris_harness.runtime.plugin_host.manifest import PluginEgressDecl, load_manifest
from iris_harness.testing import fake_http

MIB = 1024 * 1024
HOST = "api.example.org"
URL = f"https://{HOST}/big"


def _bind(tmp_path: Path, cap: int | None) -> AuditLog:
    log = AuditLog(db_path=tmp_path / "audit.db")
    register_egress_policy(
        PluginEgressPolicy(
            {"p": PluginEgress(hosts=(HostRule(HOST, data="personal"),), max_response_bytes=cap)}
        )
    )
    kernel = build_default_kernel(audit_log=log)
    bind_egress_kernel(lambda: kernel)
    return log


@pytest.fixture(autouse=True)
def _in_a_call() -> Iterator[None]:
    with egress_scope(EgressScope(run_id="r1", agent_type="chat", tool="t", tool_plugin="p")):
        yield
    register_egress_policy(None)
    bind_egress_kernel(None)


def _rows(log: AuditLog, point: str) -> list[dict[str, Any]]:
    return [json.loads(r.payload_json) for r in log.query() if r.hook_point == point]


# -- the manifest ----------------------------------------------------------------------------


def test_the_key_is_optional_and_the_default_is_ten_mib() -> None:
    assert DEFAULT_RESPONSE_BYTES == 10 * MIB and MAX_RESPONSE_BYTES_CEILING == 64 * MIB
    decl = PluginEgressDecl(hosts=("api.example.org",))  # type: ignore[arg-type]
    assert decl.max_response_bytes is None and "max_response_bytes" not in decl.summary()
    assert decl.compile().max_response_bytes is None


@pytest.mark.parametrize("value", [1, 1000, 10 * MIB, 20 * MIB, 64 * MIB])
def test_a_value_up_to_the_ceiling_is_accepted_and_compiled(value: int) -> None:
    decl = PluginEgressDecl(hosts=("api.example.org",), max_response_bytes=value)  # type: ignore[arg-type]
    assert decl.summary()["max_response_bytes"] == value
    assert decl.compile().max_response_bytes == value


@pytest.mark.parametrize("value", [64 * MIB + 1, 1024 * MIB, 0, -5])
def test_a_value_above_the_ceiling_or_not_positive_is_refused(value: int) -> None:
    with pytest.raises(ValueError):
        PluginEgressDecl(hosts=("api.example.org",), max_response_bytes=value)  # type: ignore[arg-type]


def test_a_manifest_above_the_ceiling_does_not_load(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        f"name: greedy\negress:\n  hosts: [api.example.org]\n  max_response_bytes: {65 * MIB}\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="max_response_bytes"):
        load_manifest(path)
    path.write_text(
        f"name: ok\negress:\n  hosts: [api.example.org]\n  max_response_bytes: {64 * MIB}\n",
        encoding="utf-8",
    )
    assert load_manifest(path).egress.max_response_bytes == 64 * MIB


def test_a_hand_built_declaration_is_clamped_to_the_ceiling() -> None:
    policy = PluginEgressPolicy({"p": PluginEgress(max_response_bytes=10**12)})
    assert policy.response_cap("p") == MAX_RESPONSE_BYTES_CEILING
    assert policy.response_cap("unknown") == DEFAULT_RESPONSE_BYTES


# -- the client -------------------------------------------------------------------------------


def test_a_lowered_cap_cuts_the_response_off(tmp_path: Path) -> None:
    log = _bind(tmp_path, 1000)
    with fake_http({URL: {"content": b"x" * 2000}}):
        with pytest.raises(EgressDenied, match="larger than 1000 bytes"):
            GovernedHttp("p").get(URL)
        # a response within the cap is untouched
    with fake_http({URL: {"content": b"x" * 900}}):
        assert len(GovernedHttp("p").get(URL).content) == 900
    posts = _rows(log, "post_egress")
    assert posts[0]["egress"]["aborted"] == "max_bytes" and "aborted" not in posts[1]["egress"]


def test_a_raised_cap_lets_a_larger_response_through_and_the_default_does_not(
    tmp_path: Path,
) -> None:
    body = b"y" * (11 * MIB)
    _bind(tmp_path, 20 * MIB)
    with fake_http({URL: {"content": body}}):
        assert len(GovernedHttp("p").get(URL).content) == 11 * MIB
    register_egress_policy(
        PluginEgressPolicy({"p": PluginEgress(hosts=(HostRule(HOST, data="personal"),))})
    )
    with fake_http({URL: {"content": body}}), pytest.raises(EgressDenied, match="10 MiB"):
        GovernedHttp("p").get(URL)


def test_a_cap_other_than_the_default_is_recorded_on_the_pre_row_and_the_default_is_not(
    tmp_path: Path,
) -> None:
    log = _bind(tmp_path, 20 * MIB)
    with fake_http({URL: {"text": "ok"}}):
        GovernedHttp("p").get(URL)
    [pre] = _rows(log, "pre_egress")
    assert pre["egress"]["cap"] == 20 * MIB
    register_egress_policy(
        PluginEgressPolicy({"p": PluginEgress(hosts=(HostRule(HOST, data="personal"),))})
    )
    with fake_http({URL: {"text": "ok"}}):
        GovernedHttp("p").get(URL)
    pres = _rows(log, "pre_egress")
    assert "cap" not in pres[-1]["egress"]


# -- shown to the owner ---------------------------------------------------------------------------


def test_iris_plugins_shows_a_changed_cap() -> None:
    from iris_harness.cli.plugins import egress_lines

    raised = {"open_web": False, "hosts": [], "max_response_bytes": 20 * MIB}
    [line] = egress_lines(raised)
    assert "response cap 20480 KiB" in line and "raised above" in line
    lowered = {"open_web": True, "hosts": [], "max_response_bytes": 1024}
    assert any("lowered from" in line for line in egress_lines(lowered))
    assert egress_lines({"open_web": False, "hosts": []}) == []


def test_a_hook_that_rewrites_the_egress_dict_cannot_raise_the_cap(tmp_path: Path) -> None:
    """The cap is read from the compiled policy and passed to the transfer; the dict hooks see
    is only the record of the request."""
    log = _bind(tmp_path, 1000)
    client = GovernedHttp("p")
    real = client._execute

    def tamper(request: Any, seconds: float, read: Any, egress: dict[str, Any], cap: int) -> Any:
        egress["cap"] = 10**9  # what a hook mutating the shared dict could do
        return real(request, seconds, read, egress, cap)

    client._execute = tamper  # type: ignore[method-assign]
    with (
        fake_http({URL: {"content": b"x" * 2000}}),
        pytest.raises(EgressDenied, match="1000 bytes"),
    ):
        client.get(URL)
    assert _rows(log, "pre_egress")[0]["egress"]["cap"] == 1000
