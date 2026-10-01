"""Egress allowlist for the code_exec sandbox (exp-006 GAP-16).

Covers the host matcher (both copies — host-side + the standalone proxy's), the
configured allowlist, and DockerSandbox argv in allowlist / fail-closed / isolated modes.
These do NOT exercise Docker — argv-shape + pure-logic only (the live block/allow is the
separate integration check).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.tools.sandbox import egress, egress_proxy
from iris_harness.tools.sandbox.docker_sandbox import DockerSandbox
from iris_harness.tools.sandbox.workspace import SessionWorkspace

# Both matchers must behave identically (the proxy's copy is stdlib-only and can't import
# egress, so the logic is intentionally duplicated — these tests pin them in lockstep).
_MATCHERS = [egress.host_allowed, egress_proxy._host_allowed]


@pytest.mark.parametrize("matcher", _MATCHERS)
def test_matcher_allows_exact_and_subdomain(matcher) -> None:
    al = ("pypi.org", "github.com")
    assert matcher("pypi.org", al) is True
    assert matcher("files.pypi.org", al) is True
    assert matcher("github.com", al) is True
    assert matcher("raw.github.com", al) is True


@pytest.mark.parametrize("matcher", _MATCHERS)
def test_matcher_denies_and_resists_bypass(matcher) -> None:
    al = ("github.com",)
    assert matcher("evil.com", al) is False
    assert matcher("github.com.evil.com", al) is False  # suffix-confusion
    assert matcher("notgithub.com", al) is False  # prefix-confusion
    assert matcher("1.2.3.4", al) is False  # IP literal never matches a domain allowlist
    assert matcher("github.com", ()) is False  # empty allowlist = deny all


def test_configured_allowlist_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(egress._ALLOWLIST_ENV, raising=False)
    assert egress.configured_allowlist() == egress.DEFAULT_ALLOWLIST
    monkeypatch.setenv(egress._ALLOWLIST_ENV, "example.com, foo.org ")
    assert egress.configured_allowlist() == ("example.com", "foo.org")


def _sandbox(tmp_path: Path, **kwargs) -> DockerSandbox:
    return DockerSandbox(SessionWorkspace("egress-test", root=tmp_path), **kwargs)


def test_argv_open_mode_has_no_network_flags(tmp_path: Path) -> None:
    argv = _sandbox(tmp_path)._build_argv("echo hi", container_name="c")
    assert "--network=none" not in argv
    assert egress.INTERNAL_NETWORK not in argv
    assert not any("PROXY" in a for a in argv)


def test_argv_allowlist_mode_uses_internal_net_and_proxy(tmp_path: Path) -> None:
    argv = _sandbox(tmp_path, egress_allowlist=("pypi.org",))._build_argv(
        "echo hi", container_name="c", egress_ready=True
    )
    assert "--network" in argv and egress.INTERNAL_NETWORK in argv
    assert "--network=none" not in argv
    joined = " ".join(argv)
    assert f"HTTPS_PROXY={egress.proxy_url()}" in joined
    assert f"HTTP_PROXY={egress.proxy_url()}" in joined


def test_argv_allowlist_fails_closed_when_proxy_unavailable(tmp_path: Path) -> None:
    argv = _sandbox(tmp_path, egress_allowlist=("pypi.org",))._build_argv(
        "echo hi", container_name="c", egress_ready=False
    )
    assert "--network=none" in argv  # fail CLOSED, never open
    assert egress.INTERNAL_NETWORK not in argv
    assert not any("PROXY" in a for a in argv)


def test_argv_network_false_isolates_even_with_allowlist(tmp_path: Path) -> None:
    argv = _sandbox(tmp_path, network=False, egress_allowlist=("pypi.org",))._build_argv(
        "echo hi", container_name="c"
    )
    assert "--network=none" in argv
    assert egress.INTERNAL_NETWORK not in argv
