"""The compiled egress policy: what ``decide`` allows, and fails closed otherwise (#103)."""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance.plugin_egress import (
    EgressVerdict,
    HostRule,
    PluginEgress,
    PluginEgressPolicy,
    egress_policy,
    normalize_host_pattern,
    register_egress_policy,
)


def _policy() -> PluginEgressPolicy:
    return PluginEgressPolicy(
        {
            "weather": PluginEgress(
                hosts=(
                    HostRule("api.open-meteo.com"),
                    HostRule("geo.open-meteo.com", data="personal"),
                    HostRule("*.cdn.example.org", ports=(443, 8443)),
                    HostRule("localhost", schemes=("http",), ports=(8080,)),
                )
            ),
            "fetcher": PluginEgress(open_web=True),
            "quiet": PluginEgress(),
        }
    )


@pytest.mark.parametrize(
    "plugin, scheme, host, port, allowed",
    [
        ("weather", "https", "api.open-meteo.com", 443, True),
        ("weather", "https", "evil.example", 443, False),  # undeclared host
        ("weather", "https", "open-meteo.com", 443, False),  # a prefix/suffix is not the host
        ("weather", "https", "api.open-meteo.com.evil.example", 443, False),
        ("weather", "http", "api.open-meteo.com", 80, False),  # scheme not declared
        ("weather", "https", "api.open-meteo.com", 8443, False),  # port not declared
        ("weather", "https", "a.cdn.example.org", 8443, True),
        ("weather", "https", "a.b.cdn.example.org", 443, True),
        ("weather", "https", "cdn.example.org", 443, False),  # the apex is not a subdomain
        ("weather", "https", "xcdn.example.org", 443, False),
        ("weather", "http", "localhost", 8080, True),
        ("fetcher", "https", "anything.example", 443, True),
        ("fetcher", "ftp", "anything.example", 21, False),
        ("quiet", "https", "api.open-meteo.com", 443, False),  # declares nothing: closed
        ("ghost", "https", "api.open-meteo.com", 443, False),  # not mounted
    ],
)
def test_decide(plugin: str, scheme: str, host: str, port: int, allowed: bool) -> None:
    assert _policy().decide(plugin, scheme=scheme, host=host, port=port).allowed is allowed


def test_a_run_holding_more_than_the_host_is_declared_to_receive_is_refused() -> None:
    policy = _policy()

    def send(host: str, classification: str) -> EgressVerdict:
        return policy.decide(
            "weather", scheme="https", host=host, port=443, classification=classification
        )

    assert send("api.open-meteo.com", "public").allowed
    assert send("api.open-meteo.com", "internal").allowed
    refused = send("api.open-meteo.com", "personal")
    assert not refused.allowed and "personal" in refused.reason
    assert send("geo.open-meteo.com", "personal").allowed
    # ``secret`` never leaves to a plugin's host, and an unknown label is the strictest.
    assert not send("geo.open-meteo.com", "secret").allowed
    assert not send("geo.open-meteo.com", "mystery").allowed


def test_the_registered_policy_is_none_until_one_is_registered() -> None:
    register_egress_policy(None)
    assert egress_policy() is None
    policy = _policy()
    register_egress_policy(policy)
    try:
        assert egress_policy() is policy
    finally:
        register_egress_policy(None)


@pytest.mark.parametrize("bad", ["*", "*com", "a.com/x", "a.com:1", "", "-x.com", "*.com"])
def test_a_host_pattern_must_be_a_host(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_host_pattern(bad)


def test_hosts_are_compared_lower_case_without_a_trailing_dot() -> None:
    assert normalize_host_pattern("API.Open-Meteo.COM.") == "api.open-meteo.com"


@pytest.mark.parametrize(
    "bad",
    [
        "good\n.com",
        "good.com\n",
        "good .com",
        "good\x00.com",
        "good.com..",
        "..good.com",
        "good..com",
    ],
)
def test_newlines_whitespace_and_repeated_dots_are_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_host_pattern(bad)


def test_one_trailing_dot_is_dropped() -> None:
    assert normalize_host_pattern("good.com.") == "good.com"


@pytest.mark.parametrize(
    "bad",
    [
        "999.999.999.999",
        "1.2.3.4",
        "\u0661.\u0661.\u0661.\u0661",
        "127.1",
        "0x7f.1",
        "0x7f000001",
        "0177.0.0.1",
        "2130706433",
        "localhost",
        "a.localhost",
        "*.1.2.3.4",
        "[::1]",
        "::1",
    ],
)
def test_ip_literals_and_localhost_are_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_host_pattern(bad)


@pytest.mark.parametrize("ok", ["api.example.com", "1password.com", "123.example.com", "a-b.io"])
def test_real_names_with_digits_still_pass(ok: str) -> None:
    assert normalize_host_pattern(ok) == ok


def test_a_wildcard_needs_a_multi_label_base() -> None:
    assert normalize_host_pattern("*.example.org") == "*.example.org"
    for bad in ("*.com", "*."):
        with pytest.raises(ValueError):
            normalize_host_pattern(bad)


def test_length_limits_at_the_boundary() -> None:
    label63 = "a" * 63
    assert normalize_host_pattern(f"{label63}.com") == f"{label63}.com"
    with pytest.raises(ValueError):
        normalize_host_pattern(f"{'a' * 64}.com")
    # 253 total: three 63-char labels + "." joins + a 61-char tail = 63*3 + 3 + 61 = 253.
    ok = ".".join([label63] * 3 + ["b" * 61])
    assert len(ok) == 253 and normalize_host_pattern(ok) == ok
    with pytest.raises(ValueError):
        normalize_host_pattern(ok + "c")


def test_punycode_labels_must_decode() -> None:
    assert normalize_host_pattern("xn--bcher-kva.example") == "xn--bcher-kva.example"
    with pytest.raises(ValueError):
        normalize_host_pattern("xn--zzzzzzzz.example")
    with pytest.raises(ValueError):
        normalize_host_pattern("xn--9.example")
    with pytest.raises(ValueError):
        normalize_host_pattern("b\u00fccher.example")
