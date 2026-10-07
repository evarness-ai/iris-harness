"""A wildcard egress declaration over a public suffix is refused (issue #175)."""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance.plugin_egress import normalize_host_pattern
from iris_harness.kernel.governance.public_suffix import (
    DATA,
    is_public_suffix,
    rule_count,
)


@pytest.mark.parametrize(
    "pattern",
    ["*.co.uk", "*.com.au", "*.org.uk", "*.co.jp", "*.com.br", "*.ac.uk", "*.gov.uk"],
)
def test_a_wildcard_over_a_public_suffix_is_refused(pattern: str) -> None:
    with pytest.raises(ValueError, match="public suffix"):
        normalize_host_pattern(pattern)


@pytest.mark.parametrize(
    "pattern",
    [
        "*.example.co.uk",
        "*.example.com.au",
        "*.example.org",
        "*.open-meteo.com",
        "*.github.io",  # the PRIVATE section is left out on purpose
        "*.s3.amazonaws.com",
        "*.compute.amazonaws.com",
    ],
)
def test_a_wildcard_under_a_registrable_domain_is_accepted(pattern: str) -> None:
    assert normalize_host_pattern(pattern) == pattern


def test_the_existing_single_label_refusal_still_applies() -> None:
    with pytest.raises(ValueError, match="registrable"):
        normalize_host_pattern("*.com")


def test_an_exact_host_is_not_affected() -> None:
    assert normalize_host_pattern("api.example.co.uk") == "api.example.co.uk"


def test_the_list_algorithm_handles_wildcard_and_exception_rules() -> None:
    # `*.ck` makes every second-level name under .ck a suffix; `!www.ck` is its exception.
    assert is_public_suffix("anything.ck")
    assert not is_public_suffix("www.ck")
    assert is_public_suffix("co.uk") and not is_public_suffix("example.co.uk")
    assert is_public_suffix("com") and is_public_suffix("notatld")  # the default rule
    assert not is_public_suffix("") and not is_public_suffix("example.com")
    with pytest.raises(ValueError, match="public suffix"):
        normalize_host_pattern("*.anything.ck")
    assert normalize_host_pattern("*.www.ck") == "*.www.ck"


def test_the_vendored_file_parses_is_ascii_and_keeps_its_notice() -> None:
    text = DATA.read_text(encoding="utf-8")
    assert text.isascii()
    header = [line for line in text.splitlines() if line.startswith("//")]
    joined = " ".join(line.removeprefix("//").strip() for line in header)
    assert "Mozilla Public License, v. 2.0" in joined and "mozilla.org/MPL/2.0" in joined
    assert "VERSION:" in joined and "COMMIT:" in joined and "ICANN section only" in joined
    rules = [line for line in text.splitlines() if line and not line.startswith("//")]
    assert len(rules) > 5000 and rule_count() == len(rules)
    # ICANN section only: no hosting-provider (PRIVATE section) entries.
    assert "github.io" not in rules and "blogspot.com" not in rules
    for rule in rules:
        body = rule.removeprefix("!")
        assert all(
            label == "*" or label.replace("-", "").isalnum() for label in body.split(".")
        ), rule
