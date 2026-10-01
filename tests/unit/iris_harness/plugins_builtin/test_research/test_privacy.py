"""Tests for the research privacy guard — strip the user's name before web egress."""

from __future__ import annotations

from iris_harness.plugins_builtin.research.privacy import (
    owner_name_variants,
    strip_owner_identifiers,
)


def test_strips_possessive_name_from_query() -> None:
    # The exact leak from the live session (web-3dedfff1).
    cleaned, stripped = strip_owner_identifiers(
        "latest market performance of Remy's portfolio", ["Remy"]
    )
    assert stripped is True
    assert "remy" not in cleaned.lower()
    assert cleaned == "latest market performance of portfolio"


def test_strips_lowercased_and_mid_query_name() -> None:
    cleaned, stripped = strip_owner_identifiers(
        "compare k remy's portfolio with Nifty 50", ["Remy"]
    )
    assert stripped is True
    assert "remy" not in cleaned.lower()
    assert "Nifty 50" in cleaned  # the non-personal part survives


def test_strips_full_name_and_tokens() -> None:
    cleaned, stripped = strip_owner_identifiers("Kumar portfolio returns this year", ["Remy Kumar"])
    assert stripped is True
    assert "kumar" not in cleaned.lower()


def test_leaves_clean_query_untouched() -> None:
    q = "Nifty 50 vs Nifty Next 50 comparison"
    cleaned, stripped = strip_owner_identifiers(q, ["Remy"])
    assert stripped is False
    assert cleaned == q


def test_no_names_is_noop() -> None:
    q = "anything at all"
    assert strip_owner_identifiers(q, []) == (q, False)


def test_short_tokens_not_stripped() -> None:
    # A 2-char initial must not become a strip token (would mangle unrelated queries);
    # the >= 3 char token does. Variants preserve the source case (matching is
    # case-insensitive at strip time).
    variants = owner_name_variants(["Ab Cdef"])
    assert "Ab" not in variants
    assert "Cdef" in variants
    # And the short initial is genuinely never stripped from a query.
    assert strip_owner_identifiers("ab testing framework", ["Ab Cdef"]) == (
        "ab testing framework",
        False,
    )


def test_variants_longest_first() -> None:
    variants = owner_name_variants(["Remy Kumar"])
    assert variants[0] == "Remy Kumar"  # full name removed before tokens


# --- identifiers derived from identity files (USER.md / SOUL.md), not hardcoded ---

_USER_MD = """# User Profile
## Identity
- **Name:** Remy
- **Location:** Springfield
## Auto-detected
- **email**: remy.k@example.com  <!-- auto: confidence=0.9 -->
- **name**: Remy  <!-- auto: confidence=0.95 -->
"""
_SOUL_MD = """---
name: IRIS
---
You serve the user. Contact ops at iris-ops@example.com.
"""


def test_extract_identifiers_pulls_name_and_email_from_user_md() -> None:
    from iris_harness.plugins_builtin.research.privacy import extract_personal_identifiers

    ids = extract_personal_identifiers(_USER_MD, _SOUL_MD)
    assert "Remy" in ids
    assert "remy.k@example.com" in ids


def test_extract_identifiers_never_strips_agent_name_from_soul() -> None:
    from iris_harness.plugins_builtin.research.privacy import extract_personal_identifiers

    ids = extract_personal_identifiers(_USER_MD, _SOUL_MD)
    # SOUL.md's `name: IRIS` is the AGENT — must not become a strip target.
    assert "IRIS" not in ids and "iris" not in [i.lower() for i in ids]
    # but an email anywhere (even SOUL.md) is collected as PII
    assert "iris-ops@example.com" in ids


def test_extract_identifiers_empty_profile() -> None:
    from iris_harness.plugins_builtin.research.privacy import extract_personal_identifiers

    assert extract_personal_identifiers(None, None) == []


def test_end_to_end_strip_uses_extracted_identifiers() -> None:
    from iris_harness.plugins_builtin.research.privacy import (
        extract_personal_identifiers,
        strip_owner_identifiers,
    )

    ids = extract_personal_identifiers(_USER_MD, _SOUL_MD)
    cleaned, stripped = strip_owner_identifiers(
        "compare Remy's portfolio and email remy.k@example.com to Nifty", ids
    )
    assert stripped is True
    assert "remy" not in cleaned.lower()
    assert "@example.com" not in cleaned
    assert "Nifty" in cleaned


def test_strip_emails_removes_any_address_even_unknown() -> None:
    # ADR-0102: a blanket guard — an email the identity files never listed (e.g. a
    # secondary account learned from inbox context) must still never egress to the web.
    from iris_harness.plugins_builtin.research.privacy import strip_emails

    cleaned, stripped = strip_emails("secondary@gmail.com insurance dues")
    assert stripped is True
    assert "@" not in cleaned
    assert "insurance dues" in cleaned


def test_strip_emails_noop_without_address() -> None:
    from iris_harness.plugins_builtin.research.privacy import strip_emails

    cleaned, stripped = strip_emails("latest Nifty 50 performance")
    assert stripped is False
    assert cleaned == "latest Nifty 50 performance"
