"""Per-kind matchers for the owner's identity (ADR-0125, PR 3).

Each kind is found however it is formatted, and never as part of a longer word, address
or number. Positives and negatives for every kind; the literals are synthetic.
"""

from __future__ import annotations

import re

import pytest

from iris_harness.kernel.governance.owner_identity import declared, extract, merge
from iris_harness.kernel.governance.owner_matchers import (
    Match,
    OwnerMatcher,
    canonical,
    is_first_name_alone,
    name_pattern,
    non_overlapping,
)

SECRET = "CANARY_SOUL_SECRET_DIRECTIVE_d41f8a27"
LINK = "www.web3notes.example"
EMAIL = "Owner.Canary@example.com"
PHONE = "+1 555 0100 0199"  # country code 1, national 555 0100 0199
NAME = "Robin Example"
FIRST = "Robin"
HANDLE = "@robin-gh"
HANDLE_URL = "https://social.example/in/robin-li"
ADDRESS = "1 Example Street, Springfield"

IDENTITY = merge(
    extract([f"key {SECRET} blog {LINK}"]),
    declared(
        {
            "email": [EMAIL],
            "phone": [PHONE],
            "name": [NAME, FIRST],
            "handle": [HANDLE, HANDLE_URL],
            "address": [ADDRESS],
        }
    ),
)
MATCHER = OwnerMatcher(IDENTITY)


def found(text: str) -> list[tuple[str, str]]:
    return [(text[m.start : m.end], m.kind) for m in MATCHER.find(text)]


# -- secret and link: exact, as before ---------------------------------------------------


def test_secret_and_link_match_exactly() -> None:
    assert found(f"the key is {SECRET}.") == [(SECRET, "secret")]
    assert found(f"see {LINK}/post") == [(LINK, "link")]
    assert found(SECRET.lower()) == []  # exact, case and all


# -- email -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        ("mail owner.canary@example.com.", "owner.canary@example.com"),
        ("<OWNER.CANARY@EXAMPLE.COM>", "OWNER.CANARY@EXAMPLE.COM"),
        ("mailto:owner.canary@example.com", "owner.canary@example.com"),
    ],
)
def test_email_matches_case_insensitively(text: str, hit: str) -> None:
    assert found(text) == [(hit, "email")]


@pytest.mark.parametrize(
    "text",
    [
        "xowner.canary@example.com",  # a longer local part
        "owner.canary@example.com.au",  # a longer domain
        "owner.canary@example.community",
        "a.owner.canary@example.com",
    ],
)
def test_email_never_matches_inside_another_address(text: str) -> None:
    assert [k for _, k in found(text)] == []


# -- phone -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        ("call +1 555 0100 0199 now", "+1 555 0100 0199"),
        ("call +1 (555) 0100-0199", "+1 (555) 0100-0199"),
        ("call +1-555-0100-0199", "+1-555-0100-0199"),
        ("call +1.555.0100.0199", "+1.555.0100.0199"),
        ("call +155501000199.", "+155501000199"),
        ("call 00 1 555 0100 0199", "00 1 555 0100 0199"),
        ("call 001 555 0100 0199", "001 555 0100 0199"),
        ("call 555 0100 0199", "555 0100 0199"),  # national, no country code
        ("call (555) 0100 0199", "(555) 0100 0199"),
        ("call 555.0100.0199", "555.0100.0199"),
        ("call 55501000199", "55501000199"),
        ("tel:+15550100 0199", "+15550100 0199"),
        ("call 555 0100 0199 2026 times", "555 0100 0199"),  # a second number after a space
    ],
)
def test_phone_matches_its_digits_whatever_the_format(text: str, hit: str) -> None:
    assert found(text) == [(hit, "phone")]


@pytest.mark.parametrize(
    "text",
    [
        "call 555 0100 0198",  # one digit off
        "ref 1555010001999",  # a longer number
        "ref 2555501000199",  # the digits inside a longer run
        "id 555-0100-0199-7",  # part of one hyphenated number
        "v3.555.0100.0199",  # a dotted version
        "+44 555 0100 0199",  # another country code
        "/files/55501000199",  # a path
        "amount 555,010,001",
    ],
)
def test_phone_never_matches_part_of_another_number(text: str) -> None:
    assert found(text) == []


def test_national_trunk_zero_and_the_bracketed_zero() -> None:
    uk = OwnerMatcher(declared({"phone": ["+44 20 7946 0000"]}))

    def hits(text: str) -> list[str]:
        return [text[m.start : m.end] for m in uk.find(text)]

    assert hits("ring 020 7946 0000") == ["020 7946 0000"]
    assert hits("ring +44 (0)20 7946 0000") == ["+44 (0)20 7946 0000"]
    assert hits("ring 0044 20 7946 0000") == ["0044 20 7946 0000"]
    assert hits("ring 020 7946 0001") == []


def test_a_short_phone_literal_matches_nothing() -> None:
    assert OwnerMatcher(declared({"phone": ["12 34"]})).find("12 34 and 1234") == []


def test_two_spellings_of_one_number_share_a_canonical_key() -> None:
    assert canonical("phone", "+1 (555) 0100-0199") == canonical("phone", "+1 555 0100 0199")
    assert canonical("phone", "001 555 0100 0199") == canonical("phone", "+1 555 0100 0199")


# -- name --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "hits"),
    [
        ("from robin example today", [("robin example", "name")]),
        ("from Robin\n  Example today", [("Robin\n  Example", "name")]),
        ("Robin's car", [("Robin", "name")]),
        ("Robin.", [("Robin", "name")]),
        ("(ROBIN)", [("ROBIN", "name")]),
    ],
)
def test_name_matches_whole_words_case_insensitively(
    text: str, hits: list[tuple[str, str]]
) -> None:
    assert found(text) == hits


@pytest.mark.parametrize(
    "text",
    [
        "Robinson Crusoe",  # a longer word
        "robin-hood",  # a hyphenated word
        "robin@mail.example",  # inside an address
        "mail.robin.example",  # a dotted token
        "@robin",  # a handle
        "Robin_2",
    ],
)
def test_name_never_matches_part_of_a_word(text: str) -> None:
    assert [k for _, k in found(text) if k == "name"] == []


def test_a_full_name_is_not_its_first_name() -> None:
    only_full = OwnerMatcher(declared({"name": [NAME]}))
    assert only_full.find("Robin wrote") == []
    assert [m.literal for m in only_full.find("Robin Example wrote")] == [NAME]


def test_first_name_alone_is_a_one_word_name_literal() -> None:
    full, first = MATCHER.find("Robin Example and Robin")
    assert (full.literal, is_first_name_alone(full)) == (NAME, False)
    assert (first.literal, is_first_name_alone(first)) == (FIRST, True)
    assert not is_first_name_alone(Match(0, 5, "handle", "robin"))


# -- handle ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        ("ping @robin-gh!", "@robin-gh"),
        ("ping robin-gh", "robin-gh"),
        ("ping @ROBIN-GH", "@ROBIN-GH"),
        ("see code.example/robin-gh/repo", "robin-gh"),
        ("see social.example/in/Robin-Li", "Robin-Li"),  # declared as a URL
    ],
)
def test_handle_matches_with_or_without_at_and_in_urls(text: str, hit: str) -> None:
    assert found(text) == [(hit, "handle")]


@pytest.mark.parametrize(
    "text", ["robin-ghost", "xrobin-gh", "robin-gh.example", "me@robin-gh.example"]
)
def test_handle_never_matches_part_of_a_word(text: str) -> None:
    assert [k for _, k in found(text) if k == "handle"] == []


# -- address -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "at 1 Example Street, Springfield.",
        "at 1 example street\nspringfield",
        "at 1 EXAMPLE STREET -- SPRINGFIELD",
    ],
)
def test_address_matches_whatever_the_punctuation(text: str) -> None:
    assert [k for _, k in found(text)] == ["address"]


@pytest.mark.parametrize(
    "text",
    ["1 Example Streets, Springfield", "11 Example Street, Springfield", "1 Example Street"],
)
def test_address_never_matches_a_different_or_partial_address(text: str) -> None:
    assert [k for _, k in found(text)] == []


# -- spans -------------------------------------------------------------------------------


def test_find_keeps_the_earliest_longest_occurrence() -> None:
    text = "Robin Example, Robin, owner.canary@example.com"
    assert [m.kind for m in MATCHER.find(text)] == ["name", "name", "email"]
    assert non_overlapping([Match(0, 5, "name", "x"), Match(2, 8, "email", "y")]) == [
        Match(0, 5, "name", "x")
    ]


def test_find_all_keeps_overlaps_for_the_caller_to_rank() -> None:
    both = OwnerMatcher(declared({"name": ["Robin", "Robin Example"]}))
    assert len(both.find_all("Robin Example")) == 2
    assert len(both.find("Robin Example")) == 1


def test_an_empty_corpus_or_text_finds_nothing() -> None:
    assert OwnerMatcher(declared({})).find("Robin Example") == []
    assert MATCHER.find("") == []


def test_name_pattern_is_the_research_scrubs_rule_with_single_spaces() -> None:
    """``any_space=False`` is exactly ``\\b<escaped name>\\b`` (research privacy's rule)."""
    for variant in ("Remy Kumar", "Remy", "O'Neil Smith"):
        for query in ("remy kumar's portfolio", "Remy  Kumar", "ravikumar", "o'neil smith"):
            old = re.findall(rf"\b{re.escape(variant)}\b", query, re.IGNORECASE)
            new = re.findall(name_pattern(variant, any_space=False), query, re.IGNORECASE)
            assert new == old, (variant, query)
