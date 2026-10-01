"""Owner-PII masking PR 2: the new kinds, the named sources, and invalidation (ADR-0125).

The corpus grows (``name``, ``email``, ``phone``, ``address``, ``handle``) and so does
where it comes from; no guard asks for the new kinds yet. What these pin:

- the pure helpers: declared literals, merge, ``never_match`` (``secret`` untouched), and
  the free-text ``email``/``phone`` shapes, conservative about dates and amounts;
- the seam: sources merge, a failing one costs only itself, a source bound to kinds has
  the rest dropped and reported, and nothing may declare ``secret`` (or ``link``, yet);
- invalidation: an in-process write is seen at once; another process's, through a
  fingerprint, within the interval and not before; an unchanged fingerprint re-reads
  nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest

from iris_harness.kernel.governance.identity_config import DeclarableKind
from iris_harness.kernel.governance.owner_identity import (
    DECLARABLE_KINDS,
    KINDS,
    NEVER_MATCH,
    NEVER_MATCH_SPARES,
    OWNER_PII_KINDS,
    OwnerIdentity,
    apply_never_match,
    declared,
    extract,
    merge,
)

SECRET = "CANARY_SOUL_SECRET_DIRECTIVE_d41f8a27"
LINK = "www.web3notes.example"


# -- kinds -----------------------------------------------------------------------------


def test_the_kinds() -> None:
    assert set(KINDS) == {"secret", "link", "name", "email", "phone", "address", "handle"}
    # Nothing declares a secret; a link joined the declarable kinds with PR 3 (the
    # owner's confirmed blog and website), but only the PII kinds are a plugin's to provide.
    assert set(OWNER_PII_KINDS) == {"name", "email", "phone", "address", "handle"}
    assert set(DECLARABLE_KINDS) == {*OWNER_PII_KINDS, "link"}


def test_identity_yaml_kinds_are_the_declarable_kinds() -> None:
    assert set(DeclarableKind.__args__) == set(DECLARABLE_KINDS)  # type: ignore[attr-defined]


# -- pure helpers ----------------------------------------------------------------------


def test_declared_takes_only_declarable_kinds_and_strips() -> None:
    identity = declared(
        {
            "name": ["  Robin Example ", ""],
            "email": ["robin@mail.example"],
            "secret": [SECRET],
            "link": [LINK],
            NEVER_MATCH: ["Robin"],
            "shoe_size": ["44"],
        }
    )
    assert identity.of("name") == {"Robin Example"}
    assert identity.of("email") == {"robin@mail.example"}
    assert identity.of("secret") == frozenset()
    assert identity.of("link") == {LINK}
    assert set(identity.literals) == set(KINDS)


def test_a_bare_string_is_one_literal_not_its_characters() -> None:
    assert declared({"email": "robin@mail.example"}).of("email") == {"robin@mail.example"}


def test_merge_unions_kind_by_kind() -> None:
    a = declared({"name": ["Robin"], "email": ["a@mail.example"]})
    b = declared({"name": ["Robin Example"], "phone": ["+1 555 123 4567"]})
    merged = merge(a, b, extract([f"key {SECRET}"]))
    assert merged.of("name") == {"Robin", "Robin Example"}
    assert merged.of("email") == {"a@mail.example"}
    assert merged.of("phone") == {"+1 555 123 4567"}
    assert merged.of("secret") == {SECRET}


def test_never_match_drops_case_insensitively_but_spares_what_guards_act_on() -> None:
    identity = merge(
        declared({"name": ["Robin", "Robin Example"], "handle": ["robin-gh"]}),
        extract([f"key {SECRET}, blog {LINK}"]),
    )
    out = apply_never_match(identity, ["ROBIN", "robin-GH", SECRET.lower(), LINK.upper()])
    assert out.of("name") == {"Robin Example"}
    assert out.of("handle") == frozenset()
    assert out.of("secret") == {SECRET}, "never_match must not switch a secret off"
    assert out.of("link") == {LINK}, "never_match must not take a link out of egress"
    assert set(NEVER_MATCH_SPARES) == {"secret", "link"}


def test_never_match_with_nothing_to_drop_is_the_same_identity() -> None:
    identity = declared({"name": ["Robin"]})
    assert apply_never_match(identity, ["", "  "]) is identity


# -- free text: email and phone shapes only --------------------------------------------


def test_free_text_yields_email_and_phone_but_never_a_name() -> None:
    corpus = extract(
        ["I am Robin Example. Write to robin.example+x@mail.example or call +44 20 7946 0958."]
    )
    assert corpus.of("email") == {"robin.example+x@mail.example"}
    assert corpus.of("phone") == {"+44 20 7946 0958"}
    assert corpus.of("name", "address", "handle") == frozenset()


@pytest.mark.parametrize(
    "phone",
    [
        "+1 555 123 4567",
        "+91 98765 43210",
        "+919876543210",
        "+1 555-123-4567",
        "(555) 123-4567",
        "555-123-4567",
        "555.123.4567",
        "020 7946 0958",
    ],
)
def test_phone_shapes(phone: str) -> None:
    assert extract([f"call {phone}. thanks"]).of("phone") == {phone}


@pytest.mark.parametrize(
    "text",
    [
        "2026-09-30",
        "30/09/2026",
        "30.09.2026",
        "2026-09-30T10:20:30Z",
        "2026-09-30T10:20:30+05:30",
        "2026-09-30 10:20:30",
        "2026-09-30 - 2026-10-01",
        "$1,234,567.89",
        "1,234.56 USD",
        "Rs. 12,50,000",
        "1 234 567 890",
        "192.168.100.200",
        "1727654400",  # a unix timestamp: a bare digit run is not a phone
        "9876543210",
        "12345678",
        "v3.12.1",
        "555-123-4567 2026",  # a number and then a year, not one phone
    ],
)
def test_not_phones(text: str) -> None:
    assert extract([f"x {text} y"]).of("phone") == frozenset()


@pytest.mark.parametrize("text", ["user@localhost", "@robin", "robin@", "a@b"])
def test_not_emails(text: str) -> None:
    assert extract([f"x {text} y"]).of("email") == frozenset()


# -- the seam: named sources -----------------------------------------------------------


def _source(literals: Mapping[str, Iterable[str]], calls: list[str] | None = None) -> Any:
    def provider() -> Mapping[str, Iterable[str]]:
        if calls is not None:
            calls.append("read")
        return literals

    return provider


def test_sources_merge_and_never_match_applies_across_them(owner_identity_seam: Any) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(lambda: [f"key {SECRET}; mail robin@mail.example"])
    seam.register_owner_identity_source(
        "profile", _source({"name": ["Robin", "Robin Example"], NEVER_MATCH: ["robin"]})
    )
    seam.register_owner_identity_source("other", _source({"handle": ["robin-gh"]}))
    corpus = seam.owner_identity()
    assert corpus is not None
    assert corpus.of("name") == {"Robin Example"}
    assert corpus.of("handle") == {"robin-gh"}
    assert corpus.of("email") == {"robin@mail.example"}
    assert corpus.of("secret") == {SECRET}


def test_a_bare_never_match_string_is_one_entry(owner_identity_seam: Any) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(list)
    seam.register_owner_identity_source(
        "profile", _source({"name": ["Robin", "R"], NEVER_MATCH: "Robin"})  # type: ignore[dict-item]
    )
    assert seam.owner_identity().of("name") == {"R"}


def test_without_documents_the_seam_is_unregistered(owner_identity_seam: Any) -> None:
    """A named source alone is not a registered seam: PR 1's fail-loud rule stands."""
    seam = owner_identity_seam
    seam.register_owner_identity_source("profile", _source({"name": ["Robin"]}))
    assert seam.owner_identity() is None


def test_a_failing_source_costs_only_itself(owner_identity_seam: Any) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(lambda: [f"key {SECRET}"])

    def boom() -> Mapping[str, Iterable[str]]:
        raise RuntimeError("unreadable")

    seam.register_owner_identity_source("broken", boom)
    seam.register_owner_identity_source("profile", _source({"name": ["Robin"]}))
    corpus = seam.owner_identity()
    assert corpus is not None
    assert corpus.of("secret") == {SECRET}
    assert corpus.of("name") == {"Robin"}


def test_a_bound_source_has_undeclared_kinds_dropped_and_reported(
    owner_identity_seam: Any,
) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(list)
    reported: list[frozenset[str]] = []
    seam.register_owner_identity_source(
        "plugin:mail",
        _source({"email": ["robin@mail.example"], "name": ["Robin"], NEVER_MATCH: ["x"]}),
        kinds=("email",),
        on_undeclared=reported.append,
    )
    corpus = seam.owner_identity()
    assert corpus is not None
    assert corpus.of("email") == {"robin@mail.example"}
    assert corpus.of("name") == frozenset()
    assert reported == [frozenset({"name", NEVER_MATCH})]


@pytest.mark.parametrize("kind", ["secret", "link", "shoe_size"])
def test_no_source_may_be_bound_to_an_undeclarable_kind(
    owner_identity_seam: Any, kind: str
) -> None:
    with pytest.raises(ValueError, match="cannot declare"):
        owner_identity_seam.register_owner_identity_source("p", list, kinds=(kind,))


def test_an_unbound_source_cannot_supply_a_secret(owner_identity_seam: Any) -> None:
    """The composition root's own sources may declare a link (a confirmed blog), never a
    secret."""
    seam = owner_identity_seam
    seam.register_identity_text_provider(list)
    seam.register_owner_identity_source("profile", _source({"secret": [SECRET], "link": [LINK]}))
    corpus = seam.owner_identity()
    assert corpus is not None and corpus.of("secret") == frozenset()
    assert corpus.of("link") == {LINK}


def test_documents_is_a_reserved_source_name(owner_identity_seam: Any) -> None:
    with pytest.raises(ValueError, match="documents"):
        owner_identity_seam.register_owner_identity_source("documents", list)


def test_unregistering_a_source_takes_its_literals_out(owner_identity_seam: Any) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(list)
    seam.register_owner_identity_source("profile", _source({"name": ["Robin"]}))
    assert seam.owner_identity().of("name") == {"Robin"}
    seam.unregister_owner_identity_source("profile")
    assert seam.owner_identity().of("name") == frozenset()


# -- invalidation ------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(owner_identity_seam: Any) -> _Clock:
    c = _Clock()
    owner_identity_seam.set_owner_identity_clock(c)
    return c


def test_a_moved_fingerprint_is_seen_after_the_interval_not_before(
    owner_identity_seam: Any, clock: _Clock
) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(list)
    state = {"names": ["Robin"], "version": 1}
    calls: list[str] = []

    def provider() -> Mapping[str, Iterable[str]]:
        calls.append("read")
        return {"name": list(state["names"])}  # type: ignore[call-overload]

    seam.register_owner_identity_source("profile", provider, fingerprint=lambda: state["version"])
    assert seam.owner_identity().of("name") == {"Robin"}

    # Another process writes: the file changes, nobody here invalidates.
    state.update(names=["Robin", "Robin Example"], version=2)
    clock.now += seam.FINGERPRINT_INTERVAL_S - 0.1
    assert seam.owner_identity().of("name") == {"Robin"}, "checked before the interval"
    clock.now += 0.2
    assert seam.owner_identity().of("name") == {"Robin", "Robin Example"}
    assert len(calls) == 2


def test_an_unchanged_fingerprint_reads_nothing_again(
    owner_identity_seam: Any, clock: _Clock
) -> None:
    seam = owner_identity_seam
    documents: list[str] = []
    probes: list[str] = []

    def texts() -> list[str]:
        documents.append("read")
        return [f"key {SECRET}"]

    def fingerprint() -> int:
        probes.append("probe")
        return 7

    seam.register_identity_text_provider(texts, fingerprint=fingerprint)
    calls: list[str] = []
    seam.register_owner_identity_source(
        "profile", _source({"name": ["Robin"]}, calls), fingerprint=lambda: "same"
    )
    first = seam.owner_identity()
    for _ in range(3):
        clock.now += seam.FINGERPRINT_INTERVAL_S + 1
        assert seam.owner_identity() is first
    assert documents == ["read"]
    assert calls == ["read"]
    assert len(probes) == 4  # probed each interval, read once


def test_only_the_source_that_moved_is_read_again(owner_identity_seam: Any, clock: _Clock) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(list)
    moving, still = [], []
    version = {"v": 1}
    seam.register_owner_identity_source(
        "moving", _source({"name": ["Robin"]}, moving), fingerprint=lambda: version["v"]
    )
    seam.register_owner_identity_source(
        "still", _source({"handle": ["robin-gh"]}, still), fingerprint=lambda: 1
    )
    seam.owner_identity()
    version["v"] = 2
    clock.now += seam.FINGERPRINT_INTERVAL_S + 1
    corpus = seam.owner_identity()
    assert corpus.of("handle") == {"robin-gh"}
    assert (len(moving), len(still)) == (2, 1)


def test_an_in_process_invalidation_is_seen_at_once(
    owner_identity_seam: Any, clock: _Clock
) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(list)
    names = ["Robin"]
    seam.register_owner_identity_source(
        "profile", lambda: {"name": list(names)}, fingerprint=lambda: "unchanged"
    )
    assert seam.owner_identity().of("name") == {"Robin"}
    names.append("Robin Example")
    seam.invalidate_owner_identity()
    assert seam.owner_identity().of("name") == {"Robin", "Robin Example"}  # clock never moved


def test_the_writer_decorator_invalidates_even_when_the_write_fails(
    owner_identity_seam: Any,
) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(list)
    names = ["Robin"]
    seam.register_owner_identity_source("profile", lambda: {"name": list(names)})

    @seam.invalidates_owner_identity
    def write(fail: bool) -> None:
        names.append("Robin Example")
        if fail:
            raise OSError("disk full")

    assert seam.owner_identity().of("name") == {"Robin"}
    with pytest.raises(OSError):
        write(True)
    assert seam.owner_identity().of("name") == {"Robin", "Robin Example"}


def test_a_source_without_a_fingerprint_is_read_once_until_invalidated(
    owner_identity_seam: Any, clock: _Clock
) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(list)
    calls: list[str] = []
    seam.register_owner_identity_source("profile", _source({"name": ["Robin"]}, calls))
    seam.owner_identity()
    clock.now += 10 * seam.FINGERPRINT_INTERVAL_S
    seam.owner_identity()
    assert calls == ["read"]
    seam.invalidate_owner_identity()
    seam.owner_identity()
    assert calls == ["read", "read"]


def test_a_failing_fingerprint_reads_the_source_again(
    owner_identity_seam: Any, clock: _Clock
) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(list)
    calls: list[str] = []

    def fingerprint() -> int:
        raise OSError("gone")

    seam.register_owner_identity_source(
        "profile", _source({"name": ["Robin"]}, calls), fingerprint=fingerprint
    )
    seam.owner_identity()
    clock.now += seam.FINGERPRINT_INTERVAL_S + 1
    seam.owner_identity()
    assert calls == ["read", "read"]


def test_an_identity_built_while_a_source_changed_is_not_cached(
    owner_identity_seam: Any,
) -> None:
    """A write that lands during a rebuild must not be hidden behind that rebuild."""
    seam = owner_identity_seam
    seam.register_identity_text_provider(list)
    names = ["Robin"]

    def provider() -> Mapping[str, Iterable[str]]:
        snapshot = list(names)
        if len(names) == 1:  # the write lands while this read is in flight
            names.append("Robin Example")
            seam.invalidate_owner_identity()
        return {"name": snapshot}

    seam.register_owner_identity_source("profile", provider)
    assert seam.owner_identity().of("name") == {"Robin"}
    assert seam.owner_identity().of("name") == {"Robin", "Robin Example"}


def test_empty_identity_has_every_kind() -> None:
    assert set(OwnerIdentity().literals) == set(KINDS)
