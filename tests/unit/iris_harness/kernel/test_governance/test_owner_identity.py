"""The owner's identity, extracted once and read the same way by every guard (ADR-0125).

PR 1 of owner-PII masking is a refactor with no behaviour change: the egress guard, the
response check and capability masking used to extract identity literals separately, with
two copies of the regex and two caches. These tests pin what each guard acted on before
the move, so the move cannot have changed it.
"""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance.identity_config import guard_table
from iris_harness.kernel.governance.identity_redaction import (
    owner_identity,
    register_identity_text_provider,
)
from iris_harness.kernel.governance.owner_identity import EMPTY, extract
from iris_harness.kernel.governance.owner_pii import MASK, redact_capability_text
from iris_harness.kernel.governance.plugins.network_egress import NetworkEgress
from iris_harness.kernel.governance.plugins.response_safety import identity_literals

SECRET = "CANARY_SOUL_SECRET_DIRECTIVE_d41f8a27"
LINK = "www.web3notes.example"
DOCS = [f"my key is {SECRET}; my blog is {LINK}; plain prose with no digits at all"]


@pytest.fixture(autouse=True)
def _isolate_registry(owner_identity_seam):
    NetworkEgress._reset_identity_cache()
    yield
    NetworkEgress._reset_identity_cache()


def test_extract_tags_each_secret_shaped_token_with_its_kind() -> None:
    corpus = extract(DOCS)
    assert corpus.of("secret") == frozenset({SECRET})
    assert corpus.of("link") == frozenset({LINK})
    assert corpus.of("secret", "link") == frozenset({SECRET, LINK})


def test_extract_skips_tokens_without_both_a_letter_and_a_digit() -> None:
    corpus = extract(["abcdefghijklmnopq 12345678901234567 short1a"])
    assert corpus.of("secret", "link") == frozenset()


def test_the_empty_corpus_has_every_kind() -> None:
    assert EMPTY.of("secret", "link") == frozenset()
    assert extract([]).of("secret", "link") == frozenset()


def test_egress_refuses_both_kinds_as_before() -> None:
    """The egress guard never excluded URL/email shapes: sending them out stays denied."""
    register_identity_text_provider(lambda: DOCS)
    assert NetworkEgress._identity_secret_literals() == frozenset({SECRET, LINK})


def test_the_response_check_halts_only_on_secrets_as_before() -> None:
    """Issue 0022: a blog with a digit must not halt the owner's own profile answer."""
    register_identity_text_provider(lambda: DOCS)
    assert identity_literals() == frozenset({SECRET})


def test_capability_masking_masks_what_the_response_check_halts_on() -> None:
    """The secret the response check halts on is masked in a result; the link passes."""
    register_identity_text_provider(lambda: DOCS)
    corpus, table = owner_identity(), guard_table()
    assert corpus is not None and table is not None
    out = redact_capability_text(DOCS[0], identity=corpus, table=table, grants=())
    assert identity_literals() == {SECRET}
    assert SECRET not in out.text and MASK in out.text and LINK in out.text


def test_every_guard_reads_one_corpus() -> None:
    register_identity_text_provider(lambda: DOCS)
    first = owner_identity()
    assert first is owner_identity()
    assert first is not None
    assert NetworkEgress._identity_secret_literals() == first.of("secret", "link")
    assert identity_literals() == first.of("secret")


def test_the_corpus_is_read_once_per_process() -> None:
    calls: list[int] = []

    def provider() -> list[str]:
        calls.append(1)
        return DOCS

    register_identity_text_provider(provider)
    owner_identity()
    identity_literals()
    NetworkEgress._identity_secret_literals()
    assert len(calls) == 1


def test_unregistered_is_not_cached_and_a_new_registration_takes_effect() -> None:
    assert owner_identity() is None
    assert identity_literals() == frozenset()

    register_identity_text_provider(lambda: DOCS)
    assert identity_literals() == frozenset({SECRET})

    register_identity_text_provider(lambda: [])
    assert identity_literals() == frozenset()
