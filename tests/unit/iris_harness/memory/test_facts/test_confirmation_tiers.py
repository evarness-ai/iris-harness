"""How a fact gets confirmed: said plainly, asked once, or queued.

The first cut sent everything to a review queue. That fixed precision and broke the
point — IRIS knew nothing about the owner until they did homework. Three tiers, by how
plainly the thing was stated.
"""

from __future__ import annotations

import pytest

from iris_harness.memory import fact_keys
from iris_harness.memory.fact_keys import ASK, AUTO, QUEUE, classify_capture

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")


@pytest.fixture(autouse=True)
def _fresh_config() -> None:
    fact_keys.reset_cache()


class TestTierA:
    @pytest.mark.parametrize(
        ("message", "key", "value"),
        [
            ("my name is Robin", "name", "Robin"),
            ("I live in Springfield", "city", "Springfield"),
            ("i work at Quant Academy", "employer", "Quant Academy"),
            ("my blog is tech4talk.com", "blog", "tech4talk.com"),
            ("remember that my bank is Northwind", "bank", "Northwind"),
        ],
    )
    def test_a_plain_self_statement_is_remembered_outright(
        self, message: str, key: str, value: str
    ) -> None:
        assert classify_capture(message, key, value, 0.9) == AUTO

    def test_the_value_has_to_be_in_the_message(self) -> None:
        """A normalised or invented value is not what the user said."""
        assert classify_capture("my name is Robin", "name", "Robin Kumar", 0.9) != AUTO

    def test_the_store_s_worst_row_cannot_come_back(self) -> None:
        """`name=ollama` came from a message about configuring a model."""
        verdict = classify_capture("i am running ollama for local models", "name", "ollama", 1.0)

        assert verdict != AUTO  # it can be asked about, never assumed

    def test_a_key_outside_the_allowlist_is_never_automatic(self) -> None:
        assert classify_capture("my topic is murder plot", "topic", "murder plot", 0.9) == QUEUE


class TestTierB:
    def test_an_inferred_fact_is_worth_one_question(self) -> None:
        assert classify_capture("i pay by UPI from Northwind", "bank", "Northwind", 0.8) == ASK

    def test_a_low_confidence_guess_is_not(self) -> None:
        assert classify_capture("i pay by UPI from Northwind", "bank", "Northwind", 0.3) == QUEUE


class TestTierC:
    def test_third_party_content_never_leaves_the_queue(self) -> None:
        """`employer=Department of Justice` came from a pasted article."""
        verdict = classify_capture(
            "the Department of Justice charged them", "employer", "Department of Justice", 0.9
        )

        assert verdict == QUEUE

    def test_quoted_text_is_not_the_user_speaking(self) -> None:
        assert classify_capture("> my name is Someone Else", "name", "Someone Else", 0.9) == QUEUE
