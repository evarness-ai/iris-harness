"""Display-only masking of personal identifiers (email addresses) on the way to a screen.

The regression these pin: a turn asked "wondering how my day looks like ?" summarised
the user's inbox and printed both account addresses verbatim — to the terminal, and to
every other channel and log carrying the same string. Nothing inspected the response
text on its way out; `PRE_RESPONSE` was a declared hook point with no plugin on it.
"""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance.display_mask import StreamMasker, is_enabled, mask_text


def test_an_address_is_masked_but_still_readable() -> None:
    assert mask_text("mail jordan1.kp@example.com now") == "mail jo***kp@example.com now"


def test_two_accounts_of_the_same_user_stay_distinguishable() -> None:
    """The point of the mask is to stop the address being usable, not to stop the
    reader telling which of their own accounts a line is about. Both of these share
    their first three characters, so a prefix-only mask would render them identically."""
    masked = mask_text("jordan1.kp@example.com and jordankpatel@example.com")
    assert masked == "jo***kp@example.com and jo***el@example.com"
    left, right = masked.split(" and ")
    assert left != right


@pytest.mark.parametrize(
    ("local", "expected"),
    [("a", "*"), ("ab", "**"), ("abc", "a***"), ("abcd", "a***"), ("abcde", "ab***de")],
)
def test_short_local_parts_do_not_leak_through_the_mask(local: str, expected: str) -> None:
    assert mask_text(f"{local}@example.com") == f"{expected}@example.com"


def test_every_address_in_the_text_is_masked() -> None:
    masked = mask_text("a@x.com, bbbbbb@y.org; cccccc@z.net")
    assert "@" in masked
    assert "bbbbbb" not in masked and "cccccc" not in masked


def test_text_without_an_address_is_untouched() -> None:
    text = "Today was mostly uneventful, with no events or reminders scheduled."
    assert mask_text(text) == text


def test_masking_is_on_by_default_and_turns_off_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IRIS_GOVERNANCE_DISPLAY_MASK", raising=False)
    assert is_enabled()
    assert mask_text("jordan1.kp@example.com") != "jordan1.kp@example.com"

    monkeypatch.setenv("IRIS_GOVERNANCE_DISPLAY_MASK", "0")
    assert not is_enabled()
    assert mask_text("jordan1.kp@example.com") == "jordan1.kp@example.com"


# ── streaming ─────────────────────────────────────────────────────────────────
#
# An address arrives split across LLM tokens, so masking each chunk as it lands
# misses it. The masker holds back the trailing partial word until whitespace proves
# it complete.


def _stream(chunks: list[str]) -> str:
    masker = StreamMasker()
    return "".join(masker.feed(c) for c in chunks) + masker.flush()


def test_an_address_split_across_tokens_is_still_masked() -> None:
    assert _stream(["you have mail from jord", "an1.kp@ex", "ample.com today"]) == (
        "you have mail from jo***kp@example.com today"
    )


def test_an_address_at_the_very_end_needs_the_flush() -> None:
    assert _stream(["write to ", "jordan1.kp@example", ".com"]) == "write to jo***kp@example.com"


def test_the_stream_reproduces_the_text_when_nothing_matches() -> None:
    chunks = ["Today ", "was mostly ", "uneventful,", " with no events."]
    assert _stream(chunks) == "".join(chunks)


def test_a_newline_closes_a_word_too() -> None:
    assert _stream(["- jordan1.kp@example.com\n", "- next line"]) == (
        "- jo***kp@example.com\n- next line"
    )


def test_an_unbounded_word_is_not_held_for_ever() -> None:
    """No whitespace ever arrives; the masker must emit rather than buffer without
    bound. An address is at most 320 characters, so a longer run cannot be one."""
    masker = StreamMasker()
    emitted = masker.feed("x" * 400)
    assert emitted == "x" * 400
    assert masker.flush() == ""


def test_the_stream_masker_is_a_no_op_when_masking_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_DISPLAY_MASK", "0")
    assert _stream(["jord", "an1.kp@example.com"]) == "jordan1.kp@example.com"
