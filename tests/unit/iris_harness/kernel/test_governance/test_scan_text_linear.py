"""``scan_text`` and the classifier's patterns run in linear time (issue #156).

The classifier scans untrusted text -- a document, a tool or MCP result, a file, a RAG
chunk -- with every pattern. Two of them backtracked quadratically: ``email`` (a 200 KB
``"a."*100000 + "@b."`` took 19.5 s; ``"+1-"*130000`` and ``"123-45-"*57000`` over 20 s) and
``voice_transcript_marker``. They are now linear matchers with the same meaning. Each test
here fails against the old patterns: the hostile inputs by time, the differential one by
comparing the new matchers with the original regexes on random text.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any

import pytest

from iris_harness.kernel.governance.file_scan import scan_text
from iris_harness.kernel.governance.plugins.classifier import DataClassifier
from iris_harness.kernel.governance.plugins.regex_packs import ALL_PACKS

# A gross guard only, generous for a loaded hosted runner (one run measured 1.2 s on a case that
# takes ~0.05 s on a laptop): the old patterns took 20-50 s on the issue's inputs, so a regression
# to quadratic time still crosses it. What PROVES linear time is the scaling test below, which
# does not depend on how fast the machine is.
BOUND_SECONDS = 8.0

_ORIGINAL_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ORIGINAL_VOICE = re.compile(r"\[voice_transcript:[^\]]*\]")

# The three inputs of the issue, at its sizes.
ISSUE_INPUTS = {
    "a.-run then @b.": "a." * 100_000 + "@b.",
    "+1- repeated": "+1-" * 130_000,
    "123-45- repeated": "123-45-" * 57_000,
}
# What else could make a pattern re-scan the same long run from many starts.
OTHER_HOSTILE = {
    "unclosed voice markers": "[voice_transcript:" * 30_000,
    "many @ and no domain": "a@" * 100_000,
    "dotted domain, no tld": "a@" + "b." * 100_000,
    "domain run reused by many locals": "x@y.z" * 40_000,
    "long local part, no @": "a%" * 100_000,
    "unicode word chars around runs": "é.é" * 66_000 + "@b.",
    "key prefixes": "sk-ant-" * 28_000 + "sk-or-" * 28_000 + "sk-" * 28_000,
    "pat prefixes": "github_pat_" * 18_000 + "ghp_" * 50_000,
    "jwt prefixes": "eyJ" * 66_000 + ".eyJ" * 50_000,
    "telegram-ish": "12345678:" * 22_000,
    "digit runs": "1" * 200_000,
    "phone-ish": "(123) 456-" * 20_000,
    "plus digits": "+" + "1 " * 100_000,
    "vault handles": "vault://" * 25_000,
    "pem headers": "-----BEGIN " * 18_000,
}


def _time(text: str) -> float:
    start = time.perf_counter()
    scan_text(text)
    return time.perf_counter() - start


# Runner-independent: doubling the input must not much more than double the time. Best of three on
# each side removes a noisy neighbour; the additive slack keeps millisecond timings from tripping
# the ratio. A quadratic pattern gives about 4x at 2N and fails whatever the machine.
SCALING = {
    "a.-run then @b.": lambda n: "a." * n + "@b.",
    "+1- repeated": lambda n: "+1-" * n,
    "123-45- repeated": lambda n: "123-45-" * n,
    "unclosed voice markers": lambda n: "[voice_transcript:" * n,
    "many @ and no domain": lambda n: "a@" * n,
}
_SCALING_N = 20_000
_RATIO = 3.0
_SLACK_SECONDS = 0.05


def _best_of_three(run: Any, text: str) -> float:
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        run(text)
        best = min(best, time.perf_counter() - start)
    return best


def _scales_linearly(run: Any, build: Any, n: int) -> tuple[bool, float, float]:
    small = _best_of_three(run, build(n))
    large = _best_of_three(run, build(2 * n))
    return large <= _RATIO * small + _SLACK_SECONDS, small, large


@pytest.mark.parametrize("name", list(SCALING))
def test_classification_time_grows_linearly_with_the_input(name: str) -> None:
    ok, small, large = _scales_linearly(scan_text, SCALING[name], _SCALING_N)

    assert ok, f"{name}: {small:.3f}s at N, {large:.3f}s at 2N (limit {_RATIO}x + slack)"


def test_the_scaling_check_does_catch_the_quadratic_regex_it_replaced() -> None:
    """The detector must bite: the original email regex (quadratic) fails it."""
    ok, small, large = _scales_linearly(_ORIGINAL_EMAIL.search, SCALING["a.-run then @b."], 8_000)

    assert not ok, f"the old regex scaled linearly here ({small:.3f}s -> {large:.3f}s)?"


@pytest.mark.parametrize("name", list(ISSUE_INPUTS))
def test_the_hostile_inputs_of_the_issue_classify_in_a_small_bound(name: str) -> None:
    assert _time(ISSUE_INPUTS[name]) < BOUND_SECONDS


@pytest.mark.parametrize("name", list(OTHER_HOSTILE))
def test_other_hostile_inputs_classify_in_a_small_bound(name: str) -> None:
    assert _time(OTHER_HOSTILE[name]) < BOUND_SECONDS


def test_every_pattern_in_every_pack_is_fast_on_every_hostile_input() -> None:
    """Pattern by pattern, so a slow one is named (the whole-text timing hides which)."""
    slow: list[str] = []
    for pack, entries in ALL_PACKS.items():
        for name, pattern, _ in entries:
            for label, text in {**ISSUE_INPUTS, **OTHER_HOSTILE}.items():
                start = time.perf_counter()
                pattern.search(text)
                if time.perf_counter() - start >= BOUND_SECONDS:
                    slow.append(f"{pack}/{name} on {label}")
    assert slow == []


# ------------------------------------------------------------ every true positive still matches
TRUE_POSITIVES = {
    "email": [
        "write to jane.doe@example.com today",
        "a@b.co",
        "x_y+tag%z-1@sub.domain.example.org",
        "(see a.b@c.io)",
        "first@x.io second@y.io",
        "@x.io alice@x.io",  # a bare @ before a real one
        "bad@@ok@example.com",
        "é a@b.co",  # a word character that is not in the local-part class, before the run
        "1@b.co",
        "_a@b.co" + " .a@b.co",  # the first needs the second's boundary; either alone matches
    ],
    "ssn_us": ["ssn 123-45-6789 on file"],
    "phone_us_strict": ["call (555) 123-4567", "555-123-4567", "555.123.4567"],
    "phone_intl_plus": ["+44 20 7946 0958", "+1-415-555-0100"],
    "openai_key": ["sk-" + "a" * 24],
    "anthropic_key": ["sk-ant-" + "A1_-" * 8],
    "openrouter_key": ["sk-or-" + "a1" * 12],
    "github_pat_classic": ["ghp_" + "a" * 36],
    "github_pat_finegrained": ["github_pat_" + "a_" * 30],
    "aws_access_key_id": ["AKIA" + "A" * 16],
    "jwt": ["eyJhbGciOi.eyJzdWIiOiIxIn0.sig_nature-1"],
    "pem_private_key": ["-----BEGIN RSA PRIVATE KEY-----", "-----BEGIN PRIVATE KEY-----"],
    "telegram_bot_token": ["12345678:" + "a" * 35],
    "vault_handle": ["use vault://github/token please"],
    "voice_transcript_marker": ["[voice_transcript: hello there]", "x [voice_transcript:]"],
}


@pytest.mark.parametrize(
    ("pattern_name", "text"),
    [(name, text) for name, texts in TRUE_POSITIVES.items() for text in texts],
)
def test_every_true_positive_still_matches(pattern_name: str, text: str) -> None:
    patterns = {n: p for entries in ALL_PACKS.values() for n, p, _ in entries}
    assert patterns[pattern_name].search(text), (pattern_name, text)
    assert any(m.endswith(f"/{pattern_name}") for m in scan_text(text).matched)


NEGATIVES = [
    "no address here",
    "a@b",  # no top-level label
    "a@b.c",  # one-letter label
    "a@.co",
    "@b.co",
    "foo @ bar.com",
    "[voice_transcript: never closed",
    "voice_transcript: [] no prefix bracket",
    "5551234567",  # a bare ten digits is not a phone
]


@pytest.mark.parametrize("text", NEGATIVES)
def test_the_near_misses_still_do_not_match(text: str) -> None:
    assert scan_text(text).matched == ()


def test_the_classification_of_a_mixed_text_is_unchanged() -> None:
    text = "mail jane@example.com, key sk-" + "a" * 24 + ", [voice_transcript: hi]"
    result = DataClassifier().classify(text)
    assert result.classification == "secret"
    assert set(result.matched_patterns) >= {
        "pii/email",
        "credentials/openai_key",
        "iris_specific/voice_transcript_marker",
    }


# ------------------------------------------- the new matchers mean what the old regexes meant
_ALPHABET = list("ab_.%+-@ 1é]x[\n") + ["voice_transcript:", "[voice_transcript:", "@b.", ".com"]


def test_the_new_matchers_agree_with_the_original_regexes_on_random_text() -> None:
    """A differential test: the original ``email`` and ``voice_transcript_marker`` regexes are
    the reference, kept here only to compare against. 40,000 short random strings built from
    the characters that matter (word and non-word characters of the local-part class, ``@``,
    ``]``, a non-ASCII word character, the markers); the matchers must give the same answer."""
    patterns = {n: p for entries in ALL_PACKS.values() for n, p, _ in entries}
    rng = random.Random(156)  # noqa: S311 - a seeded test generator, not a secret
    for _ in range(40_000):
        text = "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 14)))
        assert bool(patterns["email"].search(text)) == bool(_ORIGINAL_EMAIL.search(text)), text
        assert bool(patterns["voice_transcript_marker"].search(text)) == bool(
            _ORIGINAL_VOICE.search(text)
        ), text


def test_the_matchers_still_work_on_a_large_text_that_does_contain_a_match() -> None:
    big = "filler. " * 25_000
    assert scan_text(big + "ping jane@example.com " + big).matched == ("pii/email",)
    assert scan_text(big + "[voice_transcript: hi] " + big).matched == (
        "iris_specific/voice_transcript_marker",
    )
