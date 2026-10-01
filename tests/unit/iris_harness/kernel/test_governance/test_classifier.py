"""DataClassifier Stage 1 tests.

Covers per-pack matching, severity ordering, false-positive avoidance
on plain text, and integration through the kernel as a PreClassify hook.
"""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookPoint,
)
from iris_harness.kernel.governance.plugins import (
    ClassificationResult,
    DataClassifier,
    DataClassifierHook,
)


@pytest.fixture()
def classifier() -> DataClassifier:
    return DataClassifier()


def test_empty_text_is_public(classifier: DataClassifier) -> None:
    assert classifier.classify("").classification == "public"


def test_plain_conversational_text_is_public(classifier: DataClassifier) -> None:
    """Privacy-critical: don't false-positive on everyday text."""
    samples = [
        "what is the weather like today",
        "explain how transformers work in plain english",
        "write a function that reverses a list in python",
        "the meeting is at 3pm tomorrow",
    ]
    for text in samples:
        result = classifier.classify(text)
        assert result.classification == "public", f"false positive on: {text!r}"


@pytest.mark.parametrize(
    "text,expected_pattern",
    [
        ("here is my key: sk-ABC1234567890abcdef1234", "credentials/openai_key"),
        (
            "anthropic key sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
            "credentials/anthropic_key",
        ),
        (
            "openrouter sk-or-v1-abcdefghijklmnopqrstuvwxyz",
            "credentials/openrouter_key",
        ),
        (
            "github pat ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "credentials/github_pat_classic",
        ),
        (  # exp-007: fine-grained PAT was previously unclassified (-> public)
            "token github_pat_11ABCDEFG0abcdefghijklm_"
            "nopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd",
            "credentials/github_pat_finegrained",
        ),
        (
            "refresh ghr_abcdefghijklmnopqrstuvwxyz0123456789",
            "credentials/github_pat_refresh",
        ),
        ("AKIAIOSFODNN7EXAMPLE", "credentials/aws_access_key_id"),
        (
            "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2Q",
            "credentials/jwt",
        ),
        (
            "-----BEGIN RSA PRIVATE KEY-----\nMIIE...",
            "credentials/pem_private_key",
        ),
        (
            "1234567890:AAH-abcdefghijklmnopqrstuvwxyz1234567",
            "iris_specific/telegram_bot_token",
        ),
    ],
)
def test_credential_patterns_classify_as_secret(
    classifier: DataClassifier, text: str, expected_pattern: str
) -> None:
    result = classifier.classify(text)
    assert result.classification == "secret"
    assert expected_pattern in result.matched_patterns


@pytest.mark.parametrize(
    "text,expected_pattern",
    [
        ("contact me at alice@example.com", "pii/email"),
        ("my ssn is 123-45-6789", "pii/ssn_us"),
        ("call me at 415-555-1234", "pii/phone_us_strict"),
        ("(415) 555 1234 is my number", "pii/phone_us_strict"),
        ("international: +44 20 7946 0958", "pii/phone_intl_plus"),
        ("[voice_transcript: please call mom tonight]", "iris_specific/voice_transcript_marker"),
    ],
)
def test_pii_patterns_classify_as_personal(
    classifier: DataClassifier, text: str, expected_pattern: str
) -> None:
    result = classifier.classify(text)
    assert result.classification == "personal"
    assert expected_pattern in result.matched_patterns


def test_vault_handle_classifies_as_internal(classifier: DataClassifier) -> None:
    """vault:// in a prompt is a leak signal but not itself a secret value."""
    result = classifier.classify("please use vault://github-pat-coding")
    assert result.classification == "internal"
    assert "iris_specific/vault_handle" in result.matched_patterns


def test_severity_ordering_secret_beats_personal(classifier: DataClassifier) -> None:
    """When both a credential and an email appear, classification is secret."""
    text = "email: alice@example.com api key: sk-ABC1234567890abcdef1234"
    result = classifier.classify(text)
    assert result.classification == "secret"
    matches = set(result.matched_patterns)
    assert "credentials/openai_key" in matches
    assert "pii/email" in matches


def test_severity_ordering_personal_beats_internal(classifier: DataClassifier) -> None:
    text = "user alice@example.com referenced vault://foo"
    result = classifier.classify(text)
    assert result.classification == "personal"


def test_custom_packs_override_default() -> None:
    """Callers can inject their own pack set (e.g. test isolation)."""
    import re

    custom_classifier = DataClassifier(
        packs={
            "test_pack": [
                ("test_secret", re.compile(r"MY_SECRET"), "secret"),
            ]
        }
    )
    result = custom_classifier.classify("contains MY_SECRET value")
    assert result.classification == "secret"
    # Default packs disabled — email should not trigger personal
    result = custom_classifier.classify("alice@example.com")
    assert result.classification == "public"


async def test_hook_writes_classification_to_context() -> None:
    """Integration: DataClassifierHook via kernel writes ctx.classification."""
    kernel = GovernanceKernel()
    kernel.register(DataClassifierHook())
    kernel.init_lock()

    ctx = HookContext(
        hook_point=HookPoint.PRE_CLASSIFY,
        run_id="run-classify",
        agent_type="chat",
        payload={"prompt": "my email is alice@example.com"},
    )
    decision, final_ctx = await kernel.fire(HookPoint.PRE_CLASSIFY, ctx)

    assert decision.outcome == "allow"
    assert final_ctx.classification == "personal"
    assert "matched_patterns" in decision.audit_metadata


async def test_hook_extracts_text_from_alternate_payload_keys() -> None:
    kernel = GovernanceKernel()
    kernel.register(DataClassifierHook())
    kernel.init_lock()

    for key in ("text", "prompt", "input", "message"):
        ctx = HookContext(
            hook_point=HookPoint.PRE_CLASSIFY,
            run_id=f"run-{key}",
            agent_type="chat",
            payload={key: "contact alice@example.com"},
        )
        _, final_ctx = await kernel.fire(HookPoint.PRE_CLASSIFY, ctx)
        assert final_ctx.classification == "personal", f"failed for key={key}"


async def test_hook_with_no_text_payload_classifies_public() -> None:
    kernel = GovernanceKernel()
    kernel.register(DataClassifierHook())
    kernel.init_lock()

    ctx = HookContext(
        hook_point=HookPoint.PRE_CLASSIFY,
        run_id="run-empty",
        agent_type="chat",
        payload={"unrelated_key": 42},
    )
    decision, final_ctx = await kernel.fire(HookPoint.PRE_CLASSIFY, ctx)
    assert decision.outcome == "allow"
    assert final_ctx.classification == "public"


def test_classification_result_is_frozen() -> None:
    """Defense-in-depth: ClassificationResult shouldn't be mutable by callers."""
    result = ClassificationResult(classification="secret")
    with pytest.raises((AttributeError, TypeError, Exception)):
        result.classification = "public"  # type: ignore[misc]
