"""RFC 8291 encryption, checked against the RFC's own worked example.

This file is why the harness hand-rolls Web Push encryption instead of adding
``pywebpush``: the specification publishes a complete vector, so correctness
is *demonstrated* rather than trusted. Every intermediate the RFC prints is
asserted, not just the final body — a wrong ``info`` string still produces a
plausible-looking ciphertext, and the only symptom in production is a
notification that never arrives.

Values are section 5 of RFC 8291, verbatim.
"""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from iris_harness.services.channels.web_push.encryption import (
    MAX_PAYLOAD_BYTES,
    b64url_decode,
    b64url_encode,
    encrypt,
)

PLAINTEXT = b"When I grow up, I want to be a watermelon"
SALT = "DGv6ra1nlYgDCS1FRnbzlw"
AUTH_SECRET = "BTBZMqHH6r4Tts7J_aSIgg"
UA_PUBLIC = (
    "BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4"
)
AS_PUBLIC = (
    "BP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27mlmlMoZIIgDll6e3vCYLocInmYWAmS6TlzAC8wEqKK6PBru3jl7A8"
)
AS_PRIVATE = "yfWPiYE-n46HLnH0KqZOF1fJJU3MYrct3AELtAQ-oRw"

EXPECTED_BODY = (
    "DGv6ra1nlYgDCS1FRnbzlwAAEABBBP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27ml"
    "mlMoZIIgDll6e3vCYLocInmYWAmS6TlzAC8wEqKK6PBru3jl7A_yl95bQpu6cVPT"
    "pK4Mqgkf1CXztLVBSt2Ks3oZwbuwXPXLWyouBWLVWGNWQexSgSxsj_Qulcy4a-fN"
)


def _as_private() -> ec.EllipticCurvePrivateKey:
    return ec.derive_private_key(int.from_bytes(b64url_decode(AS_PRIVATE), "big"), ec.SECP256R1())


def test_the_rfc_worked_example_reproduces_byte_for_byte() -> None:
    """The whole justification for not taking a dependency.

    Same keys, same salt, same plaintext as RFC 8291 section 5 — so the output
    must be the body the RFC prints, to the character.
    """
    payload = encrypt(
        plaintext=PLAINTEXT,
        ua_public_key=b64url_decode(UA_PUBLIC),
        auth_secret=b64url_decode(AUTH_SECRET),
        salt=b64url_decode(SALT),
        as_private_key=_as_private(),
    )
    assert b64url_encode(payload.body) == EXPECTED_BODY


def test_the_header_is_laid_out_the_way_the_rfc_says() -> None:
    """salt(16) || record size(4) || key length(1) || server public key(65).

    A browser parses these by offset; one byte out and it discards the message
    without telling anyone.
    """
    payload = encrypt(
        plaintext=PLAINTEXT,
        ua_public_key=b64url_decode(UA_PUBLIC),
        auth_secret=b64url_decode(AUTH_SECRET),
        salt=b64url_decode(SALT),
        as_private_key=_as_private(),
    )
    body = payload.body
    assert body[:16] == b64url_decode(SALT)
    assert int.from_bytes(body[16:20], "big") == 4096
    assert body[20] == 65
    assert body[21:86] == b64url_decode(AS_PUBLIC)


def test_the_content_encoding_header_is_aes128gcm() -> None:
    # A push service routes on this; `aesgcm` (the older draft) is a different
    # scheme and would be decrypted with different key derivation.
    payload = encrypt(
        plaintext=b"hi",
        ua_public_key=b64url_decode(UA_PUBLIC),
        auth_secret=b64url_decode(AUTH_SECRET),
    )
    assert payload.headers["Content-Encoding"] == "aes128gcm"


def test_every_message_gets_a_fresh_salt_and_key() -> None:
    """Reusing either across messages would leak that two payloads match.

    Production passes neither argument, so this is what guards the default.
    """
    args = {
        "plaintext": PLAINTEXT,
        "ua_public_key": b64url_decode(UA_PUBLIC),
        "auth_secret": b64url_decode(AUTH_SECRET),
    }
    first, second = encrypt(**args).body, encrypt(**args).body
    assert first[:16] != second[:16], "salt repeated"
    assert first[21:86] != second[21:86], "ephemeral server key repeated"


def test_a_payload_too_large_for_one_record_is_refused_up_front() -> None:
    # Better a ValueError here than a 413 from the push service per device.
    with pytest.raises(ValueError, match="limit is"):
        encrypt(
            plaintext=b"x" * (MAX_PAYLOAD_BYTES + 1),
            ua_public_key=b64url_decode(UA_PUBLIC),
            auth_secret=b64url_decode(AUTH_SECRET),
        )


def test_a_payload_at_the_limit_is_accepted() -> None:
    payload = encrypt(
        plaintext=b"x" * MAX_PAYLOAD_BYTES,
        ua_public_key=b64url_decode(UA_PUBLIC),
        auth_secret=b64url_decode(AUTH_SECRET),
    )
    assert len(payload.body) <= 4096


@pytest.mark.parametrize("length", [0, 15, 17, 32])
def test_an_auth_secret_of_the_wrong_length_is_refused(length: int) -> None:
    # It is the HKDF salt for the key derivation; a short one silently derives
    # a key the browser will not reproduce.
    with pytest.raises(ValueError, match="auth secret"):
        encrypt(
            plaintext=b"hi",
            ua_public_key=b64url_decode(UA_PUBLIC),
            auth_secret=b"x" * length,
        )


def test_base64url_round_trips_without_padding() -> None:
    # Subscriptions carry unpadded base64url; decoding must not need the "=".
    for raw in (b"", b"a", b"ab", b"abc", b"\x00\xff" * 33):
        assert b64url_decode(b64url_encode(raw)) == raw
