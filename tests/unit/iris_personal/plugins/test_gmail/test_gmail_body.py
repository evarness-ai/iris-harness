"""Issue 0002 item C — Gmail message-body extraction (pure; no network)."""

from __future__ import annotations

import base64

from iris_personal.plugins.gmail.gmail_fetch import _extract_body_text


def _b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii")


def test_extract_plain_text_body() -> None:
    payload = {"mimeType": "text/plain", "body": {"data": _b64("Hello world body")}}
    assert _extract_body_text(payload) == "Hello world body"


def test_extract_prefers_plain_over_html_in_multipart() -> None:
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/html", "body": {"data": _b64("<p>html version</p>")}},
            {"mimeType": "text/plain", "body": {"data": _b64("plain version wins")}},
        ],
    }
    assert _extract_body_text(payload) == "plain version wins"


def test_extract_recurses_into_nested_parts() -> None:
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {"mimeType": "application/pdf", "body": {"attachmentId": "x"}},
            {
                "mimeType": "multipart/alternative",
                "parts": [{"mimeType": "text/plain", "body": {"data": _b64("nested body")}}],
            },
        ],
    }
    assert _extract_body_text(payload) == "nested body"


def test_extract_falls_back_to_stripped_html() -> None:
    html = "<style>p{}</style><p>Hi <b>there</b></p><script>evil()</script>"
    payload = {"mimeType": "text/html", "body": {"data": _b64(html)}}
    out = _extract_body_text(payload)
    assert "Hi there" in out
    assert "<" not in out and "evil()" not in out  # tags + script/style content stripped


def test_extract_empty_when_no_text_part() -> None:
    payload = {"mimeType": "application/pdf", "body": {"attachmentId": "x"}}
    assert _extract_body_text(payload) == ""


# A card issuer's statement email: a text/plain stub (a link plus a long tracking
# URL) and the figures in the HTML part alone.
_STUB = (
    "Adatum Card(R) Add cards@mail.adatumbank.test to your address book to ensure delivery. "
    "OWNER Please visit the following link to view your message: "
    "https://links.mail.adatumbank.test/go?h=" + "a1b2c3" * 60 + " Privacy Security "
    "Adatum Card Customer Service P. O. Box 0000 Anytown, ST 00000"
)
_HTML = (
    "<html><body><p>Hi, Owner. Your statement ending on Sep 16, 2026 is ready.</p>"
    "<table><tr><td>Payment Due Date:</td><td>Wednesday, October 14, 2026</td></tr>"
    "<tr><td>Statement Balance:</td><td>&#36;1,975.30</td></tr>"
    "<tr><td>Minimum Payment Due:</td><td>$45.00</td></tr></table>"
    + "<p>Adatum Card is at your side before, during and after every purchase.</p>" * 12
    + "</body></html>"
)


def test_a_plain_stub_loses_to_the_html_that_holds_the_figures() -> None:
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": _b64(_STUB)}},
            {"mimeType": "text/html", "body": {"data": _b64(_HTML)}},
        ],
    }
    body = _extract_body_text(payload)
    assert "Statement Balance: $1,975.30" in body  # the entity is decoded too
    assert "Minimum Payment Due: $45.00" in body
    assert "Please visit the following link" not in body


def test_a_full_plain_part_still_wins_over_its_html_twin() -> None:
    text = "Your statement is ready. New balance $1,357.20. Minimum $35.00 due Oct 13."
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": _b64(text)}},
            {"mimeType": "text/html", "body": {"data": _b64(f"<div><p>{text}</p></div>")}},
        ],
    }
    assert _extract_body_text(payload) == text
