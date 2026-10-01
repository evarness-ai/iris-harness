"""Tests for read-only Gmail attachment fetch (Finance F1)."""

from __future__ import annotations

import base64

from iris_personal.email.contracts import EmailAttachment
from iris_personal.plugins.gmail.gmail_attachments import (
    DownloadedAttachment,
    attachments_from_payload,
    download_attachment,
    fetch_message_attachments,
)

# ─── fake Gmail service ─────────────────────────────────────────────────


class _Req:
    def __init__(self, result: object) -> None:
        self._result = result

    def execute(self) -> object:
        return self._result


class FakeMessages:
    def __init__(self, full_payload: dict, attachment_data: dict[str, bytes]) -> None:
        self._full = full_payload
        self._attachment_data = attachment_data

    def get(self, *, userId: str, id: str, format: str = "metadata") -> _Req:
        return _Req(self._full)

    def attachments(self) -> FakeAttachments:
        return FakeAttachments(self._attachment_data)


class FakeAttachments:
    def __init__(self, data: dict[str, bytes]) -> None:
        self._data = data

    def get(self, *, userId: str, messageId: str, id: str) -> _Req:
        raw = self._data.get(id, b"")
        return _Req({"data": base64.urlsafe_b64encode(raw).decode(), "size": len(raw)})


class FakeUsers:
    def __init__(self, messages: FakeMessages) -> None:
        self._messages = messages

    def messages(self) -> FakeMessages:
        return self._messages


class FakeService:
    def __init__(self, full_payload: dict, attachment_data: dict[str, bytes]) -> None:
        self._users = FakeUsers(FakeMessages(full_payload, attachment_data))

    def users(self) -> FakeUsers:
        return self._users


def _multipart_payload() -> dict:
    return {
        "id": "msg1",
        "payload": {
            "mimeType": "multipart/mixed",
            "parts": [
                {"mimeType": "text/plain", "filename": "", "body": {"size": 12}},
                {
                    "mimeType": "application/pdf",
                    "filename": "statement.pdf",
                    "body": {"size": 4, "attachmentId": "att-pdf"},
                },
                {
                    # inline image: has body but no filename → not an attachment
                    "mimeType": "image/png",
                    "filename": "",
                    "body": {"size": 99, "attachmentId": "att-img"},
                },
            ],
        },
    }


# ─── attachments_from_payload (pure) ───────────────────────────────────


def test_attachments_from_payload_finds_named_parts() -> None:
    atts = attachments_from_payload(_multipart_payload())
    assert len(atts) == 1
    assert atts[0].filename == "statement.pdf"
    assert atts[0].mime_type == "application/pdf"
    assert atts[0].attachment_id == "att-pdf"


def test_attachments_from_payload_ignores_inline_and_text() -> None:
    """Inline image (no filename) and text body are not attachments."""
    atts = attachments_from_payload(_multipart_payload())
    ids = {a.attachment_id for a in atts}
    assert "att-img" not in ids


def test_attachments_from_payload_handles_nested_parts() -> None:
    payload = {
        "payload": {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {
                            "mimeType": "application/pdf",
                            "filename": "nested.pdf",
                            "body": {"size": 1, "attachmentId": "deep"},
                        }
                    ],
                }
            ],
        }
    }
    atts = attachments_from_payload(payload)
    assert [a.filename for a in atts] == ["nested.pdf"]


def test_attachments_from_payload_empty_when_none() -> None:
    assert attachments_from_payload({"payload": {"mimeType": "text/plain", "body": {}}}) == ()


# ─── download / fetch (fake service) ───────────────────────────────────


def test_download_attachment_decodes_bytes() -> None:
    svc = FakeService(_multipart_payload(), {"att-pdf": b"%PDF"})
    meta = EmailAttachment(
        filename="statement.pdf", mime_type="application/pdf", size_bytes=4, attachment_id="att-pdf"
    )
    got = download_attachment("gmail:x@y.com", "msg1", meta, service=svc)
    assert isinstance(got, DownloadedAttachment)
    assert got.content == b"%PDF"


def test_fetch_message_attachments_filters_mime() -> None:
    svc = FakeService(_multipart_payload(), {"att-pdf": b"%PDF-1.4"})
    out = fetch_message_attachments(
        "gmail:x@y.com", "msg1", mime_types=("application/pdf",), service=svc
    )
    assert len(out) == 1
    assert out[0].content == b"%PDF-1.4"
    assert out[0].meta.filename == "statement.pdf"
