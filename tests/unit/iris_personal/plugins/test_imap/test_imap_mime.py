"""MIME: plain / HTML / alternative bodies, attachments, header and body encodings --
parsed alone and through a sync, body read and attachment download."""

from __future__ import annotations

from iris_personal.email.store import EmailStore
from iris_personal.plugins.imap.account import ImapAccount
from iris_personal.plugins.imap.mime import message_key, parse_mail, snippet_of
from iris_personal.plugins.imap.provider import ImapProvider

from .conftest import build_message
from .fake_imap_server import FakeMailbox

LATIN1_QP = (
    b"From: =?iso-8859-1?q?Caf=E9_Ren=E9?= <rene@cafe.example>\r\n"
    b"To: owner@example.test\r\n"
    b"Subject: =?utf-8?b?UsOpc2VydmF0aW9uIGNvbmZpcm3DqWU=?=\r\n"
    b"Date: Tue, 01 Sep 2026 10:00:00 +0000\r\n"
    b"Message-ID: <qp@cafe.example>\r\n"
    b"MIME-Version: 1.0\r\n"
    b"Content-Type: text/plain; charset=iso-8859-1\r\n"
    b"Content-Transfer-Encoding: quoted-printable\r\n"
    b"\r\n"
    b"Votre table au caf=E9 est r=E9serv=E9e.\r\n"
)

BAD_CHARSET = (
    b"From: a@b.example\r\nSubject: odd\r\nMessage-ID: <odd@b.example>\r\n"
    b"Content-Type: text/plain; charset=x-no-such-charset\r\n\r\nhello \xff world\r\n"
)


def test_plain_body_and_encoded_headers() -> None:
    mail = parse_mail(LATIN1_QP)
    assert mail.subject == "Réservation confirmée"
    assert "René" in mail.from_raw
    assert mail.body_text is not None and "réservée" in mail.body_text
    assert mail.message_id == "<qp@cafe.example>"


def test_html_only_mail_gets_a_text_snippet() -> None:
    raw = build_message(
        text=None,
        html="<html><style>p{}</style><body><p>Your <b>invoice</b> &amp; receipt</p></body></html>",
        message_id="<h@x.example>",
    )
    mail = parse_mail(raw)
    assert mail.body_text is None and mail.body_html is not None
    assert snippet_of(mail) == "Your invoice & receipt"


def test_alternative_keeps_both_and_prefers_plain_for_the_snippet() -> None:
    raw = build_message(text="plain wins", html="<p>html loses</p>", message_id="<a@x.example>")
    mail = parse_mail(raw)
    assert mail.body_text is not None and mail.body_html is not None
    assert snippet_of(mail) == "plain wins"


def test_attachment_metadata_matches_the_bytes() -> None:
    pdf = b"%PDF-1.4 synthetic statement"
    raw = build_message(
        subject="Statement",
        message_id="<s@bank.example>",
        attachments=[("statement.pdf", "application/pdf", pdf), ("rows.csv", "text/csv", b"a,b\n")],
    )
    mail = parse_mail(raw)
    got = {(a.filename, a.mime_type, a.size_bytes) for a in mail.attachments}
    assert got == {("statement.pdf", "application/pdf", len(pdf)), ("rows.csv", "text/csv", 4)}
    assert all(a.attachment_id.startswith("part-") for a in mail.attachments)
    assert mail.body_text is not None and "Plain body" in mail.body_text  # csv is not the body


def test_an_unknown_charset_decodes_with_replacement_not_a_crash() -> None:
    mail = parse_mail(BAD_CHARSET)
    assert mail.body_text is not None and "hello" in mail.body_text


def test_ids_are_per_account_and_survive_without_a_message_id() -> None:
    assert message_key("imap:a@x", "<m@x>", "fp") != message_key("imap:b@x", "<m@x>", "fp")
    assert message_key("imap:a@x", None, "fp1") == message_key("imap:a@x", None, "fp1")


def test_sync_body_read_and_attachment_download(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    pdf = b"%PDF-1.4 synthetic statement"
    mailbox.add_message(LATIN1_QP)
    mailbox.add_message(
        build_message(
            subject="Your statement",
            message_id="<s@bank.example>",
            text="Statement attached.",
            attachments=[("statement.pdf", "application/pdf", pdf)],
        )
    )
    result = provider.fetch_new(account.account_id, store=store)
    first, second = result.new_message_ids

    stored = store.get(second)
    assert stored is not None and [a.filename for a in stored.attachments] == ["statement.pdf"]
    assert "réservée" in provider.fetch_message_body(account.account_id, first)
    downloads = provider.fetch_message_attachments(
        account.account_id, second, mime_types=("application/pdf",)
    )
    assert [(d.meta.filename, d.content) for d in downloads] == [("statement.pdf", pdf)]
    assert provider.fetch_message_attachments(account.account_id, second, mime_types=("x/y",)) == []
    for uid in mailbox.folders["INBOX"].uids():
        assert "\\Seen" not in mailbox.flags_of("INBOX", uid)  # BODY.PEEK throughout


def test_attachment_search_matches_terms_and_skips_mail_without_files(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox
) -> None:
    mailbox.add_message(build_message(subject="Statement reminder", message_id="<r@b.example>"))
    mailbox.add_message(
        build_message(
            subject="Statement for August",
            message_id="<a@b.example>",
            attachments=[("aug.pdf", "application/pdf", b"%PDF")],
        )
    )
    mailbox.add_message(
        build_message(
            subject="Photos",
            message_id="<p@b.example>",
            attachments=[("beach.jpg", "image/jpeg", b"\xff\xd8")],
        )
    )

    found = provider.list_attachment_candidates(account.account_id, ["statement"])

    assert [c.subject for c in found] == ["Statement for August"]
    assert found[0].attachments[0].filename == "aug.pdf"
