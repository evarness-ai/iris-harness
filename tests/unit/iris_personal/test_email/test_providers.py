"""The mail-provider registry — the core's one door to a mailbox (OSS plan M5.7 track A).

The core owns the store and every read over it; a mailbox is a plugin. These pin the
seam from the core side: lookup by account id or provider name, keyed replacement, and
that with nothing registered the answer is ``None`` — never an import of a plugin.
"""

from __future__ import annotations

import inspect

import pytest

from iris_personal.email import providers
from iris_personal.email.providers import (
    AttachmentCandidate,
    FetchResult,
    MailProvider,
    clear_mail_providers,
    mail_provider_for,
    register_mail_provider,
    registered_mail_providers,
)


class _Stub:
    name = "gmail"

    def fetch_new(self, account_id, *, store=None, max_messages=100, cold_start_days=30):  # type: ignore[no-untyped-def]
        return FetchResult(account_id, 0, (), None, False)

    def reset_cursor(self, account_id, *, store):  # type: ignore[no-untyped-def]
        return None

    def fetch_message_body(self, account_id, message_id, *, max_chars=4000):  # type: ignore[no-untyped-def]
        return ""

    def fetch_message_attachments(self, account_id, message_id, *, mime_types=None, service=None):  # type: ignore[no-untyped-def]
        return []

    def list_attachment_candidates(self, account_id, terms, *, limit=15):  # type: ignore[no-untyped-def]
        return [AttachmentCandidate("s", "f", ())]

    def trash_messages(self, account_id, message_ids):  # type: ignore[no-untyped-def]
        return list(message_ids)

    def restore_messages(self, account_id, message_ids, *, labels_before=None):  # type: ignore[no-untyped-def]
        return []


@pytest.fixture(autouse=True)
def _clean() -> None:
    clear_mail_providers()
    yield  # type: ignore[misc]
    clear_mail_providers()


def test_nothing_registered_means_none_not_an_import() -> None:
    assert mail_provider_for("gmail:user@x.com") is None
    assert mail_provider_for("gmail") is None
    assert registered_mail_providers() == ()
    assert "plugins_builtin" not in inspect.getsource(providers)


def test_lookup_by_account_id_or_bare_name() -> None:
    stub = _Stub()
    register_mail_provider(stub)

    assert mail_provider_for("gmail:user@x.com") is stub
    assert mail_provider_for("gmail") is stub
    assert mail_provider_for("outlook:user@x.com") is None
    assert isinstance(stub, MailProvider)  # the stub satisfies the runtime-checkable protocol


def test_registration_is_keyed_so_a_second_runtime_replaces_not_stacks() -> None:
    first, second = _Stub(), _Stub()
    register_mail_provider(first)
    register_mail_provider(second)

    assert mail_provider_for("gmail") is second
    assert registered_mail_providers() == ("gmail",)


def test_the_core_email_package_no_longer_holds_a_gmail_module() -> None:
    import importlib

    for gone in ("iris_personal.email.gmail_fetch", "iris_personal.email.gmail_attachments"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(gone)
