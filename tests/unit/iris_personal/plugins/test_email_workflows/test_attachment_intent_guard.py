"""The email-attachment detector must yield FileManager-domain phrasings.

"show me the file catalog" / "files in my folder" share the file/document
vocabulary with attachment asks but mean on-disk files — they should route to
the FileManager agent, not the inbox attachment search (the live misroute).
Genuine attachment asks ("get my passport copy", "the invoice pdf") stay email.
"""

from __future__ import annotations

import pytest

from iris_personal.plugins.email_workflows.agent import _is_email_attachment_intent


@pytest.mark.parametrize(
    "query",
    [
        "show me the file catalog",
        "list the files in my catalog",
        "what's in my file vault",
        "files in my downloads folder",
        "open my folder",
        "check the file manager",
    ],
)
def test_filemanager_domain_is_not_attachment(query: str) -> None:
    assert _is_email_attachment_intent(query) is False


@pytest.mark.parametrize(
    "query",
    [
        "find the passport attachment",
        "get me the invoice pdf",
        "pull up my resume",
        "get my passport copy",
        "the aadhaar card attachment",
        "download the boarding pass",
    ],
)
def test_genuine_attachment_asks_still_match(query: str) -> None:
    assert _is_email_attachment_intent(query) is True
