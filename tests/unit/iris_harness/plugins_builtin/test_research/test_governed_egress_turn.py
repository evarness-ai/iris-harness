"""A real turn that searches and reads a page leaves egress rows for both (issue #172).

The research tool runs on the governed loop with the builtin research plugin mounted, a keyed
provider enabled, and the transport faked: the search goes to Brave and the page fetch to the
site, and each is a PRE and a POST row in the ledger, attributed to the ``research`` tool.
Through both chat entries, since the REPL uses ``/chat/stream``.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from iris_harness.testing import fake_http, harness

_SCRIPT: dict[str, Any] = {
    "rules": [
        {
            "name": "answer",
            "match": {"user": r"Continue from the last Observation"},
            "reply": {"content": "Thought: Done.\nFinal Answer: It is mild in Oslo."},
        },
        {
            "name": "search",
            "match": {"user": r"User: .*weather in Oslo"},
            "reply": {
                "content": "Thought: search.\nAction: research\n"
                'Action Input: {"query": "weather in Oslo", "fetch_content": true}'
            },
        },
    ],
    "default": {"content": "Thought: x\nFinal Answer: default."},
}
_BRAVE = "https://api.search.brave.com/res/v1/web/search"
_PAGE = "https://weather.example.org/oslo"


def _answer(request: httpx.Request) -> httpx.Response:
    bare = str(request.url.copy_with(query=None))
    if bare == _BRAVE:
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {"title": "Oslo weather", "url": _PAGE, "description": "mild and dry"}
                    ]
                }
            },
        )
    if bare == _PAGE:
        return httpx.Response(
            200,
            content=b"<html><body><article><h1>Oslo</h1><p>"
            + b"Oslo is mild this week with little rain across the city. " * 8
            + b"</p></article></body></html>",
        )
    return httpx.Response(404)


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_research_turn_records_the_search_and_the_page_fetch(entry: str) -> None:
    with (
        harness(
            profile="default",
            fake_model=_SCRIPT,
            env={"BRAVE_API_KEY": "test-key", "IRIS_AGENTIC_CORE": "on"},
        ) as h,
        fake_http(_answer) as sent,
    ):
        if entry == "chat":
            assert h.chat("What is the weather in Oslo?", session_id="s1").text
        else:
            assert h.chat_stream("What is the weather in Oslo?", session_id="s1").answered
        hosts = sorted({str(r.url.host) for r in sent})
        assert hosts == ["api.search.brave.com", "weather.example.org"]
        rows = [r for r in h.audit_rows() if r.hook_point in ("pre_egress", "post_egress")]
        assert {(r.hook_point, r.egress["host"]) for r in rows if r.egress} >= {
            ("pre_egress", "api.search.brave.com"),
            ("post_egress", "api.search.brave.com"),
            ("pre_egress", "weather.example.org"),
            ("post_egress", "weather.example.org"),
        }
