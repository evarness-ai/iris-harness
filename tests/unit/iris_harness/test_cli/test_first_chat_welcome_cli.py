"""The first-chat welcome on the CLI (ADR-0127): the REPL and print mode ask, and print it once.

The harness decides (``POST /chat/welcome``); the CLI prints the text only when its call
is the one that ran it, and a welcome it cannot fetch never stops the chat.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import iris_harness.cli.modes as modes
import iris_harness.cli.repl as repl
import iris_harness.cli.welcome as welcome
from iris_harness.cli.welcome import fetch_welcome

API = "http://iris.test:8003"


def _serve(body: dict[str, Any], seen: list[httpx.Request]) -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body)

    return handle


def test_a_new_welcome_is_returned(cli_api: Any) -> None:
    seen: list[httpx.Request] = []
    cli_api(_serve({"created": True, "response": "Hi, I'm IRIS."}, seen), welcome)
    assert fetch_welcome(API) == "Hi, I'm IRIS."
    assert seen[0].method == "POST" and seen[0].url.path == "/chat/welcome"
    assert seen[0].headers.get("authorization"), "authenticated like the other chat calls"


def test_a_welcome_that_ran_before_is_not_shown_again(cli_api: Any) -> None:
    cli_api(_serve({"created": False, "response": "Hi, I'm IRIS."}, []), welcome)
    assert fetch_welcome(API) is None


def test_an_api_out_of_reach_does_not_stop_the_chat(cli_api: Any) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    cli_api(down, welcome)
    assert fetch_welcome(API) is None


def test_print_mode_prints_the_welcome_before_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(modes, "fetch_welcome", lambda api_url, channel: "WELCOME")
    monkeypatch.setattr(modes, "_post", lambda *a, **k: {"response": "ANSWER", "metadata": {}})
    monkeypatch.setattr(modes, "print_response", lambda text, **_k: printed.append(text))
    monkeypatch.setattr(modes, "render_footer", lambda *_a, **_k: None)
    session = SimpleNamespace(id="s1")
    manager = SimpleNamespace(touch=lambda _s: None)

    code = modes.run_print_mode(
        "hello",
        session=session,  # type: ignore[arg-type]
        session_manager=manager,  # type: ignore[arg-type]
        api_url=API,
    )

    assert code == 0
    assert printed == ["WELCOME", "ANSWER"]


def test_json_mode_never_adds_a_welcome(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("json mode is a machine protocol: one reply per request")

    monkeypatch.setattr(modes, "fetch_welcome", must_not)
    monkeypatch.setattr(modes, "_post", lambda *a, **k: {"response": "ANSWER"})
    session = SimpleNamespace(id="s1")
    manager = SimpleNamespace(touch=lambda _s: None)
    modes.run_json_mode(
        "hello",
        session=session,  # type: ignore[arg-type]
        session_manager=manager,  # type: ignore[arg-type]
        api_url=API,
    )


@pytest.mark.parametrize(("new", "expected"), [("WELCOME", ["WELCOME"]), (None, [])])
def test_the_repl_opens_with_the_welcome_when_it_is_new(
    monkeypatch: pytest.MonkeyPatch, new: str | None, expected: list[str]
) -> None:
    printed: list[str] = []
    asked: list[tuple[str, str]] = []

    class _Prompt:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def prompt(self, *_a: Any, **_k: Any) -> str:
            raise EOFError  # the owner leaves at once

    def fetch(api_url: str, *, channel: str) -> str | None:
        asked.append((api_url, channel))
        return new

    monkeypatch.setattr(repl, "PromptSession", _Prompt)
    monkeypatch.setattr(repl, "FileHistory", lambda *_a, **_k: None)
    monkeypatch.setattr(repl, "print_banner", lambda **_k: None)
    monkeypatch.setattr(repl, "fetch_welcome", fetch)
    monkeypatch.setattr(repl, "print_response", lambda text, **_k: printed.append(text))
    providers = SimpleNamespace(get_active=lambda: SimpleNamespace(display_name="test", model="m"))

    code = repl.run_repl(
        session=SimpleNamespace(id="s1"),  # type: ignore[arg-type]
        session_manager=SimpleNamespace(),  # type: ignore[arg-type]
        api_url=API,
        provider_manager=providers,  # type: ignore[arg-type]
    )

    assert code == 0
    assert asked == [(API, "console")]
    assert printed == expected
