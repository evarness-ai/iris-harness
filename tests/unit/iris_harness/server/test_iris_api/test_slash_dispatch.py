"""Tests for web slash-command execution and the per-session model override.

The reported bug: typing ``/model qwen3.5:latest`` in web chat returned
"unsupported slash command for web execution". The composer suggested all 22
visible commands while the dispatcher ran 6, so it offered commands it could not
execute. These pin both halves to one table, and pin that a model chosen this way
actually reaches the next chat turn.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iris_harness.cli.web_commands import ModelOverrides, web_supported_names
from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api.main import create_app


class _StubRuntime:
    """Records what the chat routes resolved, so the override can be asserted."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, message: str, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            response="ok",
            intent="general",
            agent_type="general",
            sources=[],
            has_errors=False,
            error_summary="",
            metadata={},
        )


@pytest.fixture
def client() -> TestClient:
    app = create_app(runtime=_StubRuntime())
    return TestClient(app)


def _dispatch(client: TestClient, command: str, session_id: str = "web-1") -> Any:
    return client.post(
        "/api/slash-dispatch",
        json={"command": command, "session_id": session_id},
        headers=auth_headers(),
    )


# ── the reported bug ──────────────────────────────────────────────────────────


def test_model_command_is_accepted(client: TestClient) -> None:
    """The exact command from the report, which used to be a 400."""
    res = _dispatch(client, "/model qwen3.5:latest")

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["executed"] is True
    assert body["model"] == "qwen3.5:latest"
    assert "qwen3.5:latest" in body["output"]


def test_the_chosen_model_reaches_the_next_chat_turn(client: TestClient) -> None:
    """A confirmation the next message forgets would be theatre, not a fix."""
    _dispatch(client, "/model qwen3.5:latest", session_id="web-7")

    res = client.post(
        "/chat", headers=auth_headers(), json={"message": "hi", "session_id": "web-7"}
    )
    assert res.status_code == 200, res.text

    runtime = client.app.state.runtime
    assert runtime.calls[-1]["preferred_model"] == "qwen3.5:latest"


def test_an_explicit_model_in_the_request_still_wins(client: TestClient) -> None:
    """The override fills a gap; it does not overrule a caller that stated one."""
    _dispatch(client, "/model qwen3.5:latest", session_id="web-8")

    client.post(
        "/chat",
        headers=auth_headers(),
        json={"message": "hi", "session_id": "web-8", "preferred_model": "granite4:latest"},
    )

    runtime = client.app.state.runtime
    assert runtime.calls[-1]["preferred_model"] == "granite4:latest"


def test_the_override_is_per_session(client: TestClient) -> None:
    _dispatch(client, "/model qwen3.5:latest", session_id="web-a")

    client.post("/chat", headers=auth_headers(), json={"message": "hi", "session_id": "web-b"})

    runtime = client.app.state.runtime
    assert runtime.calls[-1]["preferred_model"] is None


def test_model_reset_clears_the_override(client: TestClient) -> None:
    _dispatch(client, "/model qwen3.5:latest", session_id="web-9")
    _dispatch(client, "/model reset", session_id="web-9")

    client.post("/chat", headers=auth_headers(), json={"message": "hi", "session_id": "web-9"})

    runtime = client.app.state.runtime
    assert runtime.calls[-1]["preferred_model"] is None


def test_router_override_reaches_the_chat_turn(client: TestClient) -> None:
    _dispatch(client, "/router granite4:latest", session_id="web-r")

    client.post("/chat", headers=auth_headers(), json={"message": "hi", "session_id": "web-r"})

    runtime = client.app.state.runtime
    assert runtime.calls[-1]["router_model"] == "granite4:latest"


# ── suggestions and execution must agree ──────────────────────────────────────


def test_every_suggested_command_can_be_dispatched(client: TestClient) -> None:
    """The root cause: the composer offered commands the dispatcher refused."""
    suggested = client.get("/api/slash-commands", headers=auth_headers()).json()["commands"]
    api_handled = {"/help", "/provider", "/info", "/sessions", "/clear", "/reset"}
    executable = api_handled | web_supported_names()

    # Entries may be "<name> <completion>"; the command is the first token.
    offered = {str(c["name"]).split()[0] for c in suggested}
    assert offered <= executable, f"suggested but not executable: {offered - executable}"


def test_commands_that_cannot_run_here_are_not_suggested(client: TestClient) -> None:
    offered = {
        str(c["name"]).split()[0]
        for c in client.get("/api/slash-commands", headers=auth_headers()).json()["commands"]
    }

    # Terminal lifecycle, a write to the server's disk, and a context mutation.
    assert "/exit" not in offered
    assert "/export" not in offered
    assert "/compact" not in offered


# ── read-only enforcement ─────────────────────────────────────────────────────


def test_a_write_subcommand_is_refused(client: TestClient) -> None:
    """`/reminders list` is a query; `/reminders add` writes. Only one is allowed."""
    res = _dispatch(client, "/reminders add 2026-01-01 09:00 pay rent")

    assert res.status_code == 422
    assert "read-only" in res.json()["detail"]


def test_an_unsupported_command_still_explains_itself(client: TestClient) -> None:
    res = _dispatch(client, "/export dump.txt")

    assert res.status_code == 400
    assert "unsupported slash command" in res.json()["detail"]


# ── the override store ────────────────────────────────────────────────────────


def test_overrides_are_bounded() -> None:
    """A long-lived server must not accumulate an entry per session it ever saw."""
    overrides = ModelOverrides(max_sessions=3)
    for n in range(10):
        overrides.set(f"s{n}", model="m")

    assert len(overrides._models) <= 3


def test_setting_an_empty_model_clears_rather_than_stores() -> None:
    overrides = ModelOverrides()
    overrides.set("s1", model="m")
    overrides.set("s1", model="")

    assert overrides.get("s1") == ("", "")


# ── rendering ─────────────────────────────────────────────────────────────────


def test_table_output_is_fenced_so_markdown_keeps_its_columns(client: TestClient) -> None:
    """Web chat renders output as markdown, which would collapse rich's alignment."""
    res = _dispatch(client, "/skills")

    assert res.status_code == 200, res.text
    output = res.json()["output"]
    assert output.startswith("```text\n")
    assert output.endswith("\n```")


def test_a_one_line_confirmation_is_not_fenced(client: TestClient) -> None:
    """Nothing to align, and a code block around one line reads worse."""
    output = _dispatch(client, "/model qwen3.5:latest").json()["output"]

    assert "```" not in output
    assert "qwen3.5:latest" in output


def test_a_command_leaves_the_shared_console_width_unpinned(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capturing widens the shared console for the one command, then puts its own
    setting back. Writing back the width it had *measured* pinned it (80 off a
    terminal), so every later print in the process ignored COLUMNS and the terminal."""
    from iris_harness.foundation.console import console

    monkeypatch.setattr(console, "_width", None)  # unpinned, as a fresh process has it
    _dispatch(client, "/model qwen3.5:latest")

    assert console._width is None
