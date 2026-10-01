"""Unit tests for the chat origin-channel fallback (Phase 1)."""

from __future__ import annotations

from iris_harness.server.iris_api.chat_routes import _origin_channel


def test_explicit_non_console_channel_wins() -> None:
    assert _origin_channel("telegram", "default") == "telegram"
    # explicit channel beats a conflicting session prefix
    assert _origin_channel("web", "telegram:42") == "web"


def test_console_default_derives_from_session_prefix() -> None:
    assert _origin_channel("console", "telegram:123456789") == "telegram"
    assert _origin_channel("console", "web:abc123") == "web"


def test_console_default_without_prefix_stays_console() -> None:
    assert _origin_channel("console", "default") == "console"
    assert _origin_channel("console", "dfc18b46bab4") == "console"


def test_empty_session_is_safe() -> None:
    assert _origin_channel("console", "") == "console"
