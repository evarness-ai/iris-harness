"""Chat surfaces are channel plugins; the gateway and console are not (OSS plan M4.5).

Decision 10 says the web UI ships as a channel plugin with Telegram as a peer.
What that buys, and what these pin:

* the core no longer names a surface — ``_load_channels`` registers console rows
  and nothing else;
* console stays core, because it is the guaranteed fallback for a profile that
  mounts no channel plugin at all;
* ``web`` is a real delivery target now. The UI has always *sent*
  ``channel: "web"``; nothing answered to that name, so a brief addressed to it
  had nowhere to go.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.plugins_builtin.web_channel.connector import WebConnector
from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.services.channels.channel_config import load_channel_config
from iris_harness.services.channels.models import ChannelMessage, DeliveryStatus


class _Gateway:
    def __init__(self) -> None:
        self.registered: list[object] = []

    def register(self, connector: object) -> None:
        self.registered.append(connector)


def _api(
    tmp_path: Path,
    gateway: _Gateway,
    *,
    deliver: object = None,
    name: str = "web_channel",
) -> PluginAPI:
    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name=name, source="builtin", status=PluginStatus.LOADED))
    services = HarnessServices(
        config_dir=tmp_path,
        data_dir=tmp_path,
        tier_router=None,
        agent_executor=None,
        heartbeats=None,
        channels=gateway,
        deterministic_reply=lambda **kw: None,
        deliver_in_chat=deliver,  # type: ignore[arg-type]
    )
    return PluginAPI(plugin=name, services=services, registry=registry)


def _write_channels(tmp_path: Path, body: str) -> None:
    (tmp_path / "channels.yaml").write_text(body, encoding="utf-8")


# ─── The core names no surface ───────────────────────────────────────────────


def test_core_registers_console_only(tmp_path: Path) -> None:
    from iris_harness.runtime.bootstrap import _load_channels

    _write_channels(
        tmp_path,
        "default: telegram\nchannels:\n"
        "  - name: console\n    type: console\n"
        "  - name: telegram\n    type: telegram\n    bot_token: t\n"
        "  - name: web\n    type: web\n",
    )
    gateway, desired = _load_channels(tmp_path)
    assert gateway.channels() == ["console"]
    # The desired default is returned unvalidated — the surfaces have not mounted yet.
    assert desired == "telegram"


def test_console_is_registered_even_with_no_config(tmp_path: Path) -> None:
    """A profile that mounts no channel plugin still has somewhere to deliver."""
    from iris_harness.runtime.bootstrap import _load_channels

    gateway, desired = _load_channels(tmp_path)
    assert gateway.channels() == ["console"]
    assert desired == "console"


# ─── Telegram is a plugin ────────────────────────────────────────────────────


def test_telegram_plugin_registers_from_the_config_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.plugins_builtin.telegram_channel import plugin

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    _write_channels(
        tmp_path,
        "default: telegram\nchannels:\n  - name: telegram\n    type: telegram\n"
        "    bot_token: ${TELEGRAM_BOT_TOKEN}\n    default_chat_id: ${TELEGRAM_CHAT_ID}\n",
    )
    gateway = _Gateway()
    plugin.setup(_api(tmp_path, gateway, name="telegram_channel"))
    assert [c.name for c in gateway.registered] == ["telegram"]  # type: ignore[attr-defined]


def test_telegram_plugin_skips_the_row_when_the_token_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.plugins_builtin.telegram_channel import plugin

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    _write_channels(
        tmp_path,
        "default: telegram\nchannels:\n  - name: telegram\n    type: telegram\n"
        "    bot_token: ${TELEGRAM_BOT_TOKEN}\n",
    )
    gateway = _Gateway()
    plugin.setup(_api(tmp_path, gateway, name="telegram_channel"))
    assert gateway.registered == []


# ─── The web UI is a peer ────────────────────────────────────────────────────


def test_web_connector_delivers_into_the_named_session() -> None:
    delivered: list[tuple[str, str]] = []
    connector = WebConnector(lambda sid, text: delivered.append((sid, text)))

    receipt = connector.send(
        ChannelMessage(recipient="s-1", body="3 bills due", subject="Morning brief")
    )

    assert receipt.status is DeliveryStatus.SENT
    assert delivered == [("s-1", "Morning brief\n\n3 bills due")]


def test_web_connector_falls_back_to_the_shared_session() -> None:
    """A heartbeat addresses the user, not a conversation they happen to have open."""
    delivered: list[tuple[str, str]] = []
    connector = WebConnector(lambda sid, text: delivered.append((sid, text)), default_session="w")

    connector.send(ChannelMessage(recipient="", body="hello"))

    assert delivered == [("w", "hello")]


def test_web_plugin_does_not_mount_without_a_delivery_path(tmp_path: Path) -> None:
    """A connector that swallowed messages would be worse than an absent one."""
    from iris_harness.plugins_builtin.web_channel import plugin

    gateway = _Gateway()
    plugin.setup(_api(tmp_path, gateway, deliver=None))
    assert gateway.registered == []


def test_web_plugin_registers_the_declared_row(tmp_path: Path) -> None:
    from iris_harness.plugins_builtin.web_channel import plugin

    _write_channels(tmp_path, "default: web\nchannels:\n  - name: web\n    type: web\n")
    gateway = _Gateway()
    plugin.setup(_api(tmp_path, gateway, deliver=lambda sid, text: None))
    assert [c.name for c in gateway.registered] == ["web"]  # type: ignore[attr-defined]


# ─── The shared parser ───────────────────────────────────────────────────────


def test_unset_placeholder_resolves_to_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    _write_channels(
        tmp_path,
        "channels:\n  - name: telegram\n    type: telegram\n    bot_token: ${TELEGRAM_BOT_TOKEN}\n",
    )
    (row,) = load_channel_config(tmp_path).of_type("telegram")
    assert row.bot_token == ""


def test_a_broken_file_still_boots(tmp_path: Path) -> None:
    _write_channels(tmp_path, "channels: [oh: no: yes\n")
    config = load_channel_config(tmp_path)
    assert [r.name for r in config.rows] == ["console"]
    assert config.found is False
