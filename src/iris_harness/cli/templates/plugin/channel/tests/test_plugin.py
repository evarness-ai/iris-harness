"""__tmpl_title's tests: the connector on its own, then on a real IRIS's gateway.

``harness`` builds the runtime ``iris`` runs, in a throwaway home, with the network
refused; ``plugin`` mounts this plugin in-process with its own manifest. The gateway is
reached the way a plugin reaches it -- ``api.services.channels`` -- so the delivery runs
behind the same fault boundary it does in production.
"""

from __future__ import annotations

import json
from pathlib import Path

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.channels import ChannelMessage, DeliveryStatus, IChannelConnector
from iris_harness.testing import harness, plugin

from __tmpl_package import plugin as this_plugin

NAME = "__tmpl_name"
MANIFEST = Path(this_plugin.__file__).with_name("manifest.yaml")


def test_the_connector_implements_the_channel_protocol(tmp_path: Path) -> None:
    connector = this_plugin.Connector(outbox=tmp_path / "outbox.jsonl")
    assert isinstance(connector, IChannelConnector)
    assert connector.healthy()


def test_the_connector_delivers_to_its_outbox(tmp_path: Path) -> None:
    connector = this_plugin.Connector(outbox=tmp_path / "outbox.jsonl")
    receipt = connector.send(ChannelMessage(recipient="owner", body="Hello", subject="Hi"))
    assert receipt.status is DeliveryStatus.SENT
    assert receipt.channel == this_plugin.CHANNEL
    lines = (tmp_path / "outbox.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"recipient": "owner", "subject": "Hi", "body": "Hello"}


def test_an_empty_message_is_skipped_not_sent(tmp_path: Path) -> None:
    connector = this_plugin.Connector(outbox=tmp_path / "outbox.jsonl")
    receipt = connector.send(ChannelMessage(recipient="owner", body="  "))
    assert receipt.status is DeliveryStatus.SKIPPED
    assert not (tmp_path / "outbox.jsonl").exists()


def test_the_gateway_delivers_through_the_plugin() -> None:
    mounted: list[PluginAPI] = []

    def setup(api: PluginAPI) -> None:
        mounted.append(api)
        this_plugin.setup(api)

    with harness(plugins=[plugin(setup, manifest=MANIFEST)]) as h:
        assert h.plugin_loaded(NAME), h.plugins()[NAME]
        gateway = mounted[0].services.channels
        assert this_plugin.CHANNEL in gateway.channels()

        receipts = gateway.broadcast(
            ChannelMessage(recipient="owner", body="Your brief is ready."),
            channels=[this_plugin.CHANNEL],
        )
        assert [r.status for r in receipts] == [DeliveryStatus.SENT]
        # The connector wrote inside the harness's own home, never the owner's.
        outbox = h.data_dir / f"{this_plugin.CHANNEL}_outbox.jsonl"
        assert json.loads(outbox.read_text(encoding="utf-8"))["body"] == "Your brief is ready."
