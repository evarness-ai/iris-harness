"""__tmpl_title's tests: the provider in a real IRIS, syncing into the owner's record.

``harness`` builds the runtime ``iris`` runs, in a throwaway home, with the network
refused; ``plugin`` mounts this plugin in-process with its own manifest. The record the
provider writes (``default_sync_store``) and the account it connects live in that home.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from iris_harness.testing import harness, plugin
from iris_personal.email.provider_api import MailProvider

from __tmpl_package import plugin as this_plugin

NAME = "__tmpl_name"
MANIFEST = Path(this_plugin.__file__).with_name("manifest.yaml")
ADDRESS = "owner@example.org"
#: The ledger row a write that reached the mailbox leaves (the proof bundle reads it).
WRITE_ROW = "mailbox_write_performed"


def test_it_implements_the_mail_provider_interface() -> None:
    assert isinstance(this_plugin.Provider(), MailProvider)


def test_a_connected_account_syncs_incrementally() -> None:
    provider = this_plugin.Provider()
    with harness(plugins=[plugin(this_plugin.mount(provider), manifest=MANIFEST)]) as h:
        assert h.plugin_loaded(NAME), h.plugins()[NAME]
        account_id = provider.connect(ADDRESS)
        assert account_id == f"{this_plugin.NAME}:{ADDRESS}"

        first = provider.fetch_new(account_id)
        assert first.fetched == len(this_plugin.DEMO_MAILBOX)
        assert first.fell_back_to_cold_start
        # The cursor it saved holds: nothing new the next time.
        again = provider.fetch_new(account_id)
        assert again.fetched == 0 and not again.fell_back_to_cold_start

        message_id = first.new_message_ids[0]
        assert provider.fetch_message_body(account_id, message_id, max_chars=8) == "The seed"


def test_the_owners_address_goes_to_the_identity_guards() -> None:
    provider = this_plugin.Provider()
    with harness(plugins=[plugin(this_plugin.mount(provider), manifest=MANIFEST)]) as h:
        provider.connect(ADDRESS)
        assert provider.owner_identity() == {"email": [ADDRESS]}
        # Declared under `identity: provides` in the manifest, so it was accepted.
        assert h.plugin_loaded(NAME), h.plugins()[NAME]


def test_mailbox_writes_wait_for_the_owners_approval() -> None:
    provider = this_plugin.Provider()
    with harness(plugins=[plugin(this_plugin.mount(provider), manifest=MANIFEST)]) as h:
        account_id = provider.connect(ADDRESS)
        # The owner approves with `iris email writes approve`; until then, every write
        # is refused -- and says how to approve it.
        with pytest.raises(PermissionError, match="iris email writes approve"):
            provider.trash_messages(account_id, ["demo-1"])
        # A refused write reaches nothing, so nothing is recorded.
        assert h.audit_rows(hook_point=WRITE_ROW) == []


def _approve_writes(account_id: str) -> None:
    """What the owner runs once: `iris email writes approve --account <id>`. The harness
    put its home in the environment, so the command records the approval there; the
    `email` profile (what an install with the email slice runs) mounts `iris email`."""
    ran = subprocess.run(  # noqa: S603 - this interpreter, fixed arguments
        [sys.executable, "-m", "iris_harness.main", "email", "writes", "approve"]
        + ["--account", account_id, "--yes"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "IRIS_PROFILE": "email"},
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr


def test_an_approved_write_is_recorded_in_the_governance_ledger() -> None:
    provider = this_plugin.Provider()
    with harness(plugins=[plugin(this_plugin.mount(provider), manifest=MANIFEST)]) as h:
        account_id = provider.connect(ADDRESS)
        _approve_writes(account_id)
        assert provider.trash_messages(account_id, ["demo-1"]) == ["demo-1"]
        # mailbox_write recorded what reached the mailbox: the proof bundle's evidence
        # (`iris governance proof-bundle check`) that the write had an approval.
        [row] = h.audit_rows(hook_point=WRITE_ROW)
        assert row.decision == "allow"
        assert ADDRESS not in row.reason  # the account is never in a reason
