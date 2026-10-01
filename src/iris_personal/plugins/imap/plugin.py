"""``setup(api)`` for the IMAP plugin -- the app-password mailbox provider.

Same shape as the Gmail plugin: no ``PluginAPI`` kind means "I am how mail gets read",
so it plugs into keyed core registries instead --

* ``email.providers.register_mail_provider`` -- the :class:`ImapProvider` the sweep
  heartbeat, the read/attachment tools, the judge's labels and the trash tools reach an
  ``imap:`` account through;
* ``api.register_credential_check`` -- one System Health row per IMAP account.

Everything that talks to an IMAP server lives here. The record (``EmailStore``), the
reads over it and the provider-agnostic sweep stay in the email library.
"""

from __future__ import annotations

import logging

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.health import HealthCheck

logger = logging.getLogger(__name__)


def setup(api: PluginAPI) -> None:
    from iris_personal.email.accounts import register_account_count
    from iris_personal.email.provider_api import register_mail_provider

    from .health import imap_credential_checks
    from .provider import ImapProvider

    provider = ImapProvider()
    register_mail_provider(provider)
    # `iris system status` counts connected accounts through a core seam (also filled
    # by this plugin's CLI registration, where no runtime is built).
    register_account_count()

    def _check(net_probe: bool) -> list[HealthCheck]:
        return imap_credential_checks(provider, net_probe=net_probe)

    api.register_credential_check("imap_credentials", _check)
    logger.debug("imap: mail provider + credential health registered")
