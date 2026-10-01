"""``setup(api)`` for the Gmail plugin — the mailbox provider.

No ``PluginAPI`` registration kind fits "I am how mail gets read", and the six v1
kinds are locked, so this plugin plugs into two keyed core registries instead —
the same shape the pending-action providers (M2.6) and the health check providers
use:

* ``email.providers.register_mail_provider`` — the :class:`GmailProvider` the core's
  sweep heartbeat, the ``read_email`` / ``find_attachment`` tools, finance's statement
  ingest and email's category-discovery bootstrap all reach Gmail through;
* ``api.register_credential_check`` — the Gmail credential row of System Health,
  reported by the plugin that owns the OAuth module;
* ``connections.google.register_provider`` — Gmail in Settings > Connections, so a
  revoked token is reconnected from the console, not only by the Mac login.

Everything that talks to Gmail lives here: the fetcher, the attachment reader, the
OAuth module and ``iris auth gmail``. What stays core is the record every read
queries (``EmailStore``), the reads over it, and the provider-agnostic sweep that
keeps it current by asking whichever provider is registered.
"""

from __future__ import annotations

import logging

from iris_harness.sdk import PluginAPI

logger = logging.getLogger(__name__)


def setup(api: PluginAPI) -> None:
    from iris_personal.email.accounts import register_account_count
    from iris_personal.email.provider_api import register_mail_provider

    from .provider import GmailProvider

    register_mail_provider(GmailProvider())
    # `iris system status` counts connected accounts through a core seam (also filled
    # by this plugin's CLI registration, where no runtime is built).
    register_account_count()
    _register_web_reconnect(api)
    _register_credential_health(api)
    logger.debug("gmail: mail provider, credential health + web reconnect registered")


def _register_web_reconnect(api: PluginAPI) -> None:
    """Gmail in Settings > Connections: the server runs the OAuth flow (a Web client).

    Same scopes as ``iris auth gmail login``, read from the OAuth module; the account
    is checked with the same Gmail profile call that login makes.
    """
    from iris_personal.connections.google import GoogleProvider, register_provider

    from . import gmail_oauth

    register_provider(
        api,
        GoogleProvider(
            key=gmail_oauth.KEYRING_PROVIDER,
            label="Gmail",
            scopes=tuple(gmail_oauth.DEFAULT_SCOPES),
            identity_url="https://gmail.googleapis.com/gmail/v1/users/me/profile",
            identity_field=("emailAddress",),
            order=0,
        ),
    )


def _register_credential_health(api: PluginAPI) -> None:
    from iris_harness.sdk.health import HealthCheck
    from iris_personal.connections.google import google_credential_checks, with_reconnect

    from . import gmail_oauth

    def _check(net_probe: bool) -> list[HealthCheck]:
        # The OAuth functions are looked up per call, so patches land.
        return with_reconnect(
            google_credential_checks(
                [("Gmail", "gmail", gmail_oauth.status, gmail_oauth.load_credentials)],
                net_probe=net_probe,
            ),
            gmail_oauth.KEYRING_PROVIDER,
        )

    api.register_credential_check("gmail_credentials", _check)

    # The health watch's first fix for a red Gmail row: refresh the token now
    # (ADR-0116). A refusal means revoked — the watch then asks the owner to re-login.
    from iris_harness.sdk.health import credential_refresh_repairer, register_repairer

    def _refresh(account: str) -> bool | None:
        return gmail_oauth.force_refresh(account)  # looked up per call, so patches land

    register_repairer("gmail_credentials", credential_refresh_repairer("Gmail", _refresh))
