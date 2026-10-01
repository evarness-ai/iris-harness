"""An IMAP account: where the mailbox is and how to log in, kept in the vault.

Everything needed to connect -- host, port, security, username and the app password --
is one JSON blob in the SDK's secret store (``sdk.vault.get_secret_store()``: the OS
keychain by default, the Fernet vault with ``IRIS_SECRET_BACKEND=vault``), under the
handle ``imap`` / ``<account id>``. Nothing of it is written to a config file, and the
password never reaches a log line or a ``repr``.

The ``email_accounts`` row (``imap:<address>``) is what the sweep iterates; this blob
is what the provider resolves when it runs.
"""

from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from iris_harness.sdk.vault import SecretStore, get_secret_store

IMAP_PROVIDER = "imap"
VAULT_NAMESPACE = "imap"
DEFAULT_FOLDER = "INBOX"

Security = Literal["ssl", "starttls", "plain"]
SECURITIES: tuple[Security, ...] = ("ssl", "starttls", "plain")
DEFAULT_PORTS: dict[Security, int] = {"ssl": 993, "starttls": 143, "plain": 143}


class ImapAccountError(ValueError):
    """An account definition that cannot be used (bad security, plaintext off-host)."""


def account_id_for(address: str) -> str:
    """``imap:<address>``, lowercased: the ``email_accounts`` id for an IMAP mailbox."""
    return f"{IMAP_PROVIDER}:{address.strip().lower()}"


def address_of(account_id: str) -> str:
    return account_id.split(":", 1)[1] if ":" in account_id else account_id


def is_loopback(host: str) -> bool:
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip()).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ImapAccount:
    """One IMAP mailbox. ``password`` is kept out of ``repr`` and out of every log."""

    address: str
    host: str
    username: str
    password: str = field(repr=False)
    port: int = 993
    security: Security = "ssl"
    folder: str = DEFAULT_FOLDER

    def __post_init__(self) -> None:
        if self.security not in SECURITIES:
            raise ImapAccountError(
                f"security must be one of {', '.join(SECURITIES)}: got {self.security!r}"
            )
        if "@" not in self.address:
            raise ImapAccountError(f"address must contain '@': got {self.address!r}")
        if not self.host.strip():
            raise ImapAccountError("host is required")
        if self.security == "plain" and not is_loopback(self.host):
            # A password in clear over the network is never acceptable. Loopback is the
            # one exception: a local bridge (Proton Mail Bridge and the like) listens there.
            raise ImapAccountError(
                f"plaintext IMAP is allowed only to a local bridge (localhost); {self.host} "
                "needs security 'ssl' (port 993) or 'starttls' (port 143)"
            )

    @property
    def account_id(self) -> str:
        return account_id_for(self.address)

    def to_secret(self) -> str:
        return json.dumps(
            {
                "address": self.address,
                "host": self.host,
                "port": self.port,
                "security": self.security,
                "username": self.username,
                "password": self.password,
                "folder": self.folder,
            }
        )

    @classmethod
    def from_secret(cls, blob: str) -> ImapAccount:
        data: dict[str, Any] = json.loads(blob)
        return cls(
            address=str(data["address"]),
            host=str(data["host"]),
            username=str(data["username"]),
            password=str(data["password"]),
            port=int(data.get("port", 993)),
            security=cast(Security, str(data.get("security", "ssl"))),
            folder=str(data.get("folder") or DEFAULT_FOLDER),
        )

    def with_password(self, password: str) -> ImapAccount:
        return replace(self, password=password)


def _store(store: SecretStore | None) -> SecretStore:
    return store if store is not None else get_secret_store()


def save_account(account: ImapAccount, *, store: SecretStore | None = None) -> None:
    """Put the account (password included) in the vault under its handle."""
    _store(store).set(VAULT_NAMESPACE, account.account_id, account.to_secret())


def load_account(account_id: str, *, store: SecretStore | None = None) -> ImapAccount | None:
    """The account behind ``imap:<address>``, or ``None`` when the vault has none (or an
    unreadable blob: a corrupt entry reads as "not connected", never as a crash)."""
    blob = _store(store).get(VAULT_NAMESPACE, account_id_for(address_of(account_id)))
    if not blob:
        return None
    try:
        return ImapAccount.from_secret(blob)
    except (ValueError, KeyError, TypeError):
        return None


def delete_account(account_id: str, *, store: SecretStore | None = None) -> None:
    _store(store).delete(VAULT_NAMESPACE, account_id_for(address_of(account_id)))


__all__ = [
    "DEFAULT_FOLDER",
    "DEFAULT_PORTS",
    "IMAP_PROVIDER",
    "SECURITIES",
    "VAULT_NAMESPACE",
    "ImapAccount",
    "ImapAccountError",
    "Security",
    "account_id_for",
    "address_of",
    "delete_account",
    "is_loopback",
    "load_account",
    "save_account",
]
