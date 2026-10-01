"""One parser for ``config/channels.yaml`` (OSS plan M4.5).

The harness registers the console sink from this file and reads the default
channel name out of it; the channel plugins read their own rows out of the same
file. Both go through here, so a surface is configured one way — the reason the
calendar and finance CLIs got a public ``calendar_service`` / ``finance_store``
helper when their commands split.

``${VAR}`` placeholders resolve from the environment. A row whose placeholder is
unset resolves to ``""``, which is how "the token is not configured" reaches the
caller — the caller decides whether that means skip (Telegram) or ignore.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL = "console"
_ENV_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")


@dataclass(frozen=True)
class ChannelRow:
    """One ``channels:`` entry, with its ``${VAR}`` placeholders resolved."""

    name: str
    type: str
    description: str = ""
    params: dict[str, str] = field(default_factory=dict)

    @property
    def bot_token(self) -> str:
        return self.params.get("bot_token", "")

    @property
    def default_chat_id(self) -> str:
        return self.params.get("default_chat_id", "")


@dataclass(frozen=True)
class ChannelConfig:
    """The whole file: the declared rows plus the operator's default channel."""

    rows: tuple[ChannelRow, ...] = ()
    default: str = DEFAULT_CHANNEL
    found: bool = True

    def of_type(self, channel_type: str) -> tuple[ChannelRow, ...]:
        return tuple(r for r in self.rows if r.type == channel_type)


def _resolve(value: str) -> str:
    """Replace ``${VAR}`` with its environment value; ``""`` if any var is unset."""
    result = value
    for match in _ENV_PLACEHOLDER.finditer(value):
        env_val = os.environ.get(match.group(1), "")
        if not env_val:
            return ""
        result = result.replace(match.group(0), env_val)
    return result


def load_channel_config(config_dir: Path) -> ChannelConfig:
    """Parse ``<config_dir>/channels.yaml``. A missing or unreadable file is not
    an error — it means "console only", which is what a bare config dir should do."""
    path = config_dir / "channels.yaml"
    if not path.exists():
        logger.warning("channels.yaml not found at %s, console only", path)
        return ChannelConfig(rows=(ChannelRow(name="console", type="console"),), found=False)
    try:
        raw: Any = yaml.safe_load(path.read_text()) or {}
    except Exception:  # a broken file must not stop the runtime booting
        logger.exception("channels.yaml at %s is unreadable; console only", path)
        return ChannelConfig(rows=(ChannelRow(name="console", type="console"),), found=False)

    rows: list[ChannelRow] = []
    for entry in raw.get("channels", []) or []:
        name = str(entry.get("name") or "")
        if not name:
            logger.warning("channels.yaml: a row has no name, skipped")
            continue
        params = {
            key: _resolve(str(value))
            for key, value in entry.items()
            if key not in {"name", "type", "description"}
        }
        rows.append(
            ChannelRow(
                name=name,
                type=str(entry.get("type") or name),
                description=str(entry.get("description") or ""),
                params=params,
            )
        )
    return ChannelConfig(rows=tuple(rows), default=str(raw.get("default") or DEFAULT_CHANNEL))


def telegram_rows(config_dir: Path) -> tuple[ChannelRow, ...]:
    """Every ``type: telegram`` row — what the telegram_channel plugin registers."""
    return load_channel_config(config_dir).of_type("telegram")
