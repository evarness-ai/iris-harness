"""The address the web console is reached at, from ``IRIS_PUBLIC_URL``.

A link that leaves the console (a Telegram button, a line in an alert, an OAuth
redirect) must be absolute, and only the operator knows the host: on the cloud
harness it is the tailnet name ``set_server_env.sh --public-url`` writes. One reader,
so the digest's buttons, the health alerts and the Google reconnect agree on it.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

PUBLIC_URL_ENV = "IRIS_PUBLIC_URL"


def public_base_url() -> str:
    """``IRIS_PUBLIC_URL`` without its trailing slash, or ``""`` when unset or not a URL."""
    raw = (os.environ.get(PUBLIC_URL_ENV) or "").strip().rstrip("/")
    if not raw:
        return ""
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        logger.warning(
            "%s is not an http(s) URL; links to the console are left out", PUBLIC_URL_ENV
        )
        return ""
    return raw


def absolute_console_url(path: str) -> str:
    """``path`` on the console's public address, or ``path`` itself without one."""
    base = public_base_url()
    return f"{base}{path}" if base and path.startswith("/") else path


__all__ = ["PUBLIC_URL_ENV", "absolute_console_url", "public_base_url"]
