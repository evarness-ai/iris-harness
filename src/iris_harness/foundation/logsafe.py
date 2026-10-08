"""Make an untrusted value safe to put in a log line.

A request-supplied string (a Host header, a fact key, a slash-command name, a trace id)
that reaches a log call can carry a newline and so forge a second log entry. Pass such a
value through :func:`log_safe` at the call site; the value is still readable in the log.
"""

from __future__ import annotations

import re

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
_DEFAULT_LIMIT = 200


def log_safe(value: object, limit: int = _DEFAULT_LIMIT) -> str:
    """``value`` as one log-line-safe string: CR/LF and other control characters escaped,
    truncated to ``limit`` characters."""
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    text = _CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", text)
    return text if len(text) <= limit else text[:limit] + "..."


__all__ = ["log_safe"]
