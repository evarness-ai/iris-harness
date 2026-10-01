"""More footer lines for the morning digest, after ``learned yesterday`` (loop-proof D13).

The D13 footer says the loop is alive — which jobs ran yesterday and which did not —
beside D17's ``learned yesterday`` line. Each owner of such a fact registers a
**footer line**: a function that, given the previous local day as ``[start, end)``,
returns one short line (or None to stay silent). The ``learned_yesterday`` tool closes
the digest with its own line and then these, in registration order, one per line.

A line that fails is skipped and logged, never fatal: a footer must not take the digest
down. The registry holds no words; each line's words come from its owner's config.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from iris_harness.foundation.process_state import track_globals
from iris_harness.services.digest.learned import previous_local_day

logger = logging.getLogger(__name__)

#: ``(start, end)`` of the previous local day (tz-aware) → one line, or None.
FooterLine = Callable[[datetime, datetime], "str | None"]

_LINES: dict[str, FooterLine] = {}


def register_footer_line(name: str, line: FooterLine) -> None:
    """Add (or replace) the footer line ``name``. Order of first registration is kept."""
    _LINES[name] = line


def unregister_footer_line(name: str) -> None:
    _LINES.pop(name, None)


def footer_lines(now: datetime | None = None, tz: ZoneInfo | None = None) -> list[str]:
    """Every registered line for the local day before ``now``, blanks dropped."""
    if tz is None:
        from iris_harness.services.digest.settings import iris_timezone

        tz = iris_timezone()
    start, end = previous_local_day(now or datetime.now(UTC), tz)
    lines: list[str] = []
    for name, line in dict(_LINES).items():
        try:
            text = line(start, end)
        except Exception:  # one broken line must not blank the footer
            logger.warning("digest footer line %r failed; skipped", name, exc_info=True)
            continue
        flat = " ".join(str(text or "").split())
        if flat and flat not in lines:
            lines.append(flat)
    return lines


__all__ = ["FooterLine", "footer_lines", "register_footer_line", "unregister_footer_line"]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_LINES")
