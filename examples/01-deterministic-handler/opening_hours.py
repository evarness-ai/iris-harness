"""A deterministic handler: answer "when is the library open?" with no model call.

``api.register_intercept`` puts a handler in front of the model. It sees every message
first; returning ``None`` passes the turn on, returning a reply answers it. The reply
comes from ``api.services.deterministic_reply``, so the answer is recorded, screened by
the model-free response check and written to the audit ledger with
``deterministic: true`` and this handler's name -- the same governance a generated
answer gets, minus the model.

The hours are data (``hours.yaml`` beside this file), not code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from iris_harness.sdk import PluginAPI

HANDLER = "opening_hours"
HOURS_FILE = Path(__file__).with_name("hours.yaml")

# The questions this handler claims. Anything else falls through to the model.
_ASKS = re.compile(r"\b(open|opening|hours|close|closing)\b", re.IGNORECASE)
_PLACE = re.compile(r"\blibrary\b", re.IGNORECASE)


def load_hours(path: Path = HOURS_FILE) -> dict[str, str]:
    """``day -> hours`` in the order the file lists them."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(day): str(hours) for day, hours in (raw.get("hours") or {}).items()}


def render(hours: dict[str, str]) -> str:
    lines = [f"- {day}: {when}" for day, when in hours.items()]
    return "The library is open:\n" + "\n".join(lines)


def claims(message: str) -> bool:
    return bool(_ASKS.search(message) and _PLACE.search(message))


def setup(api: PluginAPI) -> None:
    hours = load_hours()
    reply = api.services.deterministic_reply

    def answer(message: str, *, session_id: str, span: Any = None) -> Any:
        if not claims(message):
            return None  # not ours: the next handler, then the model, gets the turn
        return reply(
            message=message,
            session_id=session_id,
            response=render(hours),
            metadata={"handler": HANDLER},
            span=span,
        )

    api.register_intercept(HANDLER, answer, trace_text="opening hours from hours.yaml")
