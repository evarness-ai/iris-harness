"""Progress for ``iris setup`` (``$IRIS_HOME/setup.json``).

A sibling of ``runtime/welcome.py``'s ``welcome.json``, not a generalization of it:
its own small file, its own atomic-write pattern, nothing shared. One record per
step, keyed by name, with three outcomes:

- ``"done"`` — ran and succeeded.
- ``"skipped"`` — the owner chose not to run it (declined, or the process wasn't a
  terminal). A deliberate choice, so it is never re-offered on its own.
- ``"failed"`` — attempted and did not succeed (a wrong Telegram token that never
  gets a message, ``start_iris.sh`` exiting non-zero, ``iris email setup`` erroring
  out). Not a choice, so unlike ``"skipped"`` it IS retried the next time ``iris
  setup`` runs, same as a step with no record at all.

Re-running resumes at the first step that is unrecorded or failed; ``--reset``
clears every record so the whole wizard runs again, including declined steps.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from iris_harness.foundation.paths import iris_home

StepStatus = Literal["done", "skipped", "failed"]

# Fixed order (decided): the two mandatory steps first, then three optional ones.
# Services is optional: `iris chat`/`iris email setup`/`iris doctor` all run
# in-process (no HTTP hop); the four servers only matter for the web UI, the REST
# API, or Telegram's poller.
STEP_ORDER: tuple[str, ...] = ("preflight", "home_secret", "services", "telegram", "email")
MANDATORY_STEPS: tuple[str, ...] = ("preflight", "home_secret")
OPTIONAL_STEPS: tuple[str, ...] = ("services", "telegram", "email")


@dataclass(frozen=True)
class StepRecord:
    status: StepStatus
    at: str
    detail: str = ""


@dataclass(frozen=True)
class SetupState:
    steps: dict[str, StepRecord] = field(default_factory=dict)

    def is_recorded(self, name: str) -> bool:
        return name in self.steps

    def needs_run(self, name: str) -> bool:
        """Whether ``iris setup`` should run this step: no record yet, or its last
        attempt failed. A ``"skipped"`` (declined) step does not need a run -- that
        was a choice, not a failure."""
        record = self.steps.get(name)
        return record is None or record.status == "failed"

    def next_step(self) -> str | None:
        """The first step that still needs a run, in ``STEP_ORDER``; ``None`` once
        every step is done or was declined."""
        for name in STEP_ORDER:
            if self.needs_run(name):
                return name
        return None

    def mandatory_done(self) -> bool:
        return all(
            name in self.steps and self.steps[name].status == "done" for name in MANDATORY_STEPS
        )

    def as_dict(self) -> dict[str, object]:
        """The read-only shape the web UI's Setup screen renders (``GET /health/setup``)."""
        return {
            "order": list(STEP_ORDER),
            "mandatory": list(MANDATORY_STEPS),
            "mandatory_done": self.mandatory_done(),
            "next_step": self.next_step(),
            "steps": {
                name: {"status": rec.status, "at": rec.at, "detail": rec.detail}
                for name, rec in self.steps.items()
            },
        }


def setup_marker_path() -> Path:
    """Where setup progress is recorded: ``$IRIS_HOME/setup.json``."""
    return iris_home() / "setup.json"


def load_state() -> SetupState:
    """The recorded progress, or an empty state when nothing has run yet.

    An unreadable or malformed file is treated the same as "nothing recorded" —
    the wizard just starts from the top again, rather than refusing to run.
    """
    path = setup_marker_path()
    if not path.exists():
        return SetupState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return SetupState()
    if not isinstance(raw, dict):
        return SetupState()
    steps: dict[str, StepRecord] = {}
    for name, entry in raw.get("steps", {}).items():
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if status not in ("done", "skipped", "failed"):
            continue
        steps[str(name)] = StepRecord(
            status=status, at=str(entry.get("at", "")), detail=str(entry.get("detail", ""))
        )
    return SetupState(steps=steps)


def _write(state: SetupState) -> None:
    """Atomic write, like ``welcome.json``'s: a reader never sees half a file."""
    path = setup_marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "steps": {
            name: {"status": rec.status, "at": rec.at, "detail": rec.detail}
            for name, rec in state.steps.items()
        }
    }
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def record_step(name: str, status: StepStatus, *, detail: str = "") -> SetupState:
    """Mark ``name`` done/skipped/failed, persist it, and return the updated state."""
    state = load_state()
    steps = dict(state.steps)
    steps[name] = StepRecord(status=status, at=datetime.now(UTC).isoformat(), detail=detail)
    new_state = SetupState(steps=steps)
    _write(new_state)
    return new_state


def clear_state() -> None:
    """``--reset``: drop every recorded step so the wizard runs from the top."""
    setup_marker_path().unlink(missing_ok=True)


__all__ = [
    "MANDATORY_STEPS",
    "OPTIONAL_STEPS",
    "STEP_ORDER",
    "SetupState",
    "StepRecord",
    "StepStatus",
    "clear_state",
    "load_state",
    "record_step",
    "setup_marker_path",
]
