"""Host pressure: one sample of what this machine has left (RAM, CPU, thermal).

Lived in ``llm/arbiter.py`` until M6.2, which is where its first consumer was -- the
arbiter flips tiers when the host is under pressure. But nothing about a psutil read or
a ``pmset`` call is LLM logic, and two other readers want the same sample: the session
log stamps it on a trace, and ``iris system status`` prints it. So it sits in
foundation, where all three import it downward (OSS plan M6, decision 6).
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime

import psutil

logger = logging.getLogger(__name__)

_PMSET_SPEED_RE = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")


@dataclass(frozen=True)
class PressureSnapshot:
    """One sample of host pressure used to drive mode transitions."""

    ram_free_gb: float
    cpu_percent: float
    cpu_speed_limit: int  # 100 = unthrottled (macOS); always 100 elsewhere
    thermal_throttled: bool  # cpu_speed_limit < 100
    sampled_at: datetime


def _read_pmset_thermal() -> int:
    """Return ``CPU_Speed_Limit`` from ``pmset -g therm`` (100 = unthrottled).

    macOS-only signal. Returns 100 (no throttle) on other platforms or when
    pmset is missing/errors. Short timeout — pmset can hang briefly under
    extreme pressure and we shouldn't compound it.
    """
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],  # noqa: S607 — fixed argv, no shell, no user input
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 100
    match = _PMSET_SPEED_RE.search(result.stdout)
    return int(match.group(1)) if match else 100


def sample_pressure() -> PressureSnapshot:
    """Capture one PressureSnapshot using psutil + pmset."""
    vm = psutil.virtual_memory()
    speed_limit = _read_pmset_thermal()
    return PressureSnapshot(
        ram_free_gb=vm.available / (1024**3),
        cpu_percent=psutil.cpu_percent(interval=None),
        cpu_speed_limit=speed_limit,
        thermal_throttled=speed_limit < 100,
        sampled_at=datetime.now(UTC),
    )
