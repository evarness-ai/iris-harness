"""Whether the model guard is on and can run, read without loading it (issue #136).

The retrieved-content injection guard (``IRIS_GOVERNANCE_PROMPT_GUARD``) is opt-in and fails
open: when its classifier cannot be loaded the hook allows the text and writes a ``guard
unavailable`` row. Nothing showed the owner either that the guard was on but could not run,
or that it was off while tools returning third-party text were mounted. This module answers
the first question for Health and ``GET /governance/state`` from one place, so they cannot
disagree.

The classifier loads lazily and only learns it is unavailable on the first use, so the
answer is read three cheap ways and never by loading it: whether ``transformers`` and
``torch`` are installed (``importlib.util.find_spec``, which imports neither), whether the
weights are in the local Hugging Face cache (a file lookup, no network), and whether a
classifier in this process has already tried and failed (``prompt_guard_load_failed``).
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from iris_harness.foundation.env import env_flag
from iris_harness.kernel.governance.threat.backends import prompt_guard_load_failed

PROMPT_GUARD_FLAG = "IRIS_GOVERNANCE_PROMPT_GUARD"

ClassifierState = Literal["available", "unavailable"]

_INSTALL_FIX = (
    "install the optional ml extra (pip install 'iris-harness[ml]') and download the "
    "classifier weights"
)


@dataclass(frozen=True)
class ModelGuardState:
    """The model guard's posture: ``on`` is the flag; ``classifier`` is only meaningful when on."""

    on: bool
    classifier: ClassifierState
    reason: str | None = None
    fix: str | None = None


def _model_id() -> str | None:
    """The Prompt Guard model id from the threat-detection config, or None when unreadable."""
    try:
        from iris_harness.foundation.paths import config_path
        from iris_harness.kernel.governance.threat.config import ThreatDetectionConfig

        override = os.getenv("IRIS_THREAT_DETECTION_CONFIG")
        path = Path(override) if override else config_path("governance", "threat-detection.yaml")
        return ThreatDetectionConfig.from_yaml(path).backend.prompt_guard.model
    except Exception:  # noqa: BLE001 - an unreadable config is reported as unknown, not raised
        return None


def _weights_cached(model_id: str) -> bool:
    """Whether the model's config is in the local cache; reads the cache directory only."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 - without the hub client nothing can be cached by it
        return False
    try:
        found = try_to_load_from_cache(model_id, "config.json")
    except Exception:  # noqa: BLE001
        return False
    return isinstance(found, str)


def model_guard_state() -> ModelGuardState:
    """The model guard's flag and, when it is on, whether its classifier can run."""
    on = env_flag(PROMPT_GUARD_FLAG, default=False)
    if not on:
        return ModelGuardState(on=False, classifier="unavailable")
    missing = [name for name in ("transformers", "torch") if importlib.util.find_spec(name) is None]
    if missing:
        return ModelGuardState(
            on=True,
            classifier="unavailable",
            reason=f"{' and '.join(missing)} not installed",
            fix=_INSTALL_FIX,
        )
    model_id = _model_id()
    if model_id is not None and not _weights_cached(model_id):
        return ModelGuardState(
            on=True,
            classifier="unavailable",
            reason=f"the weights for {model_id} are not in the local cache",
            fix=_INSTALL_FIX,
        )
    if prompt_guard_load_failed():
        return ModelGuardState(
            on=True,
            classifier="unavailable",
            reason="the classifier was tried and could not load",
            fix=_INSTALL_FIX,
        )
    return ModelGuardState(on=True, classifier="available")


__all__ = ["PROMPT_GUARD_FLAG", "ModelGuardState", "model_guard_state"]
