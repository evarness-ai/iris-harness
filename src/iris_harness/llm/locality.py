"""Where a model runs: the one fact the egress gate decides by.

The gate governs a model call by its target tier, and ``tier_3`` means the prompt leaves
the owner's machines. That used to be read off the tier's NAME: the local LM Studio model
configured as ``tier3`` in ``llm_tiers.yaml`` governed as cloud, so a personal turn needed
approval for a call that never left the Mac -- and a cloud model configured under a small
tier's name would have governed as local.

What decides it now is where the provider the client dials runs, declared in the
``providers:`` block of ``llm_tiers.yaml`` (``runs: local`` or ``runs: cloud``). The file
is per deployment, as the endpoints are: on the cloud VM ``lmstudio`` is the failover
proxy, which reaches the Mac and never hands an unmarked request to the cloud
(``llm/egress.py``). A provider the file does not declare is cloud: an unknown
destination fails closed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml

from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)

Locality = Literal["local", "cloud"]

_LOCALITIES: frozenset[str] = frozenset({"local", "cloud"})

# (path, mtime) -> the declarations read from it. Clients are built per call in the
# loop, so the file is read once per change, not once per model call.
_CACHE: dict[str, tuple[float, dict[str, Locality]]] = {}


def parse_provider_localities(raw: object) -> dict[str, Locality]:
    """The ``providers:`` block of ``llm_tiers.yaml`` as ``{provider: locality}``.

    A provider whose ``runs`` is missing or not ``local``/``cloud`` is left out, and so
    governs as cloud: a typo can make a local model ask for approval, never make a
    cloud model look local.
    """
    if not isinstance(raw, Mapping):
        return {}
    declared: dict[str, Locality] = {}
    for provider, entry in raw.items():
        runs = entry.get("runs") if isinstance(entry, Mapping) else None
        if isinstance(runs, str) and runs.strip().lower() in _LOCALITIES:
            declared[str(provider).strip().lower()] = (
                "local" if runs.strip().lower() == "local" else "cloud"
            )
        else:
            logger.warning(
                "llm_tiers.yaml: provider %r declares runs=%r; governing it as cloud",
                provider,
                runs,
            )
    return declared


def declared_localities(path: Path | None = None) -> dict[str, Locality]:
    """The providers declared in ``path``; by default, the config dir's ``llm_tiers.yaml``.

    For a client whose config did not come from the tier router (a provider profile, a
    ``/model`` override), and for a router whose own file has no ``providers:`` block:
    by default the config dir's declarations answer, and when that file declares none
    (an owner's config dir from before the block existed) the shipped file's do -- a
    block that is present is authoritative, even for a provider it leaves out. Nothing
    readable anywhere declares nothing, so everything governs as cloud.
    """
    if path is not None:
        return _read_declarations(path)
    from iris_harness.foundation.paths import config_path, packaged_config_dir, repo_root

    declared = _read_declarations(config_path("llm_tiers.yaml"))
    if declared:
        return declared
    for shipped_dir in (packaged_config_dir(), repo_root() / "config"):
        declared = _read_declarations(shipped_dir / "llm_tiers.yaml")
        if declared:
            return declared
    return {}


def _read_declarations(path: Path) -> dict[str, Locality]:
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return dict(cached[1])
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        logger.warning("could not read provider localities from %s", path, exc_info=True)
        return {}
    declared = parse_provider_localities(raw.get("providers") if isinstance(raw, dict) else None)
    _CACHE[key] = (mtime, declared)
    return dict(declared)


def provider_locality(provider: str, declared: Mapping[str, Locality] | None = None) -> Locality:
    """Where ``provider`` runs, per ``declared`` (default: the config dir's file).

    Undeclared is cloud.
    """
    table = declared_localities() if declared is None else declared
    return table.get(provider.strip().lower(), "cloud")


__all__ = [
    "Locality",
    "declared_localities",
    "parse_provider_localities",
    "provider_locality",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_CACHE")
