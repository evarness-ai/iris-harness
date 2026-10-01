"""``setup(api)`` for the research reference plugin (OSS plan M4.7, decision 9).

Three registrations and no runtime-internal imports — which is the point: research
is one of the three reference plugins release gate 2 names, so it is the worked
example of "a capability the harness ships, mounted the way a third party would
mount their own".

* **search providers** — the five built-in backends join the search-provider chain
  through ``api.register_search_provider``, the seam any plugin's provider uses
  (``config/search_providers.yaml`` orders it).
* **tool** — the guarded ``research`` tool on the governed ReAct loop.
* **intent handler** — the ``research`` persona (FMX9), mounted on the harness's
  OWN loop via ``HarnessServices.react_handler`` rather than a handler of ours.
  That is what the persona seam is for: intent biases tier and prompt, and the
  tools surface from the research skill packs.

The persona stays opt-in (``IRIS_RESEARCH_AGENT``) exactly as it was in the core:
with it off, ``search`` keeps its historical routing.
"""

from __future__ import annotations

import logging
import os

from iris_harness.sdk import PluginAPI

from . import tools
from .providers import register_builtin_providers

logger = logging.getLogger(__name__)


def _research_agent_enabled() -> bool:
    """FMX9: register the ``research`` persona and remap the ``search`` intent to it
    (see ``iris_harness.agent.intent_router.agent_for_intent``). Opt-in, default off."""
    return os.getenv("IRIS_RESEARCH_AGENT", "").strip().lower() in {"1", "true", "yes", "on"}


def setup(api: PluginAPI) -> None:
    register_builtin_providers(api)
    tools.register(api)

    if not _research_agent_enabled():
        return
    react = api.services.react_handler
    if react is None:
        # The governed loop is off (IRIS_AGENTIC_CORE not on/shadow). A persona has
        # nothing to run on, and the core applied the same guard to its own.
        logger.info("research persona not mounted: the governed loop is off")
        return
    api.register_intent_handler("research", react, stream_handler=api.services.react_stream_handler)
    logger.info("research persona mounted on the governed loop (FMX9)")
