"""``graphiti_import`` registers nothing at runtime: importing is an owner's command.

The work is in ``cli.py`` (``iris graphiti import``), ``reader.py`` (Graphiti records →
memris source records) and ``mappings.yaml`` (which Graphiti relation is which core
property). A plugin with an empty ``setup`` is still a plugin: mounting it in a profile
is what makes the command appear.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iris_harness.sdk import PluginAPI


def setup(api: PluginAPI) -> None:  # nothing to register during a turn
    return None
