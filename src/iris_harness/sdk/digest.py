"""Reading the owner's digest settings, and validating a section knob.

The digest (sections, news groups, delivery) is core; a plugin that fills a section
reads what the owner configured through `load_digest_settings` and
`news_group_topics`, and never writes the settings itself. A plugin that adds a
per-section knob registers the values it accepts with
`register_section_knob_validator` (finance registers ``categories``); a knob no
plugin validates is kept as saved. `learned_yesterday_line()` renders the footer's
"learned yesterday" line from every source registered with
`api.register_learned_source`.

A plugin that ages its own dated items out by local days declares each kind under
``expiry:`` in its manifest and reads the owner's value with `expiry_days(key)`
(``digest.yaml`` ``expiry:``, else the declared default; ``KeyError`` for a kind no
installed plugin declares).
"""

from __future__ import annotations

from iris_harness.services.digest.expiry import expiry_days
from iris_harness.services.digest.learned import learned_yesterday_line
from iris_harness.services.digest.settings import (
    DigestSettings,
    load_digest_settings,
    news_group_topics,
    register_section_knob_validator,
)

__all__ = [
    "DigestSettings",
    "expiry_days",
    "learned_yesterday_line",
    "load_digest_settings",
    "news_group_topics",
    "register_section_knob_validator",
]
