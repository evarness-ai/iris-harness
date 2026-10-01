"""The dues vocabulary: what counts as a question about money the user OWES.

Core, and deliberately so. Two very different callers read it: the finance domain,
which answers dues questions, and the ``research`` plugin's egress guard, which
REFUSES to send such a question to a web search (ADR-0102). The guard ships in the
public tree and has to work with no finance domain mounted, so the vocabulary cannot
live with the domain -- it moved here from ``finance/vocabulary.py`` at M6.1b when
the domains left the core (OSS plan M6, decision 2). The M4.2 ruling that kept it
"core so the research guard keeps working unmounted" is what this preserves.

Only the matchers the guard reads stay here. The dues *filters* -- the spoken window
phrases, the category terms and ``parse_dues_filters`` -- had no core reader left once
``configure_brief`` moved to the planner plugin, so they went to the finance domain
(core/SDK boundary plan, PR 2). The digest settings check that knob through a validator
the finance plugin registers (``services.digest.settings.register_section_knob_validator``).
"""

from __future__ import annotations

import re

# A finance query is about *amounts owed* (vs net worth or spend) when it asks
# about pending payments, bills due, or outstanding balances. Kept distinct from
# the spend regex ("how much did I pay") so the two never collide.
_DUES_QUERY = re.compile(
    r"\b(pending\s+payments?|payments?\s+due|bills?\s+due|due\s+bills?|upcoming\s+bills?|"
    r"amounts?\s+due|due\s+amounts?|minimum\s+payment|what\s+do\s+i\s+owe|"
    r"how\s+much\s+do\s+i\s+owe|"
    r"i\s+owe|owed|payable|outstanding|unpaid|overdue|dues|autopay|installments?)\b",
    re.IGNORECASE,
)


def is_dues_query(query: str) -> bool:
    """True when the user is asking what they owe / what's due, not net worth or spend."""
    return bool(_DUES_QUERY.search(query))


# Transactional / informational / authoring framings are NOT a "what do I owe"
# lookup. Consulted by the core research guard (so "how does term insurance work"
# still reaches the web) and by the plugin's dues intercept.
DUES_EXCLUDE_RE = re.compile(
    r"\b(set\s+up|create|author|schedule|routines?|when\s+i|pay\s+my|make\s+a\s+payment|"
    r"how\s+(?:do(?:es)?|to)\b|what\s+is\s+a?\b|explain|define|best|cheapest|compare\s+plans)\b",
    re.IGNORECASE,
)
