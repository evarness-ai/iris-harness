"""The owner's identity, as far as a plugin has any business with it.

Two jobs, neither of them reading the user's profile -- a plugin has no business doing
that:

- **Redacting before egress.** The research plugin's egress guard must know which
  literals in SOUL.md and USER.md are the user's, so it can refuse to send them to a
  search provider. A plugin that reaches the network needs the same list, and a plugin
  that does not should not import these. The dues vocabulary travels with them for the
  same reason: "is this question about the user's own bills" is the difference between
  a safe web search and an egress of the user's financial situation.
- **Supplying identity** (ADR-0125). A plugin that knows an address of the owner's --
  the account it signs in to -- hands it to the guards with
  ``api.register_owner_identity_source(provider)``: ``provider()`` returns
  ``{kind: literals}`` in the kinds its manifest declares under ``identity: provides``
  (``OWNER_PII_KINDS``). The types below are for annotating that provider.
- **Matching a name** with the harness's name primitive (``name_pattern``: whole words,
  case-insensitive), the one the owner-identity matchers build on.
"""

from __future__ import annotations

from iris_harness.foundation.data.dues_vocabulary import DUES_EXCLUDE_RE, is_dues_query
from iris_harness.kernel.governance.identity_redaction import Fingerprint, OwnerIdentitySource
from iris_harness.kernel.governance.owner_identity import OWNER_PII_KINDS, IdentityKind
from iris_harness.kernel.governance.owner_matchers import name_pattern
from iris_harness.memory.identity.loader import load_soul, load_user_md

__all__ = [
    "DUES_EXCLUDE_RE",
    "OWNER_PII_KINDS",
    "Fingerprint",
    "IdentityKind",
    "OwnerIdentitySource",
    "is_dues_query",
    "load_soul",
    "load_user_md",
    "name_pattern",
]
