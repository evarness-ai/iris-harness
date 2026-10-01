"""The governance audit ledger, for a plugin that makes a governed decision itself.

Most decisions reach the ledger through the kernel's hooks. A plugin that runs a
flow the kernel does not see (an OAuth connect started from the web console) records
its own rows here: ``AuditLog(db_path=audit_db_path()).record(...)`` writes to the
same ledger ``iris audit`` and the Governance screen read. Never record a token, a
code or message content.
"""

from __future__ import annotations

from iris_harness.foundation.paths import audit_db_path
from iris_harness.kernel.governance.audit import AuditLog

__all__ = ["AuditLog", "audit_db_path"]
