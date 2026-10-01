"""Phase 3 audit log — hot SQLite tier.

Every kernel hook firing records exactly one row here (allow,
transform, deny, require_approval — including exceptions). The hot
tier holds the last 30 days; Phase 5 adds the Parquet+zstd cold
archive and DuckDB analytics.
"""

from iris_harness.kernel.governance.audit.log import AuditLog, AuditRow

__all__ = ["AuditLog", "AuditRow"]
