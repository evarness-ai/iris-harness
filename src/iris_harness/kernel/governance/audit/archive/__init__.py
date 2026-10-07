"""Cold-tier audit archive helpers (Phase 5).

This package owns Parquet+zstd writes, SQLite hot-tier compaction,
and DuckDB query/export helpers over hot+cold audit data.
"""

from iris_harness.kernel.governance.audit.archive.compact import AuditCompactor, CompactResult
from iris_harness.kernel.governance.audit.archive.markers import VerifyReport, verify_archive
from iris_harness.kernel.governance.audit.archive.query import AuditQueryEngine, ExportFormat
from iris_harness.kernel.governance.audit.archive.writer import (
    ArchivePartition,
    AuditArchive,
    default_archive_root,
)

__all__ = [
    "default_archive_root",
    "ArchivePartition",
    "AuditArchive",
    "AuditCompactor",
    "CompactResult",
    "AuditQueryEngine",
    "ExportFormat",
    "VerifyReport",
    "verify_archive",
]
