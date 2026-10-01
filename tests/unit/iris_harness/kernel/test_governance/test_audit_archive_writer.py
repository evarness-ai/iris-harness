from __future__ import annotations

import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

pyarrow = pytest.importorskip("pyarrow")
pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

from iris_harness.kernel.governance.audit.archive.writer import (  # noqa: E402
    ArchivePartition,
    AuditArchive,
    audit_archive_schema,
)
from iris_harness.kernel.governance.audit.log import AuditRow  # noqa: E402


@pytest.fixture()
def sample_rows() -> list[AuditRow]:
    return [
        AuditRow(
            id=1,
            ts="2026-01-15T10:00:00+00:00",
            run_id="r1",
            step_id=1,
            agent_type="chat",
            hook_point="pre_llm_call",
            plugin="egress_gate",
            decision="allow",
            classification="public",
            tier="tier_2",
            cost_usd=0.01,
            severity="info",
            reason="ok",
            payload_json='{"k":1}',
        ),
        AuditRow(
            id=2,
            ts="2026-01-20T10:00:00+00:00",
            run_id="r2",
            step_id=None,
            agent_type="chat",
            hook_point="post_step",
            plugin="step_cap",
            decision="warn",
            classification=None,
            tier="tier_1",
            cost_usd=None,
            severity="warn",
            reason="close to cap",
            payload_json="{}",
        ),
        AuditRow(
            id=3,
            ts="2026-02-01T10:00:00+00:00",
            run_id="r3",
            step_id=9,
            agent_type="coding",
            hook_point="pre_tool_use",
            plugin="command_sandbox",
            decision="deny",
            classification="internal",
            tier="tier_2",
            cost_usd=1.23,
            severity="critical",
            reason="denied",
            payload_json='{"x":"y"}',
        ),
    ]


def test_write_partitions_by_year_month(tmp_path: Path, sample_rows: list[AuditRow]) -> None:
    archive = AuditArchive(root=tmp_path)

    counts = archive.write(sample_rows)

    assert counts[ArchivePartition(year=2026, month=1)] == 2
    assert counts[ArchivePartition(year=2026, month=2)] == 1
    jan_files = list((tmp_path / "year=2026" / "month=01").glob("audit-*.parquet"))
    feb_files = list((tmp_path / "year=2026" / "month=02").glob("audit-*.parquet"))
    assert len(jan_files) == 1
    assert len(feb_files) == 1


def test_round_trip_preserves_all_columns(tmp_path: Path) -> None:
    rows = [
        AuditRow(
            id=i,
            ts=f"2026-01-01T00:00:{i % 60:02d}+00:00",
            run_id=f"run-{i}",
            step_id=(i if i % 3 else None),
            agent_type="chat",
            hook_point="pre_llm_call",
            plugin="p",
            decision="allow",
            classification=("public" if i % 2 else None),
            tier=("tier_2" if i % 5 else None),
            cost_usd=(float(i) if i % 4 else None),
            severity="info",
            reason="ok",
            payload_json=f'{{"i": {i}}}',
        )
        for i in range(1, 101)
    ]
    archive = AuditArchive(root=tmp_path)
    archive.write(rows)

    table = pyarrow_parquet.read_table(tmp_path / "year=2026" / "month=01")
    as_rows = table.to_pylist()

    assert len(as_rows) == 100
    first = as_rows[0]
    assert set(first.keys()) == set(audit_archive_schema().names)
    assert any(row["step_id"] is None for row in as_rows)
    assert any(row["cost_usd"] is None for row in as_rows)
    assert any(row["classification"] is None for row in as_rows)


def test_write_uses_zstd_compression(tmp_path: Path, sample_rows: list[AuditRow]) -> None:
    archive = AuditArchive(root=tmp_path)
    archive.write(sample_rows)

    chunk = next((tmp_path / "year=2026" / "month=01").glob("audit-*.parquet"))
    metadata = pyarrow_parquet.ParquetFile(chunk).metadata
    assert metadata is not None
    assert metadata.row_group(0).column(0).compression.upper() == "ZSTD"


def test_partition_dir_permissions_are_restricted(
    tmp_path: Path, sample_rows: list[AuditRow]
) -> None:
    archive = AuditArchive(root=tmp_path)
    archive.write(sample_rows)

    part_dir = tmp_path / "year=2026" / "month=01"
    mode = stat.S_IMODE(part_dir.stat().st_mode)
    assert mode == 0o700


def test_second_write_creates_new_chunk_file(tmp_path: Path, sample_rows: list[AuditRow]) -> None:
    archive = AuditArchive(root=tmp_path)
    archive.write(sample_rows[:1])
    archive.write(sample_rows[:1])

    part_dir = tmp_path / "year=2026" / "month=01"
    chunks = list(part_dir.glob("audit-*.parquet"))
    assert len(chunks) == 2
    table = pyarrow_parquet.read_table(part_dir)
    assert table.num_rows == 2


def test_schema_matches_design_order_and_dictionary_columns() -> None:
    schema = audit_archive_schema()
    assert schema.names == [
        "ts",
        "run_id",
        "step_id",
        "agent_type",
        "hook_point",
        "plugin",
        "decision",
        "classification",
        "tier",
        "cost_usd",
        "severity",
        "reason",
        "payload_json",
    ]
    for field_name in (
        "agent_type",
        "hook_point",
        "plugin",
        "decision",
        "classification",
        "tier",
        "severity",
    ):
        assert pyarrow.types.is_dictionary(schema.field(field_name).type)


def test_write_empty_rows_is_noop(tmp_path: Path) -> None:
    archive = AuditArchive(root=tmp_path)
    counts = archive.write([])
    assert counts == {}
    assert list(tmp_path.glob("**/*.parquet")) == []


def test_partition_for_uses_utc_month_bucket(tmp_path: Path) -> None:
    archive = AuditArchive(root=tmp_path)
    part = archive.partition_for(datetime(2026, 3, 1, 1, tzinfo=UTC))
    assert part == ArchivePartition(year=2026, month=3)
