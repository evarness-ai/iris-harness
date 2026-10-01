"""Governance data/config path resolution + test isolation (no Path.home() leak)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.foundation.paths import governance_config_dir, governance_data_dir


def test_iris_home_relocates_data_and_config(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("IRIS_HOME", "/tmp/iris-home-abc")
    assert governance_data_dir() == Path("/tmp/iris-home-abc/governance")
    assert governance_config_dir() == Path("/tmp/iris-home-abc/governance")


def test_production_paths_when_iris_home_unset(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("IRIS_HOME", raising=False)
    assert governance_data_dir() == Path.home() / ".local" / "share" / "iris"
    assert governance_config_dir() == Path.home() / ".config" / "iris"


def test_store_defaults_resolved_under_test_home() -> None:
    # conftest sets IRIS_HOME to a throwaway temp dir before any import, so each
    # store's default points under it -- never the real profile.
    import os

    from iris_harness.kernel.governance.approvals.store import default_approvals_db_path
    from iris_harness.kernel.governance.audit.log import AuditLog
    from iris_harness.kernel.governance.cost.store import default_cost_ledger_db_path
    from iris_harness.kernel.governance.side_effects.store import default_ledger_db_path
    from iris_harness.memory.state.store import default_checkpoint_db_path

    home = os.environ["IRIS_HOME"]  # set by conftest
    gov = Path(home) / "governance"
    assert default_checkpoint_db_path() == gov / "checkpoints.db"
    assert default_cost_ledger_db_path() == gov / "cost-ledger.db"
    assert default_approvals_db_path() == gov / "approvals.db"
    assert default_ledger_db_path() == gov / "side_effects.db"
    assert AuditLog().db_path == gov / "audit.db"


def test_store_defaults_follow_a_home_moved_after_import(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Resolved on every use, never frozen at import: a process that relocates IRIS_HOME
    # after importing IRIS (iris_harness.testing's harness) writes into the new home.
    from iris_harness.foundation.observability.session_log import session_log_dir
    from iris_harness.kernel.governance.approvals.store import default_approvals_db_path
    from iris_harness.kernel.governance.audit.log import AuditLog
    from iris_harness.memory.identity.loader import user_md_path
    from iris_harness.memory.state.store import default_checkpoint_db_path

    monkeypatch.setenv("IRIS_HOME", str(tmp_path))
    monkeypatch.delenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", raising=False)
    monkeypatch.delenv("IRIS_SESSION_LOG_DIR", raising=False)
    gov = tmp_path / "governance"
    assert default_approvals_db_path() == gov / "approvals.db"
    assert default_checkpoint_db_path() == gov / "checkpoints.db"
    assert AuditLog().db_path == gov / "audit.db"
    assert user_md_path() == tmp_path / "workspace" / "USER.md"
    assert session_log_dir() == tmp_path / "logs"
