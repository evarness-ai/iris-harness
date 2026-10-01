"""The SDK names the public email slice needed (OSS plan R2; public issue #31, steps 1-2).

The email plugins ship in release 1, so they import only ``iris_harness.sdk`` from the
core. The core helpers they reached past the SDK for are published here as facades, the
same shape as the boundary plan's PRs 3a-3c: each name is the core object itself, not a
copy, so switching an import changes nothing at runtime. New modules are pinned exactly
(``sdk.activity``, ``sdk.approvals``, ``sdk.audit``, ``sdk.heartbeat``); names added to older modules are
checked by identity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.foundation import activity, auth, paths, public_url
from iris_harness.kernel.governance import approvals, audit
from iris_harness.kernel.governance.approvals import service as approvals_service
from iris_harness.llm import client as llm_client
from iris_harness.memory.knowledge import event_subscribers, models, wiki_engine
from iris_harness.sdk import activity as sdk_activity
from iris_harness.sdk import approvals as sdk_approvals
from iris_harness.sdk import audit as sdk_audit
from iris_harness.sdk import config as sdk_config
from iris_harness.sdk import health as sdk_health
from iris_harness.sdk import heartbeat as sdk_heartbeat
from iris_harness.sdk import llm as sdk_llm
from iris_harness.sdk import memory as sdk_memory
from iris_harness.sdk import types as sdk_types
from iris_harness.services.heartbeat import config as heartbeat_config
from iris_harness.services.heartbeat import models as heartbeat_models
from iris_harness.services.heartbeat import run_store, schedule_text, scheduler, slots
from iris_harness.services.heartbeat.models import HeartbeatStatus
from iris_harness.services.system import status

NEW_MODULES = [
    (sdk_activity, {"chat_in_progress": activity}),
    (sdk_audit, {"AuditLog": audit, "audit_db_path": paths}),
    (
        sdk_heartbeat,
        {
            "HeartbeatRunHistory": run_store,
            "StoredRun": run_store,
            "heartbeat_runs": run_store,
            "Tally": slots,
            "clock": slots,
            "tally": slots,
            # Email setup tells the owner when the sweep runs (OSS plan R4).
            "HeartbeatConfigError": heartbeat_config,
            "load_heartbeats": heartbeat_config,
            "describe_schedule": schedule_text,
        },
    ),
    # Email setup's governed "may IRIS change your mailbox?" (OSS plan R4, R17).
    (
        sdk_approvals,
        {
            **dict.fromkeys(
                (
                    "ApprovalAlreadyAnsweredError",
                    "ApprovalCard",
                    "ApprovalNotFoundError",
                    "ApprovalQueue",
                    "ApprovalRow",
                    "ApprovalStore",
                ),
                approvals,
            ),
            "respond_to_approval": approvals_service,
        },
    ),
]

ADDED = [
    (
        sdk_config,
        {
            "PUBLIC_URL_ENV": public_url,
            "public_base_url": public_url,
            "iris_home": paths,
            "workspace_dir": paths,
        },
    ),
    (
        sdk_health,
        {"register_account_counter": status, "register_file_root_counter": status},
    ),
    (
        sdk_memory,
        {
            "WIKI_INGEST_REQUESTED": event_subscribers,
            "subscribe_wiki_ingest_consumer": event_subscribers,
            "WikiIngestEvent": models,
            "WikiEngine": wiki_engine,
        },
    ),
    (
        sdk_llm,
        {"JsonReply": llm_client, "LLMBadReply": llm_client, "LLMUnreachable": llm_client},
    ),
    (
        sdk_types,
        {
            "Principal": auth,
            "HeartbeatRun": heartbeat_models,
            "HeartbeatStatus": heartbeat_models,
            "HeartbeatHandler": scheduler,
        },
    ),
]


@pytest.mark.parametrize(
    ("facade", "sources"), NEW_MODULES, ids=lambda v: getattr(v, "__name__", "")
)
def test_each_new_module_is_exactly_its_core_names(
    facade: object, sources: dict[str, object]
) -> None:
    assert sorted(facade.__all__) == sorted(sources)  # type: ignore[attr-defined]
    for name, source in sources.items():
        assert getattr(facade, name) is getattr(source, name), name


@pytest.mark.parametrize(("facade", "sources"), ADDED, ids=lambda v: getattr(v, "__name__", ""))
def test_each_added_name_is_the_core_object(facade: object, sources: dict[str, object]) -> None:
    assert set(sources) <= set(facade.__all__)  # type: ignore[attr-defined]
    for name, source in sources.items():
        assert getattr(facade, name) is getattr(source, name), name


def test_the_run_history_view_has_only_the_reads(tmp_path: Path) -> None:
    """The scheduler is the one writer: the view a plugin gets cannot record or prune."""
    view = sdk_heartbeat.heartbeat_runs(tmp_path)
    for writer in ("record", "note_job", "prune", "ensure_schema", "db_path"):
        assert not hasattr(view, writer), writer
    for reader in ("recent", "last", "last_success", "last_failure", "first_seen"):
        assert callable(getattr(view, reader)), reader


def test_the_run_history_view_never_creates_the_database(tmp_path: Path) -> None:
    view = sdk_heartbeat.heartbeat_runs(tmp_path)
    assert view.recent(name="email_judge") == []
    assert view.last("email_judge") is None
    assert view.first_seen("email_judge") is None
    assert not (tmp_path / run_store.DB_NAME).exists()


def test_the_run_history_view_reads_what_the_scheduler_kept(tmp_path: Path) -> None:
    store = run_store.HeartbeatRunStore(db_path=tmp_path / run_store.DB_NAME)
    store.record(heartbeat_models.HeartbeatRun(name="email_judge", status=HeartbeatStatus.SUCCESS))
    store.record(heartbeat_models.HeartbeatRun(name="email_judge", status=HeartbeatStatus.FAILED))
    view = sdk_heartbeat.heartbeat_runs(tmp_path)
    assert len(view.recent(name="email_judge")) == 2
    last_success = view.last_success("email_judge")
    last_failure = view.last_failure("email_judge")
    assert last_success is not None and last_success.ok
    assert last_failure is not None and not last_failure.ok
