"""The SDK modules a stateful plugin needs (core/SDK boundary plan, PRs 3a-3c).

`sdk.persistence`, `sdk.time`, `sdk.events`, `sdk.digest`, `sdk.tasks`,
`sdk.reminders`, `sdk.pending_actions`, `sdk.vault` and `sdk.rag` are facades, and
3c added names to `sdk.llm`, `sdk.health` and `sdk.cli`: each name
is the core object itself, not a copy, so a plugin that switches its import changes
nothing at runtime. The tests pin that, pin each `__all__` (a name added there is a
promise to keep it stable), and pin the two behaviours plugins lean on most.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from iris_harness.cli import render
from iris_harness.foundation import clock, console, eventbus, persistence
from iris_harness.foundation.persistence import embedding
from iris_harness.foundation.persistence import sqlite as persistence_sqlite
from iris_harness.kernel.governance.vault import credentials, secret_store
from iris_harness.kernel.governance.vault import keys as vault_keys
from iris_harness.llm import embeddings, narrate, tier_router
from iris_harness.sdk import cli as sdk_cli
from iris_harness.sdk import digest as sdk_digest
from iris_harness.sdk import events as sdk_events
from iris_harness.sdk import health as sdk_health
from iris_harness.sdk import llm as sdk_llm
from iris_harness.sdk import pending_actions as sdk_pending_actions
from iris_harness.sdk import persistence as sdk_persistence
from iris_harness.sdk import rag as sdk_rag
from iris_harness.sdk import reminders as sdk_reminders
from iris_harness.sdk import tasks as sdk_tasks
from iris_harness.sdk import time as sdk_time
from iris_harness.sdk import vault as sdk_vault
from iris_harness.services.digest import expiry, learned, settings
from iris_harness.services.health import models as health_models
from iris_harness.services.health import service as health_service
from iris_harness.services.notifications import events as reminder_events
from iris_harness.services.notifications import models as reminder_models
from iris_harness.services.notifications import snooze
from iris_harness.services.notifications import store as reminder_store
from iris_harness.services.rag import documents as rag_documents
from iris_harness.services.rag import index as rag_index
from iris_harness.services.rag import ingest_gate, ingest_source, retrieve
from iris_harness.services.rag import models as rag_models
from iris_harness.services.rag import store as rag_store
from iris_harness.services.system import doctor
from iris_harness.services.tasks import brief_view, pending_actions
from iris_harness.services.tasks import events as task_events
from iris_harness.services.tasks import models as task_models
from iris_harness.services.tasks import store as task_store

FACADES = [
    (
        sdk_persistence,
        {
            "collection_kwargs": embedding,
            "connect": persistence,
            "data_dir": persistence,
            "data_path": persistence,
            "ensure_columns": persistence_sqlite,
            "sqlite_conn": persistence,
            "with_locked_retry": persistence,
        },
    ),
    (
        sdk_time,
        {
            "iris_timezone": settings,
            "local_now": clock,
            "local_today": clock,
            "previous_local_day": learned,
        },
    ),
    (
        sdk_events,
        {"EventBus": eventbus, "EventHandler": eventbus, "get_default_bus": eventbus},
    ),
    (
        sdk_digest,
        {
            "DigestSettings": settings,
            "expiry_days": expiry,
            "learned_yesterday_line": learned,
            "load_digest_settings": settings,
            "news_group_topics": settings,
            "register_section_knob_validator": settings,
        },
    ),
    (
        sdk_tasks,
        {
            **dict.fromkeys(
                (
                    "ActionCard",
                    "ActionChoice",
                    "ActionEvidence",
                    "ActionFact",
                    "ActionOptionValue",
                    "ActionOptions",
                    "SourceKind",
                    "Task",
                    "TaskAction",
                    "WaitFor",
                ),
                task_models,
            ),
            "TASK_COMPLETED": task_events,
            "TaskStore": task_store,
            "short_title": brief_view,
        },
    ),
    (
        sdk_reminders,
        {
            "NOT_YET": reminder_store,
            "REMINDER_COMPLETED": reminder_events,
            "REMINDER_SNOOZED": reminder_events,
            "SNOOZE_CHOICES": snooze,
            "TERMINAL_STATUSES": reminder_models,
            "Reminder": reminder_models,
            "ReminderStore": reminder_store,
            "parse_snooze": snooze,
        },
    ),
    (
        sdk_pending_actions,
        dict.fromkeys(
            (
                "ChoiceActionProvider",
                "DesiredAction",
                "PendingActionProvider",
                "PendingActionsSummary",
                "provider_for",
                "reconcile",
                "register_provider",
            ),
            pending_actions,
        ),
    ),
]


# The names 3c added to modules that already published others: only those are pinned
# here, so the older names keep their own tests.
ADDED = [
    (
        sdk_llm,
        {
            "EMBED_MODEL_DEFAULT": embeddings,
            "embed_corpus": embeddings,
            "make_narrative_llm_call": narrate,
            "TierConfig": tier_router,
            "governance_tier_for_intent": tier_router,
            "provider_root_url": tier_router,
        },
    ),
    (
        sdk_health,
        {
            **dict.fromkeys(
                ("CheckKind", "HealthCheck", "HealthState", "Reconnect"), health_models
            ),
            "net_probe_enabled": health_service,
        },
    ),
    (sdk_cli, {"console": console, "print_error": console}),
]
FACADES += [
    (
        sdk_vault,
        {
            **dict.fromkeys(
                ("CredentialRevokedError", "delete_token", "load_token", "save_token"), credentials
            ),
            "SecretStore": secret_store,
            "get_secret_store": secret_store,
            # Email setup's vault-key check (OSS plan R4): `iris doctor --fix`'s own function.
            "MasterKeyStatus": vault_keys,
            "master_key_status": vault_keys,
            "KeyFix": doctor,
            "fix_master_key": doctor,
        },
    ),
    (
        sdk_rag,
        {
            **dict.fromkeys(
                ("DocumentCatalog", "RagDocument", "register_document_catalog"), rag_documents
            ),
            "DocumentIndex": rag_index,
            **dict.fromkeys(
                ("IngestDeniedError", "IngestProposal", "execute_rag_ingest", "propose_rag_ingest"),
                ingest_gate,
            ),
            **dict.fromkeys(
                (
                    "IndexedDocument",
                    "IngestSource",
                    "KnownFile",
                    "RemovalAwareIngestSource",
                    "RemovedDocument",
                    "register_ingest_source",
                ),
                ingest_source,
            ),
            "IngestResult": rag_models,
            "search_documents": retrieve,
            "DocumentStore": rag_store,
        },
    ),
]


@pytest.mark.parametrize(("facade", "sources"), ADDED, ids=lambda v: getattr(v, "__name__", ""))
def test_each_added_name_is_the_core_object(facade: object, sources: dict[str, object]) -> None:
    assert set(sources) <= set(facade.__all__)  # type: ignore[attr-defined]
    for name, source in sources.items():
        assert getattr(facade, name) is getattr(source, name), name


def test_the_core_cli_and_plugins_print_to_one_console() -> None:
    assert render.console is sdk_cli.console


@pytest.mark.parametrize(("facade", "sources"), FACADES, ids=lambda v: getattr(v, "__name__", ""))
def test_each_name_is_the_core_object(facade: object, sources: dict[str, object]) -> None:
    assert sorted(facade.__all__) == sorted(sources)  # type: ignore[attr-defined]
    for name, source in sources.items():
        assert getattr(facade, name) is getattr(source, name), name


def test_data_path_follows_the_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    assert sdk_persistence.data_path("x.db") == tmp_path / "x.db"


def test_sqlite_conn_commits_and_uses_wal(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    with sdk_persistence.sqlite_conn(db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES ('kept')")
    check = sqlite3.connect(db)
    try:
        assert check.execute("SELECT v FROM t").fetchall() == [("kept",)]
    finally:
        check.close()


def test_a_registered_provider_is_found_by_its_source_kind() -> None:
    class _Provider:
        source_kind = "sdk-test"

        def desired_actions(self) -> list[sdk_pending_actions.DesiredAction]:
            return []

        def invoke(self, task: sdk_tasks.Task) -> str:
            return "ok"

    provider = _Provider()
    sdk_pending_actions.register_provider(provider)  # type: ignore[arg-type]
    try:
        assert sdk_pending_actions.provider_for("sdk-test") is provider
    finally:
        pending_actions.unregister_provider("sdk-test")


def test_iris_timezone_reads_iris_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_TZ", "America/Chicago")
    assert str(sdk_time.iris_timezone()) == "America/Chicago"
    monkeypatch.setenv("IRIS_TZ", "Not/AZone")
    assert str(sdk_time.iris_timezone()) == "UTC"
