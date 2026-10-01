"""The email split, pinned from both sides (OSS plan M3.2, widened at M6.1b).

Two things have to hold and neither fails loudly on its own:

1. **The library stands alone.** The planner's daily plan, finance's ingest path and
   every email read tool go through ``EmailStore`` and the inbox digest, and none of
   that may reach back into the plugin. M3.2 called this "access is core"; at M6.1b
   the library leaves the core with its plugin (OSS plan M6, decision 2), so what the
   assertions pin now is the direction of the dependency, not its address.
2. **The subscriptions land on the bus their producer publishes on.** They are
   process-global (``iris email recategorize`` drives the same chain with no runtime
   built), and subscribing to the runtime bus instead would raise nothing and log
   nothing — the handlers would just never fire.

The sweep that feeds the whole chain registers here as of M6.1b, for the same reason.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from iris_harness.foundation.eventbus import EventBus, get_default_bus, reset_default_bus
from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_personal.email.events import (
    EMAIL_CLASSIFIED,
    EMAIL_LABELS_CHANGED,
    EMAIL_NEW_ARRIVED,
    EMAIL_SWEPT,
)
from iris_personal.plugins.email_workflows.judge_config import (
    EMAIL_JUDGED,
    EMAIL_JUDGMENT_CORRECTED,
)


@pytest.fixture(autouse=True)
def _fresh_default_bus():
    reset_default_bus()
    yield
    reset_default_bus()


class _Heartbeats:
    """Minimal stand-in for the heartbeat service the API registers into."""

    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def register_handler(self, name: str, handler: object) -> None:
        self.handlers[name] = handler


class _Executor:
    """Minimal stand-in for the agent executor the API registers into."""

    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def register(self, agent_type: str, handler: object) -> None:
        self.handlers[agent_type] = handler

    def register_stream(self, agent_type: str, handler: object) -> None:
        pass


def _api(
    registry: PluginRegistry,
    runtime_bus: EventBus,
    heartbeats: _Heartbeats | None = None,
    executor: _Executor | None = None,
) -> PluginAPI:
    registry.add_plugin(
        PluginRecord(name="email_workflows", source="builtin", status=PluginStatus.LOADED)
    )
    services = HarnessServices(
        config_dir=Path("/nonexistent"),
        data_dir=Path("/nonexistent"),
        tier_router=None,
        agent_executor=executor or _Executor(),
        heartbeats=heartbeats or _Heartbeats(),
        channels=None,
        deterministic_reply=lambda **kw: None,
        events=runtime_bus,
    )
    return PluginAPI(plugin="email_workflows", services=services, registry=registry)


# ─── 1. Access is core ───────────────────────────────────────────────────────


def test_the_email_library_imports_nothing_from_the_plugin() -> None:
    """A library module reaching into plugins_builtin is the coupling M3 removed."""
    from iris_personal.email import agent_tools, digest, events, store, sweep

    for module in (store, digest, agent_tools, sweep, events):
        source = inspect.getsource(module)
        assert "plugins_builtin" not in source, f"{module.__name__} imports a plugin"


def test_email_package_no_longer_re_exports_the_classifier() -> None:
    """``from iris_personal.email import subscribe_email_triage`` was a core->plugin edge."""
    import iris_personal.email as email_pkg

    assert set(email_pkg.__all__) == {
        "EmailStore",
        "EmailSweepHandler",
        "build_email_sweep_handler",
    }
    assert not hasattr(email_pkg, "subscribe_email_triage")


def test_reading_the_mailbox_works_with_no_plugin_mounted(tmp_path: Path) -> None:
    """Store -> digest, the path the planner and finance take, with nothing mounted."""
    from datetime import UTC, datetime

    from iris_personal.email.digest import build_inbox_digest
    from iris_personal.email.store import EmailStore

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    digest = build_inbox_digest(store, now=datetime(2026, 9, 10, 9, 0, tzinfo=UTC))
    assert digest is not None  # an empty mailbox still renders, with no plugin mounted


def test_followup_dims_are_emails_vocabulary() -> None:
    """The followup key is email's (email slice step 4); the core's recorders keep a copy
    until PR 7 moves them, pinned equal by test_email/test_feedback_keys.py."""
    from iris_personal.email.feedback_keys import email_followup_dims_from
    from iris_personal.plugins.email_workflows import followup

    assert followup.email_followup_dims_from is email_followup_dims_from
    assert email_followup_dims_from("gmail:me", "Quant <no-reply@quant.example>") == {
        "account": "gmail:me",
        "from_domain": "quant.example",
    }


def test_wiki_side_consumer_is_core() -> None:
    """ADR-0025's two-hop design exists so any producer can feed the wiki."""
    from iris_harness.memory.knowledge.event_subscribers import (
        WIKI_INGEST_REQUESTED,
        make_wiki_ingest_consumer,
    )

    assert WIKI_INGEST_REQUESTED == "wiki.ingest_requested"
    assert callable(make_wiki_ingest_consumer(object()))


# ─── 2. The subscriptions land on the right bus ──────────────────────────────


def test_setup_subscribes_all_three_on_the_process_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_personal.plugins.email_workflows import (
        followup,
        plugin,
        triage,
        wiki_ingestion,
    )

    # Don't open any store — this test is about wiring, not classification.
    monkeypatch.setattr(followup, "build_followup_handler", lambda **kw: lambda payload: None)

    registry = PluginRegistry()
    runtime_bus = EventBus()
    heartbeats = _Heartbeats()
    executor = _Executor()
    plugin.setup(_api(registry, runtime_bus, heartbeats, executor))

    # What the core used to register for this domain and no longer can (M6.1b): the
    # sweep that produces the chain these three consume, and the `email` agent.
    assert "email_sweep" in heartbeats.handlers
    assert "email" in executor.handlers

    assert sorted(registry.subscriptions()) == sorted(
        [
            ("email_workflows", EMAIL_NEW_ARRIVED, "process"),
            ("email_workflows", EMAIL_CLASSIFIED, "process"),
            ("email_workflows", EMAIL_NEW_ARRIVED, "process"),
            # Loop-proof PR 5: the judge's Gmail labels (read-back, and a correction's
            # immediate label move).
            ("email_workflows", EMAIL_LABELS_CHANGED, "process"),
            ("email_workflows", EMAIL_JUDGMENT_CORRECTED, "process"),
            # Loop-proof PR 5: the sweep's email.swept feeds the email judge's queue,
            # which releases mail as email.new_arrived.
            ("email_workflows", EMAIL_SWEPT, "process"),
            # Loop-proof PR 5: the Unsure cards follow the judge and every correction.
            ("email_workflows", EMAIL_JUDGED, "process"),
            ("email_workflows", EMAIL_JUDGMENT_CORRECTED, "process"),
        ]
    )
    assert "email_judge" in heartbeats.handlers
    # All on the process bus; none on the runtime bus. The third new-mail handler is
    # the semantic index, subscribed directly and on by default (ADR-0121 PR 4).
    assert len(get_default_bus()._handlers[EMAIL_NEW_ARRIVED]) == 3
    assert len(get_default_bus()._handlers[EMAIL_SWEPT]) == 1
    assert len(runtime_bus._handlers[EMAIL_SWEPT]) == 0
    assert "email_semantic_index" in heartbeats.handlers
    # Loop-proof PR 5: the chat correction takes its declared place in the chain.
    assert registry.intercept("email_rebucket") is not None
    assert len(get_default_bus()._handlers[EMAIL_CLASSIFIED]) == 1
    assert len(runtime_bus._handlers[EMAIL_NEW_ARRIVED]) == 0
    assert len(runtime_bus._handlers[EMAIL_CLASSIFIED]) == 0
    assert len(get_default_bus()._handlers[EMAIL_LABELS_CHANGED]) == 1
    # Two: the Gmail label moves at once (judge_labels) and the Unsure cards refresh.
    assert len(get_default_bus()._handlers[EMAIL_JUDGMENT_CORRECTED]) == 2

    # And they are the real handlers, not the module-level subscribe_* helpers.
    assert triage._handle_email_new_arrived is not None
    assert wiki_ingestion._handle_email_classified is not None
