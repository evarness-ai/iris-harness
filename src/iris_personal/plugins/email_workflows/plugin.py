"""``setup(api)`` for the email-workflows plugin — four bus subscriptions, the sweep, the judge.

The plugin is mostly *consumed services*: it subscribes to the chain the email sweep
produces, and its commands arrive separately through the CLI seam. That is the honest
shape of the domain — triage is not something a turn asks for, it is what happens when
mail lands. As of M6.1b it also registers the sweep itself, which was the core's until
the email library left for ``src/iris_personal`` (OSS plan M6, decision 2).

**All four subscribe with ``scope="process"``** (OSS plan M3.2a). The email chain
runs on ``eventbus.get_default_bus()``, not on the runtime's private bus, because
``iris email recategorize`` and ``iris email reingest-wiki`` emit ``email.classified``
with no runtime built at all. Subscribing on the runtime bus instead would raise
nothing and log nothing — the handlers would simply never fire — so the scope is
explicit here and pinned by a test.

The handlers keep the module-level ``subscribe_*`` helpers as their source of truth
for *what* to run; this module only decides *where* they are wired, so the same
functions still work when a CLI command wires them directly on a bus it owns.
"""

from __future__ import annotations

import logging

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.llm import make_narrative_llm_call
from iris_personal.email.events import EMAIL_CLASSIFIED, EMAIL_LABELS_CHANGED, EMAIL_NEW_ARRIVED

logger = logging.getLogger(__name__)


def setup(api: PluginAPI) -> None:
    # Phase 1 Track 1G — auto-classify new mail as the sweep lands it. The
    # classifier is lazy per ADR-0021 §6: MiniLM only loads on the first
    # email.new_arrived event, not at import time, which is why this imports the
    # handler rather than building a classifier here.
    from .triage import _handle_email_new_arrived as handle_new_mail

    api.subscribe(EMAIL_NEW_ARRIVED, handle_new_mail, scope="process")

    # Phase 1 Track 1K — bridge email.classified → WikiIngestEvent so the wiki
    # picks up entities from each classified email (ADR-0025). The wiki-side
    # consumer is core and is wired by build_runtime next to the WikiEngine.
    from .wiki_ingestion import _handle_email_classified as handle_classified

    api.subscribe(EMAIL_CLASSIFIED, handle_classified, scope="process")

    # Phase 2 Track 2B — followup auto-resolution. On every email.new_arrived, if
    # any open followup tracks the same thread, mark its wait resolved. Detection
    # itself is CLI-invoked (`iris email detect-followups`) per ADR-0022's
    # Tier-3-on-demand stance.
    from .followup import build_followup_handler

    api.subscribe(EMAIL_NEW_ARRIVED, build_followup_handler(), scope="process")

    # Phase 1 Track 1D — the sweep that keeps email.db current (canonical §3.1).
    # It was the core's until M6.1b: the store is what the planner, finance and every
    # email read tool query, so M3.2 called the sweep core maintenance. Decision 2
    # moves the whole email library out, so the sweep registers here, beside the
    # triage that classifies what it lands. Its schedule stays in the core's
    # config/heartbeats.yaml, and an entry with no registered handler is skipped.
    from iris_personal.email import build_email_sweep_handler

    api.register_heartbeat("email_sweep", build_email_sweep_handler())

    # Loop-proof PR 5 — the email judge (judge_wiring.py). The sweep now emits
    # `email.swept`: the queue makes each email the judge will judge a `waiting` row,
    # hidden until judged, and releases the rest at once as `email.new_arrived`. The
    # `email_judge` job (its own schedule in config/heartbeats.yaml) judges the queue and
    # releases what it judged, so every new-mail subscriber above sees mail only then.
    from iris_personal.email.events import EMAIL_SWEPT

    from .judge_wiring import build_judge_job, build_queue_handler

    api.subscribe(EMAIL_SWEPT, build_queue_handler(api), scope="process")
    api.register_heartbeat("email_judge", build_judge_job(api))

    # ADR-0071 slice 1 — keep the semantic email index current as mail arrives. On by
    # default since ADR-0121 PR 4 (IRIS_EMAIL_SEMANTIC_SEARCH=0 turns it off); the
    # email_semantic_index heartbeat fills in what arrived before, in bounded batches.
    from iris_personal.email.semantic_index import semantic_search_enabled

    if semantic_search_enabled():
        from iris_personal.email.semantic_index import subscribe_email_semantic_index

        subscribe_email_semantic_index()
    from .semantic_heartbeat import build_semantic_index_handler

    api.register_heartbeat("email_semantic_index", build_semantic_index_handler(api))

    # The email tools on the shared loop (search_inbox / read_email / …). The core
    # built them in `_domain_tools` until M6.1b; they join the same pool from here.
    from . import tools

    tools.register(api)

    # ADR-0119 — senders whose mail is mostly promotions stay off the memory Map. A
    # "check my email" chat lists every sender it read, so a daily shop recurs in every
    # summary; this plugin classified the mail, so it is the one that can say so.
    from iris_harness.sdk.memory import register_map_exclusions

    from .map_exclusions import build_provider

    register_map_exclusions("email_workflows.promotional_senders", build_provider())

    _wire_judge_labels(api)
    # OSS plan R4 — a mailbox with no write approval gets a System Health row naming
    # the command that approves it, so labels stopping is never silent.
    from .writes_health import write_approval_checks

    api.register_credential_check(
        "email_write_approvals", lambda net_probe: write_approval_checks()
    )
    # Owner decision 2026-09-30 — an account whose email setup has not turned the sweep
    # on is not fetched on the schedule; a System Health row says so and how to go on.
    from .sweep_health import sweep_wait_checks

    api.register_credential_check("email_sweep_setup", lambda net_probe: sweep_wait_checks())
    # Loop-proof PR 5 — the owner corrects the email judge: the Action Center card on
    # Unsure emails, /api/v1/email/judgments (the web list), the `email_rebucket` chat
    # intercept, and the digest footer's learned source (judge_surfaces.py).
    from .judge_surfaces import register as register_judge_surfaces

    register_judge_surfaces(api)

    # OSS plan R4 — email setup's API (/api/v1/email/onboarding), the same state machine
    # `iris email setup` drives (onboarding.py).
    from .onboarding_api import register as register_onboarding_api

    register_onboarding_api(api)

    # The `email` agent itself, also the core's until M6.1b: the digest lane, and the
    # governed loop's degrade path for the `email` intent the plugin puts on it.
    _register_agent(api)

    # Loop-proof D13 — hidden mail: `judge_reachable` (paged by the health watch) and
    # the digest footer's "Email jobs" line, over the core's kept heartbeat runs.
    from .job_watch import register as register_job_watch

    register_job_watch(api)

    logger.debug(
        "email_workflows: 4 subscriptions, the sweep, the judge, the email tools, the Map "
        "exclusions and the agent registered"
    )


def _wire_judge_labels(api: PluginAPI) -> None:
    """Loop-proof PR 5 — the judge's IRIS/* Gmail labels (``judge_labels.py``). The
    sweep's ``email.labels_changed`` carries the owner's Gmail relabels back as
    corrections; a correction from the card, chat or web moves the label right away.
    Both on the process bus, where the sweep and ``apply_correction``'s emit publish."""
    from .judge_config import EMAIL_JUDGMENT_CORRECTED
    from .judge_labels import LabelHandlers

    handlers = LabelHandlers(config_dir_for=lambda: api.services.config_dir)
    api.subscribe(EMAIL_LABELS_CHANGED, handlers.on_labels_changed, scope="process")
    api.subscribe(EMAIL_JUDGMENT_CORRECTED, handlers.on_corrected, scope="process")


def _register_agent(api: PluginAPI) -> None:
    """Register the chat ``email`` agent (OSS plan M6.1b) and put it on the loop.

    The digest is the ``email`` lane when the harness's governed loop is off. With it
    on, the loop answers ``email`` (only it can pause on a destructive call's approval
    and resume, ADR-0118 step 5) and the fallback is its degrade path.
    """
    from .agent import make_email_fallback_handler, make_email_handler

    services = api.services
    narrative = make_narrative_llm_call(services.tier_router) if services.tier_router else None
    handler, stream_handler = make_email_handler(llm_call=narrative, data_dir=services.data_dir)
    api.register_intent_handler("email", handler, stream_handler=stream_handler)
    # The fallback's tool calls go through `api.tools`: governed, as this plugin.
    fallback = make_email_fallback_handler(
        api.tools, data_dir=services.data_dir, llm_call=narrative
    )
    api.register_loop_intent("email", fallback=fallback)
