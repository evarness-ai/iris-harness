"""IRIS API service entrypoint."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from iris_harness.cli.web_commands import ModelOverrides
from iris_harness.foundation.paths import config_root, repo_root

# Load the repo-root .env before importing iris_harness.runtime so provider profiles
# (OPENROUTER_API_KEY, GITHUB_TOKEN, ANTHROPIC_API_KEY, IRIS_ENABLE_COPILOT_BACKEND, …)
# are visible inside the uvicorn process — without this, the server resolves
# every key as missing and every cloud LLM call returns 401.
_REPO_ROOT_ENV = repo_root() / ".env"
load_dotenv(_REPO_ROOT_ENV)

# Install the root log handler (IRIS_LOG_LEVEL, default INFO) before anything else imports
# so every iris.* INFO line — including the ingress/egress audit trail — actually appears
# in the service log instead of being dropped at the default WARNING level.
from iris_harness.foundation.observability.logging_setup import configure_logging  # noqa: E402

configure_logging(service="iris_api")

from iris_harness.foundation.auth import PROBE_ONLY_EXEMPT_PATHS, Principal  # noqa: E402
from iris_harness.foundation.settings.catalog import load_sidecar_catalogs  # noqa: E402
from iris_harness.kernel.governor import IRISGovernorService  # noqa: E402
from iris_harness.runtime import IrisRuntime, build_runtime  # noqa: E402
from iris_harness.runtime.api_routes import is_public_callback  # noqa: E402
from iris_harness.server.auth import (  # noqa: E402
    install_bearer_auth,
    install_callback_log_redaction,
    routed_path,
)
from iris_harness.server.iris_api.action_center_routes import (  # noqa: E402
    install_action_center_routes,
)
from iris_harness.server.iris_api.agent_routes import (  # noqa: E402
    install_agent_routes,
)
from iris_harness.server.iris_api.chat_routes import (  # noqa: E402
    install_chat_routes,
)
from iris_harness.server.iris_api.detached_turns import DetachedTurns  # noqa: E402
from iris_harness.server.iris_api.device_routes import (  # noqa: E402
    PAIR_CLAIM_PATH,
    LazyDeviceService,
    install_device_routes,
    is_device_admin_write,
)
from iris_harness.server.iris_api.digest_feedback_routes import (  # noqa: E402
    NOT_USEFUL_PATH as DIGEST_NOT_USEFUL_PATH,
)
from iris_harness.server.iris_api.digest_feedback_routes import (  # noqa: E402
    install_digest_feedback_routes,
)
from iris_harness.server.iris_api.digest_routes import install_digest_routes  # noqa: E402
from iris_harness.server.iris_api.governance_routes import (  # noqa: E402
    _env_flag,
    _flag_payload,
    install_governance_routes,
)
from iris_harness.server.iris_api.health_routes import install_health_routes  # noqa: E402
from iris_harness.server.iris_api.learning_routes import install_learning_routes  # noqa: E402
from iris_harness.server.iris_api.memory_routes import (  # noqa: E402
    RAG_UPLOAD_PATH,
    _upload_limit_detail,
    _upload_too_large,
    install_memory_routes,
)
from iris_harness.server.iris_api.ops_routes import (  # noqa: E402
    install_ops_routes,
)
from iris_harness.server.iris_api.playground_routes import (  # noqa: E402
    install_playground_routes,
)
from iris_harness.server.iris_api.push_routes import (  # noqa: E402
    SUBSCRIBE_PATH as PUSH_SUBSCRIBE_PATH,
)
from iris_harness.server.iris_api.push_routes import install_push_routes  # noqa: E402
from iris_harness.server.iris_api.routines_routes import (  # noqa: E402
    install_routines_routes,
)
from iris_harness.server.iris_api.runtime_access import (  # noqa: E402
    runtime_or_503 as _runtime_or_503,
)
from iris_harness.server.iris_api.session_routes import (  # noqa: E402
    install_session_routes,
)
from iris_harness.server.iris_api.settings_routes import (  # noqa: E402
    install_settings_routes,
)
from iris_harness.server.iris_api.static_ui import install_static_ui  # noqa: E402
from iris_harness.server.iris_api.write_guard import _may_write  # noqa: E402
from iris_harness.tools.mcp_bridge import MCPBridge, load_mcp_bridge_config  # noqa: E402

if TYPE_CHECKING:
    from iris_harness.kernel.governance.devices import DeviceService

try:
    from iris_harness.server.iris_api.mcp_server import create_mcp_router
except ModuleNotFoundError:
    MCP_SERVER_PATH = Path(__file__).resolve().with_name("mcp_server.py")
    MCP_SERVER_SPEC = importlib.util.spec_from_file_location(
        "iris_api_mcp_server",
        MCP_SERVER_PATH,
    )
    if MCP_SERVER_SPEC is None or MCP_SERVER_SPEC.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load MCP server module: {MCP_SERVER_PATH}") from None
    MCP_SERVER_MODULE = importlib.util.module_from_spec(MCP_SERVER_SPEC)
    MCP_SERVER_SPEC.loader.exec_module(MCP_SERVER_MODULE)
    create_mcp_router = MCP_SERVER_MODULE.create_mcp_router

REPO_ROOT = repo_root()

logger = logging.getLogger(__name__)


class ObservabilityBackendStatus(BaseModel):
    """Backend health for rollout observability."""

    kind: str = "otlp"
    enabled: bool
    healthy: bool
    # The OTLP traces endpoint spans export to (credentials stripped); None when unset.
    endpoint: str | None = None
    instrumented_targets: list[str] = Field(default_factory=list)
    error: str | None = None


class ObservabilityMetricsResponse(BaseModel):
    """Stable operator-facing payload for recent LLM activity."""

    backend: ObservabilityBackendStatus
    summary: dict[str, Any]


# ---- Governance + settings read surfaces (Phase 6, read-only) ----------------


# Curated non-secret runtime behavior flags for the read-only Settings view.
_RUNTIME_FLAGS: tuple[tuple[str, str, bool], ...] = (
    ("IRIS_ADAPTIVE_TIERS", "Adaptive tier downshift", False),
    ("IRIS_OBSERVABILITY_METRICS_ENABLED", "Observability metrics", False),
    ("IRIS_LESSON_CAPTURE_ENABLED", "Lesson capture", True),
    ("IRIS_CURATOR_FAITHFULNESS_LLM", "Faithfulness judge", False),
    ("IRIS_CURATOR_GROUNDING_LLM", "Grounding judge", False),
    ("IRIS_CURATOR_LEAK_JUDGE", "Leak judge", True),
    ("IRIS_SKILL_SYNTHESIS", "Skill synthesis", False),
    ("IRIS_DISABLE_WARMUP", "Warmup disabled", False),
    # Plugin switches (semantic email search, finance dues) are not listed here: the
    # core carries no plugin's vocabulary; /settings/catalog lists them (ADR-0120).
    ("IRIS_FEEDBACK_CAPTURE", "Answer feedback (thumbs)", False),
    ("IRIS_FEEDBACK_CLARIFY", "Judge-gated clarify nudge", False),
)


# ── Web-UI write gate ─────────────────────────────────────────────────────────
# This switch stood in for authentication: with no way to know who was asking,
# UI-driven control mutations were off unless the operator opted in. Paired
# devices (ADR-0117) are that authentication, so a request that carries one is
# now judged on its own scope and the switch does not apply to it:
#
#   paired `control` device  → may write
#   paired `read` device     → refused, however the switch is set
#   shared secret / no creds → the switch still decides (local dev, the Vite
#                              proxy, service-to-service — a secret says which
#                              *process* is calling, never which person)
#
# Reads, chat and RAG upload are always allowed. MCP writes are gated like any other:
# they run external tools, so a read-only device must not reach them.


# Done / Snooze / Undo on a reminder (loop-proof D14, PR 3b) — by id, or by the message
# that delivered it (a Telegram reply-to). The owner's decision in the PR 3b prototype:
# "Done and Snooze are allowed without control access, and nothing else is" — so a
# READ-ONLY paired phone may make exactly these writes. Exact shapes only: no other
# reminder write (and no other route) rides along.
_REMINDER_ACTION_PATH = re.compile(
    r"^/api/v1/reminders/(?:"
    r"[0-9A-Za-z-]{1,64}/(?:done|snooze|undo)"
    r"|by-message/[a-z_]{1,32}/-?[0-9A-Za-z_]{1,64}/[0-9A-Za-z_-]{1,64}/(?:done|snooze)"
    r")$"
)


def _reminder_action(method: str, path: str) -> bool:
    return method == "POST" and bool(_REMINDER_ACTION_PATH.match(path))


def _device_may_act_on_reminder(principal: Principal | None, method: str, path: str) -> bool:
    """Any paired device — read-only included — may Done / Snooze a reminder."""
    return principal is not None and principal.kind == "device" and _reminder_action(method, path)


# The gated writes the shared service secret may make without IRIS_WEBUI_ALLOW_WRITES:
# answering an approval — the channel gateway does it for the owner's allowlisted
# Telegram chat when they tap a button (owner's decision, 2026-09-21) — and Done /
# Snooze on a reminder, for the same chat's reminder buttons and replies (PR 3b).
# Nothing else: the secret still cannot change settings, flags or tasks unless the
# operator opts in.
_SERVICE_WRITE_PATH = re.compile(
    r"^/governance/approvals/[^/]+/respond$|" + _REMINDER_ACTION_PATH.pattern
)


def _service_may_write(principal: Principal | None, method: str, path: str) -> bool:
    return (
        principal is not None
        and principal.kind == "service"
        and method == "POST"
        and bool(_SERVICE_WRITE_PATH.match(path))
    )


# Always-permitted mutations — the conversational product surface. Everything
# else that mutates state is gated behind IRIS_WEBUI_ALLOW_WRITES, so a NEW
# mutating route is gated automatically instead of silently open (deny by
# default; previously this was an allowlist of gated paths and routes like
# POST /portfolio/import-holdings slipped through).
_UNGATED_WRITE_PATHS = frozenset(
    {
        # Chat IS the product; its actions run through the governance kernel.
        "/chat",
        "/chat/stream",
        "/chat/cancel",
        # The first-chat welcome (ADR-0127) is a chat turn the harness opens itself, once
        # per home, through the same governed pipeline.
        "/chat/welcome",
        # Model preload — no user-visible state mutation.
        "/warmup",
        # Document upload + search were always allowed for read-only consoles.
        RAG_UPLOAD_PATH,
        "/rag/search",
        # Local learning telemetry only (ADR-0072 / issue 0028) — intentionally
        # open so a read-only user can still rate answers and suppress noise.
        "/api/feedback",
        "/surface-feedback",
        # The digest's 👎 on a Focus line is the same suppression signal (loop-proof D17).
        DIGEST_NOT_USEFUL_PATH,
        # Subscribing to notifications asks to RECEIVE what IRIS already
        # decided to say; it changes where the harness's voice carries, not
        # its state (track 2b PR 9). A read-only phone that cannot be told
        # its calendar credential broke is a read-only phone nobody looks at.
        PUSH_SUBSCRIBE_PATH,
        # A dry run of a removal (ADR-0119): computes what WOULD go and changes
        # nothing, so a read-only console can still show it; the removal is gated.
        "/memory/removed/preview",
    }
)


_push_subscription_store: Any = None


def _push_store() -> Any:
    """The shared push-subscription store, opened on first use.

    Lazy for the same reason the devices service is: opening it is a disk
    write, and a harness nobody has subscribed to should not make the file.
    """
    global _push_subscription_store  # one process-wide store
    if _push_subscription_store is None:
        from iris_harness.services.channels.web_push import PushSubscriptionStore

        _push_subscription_store = PushSubscriptionStore()
    return _push_subscription_store


def _digest_store() -> Any:
    """The stored-digest store the brief handler writes (one per process)."""
    from iris_harness.services.digests import shared_digest_store

    return shared_digest_store()


def _is_gated_write(method: str, path: str) -> bool:
    """Deny-by-default server-side write gate.

    Every non-GET route is gated behind IRIS_WEBUI_ALLOW_WRITES unless it is
    part of the always-allowed product surface declared above.

    Pairing and revoking a device are not console writes but authentication
    administration, and enforce scope in the route (ADR-0117, "Pairing flow"): behind
    the switch, a default install could never pair its first device and a read-only
    console could not revoke a lost phone."""
    if method not in {"POST", "PATCH", "PUT", "DELETE"}:
        return False
    if path in _UNGATED_WRITE_PATHS:
        return False
    # The MCP routes were once exempt here on the grounds that their tools carry their
    # own governance. They do not stop a read-only device: the governor's approval was
    # a body flag the caller set itself. They are gated writes like everything else now.
    return not is_device_admin_write(method, path)


# `/healthz` is the probe; the pairing claim is the one data route without a
# credential — an unpaired device has none to present, the code is what it presents.
# iris_api only: the governor and the evaluator keep their own exempt set.
# Readiness, not liveness: 200 only once the runtime has been built. /healthz answers
# 200 as soon as the process serves, so a runtime that failed to build (the API then
# 503s every data route) still looked healthy to Docker and so to roll_vm.sh. The
# container's healthcheck asks this instead; /healthz keeps meaning "the process is up"
# for the health watch and the other probes. Open like /healthz: it says only whether
# the runtime is up.
READY_PATH = "/readyz"
_BEARER_EXEMPT_PATHS = PROBE_ONLY_EXEMPT_PATHS | {PAIR_CLAIM_PATH, READY_PATH}


def _default_mcp_bridge() -> MCPBridge:
    """The MCP bridge for the HTTP routes, with the governance kernel when MCP is on.

    Built without a kernel, the bridge never fires PreToolUse, so ``MCPAllowlistHook``
    (the per-server, per-tool allowlist) never saw an HTTP call. The kernel is built
    only when the bridge is enabled: with MCP off (the shipped default) every route
    refuses before it would be used, and building it costs startup time."""
    # Config (servers, skills, signing, the governor policy) from the resolved config
    # dir -- its root is config_root(), see foundation/paths.py; the governor's audit log
    # stays under REPO_ROOT/data, where it always was.
    root = config_root()
    config = load_mcp_bridge_config(root)
    kernel = None
    if config.enabled:
        from iris_harness.kernel.governance import kernel_from_env

        kernel = kernel_from_env()
    governor = IRISGovernorService.from_repo_root(
        REPO_ROOT, policy_path=root / "config" / "governor" / "policy.yaml"
    )
    return MCPBridge(root, config=config, governance_kernel=kernel, governor_service=governor)


def create_app(
    *,
    mcp_bridge: MCPBridge | None = None,
    runtime: IrisRuntime | None = None,
    auto_start_runtime: bool = True,
    device_service: DeviceService | None = None,
) -> FastAPI:
    """Create the IRIS API service.

    Parameters
    ----------
    mcp_bridge
        Optional pre-built MCP bridge. Defaults to one loaded from the repo root.
    runtime
        Optional pre-built ``IrisRuntime``. When provided, its lifecycle is the
        caller's responsibility (no startup/shutdown calls). When ``None``, a
        runtime is built lazily and managed by the lifespan.
    auto_start_runtime
        When ``True`` and ``runtime`` is None, the lifespan calls
        ``runtime.startup()`` on startup and ``runtime.shutdown()`` on shutdown.
    device_service
        Optional pre-built paired-devices service (ADR-0117); tests inject a store and
        a clock through it. Defaults to one built on first use from the governance
        data dir. Either way the auth middleware and the device routes share it.
    """
    bridge = mcp_bridge or _default_mcp_bridge()
    externally_managed_runtime = runtime is not None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Setup tracing before runtime so LangChain calls are auto-instrumented
        try:
            from iris_harness.foundation.observability.tracer import setup_tracing_state

            app.state.observability = setup_tracing_state()
            app.state.tracer = app.state.observability.tracer
        except Exception:  # noqa: BLE001
            logger.warning("tracing setup failed; running without trace export")
            app.state.observability = None
            app.state.tracer = None

        if app.state.runtime is None:
            try:
                app.state.runtime = build_runtime()
            except Exception:
                logger.exception("runtime build failed; /chat will be unavailable")
                app.state.runtime = None
        if app.state.runtime is not None:
            app.state.runtime.tracer = app.state.tracer
            # Plugins registered their routers while the runtime was built just now.
            _mount_plugin_routes(app)
        if app.state.runtime is not None and auto_start_runtime and not externally_managed_runtime:
            try:
                app.state.runtime.startup()
            except Exception:
                logger.exception("runtime startup failed")
        try:
            yield
        finally:
            if (
                app.state.runtime is not None
                and auto_start_runtime
                and not externally_managed_runtime
            ):
                try:
                    app.state.runtime.shutdown()
                except Exception:
                    logger.exception("runtime shutdown failed")

    app = FastAPI(title="IRIS API", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.observability = None
    # Remembers what `/model` and `/router` were set to, per session. A browser
    # has no REPL Session to hold the choice, so without this a `/model` would be
    # forgotten by the next message; the chat routes below read it.
    app.state.model_overrides = ModelOverrides()
    app.state.turns = DetachedTurns()
    # ONE paired-devices service for the verifier and the routes: the failed-claim
    # throttle lives in it. Built on first use — opening the devices DB is a disk
    # write, and a request carrying the shared secret never needs it.
    devices = LazyDeviceService(device_service)

    @app.middleware("http")
    async def _upload_size_guard(request: Request, call_next: Any) -> Any:
        # Registered first, so it runs innermost: after the bearer check, before the
        # body is parsed. See _upload_too_large for why the route cannot do this.
        if _upload_too_large(
            request.method, routed_path(request), request.headers.get("content-length")
        ):
            return JSONResponse(status_code=413, content={"detail": _upload_limit_detail()})
        return await call_next(request)

    @app.middleware("http")
    async def _control_writes_guard(request: Request, call_next: Any) -> Any:
        # Block control mutations unless the operator opted in. Allowlist-based,
        # so chat / uploads / reads pass straight through.
        if not _is_gated_write(request.method, routed_path(request)):
            return await call_next(request)
        # The bearer check has already run, so the principal is there for every
        # gated write that carried a credential.
        principal = getattr(request.state, "principal", None)
        if _device_may_act_on_reminder(principal, request.method, routed_path(request)):
            return await call_next(request)
        if principal is not None and principal.kind == "device" and not principal.can_control:
            return JSONResponse(
                status_code=403,
                content={"detail": "this device is paired read-only"},
            )
        if not _may_write(principal) and not _service_may_write(
            principal, request.method, routed_path(request)
        ):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "control writes are disabled — pair a control device, "
                    "or set IRIS_WEBUI_ALLOW_WRITES=1 to enable editing without pairing"
                },
            )
        return await call_next(request)

    # Registered between the write guard and the ingress log: Starlette runs
    # the last-registered middleware first, so requests flow ingress-log →
    # bearer-auth → write-guard and refused attempts still hit the trail.
    #
    # Paired devices authenticate here as well as the shared secret (ADR-0117); the
    # governor and the evaluator do not pass a verifier and stay secret-only. `/health`
    # on this service is the System Health snapshot, not a probe, so only `/healthz`
    # and the pairing claim are open.
    # A plugin's OAuth callback (``register_public_callback``) is the one other way in:
    # the provider redirects the browser there without our cookie or header, and the
    # route checks the one-time state it issued instead.
    install_bearer_auth(
        app,
        exempt_paths=_BEARER_EXEMPT_PATHS,
        device_verifier=devices.verify,
        public_route=is_public_callback,
    )
    # Its query (state + one-time code) is that request's credential: keep it out of
    # uvicorn's access log, which otherwise prints the full request line.
    install_callback_log_redaction(is_public_callback)

    # The built web console, when the deployment ships one. Ahead of the bearer
    # check because the shell and its assets carry no data; behind the ingress
    # log so those requests are in the trail too. A no-op without a build.
    install_static_ui(app)

    @app.middleware("http")
    async def _ingress_log(request: Request, call_next: Any) -> Any:
        # Log EVERY inbound request crossing the API boundary (all routes, no allowlist),
        # so there is a complete ingress trail. Health probes are demoted to DEBUG to keep
        # the steady-state log readable.
        import time as _time

        from iris_harness.foundation.observability.logging_setup import (
            ingress_logger,
            log_ingress,
        )

        start = _time.monotonic()
        path = routed_path(request)
        is_probe = path in {"/healthz", "/health", READY_PATH}
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            log_ingress(
                method=request.method,
                path=path,
                source=request.client.host if request.client else "",
                status=status,
                duration_ms=(_time.monotonic() - start) * 1000,
            )
            raise
        dur_ms = (_time.monotonic() - start) * 1000
        if is_probe:
            ingress_logger.debug("INGRESS %s %s status=%s", request.method, path, status)
        else:
            log_ingress(
                method=request.method,
                path=path,
                source=request.client.host if request.client else "",
                status=status,
                duration_ms=dur_ms,
            )
        return response

    install_health_routes(app, lambda: _runtime_or_503(app))

    install_learning_routes(app, lambda: _runtime_or_503(app))

    @app.get("/context-health")
    def context_health(session_id: str = "default") -> dict[str, Any]:
        """How the harness is managing its context (ADR-0081): conversation-window fill +
        last compaction, the in-loop transcript/memory budget split + latest eviction, and
        the surface-feedback suppression roll-up. Read-only; one view of the brain's memory
        management. `session_id` scopes the window fill to one conversation."""
        rt = getattr(app.state, "runtime", None)
        if rt is None:
            return {"available": False}
        return {"available": True, **rt.sessions.context_health(session_id)}

    @app.post("/context-health/compact")
    def context_health_compact(session_id: str = "default") -> dict[str, Any]:
        """Force-compact a session's conversation now (ADR-0084) — the action half of the
        context-health control loop: see the window fill, reclaim it on demand. Summarizes
        the older span even below the auto trigger; a short conversation is a no-op.
        Write-gated by IRIS_WEBUI_ALLOW_WRITES."""
        rt = _runtime_or_503(app)
        return rt.sessions.compact_now(session_id)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        observability = getattr(app.state, "observability", None)
        return {
            "status": "ok",
            "mcp_bridge_enabled": bridge.config.enabled,
            "enabled_server_count": len(bridge.list_enabled_servers()),
            "runtime_ready": app.state.runtime is not None,
            "tracing_enabled": bool(getattr(observability, "enabled", False)),
            "otlp_endpoint": getattr(observability, "endpoint", None),
            "otel_targets": list(getattr(observability, "instrumented_targets", ())),
            "observability_error": getattr(observability, "error", None),
        }

    @app.get(READY_PATH)
    def readyz() -> JSONResponse:
        """Ready = the runtime was built; 503 otherwise (see READY_PATH)."""
        if app.state.runtime is None:
            return JSONResponse(
                status_code=503, content={"ready": False, "reason": "runtime not built"}
            )
        return JSONResponse(status_code=200, content={"ready": True})

    @app.get("/observability/llm-metrics", response_model=ObservabilityMetricsResponse)
    def llm_metrics() -> ObservabilityMetricsResponse:
        if not _env_flag("IRIS_OBSERVABILITY_METRICS_ENABLED", default=False):
            raise HTTPException(status_code=404, detail="observability metrics disabled")

        from iris_harness.foundation.observability.metrics_summary import summarize_llm_metrics

        observability = getattr(app.state, "observability", None)
        backend = ObservabilityBackendStatus(
            enabled=bool(getattr(observability, "enabled", False)),
            healthy=getattr(observability, "tracer", None) is not None,
            endpoint=getattr(observability, "endpoint", None),
            instrumented_targets=list(getattr(observability, "instrumented_targets", ())),
            error=getattr(observability, "error", None),
        )
        runtime = getattr(app.state, "runtime", None)
        learning_store = getattr(runtime, "learning_store", None)
        # The in-process signal counters come from here, not from observability itself:
        # foundation is the bottom layer and may not reach up into learning (M6.2).
        try:
            from iris_harness.services.learning.signals import signal_health

            process_signals = signal_health()
        except Exception:  # noqa: BLE001 — best-effort meta-observability
            process_signals = {}
        return ObservabilityMetricsResponse(
            backend=backend,
            summary=summarize_llm_metrics(
                learning_store=learning_store, process_signals=process_signals
            ),
        )

    install_session_routes(app, lambda: _runtime_or_503(app))

    install_chat_routes(app, lambda: _runtime_or_503(app))

    install_action_center_routes(app, lambda: _runtime_or_503(app))

    install_agent_routes(app, lambda: _runtime_or_503(app))

    install_routines_routes(app, lambda: _runtime_or_503(app))

    install_ops_routes(app, lambda: _runtime_or_503(app))

    install_memory_routes(app, lambda: _runtime_or_503(app))

    install_governance_routes(app, lambda: _runtime_or_503(app))

    # ── Paired devices: pair, list, revoke (ADR-0117) ────────────────────────
    # `/api/v1/devices/*`. The same rule as the approvals above: the capability is
    # `governance.devices.DeviceService`; these routes and `iris device` (which calls
    # them) are surfaces.
    install_device_routes(app, devices.get)

    # ── Web Push: the browser subscribes, the harness notifies (track 2b PR 9)
    # The capability is `services.channels.web_push`; these routes are a surface,
    # and `WebPushConnector` is the other one — the health watch reaches it through
    # the ordinary channel gateway, unchanged.
    install_push_routes(app, _push_store)

    # ── Stored digests: the full web copy a push notification lands on ──────
    # The capability is `services.digests.DigestStore`, written by the skill_brief
    # handler; these read routes and the console's /digest view are surfaces.
    install_digest_routes(app, _digest_store)

    # ── The digest's 👎 on a Focus line → surface suppression (loop-proof D17) ──
    install_digest_feedback_routes(app)

    # ── Settings the app changes, and their history (ADR-0120) ───────────────
    # The capability is the heartbeat scheduler's update/reset over the settings
    # store on the data volume; these routes and `/heartbeats set|reset` are surfaces.
    install_settings_routes(
        app,
        lambda: _runtime_or_503(app),
        devices.get,
        sidecar_catalogs=load_sidecar_catalogs(),
    )

    @app.get("/settings")
    def settings() -> dict[str, Any]:
        from iris_harness.services.system.status import system_report

        rt = _runtime_or_503(app)
        report = system_report()
        tiers = [
            {
                "name": t.name,
                "provider": t.provider,
                "model": t.model,
                "max_tokens": t.max_tokens,
                "temperature": t.temperature,
                "use_for": list(t.use_for),
            }
            for t in rt.tier_router._tiers.values()
        ]
        # Providers: presence only — never surface secret values.
        providers = {
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
            "github": bool(os.getenv("GITHUB_TOKEN")),
        }
        return {
            "tiers": tiers,
            "intent_tier_map": rt.tier_router.intent_tier_map(),
            "providers": providers,
            "paths": {"config_dir": str(rt.config_dir), "data_dir": str(rt.data_dir)},
            "host": {
                "ram_total_gb": report.host.ram_total_gb,
                "ram_free_gb": report.host.ram_free_gb,
                "cpu_percent": report.host.cpu_percent,
                "thermal_throttled": report.host.thermal_throttled,
            },
            "stores": {
                "skill_count": report.iris.skill_count,
                "heartbeat_count": report.iris.heartbeat_count,
                "database_sizes": report.iris.database_sizes,
                "filemanager_roots": report.iris.filemanager_roots,
                "accounts": report.iris.accounts,
            },
            "flags": _flag_payload(_RUNTIME_FLAGS),
        }

    install_playground_routes(app, lambda: _runtime_or_503(app))

    app.include_router(create_mcp_router(bridge))
    # Plugin-contributed routes (OSS plan M5.7 track A slice 4): a capability that left
    # the core keeps its API by registering a router factory in `iris_harness.runtime.api_routes`
    # at plugin setup; the service mounts whatever is registered and imports no plugin.
    _mount_plugin_routes(app)
    return app


def _mount_plugin_routes(app: FastAPI) -> None:
    """Mount every registered plugin router not mounted yet (each key once).

    Plugins register their routers in ``setup()``, which runs when the runtime is
    built — at app build only when a runtime was passed in (tests), otherwise inside
    the lifespan, after ``create_app`` returned. So this runs at both points; a key
    already mounted is skipped, so nothing is mounted twice.
    """
    from iris_harness.runtime.api_routes import registered_api_routers

    mounted: set[str] = app.state.__dict__.setdefault("plugin_route_keys", set())
    for key, factory in registered_api_routers().items():
        if key in mounted:
            continue
        try:
            app.include_router(factory())
            mounted.add(key)
        except Exception:  # one plugin's routes must not take the API down
            logger.exception("plugin API routes %r failed to build; skipped", key)


app = create_app()
