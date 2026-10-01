"""Non-interactive output modes: print, json, stdio (JSON-RPC).

These modes are used when IRIS is driven programmatically or when
a one-shot response is needed (e.g. scripting, piping, or external
tool integration via the --stdio JSON-RPC bridge).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from iris_harness.cli.api_client import harness_urlopen
from iris_harness.foundation.auth import auth_headers

from .render import (
    FooterMetrics,
    print_error,
    print_response,
    render_footer,
    sample_system_metrics,
    spinner,
)
from .session import Session, SessionManager
from .welcome import fetch_welcome


def _post(
    message: str,
    session_id: str,
    api_url: str,
    *,
    strict: bool = False,
) -> dict[str, object]:
    payload = json.dumps({"message": message, "session_id": session_id, "strict": strict}).encode()
    req = urllib.request.Request(  # noqa: S310
        f"{api_url}/chat",
        data=payload,
        headers={"Content-Type": "application/json", **auth_headers()},
        method="POST",
    )
    with harness_urlopen(req, purpose="chat", timeout=120) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def run_print_mode(
    message: str,
    *,
    session: Session,
    session_manager: SessionManager,
    api_url: str,
    strict: bool = False,
) -> int:
    # A first chat on this install opens with IRIS's welcome (ADR-0127), printed ahead of
    # the answer. Print mode only: json and stdio are machine protocols, one reply per
    # request, and an unasked-for welcome would break them.
    welcome = fetch_welcome(api_url, channel="console")
    if welcome:
        print_response(welcome)
    try:
        with spinner("thinking…"):
            body = _post(message, session.id, api_url, strict=strict)
    except OSError as exc:
        print_error(f"Cannot reach IRIS API at {api_url} — is the server running?\n  {exc}")
        return 1
    except urllib.error.HTTPError as exc:
        print_error(f"HTTP {exc.code}: {exc.read().decode()}")
        return 1

    session_manager.touch(session)
    meta: dict[str, object] = body.get("metadata", {}) or {}  # type: ignore[assignment]
    sys_metrics = sample_system_metrics()
    print_response(
        str(body.get("response", "")),
        intent=str(body.get("intent", "")),
        agent=str(body.get("agent_type", "")),
        has_errors=bool(body.get("has_errors", False)),
    )
    render_footer(
        FooterMetrics(
            prompt_tokens=int(str(meta.get("prompt_tokens") or 0)),
            completion_tokens=int(str(meta.get("completion_tokens") or 0)),
            model=str(meta.get("model", "")),
            latency_ms=float(meta.get("total_latency_ms", 0) or 0),  # type: ignore[arg-type]
            cpu_pct=sys_metrics.get("cpu_pct", -1.0),
            mem_used_gb=sys_metrics.get("mem_used_gb", -1.0),
            mem_total_gb=sys_metrics.get("mem_total_gb", -1.0),
            gpu_pct=sys_metrics.get("gpu_pct", -1.0),
            gpu_mem_used_gb=sys_metrics.get("gpu_mem_used_gb", -1.0),
            gpu_mem_total_gb=sys_metrics.get("gpu_mem_total_gb", -1.0),
        )
    )
    return 1 if body.get("has_errors") else 0


def run_json_mode(
    message: str,
    *,
    session: Session,
    session_manager: SessionManager,
    api_url: str,
    strict: bool = False,
) -> int:
    try:
        body = _post(message, session.id, api_url, strict=strict)
    except OSError as exc:
        sys.stdout.write(json.dumps({"error": str(exc)}) + "\n")
        sys.stdout.flush()
        return 1
    except urllib.error.HTTPError as exc:
        sys.stdout.write(json.dumps({"error": f"HTTP {exc.code}"}) + "\n")
        sys.stdout.flush()
        return 1

    session_manager.touch(session)
    sys.stdout.write(json.dumps(body, indent=2) + "\n")
    sys.stdout.flush()
    return 1 if body.get("has_errors") else 0


def run_stdio_mode(*, api_url: str, session_manager: SessionManager, strict: bool = False) -> int:
    """JSON-RPC over stdin/stdout — one JSON object per line in each direction.

    Request format:
        {"id": "1", "message": "hello", "session_id": "optional-id"}

    Response format:
        {"id": "1", "session_id": "...", "response": "...", "intent": "...",
         "agent_type": "...", "has_errors": false, "metadata": {...}}

    This is the portability bridge: any external tool (scripts, Claude Code,
    other agents) can drive IRIS by reading/writing newline-delimited JSON.
    """
    session_cache: dict[str, Session] = {}

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        try:
            req = json.loads(raw)
        except json.JSONDecodeError as exc:
            _write({"error": f"invalid JSON: {exc}"})
            continue

        req_id = req.get("id", "")
        message = str(req.get("message", ""))
        session_key = str(req.get("session_id", "default"))

        if not message:
            _write({"id": req_id, "error": "message is required"})
            continue

        if session_key not in session_cache:
            existing = session_manager.load(session_key)
            session_cache[session_key] = existing or session_manager.create()

        sess = session_cache[session_key]

        try:
            body = _post(message, sess.id, api_url, strict=strict)
        except Exception as exc:  # noqa: BLE001
            _write({"id": req_id, "error": str(exc)})
            continue

        session_manager.touch(sess)
        _write(
            {
                "id": req_id,
                "session_id": sess.id,
                "response": body.get("response", ""),
                "intent": body.get("intent", ""),
                "agent_type": body.get("agent_type", ""),
                "has_errors": body.get("has_errors", False),
                "metadata": body.get("metadata", {}),
            }
        )

    return 0


def _write(obj: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()
