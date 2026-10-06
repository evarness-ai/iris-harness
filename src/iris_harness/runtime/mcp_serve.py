"""``iris mcp serve``, governed: every tool call an MCP client makes is a governed call.

The protocol is ``tools/mcp/server.py``'s. What it serves, and how a call runs, is here:

**What is served.** The registered tool catalogue -- the tools plugins register with
``api.register_tool``, each carrying its manifest declaration (effect, confirm, content,
``sends_to``, ``executes_code``, describe/validate). That catalogue is the stable surface
(OSS plan R16): a third-party plugin's tools are served the same way as the core's. With
no ``--tool`` only the *contained* reads are served (owner's decision, 2026-09-30): a
``read`` tool whose arguments go nowhere (no ``sends_to``), whose output is not text a
third party wrote (``content`` not ``external``) and that runs no code
(``executes_code`` unset). ``research`` and ``code_exec`` -- and the email reads -- are
served only when named with ``--tool``, which names exactly which (any effect --
governance still decides each call). Skill packages (``config/skills``) are served only
when named with ``--skill``: a skill manifest declares none of those properties, so no
skill tool can be shown to be contained. Even then only its tools whose manifest route is
``<domain>/read``: a skill manifest cannot declare a write's confirm, approval or undo,
so a skill write could not be governed as its kind requires and is not served.

**Where the result may go.** The owner's MCP serve config (``config/governance/
mcp-serve.yaml``, overlaid by ``$IRIS_HOME/mcp-serve.yaml``) lists the clients and
whether each runs on the owner's machine (``local``, default false). ``--client`` must
name one. ``POST_TOOL_USE``'s ``McpClientEgressHook`` withholds a result labelled
``personal`` from a client not declared local, and a ``secret`` one from every client.

**How a call runs.** Through ``ToolService.call_for_client`` -- the same
``GovernedToolRunner`` the loop and ``api.tools`` use: the keyed-digest audit key check
(no vault master key, nothing runs), ``PRE_TOOL_USE`` (caller policy, tool policy,
approvals, credential broker, owner-PII guards), the call, ``POST_TOOL_USE`` whose verdict
is enforced (a withheld result is answered as an MCP error; a rewritten one is what the
client receives), and the audit rows.

**Who the caller is.** ``mcp:<client>``, where ``<client>`` is the name the operator gave
``iris mcp serve --client`` (default ``stdio``) -- in the client's own config, which the
owner controls, and one the owner's MCP serve config lists. The ``clientInfo`` a client
sends at ``initialize`` is a claim: it is logged, and nothing is decided by it.

**What cannot run.** A destructive tool, a pinned write (``approval: pinned``) and a
``confirm: once`` write need the owner's answer, which an MCP client cannot give. They are
refused by governance at ``PRE_TOOL_USE`` (audited, answered as an MCP error), not queued:
a queued call would run later, out of the client's sight, its result never reaching it.

**The label.** One served session (one stdio connection) is one turn for the label: what a
result earns (``POST_TOOL_USE``'s classifier) lifts the session's label, and every later
call is governed at that floor -- so a client that read personal data cannot then send it
to a web search as if it were public.

**The log.** Each call is an ingress (``INGRESS MCP tools/call``) and its answer an
egress to the client (``EGRESS mcp``): tool names, caller and outcome; never arguments or
results.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import yaml

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.foundation.observability.logging_setup import log_egress, log_ingress
from iris_harness.foundation.observability.session_log import session_scope
from iris_harness.foundation.paths import config_dir, default_config_dir, iris_home
from iris_harness.kernel.governance.caller_policy import MCP_CALLER_PREFIX
from iris_harness.kernel.governance.mcp_clients import register_local_mcp_clients
from iris_harness.kernel.governance.turn_label import turn_label_scope
from iris_harness.tools.mcp.server import (
    IrisMCPServer,
    McpReply,
    McpTool,
    SkillTool,
    load_skill_tools,
    serve_stdio,
)

if TYPE_CHECKING:
    from iris_harness.runtime.tool_service import ToolService

#: The client name when the operator gives none.
DEFAULT_CLIENT = "stdio"
_CLIENT_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

# A plugin tool takes free-form arguments (``args.get("query")`` is the one every tool in
# the tree reads); a ToolSpec carries no schema, so the client gets this honest floor.
_PLUGIN_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The request to pass to this tool, in full."}
    },
    "additionalProperties": True,
}


class McpServeError(ValueError):
    """The operator asked for something ``iris mcp serve`` cannot serve."""


def mcp_caller(client: str) -> str:
    """``mcp:<client>`` for an operator-given client name (lowercase, ``[a-z0-9._-]``)."""
    if not _CLIENT_NAME.match(client):
        raise McpServeError(
            f"--client {client!r}: use 1-64 lowercase letters, digits, '.', '_' or '-'"
        )
    return f"{MCP_CALLER_PREFIX}{client}"


#: The owner's MCP serve config: shipped under ``config/governance/``, overlaid by
#: ``$IRIS_HOME/mcp-serve.yaml`` (the file an installed IRIS's owner edits).
CLIENTS_FILE = ("governance", "mcp-serve.yaml")
HOME_OVERLAY = "mcp-serve.yaml"


def _read_clients(path: Path) -> dict[str, bool]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    clients = raw.get("clients") if isinstance(raw, dict) else None
    if clients is None:
        return {}
    if not isinstance(clients, dict):
        raise McpServeError(f"{path}: `clients` must map a client name to its settings")
    out: dict[str, bool] = {}
    for name, settings in clients.items():
        mcp_caller(str(name))  # the same name rule as --client
        settings = settings or {}
        if not isinstance(settings, dict) or set(settings) - {"local"}:
            raise McpServeError(f"{path}: client {name!r} takes only `local: true|false`")
        local = settings.get("local", False)
        if not isinstance(local, bool):
            raise McpServeError(f"{path}: client {name!r}: `local` must be true or false")
        out[str(name)] = local
    return out


def load_mcp_clients(*, config: Path | None = None, home: Path | None = None) -> dict[str, bool]:
    """``{client name: local}`` from the owner's MCP serve config.

    The shipped file (``config_dir()``, else the defaults when an override directory has
    none), then ``<home>/mcp-serve.yaml``, whose entries win. Unreadable: an error, never
    an empty (or a wider) list.
    """
    shipped = config or config_dir().joinpath(*CLIENTS_FILE)
    if config is None and not shipped.exists():
        shipped = default_config_dir().joinpath(*CLIENTS_FILE)
    overlay = (home or iris_home()) / HOME_OVERLAY
    clients: dict[str, bool] = {}
    for path in (shipped, overlay):
        if not path.exists():
            continue
        try:
            clients |= _read_clients(path)
        except (OSError, yaml.YAMLError) as exc:
            raise McpServeError(f"{path}: unreadable ({exc})") from exc
    return clients


def client_is_local(client: str, clients: Mapping[str, bool]) -> bool:
    """Whether the owner declared ``client`` local; an unlisted client is an error."""
    mcp_caller(client)
    if client not in clients:
        listed = ", ".join(sorted(clients)) or "none"
        raise McpServeError(
            f"--client {client!r} is not in the owner's MCP serve config (listed: {listed}); "
            f"add it under `clients:` in $IRIS_HOME/{HOME_OVERLAY}"
        )
    return clients[client]


def contained(spec: ToolSpec) -> bool:
    """Whether ``spec`` is served without being named: a read whose arguments go nowhere,
    whose output no third party wrote, and that runs no code."""
    return (
        spec.effect == "read"
        and spec.sends_to is None
        and spec.content != "external"
        and not spec.executes_code
    )


@dataclass(frozen=True)
class ServedTool:
    """One tool as served: its governed spec, and the schema the client sees."""

    spec: ToolSpec
    input_schema: dict[str, Any]
    source: str  # "plugin" or "skill:<name>"


@dataclass(frozen=True)
class Selection:
    """What ``select_tools`` chose, and what it left out and why (for the operator)."""

    served: tuple[ServedTool, ...]
    skipped: tuple[str, ...]


def _skill_spec(tool: SkillTool) -> ToolSpec:
    def call(args: dict[str, Any]) -> str:
        result = tool.instance.invoke(args or {})
        return result if isinstance(result, str) else str(result)

    # Served only when its route declares a read (``select_tools``).
    return ToolSpec(
        tool.name,
        tool.description,
        call,
        effect="read",
        confirm="never",
        plugin=f"skill:{tool.skill}",
    )


def select_tools(
    catalogue: Sequence[ToolSpec],
    *,
    tools: Sequence[str] = (),
    skills: Sequence[str] | None = None,
    skill_tools: Sequence[SkillTool] | None = None,
) -> Selection:
    """The tools ``iris mcp serve`` exposes (module docstring, "What is served").

    ``catalogue`` is the registered tools. ``tools`` names catalogue tools to serve; with
    none, every ``contained`` tool in it is. ``skills`` names skill packages (none unless
    named); ``skill_tools`` is what was loaded for them (loaded here when not given). A
    name nobody registered is an error, not a silent omission.
    """
    by_name = {spec.name: spec for spec in catalogue}
    unknown = [name for name in tools if name not in by_name]
    if unknown:
        raise McpServeError(f"no registered tool named {', '.join(map(repr, unknown))}")
    skipped: list[str] = []
    if tools:
        chosen = [by_name[name] for name in dict.fromkeys(tools)]
    else:
        chosen = [spec for spec in catalogue if contained(spec)]
        skipped += [
            f"tool {spec.name!r}: not a contained read (name it with --tool to serve it)"
            for spec in catalogue
            if not contained(spec)
        ]
    served = [ServedTool(spec, dict(_PLUGIN_TOOL_SCHEMA), "plugin") for spec in chosen]

    wanted_skills = tuple(skills or ())
    loaded = list(skill_tools) if skill_tools is not None else load_skill_tools(wanted_skills)
    names = {tool.spec.name for tool in served}
    for skill_tool in loaded:
        where = f"skill {skill_tool.skill!r} tool {skill_tool.name!r}"
        if skill_tool.verb != "read":
            route = skill_tool.governor_route or "none"
            skipped.append(f"{where}: its route ({route}) is not a read")
        elif skill_tool.name in names:
            skipped.append(f"{where}: a registered tool has the same name")
        else:
            names.add(skill_tool.name)
            served.append(
                ServedTool(
                    _skill_spec(skill_tool),
                    skill_tool.input_schema,
                    f"skill:{skill_tool.skill}",
                )
            )
    return Selection(served=tuple(served), skipped=tuple(skipped))


def governed_mcp_tools(
    service: ToolService, caller: str, served: Sequence[ServedTool]
) -> list[McpTool]:
    """Each served tool as an ``McpTool`` whose every call is governed as ``caller``."""

    def invoker(tool: ServedTool) -> Any:
        def invoke(arguments: dict[str, Any]) -> McpReply:
            started = time.monotonic()
            result = service.call_for_client(caller, tool.spec, arguments)
            if result.ok:
                outcome = "ok"
            elif result.held:
                outcome = "refused"
            else:
                outcome = "error"
            elapsed = (time.monotonic() - started) * 1000
            log_ingress(
                method="MCP",
                path=f"tools/call {tool.spec.name}",
                source=caller,
                status=200 if result.ok else 403 if result.held else 500,
                duration_ms=elapsed,
            )
            # The answer leaves IRIS for the client: what went where, never the text.
            log_egress(
                destination=caller,
                method="RESULT",
                kind="mcp",
                purpose="tools/call",
                status=outcome,
                tool=tool.spec.name,
                chars=len(result.text),
            )
            return McpReply(result.text, is_error=not result.ok)

        return invoke

    return [
        McpTool(
            name=tool.spec.name,
            description=tool.spec.description,
            input_schema=tool.input_schema,
            invoke=invoker(tool),
        )
        for tool in served
    ]


@contextmanager
def mcp_session(caller: str) -> Iterator[str]:
    """One served connection: its own session id, and one label for all its calls."""
    session_id = f"mcp-{uuid.uuid4().hex[:12]}"
    with session_scope(session_id), turn_label_scope():
        log_ingress(method="MCP", path="session/open", source=caller, session_id=session_id)
        try:
            yield session_id
        finally:
            log_ingress(method="MCP", path="session/close", source=caller, session_id=session_id)


def serve(
    service: ToolService,
    served: Sequence[ServedTool],
    *,
    client: str = DEFAULT_CLIENT,
    local: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> IrisMCPServer:
    """Serve ``served`` over stdio until stdin closes, every call governed as the client.

    ``local``: what the owner's MCP serve config declares for ``client``
    (``client_is_local``). The kernel's ``McpClientEgressHook`` reads it for the life of
    the server; it is withdrawn when the server stops.
    """
    caller = mcp_caller(client)
    server = IrisMCPServer(tools=governed_mcp_tools(service, caller, served))
    register_local_mcp_clients([caller] if local else [])
    try:
        with mcp_session(caller):
            serve_stdio(server, stdin=stdin, stdout=stdout)
    finally:
        register_local_mcp_clients([])
    return server


__all__ = [
    "CLIENTS_FILE",
    "DEFAULT_CLIENT",
    "HOME_OVERLAY",
    "McpServeError",
    "Selection",
    "ServedTool",
    "client_is_local",
    "contained",
    "governed_mcp_tools",
    "load_mcp_clients",
    "mcp_caller",
    "mcp_session",
    "select_tools",
    "serve",
]
