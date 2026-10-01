"""A real, governed IRIS in a throwaway home, for a plugin's tests and the examples.

Stable tier (OSS plan R16, L2). :func:`harness` builds the runtime the servers and the CLI
build -- ``build_runtime``, the composition root, not a look-alike -- so a chat turn runs
the whole pipeline: the PRE_TURN screen, routing, the loop, the curator, PRE_RESPONSE,
the audit ledger. Only the edges are replaced:

* **the home** -- a fresh ``IRIS_HOME`` (its own data dir, governance stores and audit
  ledger); nothing is read from or written to the owner's profile;
* **the environment** -- every ``IRIS_*`` setting and every variable that looks like a
  credential is dropped for the harness's lifetime and the process environment is put
  back exactly on exit, so the owner's ``.env`` flags cannot change what a test sees;
* **the model** -- every tier on the scripted fake (:func:`use_fake_model`); with no
  script, any model call fails loudly instead of reaching a model server;
* **the vault master key** -- a throwaway key in the environment; the OS keyring is
  swapped for one that refuses every call, so the owner's Keychain is never read;
* **the network** -- refused (:func:`no_network`) for the harness's lifetime.

Plugins are supplied in-process: :func:`plugin` wraps a ``setup`` callable and its
manifest, and the harness mounts it after the profile's own plugins, through the same
loader, fault boundary and manifest checks as an installed one -- no entry point, no
packaging.

The harness answers the R14 question directly: :meth:`Harness.audit_gaps` lists every
model call without an audit row and every answer without a PRE_RESPONSE row.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iris_harness.foundation.process_state import (
    restore_process_state,
    snapshot_process_state,
)
from iris_harness.kernel.governance.audit.log import AuditLog, AuditRow
from iris_harness.llm.fake import FakeCall, Script, transcript
from iris_harness.runtime.plugin_host.loader import InProcessPlugin, in_process_plugin
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.runtime.plugin_host.registry import PluginStatus

if TYPE_CHECKING:
    from iris_harness.runtime.facade import IrisRuntime
    from iris_harness.runtime.types import ChatResult, StreamEvent
    from iris_harness.sdk import PluginAPI

# The hook points whose rows answer "was this model call audited" and "was this answer".
LLM_CALL_HOOK = "pre_llm_call"
ANSWER_HOOK = "pre_response"

_CREDENTIAL_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


def plugin(
    setup: Callable[[PluginAPI], None] | None = None,
    *,
    name: str | None = None,
    manifest: PluginManifest | Mapping[str, Any] | Path | str | None = None,
) -> InProcessPlugin:
    """A plugin for :func:`harness`, supplied as its ``setup`` callable.

    ``manifest`` is what the plugin's ``manifest.yaml`` would say -- a mapping of that
    shape or the file's path -- and is checked as that file is: a tool the plugin
    registers must be declared under ``tools:``. ``None`` is the minimal manifest (just
    ``name``), enough for a plugin that registers no tool. A ``flavor: declarative``
    plugin is its manifest alone: pass ``manifest`` and no ``setup``, and the harness binds
    its tools as the loader does for an installed one.
    """
    return in_process_plugin(setup, name=name, manifest=manifest)


@dataclass(frozen=True)
class TurnEvent:
    """One step of a streamed turn, in the order the pipeline yielded it.

    ``kind`` is ``"token"`` (answer text as it is produced), ``"activity"`` (a short
    status line), ``"trace"`` (a pipeline step, for inspection), ``"done"`` (the turn
    finished; ``text`` is the final answer) or ``"error"`` (the turn failed; ``text``
    says why). ``done`` or ``error`` is always last.
    """

    kind: str
    text: str


@dataclass(frozen=True)
class TurnResult:
    """What one harness turn produced.

    ``text`` is the answer (empty when the turn failed), ``agent`` the agent that
    answered and ``intent`` what the router classified the message as. ``answered`` is
    false only when the turn ended without an answer (a streamed turn's terminal
    ``error``); ``error`` then says why -- and on an answered turn, what went wrong on
    the way, if anything. ``audit_refs`` are the ids of the audit-ledger rows the turn
    wrote in its session, oldest first (see :meth:`Harness.audit_rows`). ``events`` is
    every step of a :meth:`Harness.chat_stream` turn; empty for :meth:`Harness.chat`.
    """

    text: str
    agent: str
    intent: str
    sources: tuple[str, ...]
    answered: bool
    error: str | None
    session_id: str
    audit_refs: tuple[int, ...]
    events: tuple[TurnEvent, ...] = ()


@dataclass(frozen=True)
class TurnAuditRow:
    """One row of the harness's audit ledger, as the stable tier promises it.

    The ledger's own row type is internal (it carries the raw payload, whose keys belong
    to each governance check); this is the part a plugin's test may rely on.

    ``hook_point`` is where governance ran (``"pre_turn"``, ``"pre_llm_call"``,
    ``"pre_tool_use"``, ``"post_tool_use"``, ``"pre_response"``, ...); ``plugin`` is the
    governance check that wrote the row (``"egress_gate"``, ``"response_safety"``, ...)
    and ``decision`` what it decided (``"allow"``, ``"deny"``, ...), with ``reason``.
    ``run_id`` and ``step_id`` identify the call: each model call of a loop run is its
    own ``step_id``. ``classification`` is the data label the check saw (``"public"``,
    ``"personal"``, ...) and ``tier`` the model tier a model call was bound for, when the
    row is about one. ``session_id`` is the chat session the row belongs to, ``tool`` the
    tool a tool-use row is about, and ``deterministic`` / ``handler`` mark an answer a
    deterministic handler gave (``register_intercept``), with that handler's name.
    """

    id: int
    created_at: datetime
    hook_point: str
    plugin: str
    decision: str
    reason: str
    run_id: str
    step_id: int | None
    classification: str | None
    tier: str | None
    session_id: str | None
    tool: str | None
    deterministic: bool
    handler: str | None


def _payload(row: AuditRow) -> dict[str, Any]:
    try:
        payload = json.loads(row.payload_json)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _turn_audit_row(row: AuditRow) -> TurnAuditRow:
    payload = _payload(row)
    return TurnAuditRow(
        id=row.id,
        created_at=datetime.fromisoformat(row.ts),
        hook_point=row.hook_point,
        plugin=row.plugin,
        decision=row.decision,
        reason=row.reason,
        run_id=row.run_id,
        step_id=row.step_id,
        classification=row.classification,
        tier=row.tier,
        session_id=_optional_str(payload.get("session_id")),
        tool=_optional_str(payload.get("tool_name")),
        deterministic=payload.get("deterministic") is True,
        handler=_optional_str(payload.get("handler")),
    )


@dataclass(frozen=True)
class TurnRecord:
    """One turn the harness ran: what was asked, in which session, and its result."""

    message: str
    session_id: str
    result: TurnResult


def _event(event: StreamEvent) -> TurnEvent:
    if event.kind == "done":
        return TurnEvent(kind="done", text=event.result.response if event.result else "")
    if event.kind == "error":
        return TurnEvent(kind="error", text=event.error or "")
    return TurnEvent(kind=event.kind, text=event.text)


def _turn_result(
    result: ChatResult | None,
    *,
    session_id: str,
    audit_refs: tuple[int, ...],
    events: tuple[TurnEvent, ...] = (),
    failure: str | None = None,
) -> TurnResult:
    if result is None:
        return TurnResult(
            text="",
            agent="",
            intent="",
            sources=(),
            answered=False,
            error=failure or "the turn ended without an answer",
            session_id=session_id,
            audit_refs=audit_refs,
            events=events,
        )
    error = (result.error_summary or "the turn reported an error") if result.has_errors else None
    return TurnResult(
        text=result.response,
        agent=result.agent_type,
        intent=result.intent,
        sources=tuple(result.sources),
        answered=True,
        error=error,
        session_id=session_id,
        audit_refs=audit_refs,
        events=events,
    )


class Harness:
    """A built runtime in its own home. Get one from :func:`harness`."""

    def __init__(self, runtime: IrisRuntime, home: Path, started: datetime) -> None:
        self._runtime = runtime
        self._home = home
        self._started = started
        self._turns: list[TurnRecord] = []

    @property
    def home(self) -> Path:
        """The harness's ``IRIS_HOME``; every store it writes is under it."""
        return self._home

    @property
    def data_dir(self) -> Path:
        return self._home / "data"

    @property
    def audit_db(self) -> Path:
        return self._home / "governance" / "audit.db"

    @property
    def turns(self) -> tuple[TurnRecord, ...]:
        return tuple(self._turns)

    def _session(self, session_id: str | None) -> str:
        return session_id or f"harness-{len(self._turns) + 1}"

    def _audit_high_water(self) -> int:
        rows = AuditLog(db_path=self.audit_db).query(since=self._started)
        return max((row.id for row in rows), default=0)

    def _audit_refs(self, session_id: str, after: int) -> tuple[int, ...]:
        return tuple(row.id for row in self.audit_rows(session_id=session_id) if row.id > after)

    def chat(self, message: str, *, session_id: str | None = None, **kwargs: Any) -> TurnResult:
        """One turn through the full pipeline (``IrisRuntime.chat``); returns its result.

        Each turn gets its own session unless ``session_id`` is given, so its audit rows
        can be told apart. Extra keywords (``channel``, ``audience``, ...) pass through.
        """
        session = self._session(session_id)
        mark = self._audit_high_water()
        answer = self._runtime.chat(message, session_id=session, **kwargs)
        result = _turn_result(
            answer, session_id=session, audit_refs=self._audit_refs(session, mark)
        )
        self._turns.append(TurnRecord(message=message, session_id=session, result=result))
        return result

    def chat_stream(
        self, message: str, *, session_id: str | None = None, **kwargs: Any
    ) -> TurnResult:
        """One streamed turn (``IrisRuntime.chat_stream``, the REPL's and the web's path),
        drained: the result carries every event, the terminal ``done`` or ``error`` last."""
        session = self._session(session_id)
        mark = self._audit_high_water()
        raw = list(self._runtime.chat_stream(message, session_id=session, **kwargs))
        final = raw[-1] if raw else None
        answer = final.result if final is not None and final.kind == "done" else None
        failure = final.error if final is not None and final.kind == "error" else None
        result = _turn_result(
            answer,
            session_id=session,
            audit_refs=self._audit_refs(session, mark),
            events=tuple(_event(e) for e in raw),
            failure=failure,
        )
        self._turns.append(TurnRecord(message=message, session_id=session, result=result))
        return result

    def respond_to_approval(
        self, approval_id: str, *, approve: bool, actor: str = "harness-owner"
    ) -> str:
        """Answer a pending approval as the owner would, and let the run go on.

        What the Governance screen's Approve / Reject buttons do: the answer is recorded
        (and audited) first; then a turn halted for it resumes from its checkpoint -- the
        approved call runs, governed again -- and a call a plugin's code queued runs
        once. Returns what the owner is told: the resumed turn's answer, or why nothing
        continued. The pending ids are on ``ApprovalQueue().list_pending()``
        (``iris_harness.sdk.approvals``).
        """
        from iris_harness.kernel.governance.approvals.service import (
            respond_to_approval as respond,
        )

        outcome = respond(
            approval_id,
            status="approved" if approve else "rejected",
            actor=actor,
            resumer=self._runtime,
            executor=self._runtime.tool_service,
        )
        return str(outcome.detail)

    def plugins(self) -> dict[str, tuple[str, str | None]]:
        """Every plugin the harness tried to mount: ``name -> (status, load error)``."""
        return {
            record.name: (record.status.value, record.load_error)
            for record in self._runtime.plugin_registry.plugins()
        }

    def plugin_loaded(self, name: str) -> bool:
        return self.plugins().get(name, ("", None))[0] == PluginStatus.LOADED.value

    def model_calls(self) -> tuple[FakeCall, ...]:
        """Every call the scripted model answered since the harness started."""
        return transcript()

    def audit_rows(
        self, *, hook_point: str | None = None, session_id: str | None = None
    ) -> list[TurnAuditRow]:
        """The harness ledger's rows since it started, oldest first, optionally narrowed
        to one hook point (``"pre_llm_call"``, ``"pre_response"``, ...) and one session."""
        ledger = AuditLog(db_path=self.audit_db).query(since=self._started)
        rows = [_turn_audit_row(row) for row in ledger]
        return [
            row
            for row in rows
            if (hook_point is None or row.hook_point == hook_point)
            and (session_id is None or row.session_id == session_id)
        ]

    def audit_gaps(self) -> list[str]:
        """What the R14 invariant finds missing; empty when every model call and every
        answer of the harness's turns has its audit row.

        * a model call: each governed call fires ``PRE_LLM_CALL`` once, under its own
          ``(run_id, step_id)`` -- one row per hook registered there -- before the
          transport runs, so there must be at least one key per call the scripted model
          answered. More keys than answers is not a gap: a call governance refused, or
          one the transport failed, is audited and never answered;
        * an answer: each turn that produced one has a ``PRE_RESPONSE`` row in its
          session.
        """
        gaps: list[str] = []
        calls = self.model_calls()
        keys = {(row.run_id, row.step_id) for row in self.audit_rows(hook_point=LLM_CALL_HOOK)}
        if len(keys) < len(calls):
            gaps.append(
                f"{len(calls)} model call(s) answered, {len(keys)} audited at {LLM_CALL_HOOK}"
            )
        for turn in self._turns:
            if not turn.result.answered:
                continue
            if not self.audit_rows(hook_point=ANSWER_HOOK, session_id=turn.session_id):
                gaps.append(f"no {ANSWER_HOOK} row for the answer to {turn.message!r}")
        return gaps


def _looks_like_credential(name: str) -> bool:
    upper = name.upper()
    return any(word in upper for word in _CREDENTIAL_WORDS)


@contextmanager
def _environment(values: Mapping[str, str]) -> Iterator[None]:
    """The process environment minus the owner's IRIS settings and credentials, plus
    ``values``; restored exactly on exit."""
    saved = dict(os.environ)
    for name in list(os.environ):
        if name.startswith("IRIS_") or _looks_like_credential(name):
            del os.environ[name]
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@contextmanager
def _refusing_keyring() -> Iterator[None]:
    """Swap the OS keyring for one that refuses every call; put the old one back."""
    try:
        import keyring
        from keyring.backends import fail
    except Exception:  # noqa: BLE001 -- no keyring installed: nothing to protect
        yield
        return
    previous = keyring.get_keyring()
    keyring.set_keyring(fail.Keyring())  # type: ignore[no-untyped-call]  # keyring is untyped
    try:
        yield
    finally:
        keyring.set_keyring(previous)


def _throwaway_master_key() -> str:
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode("ascii")


@contextmanager
def harness(
    *,
    profile: str = "minimal",
    plugins: Sequence[InProcessPlugin] = (),
    fake_model: Script | Mapping[str, Any] | Path | str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    config_dir: Path | None = None,
    embeddings: bool = False,
) -> Iterator[Harness]:
    """Build a governed IRIS in a throwaway home and yield it; tear it down on exit.

    ``profile`` is a shipped profile name (``minimal``, ``default``, ``email``, ...);
    ``plugins`` are mounted after its own (see :func:`plugin`). ``fake_model`` is the
    scripted model every tier runs on (a :class:`Script`, a mapping or a YAML path).
    ``env`` sets ``IRIS_*`` flags for the run -- applied after the scrub, so they are the
    only ones set. ``home`` is an empty directory to use (a fresh temporary one, removed
    on exit, by default). ``embeddings=False`` stubs the embedding model, as the suite
    does, so the build loads no ~80 MB model.
    """
    from iris_harness.runtime import build_runtime
    from iris_harness.testing import no_network, use_fake_model

    with ExitStack() as stack:
        if home is None:
            home = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="iris-harness-")))
        home = home.resolve()
        if home.exists() and any(home.iterdir()):
            raise ValueError(f"harness home {home} is not empty")
        (home / "data").mkdir(parents=True, exist_ok=True)
        values = {
            "IRIS_HOME": str(home),
            "IRIS_DATA_DIR": str(home / "data"),
            "IRIS_GOVERNANCE_AUDIT_DB_PATH": str(home / "governance" / "audit.db"),
            "IRIS_VAULT_MASTER_KEY": _throwaway_master_key(),
            "IRIS_PROFILE": profile,
            "IRIS_DISABLE_WARMUP": "1",
            # The resource arbiter polls a model server over HTTP.
            "IRIS_DISABLE_ARBITER": "1",
            # Anything the run starts can't reach the Keychain either.
            "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring",
        }
        if not embeddings:
            values["IRIS_TEST_NULL_EMBEDDINGS"] = "1"
        values.update(env or {})
        stack.enter_context(_environment(values))
        stack.enter_context(_refusing_keyring())
        stack.enter_context(no_network())
        stack.enter_context(use_fake_model(fake_model if fake_model is not None else Script(())))
        started = datetime.now(UTC)
        # Building the runtime, starting it and mounting plugins fill process-wide
        # registries and caches: mail providers, API routes, health checks and the health
        # watcher, the process bus's subscribers, the identity-redaction seams, config
        # caches read from this home... Each owner declares its state with
        # foundation/process_state.py; all of it is put back on exit, after the runtime
        # shut down, so a later harness -- or anything else in the process -- never sees
        # this run's plugins, or a store in a home that is gone.
        stack.callback(restore_process_state, snapshot_process_state())
        runtime = build_runtime(
            config_dir=config_dir,
            data_dir=home / "data",
            use_background_scheduler=False,
            in_process_plugins=plugins,
        )
        stack.callback(runtime.shutdown)
        runtime.startup()
        yield Harness(runtime, home, started)


__all__ = [
    "ANSWER_HOOK",
    "LLM_CALL_HOOK",
    "Harness",
    "TurnAuditRow",
    "TurnEvent",
    "TurnRecord",
    "TurnResult",
    "harness",
    "plugin",
]
