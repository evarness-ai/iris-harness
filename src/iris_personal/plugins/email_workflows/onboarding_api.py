"""``/api/v1/email/onboarding`` -- email setup's API (OSS plan R4).

The same state machine ``iris email setup`` drives (``onboarding.py``); these routes only
translate. The web Setup screen (L2) is built on them::

    GET  /api/v1/email/onboarding
         -> {"steps": [{"step", "title"}], "setups": [state, ...], "accounts": [id, ...]}
    GET  /api/v1/email/onboarding/{account_id}                 -> state (404: none yet)
    POST /api/v1/email/onboarding/{account_id}/advance
         {"create_master_key"?, "accept_categories"?, "accept_defaults"?, "approve_writes"?,
          "until_waiting"?, "actor"?}                            -> state
    GET  /api/v1/email/onboarding/{account_id}/label-preview   -> preview
    POST /api/v1/email/onboarding/{account_id}/approve-writes  {"approve", "actor"?} -> state
    POST /api/v1/email/onboarding/{account_id}/restart         -> {"restarted": bool}

The overview also carries ``connect_hints`` (each provider's login command, for an owner
with no mailbox yet) and ``demo_account`` (the synthetic mailbox's id inside a demo home,
else null). A state is ``OnboardingState.as_dict`` plus ``sweep``
(``Onboarding.sweep_status``: does the scheduled sweep take the account yet, and why),
``connect_command`` (the account's provider login, "" when it has none) and
``rendered`` (each finished step in plain lines, the words the CLI prints). Its fields:
``step`` (the current one, ``complete`` at the end), ``status`` (``in_progress`` /
``waiting`` / ``done``), ``waiting_kind`` (``decision``: the owner decides;
``blocked``: something outside setup first), ``waiting_for``, ``steps`` and each
finished step's ``results``.

``accept_defaults`` is ``--yes``: every decision's default except mailbox writes, which
only ``approve_writes: true`` (or ``approve-writes`` with ``approve: true``) approves.
``approve-writes`` answers step 6 only (409 at any other step).

Every POST here is a gated write (OSS plan R17): the console's write guard lets a
control-paired device or ``IRIS_WEBUI_ALLOW_WRITES`` through, exactly as it does the
Action Center's answer to the same approval (``/governance/approvals/{id}/respond``).
Setup's confirmed action -- the mailbox-write approval -- is a row on the governed
approval queue either way; opening these routes to a read-only console would let it
answer that row here when it cannot in the Action Center.

Long steps (the first fetch, discovery, the judge) run inside the request.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .onboarding import (
    STEPS,
    Inputs,
    Onboarding,
    OnboardingDeps,
    OnboardingError,
    candidate_accounts,
    step_body,
)

PATH = "/api/v1/email/onboarding"
_ACTOR_MAX = 80


class AdvanceBody(BaseModel):
    create_master_key: bool | None = None
    accept_categories: list[int] | None = Field(default=None, max_length=500)
    accept_defaults: bool = False
    approve_writes: bool | None = None
    until_waiting: bool = False
    actor: str | None = Field(default=None, max_length=_ACTOR_MAX)


class ApproveBody(BaseModel):
    approve: bool
    actor: str | None = Field(default=None, max_length=_ACTOR_MAX)


def build_router(deps: Callable[[], OnboardingDeps]) -> APIRouter:
    """The router; ``deps`` is called per request (services may fill in after setup)."""
    router = APIRouter(tags=["email"])

    def _machine() -> Onboarding:
        return Onboarding(deps())

    def _view(machine: Onboarding, state: Any) -> dict[str, Any]:
        return {
            **state.as_dict(machine.config),
            "sweep": machine.sweep_status(state.account_id),
            "connect_command": machine.connect_command(state.account_id),
            "rendered": {step: step_body(step, r) for step, r in state.results.items()},
        }

    def _state_or_404(machine: Onboarding, account_id: str) -> Any:
        state = machine.state(account_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"no email setup for {account_id}")
        return state

    @router.get(PATH)
    def overview() -> dict[str, Any]:
        machine = _machine()
        return {
            "steps": [{"step": s, "title": machine.config.titles[s]} for s in STEPS],
            "setups": [_view(machine, s) for s in machine.states()],
            "accounts": candidate_accounts(machine.deps),
            "connect_hints": machine.connect_hints(),
            "demo_account": machine.demo_account(),
        }

    @router.get(PATH + "/{account_id}")
    def get_state(account_id: str) -> dict[str, Any]:
        machine = _machine()
        return _view(machine, _state_or_404(machine, account_id))

    @router.post(PATH + "/{account_id}/advance")
    def advance(account_id: str, body: AdvanceBody) -> dict[str, Any]:
        machine = _machine()
        inputs = Inputs(
            assume_defaults=body.accept_defaults,
            create_master_key=body.create_master_key,
            accept_categories=(
                tuple(body.accept_categories) if body.accept_categories is not None else None
            ),
            approve_writes=body.approve_writes,
            actor=body.actor or "web",
            channel="web",
        )
        try:
            state = (machine.run if body.until_waiting else machine.advance)(account_id, inputs)
        except OnboardingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _view(machine, state)

    @router.get(PATH + "/{account_id}/label-preview")
    def label_preview(account_id: str) -> dict[str, Any]:
        machine = _machine()
        _state_or_404(machine, account_id)
        return machine.label_preview(account_id).as_dict(machine.config)

    @router.post(PATH + "/{account_id}/approve-writes")
    def approve_writes(account_id: str, body: ApproveBody) -> dict[str, Any]:
        machine = _machine()
        state = _state_or_404(machine, account_id)
        if state.step != "label_approval":
            raise HTTPException(
                status_code=409,
                detail=f"setup is at {state.step!r}, not the label approval",
            )
        inputs = Inputs(approve_writes=body.approve, actor=body.actor or "web", channel="web")
        return _view(machine, machine.advance(account_id, inputs))

    @router.post(PATH + "/{account_id}/restart")
    def restart(account_id: str) -> dict[str, Any]:
        return {"restarted": _machine().restart(account_id)}

    return router


def runtime_deps(services: Any) -> OnboardingDeps:
    """Setup's calls from the runtime's services: its data and config dirs, its tier
    router's ``email_judge`` tier (the judge and the category namer) and its narrative
    call. A server never reads the OS keyring (no Keychain dialog)."""
    from iris_harness.sdk.llm import make_narrative_llm_call

    from .discovery import GovernedNamingClient
    from .judge import llm_from_router

    router = getattr(services, "tier_router", None)
    data_dir: Path = services.data_dir

    def judge() -> Any:
        return llm_from_router(router)

    def namer() -> Any:
        call = llm_from_router(router)
        return GovernedNamingClient(call) if call is not None else None

    def narrate() -> Any:
        return make_narrative_llm_call(router) if router is not None else None

    return OnboardingDeps(
        data_dir=data_dir,
        config_dir=services.config_dir,
        judge_llm=judge,
        naming_client=namer,
        narrate=narrate,
        read_keyring=False,
    )


def register(api: Any) -> None:
    """Mount the routes on the API service (``plugin.setup``). The services are read per
    request: the host may fill them in after setup."""
    api.register_api_router(
        "email_onboarding", lambda: build_router(lambda: runtime_deps(api.services))
    )


__all__ = ["PATH", "build_router", "register", "runtime_deps"]
