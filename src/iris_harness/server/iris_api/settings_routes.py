"""Routes that change settings from the app, and the history of those changes (ADR-0120).

    GET    /heartbeat                     every heartbeat, disabled ones included
    PATCH  /heartbeat/{name}              change its schedule and/or turn it on or off
    DELETE /heartbeat/{name}/override     back to the schedule the deploy ships
    GET    /settings/history              what changed, when, and from which device
    GET    /settings/catalog              every IRIS_* setting, described, with its value
    PATCH  /settings/{name}               change one (guarded ones need confirm: true)
    DELETE /settings/{name}               back to the deploy's value
    GET    /models                        every tier (live and file values) and intent map
    PATCH  /models/tiers/{name}           change a tier's fields (a route change: confirm)
    DELETE /models/tiers/{name}/override  back to llm_tiers.yaml
    PATCH  /models/intents/{intent}       run an intent on another tier (confirm if the
                                          route changes)
    DELETE /models/intents/{intent}/override
    GET    /health/watch/config           the health watch's editable fields, file values
    PATCH  /health/watch/config           change them; the running watcher uses them at
                                          its next tick
    DELETE /health/watch/config           back to health_watch.yaml
    GET    /digest/config                 the morning digest's settings, file values,
                                          every section it knows, IRIS_TZ
    PATCH  /digest/config                 change them; the next digest uses them
    DELETE /digest/config                 back to digest.yaml
    GET    /system/restart                supervised? which saved changes wait for one
    POST   /system/restart                restart every server (confirm: true)

The capability is the heartbeat scheduler (``update`` / ``reset``) and the settings
store under it; these routes and ``/heartbeats set|reset`` in the REPL are surfaces.

Who may change: the write guard in ``main`` gates every PATCH and DELETE, so a paired
``control`` device may, a ``read`` device may not, and the shared secret only when the
operator opted in. Nothing here re-checks that; a route that did would be a second
policy to keep in step with the first.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from iris_harness.foundation.auth import Principal
from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.foundation.settings.catalog import (
    NEVER_EDITABLE,
    Catalog,
    CatalogEntry,
    SettingDeclaration,
    build_catalog,
    load_core_catalog,
)
from iris_harness.foundation.settings.env_overrides import (
    ENV_SECTION,
    SettingValueError,
    clear_env_override,
    deploy_value,
    normalize,
    set_env_override,
)
from iris_harness.foundation.settings.restart import (
    PROCESS_STARTED_AT,
    request_restart,
    supervised,
)
from iris_harness.kernel.governance.devices import DeviceService
from iris_harness.runtime.plugin_host.profile import DISABLE_ENV
from iris_harness.services.heartbeat import HeartbeatDefinition
from iris_harness.services.heartbeat.schedule_text import describe_schedule

_HISTORY_LIMIT_MAX = 500


class SettingPatch(BaseModel):
    """Body of ``PATCH /settings/{name}``. ``confirm`` is the owner's yes to a guarded
    change; the app asks for it, and the API refuses a guarded change without it."""

    model_config = ConfigDict(extra="forbid")

    value: str | bool | int | float
    confirm: bool = False


class TierPatch(BaseModel):
    """Body of ``PATCH /models/tiers/{name}``: any of the editable fields."""

    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    timeout_seconds: int | None = None
    think: bool | None = None
    confirm: bool = False


class IntentMove(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: str = Field(min_length=1, max_length=80)
    confirm: bool = False


class RestartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: bool = False


class HeartbeatPatch(BaseModel):
    """Body of ``PATCH /heartbeat/{name}``: send one field or both."""

    model_config = ConfigDict(extra="forbid")

    schedule: str | None = Field(default=None, min_length=1, max_length=120)
    enabled: bool | None = None


def actor_of(request: Request) -> str:
    """Who is making this change, as the history records it."""
    principal = getattr(request.state, "principal", None)
    if isinstance(principal, Principal) and principal.kind == "device":
        return f"device:{principal.device_id}"
    return "service"


def heartbeat_payload(heartbeats: Any, definition: HeartbeatDefinition) -> dict[str, Any]:
    """One heartbeat as the app, the REPL and the chat tool see it."""
    declared = heartbeats.declared(definition.name) or definition
    next_run: datetime | None = heartbeats.next_run_at(definition.name)
    return {
        "name": definition.name,
        "schedule": definition.schedule,
        "schedule_text": describe_schedule(definition.schedule),
        "enabled": definition.enabled,
        "description": definition.description,
        "default_schedule": declared.schedule,
        "default_enabled": declared.enabled,
        "overridden": (definition.schedule, definition.enabled)
        != (declared.schedule, declared.enabled),
        "runnable": heartbeats.unavailable_reason(definition) is None,
        "unavailable_reason": heartbeats.unavailable_reason(definition),
        "platforms": list(definition.platforms),
        "next_run_at": next_run.isoformat() if next_run is not None else None,
        # Loop-proof D13, from the runs kept in heartbeat_runs.db (they survive a
        # restart): the last run, the last success, and whether the most recent slot
        # ran — "Missed 12:15 — last success 06:15". None where runs are not kept.
        **_run_fields(heartbeats, definition),
    }


def _run_fields(heartbeats: Any, definition: HeartbeatDefinition) -> dict[str, Any]:
    job_status = getattr(heartbeats, "job_status", None)
    if job_status is None:
        return {"last_run": None, "last_success_at": None, "job": None}
    from iris_harness.services.health.jobs import grace_for, load_watched_jobs

    watched = any(job.name == definition.name for job in load_watched_jobs())
    try:
        status = job_status(definition.name, grace=grace_for(definition.name))
    except Exception:  # noqa: BLE001 — a broken run store must not hide the heartbeats
        status = None
    if status is None:
        return {"last_run": None, "last_success_at": None, "job": None, "watched": watched}
    job = status.as_dict()
    return {
        "watched": watched,
        "last_run": job["last_run"],
        "last_success_at": job["last_success_at"],
        "job": {k: v for k, v in job.items() if k not in ("last_run", "last_success_at")},
    }


def models_payload(router: Any) -> dict[str, Any]:
    """Every tier (what runs now, and the file's value for each edited field) and where
    each intent runs, for the Models tab."""
    from iris_harness.llm.tier_edits import EDITABLE_FIELDS, PROVIDERS

    tiers = []
    for name, tier in router._tiers.items():  # read the router's own state
        declared = router._declared_tiers.get(name, tier)
        tiers.append(
            {
                "name": name,
                "title": tier.name,
                **{f: getattr(tier, f) for f in EDITABLE_FIELDS},
                "use_for": list(tier.use_for),
                "file": {f: getattr(declared, f) for f in EDITABLE_FIELDS},
                "changed": [f for f in EDITABLE_FIELDS if getattr(tier, f) != getattr(declared, f)],
            }
        )
    intents = router.intent_tier_map()
    declared_intents = dict(router._declared_intents)
    return {
        "tiers": tiers,
        "intents": intents,
        "moved_intents": sorted(i for i, t in intents.items() if declared_intents.get(i) != t),
        "providers": list(PROVIDERS),
    }


def locked_plugins(rt: Any) -> dict[str, str]:
    """Plugins whose manifest says the app may not turn them off, with the reason."""
    registry = getattr(rt, "plugin_registry", None)
    return {
        record.manifest.name: record.manifest.locked
        for record in (registry.plugins() if registry is not None else [])
        if record.manifest is not None and record.manifest.locked
    }


SidecarCatalogs = Sequence[tuple[str, dict[str, SettingDeclaration]]]


def runtime_catalog(rt: Any, sidecars: SidecarCatalogs = ()) -> Catalog:
    """The core catalog, every loaded plugin's declarations, and each sidecar's
    (``load_sidecar_catalogs``: what the deployment says runs beside the harness)."""
    registry = getattr(rt, "plugin_registry", None)
    plugins = [
        (f"plugin:{record.manifest.name}", dict(record.manifest.settings))
        for record in (registry.plugins() if registry is not None else [])
        if record.manifest is not None
    ]
    return build_catalog(load_core_catalog(), [*plugins, *sidecars])


def catalog_payload(entry: CatalogEntry, overridden: bool = False) -> dict[str, Any]:
    """One setting for the app. The value is shown only for plain settings: a secret,
    path or URL reports whether it is set, never what it is."""
    decl = entry.declaration
    raw = os.environ.get(entry.name)
    plain = decl.kind not in NEVER_EDITABLE
    return {
        "name": entry.name,
        "owner": entry.owner,
        **decl.model_dump(),
        "is_set": raw is not None,
        "value": raw if plain else None,
        # ADR-0120: changed from the app (saved on the data volume), and what the
        # deploy's own environment says underneath.
        "overridden": overridden,
        "deploy_value": deploy_value(entry.name) if plain else None,
    }


def install_settings_routes(
    app: FastAPI,
    runtime: Callable[[], Any],
    devices: Callable[[], DeviceService],
    *,
    sidecar_catalogs: SidecarCatalogs = (),
) -> None:
    """Register the edit + history routes. ``runtime`` returns the live runtime or
    raises 503; ``devices`` names the device behind an actor in the history;
    ``sidecar_catalogs`` are the settings of the processes beside the harness, listed
    and never changed from the app."""
    sidecar_owners = {owner for owner, _ in sidecar_catalogs}

    def _heartbeats() -> Any:
        return runtime().heartbeats

    def _one(name: str) -> dict[str, Any]:
        heartbeats = _heartbeats()
        found = next((d for d in heartbeats.all_definitions() if d.name == name), None)
        if found is None:
            raise HTTPException(status_code=404, detail=f"heartbeat '{name}' not found")
        return heartbeat_payload(heartbeats, found)

    @app.patch("/heartbeat/{name}")
    def patch_heartbeat(name: str, body: HeartbeatPatch, request: Request) -> dict[str, Any]:
        if body.schedule is None and body.enabled is None:
            raise HTTPException(status_code=422, detail="send 'schedule', 'enabled', or both")
        try:
            _heartbeats().update(
                name, schedule=body.schedule, enabled=body.enabled, actor=actor_of(request)
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"heartbeat '{name}' not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _one(name)

    @app.delete("/heartbeat/{name}/override")
    def reset_heartbeat(name: str, request: Request) -> dict[str, Any]:
        try:
            _heartbeats().reset(name, actor=actor_of(request))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"heartbeat '{name}' not found") from exc
        return _one(name)

    def _store() -> SettingsStore:
        return SettingsStore(db_path=runtime().data_dir / SETTINGS_DB_NAME)

    def _editable(name: str) -> CatalogEntry:
        entry = runtime_catalog(runtime(), sidecar_catalogs).get(name)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"no setting {name!r} in the catalog")
        if not entry.declaration.editable:
            raise HTTPException(
                status_code=403,
                detail=f"{name} is not changed from the app: "
                f"{entry.declaration.not_editable_reason}",
            )
        if entry.owner in sidecar_owners:
            raise HTTPException(
                status_code=409,
                detail=f"{name} belongs to the {entry.owner} process, which does not read "
                "app changes; set it in that process's environment",
            )
        return entry

    @app.get("/settings/catalog")
    def settings_catalog(tab: str | None = None) -> dict[str, Any]:
        catalog = runtime_catalog(runtime(), sidecar_catalogs)
        saved = _store().section(ENV_SECTION)
        rows = [catalog_payload(e, e.name in saved) for e in catalog.entries.values()]
        if tab is not None:
            rows = [r for r in rows if r["tab"] == tab]
        return {"count": len(rows), "settings": rows}

    @app.patch("/settings/{name}")
    def patch_setting(name: str, body: SettingPatch, request: Request) -> dict[str, Any]:
        entry = _editable(name)
        if entry.declaration.guarded and not body.confirm:
            raise HTTPException(
                status_code=409,
                detail=f"{name} is guarded ({entry.declaration.guard_reason}); "
                "confirm the change and send confirm: true",
            )
        try:
            value = normalize(entry.declaration, body.value)
        except SettingValueError as exc:
            raise HTTPException(status_code=422, detail=f"{name}: {exc}") from exc
        if name == DISABLE_ENV:
            locked = locked_plugins(runtime())
            refused = [p for p in value.split(",") if p in locked]
            if refused:
                raise HTTPException(
                    status_code=409,
                    detail="; ".join(
                        f"{p} cannot be turned off here: {locked[p]}" for p in refused
                    ),
                )
        if value != os.environ.get(name):
            set_env_override(name, value, actor=actor_of(request), store=_store())
        return {
            **catalog_payload(entry, overridden=True),
            "restart_required": entry.declaration.applies == "restart",
        }

    @app.delete("/settings/{name}")
    def reset_setting(name: str, request: Request) -> dict[str, Any]:
        entry = _editable(name)
        change = clear_env_override(name, actor=actor_of(request), store=_store())
        return {
            **catalog_payload(entry, overridden=False),
            "restart_required": change is not None and entry.declaration.applies == "restart",
        }

    def _router() -> Any:
        return runtime().tier_router

    @app.get("/models")
    def get_models() -> dict[str, Any]:
        return models_payload(_router())

    @app.patch("/models/tiers/{name}")
    def patch_tier(name: str, body: TierPatch, request: Request) -> dict[str, Any]:
        from iris_harness.llm import tier_edits

        changes = body.model_dump(exclude={"confirm"}, exclude_none=True)
        if not changes:
            raise HTTPException(status_code=422, detail="send at least one field to change")
        router = _router()
        try:
            if tier_edits.is_guarded_tier_change(router, name, changes) and not body.confirm:
                raise HTTPException(
                    status_code=409,
                    detail="changing a tier's provider changes where its prompts go; "
                    "confirm the change and send confirm: true",
                )
            tier_edits.update_tier(router, _store(), name, changes, actor=actor_of(request))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no tier {name!r}") from exc
        except tier_edits.TierEditError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return models_payload(router)

    @app.delete("/models/tiers/{name}/override")
    def reset_tier(name: str, request: Request) -> dict[str, Any]:
        from iris_harness.llm import tier_edits

        router = _router()
        try:
            tier_edits.reset_tier(router, _store(), name, actor=actor_of(request))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no tier {name!r}") from exc
        return models_payload(router)

    @app.patch("/models/intents/{intent}")
    def move_intent(intent: str, body: IntentMove, request: Request) -> dict[str, Any]:
        from iris_harness.llm import tier_edits

        router = _router()
        if tier_edits.is_guarded_move(router, intent, body.tier) and not body.confirm:
            raise HTTPException(
                status_code=409,
                detail=f"{body.tier} is on a different route, so {intent} prompts would go "
                "somewhere else; confirm the change and send confirm: true",
            )
        try:
            tier_edits.move_intent(router, _store(), intent, body.tier, actor=actor_of(request))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no such {exc.args[0]!r}") from exc
        return models_payload(router)

    @app.delete("/models/intents/{intent}/override")
    def reset_intent(intent: str, request: Request) -> dict[str, Any]:
        from iris_harness.llm import tier_edits

        router = _router()
        try:
            tier_edits.reset_intent(router, _store(), intent, actor=actor_of(request))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no intent {intent!r}") from exc
        return models_payload(router)

    def _watch_state() -> tuple[Any, Any, Any]:
        """(file config, current config, running watcher or None)."""
        from iris_harness.services.health.watch import (
            current_watcher,
            load_watch_config,
        )

        config_dir = getattr(runtime(), "config_dir", None)
        watcher = current_watcher()
        file_config = load_watch_config(config_dir, with_edits=False)
        current = (
            watcher.config
            if watcher is not None
            else load_watch_config(config_dir, settings=_store())
        )
        return file_config, current, watcher

    def _watch_payload(file_config: Any, current: Any, watcher: Any) -> dict[str, Any]:
        from iris_harness.services.health.watch_edits import EDITABLE, fields

        now, was = fields(current), fields(file_config)
        return {
            "fields": [
                {
                    "name": f,
                    "kind": kind,
                    "min": low if kind != "bool" else None,
                    "max": high if kind != "bool" else None,
                    "value": now[f],
                    "file": was[f],
                    "changed": now[f] != was[f],
                }
                for f, (kind, low, high) in EDITABLE.items()
            ],
            "enabled": current.enabled,
            "running": watcher is not None,
        }

    @app.get("/health/watch/config")
    def get_watch_config() -> dict[str, Any]:
        return _watch_payload(*_watch_state())

    @app.patch("/health/watch/config")
    def patch_watch_config(body: dict[str, Any], request: Request) -> dict[str, Any]:
        from iris_harness.services.health import watch_edits

        if not body:
            raise HTTPException(status_code=422, detail="send at least one field to change")
        file_config, current, watcher = _watch_state()
        try:
            new = watch_edits.update(file_config, current, _store(), body, actor=actor_of(request))
        except watch_edits.WatchEditError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if watcher is not None:
            watcher.config = new  # read on every tick: applies at the next one
        return _watch_payload(file_config, new, watcher)

    @app.delete("/health/watch/config")
    def reset_watch_config(request: Request) -> dict[str, Any]:
        from iris_harness.services.health import watch_edits

        file_config, current, watcher = _watch_state()
        new = watch_edits.reset(file_config, current, _store(), actor=actor_of(request))
        if watcher is not None:
            watcher.config = new
        return _watch_payload(file_config, new, watcher)

    def _digest_dirs() -> tuple[Path, Path | None]:
        rt = runtime()
        return rt.data_dir, getattr(rt, "config_dir", None)

    @app.get("/digest/config")
    def get_digest_config() -> dict[str, Any]:
        from iris_harness.services.digest import edits as digest_edits

        data_dir, config_dir = _digest_dirs()
        return digest_edits.payload(data_dir, config_dir)

    @app.patch("/digest/config")
    def patch_digest_config(body: dict[str, Any], request: Request) -> dict[str, Any]:
        from iris_harness.services.digest import edits as digest_edits

        data_dir, config_dir = _digest_dirs()
        try:
            digest_edits.update(data_dir, body, actor=actor_of(request), config_dir=config_dir)
        except digest_edits.DigestEditError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return digest_edits.payload(data_dir, config_dir)

    @app.delete("/digest/config")
    def reset_digest_config(request: Request) -> dict[str, Any]:
        from iris_harness.services.digest import edits as digest_edits

        data_dir, config_dir = _digest_dirs()
        digest_edits.reset(data_dir, actor=actor_of(request), config_dir=config_dir)
        return digest_edits.payload(data_dir, config_dir)

    @app.get("/system/restart")
    def restart_status() -> dict[str, Any]:
        """Whether the app may restart this harness, and which saved changes wait for it."""
        catalog = runtime_catalog(runtime(), sidecar_catalogs)
        store = _store()
        waiting = sorted(
            {
                change.key
                for change in store.history(section=ENV_SECTION, limit=0)
                if change.at > PROCESS_STARTED_AT
                and (entry := catalog.get(change.key)) is not None
                and entry.declaration.applies == "restart"
            }
            # A tier edit applies to the next turn at once; helpers that built their
            # client at startup (the curator's judges, the email summariser) need one.
            | {
                f"llm_tiers:{change.key}"
                for change in store.history(section="llm_tiers", limit=0)
                if change.at > PROCESS_STARTED_AT
            }
        )
        return {
            "supervised": supervised(),
            "started_at": PROCESS_STARTED_AT.isoformat(),
            "waiting_for_restart": waiting,
        }

    @app.post("/system/restart")
    def restart(body: RestartRequest, request: Request) -> dict[str, Any]:
        if not supervised():
            raise HTTPException(
                status_code=409,
                detail="nothing would bring this harness back (IRIS_SUPERVISED is not set); "
                "restart it yourself",
            )
        if not body.confirm:
            raise HTTPException(
                status_code=409, detail="restarting pauses chat for about a minute; confirm it"
            )
        at = request_restart(_store(), actor=actor_of(request))
        return {"restarting": True, "requested_at": at.isoformat()}

    @app.get("/settings/history")
    def settings_history(section: str | None = None, limit: int = 100) -> dict[str, Any]:
        limit = max(1, min(limit, _HISTORY_LIMIT_MAX))
        store = SettingsStore(db_path=runtime().data_dir / SETTINGS_DB_NAME)
        changes = [change.as_dict() for change in store.history(section=section, limit=limit)]
        names: dict[str, str | None] = {}
        for change in changes:
            actor = str(change["actor"])
            if actor not in names:
                names[actor] = _actor_name(actor, devices)
            change["actor_name"] = names[actor]
        return {"count": len(changes), "changes": changes}


def _actor_name(actor: str, devices: Callable[[], DeviceService]) -> str | None:
    """The device's name for ``device:<id>``, looked up now so a rename shows."""
    if actor == "service":
        return "API secret"
    if not actor.startswith("device:"):
        return None
    try:
        row = devices().get(actor.split(":", 1)[1])
    except Exception:  # noqa: BLE001 — a name is a nicety; the id is the record
        return None
    if row is None:
        return None
    return f"{row.name} (revoked)" if row.revoked else row.name
