"""``iris doctor`` -- the install preflight (OSS plan R5, milestone L1).

Answers one question for a new install: can IRIS run here, and if not, what to fix. Each
check is pass / warn / fail / info with a one-line fix; the report ends in a verdict:

* ``ready`` -- everything the real assistant needs is in place;
* ``demo_only`` -- the synthetic demo runs (it uses a scripted model, not Ollama), but
  something real use needs is missing: local models, Ollama, or the 16 GB floor;
* ``not_ready`` -- even the demo cannot run (wrong Python, native Windows, under 8 GB,
  no writable IRIS_HOME, or no vault master key -- without it every governed tool call
  is refused, #741).

The states are System Health's (``services/health/models.HealthState``): green = pass,
yellow = warn, red = fail, grey = info. Doctor is not a second health system: health
watches a running install; doctor runs before there is one, and most of what it checks
(the Python range, total RAM, the starter models) health never looks at.

Thresholds and the starter model set come from ``config/doctor.yaml``. Everything that
reaches the machine is injectable, so the report builds deterministically under test.
The only fixes offered are safe ones: pull a missing starter model, and create a master
key when there is none (a stored key is never overwritten -- secrets are encrypted
under it).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml

from iris_harness.foundation.observability.logging_setup import log_egress
from iris_harness.foundation.paths import config_path
from iris_harness.foundation.plugin_dirs import iris_home
from iris_harness.kernel.governance.audit.digest import AuditKeyState, audit_key_status
from iris_harness.kernel.governance.vault import keys as vault_keys
from iris_harness.services.health.models import HealthState

_GIB = 1024**3
_DIST_NAME = "iris-harness"
_OLLAMA_TIMEOUT_S = 3.0
_PULL_TIMEOUT_S = 60.0 * 60

# ── config ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StarterModel:
    name: str
    size_gb: float


@dataclass(frozen=True)
class OptionalExtra:
    name: str
    module: str
    purpose: str


@dataclass(frozen=True)
class DoctorConfig:
    """``config/doctor.yaml``."""

    ram_floor_gb: float
    ram_demo_gb: float
    platforms: dict[str, tuple[str, ...]]
    data_headroom_gb: float
    warn_free_gb: float
    starter_models: tuple[StarterModel, ...]
    optional_extras: tuple[OptionalExtra, ...] = ()


class DoctorConfigError(ValueError):
    """``doctor.yaml`` is missing a value doctor cannot guess."""


def load_doctor_config(path: Path | None = None) -> DoctorConfig:
    """Read ``doctor.yaml`` (the config dir's, unless ``path`` is given).

    No defaults in code: a missing key is an error naming it, so a threshold is only
    ever the one the file states.
    """
    target = path or config_path("doctor.yaml")
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}

    def need(section: str, key: str) -> Any:
        try:
            return raw[section][key]
        except (KeyError, TypeError) as exc:
            raise DoctorConfigError(f"{target}: {section}.{key} is required") from exc

    starters = raw.get("starter_models")
    if not starters:
        raise DoctorConfigError(f"{target}: starter_models is required")
    platforms = raw.get("platforms")
    if not isinstance(platforms, dict) or not platforms:
        raise DoctorConfigError(f"{target}: platforms is required")
    return DoctorConfig(
        ram_floor_gb=float(need("ram", "floor_gb")),
        ram_demo_gb=float(need("ram", "demo_gb")),
        platforms={
            str(system).lower(): tuple(str(m).lower() for m in machines or ())
            for system, machines in platforms.items()
        },
        data_headroom_gb=float(need("disk", "data_headroom_gb")),
        warn_free_gb=float(need("disk", "warn_free_gb")),
        starter_models=tuple(
            StarterModel(name=str(m["name"]), size_gb=float(m["size_gb"])) for m in starters
        ),
        optional_extras=tuple(
            OptionalExtra(
                name=str(e["name"]), module=str(e["module"]), purpose=str(e.get("purpose", ""))
            )
            for e in raw.get("optional_extras") or ()
        ),
    )


# ── the report ──────────────────────────────────────────────────────────────────

#: What a failing check stops: ``all`` (even the demo), ``use`` (real use; the demo
#: still runs) or ``none`` (a warning or information).
Blocks = Literal["all", "use", "none"]

_STATUS = {
    HealthState.GREEN: "pass",
    HealthState.YELLOW: "warn",
    HealthState.RED: "fail",
    HealthState.GREY: "info",
}


@dataclass(frozen=True)
class DoctorCheck:
    """One preflight row: a verdict, what was seen, and the one-line fix."""

    name: str
    state: HealthState
    detail: str
    fix: str | None = None
    blocks: Blocks = "none"

    @property
    def status(self) -> str:
        return _STATUS[self.state]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "state": self.state.value,
            "detail": self.detail,
            "fix": self.fix,
            "blocks": self.blocks,
        }


class Verdict(StrEnum):
    READY = "ready"
    DEMO_ONLY = "demo_only"
    NOT_READY = "not_ready"

    @property
    def exit_code(self) -> int:
        """For scripts and CI: 0 ready, 1 demo only, 2 not ready."""
        return {"ready": 0, "demo_only": 1, "not_ready": 2}[self.value]

    @property
    def headline(self) -> str:
        return {
            "ready": "Ready.",
            "demo_only": "Ready for the demo only.",
            "not_ready": "Not ready.",
        }[self.value]


def verdict_for(checks: Iterable[DoctorCheck]) -> Verdict:
    """``not_ready`` if a failure blocks everything, ``demo_only`` if one blocks real
    use, else ``ready``. Warnings and information never change the verdict."""
    failing = [c for c in checks if c.state is HealthState.RED]
    if any(c.blocks == "all" for c in failing):
        return Verdict.NOT_READY
    if any(c.blocks == "use" for c in failing):
        return Verdict.DEMO_ONLY
    return Verdict.READY


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]
    verdict: Verdict
    #: Starter models Ollama does not have (what ``--fix`` would pull).
    missing_models: tuple[StarterModel, ...] = ()
    #: Where the vault master key stands (what ``--fix`` would do about it).
    key_status: vault_keys.MasterKeyStatus = field(
        default_factory=lambda: vault_keys.MasterKeyStatus("not_read")
    )
    ollama_url: str = ""

    @property
    def fixable(self) -> bool:
        """Something ``--fix`` can do: pull a model or create a missing key."""
        return bool(self.missing_models) or key_needs_fix(self.key_status)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "exit_code": self.verdict.exit_code,
            "checks": [c.as_dict() for c in self.checks],
            "missing_models": [{"name": m.name, "size_gb": m.size_gb} for m in self.missing_models],
            "key": {"source": self.key_status.source, "detail": self.key_status.detail},
            "ollama_url": self.ollama_url,
            "fixable": self.fixable,
        }


def key_needs_fix(status: vault_keys.MasterKeyStatus) -> bool:
    """Doctor can act: no key and nothing in the way (an invalid env key is the owner's;
    an unread keyring is read first by the fix itself)."""
    return status.source in ("absent", "no_keyring", "not_read")


# ── facts ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HostFacts:
    python_version: str
    system: str  # platform.system(), e.g. Darwin / Linux / Windows
    machine: str  # platform.machine(), e.g. arm64 / x86_64
    release: str  # platform.release(); "microsoft" in it = WSL
    ram_total_bytes: int


def host_facts() -> HostFacts:
    import psutil

    return HostFacts(
        python_version=platform.python_version(),
        system=platform.system(),
        machine=platform.machine(),
        release=platform.release(),
        ram_total_bytes=int(psutil.virtual_memory().total),
    )


@dataclass(frozen=True)
class OllamaFacts:
    url: str
    reachable: bool
    version: str | None = None
    models: frozenset[str] = frozenset()
    error: str | None = None


def ollama_root_url() -> str:
    """The Ollama the model calls go to (``OLLAMA_BASE_URL``, else localhost)."""
    from iris_harness.llm.tier_router import provider_root_url

    return provider_root_url("ollama")


def _get_json(client: Any, url: str, *, purpose: str) -> Any:
    log_egress(destination=urlparse(url).netloc, method="GET", kind="llm", purpose=purpose)
    resp = client.get(url, timeout=_OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def ollama_facts(client: Any, root: str) -> OllamaFacts:
    """Ollama's version and pulled models. Two cheap reads; loads no model."""
    try:
        version = _get_json(client, f"{root}/api/version", purpose="doctor").get("version")
        tags = _get_json(client, f"{root}/api/tags", purpose="doctor")
    except Exception as exc:  # noqa: BLE001 — any transport or payload error = unreachable
        return OllamaFacts(url=root, reachable=False, error=str(exc) or type(exc).__name__)
    names = {
        str(m.get("name") or m.get("model"))
        for m in (tags.get("models") or [])
        if isinstance(m, dict) and (m.get("name") or m.get("model"))
    }
    return OllamaFacts(
        url=root, reachable=True, version=str(version) if version else None, models=frozenset(names)
    )


def model_key(name: str) -> str:
    """Ollama's own name for a model: ``qwen2.5`` is ``qwen2.5:latest``."""
    return name if ":" in name else f"{name}:latest"


def disk_free_bytes(path: Path) -> tuple[Path, int]:
    """Free bytes on the volume ``path`` is (or would be) on; walks up to what exists."""
    probe = path.expanduser().resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe, int(shutil.disk_usage(str(probe)).free)


def extra_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def requires_python() -> str | None:
    """The supported Python range from the installed package's metadata (pyproject)."""
    try:
        return importlib.metadata.metadata(_DIST_NAME).get("Requires-Python")
    except importlib.metadata.PackageNotFoundError:
        return None


def configured_ollama_models(path: Path | None = None) -> tuple[str, ...]:
    """Every Ollama model ``llm_tiers.yaml`` names, in file order, deduplicated."""
    target = path or config_path("llm_tiers.yaml")
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return ()
    seen: dict[str, None] = {}
    for cfg in (raw.get("tiers") or {}).values():
        if isinstance(cfg, dict) and cfg.get("provider", "ollama") == "ollama" and cfg.get("model"):
            seen.setdefault(str(cfg["model"]), None)
    return tuple(seen)


# ── checks (pure: facts in, a row out) ─────────────────────────────────────────


def _version_tuple(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in text.strip().split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _satisfies(version: str, spec: str) -> bool:
    """Does ``version`` meet a PEP 440 range like ``>=3.12,<3.14``? (Comparison
    operators only -- what a ``Requires-Python`` uses; no ``packaging`` dependency.)"""
    have = _version_tuple(version)
    for clause in (c.strip() for c in spec.split(",") if c.strip()):
        for op in (">=", "<=", "==", "!=", ">", "<"):
            if clause.startswith(op):
                want = _version_tuple(clause[len(op) :])
                cut = have[: len(want)] if op in ("==", "!=") else have
                ok = {
                    ">=": cut >= want,
                    "<=": cut <= want,
                    ">": cut > want,
                    "<": cut < want,
                    "==": cut == want,
                    "!=": cut != want,
                }[op]
                if not ok:
                    return False
                break
    return True


def check_python(version: str, spec: str | None) -> DoctorCheck:
    if spec is None:
        return DoctorCheck(
            "Python",
            HealthState.YELLOW,
            f"Python {version}; the supported range is unknown (iris-harness is not installed)",
        )
    if _satisfies(version, spec):
        return DoctorCheck("Python", HealthState.GREEN, f"Python {version} (supported: {spec})")
    return DoctorCheck(
        "Python",
        HealthState.RED,
        f"Python {version} is outside the supported range {spec}",
        fix=f"Reinstall IRIS on a supported Python ({spec}): uv tool install --python <version>",
        blocks="all",
    )


def _is_wsl(facts: HostFacts) -> bool:
    return facts.system.lower() == "linux" and "microsoft" in facts.release.lower()


def check_platform(facts: HostFacts, config: DoctorConfig) -> DoctorCheck:
    system, machine = facts.system.lower(), facts.machine.lower()
    label = f"{facts.system} {facts.machine}"
    if system == "windows":
        return DoctorCheck(
            "Platform",
            HealthState.RED,
            f"{label}: native Windows is not supported",
            fix="Install WSL2 (wsl --install) and run IRIS inside it",
            blocks="all",
        )
    if machine in config.platforms.get(system, ()):
        if _is_wsl(facts):
            return DoctorCheck(
                "Platform",
                HealthState.GREEN,
                f"{label} (WSL2: no OS keyring, so the vault key lives in an env var)",
            )
        return DoctorCheck("Platform", HealthState.GREEN, label)
    return DoctorCheck(
        "Platform",
        HealthState.YELLOW,
        f"{label} is not a supported platform (Apple Silicon or Linux); it may still work",
    )


def check_ram(total_bytes: int, config: DoctorConfig) -> DoctorCheck:
    """Total RAM against the floor. Rounded to whole GiB: a 16 GB Linux box reports
    ~15.6 GiB after the kernel's reservations, and it is a 16 GB machine."""
    gib = total_bytes / _GIB
    shown = f"{gib:.1f} GB RAM"
    if round(gib) >= config.ram_floor_gb:
        return DoctorCheck("Memory", HealthState.GREEN, shown)
    if round(gib) >= config.ram_demo_gb:
        return DoctorCheck(
            "Memory",
            HealthState.RED,
            f"{shown}: enough for the demo, below the {config.ram_floor_gb:g} GB "
            "needed for local models",
            fix=f"Use a machine with {config.ram_floor_gb:g} GB or more for real use",
            blocks="use",
        )
    return DoctorCheck(
        "Memory",
        HealthState.RED,
        f"{shown}: below the {config.ram_demo_gb:g} GB minimum",
        fix=f"Use a machine with {config.ram_floor_gb:g} GB or more",
        blocks="all",
    )


def check_disk(
    volume: Path,
    free_bytes: int,
    missing: tuple[StarterModel, ...],
    config: DoctorConfig,
) -> DoctorCheck:
    free_gb = free_bytes / 1e9
    need_gb = sum(m.size_gb for m in missing) + config.data_headroom_gb
    shown = f"{free_gb:.1f} GB free on {volume}"
    if free_gb < need_gb:
        return DoctorCheck(
            "Disk",
            HealthState.RED,
            f"{shown}; {need_gb:.1f} GB needed (missing models + data)",
            fix=f"Free at least {need_gb - free_gb:.1f} GB on that volume",
            blocks="use",
        )
    if free_gb < config.warn_free_gb:
        return DoctorCheck(
            "Disk",
            HealthState.YELLOW,
            f"{shown}; below {config.warn_free_gb:g} GB, mail stores and models grow",
        )
    return DoctorCheck("Disk", HealthState.GREEN, shown)


def check_home(home: Path, writable: Callable[[Path], bool] | None = None) -> DoctorCheck:
    """IRIS_HOME is writable, or can be created (nearest existing parent writable)."""
    is_writable = writable or (lambda p: os.access(p, os.W_OK))
    probe = home.expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if probe.exists() and probe.is_dir() and is_writable(probe):
        created = "" if probe == home.expanduser() else " (will be created)"
        return DoctorCheck("IRIS_HOME", HealthState.GREEN, f"{home} is writable{created}")
    return DoctorCheck(
        "IRIS_HOME",
        HealthState.RED,
        f"{home} is not writable",
        fix="Set IRIS_HOME to a directory you can write to",
        blocks="all",
    )


def check_ollama(facts: OllamaFacts) -> DoctorCheck:
    if not facts.reachable:
        return DoctorCheck(
            "Ollama",
            HealthState.RED,
            f"not reachable at {facts.url} ({facts.error})",
            fix="Install Ollama from https://ollama.com and run `ollama serve` "
            "(or set OLLAMA_BASE_URL)",
            blocks="use",
        )
    version = f"Ollama {facts.version}" if facts.version else "Ollama"
    return DoctorCheck("Ollama", HealthState.GREEN, f"{version} at {facts.url}")


def missing_starter_models(
    facts: OllamaFacts, starter: tuple[StarterModel, ...]
) -> tuple[StarterModel, ...]:
    """Starter models Ollama does not have (none known when it is unreachable)."""
    if not facts.reachable:
        return ()
    have = {model_key(n) for n in facts.models}
    return tuple(m for m in starter if model_key(m.name) not in have)


def check_starter_models(
    facts: OllamaFacts, starter: tuple[StarterModel, ...], missing: tuple[StarterModel, ...]
) -> DoctorCheck:
    names = ", ".join(m.name for m in starter)
    if not facts.reachable:
        return DoctorCheck(
            "Starter models", HealthState.GREY, f"not checked: Ollama is not reachable ({names})"
        )
    if not missing:
        return DoctorCheck("Starter models", HealthState.GREEN, f"pulled: {names}")
    size = sum(m.size_gb for m in missing)
    return DoctorCheck(
        "Starter models",
        HealthState.RED,
        f"missing: {', '.join(m.name for m in missing)} (~{size:.1f} GB to download)",
        fix="iris doctor --fix",
        blocks="use",
    )


def check_tier_models(
    facts: OllamaFacts, configured: tuple[str, ...], starter: tuple[StarterModel, ...]
) -> DoctorCheck | None:
    """Models ``llm_tiers.yaml`` routes to beyond the starter set, when not pulled.

    A warning, not a failure: R5 makes the starter set the install's promise. Until the
    models-by-capability change narrows the tiers to it, intents routed to these tiers
    fail until their models are pulled, and the owner should know that.
    """
    if not facts.reachable:
        return None
    have = {model_key(n) for n in facts.models} | {model_key(m.name) for m in starter}
    absent = [m for m in configured if model_key(m) not in have]
    if not absent:
        return None
    return DoctorCheck(
        "Tier models",
        HealthState.YELLOW,
        f"llm_tiers.yaml also routes to {len(absent)} model(s) not pulled: {', '.join(absent)}",
        fix=" && ".join(f"ollama pull {m}" for m in absent),
    )


def check_vault_key(
    status: vault_keys.MasterKeyStatus, audit_state: AuditKeyState = "unresolved"
) -> DoctorCheck:
    """The master key every governed tool call needs (#741).

    ``audit_state`` is this process's own answer (``audit_key_status``): a process that
    already resolved the key (the API) needs no further look.
    """
    if audit_state == "ready" or status.present:
        where = "resolved by this process" if audit_state == "ready" else status.detail
        return DoctorCheck("Vault key", HealthState.GREEN, f"master key present ({where})")
    if status.source == "invalid":
        return DoctorCheck(
            "Vault key",
            HealthState.RED,
            status.detail,
            fix="Unset IRIS_VAULT_MASTER_KEY or set it to the valid Fernet key your vault uses",
            blocks="all",
        )
    if status.source == "not_read" and audit_state != "unavailable":
        return DoctorCheck(
            "Vault key",
            HealthState.YELLOW,
            "an OS keyring is present but was not read (non-interactive run)",
            fix="Run `iris doctor` in a terminal, or set IRIS_VAULT_MASTER_KEY",
        )
    how = "prints an export line" if status.source == "no_keyring" else "stores one in the keyring"
    return DoctorCheck(
        "Vault key",
        HealthState.RED,
        f"no master key: governed tool calls are refused ({status.detail})",
        fix=f"iris doctor --fix  ({how})",
        blocks="all",
    )


def check_extra(extra: OptionalExtra, installed: bool) -> DoctorCheck:
    state = (
        "installed" if installed else f'not installed (pip install "iris-harness[{extra.name}]")'
    )
    return DoctorCheck(f"Extra: {extra.name}", HealthState.GREY, f"{extra.purpose}: {state}")


# ── the run ─────────────────────────────────────────────────────────────────────


def http_client() -> Any:
    """The HTTP client doctor talks to Ollama with (one place for a test to replace)."""
    import httpx

    return httpx.Client()


def run_doctor(
    *,
    read_keyring: bool,
    config: DoctorConfig | None = None,
    client: Any = None,
    home: Path | None = None,
    audit_state: AuditKeyState | None = None,
) -> DoctorReport:
    """Build the preflight report. Reads only; installs, pulls and stores nothing.

    ``read_keyring=False`` keeps the OS keyring closed (a Keychain dialog must not open
    in a script or a server). ``client`` is an ``httpx.Client``-like object, else
    :func:`http_client`. Each fact comes from one module-level reader (``host_facts``,
    ``disk_free_bytes``, ``requires_python``, ...), which is what a test replaces.
    """
    cfg = config or load_doctor_config()
    facts = host_facts()
    home_dir = home or iris_home()
    spec = requires_python()
    if audit_state is None:
        audit_state = audit_key_status()[0]

    root = ollama_root_url()
    if client is None:
        with http_client() as own:
            ollama = ollama_facts(own, root)
    else:
        ollama = ollama_facts(client, root)
    missing = missing_starter_models(ollama, cfg.starter_models)
    key_status = vault_keys.master_key_status(read_keyring=read_keyring)
    volume, free = disk_free_bytes(home_dir)

    checks: list[DoctorCheck] = [
        check_python(facts.python_version, spec),
        check_platform(facts, cfg),
        check_ram(facts.ram_total_bytes, cfg),
        check_disk(volume, free, missing, cfg),
        check_home(home_dir),
        check_vault_key(key_status, audit_state),
        check_ollama(ollama),
        check_starter_models(ollama, cfg.starter_models, missing),
    ]
    tier_row = check_tier_models(
        ollama,
        configured_ollama_models(),
        cfg.starter_models,
    )
    if tier_row is not None:
        checks.append(tier_row)
    checks.extend(check_extra(e, extra_installed(e.module)) for e in cfg.optional_extras)
    return DoctorReport(
        checks=tuple(checks),
        verdict=verdict_for(checks),
        missing_models=missing,
        key_status=key_status,
        ollama_url=root,
    )


# ── safe fixes ──────────────────────────────────────────────────────────────────


class DoctorFixError(RuntimeError):
    """A fix could not complete; the message says why."""


@dataclass(frozen=True)
class PullProgress:
    status: str
    completed: int | None = None
    total: int | None = None


def pull_model(
    name: str,
    *,
    client: Any,
    root: str | None = None,
    on_progress: Callable[[PullProgress], None] | None = None,
) -> None:
    """Pull ``name`` through Ollama's ``/api/pull`` stream. Raises :class:`DoctorFixError`."""
    base = root or ollama_root_url()
    url = f"{base}/api/pull"
    log_egress(destination=urlparse(url).netloc, method="POST", kind="llm", purpose="doctor-pull")
    try:
        with client.stream(
            "POST", url, json={"model": name, "stream": True}, timeout=_PULL_TIMEOUT_S
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise DoctorFixError(f"pull {name}: HTTP {resp.status_code} {resp.text[:200]}")
            last = ""
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise DoctorFixError(f"pull {name}: {event['error']}")
                last = str(event.get("status", ""))
                if on_progress is not None:
                    on_progress(PullProgress(last, event.get("completed"), event.get("total")))
    except DoctorFixError:
        raise
    except Exception as exc:  # transport/JSON errors become one fix error
        raise DoctorFixError(f"pull {name}: {exc}") from exc
    if last != "success":
        raise DoctorFixError(f"pull {name}: the stream ended without success ({last or 'empty'})")


KeyFixOutcome = Literal["kept", "stored", "export", "invalid"]

#: Where a key printed for an env var should be kept (no keyring on this host).
PERSIST_HINT = (
    "No OS keyring here (WSL2, headless Linux, containers), so the key lives in an env "
    "var. Add the export line to your shell profile (~/.bashrc, ~/.zshrc or ~/.profile) "
    "and to the environment of whatever runs IRIS (a systemd Environment= line, Docker "
    "-e or env_file). Keep a copy somewhere safe: vault secrets cannot be read without it."
)


@dataclass(frozen=True)
class KeyFix:
    outcome: KeyFixOutcome
    detail: str
    export_line: str | None = None


def fix_master_key(
    *,
    generate: Callable[[], str] = vault_keys.generate_master_key,
) -> KeyFix:
    """Make sure a master key exists, never replacing one.

    Reads the keyring (the owner asked for the fix, so a Keychain prompt here is theirs
    to answer). A key already there, in the env or the keyring, is kept. With a working
    keyring a new key is stored in it; without one the key is returned as an export line
    for the owner to persist (:data:`PERSIST_HINT`).
    """
    status = vault_keys.master_key_status(read_keyring=True)
    if status.present:
        return KeyFix("kept", f"a master key is already set ({status.detail}); left as is")
    if status.source == "invalid":
        return KeyFix("invalid", status.detail)
    key = generate()
    if status.source == "absent":
        try:
            vault_keys.store_master_key_in_keyring(key)
        except vault_keys.MasterKeyExistsError:
            return KeyFix("kept", "a master key appeared in the OS keyring; left as is")
        except vault_keys.VaultMasterKeyUnavailableError:
            pass  # the keyring refused the write: fall through to the export line
        else:
            return KeyFix("stored", "a new master key is stored in the OS keyring")
    return KeyFix(
        "export",
        PERSIST_HINT,
        export_line=f"export {vault_keys.MASTER_KEY_ENV}={key}",
    )


__all__ = [
    "PERSIST_HINT",
    "DoctorCheck",
    "DoctorConfig",
    "DoctorConfigError",
    "DoctorFixError",
    "DoctorReport",
    "HostFacts",
    "KeyFix",
    "OllamaFacts",
    "OptionalExtra",
    "PullProgress",
    "StarterModel",
    "Verdict",
    "check_disk",
    "check_extra",
    "check_home",
    "check_ollama",
    "check_platform",
    "check_python",
    "check_ram",
    "check_starter_models",
    "check_tier_models",
    "check_vault_key",
    "fix_master_key",
    "http_client",
    "key_needs_fix",
    "load_doctor_config",
    "missing_starter_models",
    "ollama_facts",
    "pull_model",
    "run_doctor",
    "verdict_for",
]
