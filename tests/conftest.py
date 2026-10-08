"""Shared test bootstrap configuration."""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
import tempfile
import threading
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# CRITICAL test isolation — strip the developer's IRIS_* feature flags from the
# environment BEFORE anything resolves them. A real IRIS install exports dozens of
# them (IRIS_CALENDAR_AUTO_APPROVE_INVITES, IRIS_INTENT_ROUTER_SEMANTIC,
# IRIS_LEARNING_ANALYST/BEHAVIOR_MINER/INTENTION_ROLLUP, IRIS_CALENDAR_APPLE_WRITE,
# IRIS_PORTFOLIO_INTERCEPT, IRIS_ON_DEMAND_BRIEF, ...). Inherited, they make the
# suite non-deterministic and machine-dependent: tests assert default/off behavior
# while the shell turns features ON. Worse, some pull REAL local state into a test
# (IRIS_CALENDAR_APPLE_WRITE reads the developer's macOS Calendar; IRIS_SEARXNG_URL
# hits a live search box; IRIS_SPIKE_GMAIL_PASSWORD is a live secret). Keep only the
# handful the suite itself controls — a test that needs a flag opts in explicitly via
# ``monkeypatch.setenv``. Mirrors the IRIS_HOME/IRIS_DATA_DIR relocation below.
_KEEP_IRIS_ENV = {
    "IRIS_HOME",
    "IRIS_DATA_DIR",
    "IRIS_AUTH_SECRET",
    "IRIS_DISABLE_WARMUP",
    "IRIS_DISABLE_ARBITER",
    "IRIS_TEST_NULL_EMBEDDINGS",
}
for _key in [k for k in os.environ if k.startswith("IRIS_") and k not in _KEEP_IRIS_ENV]:
    del os.environ[_key]


# CRITICAL test isolation — git's own environment. A git hook runs with GIT_DIR,
# GIT_INDEX_FILE, GIT_WORK_TREE (and GIT_PREFIX, GIT_COMMON_DIR, GIT_OBJECT_DIRECTORY, ...)
# exported, and the pre-push gate runs this suite from one. Inherited, they aim EVERY git
# call a test makes at the developer's repository, whatever its `cwd`: on 2026-09-26 a
# test's `git init` in a temp dir re-initialised the owner's real repo through the hook's
# GIT_DIR and, with no work tree named, set `core.bare=true` on it. No test needs an
# inherited GIT_* value (a test that wants one sets it itself), so all of them go, here,
# before any test or fixture can start a git process.
def _strip_git_env(environ: dict[str, str] | os._Environ[str]) -> list[str]:
    """Remove every ``GIT_*`` variable from ``environ``; return the names removed."""
    removed = [key for key in environ if key.startswith("GIT_")]
    for key in removed:
        del environ[key]
    return removed


_strip_git_env(os.environ)

# CRITICAL test isolation — the OS keyring. With IRIS_VAULT_MASTER_KEY stripped above,
# the vault (kernel/governance/vault/keys.py) and the credential helpers fall back to the
# OS keyring, which on macOS is the developer's login Keychain: a test that builds a vault
# without faking it read the REAL `iris-vault` master key (and the credential helpers can
# set and delete real entries). Found 2026-09-26 when a fresh python3.13 test env made
# macOS ask for the login password to read `iris-vault`. Every test now gets an
# in-memory keyring, cleared per test (`_fresh_keyring` below); subprocesses get the
# "fail" backend, so nothing a test starts can reach the Keychain either. Tests that
# need a specific keyring still monkeypatch it, as before.
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
try:
    import keyring as _keyring
    from keyring.backend import KeyringBackend as _KeyringBackend
    from keyring.errors import PasswordDeleteError as _PasswordDeleteError
except ImportError:  # keyring is a main dependency; tolerate a trimmed env
    _keyring = None
else:

    class _TestKeyring(_KeyringBackend):
        """The only keyring a test process sees: a dict, never the OS store."""

        priority = 1  # type: ignore[assignment]

        def __init__(self) -> None:
            super().__init__()
            self.entries: dict[tuple[str, str], str] = {}

        def get_password(self, service: str, username: str) -> str | None:
            return self.entries.get((service, username))

        def set_password(self, service: str, username: str, password: str) -> None:
            self.entries[(service, username)] = password

        def delete_password(self, service: str, username: str) -> None:
            if self.entries.pop((service, username), None) is None:
                raise _PasswordDeleteError(f"no {service}/{username} in the test keyring")

    _keyring.set_keyring(_TestKeyring())

# The services enforce bearer auth on every route (security floor, phase 0), so the
# suite always needs a secret — the documented test value unless the shell already
# exports one. Tests reach it via ``iris_harness.foundation.auth.auth_headers()``.
os.environ.setdefault("IRIS_AUTH_SECRET", "test-secret-for-testing")

# The services refuse a Host name they are not reached by (DNS rebinding). Many API
# tests use ``TestClient(app, base_url="http://iris.test")``; ``.test`` is reserved
# (RFC 2606), and the allowlist tests set their own value.
os.environ["IRIS_ALLOWED_HOSTS"] = "iris.test"

# Deterministic terminal rendering. Rich/Click emit ANSI colour codes when the
# shell exports FORCE_COLOR / CLICOLOR_FORCE (Rich forces colour when the var is
# merely PRESENT, even ``FORCE_COLOR=0``). Many CLI tests parse Rich-rendered
# tables as plain text (e.g. ``goal list`` id extraction), so a coloured shell
# makes them fail with unpack/index errors. Strip the force-colour vars so
# CliRunner (a non-tty) renders plain text regardless of the developer's shell.
for _color_var in ("FORCE_COLOR", "CLICOLOR_FORCE", "COLORTERM"):
    os.environ.pop(_color_var, None)
# The same for width and for CI's own forcing, which a hosted runner (GitHub Actions)
# turns on without asking:
#   * a Rich ``Console()`` built while COLUMNS/LINES are set freezes that width for the
#     life of the process, so a module-level console (``cli/device.py``, the shared
#     ``foundation.console``) ignores the ``CliRunner(env={"COLUMNS": ...})`` a test
#     gives it and wraps at the shell's width. Unset here, before any CLI module is
#     imported, a console resolves its width at print time: the test's COLUMNS, or 80
#     (Rich's default off a terminal).
#   * Typer forces its help console into terminal mode (ANSI escapes) when GITHUB_ACTIONS,
#     FORCE_COLOR or PY_COLORS is set, and TERMINAL_WIDTH pins its width.
#     ``_TYPER_FORCE_DISABLE_TERMINAL`` is Typer's own switch for exactly this; it is read
#     when ``typer.rich_utils`` is imported, so it is set here, first.
for _width_var in ("COLUMNS", "LINES", "TERMINAL_WIDTH", "PY_COLORS"):
    os.environ.pop(_width_var, None)
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"

# Never load the developer's repo-root ``.env`` into the test process. Several modules
# call ``load_dotenv()`` at import time (e.g. ``iris_harness.server.iris_api.main`` so uvicorn sees
# provider keys) — but that ``.env`` holds the developer's real feature flags
# (``IRIS_BEHAVIOR_MINER``, ``IRIS_CALENDAR_AUTO_APPROVE_INVITES``, …) and live secrets
# (``IRIS_SPIKE_GMAIL_PASSWORD``). Loaded at import, it would repopulate ``os.environ``
# right after the strip above and make tests machine-dependent. Neutralize it session-wide,
# BEFORE any test module imports a module that calls it, so ``from dotenv import load_dotenv``
# binds to this no-op. Tests set what they need explicitly (pytest env / ``monkeypatch``).
try:
    import dotenv

    dotenv.load_dotenv = lambda *args, **kwargs: False  # type: ignore[assignment]
    if getattr(dotenv, "main", None) is not None:
        dotenv.main.load_dotenv = dotenv.load_dotenv  # type: ignore[attr-defined]
except ImportError:
    pass

# ...and strip the same secrets when the SHELL exports them. Neutralizing
# ``load_dotenv`` only closes the file path into the test process; a developer who
# sources ``.env`` in their profile (or runs the suite from a shell that has)
# hands every test the live values anyway, which is the same machine-dependence
# the block above exists to prevent — and worse, because these credentials reach
# real services.
#
# The concrete risk is not theoretical: with ``TELEGRAM_BOT_TOKEN`` and
# ``TELEGRAM_CHAT_ID`` both present, the only thing standing between a
# runtime-building test and a live long-poll against the developer's real bot is
# the default of ``IRIS_CHANNEL_GATEWAY_TELEGRAM_ENABLED``. One test that flips
# that flag and builds a runtime would start polling the real bot, steal the
# long-poll from a running channel_gateway (Telegram allows one ``getUpdates``
# consumer per token) and answer the developer's real messages. Several tests
# already open with ``monkeypatch.delenv("TELEGRAM_BOT_TOKEN")`` — defending by
# hand against exactly this. Do it once, here, for everyone.
#
# ``IRIS_AUTH_SECRET`` is deliberately NOT in this list: the suite requires it and
# it is set to a documented dummy above.
for _live_secret in (
    # Channel credentials — these drive real outbound messages.
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    # Mailbox credentials from the email spikes.
    "IRIS_SPIKE_GMAIL_PASSWORD",
    # Provider keys — a stray real call is billable and makes results
    # machine-dependent. Tests that want a provider set a fake key themselves.
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "LM_STUDIO_API_KEY",
    "EXA_API_KEY",
    "HF_TOKEN",
    "GITHUB_TOKEN",
):
    os.environ.pop(_live_secret, None)

# Prevent ~/.iris/providers.json from bleeding into tests.
# Set to empty string so _load_global_provider_override returns None.
os.environ.setdefault("IRIS_PROVIDERS_CONFIG", "")

# Which profile the suite runs as. `default` is core-only as of M6.1b (OSS plan M6,
# decision 10) and the domains mount under `personal-assistant`; the suite exercises the
# assembled product, so it asks for that one. It must be set HERE, before any import:
# `iris_harness.main` registers the profile's plugin CLI commands at import time, so a
# test that sets it later gets an `iris email` group that was never built.
#
# The public tree (release 1) ships the email slice and its `email` profile, not
# `personal-assistant` (OSS plan R1/R2): there the assembled product is `email`. Asking
# for a profile the tree lacks would silently run the suite on the built-in fallback.
#
# This only decides which SHIPPED profile is read. A runtime built against a temp config
# dir finds no profile file at all and falls back to what the installation ships
# (`profile._installed_plugins`).
_PROFILES = Path(__file__).resolve().parents[1] / "config" / "profiles"
os.environ.setdefault(
    "IRIS_PROFILE",
    next(
        (p for p in ("personal-assistant", "email") if (_PROFILES / f"{p}.yaml").is_file()),
        "default",
    ),
)

# CRITICAL test isolation — set BEFORE any iris module is imported, because
# IRIS_HOME (identity tree) and the data dir are resolved at import / build time.
# Relocating both to throwaway temp dirs guarantees the suite can never read or
# corrupt the developer's REAL ~/.iris profile (USER.md / SOUL.md) or data
# stores. This closes a real incident where a runtime-building test wrote bogus
# auto-detected facts into the real USER.md. Tests that need a specific data dir
# still override via the ``data_dir`` arg to ``build_runtime``.
#
# Per xdist worker, not per run. The controller imports this file first and exports
# the dirs; every worker then inherits them, so without the second branch all workers
# shared ONE data dir — and stores that live there by name (``filemanager.json``'s
# allowed roots, the SQLite stores) raced across workers: a root one worker added was
# clobbered by another's read-modify-write, and `test_move_intercept` failed
# `needs_root` under the full gate while passing alone. A dir is only replaced when
# its name says this file made it, so an explicitly exported location is honoured.
_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
for _var, _prefix in (("IRIS_HOME", "iris-test-home-"), ("IRIS_DATA_DIR", "iris-test-data-")):
    if _var not in os.environ:
        os.environ[_var] = tempfile.mkdtemp(prefix=_prefix)
    elif _XDIST_WORKER and Path(os.environ[_var]).name.startswith(_prefix):
        os.environ[_var] = tempfile.mkdtemp(prefix=f"{_prefix}{_XDIST_WORKER}-")


@pytest.fixture(autouse=True)
def _isolate_llm_and_embeddings(monkeypatch, request):
    """Default: tests never spawn the Ollama warmup thread, never load embeddings.

    The biggest incident trigger was ``IrisRuntime._warm_models`` firing real
    Ollama requests in a daemon thread on every ``build_runtime`` call.
    ``IRIS_DISABLE_WARMUP=1`` short-circuits that. ``IRIS_TEST_NULL_EMBEDDINGS=1``
    skips the ~80 MB ChromaDB embedding model.

    Opt out per-test:
      * ``@pytest.mark.real_llm``        — allow the warmup thread to run
      * ``@pytest.mark.real_embeddings`` — load ChromaDB and embeddings
    """
    if "real_llm" not in request.keywords:
        monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
        monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    if "real_embeddings" not in request.keywords:
        monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")


@pytest.fixture(autouse=True)
def _reset_llm_circuit_breaker():
    """Clear the process-wide Ollama circuit breaker between tests.

    Exactly the same class of bug as ``_restore_os_environ`` below, in different
    state. The breaker is global on purpose -- one DOWN endpoint should fail fast
    for every caller in the process -- but a test that trips it on a real
    connection failure then fails every later test in the same xdist worker that
    builds a client, reporting "local model server appears down" instead of
    whatever that test was asserting. Pass/fail depends on which worker got which
    module, so it surfaces when the module set changes and looks like the change
    caused it. It does not; the leak did.
    """
    from iris_harness.llm.arbiter import reset_ollama_breaker

    reset_ollama_breaker()
    yield
    reset_ollama_breaker()


#: Where the vault keeps its master key in a keyring (``kernel/governance/vault/keys.py``).
VAULT_KEYRING_ENTRY = ("iris-vault", "master-key")


@pytest.fixture(autouse=True)
def _fresh_keyring():
    """Every test starts with a fresh in-memory keyring holding only a newly generated
    vault master key (see the isolation note above).

    The key is there because every governed tool and capability call needs the audit key
    derived from it, and is refused without one (``kernel/governance/audit/digest.py``) --
    as an owner's install has one in its OS keyring. It is generated per test, never a
    literal. A test about the no-key path removes it (``no_vault_master_key``). The
    process's cached audit digester is forgotten too, so each test resolves its own key.

    Re-installed each time, so a test that swapped the backend cannot leave the next one
    pointed at the OS store.
    """
    if _keyring is None:
        yield
        return
    from cryptography.fernet import Fernet

    from iris_harness.kernel.governance.audit import digest as audit_digest

    backend = _TestKeyring()
    backend.entries[VAULT_KEYRING_ENTRY] = Fernet.generate_key().decode("utf-8")
    _keyring.set_keyring(backend)
    audit_digest._reset_for_tests()
    yield
    backend.entries.clear()
    audit_digest._reset_for_tests()


@pytest.fixture
def no_vault_master_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No vault master key anywhere: not in the env, not in the keyring, nothing cached."""
    from iris_harness.kernel.governance.audit import digest as audit_digest
    from iris_harness.kernel.governance.vault import keys as vault_keys

    monkeypatch.delenv("IRIS_VAULT_MASTER_KEY", raising=False)
    monkeypatch.setattr(vault_keys, "keyring", None)
    audit_digest._reset_for_tests()


@pytest.fixture(autouse=True)
def _restore_os_environ():
    """Revert any direct ``os.environ`` mutation a test (or the code it drives) makes.

    Some runtime code writes flags straight into ``os.environ`` rather than reading
    them — ``agent_settings_store.apply_overrides_to_env`` / ``set_toggle`` seed
    ``IRIS_*`` toggles into the process env so persisted edits take effect. Those are
    direct writes, which ``monkeypatch`` cannot undo. Without this, a settings/toggle
    test leaks a flag (e.g. ``IRIS_CALENDAR_AUTO_APPROVE_INVITES``) into every later
    test and makes pass/fail depend on run order. Snapshotting and restoring keeps each
    test's env changes local to that test — the complement to the session-level
    ``IRIS_*`` strip at the top of this file.
    """
    saved = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture
def owner_identity_seam() -> Iterator[Any]:
    """The owner-identity seam, emptied for one test and put back after it (ADR-0125).

    Inside the test no source is registered (``owner_identity()`` is ``None``); register
    what the test needs through the public functions. Afterwards the documents provider
    and every named source the process had (the composition root's, a plugin's) are
    restored and the cached corpus dropped.
    """
    from iris_harness.kernel.governance import identity_redaction as seam

    saved = (seam._provider, seam._documents_fingerprint, dict(seam._sources))
    seam.clear_identity_text_provider()
    for name in seam.owner_identity_sources():
        seam.unregister_owner_identity_source(name)
    yield seam
    provider, fingerprint, sources = saved
    seam.set_owner_identity_clock(None)
    if provider is None:
        seam.clear_identity_text_provider()
    else:
        seam.register_identity_text_provider(provider, fingerprint=fingerprint)
    for name in seam.owner_identity_sources():
        seam.unregister_owner_identity_source(name)
    seam._sources.update(sources)
    seam.invalidate_owner_identity()


@pytest.fixture
def owner_identity_documents(owner_identity_seam: Any) -> Any:
    """Pin the identity documents to ``texts`` for one test: ``owner_identity_documents([..])``."""

    def pin(texts: list[str]) -> None:
        owner_identity_seam.register_identity_text_provider(lambda: list(texts))

    return pin


TEST_VOCABULARY_DIR = PROJECT_ROOT / "tests" / "fixtures" / "test_vocabulary"


@pytest.fixture
def test_vocabulary(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Memory's vocabulary is the core ontology plus the TEST fragment -- in any tree.

    ``tests/fixtures/test_vocabulary`` is a plugin directory that ships only an
    ``ontology:`` fragment (fact keys ``bank``, ``broker``, ``insurer``, ``credit_card``,
    ``currency`` under ``tv:``). It is installed through the seam every plugin's
    vocabulary uses, ``iris_harness.memory.ontology.vocabulary_fragments()``, which reads
    the plugin directories ``installed_plugin_dirs()`` returns. The built-in plugins stay;
    entry-point and ``$IRIS_HOME`` plugins (the domain plugins, where they are installed)
    are left out, so a memory test proves the same thing whichever domains the tree
    carries. Use it with ``pytest.mark.usefixtures("test_vocabulary")``.
    """
    from iris_harness.foundation import plugin_dirs
    from iris_harness.memory import fact_keys, ontology

    def _installed() -> list[tuple[str, Path]]:
        return [*plugin_dirs._builtin_dirs(), ("test_vocabulary", TEST_VOCABULARY_DIR)]

    monkeypatch.setattr(ontology, "installed_plugin_dirs", _installed)
    ontology.reset_cache()
    fact_keys.reset_cache()
    yield TEST_VOCABULARY_DIR
    ontology.reset_cache()
    fact_keys.reset_cache()


@pytest.fixture(autouse=True)
def _minilm_on_disk_or_skip(request: pytest.FixtureRequest) -> None:
    """``@pytest.mark.minilm`` tests assert on real all-MiniLM-L6-v2 similarities.

    The semantic routers embed with chromadb's ONNX copy of the model. The suite never
    downloads it (the network guard below refuses the fetch, and ``DefaultEmbedder``
    does not try under ``IRIS_TEST_NULL_EMBEDDINGS``), so these tests run where the
    model is already on disk -- a machine that has run IRIS once -- and skip, saying
    why, on a fresh clone. Everything else in the suite runs either way.
    """
    if request.node.get_closest_marker("minilm") is None:
        return
    from iris_harness.kernel.governance.evaluator.embeddings import (
        default_model_on_disk,
        default_model_path,
    )

    if not default_model_on_disk():
        pytest.skip(
            f"needs the all-MiniLM-L6-v2 ONNX model at {default_model_path()} (the suite "
            "never downloads it; running IRIS once, or chromadb's DefaultEmbeddingFunction, "
            "fetches it)"
        )


# --------------------------------------------------------------------------------
# Network guard: no test talks to a real model server or leaves the machine.
#
# A test that reaches a live Ollama / LM Studio / llama-server, or the developer's
# running IRIS stack, gets a real, slow, machine-dependent answer — it passes on the developer's box with a model loaded
# and fails (or hangs on a timeout) everywhere else, and it is not testing IRIS
# code at all. The same goes for any non-loopback host. So every TCP ``connect``
# in the test process is checked:
#
#   * loopback (127.0.0.0/8, ::1, ``localhost``) on any port EXCEPT the model
#     and IRIS service ports below — allowed. That is how in-process uvicorn servers, fake HTTP
#     servers and anything else a test starts on an ephemeral port are reached.
#     (TestClient/ASGI transports open no socket at all.)
#   * loopback on a model or IRIS service port, or ANY non-loopback address —
#     refused.
#   * Unix sockets and UDP — not checked (a UDP ``connect`` sends nothing, and
#     DNS lookups are not connections; the guard is about traffic to servers).
#
# A refused connect raises ``ConnectionRefusedError`` in the code that tried it
# (so it behaves exactly like a model server that is down) AND the test fails
# with the host:port, even when the code under test swallowed the error — that
# is the case that hid these calls before: an ``except Exception`` fallback turned
# a live call into a pass that only worked with Ollama running.
#
# Opt out with ``@pytest.mark.real_llm`` (deselected by the default ``addopts``),
# for tests that exist to talk to a real model.
#
# Cost: one Python-level check per TCP connect; nothing on any other call.
# --------------------------------------------------------------------------------
# Ollama, LM Studio, llama-server. 8090 is also the IRIS evaluator's port — a live
# service either way.
MODEL_SERVER_PORTS = frozenset({11434, 1234, 8090})
# The IRIS services a developer keeps running: Governor, IRIS API, channel gateway.
IRIS_SERVICE_PORTS = frozenset({8080, 8003, 8006})


class RealNetworkBlocked(ConnectionRefusedError):
    """Raised by the test network guard in place of a real outbound connection."""


class _NetGuard:
    """Which test is running, whether it may use the network, what it tried."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.test_id: str | None = None
        # Outside a test (collection, session hooks, a daemon thread still running
        # after its test ended) nothing is checked: there is no test to fail.
        self.allowed = True
        self.violations: list[str] = []


_NET_GUARD = _NetGuard()
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False  # an unresolved hostname: treat as remote


def _blocked_reason(sock: socket.socket, address: Any) -> str | None:
    """Why this connect must not happen, or None when it is allowed."""
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return None
    if sock.type != socket.SOCK_STREAM:
        return None
    if not isinstance(address, tuple) or len(address) < 2:
        return None
    host, port = str(address[0]), address[1]
    if _is_loopback(host):
        if port in MODEL_SERVER_PORTS:
            return f"model server port {host}:{port}"
        if port in IRIS_SERVICE_PORTS:
            return f"IRIS service port {host}:{port}"
        return None
    return f"non-loopback host {host}:{port}"


def _calling_site() -> str:
    """The innermost repo frame (src/ or tests/) behind a blocked connect."""
    for frame in reversed(traceback.extract_stack()[:-3]):
        path = frame.filename
        if path == __file__ or "site-packages" in path:
            continue
        if f"{os.sep}src{os.sep}" in path or f"{os.sep}tests{os.sep}" in path:
            return f"{Path(path).name}:{frame.lineno} ({frame.name})"
    return "unknown site"


def _check_connect(sock: socket.socket, address: Any) -> None:
    guard = _NET_GUARD
    if guard.allowed:
        return
    reason = _blocked_reason(sock, address)
    if reason is None:
        return
    message = (
        f"{guard.test_id} tried to open a real connection to {reason} "
        f"from {_calling_site()} (thread {threading.current_thread().name}). "
        "Stub the model/network call, or mark the test @pytest.mark.real_llm "
        "if it genuinely needs a live server."
    )
    with guard.lock:
        guard.violations.append(message)
    raise RealNetworkBlocked(message)


def _guarded_connect(self: socket.socket, address: Any) -> None:
    _check_connect(self, address)
    _REAL_CONNECT(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    _check_connect(self, address)
    return _REAL_CONNECT_EX(self, address)


# ``socket.create_connection``, http.client, urllib, httpx/httpcore (sync) and
# asyncio's ``sock_connect`` (httpx async, aiohttp, websockets) all end in one of
# these two methods, so patching them covers every client library in use here.
socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]


def _take_violations() -> list[str]:
    with _NET_GUARD.lock:
        found, _NET_GUARD.violations = _NET_GUARD.violations, []
    return found


def _raised_by_guard(exc: BaseException) -> bool:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, RealNetworkBlocked):
            return True
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False


def _violation_report(found: list[str]) -> str:
    unique = list(dict.fromkeys(found))
    return "real network connection blocked by tests/conftest.py guard:\n  " + "\n  ".join(unique)


# Armed by runtest hooks rather than an autouse fixture: a module- or
# session-scoped fixture is set up BEFORE any function-scoped fixture, so a
# fixture-armed guard would miss exactly the "probe the server once per module"
# connects (tests/integration/test_lmstudio_gemma.py has one).
@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Any:
    """Arm the network guard for everything this test runs: setup, call, teardown."""
    guard = _NET_GUARD
    with guard.lock:
        guard.test_id = item.nodeid
        guard.allowed = "real_llm" in item.keywords
        guard.violations = []
    try:
        return (yield)
    finally:
        with guard.lock:
            guard.allowed = True
            guard.test_id = None
            guard.violations = []


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Any:
    """Fail the test itself when it (or its setup) tried to reach a real server.

    If the test raised something else, the violations stay queued and the teardown
    hook reports them, so the host:port is never lost behind an unrelated failure.
    A test that died of the guard's own error already names it: nothing to add.
    """
    try:
        result = yield
    except BaseException as exc:
        if _raised_by_guard(exc):
            _take_violations()
        raise
    found = _take_violations()
    if found:
        pytest.fail(_violation_report(found), pytrace=False)
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> Any:
    """Report connects the call phase did not: from teardown, a failed or skipped
    setup (a probe that found the server "down" and skipped), or a failed call."""
    result = yield
    found = _take_violations()
    if found:
        pytest.fail(_violation_report(found), pytrace=False)
    return result


# --------------------------------------------------------------------------------
# Stubs for the paths the guard caught. Opt-in, so a file says what it replaces.
# --------------------------------------------------------------------------------
class _OfflineChatModel:
    """A chat model whose server is down: every call fails like a refused connect.

    This is the state hosted CI always ran these tests in (no Ollama on the
    runner), so the tests keep asserting what they asserted there — the degrade
    path — instead of whatever a live local model happened to answer.
    """

    def bind_tools(self, _tools: Any) -> _OfflineChatModel:
        return self

    def invoke(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ConnectionRefusedError("offline_llm: tests run without a model server")

    def stream(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ConnectionRefusedError("offline_llm: tests run without a model server")


@pytest.fixture()
def offline_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every ``CodingLLMClient`` built without its own factory gets a dead model.

    Patches the default factory, which a client binds at construction, so it
    must be active before the runtime is built (use it as a fixture, or via
    ``pytestmark = pytest.mark.usefixtures("offline_llm")``). A test that passes
    its own ``model_factory`` or stubs a caller keeps its own stub.
    """
    from iris_harness.llm import client

    monkeypatch.setattr(client, "_default_model_factory", lambda **_kw: _OfflineChatModel())


@pytest.fixture()
def offline_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Health snapshots see every IRIS service and Ollama as unreachable.

    ``refresh()`` builds the snapshot with the real HTTP prober, which would ask
    the developer's running stack (Governor, API, gateway, evaluator, Ollama).
    """
    import functools

    from iris_harness.services.health import checks, service

    monkeypatch.setattr(
        service,
        "build_snapshot",
        functools.partial(checks.build_snapshot, service_prober=lambda _url: None),
    )


@pytest.fixture()
def offline_ollama_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """``runtime_inventory()`` lists no local models instead of asking the live Ollama.

    Reached by ``/runtime/inventory``, ``/api/chat-status`` and the ``/info`` command.
    """
    from iris_harness.services.system import inventory

    def _no_ollama(_url: str) -> dict[str, object]:
        raise ConnectionRefusedError("offline_ollama_inventory: no Ollama in tests")

    monkeypatch.setattr(inventory, "_fetch_ollama_tags", _no_ollama)


@pytest.fixture()
def cli_api(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return ``route(handler, *modules)``: the CLI's API client answers from ``handler``.

    ``handler`` is an ``httpx.MockTransport`` handler (``httpx.Request -> httpx.Response``;
    raise ``httpx.ConnectError`` for "the API is down"). The real ``harness_api_client``
    still builds the client, so its egress hook and auth headers run as in production.
    Pass each CLI module that imported the helper by name; a module that imports it
    inside a function (``cli.approvals``) resolves it on ``api_client`` itself.
    """
    import httpx

    from iris_harness.cli import api_client

    real = api_client.harness_api_client

    def route(handler: Any, *modules: Any) -> None:
        transport = httpx.MockTransport(handler)

        def client(**kwargs: Any) -> httpx.Client:
            return real(transport=transport, **kwargs)

        for module in (api_client, *modules):
            monkeypatch.setattr(module, "harness_api_client", client)

    return route


@pytest.fixture()
def offline_web_fetch(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return ``stub(runtime)``: the runtime's web-fetch skill tool gets no network.

    The skill's module is loaded from ``config/skills`` at discovery, so it can
    only be patched after ``skill_registry.discover()``. Its ``_http_get``
    returning ``None`` is the skill's own "request failed" path.
    """
    from iris_harness.runtime.handlers.skill_brief import _build_tool_index

    def stub(runtime: Any) -> None:
        index = _build_tool_index(runtime.skill_registry)
        tool_class = index.get(("web-fetch", "fetch_web_content"))
        assert tool_class is not None, "web-fetch skill not discovered"
        module = sys.modules[tool_class.__module__]
        monkeypatch.setattr(module, "_http_get", lambda *_a, **_kw: None)

    return stub


@pytest.fixture(autouse=True)
def _restore_process_state():
    """Put back every declared piece of process-wide state after each test
    (``foundation/process_state.py``): plugin API routes, mail providers, health checks,
    the process bus's subscribers, the identity-redaction seams...

    ``iris_harness.testing.harness`` does this for its own runs, but a test that calls
    ``build_runtime`` or a plugin's ``setup`` directly does not, and what it mounted
    stayed for every later test in the worker: a default-profile runtime (which prefers
    ``email`` when the email plugins are installed) left the email routes registered,
    so a later app served /api/v1/email/* for a profile without the email plugins. Same
    class as ``_restore_os_environ``; higher-scoped fixtures run first, so what they
    register is part of the snapshot and survives.
    """
    from iris_harness.foundation.process_state import (
        restore_process_state,
        snapshot_process_state,
    )

    snapshot = snapshot_process_state()
    yield
    restore_process_state(snapshot)


@pytest.fixture(autouse=True)
def _no_chat_turn_in_progress():  # a pytest generator fixture
    """Each test starts with no chat turn marked (foundation/activity.py): a chat test
    that just ended would otherwise make the next test's background sweep yield."""
    from iris_harness.foundation import activity

    activity._reset_for_tests()
    yield
    activity._reset_for_tests()


# --------------------------------------------------------------------------- scaling checks


class ScalingCheck:
    """Does a scan's cost grow in proportion to its input, whatever the machine?

    An absolute wall-clock bound fails on a loaded runner. A ratio of two timings taken back to
    back survives that, but only if the timings do not include time the thread spent descheduled,
    so the clock is the thread's CPU time (``time.thread_time``: ``CLOCK_THREAD_CPUTIME_ID``,
    nanosecond resolution on macOS and Linux); the collector is run before and switched off
    during each timed call. Even CPU time is not constant on a shared VM (SMT, frequency, steal
    change the work done per CPU-second), so the margins are wide: the input is built at N and at
    8N and the best of three timings of each side compared. Linear gives about 8x, a quadratic
    scan about 64x, and the limit is 24x: three times the linear ratio of headroom for noise,
    2.7x below a quadratic one. N doubles until the small side takes at least 20 ms, so a fast
    machine is not measuring timer noise.

    The clock is injectable. The decision logic (the ratio rule and the doubling floor) is tested
    with a COUNTING clock, where "time" is the number of steps a stand-in took, so the detector
    tests involve no real timing and cannot flake: ``assert_detects_quadratic``.
    """

    FACTOR = 8
    LIMIT = 24.0
    SLACK_SECONDS = 0.05
    REPEATS = 3
    MIN_SMALL_SECONDS = 0.02
    MAX_DOUBLINGS = 6

    def __init__(
        self,
        *,
        clock: Any = None,
        repeats: int | None = None,
        min_small: float | None = None,
        slack: float | None = None,
    ) -> None:
        import time

        self._clock = clock or time.thread_time
        self._repeats = self.REPEATS if repeats is None else repeats
        self._min_small = self.MIN_SMALL_SECONDS if min_small is None else min_small
        self._slack = self.SLACK_SECONDS if slack is None else slack

    def _best_of(self, run: Any, text: Any) -> float:
        """The least clock reading over the repeats of ``run(text)`` (CPU seconds by default)."""
        import gc

        best = float("inf")
        for _ in range(self._repeats):
            gc.collect()
            was_enabled = gc.isenabled()
            gc.disable()
            try:
                start = self._clock()
                run(text)
                best = min(best, self._clock() - start)
            finally:
                if was_enabled:
                    gc.enable()
        return best

    def is_linear(self, run: Any, build: Any, n: int) -> tuple[bool, float, float]:
        """``(linear, clock units at N, clock units at 8N)``; N is raised until N reaches the floor."""
        small = self._best_of(run, build(n))
        for _ in range(self.MAX_DOUBLINGS):
            if small >= self._min_small:
                break
            n *= 2
            small = self._best_of(run, build(n))
        large = self._best_of(run, build(self.FACTOR * n))
        return large <= self.LIMIT * small + self._slack, small, large

    def assert_linear(self, name: str, run: Any, build: Any, n: int) -> None:
        ok, small, large = self.is_linear(run, build, n)
        assert ok, (
            f"{name}: {small:.3f}s at N, {large:.3f}s at {self.FACTOR}N "
            f"(limit {self.LIMIT}x + {self._slack}s)"
        )

    def assert_detects_quadratic(self) -> None:
        """The decision logic accepts a linear stand-in and rejects a quadratic one.

        Both stand-ins count their steps and the clock reads that counter, so the result does not
        depend on any real timing. The floor is 20,000 steps; N starts below it, so the doubling
        is exercised too.
        """
        steps = [0]

        def linear(text: str) -> None:
            for _ in text:
                steps[0] += 1

        def quadratic(text: str) -> None:
            for i in range(len(text)):
                for _ in range(i, len(text)):
                    steps[0] += 1

        counting = ScalingCheck(
            clock=lambda: float(steps[0]), repeats=1, min_small=20_000.0, slack=0.0
        )
        ok, small, large = counting.is_linear(linear, lambda n: "a" * n, 1_000)
        assert ok and small >= 20_000, f"a linear stand-in failed ({small} -> {large} steps)"
        ok, small, large = counting.is_linear(quadratic, lambda n: "a" * n, 50)
        assert small >= 20_000, f"the doubling did not reach the floor ({small} steps)"
        assert not ok, f"a quadratic stand-in passed as linear ({small} -> {large} steps)"


@pytest.fixture
def scaling() -> ScalingCheck:
    return ScalingCheck()
