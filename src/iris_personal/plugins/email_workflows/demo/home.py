"""The demo's isolated home and the environment its run gets.

``iris email demo`` never runs in the owner's profile. It runs in its own ``IRIS_HOME``
(``$IRIS_DEMO_HOME``, else ``~/.iris-demo``): its own data dir, governance stores,
audit ledger and vault master key. The run is a child process started with an
environment built here from scratch, so nothing of the owner's leaks in:

* every ``IRIS_*`` variable of the parent is dropped (the owner's ``.env`` flags, a
  data-dir or audit-DB override, a config dir), except the time zone;
* every variable that looks like a credential (``*KEY*``, ``*TOKEN*``, ``*SECRET*``,
  ``*PASSWORD*``, ``*CREDENTIAL*``) is dropped: the demo needs none;
* the demo's own settings are set explicitly, the model among them: every tier on the
  scripted fake (``IRIS_LLM_PROVIDER=fake``), answering from ``fake_model.yaml``.

The vault master key (#741: without one no governed tool call runs) is a throwaway
Fernet key generated into the demo home on first run, mode 0600. It never touches the
OS keyring and is never the owner's key.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from iris_harness.sdk.llm import FAKE_MODEL_SCRIPT_ENV, FAKE_PROVIDER, FORCED_PROVIDER_ENV

HOME_ENV = "IRIS_DEMO_HOME"
# Written into a demo home on creation; ``--reset`` deletes only a directory that has it.
MARKER = ".iris-demo"
KEY_FILENAME = "vault-master.key"
FAKE_SCRIPT = Path(__file__).with_name("fake_model.yaml")

# Kept from the parent's IRIS_* settings: where "today" is.
_KEEP_IRIS = frozenset({"IRIS_TZ"})
_CREDENTIAL_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PASSWD")


class DemoHomeError(RuntimeError):
    """The demo home cannot be used (a non-demo directory is in the way)."""


def default_home() -> Path:
    """``$IRIS_DEMO_HOME``, else ``~/.iris-demo`` -- a sibling of the owner's ``~/.iris``,
    never inside it."""
    raw = os.environ.get(HOME_ENV, "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".iris-demo"


def in_demo_home() -> bool:
    """This process runs inside a demo home: ``IRIS_HOME`` is ``$IRIS_DEMO_HOME``, as
    ``demo_environment`` sets them. Anything else is the owner's own profile."""
    home = os.environ.get(HOME_ENV, "").strip()
    return bool(home) and os.environ.get("IRIS_HOME", "").strip() == home


def prepare_home(home: Path, *, reset: bool = False) -> Path:
    """Create (or, with ``reset``, recreate) the demo home; return it resolved.

    Refuses a directory that exists, is not empty and was not made by the demo: the
    demo writes and ``--reset`` deletes, so it only ever does either in its own home.
    """
    home = home.expanduser().resolve()
    if home.exists() and any(home.iterdir()) and not (home / MARKER).is_file():
        raise DemoHomeError(
            f"{home} exists and is not a demo home (no {MARKER} marker); "
            f"pick another with --home or {HOME_ENV}"
        )
    if reset and home.exists():
        shutil.rmtree(home)
    (home / "data").mkdir(parents=True, exist_ok=True)
    (home / MARKER).write_text("iris email demo\n", encoding="utf-8")
    return home


def master_key(home: Path) -> str:
    """The demo home's throwaway vault master key, generated on first use."""
    from cryptography.fernet import Fernet

    path = home / KEY_FILENAME
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    key = Fernet.generate_key().decode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")
    return key


def _looks_like_credential(name: str) -> bool:
    upper = name.upper()
    return any(word in upper for word in _CREDENTIAL_WORDS)


def demo_environment(home: Path, base: Mapping[str, str]) -> dict[str, str]:
    """The environment of a demo run in ``home``: ``base`` minus the owner's IRIS
    settings and credentials, plus the demo's own."""
    env = {
        name: value
        for name, value in base.items()
        if not _looks_like_credential(name) and (not name.startswith("IRIS_") or name in _KEEP_IRIS)
    }
    # The run's working directory is the demo home, so a relative import path (a
    # checkout's PYTHONPATH=src) is made absolute against the caller's directory.
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.pathsep.join(
            str(Path(p).resolve()) if p else p for p in env["PYTHONPATH"].split(os.pathsep)
        )
    env.update(
        {
            "IRIS_HOME": str(home),
            "IRIS_DATA_DIR": str(home / "data"),
            "IRIS_GOVERNANCE_AUDIT_DB_PATH": str(home / "governance" / "audit.db"),
            "IRIS_VAULT_MASTER_KEY": master_key(home),
            FORCED_PROVIDER_ENV: FAKE_PROVIDER,
            FAKE_MODEL_SCRIPT_ENV: str(FAKE_SCRIPT),
            # The judge judges; labels are on, and the demo writes them after walking
            # setup's step 6 for its own account. The semantic index and model warm-up
            # would reach for an embedding model / a model server.
            "IRIS_EMAIL_JUDGE": "1",
            "IRIS_EMAIL_JUDGE_LABELS": "1",
            "IRIS_EMAIL_SEMANTIC_SEARCH": "0",
            "IRIS_DISABLE_WARMUP": "1",
            "IRIS_PROFILE": "minimal",
            # Marks the run as the demo's, for the run's own isolation check.
            HOME_ENV: str(home),
        }
    )
    return env


__all__ = [
    "FAKE_SCRIPT",
    "HOME_ENV",
    "KEY_FILENAME",
    "MARKER",
    "DemoHomeError",
    "default_home",
    "demo_environment",
    "in_demo_home",
    "master_key",
    "prepare_home",
]
