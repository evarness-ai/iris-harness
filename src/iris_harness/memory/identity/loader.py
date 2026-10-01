"""Loader for the four-layer agentic memory files at ``~/.iris/``.

The runtime calls :func:`bootstrap_identity_files` once at startup to seed
``~/.iris/identity/`` and ``~/.iris/memory/`` from the repo's ``config/identity/``
defaults, then calls :func:`load_soul` and :func:`load_user_md` per request to
inject the user-curated content into the system prompt.

Files are user-owned:  the bootstrap step never overwrites an existing file.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from iris_harness.foundation.clock import local_today
from iris_harness.foundation.paths import config_path, default_config_dir, repo_root
from iris_harness.foundation.process_state import track_globals
from iris_harness.kernel.governance.identity_redaction import invalidates_owner_identity


class UserFact(Protocol):
    """The three fields this module renders into ``USER.md``.

    Read-only properties, not bare attributes: a protocol with mutable
    attributes is invariant, and a store returning ``list[ConcreteFact]`` then
    fails to match ``Sequence[UserFact]``. This module only reads.
    """

    @property
    def key(self) -> str: ...

    @property
    def value(self) -> str: ...

    @property
    def confidence(self) -> float: ...


class UserFactSource(Protocol):
    """Where the auto-detected half of ``USER.md`` comes from.

    Structural on purpose. The identity loader renders markdown; it has no
    business knowing that the facts arrive from a SQLite store, and saying so as
    a type would put an import on the memory layer -- which reads this module
    back, from three places. ``MemoryStore`` satisfies this protocol without
    being named here, and without being told to.
    """

    def fetch_all_user_facts(self) -> Sequence[UserFact]: ...

    def fetch_fact_projections(self) -> Sequence[UserFact]:
        """One fact per key; a key with several values lists them all."""
        ...


logger = logging.getLogger(__name__)


# ``$IRIS_HOME`` relocates the entire identity tree (SOUL.md, USER.md,
# behaviors) so an eval / sandbox instance never reads or writes the real
# user's profile. Defaults to ``~/.iris``. Resolved on every use, not once at import:
# a process that relocates ``IRIS_HOME`` after importing IRIS (``iris_harness.testing``'s
# harness, a test) must not keep writing into the home that was current at import.
# See also ``$IRIS_DATA_DIR`` for the SQLite / Chroma stores.
#
# Each name below is an override slot, ``None`` by default: set one (the playground's
# isolated runs, a test's ``monkeypatch.setattr``) and its accessor returns it instead
# of the path derived from ``IRIS_HOME``. Read the paths through the accessors.
IRIS_HOME: Path | None = None
WORKSPACE_DIR: Path | None = None
# Legacy locations — pre-2026-05-19 split kept soul/user under
# identity/memory subdirs. Loader still reads from them as a fallback
# when the workspace copy is missing, so existing installs keep
# working until they migrate.
IDENTITY_DIR: Path | None = None
MEMORY_DIR: Path | None = None
BEHAVIORS_DIR: Path | None = None

SOUL_PATH: Path | None = None
USER_MD_PATH: Path | None = None
AGENTS_MD_PATH: Path | None = None
_LEGACY_SOUL_PATH: Path | None = None
_LEGACY_USER_MD_PATH: Path | None = None
ACTIVE_MD_PATH: Path | None = None
EPISODIC_MD_PATH: Path | None = None


def iris_home() -> Path:
    """``$IRIS_HOME``, else ``~/.iris`` (or the ``IRIS_HOME`` override slot)."""
    if IRIS_HOME is not None:
        return IRIS_HOME
    return Path(os.environ.get("IRIS_HOME") or (Path.home() / ".iris"))


def workspace_dir() -> Path:
    return WORKSPACE_DIR if WORKSPACE_DIR is not None else iris_home() / "workspace"


def identity_dir() -> Path:
    return IDENTITY_DIR if IDENTITY_DIR is not None else iris_home() / "identity"


def memory_dir() -> Path:
    return MEMORY_DIR if MEMORY_DIR is not None else iris_home() / "memory"


def behaviors_dir() -> Path:
    return BEHAVIORS_DIR if BEHAVIORS_DIR is not None else iris_home() / "behaviors"


def soul_path() -> Path:
    return SOUL_PATH if SOUL_PATH is not None else workspace_dir() / "SOUL.md"


def user_md_path() -> Path:
    return USER_MD_PATH if USER_MD_PATH is not None else workspace_dir() / "USER.md"


def agents_md_path() -> Path:
    return AGENTS_MD_PATH if AGENTS_MD_PATH is not None else workspace_dir() / "AGENTS.md"


def _legacy_soul_path() -> Path:
    return _LEGACY_SOUL_PATH if _LEGACY_SOUL_PATH is not None else identity_dir() / "soul.md"


def _legacy_user_md_path() -> Path:
    if _LEGACY_USER_MD_PATH is not None:
        return _LEGACY_USER_MD_PATH
    return memory_dir() / "user.md"


def active_md_path() -> Path:
    return ACTIVE_MD_PATH if ACTIVE_MD_PATH is not None else memory_dir() / "active.md"


def episodic_md_path() -> Path:
    return EPISODIC_MD_PATH if EPISODIC_MD_PATH is not None else memory_dir() / "episodic.md"


# The override slots and their accessors, for a caller that redirects the whole tree
# (the playground's isolated runs).
IDENTITY_PATH_ACCESSORS: dict[str, Callable[[], Path]] = {
    "WORKSPACE_DIR": workspace_dir,
    "IDENTITY_DIR": identity_dir,
    "MEMORY_DIR": memory_dir,
    "BEHAVIORS_DIR": behaviors_dir,
    "SOUL_PATH": soul_path,
    "USER_MD_PATH": user_md_path,
    "AGENTS_MD_PATH": agents_md_path,
    "_LEGACY_SOUL_PATH": _legacy_soul_path,
    "_LEGACY_USER_MD_PATH": _legacy_user_md_path,
    "ACTIVE_MD_PATH": active_md_path,
    "EPISODIC_MD_PATH": episodic_md_path,
}

# Shipped defaults — populated by `bootstrap_identity_files` on first run. The checkout's
# config/ or, installed from a wheel, the packaged copy (foundation/paths.py).
_REPO_ROOT = repo_root()
_DEFAULTS_DIR = default_config_dir() / "identity"
_SOUL_DEFAULT = _DEFAULTS_DIR / "soul.default.md"
_ACTIVE_DEFAULT = _DEFAULTS_DIR / "active.default.md"
_EPISODIC_DEFAULT = _DEFAULTS_DIR / "episodic.default.md"
_BEHAVIORS_DEFAULTS_DIR = default_config_dir() / "behaviors"

# In-repo operational reference — read by the `iris_doc("HARNESS")` tool.
# Living document; ships with code and changes when the runtime does.
HARNESS_DOC_PATH = _REPO_ROOT / "docs" / "architecture" / "iris-harness.md"

# Matches a checkbox bullet:  `- [ ] text`  or  `- [x] text`.
_CHECKBOX_RE = re.compile(r"^(?P<indent>\s*)- \[(?P<mark>[ xX])\] (?P<text>.*)$")
# Matches a plain bullet (NOT a checkbox) — used for episodic patterns.
_BULLET_RE = re.compile(r"^\s*-\s+(?!\[[ xX]\])(?P<text>.+?)\s*$")
_EPISODIC_INDEXED_SECTIONS = frozenset(
    {
        "long term patterns",
        "patterns",
        "routines",
        "source preferences",
        "automation candidates",
        "wiki index",
    }
)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@invalidates_owner_identity
def bootstrap_identity_files(*, memory_store: UserFactSource | None = None) -> None:
    """Create ``~/.iris/`` directories and seed missing files.

    Idempotent: never overwrites an existing user file.  Run once at runtime
    startup before the first prompt is built. As of 2026-05-19 identity
    lives in ``~/.iris/workspace/`` (SOUL.md + USER.md); legacy
    ``~/.iris/identity/`` and ``~/.iris/memory/`` paths are auto-migrated
    on first encounter so existing installs keep working.
    """
    workspace_dir().mkdir(parents=True, exist_ok=True)
    identity_dir().mkdir(parents=True, exist_ok=True)
    memory_dir().mkdir(parents=True, exist_ok=True)
    behaviors_dir().mkdir(parents=True, exist_ok=True)

    # One-time migration: pull legacy soul/user content into the workspace
    # the first time we see it. Never clobber an existing workspace file.
    if not soul_path().exists() and _legacy_soul_path().exists():
        shutil.copy2(_legacy_soul_path(), soul_path())
        logger.info("migrated %s -> %s", _legacy_soul_path(), soul_path())
    if not user_md_path().exists() and _legacy_user_md_path().exists():
        shutil.copy2(_legacy_user_md_path(), user_md_path())
        logger.info("migrated %s -> %s", _legacy_user_md_path(), user_md_path())

    if not soul_path().exists() and _SOUL_DEFAULT.exists():
        shutil.copy2(_SOUL_DEFAULT, soul_path())
        logger.info("seeded %s from %s", soul_path(), _SOUL_DEFAULT)

    if not active_md_path().exists() and _ACTIVE_DEFAULT.exists():
        shutil.copy2(_ACTIVE_DEFAULT, active_md_path())
        logger.info("seeded %s from %s", active_md_path(), _ACTIVE_DEFAULT)

    if not episodic_md_path().exists() and _EPISODIC_DEFAULT.exists():
        shutil.copy2(_EPISODIC_DEFAULT, episodic_md_path())
        logger.info("seeded %s from %s", episodic_md_path(), _EPISODIC_DEFAULT)

    if _BEHAVIORS_DEFAULTS_DIR.exists():
        for default_file in _BEHAVIORS_DEFAULTS_DIR.glob("*.default.md"):
            target = behaviors_dir() / default_file.name.replace(".default.md", ".md")
            if not target.exists():
                shutil.copy2(default_file, target)
                logger.info("seeded %s from %s", target, default_file)

    if not user_md_path().exists() and memory_store is not None:
        sync_user_facts_to_md(memory_store)


# ---------------------------------------------------------------------------
# Identity (Layer 1)
# ---------------------------------------------------------------------------


def load_soul() -> str | None:
    """Return the body of ``~/.iris/workspace/SOUL.md`` or None if absent.

    ``SOUL.md`` may include YAML frontmatter (``name``, ``classification``,
    ``load``, ``egress``, ``version``). Frontmatter is metadata for the
    loader / governance kernel and is never injected into prompts.

    Falls back to the legacy ``~/.iris/identity/soul.md`` path when the
    workspace copy is missing — keeps pre-2026-05-19 installs working
    until ``bootstrap_identity_files`` migrates them.
    """
    path = soul_path() if soul_path().exists() else _legacy_soul_path()
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None
    _meta, body = _split_frontmatter(text)
    return body.strip() or None


_SOUL_SECTION_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
_SOUL_LAYERS_CACHE: dict[str, object] | None = None


def _soul_layers() -> dict[str, object]:
    """Read ``config/identity/soul_layers.yaml`` (cached).

    Missing or unreadable config means "nothing is extended" — the whole soul keeps
    riding in the prompt, which is the pre-split behavior.
    """
    global _SOUL_LAYERS_CACHE
    if _SOUL_LAYERS_CACHE is not None:
        return _SOUL_LAYERS_CACHE
    path = config_path("identity", "soul_layers.yaml")
    data: dict[str, object] = {}
    if not path.exists():
        path = default_config_dir() / "identity" / "soul_layers.yaml"
    if path.exists():
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s: %s — keeping the whole soul in L0", path, exc)
    _SOUL_LAYERS_CACHE = data
    return data


def _split_soul_sections(body: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a soul body into (preamble, [(title, section_text), ...])."""
    matches = list(_SOUL_SECTION_RE.finditer(body))
    if not matches:
        return body, []
    preamble = body[: matches[0].start()]
    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        sections.append((match.group("title").strip(), body[match.start() : end]))
    return preamble, sections


def _extended_titles() -> set[str]:
    raw = _soul_layers().get("extended_sections") or []
    if not isinstance(raw, list):
        return set()
    return {str(t).strip().lower() for t in raw if str(t).strip()}


def load_soul_core() -> str | None:
    """The part of SOUL.md that rides in every prompt (L0).

    Identity, mission and the hard rules stay; the internals tour, style notes and
    the full tool policy move to ``iris_doc("OPERATING")`` and are replaced by the
    short tool-policy summary from the config. A section the user added and did not
    list as extended stays here — hiding someone's own edits would be the wrong
    default.
    """
    body = load_soul()
    if body is None:
        return None
    extended = _extended_titles()
    if not extended:
        return body
    preamble, sections = _split_soul_sections(body)
    kept = [preamble.rstrip()] if preamble.strip() else []
    moved = False
    for title, text in sections:
        if title.lower() in extended:
            moved = True
            continue
        kept.append(text.rstrip())
    if not moved:
        return body
    summary = str(_soul_layers().get("tool_policy_summary") or "").strip()
    if summary:
        kept.append(summary)
    return "\n\n".join(part for part in kept if part).strip() or None


def load_soul_extended() -> str | None:
    """The sections held back from the prompt — served by ``iris_doc("OPERATING")``."""
    body = load_soul()
    if body is None:
        return None
    extended = _extended_titles()
    if not extended:
        return None
    _preamble, sections = _split_soul_sections(body)
    kept = [text.rstrip() for title, text in sections if title.lower() in extended]
    if not kept:
        return None
    return ("# Operating detail (held out of the per-turn prompt)\n\n" + "\n\n".join(kept)).strip()


@invalidates_owner_identity
def write_curated_profile(text: str) -> Path:
    """Replace the hand-curated head of USER.md, keeping the auto-detected block.

    The block below ``## Auto-detected`` is the fact store's projection — it is
    rewritten whenever a fact changes and must not be edited by hand, so an editor
    (the web UI, a CLI) hands us only the part above it.
    """
    path = user_md_path() if user_md_path().exists() else _legacy_user_md_path()
    existing = ""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("could not read %s: %s", path, exc)
    match = _AUTO_HEADER_RE.search(existing)
    tail = existing[match.start() :] if match else ""
    body = text.strip() + ("\n\n" + tail.strip() + "\n" if tail.strip() else "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def load_curated_profile() -> str:
    """The hand-curated head of USER.md — what an editor should show."""
    from iris_harness.memory.retriever import _curated_profile

    return _curated_profile(load_user_md() or "")


def load_agent_name(default: str = "IRIS") -> str:
    """Return the configured agent name from ``soul.md`` frontmatter.

    Falls back to the repo default, then a first-level heading like
    ``# IRIS - Soul``, then ``default``. This keeps older user-owned soul files
    working even before they are edited to include frontmatter.
    """
    candidates = [soul_path(), _SOUL_DEFAULT]
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("could not read %s: %s", path, exc)
            continue

        meta, body = _split_frontmatter(text)
        raw_name = meta.get("name")
        if raw_name is not None:
            name = _clean_agent_name(str(raw_name))
            if name:
                return name

        heading_match = re.search(r"^#\s+(.+?)(?:\s+[—-]\s+Soul)?\s*$", body, re.MULTILINE)
        if heading_match:
            name = _clean_agent_name(heading_match.group(1))
            if name:
                return name

    fallback = _clean_agent_name(default)
    return fallback or "IRIS"


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Split optional YAML frontmatter from a markdown document."""
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    closing_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        return {}, text

    import yaml

    meta_text = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :]).lstrip("\n")
    try:
        raw_meta = yaml.safe_load(meta_text) or {}
    except yaml.YAMLError as exc:
        logger.warning("soul frontmatter YAML error: %s", exc)
        return {}, body
    if not isinstance(raw_meta, dict):
        return {}, body
    return {str(key): value for key, value in raw_meta.items()}, body


def _clean_agent_name(value: str) -> str:
    """Return a prompt-safe single-line agent name."""
    cleaned = re.sub(r"[\r\n]+", " ", value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:80].strip()


# ---------------------------------------------------------------------------
# Long-term user profile (Layer 2)
# ---------------------------------------------------------------------------


def load_user_md() -> str | None:
    """Return the contents of ``~/.iris/workspace/USER.md`` or None if absent.

    Falls back to legacy ``~/.iris/memory/user.md`` path when the workspace
    copy is missing — keeps pre-2026-05-19 installs working until they
    migrate. Frontmatter (classification / load / egress) is stripped
    before returning so callers get just the body.
    """
    path = user_md_path() if user_md_path().exists() else _legacy_user_md_path()
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None
    _meta, body = _split_frontmatter(text)
    body = body.strip()
    return body or None


class UserIdentityBlock(BaseModel):
    """The ``identity:`` block of the USER.md frontmatter, as the owner writes it.

    The owner's own statement of who they are (ADR-0125): what owner-PII masking treats as
    theirs, and ``never_match`` for what it must never treat as theirs (a first name a
    contact shares). The only source of postal addresses. Lists of strings; anything else
    is a mistake the owner must hear about, never a crash.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    names: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()
    phones: tuple[str, ...] = ()
    addresses: tuple[str, ...] = ()
    handles: tuple[str, ...] = ()
    never_match: tuple[str, ...] = ()


def _user_md_path() -> Path:
    return user_md_path() if user_md_path().exists() else _legacy_user_md_path()


def load_user_identity() -> UserIdentityBlock:
    """The USER.md frontmatter's ``identity:`` block; empty when absent or invalid.

    ``load_user_md`` strips the frontmatter, so this is the one reader of it. A block that
    does not validate is logged at WARNING and read as empty: a typo must not take the
    profile (or a guard) down.
    """
    path = _user_md_path()
    if not path.exists():
        return UserIdentityBlock()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read %s: %s", path, exc)
        return UserIdentityBlock()
    meta, _body = _split_frontmatter(text)
    raw = meta.get("identity")
    if raw is None:
        return UserIdentityBlock()
    try:
        return UserIdentityBlock.model_validate(raw)
    except ValidationError as exc:
        logger.warning("%s: the identity: block is invalid and was ignored: %s", path, exc)
        return UserIdentityBlock()


def identity_document_paths() -> tuple[Path, ...]:
    """Every file the identity documents are read from, legacy fallbacks included.

    For a cheap change probe (stat each one): moving from the legacy file to the workspace
    copy changes the answer as surely as editing either.
    """
    return (
        soul_path(),
        _legacy_soul_path(),
        user_md_path(),
        _legacy_user_md_path(),
        agents_md_path(),
    )


def user_md_paths() -> tuple[Path, ...]:
    """The files USER.md is read from (workspace, then legacy)."""
    return (user_md_path(), _legacy_user_md_path())


def load_agents_md() -> str | None:
    """Return the body of ``~/.iris/workspace/AGENTS.md`` or None if absent.

    Per its ``load: on-demand`` frontmatter, AGENTS.md is NOT auto-injected
    into every system prompt — the model fetches it via the ``iris_doc``
    tool when it needs the registry (handoff decisions, "what agents
    exist", etc.). Frontmatter is stripped before returning.
    """
    if not agents_md_path().exists():
        return None
    try:
        text = agents_md_path().read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read %s: %s", agents_md_path(), exc)
        return None
    _meta, body = _split_frontmatter(text)
    body = body.strip()
    return body or None


def load_harness_md() -> str | None:
    """Return the body of ``docs/architecture/iris-harness.md`` or None.

    The harness doc is in-repo (ships with code) — not in the user's
    workspace. Loaded on-demand via the ``iris_doc("HARNESS")`` tool
    when the model needs the full operational reference (typically
    only during onboarding or deep debugging).
    """
    if not HARNESS_DOC_PATH.exists():
        return None
    try:
        text = HARNESS_DOC_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read %s: %s", HARNESS_DOC_PATH, exc)
        return None
    return text.strip() or None


_AUTO_HEADER = "## Auto-detected"
# Match the header ONLY as a real Markdown heading on its own line, so a
# backtick-quoted mention in the curated intro prose ("_Everything above the
# `## Auto-detected` heading..._") is never mistaken for the section boundary —
# that substring match truncated the user's curated head on every fact write.
_AUTO_HEADER_RE = re.compile(r"^##\s+Auto-detected\s*$", re.MULTILINE)
# Matches an existing auto-bullet so the appender can rewrite confidence in place.
_AUTO_BULLET_RE = re.compile(
    r"^- \*\*(?P<key>[^*]+)\*\*: (?P<value>.+?)  <!-- auto: confidence=(?P<conf>[0-9.]+) -->\s*$"
)


@invalidates_owner_identity
def append_user_fact_to_md(key: str, value: str, confidence: float) -> bool:
    """Upsert a fact into the ``## Auto-detected`` section of ``user.md``.

    Returns True when the file was written.  User-curated sections above the
    auto block are never touched — this function only reads/rewrites the
    auto block.  When ``user.md`` does not exist yet the function creates a
    minimal header so the first auto-fact has somewhere to live.
    """
    cleaned_key = _normalise_fact_key(key)
    cleaned_value = value.strip()
    if not cleaned_key or not cleaned_value:
        return False
    confidence = max(0.0, min(1.0, float(confidence)))

    memory_dir().mkdir(parents=True, exist_ok=True)
    if not user_md_path().exists():
        user_md_path().parent.mkdir(parents=True, exist_ok=True)
        user_md_path().write_text(
            "---\n"
            "classification: personal\n"
            "load: always\n"
            "egress: local-only\n"
            "---\n\n"
            "# User Profile\n\n"
            "_Everything above the Auto-detected heading below is curated by "
            "you and is never overwritten by the agent. Beneath that heading, "
            "the system appends facts it extracts from conversation._\n",
            encoding="utf-8",
        )

    text = user_md_path().read_text(encoding="utf-8")
    head, auto_lines, tail = _split_auto_section(text)
    if _curated_section_has_key(head, cleaned_key):
        cleaned_auto_lines = _remove_auto_bullet(auto_lines, cleaned_key)
        if cleaned_auto_lines == auto_lines:
            return False
        user_md_path().write_text(
            _compose_user_md(head, cleaned_auto_lines, tail),
            encoding="utf-8",
        )
        return True

    auto_lines = _upsert_auto_bullet(auto_lines, cleaned_key, cleaned_value, confidence)
    user_md_path().write_text(_compose_user_md(head, auto_lines, tail), encoding="utf-8")
    return True


def _split_auto_section(text: str) -> tuple[str, list[str], str]:
    """Split ``user.md`` into head, existing auto-bullets, and trailing sections.

    The header is matched only as a real heading on its own line
    (``_AUTO_HEADER_RE``), so a backtick-quoted mention inside the curated intro
    is not treated as the boundary — that bug truncated the user's curated head
    on every fact write.
    """
    match = _AUTO_HEADER_RE.search(text)
    if match is None:
        return text, [], ""
    head = text[: match.start()]
    tail = text[match.end() :]
    auto_lines: list[str] = []
    trailing_lines: list[str] = []
    raw_lines = tail.splitlines()
    for index, raw_line in enumerate(raw_lines):
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("## "):
            trailing_lines = raw_lines[index:]
            break
        if _AUTO_BULLET_RE.match(line):
            auto_lines.append(line)
    return head, auto_lines, "\n".join(trailing_lines)


def _compose_user_md(head: str, auto_lines: list[str], tail: str) -> str:
    """Render ``user.md`` while preserving user-owned head and trailing sections."""
    sections = [head.rstrip()]
    if auto_lines:
        sections.append("\n".join([_AUTO_HEADER, "", *auto_lines]))
    if tail.strip():
        sections.append(tail.strip())
    return "\n\n".join(section for section in sections if section.strip()).rstrip() + "\n"


def _curated_section_has_key(head: str, key: str) -> bool:
    """Return whether a user-curated section already owns ``key``."""
    for raw_line in head.splitlines():
        line = raw_line.strip()
        if not line or "<!-- auto:" in line:
            continue
        bold_match = re.match(r"^(?:[-*]\s*)?\*\*(?P<key>[^*]+)\*\*\s*:", line)
        if bold_match and _normalise_fact_key(bold_match.group("key")) == key:
            return True
        plain_match = re.match(r"^(?:[-*]\s*)?(?P<key>[A-Za-z][\w -]{0,80})\s*:", line)
        if plain_match and _normalise_fact_key(plain_match.group("key")) == key:
            return True
    return False


def _normalise_fact_key(key: str) -> str:
    """Normalize user-facing fact labels for conflict checks."""
    return re.sub(r"\s+", "_", key.strip().lower().replace("-", "_"))


def _remove_auto_bullet(existing: list[str], key: str) -> list[str]:
    """Remove any auto-generated bullet for ``key``."""
    out: list[str] = []
    for line in existing:
        match = _AUTO_BULLET_RE.match(line)
        if match and _normalise_fact_key(match.group("key")) == key:
            continue
        out.append(line)
    return out


def _upsert_auto_bullet(existing: list[str], key: str, value: str, confidence: float) -> list[str]:
    """Replace the bullet for ``key`` if present, otherwise append it."""
    new_bullet = f"- **{key}**: {value}  <!-- auto: confidence={confidence:.2f} -->"
    out: list[str] = []
    replaced = False
    for line in existing:
        match = _AUTO_BULLET_RE.match(line)
        if match and _normalise_fact_key(match.group("key")) == key:
            out.append(new_bullet)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(new_bullet)
    return out


@invalidates_owner_identity
def sync_user_facts_to_md(memory_store: UserFactSource) -> None:
    """Render SQLite ``user_facts`` into ``~/.iris/memory/user.md``.

    Only runs when the file is absent — once the user has edited their profile
    directly, the markdown is the canonical source and this is a no-op.
    """
    if user_md_path().exists():
        return

    # One bullet per key: a key with several values (two cards) lists them all.
    facts = memory_store.fetch_fact_projections()
    # Ensure the *target* dir exists. USER_MD_PATH lives under ``workspace/`` but
    # MEMORY_DIR is the legacy ``memory/`` dir; mkdir'ing only the latter left a
    # missing-parent crash when this is called outside ``bootstrap_identity_files``
    # (which pre-creates the workspace) — e.g. from ``reconcile_user_facts_md``.
    user_md_path().parent.mkdir(parents=True, exist_ok=True)

    if not facts:
        user_md_path().write_text(
            "# User Profile\n\n"
            "_This file is your long-term profile. IRIS reads it every turn — "
            "edit freely to correct or enrich what the agent knows about you._\n",
            encoding="utf-8",
        )
        logger.info("seeded empty %s", user_md_path())
        return

    lines = [
        "# User Profile",
        "",
        "_Auto-generated from prior sessions. Edit freely — your edits win on conflict._",
        "",
        _AUTO_HEADER,
        "",
    ]
    for fact in sorted(facts, key=lambda f: f.key):
        lines.append(
            f"- **{fact.key}**: {fact.value}  <!-- auto: confidence={fact.confidence:.2f} -->"
        )
    lines.append("")
    user_md_path().write_text("\n".join(lines), encoding="utf-8")
    logger.info("seeded %s with %d facts", user_md_path(), len(facts))


@invalidates_owner_identity
def drop_user_fact_from_md(key: str) -> bool:
    """Remove the auto-detected bullet for ``key`` from ``USER.md``.

    The projection-delete leg that pairs with ``append_user_fact_to_md``: when a
    fact is forgotten from the SQLite truth, its auto-bullet must leave the
    markdown projection too, or the prompt keeps asserting a fact the store no
    longer holds. User-curated head/tail sections are never touched. Returns
    True when a bullet was actually removed.
    """
    cleaned_key = _normalise_fact_key(key)
    if not cleaned_key or not user_md_path().exists():
        return False
    text = user_md_path().read_text(encoding="utf-8")
    head, auto_lines, tail = _split_auto_section(text)
    pruned = _remove_auto_bullet(auto_lines, cleaned_key)
    if pruned == auto_lines:
        return False
    user_md_path().write_text(_compose_user_md(head, pruned, tail), encoding="utf-8")
    return True


def read_auto_fact_keys() -> dict[str, tuple[str, float]]:
    """Parse the ``## Auto-detected`` block into ``{key: (value, confidence)}``.

    Reads the markdown projection so the coherence checker can diff it against
    the SQLite truth. Keys are normalised the same way writes normalise them, so
    a round-trip is stable. Returns an empty mapping when the file or block is
    absent.
    """
    out: dict[str, tuple[str, float]] = {}
    if not user_md_path().exists():
        return out
    _, auto_lines, _ = _split_auto_section(user_md_path().read_text(encoding="utf-8"))
    for line in auto_lines:
        match = _AUTO_BULLET_RE.match(line)
        if match is None:
            continue
        try:
            conf = float(match.group("conf"))
        except (TypeError, ValueError):
            conf = 0.0
        out[_normalise_fact_key(match.group("key"))] = (match.group("value").strip(), conf)
    return out


def curated_fact_keys() -> set[str]:
    """Return fact keys a human has curated by hand in ``USER.md``'s head section.

    These intentionally live *outside* the ``## Auto-detected`` block (the user's
    edits win on conflict), so the coherence checker must not flag them as
    "missing from the projection". Mirrors ``_curated_section_has_key``.
    """
    if not user_md_path().exists():
        return set()
    head, _, _ = _split_auto_section(user_md_path().read_text(encoding="utf-8"))
    keys: set[str] = set()
    for raw_line in head.splitlines():
        line = raw_line.strip()
        if not line or "<!-- auto:" in line:
            continue
        bold = re.match(r"^(?:[-*]\s*)?\*\*(?P<key>[^*]+)\*\*\s*:", line)
        if bold:
            keys.add(_normalise_fact_key(bold.group("key")))
            continue
        plain = re.match(r"^(?:[-*]\s*)?(?P<key>[A-Za-z][\w -]{0,80})\s*:", line)
        if plain:
            keys.add(_normalise_fact_key(plain.group("key")))
    return keys


@invalidates_owner_identity
def reconcile_user_facts_md(memory_store: UserFactSource) -> dict[str, int]:
    """Rewrite the ``## Auto-detected`` block to exactly match the SQLite truth.

    The projection-rebuild leg: unlike ``sync_user_facts_to_md`` (seeds only when
    the file is absent), this reconciles an *existing* profile — adding bullets
    for store facts that are missing, refreshing changed values, and dropping
    orphan bullets whose fact no longer exists in the store. Facts a human has
    curated by hand in the head section are left to the head (never duplicated
    into the auto block). User-owned head/tail are preserved verbatim.

    Returns ``{"added", "updated", "removed"}`` counts.
    """
    # One bullet per key: a key with several values (two cards) lists them all.
    facts = memory_store.fetch_fact_projections()
    if not user_md_path().exists():
        sync_user_facts_to_md(memory_store)
        return {"added": len(facts), "updated": 0, "removed": 0}

    text = user_md_path().read_text(encoding="utf-8")
    head, auto_lines, tail = _split_auto_section(text)
    existing = {
        _normalise_fact_key(m.group("key")): line
        for line in auto_lines
        if (m := _AUTO_BULLET_RE.match(line)) is not None
    }
    # Desired auto-bullets: every store fact NOT already owned by a curated head row.
    desired: dict[str, str] = {}
    for fact in sorted(facts, key=lambda f: f.key):
        norm = _normalise_fact_key(fact.key)
        if _curated_section_has_key(head, norm):
            continue
        conf = max(0.0, min(1.0, float(fact.confidence)))
        desired[norm] = (
            f"- **{fact.key}**: {fact.value.strip()}  <!-- auto: confidence={conf:.2f} -->"
        )

    added = sum(1 for k in desired if k not in existing)
    removed = sum(1 for k in existing if k not in desired)
    updated = sum(1 for k, line in desired.items() if k in existing and existing[k] != line)
    if not (added or removed or updated):
        return {"added": 0, "updated": 0, "removed": 0}

    new_auto = [desired[k] for k in sorted(desired)]
    user_md_path().write_text(_compose_user_md(head, new_auto, tail), encoding="utf-8")
    logger.info("reconciled %s auto-facts: +%d ~%d -%d", user_md_path(), added, updated, removed)
    return {"added": added, "updated": updated, "removed": removed}


# ---------------------------------------------------------------------------
# Active context (Layer 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveItem:
    """One checkbox entry parsed from ``active.md``."""

    index: int  # 1-based position in the file (in document order)
    done: bool
    text: str
    line_no: int  # 0-based line number within the file


def load_active_md() -> str | None:
    """Return the contents of ``~/.iris/memory/active.md`` or None if absent."""
    if not active_md_path().exists():
        return None
    try:
        text = active_md_path().read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("could not read %s: %s", active_md_path(), exc)
        return None
    return text or None


def list_active_items() -> list[ActiveItem]:
    """Parse ``active.md`` and return all checkbox entries in document order."""
    if not active_md_path().exists():
        return []
    items: list[ActiveItem] = []
    for line_no, line in enumerate(active_md_path().read_text(encoding="utf-8").splitlines()):
        match = _CHECKBOX_RE.match(line)
        if match is None:
            continue
        items.append(
            ActiveItem(
                index=len(items) + 1,
                done=match.group("mark").lower() == "x",
                text=match.group("text").strip(),
                line_no=line_no,
            )
        )
    return items


def add_active_item(text: str) -> ActiveItem:
    """Append a new in-flight item to ``active.md``.

    Creates the file (and the ``## In Flight`` section) if absent.  The line
    written is ``- [ ] YYYY-MM-DD — <text>`` so the date prefix is stable for
    later parsing and grep-based review.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("active item text must not be empty")

    today = local_today().isoformat()
    bullet = f"- [ ] {today} — {cleaned}"

    memory_dir().mkdir(parents=True, exist_ok=True)
    if not active_md_path().exists():
        active_md_path().write_text(
            "# Active Context\n\n## In Flight\n\n" + bullet + "\n",
            encoding="utf-8",
        )
    else:
        existing = active_md_path().read_text(encoding="utf-8")
        if not existing.endswith("\n"):
            existing += "\n"
        active_md_path().write_text(existing + bullet + "\n", encoding="utf-8")

    items = list_active_items()
    return items[-1] if items else ActiveItem(1, False, cleaned, 0)


def mark_active_done(index: int) -> ActiveItem | None:
    """Flip the Nth (1-based) checkbox in ``active.md`` to ``[x]``.

    Returns the updated item, or None if ``index`` is out of range or the
    item is already done.
    """
    if not active_md_path().exists():
        return None
    items = list_active_items()
    target = next((item for item in items if item.index == index), None)
    if target is None or target.done:
        return None

    lines = active_md_path().read_text(encoding="utf-8").splitlines()
    original = lines[target.line_no]
    lines[target.line_no] = original.replace("- [ ] ", "- [x] ", 1)
    active_md_path().write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ActiveItem(target.index, True, target.text, target.line_no)


# ---------------------------------------------------------------------------
# Episodic memory (Layer 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodicPattern:
    """One bullet entry parsed from ``episodic.md``."""

    pattern_id: str  # stable hash of the text — used as the ChromaDB doc id
    text: str
    line_no: int  # 0-based line number within the file


def load_episodic_md() -> str | None:
    """Return the contents of ``~/.iris/memory/episodic.md`` or None if absent."""
    if not episodic_md_path().exists():
        return None
    try:
        text = episodic_md_path().read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("could not read %s: %s", episodic_md_path(), exc)
        return None
    return text or None


def list_episodic_patterns() -> list[EpisodicPattern]:
    """Parse ``episodic.md`` and return searchable indexed entries.

    Plain bullets are indexed from pattern sections. Markdown table rows are
    indexed from routine/source/wiki sections. Other sections, including
    retrieval hints, are skipped so template guidance does not pollute recall.
    """
    if not episodic_md_path().exists():
        return []
    items: list[EpisodicPattern] = []
    lines = episodic_md_path().read_text(encoding="utf-8").splitlines()
    current_section = ""
    current_section_title = ""
    for line_no, line in enumerate(lines):
        if line.startswith("## "):
            current_section_title = line.lstrip("#").strip()
            current_section = _normalise_heading(current_section_title)
            continue
        if current_section not in _EPISODIC_INDEXED_SECTIONS:
            continue

        match = _BULLET_RE.match(line)
        if match is not None:
            text = match.group("text").strip()
            if not text:
                continue
            items.append(
                EpisodicPattern(
                    pattern_id=_episodic_id(text),
                    text=text,
                    line_no=line_no,
                )
            )
            continue

        next_line = lines[line_no + 1] if line_no + 1 < len(lines) else ""
        text = _episodic_table_row_text(
            line,
            current_section_title=current_section_title,
            next_line=next_line,
        )
        if not text:
            continue
        items.append(
            EpisodicPattern(
                pattern_id=_episodic_id(text),
                text=text,
                line_no=line_no,
            )
        )
    return items


def load_episodic_digest(*, max_chars: int = 500, max_items: int = 5) -> str | None:
    """Return a compact digest of the most recent episodic patterns for prompt injection.

    The digest is a small markdown block — at most ``max_items`` bullets and
    ``max_chars`` characters total — designed to fit inside the system
    prompt without crowding out other context.  Returns None when there are
    no patterns yet.
    """
    patterns = list_episodic_patterns()
    if not patterns:
        return None
    # Most recent first — the file is append-mostly, so reverse order.
    selected = list(reversed(patterns))[:max_items]
    header = "## What I've noticed about you"
    body_lines = [f"- {p.text}" for p in selected]
    digest = "\n".join([header, *body_lines])
    if len(digest) > max_chars:
        digest = digest[: max_chars - 1].rstrip() + "…"
    return digest


def append_episodic_pattern(text: str) -> EpisodicPattern:
    """Append a new pattern bullet to ``episodic.md`` and return it.

    Creates the file (and the ``## Patterns`` section) if absent.  Bullet is
    written as ``- YYYY-MM-DD — <text>`` so the date prefix gives temporal
    ordering for ``load_episodic_digest``.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("episodic pattern text must not be empty")

    today = local_today().isoformat()
    bullet_text = f"{today} — {cleaned}"
    bullet = f"- {bullet_text}"

    memory_dir().mkdir(parents=True, exist_ok=True)
    if not episodic_md_path().exists():
        episodic_md_path().write_text(
            "# Episodic Memory\n\n## Long-Term Patterns\n\n" + bullet + "\n",
            encoding="utf-8",
        )
    else:
        existing = episodic_md_path().read_text(encoding="utf-8")
        episodic_md_path().write_text(_insert_episodic_bullet(existing, bullet), encoding="utf-8")

    items = list_episodic_patterns()
    return items[-1] if items else EpisodicPattern(_episodic_id(bullet_text), bullet_text, 0)


def _episodic_id(text: str) -> str:
    """Stable short id for an episodic pattern — first 16 chars of sha256 hex."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalise_heading(value: str) -> str:
    """Return a lowercase heading key for section-aware episodic parsing."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _episodic_table_row_text(
    line: str,
    *,
    current_section_title: str,
    next_line: str,
) -> str:
    """Convert a markdown table data row into a searchable episodic entry."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return ""
    if _is_table_separator(stripped) or _is_table_separator(next_line.strip()):
        return ""
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    cells = [cell for cell in cells if cell]
    if len(cells) < 2:
        return ""
    return f"{current_section_title}: " + " | ".join(cells)


def _is_table_separator(line: str) -> bool:
    """Return True for markdown table separator rows."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def _insert_episodic_bullet(existing: str, bullet: str) -> str:
    """Insert a pattern bullet into the pattern section without moving other sections."""
    lines = existing.splitlines()
    for heading in ("## Long-Term Patterns", "## Patterns"):
        heading_key = heading.lower()
        for index, line in enumerate(lines):
            if line.strip().lower() != heading_key:
                continue
            insert_at = len(lines)
            for next_index in range(index + 1, len(lines)):
                if lines[next_index].startswith("## "):
                    insert_at = next_index
                    break
            before = lines[:insert_at]
            after = lines[insert_at:]
            if before and before[-1].strip():
                before.append("")
            before.append(bullet)
            if after and after[0].strip():
                before.append("")
            return "\n".join([*before, *after]).rstrip() + "\n"
    prefix = existing.rstrip()
    return (
        f"{prefix}\n\n## Long-Term Patterns\n\n{bullet}\n"
        if prefix
        else f"## Long-Term Patterns\n\n{bullet}\n"
    )


# ---------------------------------------------------------------------------
# Behaviors (parallel to memory — operational recipes loaded on intent match)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Behavior:
    """One operational recipe parsed from ``~/.iris/behaviors/*.md``."""

    name: str
    description: str
    match_intents: tuple[str, ...]
    match_keywords: tuple[str, ...]
    body: str
    path: Path
    # One line for the L1 pointer and for search hits, so the agent can see that a
    # lesson exists without spending the whole recipe's tokens on it. Falls back to
    # ``description``.
    summary: str = ""

    @property
    def headline(self) -> str:
        return self.summary or self.description or self.name


# Relevance weights for behavior matching, tunable without touching code.
#
# A SPECIFIC intent (calendar, finance, email) still matches on its own — that is what
# `match_intents` is for. A CATCH-ALL intent does not: `reminders.md` lists
# `match_intents: [calendar, general]`, and `general` is what most turns carry, so every
# other turn was injecting 2K characters about reminders and no other behavior could
# ever be seen.
_INTENT_WEIGHT = float(os.environ.get("IRIS_BEHAVIOR_INTENT_WEIGHT", "2.0"))
_BROAD_INTENT_WEIGHT = float(os.environ.get("IRIS_BEHAVIOR_BROAD_INTENT_WEIGHT", "0.5"))
_KEYWORD_WEIGHT = float(os.environ.get("IRIS_BEHAVIOR_KEYWORD_WEIGHT", "1.0"))
_MATCH_THRESHOLD = float(os.environ.get("IRIS_BEHAVIOR_MATCH_THRESHOLD", "2.0"))
_BROAD_INTENTS = frozenset(
    i.strip().lower()
    for i in os.environ.get("IRIS_BEHAVIOR_BROAD_INTENTS", "general,system,chat,other").split(",")
    if i.strip()
)


def list_behaviors() -> list[Behavior]:
    """Return every behavior file under ``~/.iris/behaviors/`` in name order.

    Files without valid frontmatter are skipped with a warning so a malformed
    recipe never crashes prompt assembly.
    """
    if not behaviors_dir().exists():
        return []
    behaviors: list[Behavior] = []
    for path in sorted(behaviors_dir().glob("*.md")):
        behavior = _parse_behavior(path)
        if behavior is not None:
            behaviors.append(behavior)
    return behaviors


def score_behaviors(intent: str, query: str) -> list[tuple[Behavior, float]]:
    """Every behavior with a relevance score for this turn, best first.

    Matching used to return the FIRST behavior whose intent or keyword matched, and
    ``reminders.md`` lists ``match_intents: [calendar, general]`` — so almost every
    general turn injected 2K characters about reminders whether or not the user asked
    about one, and no other behavior could ever be seen.

    Score: a specific intent match is worth ``_INTENT_WEIGHT`` (a catch-all intent like
    ``general`` is worth much less); each keyword hit adds ``_KEYWORD_WEIGHT``, doubled
    when the keyword appears as a whole word — so "remind" inside "reminder" still
    helps, and "remind me" scores far higher.
    """
    behaviors = list_behaviors()
    if not behaviors:
        return []
    intent_norm = (intent or "").strip().lower()
    query_norm = (query or "").lower()
    query_words = set(re.findall(r"[a-z0-9']+", query_norm))

    scored: list[tuple[Behavior, float]] = []
    for behavior in behaviors:
        score = 0.0
        if intent_norm and intent_norm in behavior.match_intents:
            score += _BROAD_INTENT_WEIGHT if intent_norm in _BROAD_INTENTS else _INTENT_WEIGHT
        for keyword in behavior.match_keywords:
            if not keyword:
                continue
            if keyword in query_norm:
                score += _KEYWORD_WEIGHT
                if keyword in query_words or f" {keyword} " in f" {query_norm} ":
                    score += _KEYWORD_WEIGHT
        if score > 0:
            scored.append((behavior, score))
    scored.sort(key=lambda pair: (-pair[1], pair[0].name))
    return scored


def match_behavior(intent: str, query: str) -> Behavior | None:
    """The single best behavior for this turn, or None below the relevance threshold.

    An intent match alone no longer clears the bar: the turn has to look like the one
    the behavior is about.
    """
    scored = score_behaviors(intent, query)
    if not scored:
        return None
    best, score = scored[0]
    return best if score >= _MATCH_THRESHOLD else None


def other_matching_behaviors(intent: str, query: str, *, limit: int = 3) -> list[Behavior]:
    """Behaviors that also look relevant — the L1 pointer, one line each."""
    scored = score_behaviors(intent, query)
    if not scored:
        return []
    best = match_behavior(intent, query)
    return [b for b, _score in scored if best is None or b.name != best.name][:limit]


def _parse_behavior(path: Path) -> Behavior | None:
    """Parse a behavior markdown file with YAML frontmatter."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read behavior %s: %s", path, exc)
        return None

    if not text.startswith("---"):
        logger.warning("behavior %s missing frontmatter — skipping", path)
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        logger.warning("behavior %s has malformed frontmatter — skipping", path)
        return None

    import yaml

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        logger.warning("behavior %s frontmatter YAML error: %s", path, exc)
        return None

    if not isinstance(meta, dict):
        logger.warning("behavior %s frontmatter not a mapping — skipping", path)
        return None

    name = str(meta.get("name") or path.stem).strip()
    description = str(meta.get("description") or "").strip()
    intents = _normalise_str_list(meta.get("match_intents"))
    keywords = _normalise_str_list(meta.get("match_keywords"))
    body = parts[2].strip()

    return Behavior(
        name=name,
        description=description,
        match_intents=intents,
        match_keywords=keywords,
        body=body,
        path=path,
        summary=str(meta.get("summary") or "").strip(),
    )


def write_behavior(
    name: str,
    body: str,
    *,
    match_keywords: tuple[str, ...] = (),
    match_intents: tuple[str, ...] = (),
    description: str = "",
    source: str = "taught",
) -> Path:
    """Persist (or update) a behavior recipe under ``~/.iris/behaviors/<slug>.md``.

    This is the write side of the same store ``list_behaviors``/``match_behavior``
    read from, so a behavior written here is matchable on the very next turn. The
    filename is derived from ``name`` (slugified), so re-teaching the same rule
    overwrites it in place rather than accumulating near-duplicates. ``source`` is
    recorded in frontmatter for provenance ("taught" = captured from an explicit
    in-chat instruction) and to distinguish user-taught recipes from shipped
    defaults when listing.
    """
    behaviors_dir().mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "behavior"
    path = behaviors_dir() / f"{slug}.md"

    import yaml

    meta = {
        "name": name.strip() or slug,
        "description": description.strip(),
        "match_intents": list(match_intents),
        "match_keywords": list(match_keywords),
        "source": source,
    }
    frontmatter = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    content = f"---\n{frontmatter}\n---\n\n{body.strip()}\n"
    path.write_text(content, encoding="utf-8")
    logger.info("behavior written: %s (keywords=%s, source=%s)", path.name, match_keywords, source)
    return path


def _normalise_str_list(raw: object) -> tuple[str, ...]:
    """Coerce frontmatter list/scalar into a lowercased tuple of strings."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw]
    else:
        return ()
    return tuple(item.strip().lower() for item in items if str(item).strip())


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_SOUL_LAYERS_CACHE")
