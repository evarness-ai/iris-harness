"""What the morning digest shows and when (loop-proof plan D4, ADR-0122).

``config/digest.yaml`` is the default. The owner's changes, from Settings -> Digest or
from chat ("add X to my digest", the ``configure_brief`` tool), are saved in the
settings store (ADR-0120) under section ``digest``, key ``config``: only the fields that
differ from the file, the same way the health watch keeps its edits. On the VM the
file ships in the image and a deploy replaces it; the store is on the data volume.

Sections are an ordered list of the brief's tool-slot names. The owner's list is kept
as two lists, ``sections`` (on, in order) and ``sections_off``, so a section the file
gains later can be told apart from one the owner turned off: a section in neither list
is new to the owner and slots in after the section it follows in the file.

``load_digest_settings`` also moves ``data/brief_prefs.json`` (the brief preferences
before this store existed) into the store once, and marks that done
(key ``brief_prefs_migration``). It never raises: a broken file or store gives the
defaults and a warning, because a digest with default settings beats no digest.

The file also lays the digest out (digest v5): ``groups`` put the sections under a few
headed groups (Today, Money, ...), ``channels`` say how each channel renders them, and
``news_groups`` split the news into slots (AI / Tech, Global, Local), each its own
topics. Groups and channels are the file's only; the owner edits the sections, the
news groups' topics and ``news_local_area``. :func:`group_sections` is what a renderer
asks: the rendered sections under their groups.

``renamed_sections`` maps an old section name to the one(s) that replaced it
(``news`` -> the three news slots). It applies to the owner's saved settings whenever
they are read, and to the brief_prefs.json move.

``iris_timezone`` is the one reader of ``IRIS_TZ``: the owner's local zone, which
compose also uses as the container clock (``TZ``).
"""

from __future__ import annotations

import json
import logging
import re
import string
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iris_harness.foundation.clock import iris_timezone
from iris_harness.foundation.paths import config_dir as resolve_config_dir
from iris_harness.foundation.process_state import track_globals
from iris_harness.services.digest.expiry import ExpiryPolicy, parse_expiry

if TYPE_CHECKING:
    from iris_harness.foundation.settings.store import SettingsStore

logger = logging.getLogger(__name__)

SETTINGS_SECTION = "digest"
KEY = "config"
MIGRATION_KEY = "brief_prefs_migration"
DIGEST_FILE = "digest.yaml"
BRIEF_PREFS_FILE = "brief_prefs.json"
MIGRATION_ACTOR = "migration:brief_prefs"

# Every field the owner may change, including ``sections_off``, which is how the store
# remembers a section turned off (DigestSettings carries only the effective list).
EDITABLE: tuple[str, ...] = (
    "enabled",
    "time",
    "channel",
    "sections",
    "sections_off",
    "section_config",
    "news_topics",
    "news_groups",
    "news_local_area",
    "news_sources",
    "news_language",
    "focus_categories",
    "focus_limit",
    "focus_per_account",
)

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CHANNEL_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_DOMAIN_RE = re.compile(r"^(?=.{4,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_CATEGORY_RE = re.compile(r"^[a-z0-9][a-z0-9_/-]{0,127}$")
_LANGUAGE_RE = re.compile(r"^[a-z]{2}$")  # ISO 639-1
ANY_LANGUAGE = "any"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
MAX_TOPICS = 12
MAX_TOPIC_LEN = 80
MAX_SOURCES = 20
MAX_CATEGORIES = 20
FOCUS_LIMIT_RANGE = (1, 20)
# ``focus_per_account``: the newest this many from each inbox, before ``focus_limit``.
FOCUS_PER_ACCOUNT_RANGE = (1, 20)
# ``section_config.<section>.line_cap``: how many lines a section shows.
LINE_CAP_RANGE = (1, 50)
MAX_AREA_LEN = 80
# The one placeholder a news group's title and topics may hold.
AREA_PLACEHOLDER = "news_local_area"
DEFAULT_LOCAL_AREA = "St. Louis"


@dataclass(frozen=True)
class DigestGroup:
    """A headed group of sections (``digest.yaml`` ``groups``), e.g. Today or Money."""

    id: str
    title: str
    icon: str = ""
    sections: tuple[str, ...] = ()
    push: str = ""  # format string over section item counts, "" = no push line


# Sections in no group go last, under this one (the contract with the renderers).
MORE_GROUP = DigestGroup(id="more", title="More")


@dataclass(frozen=True)
class DigestSettings:
    """The digest settings in effect: the file's defaults with the owner's changes."""

    enabled: bool = True
    time: str = "07:00"  # HH:MM local (IRIS_TZ)
    channel: str = "all"
    sections: tuple[str, ...] = ()  # ordered brief tool-slot names; () = full manifest
    section_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    news_topics: tuple[str, ...] = ("AI", "world")
    news_sources: tuple[str, ...] = ()
    # ISO 639-1 code the news is in (provider hint + a script filter); "any" = no filter.
    news_language: str = "en"
    focus_categories: tuple[str, ...] = ()
    focus_limit: int = 10
    # The newest this many per inbox, then newest-first overall up to ``focus_limit``.
    focus_per_account: int = 5
    # The layout (the file's only): the groups in order, and each channel's knobs.
    groups: tuple[DigestGroup, ...] = ()
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    # The news slots: slot name -> {"title": str, "topics": tuple[str, ...]}; the title
    # and topics may hold ``{news_local_area}`` (see :func:`news_group_topics`).
    news_groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    news_local_area: str = DEFAULT_LOCAL_AREA
    # Closes the digest, after every group (the D17 footer); never in a group.
    footer_sections: tuple[str, ...] = ("learned_yesterday",)
    # When dated items age out of the digest (the file's ``expiry``; see expiry.py).
    expiry: ExpiryPolicy = field(default_factory=ExpiryPolicy)


@dataclass(frozen=True)
class DigestDefaults:
    """``config/digest.yaml``, read: the defaults, plus what the migration needs."""

    settings: DigestSettings
    # Always shown and always last, whatever the owner's list says (the D17 footer).
    locked_sections: tuple[str, ...] = ()
    # Known, off until the owner turns them on (the file's ``sections_off``).
    sections_off: tuple[str, ...] = ()
    # brief_prefs.json: sections it predates, old section names -> new ones, and
    # sections it had on that the move leaves off (the owner turns them back on).
    new_sections: tuple[str, ...] = ()
    # Old section name -> the section(s) that replaced it (``renamed_sections``): the
    # owner's saved settings and brief_prefs.json are read under today's names.
    renamed: dict[str, tuple[str, ...]] = field(default_factory=dict)
    dropped_sections: tuple[str, ...] = ()
    # False when digest.yaml was missing or unreadable (built-in values). The one-time
    # brief_prefs.json move waits for the real file: without its renames and new
    # sections it would record a wrong layout as done, for good.
    from_file: bool = True


class DigestEditError(ValueError):
    """A change that does not fit a digest field."""


# -- validation ------------------------------------------------------------------


def _str_list(key: str, raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list | tuple):
        raise DigestEditError(f"{key} must be a list")
    return [str(v).strip() for v in raw if str(v).strip()]


def _dedupe(items: list[str], *, fold: bool = False) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        k = item.lower() if fold else item
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


def _section_names(key: str, raw: Any, known: frozenset[str] | None) -> list[str]:
    names = [n.lower() for n in _str_list(key, raw)]
    for name in names:
        if not _NAME_RE.match(name):
            raise DigestEditError(f"{key}: {name!r} is not a section name")
        if known is not None and name not in known:
            raise DigestEditError(f"{key}: no section {name!r}")
    if len(set(names)) != len(names):
        raise DigestEditError(f"{key} lists a section twice")
    return names


def _line_cap(section: str, raw: Any) -> int:
    low, high = LINE_CAP_RANGE
    try:
        if isinstance(raw, bool):
            raise TypeError
        number = int(str(raw).strip())
    except (TypeError, ValueError):
        raise DigestEditError(f"section_config: {section}.line_cap must be a number") from None
    if not low <= number <= high:
        raise DigestEditError(f"section_config: {section}.line_cap must be {low}-{high}")
    return number


def _placeholders(key: str, text: str) -> None:
    """``text`` may hold ``{news_local_area}`` and nothing else in braces."""
    try:
        names = [f for _, f, _, _ in string.Formatter().parse(text) if f is not None]
    except ValueError:
        raise DigestEditError(f"{key}: {text!r} has an unmatched brace") from None
    bad = [n for n in names if n != AREA_PLACEHOLDER]
    if bad:
        raise DigestEditError(
            f"{key}: {{{bad[0]}}} is not a placeholder (only {{{AREA_PLACEHOLDER}}})"
        )


def _topics(key: str, raw: Any) -> list[str]:
    topics = _dedupe(_str_list(key, raw), fold=True)
    if len(topics) > MAX_TOPICS:
        raise DigestEditError(f"{key}: at most {MAX_TOPICS}")
    if any(len(t) > MAX_TOPIC_LEN for t in topics):
        raise DigestEditError(f"{key}: a topic is at most {MAX_TOPIC_LEN} characters")
    for topic in topics:
        _placeholders(key, topic)
    return topics


def _news_groups(raw: Any) -> dict[str, dict[str, Any]]:
    """``news_groups``: slot -> {title?, topics?}; either key may be left out (the
    file's stands)."""
    if not isinstance(raw, dict):
        raise DigestEditError("news_groups must map a news section to its title and topics")
    out: dict[str, dict[str, Any]] = {}
    for slot, spec in raw.items():
        name = _section_names("news_groups", [slot], None)[0]
        if not isinstance(spec, dict):
            raise DigestEditError(f"news_groups: {name} must be a mapping (title, topics)")
        unknown = set(spec) - {"title", "topics"}
        if unknown:
            raise DigestEditError(f"news_groups: {name}.{sorted(unknown)[0]} is not a setting")
        group: dict[str, Any] = {}
        if "title" in spec:
            title = " ".join(str(spec["title"] or "").split())
            if not title or len(title) > MAX_TOPIC_LEN:
                raise DigestEditError(
                    f"news_groups: {name}.title must be 1-{MAX_TOPIC_LEN} characters"
                )
            _placeholders(f"news_groups: {name}.title", title)
            group["title"] = title
        if "topics" in spec:
            group["topics"] = _topics(f"news_groups: {name}.topics", spec["topics"])
        out[name] = group
    return out


def _area(raw: Any) -> str:
    text = " ".join(str(raw if raw is not None else "").split())
    if not text:
        raise DigestEditError("news_local_area must name a place (e.g. St. Louis)")
    if len(text) > MAX_AREA_LEN:
        raise DigestEditError(f"news_local_area: at most {MAX_AREA_LEN} characters")
    if "{" in text or "}" in text:
        raise DigestEditError("news_local_area cannot hold braces")
    return text


def _domain(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"^[a-z]+://", "", text).split("/", 1)[0].split(":", 1)[0]
    text = text.removeprefix("www.")
    if not _DOMAIN_RE.match(text):
        raise DigestEditError(f"news_sources: {raw!r} is not a domain (e.g. bbc.com)")
    return text


def validate(
    changes: dict[str, Any], *, known_sections: frozenset[str] | None = None
) -> dict[str, Any]:
    """``changes`` checked and normalized to their stored (JSON) form.

    ``known_sections`` limits section names to those the digest knows (the file's
    list plus the owner's); ``None`` accepts any well-formed name (chat adding a slot
    the brief skill has and the file does not).
    """
    out: dict[str, Any] = {}
    for key, raw in changes.items():
        if key not in EDITABLE:
            raise DigestEditError(f"{key!r} is not a digest setting")
        if key == "enabled":
            text = str(raw).strip().lower()
            if isinstance(raw, bool):
                out[key] = raw
            elif text in _TRUE or text in _FALSE:
                out[key] = text in _TRUE
            else:
                raise DigestEditError("enabled must be on or off")
        elif key == "time":
            match = _TIME_RE.match(str(raw).strip())
            if match is None:
                raise DigestEditError("time must be HH:MM, 24-hour (e.g. 07:00)")
            out[key] = f"{int(match.group(1)):02d}:{match.group(2)}"
        elif key == "channel":
            text = str(raw).strip().lower()
            if text == "all":
                out[key] = "all"
                continue
            names = _dedupe([p.strip() for p in text.split(",") if p.strip()])
            if not names or not all(_CHANNEL_RE.match(n) for n in names):
                raise DigestEditError("channel must be 'all' or channel names, comma-separated")
            out[key] = ",".join(names)
        elif key in ("sections", "sections_off"):
            out[key] = _section_names(key, raw, known_sections)
        elif key == "section_config":
            if not isinstance(raw, dict):
                raise DigestEditError("section_config must map a section to its settings")
            config: dict[str, dict[str, Any]] = {}
            for name, knobs in raw.items():
                _section_names("section_config", [name], known_sections)
                if not isinstance(knobs, dict):
                    raise DigestEditError(f"section_config: {name} must be a mapping")
                try:
                    json.dumps(knobs)
                except (TypeError, ValueError):
                    raise DigestEditError(f"section_config: {name} is not plain data") from None
                knobs = dict(knobs)
                if "line_cap" in knobs:
                    knobs["line_cap"] = _line_cap(name, knobs["line_cap"])
                # Kept when empty: {} clears the file's knobs for that section.
                config[str(name).lower()] = knobs
            out[key] = config
        elif key == "news_topics":
            out[key] = _topics(key, raw)
        elif key == "news_groups":
            out[key] = _news_groups(raw)
        elif key == "news_local_area":
            out[key] = _area(raw)
        elif key == "news_sources":
            sources = _dedupe([_domain(s) for s in _str_list(key, raw)])
            if len(sources) > MAX_SOURCES:
                raise DigestEditError(f"news_sources: at most {MAX_SOURCES}")
            out[key] = sources
        elif key == "news_language":
            text = str(raw if raw is not None else "").strip().lower() or ANY_LANGUAGE
            if text != ANY_LANGUAGE and not _LANGUAGE_RE.match(text):
                raise DigestEditError(
                    "news_language must be a two-letter language code (e.g. en) or 'any'"
                )
            out[key] = text
        elif key == "focus_categories":
            cats = _dedupe([c.lower().strip("/") for c in _str_list(key, raw)])
            if len(cats) > MAX_CATEGORIES:
                raise DigestEditError(f"focus_categories: at most {MAX_CATEGORIES}")
            bad = [c for c in cats if not _CATEGORY_RE.match(c)]
            if bad:
                raise DigestEditError(f"focus_categories: {bad[0]!r} is not a category path")
            out[key] = cats
        elif key in ("focus_limit", "focus_per_account"):
            low, high = FOCUS_LIMIT_RANGE if key == "focus_limit" else FOCUS_PER_ACCOUNT_RANGE
            try:
                if isinstance(raw, bool):
                    raise TypeError
                number = int(str(raw).strip())
            except (TypeError, ValueError):
                raise DigestEditError(f"{key} must be a whole number") from None
            if not low <= number <= high:
                raise DigestEditError(f"{key} must be between {low} and {high}")
            out[key] = number
    return out


# -- the file ---------------------------------------------------------------------


def _config_dir(config_dir: Path | None) -> Path:
    return config_dir or resolve_config_dir()


def load_defaults(config_dir: Path | None = None) -> DigestDefaults:
    """``digest.yaml`` as defaults; a missing or broken file (or field) gives the
    built-in value for it, with a warning."""
    path = _config_dir(config_dir) / DIGEST_FILE
    raw: dict[str, Any] = {}
    from_file = False
    if path.exists():
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
                from_file = True
            else:
                logger.warning("digest: %s is not a mapping; using built-in defaults", path)
        except Exception as exc:  # noqa: BLE001 — a bad file must not stop the digest
            logger.warning("digest: could not read %s (%s); using built-in defaults", path, exc)
    else:
        logger.warning("digest: no %s; using built-in defaults", path)
    fields_ = _good_fields(raw, source=str(path), known_sections=None)
    base = DigestSettings()
    settings = replace(
        base,
        **{k: _as_field(k, v) for k, v in fields_.items() if k not in ("sections_off",)},
    )
    locked: tuple[str, ...] = ()
    try:
        locked = tuple(_section_names("locked_sections", raw.get("locked_sections") or [], None))
    except DigestEditError as exc:
        logger.warning("digest: ignoring locked_sections from %s (%s)", path, exc)
    # The locked sections are the footer: always shown, always last, in no group.
    footer = locked if from_file else base.footer_sections
    settings = replace(
        settings,
        footer_sections=footer,
        groups=_groups(raw.get("groups"), footer, source=str(path)),
        channels=_channels(raw.get("channels"), source=str(path)),
        news_groups={
            slot: {"title": spec.get("title") or slot, "topics": tuple(spec.get("topics") or ())}
            for slot, spec in settings.news_groups.items()
        },
    )
    try:
        settings = replace(settings, expiry=parse_expiry(raw.get("expiry")))
    except ValueError as exc:
        logger.warning("digest: ignoring expiry from %s (%s)", path, exc)
    # Off by default: known (Settings lists them to turn on), not shown. Listed in both
    # ``sections`` and ``sections_off``, a section is off; a locked one is never off.
    default_off = tuple(s for s in fields_.get("sections_off") or () if s not in locked)
    if default_off:
        settings = replace(
            settings, sections=tuple(s for s in settings.sections if s not in default_off)
        )
    migration = raw.get("brief_prefs_migration") or {}
    new_sections: tuple[str, ...] = ()
    dropped_sections: tuple[str, ...] = ()
    renamed: dict[str, tuple[str, ...]] = {}
    if isinstance(migration, dict):
        try:
            new_sections = tuple(
                _section_names("new_sections", migration.get("new_sections") or [], None)
            )
            dropped_sections = tuple(
                _section_names("dropped_sections", migration.get("dropped_sections") or [], None)
            )
            # The older spelling of ``renamed_sections``, read so an old file still moves.
            renamed = _renames(migration.get("renamed"))
        except DigestEditError as exc:
            logger.warning("digest: %s brief_prefs_migration ignored (%s)", path, exc)
            new_sections, dropped_sections, renamed = (), (), {}
    try:
        renamed = {**renamed, **_renames(raw.get("renamed_sections"))}
    except DigestEditError as exc:
        logger.warning("digest: %s renamed_sections ignored (%s)", path, exc)
    defaults = DigestDefaults(
        settings=settings,
        locked_sections=locked,
        sections_off=default_off,
        new_sections=new_sections,
        renamed=renamed,
        dropped_sections=tuple(s for s in dropped_sections if s not in locked),
        from_file=from_file,
    )
    return replace(defaults, settings=_with_locked(defaults, settings))


def _renames(raw: Any) -> dict[str, tuple[str, ...]]:
    """``old: new`` or ``old: [new, ...]`` (one section split into several)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise DigestEditError("renamed must map an old section name to its new name(s)")
    out: dict[str, tuple[str, ...]] = {}
    for old, new in raw.items():
        names = [new] if isinstance(new, str) else new
        old_name = _section_names("renamed", [old], None)[0]
        targets = tuple(_section_names(f"renamed.{old_name}", names, None))
        if not targets or old_name in targets:
            raise DigestEditError(f"renamed: {old_name} needs a new name")
        out[old_name] = targets
    return out


def _groups(raw: Any, footer: tuple[str, ...], *, source: str) -> tuple[DigestGroup, ...]:
    """``groups`` from the file. A bad group is skipped with a warning; a section a group
    lists twice, or an earlier group already has, stays with the first; the footer is
    never in a group; a ``push`` naming a section outside its group is dropped."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        logger.warning("digest: ignoring groups from %s (not a list)", source)
        return ()
    groups: list[DigestGroup] = []
    claimed: set[str] = set(footer)
    for entry in raw:
        try:
            if not isinstance(entry, dict):
                raise DigestEditError("a group must be a mapping")
            gid = _section_names("groups.id", [entry.get("id") or ""], None)[0]
            if gid in {g.id for g in groups} or gid == MORE_GROUP.id:
                raise DigestEditError(f"group id {gid!r} is taken")
            title = " ".join(str(entry.get("title") or "").split())
            if not title:
                raise DigestEditError(f"group {gid} needs a title")
            names = [
                n.lower() for n in _str_list(f"groups.{gid}.sections", entry.get("sections") or [])
            ]
            for name in names:
                if not _NAME_RE.match(name):
                    raise DigestEditError(f"groups.{gid}: {name!r} is not a section name")
        except DigestEditError as exc:
            logger.warning("digest: ignoring a group in %s (%s)", source, exc)
            continue
        taken = [n for n in names if n in claimed]
        if taken:
            logger.warning("digest: group %s leaves out %s (footer or another group's)", gid, taken)
        sections = tuple(dict.fromkeys(n for n in names if n not in claimed))
        claimed.update(sections)
        push = str(entry.get("push") or "")
        if push:
            try:
                fields_ = [f for _, f, _, _ in string.Formatter().parse(push) if f is not None]
                outside = [f for f in fields_ if f not in sections]
                if outside:
                    raise ValueError(f"{{{outside[0]}}} is not a section of the group")
            except ValueError as exc:
                logger.warning("digest: group %s push line dropped (%s)", gid, exc)
                push = ""
        groups.append(
            DigestGroup(
                id=gid,
                title=title,
                icon=str(entry.get("icon") or ""),
                sections=sections,
                push=push,
            )
        )
    return tuple(groups)


def _channels(raw: Any, *, source: str) -> dict[str, dict[str, Any]]:
    """``channels`` from the file: channel name -> its rendering knobs (plain data)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning("digest: ignoring channels from %s (not a mapping)", source)
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, knobs in raw.items():
        key = str(name).strip().lower()
        try:
            if not _CHANNEL_RE.match(key) or not isinstance(knobs, dict):
                raise ValueError("a channel name mapped to its knobs")
            json.dumps(knobs)
        except (TypeError, ValueError) as exc:
            logger.warning("digest: ignoring channel %r from %s (%s)", name, source, exc)
            continue
        out[key] = dict(knobs)
    return out


def _with_locked(defaults: DigestDefaults, settings: DigestSettings) -> DigestSettings:
    """``settings`` with every locked section shown, last, in the file's order. An
    empty list (the full manifest) stays empty: the manifest has them."""
    if not settings.sections or not defaults.locked_sections:
        return settings
    locked = defaults.locked_sections
    rest = tuple(s for s in settings.sections if s not in locked)
    return replace(settings, sections=rest + locked)


def _good_fields(
    raw: dict[str, Any], *, source: str, known_sections: frozenset[str] | None
) -> dict[str, Any]:
    """The fields of ``raw`` that validate; each bad one is dropped with a warning."""
    good: dict[str, Any] = {}
    for key in EDITABLE:
        if key not in raw or raw[key] is None:
            continue
        try:
            good.update(validate({key: raw[key]}, known_sections=known_sections))
        except DigestEditError as exc:
            logger.warning("digest: ignoring %s from %s (%s)", key, source, exc)
    return good


def _as_field(key: str, value: Any) -> Any:
    """A stored (JSON) value as the DigestSettings field holds it."""
    if isinstance(value, list):
        return tuple(value)
    if key == "section_config":
        return {k: dict(v) for k, v in value.items()}
    if key == "news_groups":
        return {
            k: {**v, **({"topics": tuple(v["topics"])} if "topics" in v else {})}
            for k, v in value.items()
        }
    return value


# Fields merged entry by entry over the file's: the owner's entry for a section (or a
# news group) replaces the file's for that one, and the file's others still apply, so a
# section the file gains later brings its knobs to an owner who changed another one.
_PER_ENTRY = ("section_config", "news_groups")


# -- merging ----------------------------------------------------------------------


def merge_sections(
    defaults: tuple[str, ...], on: list[str] | None, off: list[str] | None
) -> tuple[str, ...]:
    """The effective section order.

    ``on`` is the owner's ordered list (``None``: the file's), ``off`` the sections
    they turned off. A file section in neither list is new to the owner. It goes just
    after the nearest section before it in the file that is shown; one with nothing
    shown after it in the file goes last (so a new footer stays the footer), and one
    with nothing shown before it goes first.
    """
    off_set = set(off or ())
    result = [s for s in (defaults if on is None else on) if s not in off_set]
    seen = set(result) | off_set
    for index, name in enumerate(defaults):
        if name in seen:
            continue
        if not any(f in result for f in defaults[index + 1 :]):
            at = len(result)
        else:
            at = 0
            for previous in reversed(defaults[:index]):
                if previous in result:
                    at = result.index(previous) + 1
                    break
        result.insert(at, name)
        seen.add(name)
    return tuple(result)


def known_sections(defaults: DigestDefaults, saved: dict[str, Any]) -> frozenset[str]:
    """Every section the digest knows: the file's (on or off) and any the owner has
    listed."""
    names = set(defaults.settings.sections) | set(defaults.sections_off)
    for key in ("sections", "sections_off"):
        names.update(saved.get(key) or ())
    names.update((saved.get("section_config") or {}).keys())
    return frozenset(names)


def apply_saved(defaults: DigestDefaults, saved: dict[str, Any]) -> DigestSettings:
    """``defaults`` with the owner's saved fields laid over them."""
    base = defaults.settings
    plain = {
        k: _as_field(k, v)
        for k, v in saved.items()
        if k in EDITABLE and k not in ("sections", "sections_off", *_PER_ENTRY)
    }
    if "section_config" in saved:
        plain["section_config"] = {
            **{k: dict(v) for k, v in base.section_config.items()},
            **_as_field("section_config", saved["section_config"]),
        }
    if "news_groups" in saved:
        groups = {k: dict(v) for k, v in base.news_groups.items()}
        for slot, spec in _as_field("news_groups", saved["news_groups"]).items():
            merged = {"title": slot, "topics": (), **groups.get(slot, {}), **spec}
            groups[slot] = merged
        plain["news_groups"] = groups
    sections = base.sections
    if "sections" in saved or "sections_off" in saved:
        sections = merge_sections(base.sections, saved.get("sections"), saved.get("sections_off"))
    return _with_locked(defaults, replace(base, **plain, sections=sections))


def sections_off(defaults: DigestDefaults, saved: dict[str, Any]) -> tuple[str, ...]:
    """The sections turned off, in file order then the owner's: the owner's, plus the
    file's off-by-default ones the owner has not turned on."""
    on = set(saved.get("sections") or ())
    off = list(saved.get("sections_off") or ())
    off += [s for s in defaults.sections_off if s not in on and s not in off]
    off = [s for s in off if s not in defaults.locked_sections]
    names = (*defaults.settings.sections, *defaults.sections_off)
    order = {s: i for i, s in enumerate(names)}
    return tuple(sorted(off, key=lambda s: order.get(s, len(order))))


def _renamed_list(names: Any, renamed: Mapping[str, tuple[str, ...]]) -> Any:
    """A section list under today's names (one old name may become several)."""
    if not isinstance(names, list):
        return names
    out: list[Any] = []
    for name in names:
        key = str(name).strip().lower()
        out.extend(renamed.get(key, (name,)))
    return list(dict.fromkeys(out))


def _renamed_config(raw: Any, renamed: Mapping[str, tuple[str, ...]]) -> tuple[Any, list[str]]:
    """``section_config`` under today's names, and the old sections whose knobs were
    left behind. A section renamed to one keeps its knobs (the new name's own win a
    clash); one split into several leaves them behind: a cap on the whole old section
    is not a cap on each part, so each part takes the file's."""
    if not isinstance(raw, dict):
        return raw, []
    items = [(str(k).strip().lower(), v) for k, v in raw.items()]
    out: dict[str, Any] = {name: knobs for name, knobs in items if name not in renamed}
    left: list[str] = []
    for name, knobs in items:
        targets = renamed.get(name)
        if targets is None:
            continue
        if len(targets) > 1:
            left.append(name)
        elif isinstance(knobs, dict):
            own = out.get(targets[0])
            out[targets[0]] = {**knobs, **(own if isinstance(own, dict) else {})}
    return out, left


def renamed_fields(saved: dict[str, Any], renamed: Mapping[str, tuple[str, ...]]) -> dict[str, Any]:
    """``saved`` (stored JSON) with old section names mapped to today's."""
    if not renamed:
        return saved
    out = dict(saved)
    for key in ("sections", "sections_off"):
        if key in out:
            out[key] = _renamed_list(out[key], renamed)
    if "sections" in out and isinstance(out.get("sections_off"), list):
        # One old name split into several: on stays on, whatever the off list says.
        on = set(out["sections"]) if isinstance(out["sections"], list) else set()
        out["sections_off"] = [s for s in out["sections_off"] if s not in on]
    if "section_config" in out:
        out["section_config"], left = _renamed_config(out["section_config"], renamed)
        if left:
            logger.info("digest: section_config of %s left behind (section renamed)", left)
    return out


def read_saved(store: SettingsStore, defaults: DigestDefaults | None = None) -> dict[str, Any]:
    """The owner's saved fields that still validate, under today's section names
    (``defaults.renamed``); a bad one is ignored with a warning (the file's value is a
    safe place to land)."""
    try:
        saved = store.get(SETTINGS_SECTION, KEY)
    except Exception:  # a broken store must not stop the digest
        logger.exception("digest: could not read saved settings; using the file")
        return {}
    if not isinstance(saved, dict):
        return {}
    if defaults is not None:
        saved = renamed_fields(saved, defaults.renamed)
    return _good_fields(saved, source="settings.db", known_sections=None)


def store_for(data_dir: Path) -> SettingsStore:
    from iris_harness.foundation.settings.store import (
        SETTINGS_DB_NAME,
        SettingsStore,
    )

    return SettingsStore(db_path=data_dir / SETTINGS_DB_NAME)


def load_digest_settings(data_dir: Path, config_dir: Path | None = None) -> DigestSettings:
    """The digest settings in effect: ``digest.yaml`` <- the owner's saved changes.

    The first call against a data dir moves ``brief_prefs.json`` into the store. Never
    raises: anything broken gives the defaults (or built-ins) and a warning.
    """
    try:
        defaults = load_defaults(config_dir)
    except Exception:  # last resort; load_defaults already guards
        logger.exception("digest: defaults unreadable; using built-ins")
        return DigestSettings()
    try:
        store = store_for(data_dir)
        migrate_brief_prefs(data_dir, store, defaults)
        return apply_saved(defaults, read_saved(store, defaults))
    except Exception:  # a digest with defaults beats no digest
        logger.exception("digest: saved settings unusable; using %s", DIGEST_FILE)
        return defaults.settings


# -- brief_prefs.json -> the store (once) -------------------------------------------


#: ``section_config`` knobs whose values come from a plugin's vocabulary: knob -> the
#: values it may hold. The core names no knob and no value; the plugin whose section
#: reads a knob registers it. A knob nobody registered is kept as saved.
_knob_validators: dict[str, Callable[[], Iterable[str]]] = {}


def register_section_knob_validator(knob: str, allowed: Callable[[], Iterable[str]] | None) -> None:
    """Register the values ``section_config.<section>.<knob>`` may hold (``None`` clears).

    Read when ``brief_prefs.json`` moves into the store: a listed value outside
    ``allowed()`` is dropped, and the knob with it when none is left.
    """
    name = knob.strip().lower()
    if allowed is None:
        _knob_validators.pop(name, None)
    else:
        _knob_validators[name] = allowed


def _knob_values() -> dict[str, frozenset[str]]:
    return {
        knob: frozenset(str(v).strip().lower() for v in allowed())
        for knob, allowed in _knob_validators.items()
    }


def _clean_section_config(
    raw: Any, renamed: Mapping[str, tuple[str, ...]]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """``section_config`` from the old file, under today's section names, minus what no
    section can use.

    A knob a plugin registered a validator for keeps only the values it allows
    (:func:`register_section_knob_validator`): a list knob that narrows a section to the
    plugin's categories filters everything out on a value outside its vocabulary (a
    mis-saved ``["images"]``), so that value is dropped. Knobs of two old sections renamed to one are merged (the
    first listed wins a clash); knobs of a section split into several are dropped (each
    part takes the file's).
    """
    dropped: list[str] = []
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out, dropped
    validators = _knob_values()
    for section, knobs in raw.items():
        name = str(section).strip().lower()
        if not _NAME_RE.match(name) or not isinstance(knobs, dict):
            dropped.append(f"{section}")
            continue
        targets = renamed.get(name, (name,))
        if len(targets) > 1:
            dropped.append(f"{name}.section_config")
            continue
        name = targets[0]
        kept = {**dict(knobs), **out.get(name, {})}
        for knob, valid in validators.items():
            if knob not in kept:
                continue
            values = kept.get(knob)
            items = [str(c).strip().lower() for c in values] if isinstance(values, list) else []
            good = [c for c in items if c in valid]
            dropped.extend(f"{name}.{knob}:{c}" for c in items if c not in valid)
            if good:
                kept[knob] = good
            else:
                kept.pop(knob)
        if kept:
            out[name] = kept
    return out, dropped


def _legacy_sections(
    raw: dict[str, Any], defaults: DigestDefaults
) -> tuple[list[str] | None, list[str]]:
    """(on, off) from the old file's ``enabled_sections`` + ``section_order``.

    ``section_order`` is the order the owner asked for; the order of
    ``enabled_sections`` was the manifest's (the tool wrote it), so the rest follow the
    file's order here. Old names are mapped to today's (``renamed``: ai_news is
    news_ai, ``news`` the three news slots), and ``dropped_sections`` are left off even when
    the old file had them on (the old tool wrote every section on). ``None`` for ``on``
    means the old file had no selection.
    """
    enabled = raw.get("enabled_sections")
    if not isinstance(enabled, list):
        return None, []
    renamed = defaults.renamed

    def today(names: Any) -> list[str]:
        out: list[str] = []
        for name in names:
            text = str(name).strip().lower()
            if _NAME_RE.match(text):
                out.extend(renamed.get(text, (text,)))
        return out

    names = _dedupe(today(enabled))
    names = [s for s in names if s not in defaults.dropped_sections]
    order = today(raw.get("section_order") or [])
    first = _dedupe([s for s in order if s in names])
    rank = {s: i for i, s in enumerate(defaults.settings.sections)}
    rest = sorted(
        (s for s in names if s not in first),
        key=lambda s: (rank.get(s, len(rank)), names.index(s)),
    )
    on = first + rest
    new = set(defaults.new_sections) | set(defaults.locked_sections)
    off = [s for s in defaults.settings.sections if s not in on and s not in new]
    off += [s for s in defaults.dropped_sections if s not in off]
    return on, off


def _legacy_names(enabled: Any, renamed: Mapping[str, tuple[str, ...]]) -> set[str]:
    """The old file's enabled section names, under today's names."""
    if not isinstance(enabled, list):
        return set()
    names = (str(s).strip().lower() for s in enabled)
    return {t for n in names for t in renamed.get(n, (n,))}


def migrate_brief_prefs(data_dir: Path, store: SettingsStore, defaults: DigestDefaults) -> bool:
    """Move ``<data_dir>/brief_prefs.json`` into the store, once. True when it ran now.

    Idempotent: a marker (``digest`` / ``brief_prefs_migration``) records that it ran,
    unreadable file included. The old file stays where it is, unread after this.
    Fields the owner already saved win over the old file's.
    """
    path = data_dir / BRIEF_PREFS_FILE
    # No old file, nothing to move: no marker either, so a read creates no settings.db.
    if not path.exists() or store.get(SETTINGS_SECTION, MIGRATION_KEY) is not None:
        return False
    if not defaults.from_file:
        logger.warning(
            "digest: %s not moved yet — %s was not loaded, so its section renames are unknown",
            path,
            DIGEST_FILE,
        )
        return False
    moved: dict[str, Any] = {}
    dropped: list[str] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("not a JSON object")
    except Exception as exc:  # noqa: BLE001 — a bad old file is skipped, not fatal
        logger.warning("digest: %s unreadable (%s); nothing migrated", path, exc)
        status = "unreadable"
    else:
        status = "migrated"
        on, off = _legacy_sections(raw, defaults)
        if on is not None:
            moved["sections"] = on
            if off:
                moved["sections_off"] = off
        config, dropped = _clean_section_config(raw.get("section_config"), defaults.renamed)
        dropped.extend(
            s
            for s in defaults.dropped_sections
            if s in _legacy_names(raw.get("enabled_sections"), defaults.renamed)
        )
        if config:
            moved["section_config"] = config
    if moved:
        saved = read_saved(store, defaults)
        before = apply_saved(defaults, saved)
        target = {**moved, **saved}
        diff = diff_from_defaults(defaults, target)
        if diff != saved:
            store.set(
                SETTINGS_SECTION,
                KEY,
                diff,
                old=view(before, sections_off(defaults, saved)),
                new=view(apply_saved(defaults, diff), sections_off(defaults, diff)),
                actor=MIGRATION_ACTOR,
            )
    store.set(
        SETTINGS_SECTION,
        MIGRATION_KEY,
        {
            "at": datetime.now(UTC).isoformat(),
            "source": BRIEF_PREFS_FILE,
            "status": status,
            "moved": sorted(moved),
            "dropped": dropped,
        },
        old=None,
        actor=MIGRATION_ACTOR,
    )
    if status == "migrated":
        logger.info(
            "digest: moved %s into settings (%s); dropped %s",
            path,
            ", ".join(sorted(moved)) or "nothing different",
            ", ".join(dropped) or "nothing",
        )
    return True


# -- views and diffs ------------------------------------------------------------------


def file_view(defaults: DigestDefaults) -> dict[str, Any]:
    """``digest.yaml``'s values as :func:`view` shows them (its off sections included)."""
    return view(defaults.settings, defaults.sections_off)


def view(settings: DigestSettings, off: tuple[str, ...] = ()) -> dict[str, Any]:
    """Every editable field as plain JSON (what the app shows and the history keeps)."""
    return {
        "enabled": settings.enabled,
        "time": settings.time,
        "channel": settings.channel,
        "sections": list(settings.sections),
        "sections_off": list(off),
        "section_config": {k: dict(v) for k, v in settings.section_config.items()},
        "news_topics": list(settings.news_topics),
        "news_groups": {
            k: {"title": v.get("title", k), "topics": list(v.get("topics") or ())}
            for k, v in settings.news_groups.items()
        },
        "news_local_area": settings.news_local_area,
        "news_sources": list(settings.news_sources),
        "news_language": settings.news_language,
        "focus_categories": list(settings.focus_categories),
        "focus_limit": settings.focus_limit,
        "focus_per_account": settings.focus_per_account,
    }


def diff_from_defaults(defaults: DigestDefaults, saved: dict[str, Any]) -> dict[str, Any]:
    """``saved`` minus the fields that equal the file's, so a later change to the file
    reaches every field the owner has not changed (sections that come out as the
    file's, in its order and with its off ones off, are no change)."""
    from_file = file_view(defaults)
    diff = {k: v for k, v in saved.items() if k in EDITABLE and v != from_file.get(k)}
    if "section_config" in diff:
        # Per section: only those that differ from the file's; an empty mapping for a
        # section the file has no knobs for is no change.
        file_config = from_file["section_config"]
        config = {k: v for k, v in diff["section_config"].items() if v != file_config.get(k, {})}
        if config:
            diff["section_config"] = config
        else:
            diff.pop("section_config")
    if "news_groups" in diff:
        file_groups = from_file["news_groups"]
        groups: dict[str, Any] = {}
        for slot, spec in diff["news_groups"].items():
            base = file_groups.get(slot, {})
            changed = {k: v for k, v in spec.items() if v != base.get(k)}
            if changed:
                groups[slot] = changed
        if groups:
            diff["news_groups"] = groups
        else:
            diff.pop("news_groups")
    if ("sections" in diff or "sections_off" in diff) and (
        apply_saved(defaults, diff).sections == defaults.settings.sections
        and sections_off(defaults, diff) == defaults.sections_off
    ):
        # The file's order (short only of sections new to the owner), the file's off.
        diff.pop("sections", None)
        diff.pop("sections_off", None)
    return diff


# -- the layout (digest v5) ------------------------------------------------------------


def group_sections(
    settings: DigestSettings, rendered: Sequence[str]
) -> list[tuple[DigestGroup, list[str]]]:
    """The rendered sections under their groups: ``[(group, [sections])]``.

    Groups come in ``digest.yaml`` order; within a group, its sections come in
    ``rendered`` order (the owner's order). A group with nothing rendered is left out.
    Rendered sections in no group go last, under :data:`MORE_GROUP`. The footer
    (``footer_sections``) is in no group: the renderer puts it after every group.
    """
    footer = set(settings.footer_sections)
    names = [n for n in dict.fromkeys(rendered) if n not in footer]
    owner: dict[str, str] = {}
    for group in settings.groups:
        for section in group.sections:
            owner.setdefault(section, group.id)
    out: list[tuple[DigestGroup, list[str]]] = []
    for group in settings.groups:
        members = [n for n in names if owner.get(n) == group.id]
        if members:
            out.append((group, members))
    rest = [n for n in names if n not in owner]
    if rest:
        out.append((MORE_GROUP, rest))
    return out


def _fill_area(text: str, area: str) -> str:
    return text.replace("{" + AREA_PLACEHOLDER + "}", area)


def news_group_topics(settings: DigestSettings, slot: str) -> tuple[str, ...]:
    """The topics of news slot ``slot`` with ``{news_local_area}`` filled in; () when
    the digest has no such slot."""
    spec = settings.news_groups.get(slot) or {}
    topics = (_fill_area(str(t), settings.news_local_area) for t in spec.get("topics") or ())
    return tuple(t for t in (" ".join(t.split()) for t in topics) if t)


def news_group_title(settings: DigestSettings, slot: str) -> str:
    """News slot ``slot``'s heading ("Local — St. Louis"); the slot name when unknown."""
    spec = settings.news_groups.get(slot) or {}
    return _fill_area(str(spec.get("title") or slot), settings.news_local_area)


__all__ = [
    "EDITABLE",
    "KEY",
    "MIGRATION_KEY",
    "SETTINGS_SECTION",
    "DigestDefaults",
    "MORE_GROUP",
    "DigestEditError",
    "DigestGroup",
    "DigestSettings",
    "apply_saved",
    "diff_from_defaults",
    "file_view",
    "group_sections",
    "iris_timezone",
    "known_sections",
    "load_defaults",
    "load_digest_settings",
    "store_for",
    "merge_sections",
    "migrate_brief_prefs",
    "news_group_title",
    "news_group_topics",
    "read_saved",
    "register_section_knob_validator",
    "renamed_fields",
    "sections_off",
    "validate",
    "view",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_knob_validators")
