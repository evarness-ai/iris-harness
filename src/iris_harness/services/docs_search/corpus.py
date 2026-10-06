"""The searchable corpus: an allow-list of roots, contained paths, scanned documents.

Everything that decides *what may be read* lives here, so the search module only ever sees
documents that passed. The rules, in order:

1. The roots come from ``config/docs_search.yaml`` and are relative to the checkout, each
   under ``docs/``. An absolute path, ``~``, a ``..`` component or a backslash is a config
   error. The shipped file is the allow-list; an ``IRIS_CONFIG_DIR`` copy can only NARROW it
   (roots and extensions intersect, limits take the smaller value).
2. A root is used only if it is a real directory reached WITHOUT a symlink, and it neither
   contains nor sits inside ``IRIS_HOME``, the data directory or the governance
   directories (the identity files, the vault and every store live there).
3. A file is read only if it is a regular file, not a symlink, with an allowed
   extension, whose resolved path is inside the resolved root it was found under. The
   check and the read share one file descriptor (``O_NOFOLLOW``, ``fstat``, the path the
   descriptor really names), so a file swapped for a symlink after the listing is not read.
4. The text must not carry frontmatter classifying it ``secret`` or ``personal`` (identity
   files do; an unparseable frontmatter withholds the document), and must not classify
   ``secret`` under the kernel's content scan. The scan runs in bounded chunks under a time
   budget: a document that exhausts it is withheld (fail closed).

Nothing a caller passes (the query, a section filter) is ever used as a path.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import stat as stat_mod
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from iris_harness.foundation.paths import (
    checkout_docs_dir,
    config_path,
    data_dir,
    default_config_dir,
    governance_config_dir,
    governance_data_dir,
    iris_home,
)
from iris_harness.kernel.governance.file_scan import scan_text

logger = logging.getLogger(__name__)

CONFIG_FILE = "docs_search.yaml"

# Headings are split by hand (_parse_heading): the regex this replaced backtracked
# super-quadratically on a long line of blanks. A heading line is read to this length only.
_MAX_HEADING_LINE = 300
_MAX_FRONTMATTER_CHARS = 8000
_MAX_FRONTMATTER_NODES = 500
_FENCE = re.compile(r"^\s*(```|~~~)")
_WITHHELD_CLASSES = frozenset({"secret", "personal"})


class DocsSearchConfigError(ValueError):
    """``config/docs_search.yaml`` is missing, unreadable or names an unsafe root."""


@dataclass(frozen=True)
class Limits:
    max_files: int = 400
    max_file_bytes: int = 400_000
    max_total_bytes: int = 8_000_000
    max_query_chars: int = 200
    max_query_terms: int = 8
    default_results: int = 5
    max_results: int = 10
    snippet_chars: int = 240
    max_output_chars: int = 4000
    scan_chunk_chars: int = 4000  # the content scan sees at most this much at once
    scan_budget_ms: int = 2000  # per document; exhausting it withholds the document
    scan_total_ms: int = 3000  # per call; documents not yet scanned wait for the next call


@dataclass(frozen=True)
class Config:
    roots: tuple[str, ...]
    extensions: frozenset[str]
    limits: Limits


@dataclass(frozen=True)
class Section:
    heading: str  # "## Escalation"; "(top)" for text before the first heading
    start_line: int  # 1-based
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Doc:
    name: str  # repo-relative, e.g. docs/architecture/iris-harness.md
    sections: tuple[Section, ...]


@dataclass
class Corpus:
    docs: list[Doc] = field(default_factory=list)
    roots_missing: bool = True  # no allow-listed root exists on this install
    dropped: int = 0  # documents withheld (secret/personal, frontmatter, non-UTF8, scan budget)
    skipped: int = 0  # allowed-type files refused before reading (hardlink, oversize, unreadable)
    pending: int = 0  # documents not scanned yet this call (scan time ran out); next call
    truncated: bool = False  # a cap stopped the walk


def _read_config(path: Path) -> Config:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise DocsSearchConfigError(f"{path} is unreadable: {exc}") from exc
    if not isinstance(raw, dict):
        raise DocsSearchConfigError(f"{path}: expected a mapping")
    roots = raw.get("roots")
    if not isinstance(roots, list) or not roots:
        raise DocsSearchConfigError(f"{path}: `roots` must be a non-empty list")
    clean: list[str] = []
    for entry in roots:
        raw_entry = str(entry).strip()
        # Judged as written: stripping a leading "/" first would turn "/etc" into a
        # relative root.
        text = raw_entry.rstrip("/") if not raw_entry.startswith("/") else raw_entry
        parts = text.replace("\\", "/").split("/")
        if (
            not text
            or "\\" in text
            or Path(text).is_absolute()
            or text.startswith("~")
            or ".." in parts
            or "." in parts
            or parts[0] != "docs"
            or len(parts) < 2
        ):
            raise DocsSearchConfigError(
                f"{path}: root {entry!r} must be a plain relative path under docs/"
            )
        clean.append(text)
    extensions = raw.get("extensions") or [".md"]
    if not isinstance(extensions, list) or not all(str(e).startswith(".") for e in extensions):
        raise DocsSearchConfigError(f"{path}: `extensions` must be a list like ['.md']")
    limits_raw = raw.get("limits") or {}
    if not isinstance(limits_raw, dict):
        raise DocsSearchConfigError(f"{path}: `limits` must be a mapping")
    values: dict[str, int] = {}
    for key, default in vars(Limits()).items():
        value = limits_raw.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise DocsSearchConfigError(f"{path}: limits.{key} must be a positive integer")
        values[key] = value
    return Config(
        roots=tuple(dict.fromkeys(clean)),
        extensions=frozenset(str(e).lower() for e in extensions),
        limits=Limits(**values),
    )


def load_config() -> Config:
    """The allow-list: the SHIPPED file, narrowed (never widened) by an override copy.

    The shipped ``docs_search.yaml`` (the checkout's, else the packaged one) is the
    allow-list. A copy under ``IRIS_CONFIG_DIR`` may remove roots and extensions and lower
    limits; a root it names that the shipped file does not is ignored. Missing or malformed
    config fails closed (nothing searched).
    """
    shipped_path = default_config_dir() / CONFIG_FILE
    shipped = _read_config(shipped_path)
    override_path = config_path(CONFIG_FILE)
    if not override_path.is_file() or override_path.resolve() == shipped_path.resolve():
        return shipped
    override = _read_config(override_path)
    extra = [r for r in override.roots if r not in shipped.roots]
    if extra:
        logger.warning("docs_search: override roots %s are not in the shipped allow-list", extra)
    roots = tuple(r for r in shipped.roots if r in override.roots)
    extensions = shipped.extensions & override.extensions
    if not roots or not extensions:
        raise DocsSearchConfigError(
            f"{override_path} leaves no root or extension in common with the shipped allow-list"
        )
    mine, theirs = vars(shipped.limits), vars(override.limits)
    return Config(
        roots=roots,
        extensions=extensions,
        limits=Limits(**{k: min(mine[k], theirs[k]) for k in mine}),
    )


def _protected_dirs() -> list[str]:
    """Where the owner's identity, data, vault and stores live: never searchable."""
    out: list[str] = []
    for fn in (iris_home, data_dir, governance_data_dir, governance_config_dir):
        try:
            out.append(os.path.realpath(fn()))
        except OSError:  # an unresolvable protected dir cannot be overlapped
            continue
    return out


def _overlaps(a: str, b: str) -> bool:
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def _resolve_root(base: Path, rel: str, protected: list[str]) -> Path | None:
    """The real directory for an allow-listed root, or ``None`` when it must not be read."""
    real_base = os.path.realpath(base)
    candidate = os.path.join(real_base, *rel.split("/"))
    if not os.path.isdir(candidate):
        return None
    # A symlink anywhere in the root's own path would let it point out of the checkout.
    if os.path.realpath(candidate) != os.path.normpath(candidate):
        logger.warning("docs_search: root %s is reached through a symlink; skipped", rel)
        return None
    if any(_overlaps(candidate, p) for p in protected):
        logger.warning("docs_search: root %s overlaps a protected directory; skipped", rel)
        return None
    return Path(candidate)


def _fd_path(fd: int) -> str | None:
    """The path the open descriptor really names (``/proc`` on Linux, ``F_GETPATH`` on macOS)."""
    try:
        return os.path.realpath(os.readlink(f"/proc/self/fd/{fd}"))
    except OSError:
        pass
    try:
        import fcntl

        getpath = getattr(fcntl, "F_GETPATH", None)  # macOS; absent on Linux
        if getpath is None:
            return None
        raw = fcntl.fcntl(fd, getpath, b"\0" * 1024)
        return os.path.realpath(raw.split(b"\0", 1)[0].decode("utf-8", "surrogateescape"))
    except (OSError, ImportError, AttributeError):
        return None


def read_contained(
    path: Path, root: Path, extensions: frozenset[str], max_bytes: int
) -> tuple[bytes, os.stat_result, str] | None:
    """``(bytes, stat, real path)`` of ``path`` if a plain allowed file inside ``root``.

    One open descriptor carries the whole decision: ``O_NOFOLLOW`` refuses a symlink at
    the last component, ``fstat`` must say regular file within the size cap, and the path
    the descriptor names (not the one that was listed) must sit inside ``root``. The bytes
    are read from that descriptor, so nothing swapped in after the listing is read. The
    path returned is the one the descriptor names, so a name or cache key is never the
    listed path of a file that was swapped. A file with more than one hard link is refused:
    its other names may sit anywhere.
    """
    if path.suffix.lower() not in extensions:
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat_mod.S_ISREG(st.st_mode) or st.st_nlink > 1 or st.st_size > max_bytes:
            return None
        named = _fd_path(fd)
        if named is None:
            # No way to ask the descriptor: fall back to comparing it with what the
            # resolved path says now (a swap between the two shows as a different file).
            named = os.path.realpath(path)
            if not os.path.samestat(st, os.stat(named)):
                return None
        if not Path(named).is_relative_to(root):
            return None
        data = b""
        while len(data) <= max_bytes:
            block = os.read(fd, 65536)
            if not block:
                break
            data += block
        return (data, st, named) if len(data) <= max_bytes else None
    except OSError:
        return None
    finally:
        os.close(fd)


def _frontmatter_withheld(text: str) -> bool:
    """:func:`_check_frontmatter`, never raising: a structure too deep for the parser or the
    walk (``RecursionError``, ``MemoryError``) withholds the document like any other doubt."""
    try:
        return _check_frontmatter(text)
    except (RecursionError, MemoryError):
        return True


def _check_frontmatter(text: str) -> bool:
    """Whether the document opens with frontmatter that must keep it out (fail closed).

    Same shape as the identity loader's ``_split_frontmatter`` (``yaml.safe_load`` between
    ``---`` lines; the loader reads the ``classification`` key), but stricter: a BOM or
    blank lines before the ``---`` do not hide it, ``...`` also closes it, keys compare
    case-insensitively at any depth, a ``classification`` whose value is not a plain
    scalar withholds, and frontmatter that cannot be parsed, is not a mapping, never closes,
    is oversized, or uses anchors/aliases (the way to build an exponentially shared
    structure) withholds the document.
    """
    body = text.lstrip("\ufeff").lstrip()
    lines = body.splitlines()
    if not lines or lines[0].strip() != "---":
        return False
    close = next((i for i, ln in enumerate(lines[1:], 1) if ln.strip() in ("---", "...")), None)
    if close is None:
        return True
    meta_text = "\n".join(lines[1:close])
    if len(meta_text) > _MAX_FRONTMATTER_CHARS:
        return True
    try:
        # The event stream first: an anchor or alias is refused before any tree exists,
        # so nothing is ever walked that could be exponentially shared.
        events = 0
        for event in yaml.parse(meta_text, Loader=yaml.SafeLoader):
            events += 1
            if events > _MAX_FRONTMATTER_NODES or getattr(event, "anchor", None) is not None:
                return True
            if isinstance(event, yaml.AliasEvent):
                return True
        meta = yaml.safe_load(meta_text)
    except yaml.YAMLError:
        return True
    if meta is None:
        return False
    return not isinstance(meta, dict) or _classified(meta, [_MAX_FRONTMATTER_NODES], set())


def _classified(node: object, budget: list[int], seen: set[int]) -> bool:
    """Whether ``node`` carries a withheld ``classification`` (any depth), failing closed.

    A visited set and a node budget bound the walk whatever the tree's shape. A
    ``classification`` that is not a plain scalar (a list, mapping, set) is withheld: it
    cannot be read the way the loader reads it.
    """
    budget[0] -= 1
    if budget[0] < 0:
        return True
    if isinstance(node, dict | list | set | tuple):
        if id(node) in seen:
            return False
        seen.add(id(node))
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).strip().lower() == "classification":
                if isinstance(value, dict | list | set | tuple | frozenset):
                    return True
                if str(value).strip().lower() in _WITHHELD_CLASSES:
                    return True
            if _classified(value, budget, seen):
                return True
    elif isinstance(node, list | tuple | set):
        return any(_classified(v, budget, seen) for v in node)
    return False


def _parse_heading(line: str) -> tuple[int, str] | None:
    """``(level, text)`` for an ATX heading line, else None. Linear; reads 300 chars."""
    line = line[:_MAX_HEADING_LINE]
    body = line.lstrip("#")
    level = len(line) - len(body)
    if not 1 <= level <= 6 or not body or body[0] not in " \t":
        return None
    text = body.strip(" \t").rstrip("#").rstrip(" \t")
    return (level, text) if text else None


def split_sections(text: str) -> tuple[Section, ...]:
    """Cut markdown at headings (not inside code fences) into line-numbered sections."""
    sections: list[Section] = []
    heading = "(top)"
    start = 1
    buf: list[str] = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line[:_MAX_HEADING_LINE]):
            in_fence = not in_fence
        parsed = None if in_fence else _parse_heading(line)
        if parsed:
            if buf:
                sections.append(Section(heading, start, tuple(buf)))
            heading = f"{'#' * parsed[0]} {parsed[1]}"
            start = number
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append(Section(heading, start, tuple(buf)))
    return tuple(s for s in sections if any(line.strip() for line in s.lines))


_TOKEN_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
# marker, whether it must start a token, and how long a token run after it looks like a secret
_MARKERS: tuple[tuple[str, bool, int], ...] = (
    ("eyJ", False, 30),
    ("sk-", True, 20),
    ("ghp_", True, 20),
    ("AKIA", True, 16),
)


def _long_line_marker(line: str) -> bool:
    """A cheap, regex-free look at a line too long to scan whole: does it carry something
    shaped like a JWT, API key or private-key header? Substring tests only."""
    if "-----BEGIN" in line:
        return True
    for marker, boundary, minimum in _MARKERS:
        at = line.find(marker)
        while at != -1:
            if not (boundary and at > 0 and line[at - 1] in _TOKEN_CHARS):
                end = at + len(marker)
                while end < len(line) and line[end] in _TOKEN_CHARS:
                    end += 1
                if end - at >= minimum:
                    return True
            at = line.find(marker, at + 1)
    return False


def _cut(line: str, size: int) -> int:
    """Where to cut ``line`` (longer than ``size``): just after a non-token character, so a
    run of ``[A-Za-z0-9._-]`` (a JWT, a key) is not split; ``size`` when it has none."""
    for i in range(size, size // 2, -1):
        if line[i - 1] not in _TOKEN_CHARS:
            return i
    return size


def _chunks(text: str, size: int) -> Iterator[str]:
    """Pieces of at most ``size`` chars cut at line ends, so a secret that sits on one line
    is never split; a longer line is cut only outside token runs (with an overlap too)."""
    overlap = min(256, size // 4)
    buf: list[str] = []
    used = 0
    for line in text.splitlines():
        while len(line) > size:
            if buf:
                yield "\n".join(buf)
                buf, used = [], 0
            cut = _cut(line, size)
            yield line[:cut]
            line = line[max(1, cut - overlap) :]
        if used + len(line) + 1 > size and buf:
            yield "\n".join(buf)
            buf, used = [], 0
        buf.append(line)
        used += len(line) + 1
    if buf:
        yield "\n".join(buf)


def _scan_verdict(text: str, limits: Limits, corpus_deadline: float, resume: list[int]) -> str:
    """``"secret"`` (withhold), ``"budget"`` (this document used its time: withhold),
    ``"defer"`` (the call's scan time ran out: decide on a later call) or ``"ok"``.

    The kernel's patterns are super-linear on hostile text, so they only ever see a chunk
    (cost bounded by the chunk) and each document gets a time budget; exhausting it is
    refused, not waved through. Lines longer than a chunk also get a regex-free marker test.
    ``resume`` is ``[chunks already scanned clean]``: a deferred document picks up where the
    last call stopped, so a document longer than one call's time still finishes.
    """
    stop = time.monotonic() + limits.scan_budget_ms / 1000
    for line in text.splitlines():
        if len(line) > limits.scan_chunk_chars and _long_line_marker(line):
            return "secret"
    for index, chunk in enumerate(_chunks(text, limits.scan_chunk_chars)):
        if index < resume[0]:
            continue
        if scan_text(chunk).is_secret:
            return "secret"
        resume[0] = index + 1
        now = time.monotonic()
        if now > corpus_deadline:
            return "defer"
        if now > stop:
            return "budget"
    return "ok"


# A file's identity for the caches: stat fields AND the sha256 of the bytes read, plus the
# chunk size the scan used. stat alone is defeatable (a same-size rewrite with mtime put
# back), and a verdict reached at one chunk size is not evidence at another.
_Key = tuple[int, int, int, int, str, int]

# (real path) -> (key, Doc | None); a changed file is re-read and re-scanned, and a document
# withheld once (secret, or scan budget exhausted) stays withheld without being scanned again.
_CACHE: dict[str, tuple[_Key, Doc | None]] = {}
# (real path) -> (key, chunks scanned clean so far) for a document the call's scan time
# interrupted. Resumed only on an identical key (same bytes, same chunk size, so a chunk
# index means the same text); dropped when the scan finishes.
_PROGRESS: dict[str, tuple[_Key, list[int]]] = {}


def build_corpus(config: Config, base: Path | None = None) -> Corpus:
    """Walk the allow-listed roots under ``base`` (the checkout's ``docs/`` parent).

    ``base`` defaults to the checkout; ``None`` there (an installed wheel) is an empty,
    ``roots_missing`` corpus. Tests pass a temp tree.
    """
    corpus = Corpus()
    if base is None:
        docs = checkout_docs_dir()
        if docs is None:
            return corpus
        base = docs.parent
    protected = _protected_dirs()
    limits = config.limits
    total = 0
    real_base = os.path.realpath(base)
    corpus_deadline = time.monotonic() + limits.scan_total_ms / 1000
    for rel in config.roots:
        root = _resolve_root(base, rel, protected)
        if root is None:
            continue
        corpus.roots_missing = False
        stack = [root]
        while stack:
            directory = stack.pop()
            try:
                entries = sorted(os.scandir(directory), key=lambda e: e.name)
            except OSError:
                continue
            subdirs: list[Path] = []
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                path = Path(entry.path)
                if entry.is_symlink():
                    continue  # never followed, file or directory
                if entry.is_dir(follow_symlinks=False):
                    subdirs.append(path)
                    continue
                if len(corpus.docs) >= limits.max_files or total >= limits.max_total_bytes:
                    corpus.truncated = True
                    return corpus
                if path.suffix.lower() not in config.extensions:
                    continue
                read = read_contained(path, root, config.extensions, limits.max_file_bytes)
                if read is None:
                    corpus.skipped += 1  # hardlink, oversize, swapped, unreadable: not silent
                    continue
                data, st, real = read
                if total + len(data) > limits.max_total_bytes:
                    corpus.truncated = True
                    return corpus
                total += len(data)
                key: _Key = (
                    st.st_dev,
                    st.st_ino,
                    st.st_mtime_ns,
                    st.st_size,
                    hashlib.sha256(data).hexdigest(),
                    limits.scan_chunk_chars,
                )
                cached = _CACHE.get(real)
                if cached is not None and cached[0] == key:
                    doc = cached[1]
                elif time.monotonic() > corpus_deadline:
                    corpus.pending += 1  # out of scan time: picked up on a later call
                    corpus.truncated = True
                    continue
                else:
                    doc = None
                    try:
                        text: str | None = data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = None
                    verdict = "secret"
                    if text is not None and not _frontmatter_withheld(text):
                        progress = _PROGRESS.get(real)
                        resume = progress[1] if progress and progress[0] == key else [0]
                        verdict = _scan_verdict(text, limits, corpus_deadline, resume)
                        if verdict == "defer":
                            if len(_PROGRESS) > 2048:
                                _PROGRESS.clear()
                            _PROGRESS[real] = (key, resume)
                        else:
                            _PROGRESS.pop(real, None)
                    if verdict == "defer":
                        corpus.pending += 1  # out of scan time: decided on a later call
                        corpus.truncated = True
                        continue
                    if verdict == "ok" and text is not None:
                        name = os.path.relpath(real, real_base).replace(os.sep, "/")
                        doc = Doc(name=name, sections=split_sections(text))
                    if len(_CACHE) > 2048:
                        _CACHE.clear()
                    _CACHE[real] = (key, doc)
                if doc is None:
                    corpus.dropped += 1
                else:
                    corpus.docs.append(doc)
            stack.extend(reversed(subdirs))
    return corpus
