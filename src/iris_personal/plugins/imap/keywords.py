"""IRIS labels on IMAP mail: ``$``-keywords, not folders.

The email judge puts exactly one ``IRIS/<bucket>`` label on each email it judged
(``judge.yaml``). Gmail has labels; IMAP has two things that could stand in for one:

* **a folder** -- but a message lives in one folder. Moving it to ``IRIS/Bill`` takes it
  out of the inbox (IRIS never archives), and copying it there doubles the mail.
* **a keyword** (RFC 3501 flag without a backslash) -- set per message, as many as you
  like, removed as easily, and it moves nothing. Thunderbird shows them as tags,
  Dovecot / Fastmail / Cyrus keep them. The ``$`` prefix is the convention for
  client-defined keywords (RFC 5788).

So an IRIS label is a keyword: ``IRIS/Needs-Reply`` becomes ``$IRIS_Needs-Reply``. The
keyword is the provider's "label id" (what ``ensure_labels`` returns and what
``emails.labels`` stores), the same role a Gmail ``Label_123`` plays. A server whose
mailbox does not keep custom keywords (``PERMANENTFLAGS`` without ``\\*``) cannot be
labelled; the provider says so rather than fall back to moving mail.

The mapping is a pure function of the name, so it needs no server round trip and reads
back the same on every machine.
"""

from __future__ import annotations

import re

#: Characters an IMAP ``atom`` may not contain (RFC 3501 ``atom-specials``), plus ``/``
#: which is legal but reads as a folder separator in most clients.
_NOT_ATOM = re.compile(r"[\s(){%*\"\\\]/\x00-\x1f\x7f]")

#: Stored labels that are not keywords: where the message lives and its read state,
#: synthesised from the folder and the system flags on fetch (Gmail's names, so readers
#: that look for ``UNREAD`` or ``INBOX`` work on IMAP mail unchanged).
INBOX = "INBOX"
UNREAD = "UNREAD"
STARRED = "STARRED"
SYNTHETIC_LABELS = frozenset({INBOX, UNREAD, STARRED})


def keyword_for(name: str) -> str:
    """The IMAP keyword for an IRIS label name: ``IRIS/Bill`` -> ``$IRIS_Bill``."""
    cleaned = _NOT_ATOM.sub("_", name.strip())
    if not cleaned:
        raise ValueError("a label name cannot be empty")
    return cleaned if cleaned.startswith("$") else f"${cleaned}"


def is_writable_keyword(flag: str) -> bool:
    """True for a keyword IRIS may add or remove: never a system flag (``\\Seen``,
    ``\\Deleted``, ...), never one of the synthetic location/read-state labels."""
    return bool(flag) and not flag.startswith("\\") and flag not in SYNTHETIC_LABELS


def labels_from_flags(folder: str, flags: tuple[str, ...]) -> tuple[str, ...]:
    """What a fetched message's ``labels`` are: its folder (``INBOX``), ``UNREAD``
    unless ``\\Seen``, ``STARRED`` when ``\\Flagged``, then its keywords as-is."""
    out: list[str] = [INBOX if folder.upper() == "INBOX" else folder]
    lowered = {f.lower() for f in flags}
    if "\\seen" not in lowered:
        out.append(UNREAD)
    if "\\flagged" in lowered:
        out.append(STARRED)
    out.extend(f for f in flags if not f.startswith("\\"))
    return tuple(dict.fromkeys(out))


__all__ = [
    "INBOX",
    "STARRED",
    "SYNTHETIC_LABELS",
    "UNREAD",
    "is_writable_keyword",
    "keyword_for",
    "labels_from_flags",
]
