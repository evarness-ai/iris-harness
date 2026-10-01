"""Display-only masking of personal identifiers on their way to a screen.

The governance kernel already stops secrets leaving for a model (``redaction_filter``
at ``pre_llm_call``). Nothing looked at text coming *back*: a turn that summarised an
inbox printed the account addresses verbatim to the terminal, to Telegram, to the web
UI and into every session log that recorded the reply.

This module masks that text at the display boundary and **only** there. The audit
ledger, the session log and the stored ``ChatResult`` the ``record`` stage writes keep
the original bytes, because an audit that cannot be reconciled with what actually
happened is not an audit. The masking is applied to the events the turn pipeline
yields, which run after ``record`` — see ``runtime/turn/pipeline.py``.

Masking is on by default and turns off with ``IRIS_GOVERNANCE_DISPLAY_MASK=0``.

Streaming is the hard part: an address arrives split across LLM tokens
(``jord`` + ``an1.kp@ex`` + ``ample.com``), so masking each chunk as it lands would
miss it. ``StreamMasker`` holds back the trailing partial word until a whitespace
character proves it complete. An address contains no whitespace, so a whole one is
always inside one held word.

Not the owner-identity matchers (ADR-0125): this masks EVERY address, the owner's and
anyone else's, keeping enough of the local part to tell two accounts apart; the matchers
find only the owner's own literals. The two share nothing but the intent, so this keeps its
own pattern and its behaviour exactly.
"""

from __future__ import annotations

import re

from iris_harness.foundation.env import env_flag

# Deliberately narrower than a full RFC 5322 address. A display mask that swallows
# neighbouring punctuation is worse than one that misses an exotic address: the
# audit ledger still holds the original either way.
_EMAIL = re.compile(
    r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b",
)

# An address is at most 320 characters (RFC 3696). A "word" longer than this cannot
# be one, so the stream masker stops holding it rather than buffering without bound.
_MAX_HELD_WORD = 320


def is_enabled() -> bool:
    """Whether display masking is active. On unless explicitly turned off."""
    return env_flag("IRIS_GOVERNANCE_DISPLAY_MASK", default=True)


def _mask_local_part(local: str) -> str:
    """Keep enough of the local part to tell two of the user's own accounts apart.

    ``jordan1.kp`` and ``jordankpatel`` share their first three characters, so a
    leading-prefix-only mask renders both identically and the reader cannot tell which
    account a line is about. Keeping the tail as well restores that distinction
    without restoring a usable address.
    """
    if len(local) <= 2:
        return "*" * len(local)
    if len(local) <= 4:
        return f"{local[0]}***"
    return f"{local[:2]}***{local[-2:]}"


def mask_text(text: str) -> str:
    """Mask every email address in ``text``. A no-op when masking is turned off."""
    if not text or not is_enabled():
        return text
    return _EMAIL.sub(lambda m: f"{_mask_local_part(m.group(1))}@{m.group(2)}", text)


class StreamMasker:
    """Masks a token stream, holding back the trailing partial word.

    ``feed`` returns the text that is safe to display now; ``flush`` returns whatever
    is still held once the stream ends. Callers must call ``flush`` — the final word
    of a response is otherwise never emitted.
    """

    def __init__(self) -> None:
        self._held = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if not is_enabled():
            return chunk
        buffer = self._held + chunk
        # Everything up to the last whitespace is complete; the tail may still grow.
        split = max(buffer.rfind(" "), buffer.rfind("\n"), buffer.rfind("\t"))
        if split == -1:
            if len(buffer) > _MAX_HELD_WORD:
                self._held = ""
                return mask_text(buffer)
            self._held = buffer
            return ""
        self._held = buffer[split + 1 :]
        return mask_text(buffer[: split + 1])

    def flush(self) -> str:
        held, self._held = self._held, ""
        return mask_text(held)


__all__ = ["StreamMasker", "is_enabled", "mask_text"]
