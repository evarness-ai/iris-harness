"""Time-sortable unique ids (ULID), with the standard library only.

A ULID is 26 characters of Crockford base32: 48 bits of Unix milliseconds, then 80 bits of
randomness. Ids sort by creation time as plain strings. Within one process they are strictly
increasing: two ids minted in the same millisecond differ by incrementing the random part,
and a clock that steps backwards never makes an id sort before one already handed out.

``new_ulid`` is what the governed tool runner mints one call id with (issue #134); it is
thread safe, and nothing a caller passes in can choose the value.
"""

from __future__ import annotations

import os
import threading
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32: no I, L, O, U
_RANDOM_BITS = 80
_RANDOM_MAX = (1 << _RANDOM_BITS) - 1
_TIME_MAX = (1 << 48) - 1

ULID_LENGTH = 26

_lock = threading.Lock()
_last_ms = -1
_last_random = 0


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def new_ulid() -> str:
    """A new 26-character ULID, strictly greater than every one this process made before."""
    global _last_ms, _last_random
    with _lock:
        now_ms = min(int(time.time() * 1000), _TIME_MAX)
        if now_ms > _last_ms:
            _last_ms = now_ms
            _last_random = int.from_bytes(os.urandom(10), "big")
        else:
            # Same millisecond (or the clock stepped back): keep the later time and count up.
            _last_random += 1
            if _last_random > _RANDOM_MAX:
                _last_ms += 1
                _last_random = int.from_bytes(os.urandom(10), "big")
        return _encode(_last_ms, 10) + _encode(_last_random, 16)


def is_ulid(value: object) -> bool:
    """Whether ``value`` has the shape of a ULID (26 Crockford base32 characters)."""
    return (
        isinstance(value, str)
        and len(value) == ULID_LENGTH
        and all(ch in _ALPHABET for ch in value)
    )


__all__ = ["ULID_LENGTH", "is_ulid", "new_ulid"]
