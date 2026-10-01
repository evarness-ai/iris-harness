"""Process-wide state, declared by the module that owns it, so it can be put back.

Building a runtime, starting it and mounting plugins fill process-wide registries and
caches: the mail-provider registry, API route factories, health check providers, the
process event bus's subscribers, the identity-redaction seams, config caches read from
the run's home... A second runtime in the same process -- the testing harness runs one
per ``with`` block -- would otherwise inherit whatever the first left behind: a mail
provider from a plugin it never mounted, a route bound to a deleted home.

The owner of each piece of state declares it here once, at import, with how to save and
how to restore it: :func:`register_process_state` for anything, :func:`track_globals`
for plain module globals (containers are restored in place, so a reference another
module holds stays valid). The harness then takes a :func:`snapshot_process_state`
before it builds and hands it to :func:`restore_process_state` on exit. The harness
never pokes a private name; a module that adds state adds its declaration beside it.

State first registered *during* a run (its module was imported by the run) is restored
to its value when it registered -- the import-time value -- since nothing held it before.
The limit that follows: a module first imported during a run that, at import, registers
into *another* module's tracked state loses that registration on restore (the import
does not run again). The harness imports the composition root before its snapshot, so
the import-time registrations the root pulls in are part of the "before".
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class _Entry:
    save: Callable[[], Any]
    restore: Callable[[Any], None]
    initial: Any


_lock = Lock()
_entries: dict[str, _Entry] = {}


@dataclass(frozen=True)
class ProcessStateSnapshot:
    """Every registered piece of state's saved value, by name."""

    values: dict[str, Any]


def register_process_state(
    name: str, save: Callable[[], Any], restore: Callable[[Any], None]
) -> None:
    """Declare one piece of process-wide state.

    ``save()`` returns a value that does not change when the state does (a copy of a
    container, not the container); ``restore(value)`` puts that value back. ``save`` is
    called once now, and that value is what the state is restored to after a run that
    started before it was registered. Keyed by ``name``: registering it again replaces
    the declaration (a module reloaded in a test).
    """
    with _lock:
        _entries[name] = _Entry(save=save, restore=restore, initial=save())


_NO_CONTENT = object()


def _copy(value: Any) -> Any:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, set):
        return set(value)
    return _NO_CONTENT


def _save_global(module: str, name: str) -> tuple[Any, Any]:
    value = getattr(sys.modules[module], name)
    return value, _copy(value)


def _restore_global(module: str, name: str, saved: tuple[Any, Any]) -> None:
    value, content = saved
    setattr(sys.modules[module], name, value)
    if content is _NO_CONTENT:
        return
    value.clear()
    if isinstance(value, list):
        value.extend(content)
    else:
        value.update(content)


def track_globals(module: str, *names: str) -> None:
    """Declare module globals of ``module`` (pass ``__name__``) as process state.

    Each is restored to the object it was bound to; a ``dict``, ``list`` or ``set`` also
    gets its contents back, in place. Call it at the bottom of the module, after the
    globals (and any import-time registrations into them) exist.
    """
    for name in names:
        if not hasattr(sys.modules[module], name):
            raise AttributeError(f"{module} has no global {name!r} to track")
        register_process_state(
            f"{module}.{name}",
            save=partial(_save_global, module, name),
            restore=partial(_restore_global, module, name),
        )


def registered_process_state() -> tuple[str, ...]:
    """The names of every declared piece of state."""
    with _lock:
        return tuple(sorted(_entries))


def snapshot_process_state() -> ProcessStateSnapshot:
    """Save every declared piece of state."""
    with _lock:
        entries = dict(_entries)
    return ProcessStateSnapshot(values={name: e.save() for name, e in entries.items()})


def restore_process_state(snapshot: ProcessStateSnapshot) -> None:
    """Put every declared piece of state back to ``snapshot`` -- or, for state
    registered after it was taken, to its value at registration."""
    with _lock:
        entries = dict(_entries)
    for name, entry in entries.items():
        entry.restore(snapshot.values.get(name, entry.initial))


__all__ = [
    "ProcessStateSnapshot",
    "register_process_state",
    "registered_process_state",
    "restore_process_state",
    "snapshot_process_state",
    "track_globals",
]
