"""``foundation/process_state.py``: declared process-wide state, saved and put back."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from iris_harness.foundation import process_state
from iris_harness.foundation.process_state import (
    register_process_state,
    registered_process_state,
    restore_process_state,
    snapshot_process_state,
    track_globals,
)


@pytest.fixture
def module() -> Iterator[types.ModuleType]:
    mod = types.ModuleType("iris_process_state_probe")
    mod._registry = {"core": 1}  # type: ignore[attr-defined]
    mod._items = ["a"]  # type: ignore[attr-defined]
    mod._singleton = None  # type: ignore[attr-defined]
    sys.modules[mod.__name__] = mod
    yield mod
    del sys.modules[mod.__name__]
    for name in [n for n in registered_process_state() if n.startswith(mod.__name__)]:
        process_state._entries.pop(name)


def test_tracked_globals_come_back_in_place(module: types.ModuleType) -> None:
    track_globals(module.__name__, "_registry", "_items", "_singleton")
    registry, items = module._registry, module._items
    before = snapshot_process_state()

    module._registry["plugin"] = 2
    module._items.append("b")
    module._singleton = object()
    restore_process_state(before)

    assert module._registry == {"core": 1} and module._registry is registry
    assert module._items == ["a"] and module._items is items
    assert module._singleton is None


def test_state_registered_after_the_snapshot_goes_back_to_its_import_value(
    module: types.ModuleType,
) -> None:
    before = snapshot_process_state()
    track_globals(module.__name__, "_registry")  # the module was imported mid-run
    module._registry["plugin"] = 2

    restore_process_state(before)

    assert module._registry == {"core": 1}


def test_a_custom_pair(module: types.ModuleType) -> None:
    box = {"value": 1}
    register_process_state(
        f"{module.__name__}.box",
        save=lambda: box["value"],
        restore=lambda value: box.__setitem__("value", value),
    )
    before = snapshot_process_state()
    box["value"] = 5
    restore_process_state(before)
    assert box["value"] == 1


def test_tracking_a_missing_global_is_refused(module: types.ModuleType) -> None:
    with pytest.raises(AttributeError, match="_nope"):
        track_globals(module.__name__, "_nope")


def test_the_process_bus_gets_its_subscribers_back() -> None:
    from iris_harness.foundation.eventbus import get_default_bus

    bus = get_default_bus()
    before = snapshot_process_state()
    bus.on("process-state.probe", lambda payload: None)
    restore_process_state(before)
    assert get_default_bus() is bus
    assert not bus._handlers.get("process-state.probe")
