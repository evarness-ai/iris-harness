"""``bool_setting``: an on/off setting read with its declared default (ADR-0120)."""

from __future__ import annotations

import pytest

from iris_harness.foundation.settings.catalog import (
    SettingDeclaration,
    build_catalog,
    load_core_catalog,
)
from iris_harness.runtime.settings_catalog import bool_setting


def _decl(kind: str, default: object) -> SettingDeclaration:
    return SettingDeclaration(
        kind=kind,  # type: ignore[arg-type]
        default=default,
        applies="next_run",
        label="A test switch",
        description="Switches a test job.",
        tab="features",
    )


_CATALOG = build_catalog(
    load_core_catalog(),
    [
        (
            "plugin:test",
            {
                "IRIS_TEST_ON_SWITCH": _decl("bool", True),
                "IRIS_TEST_OFF_SWITCH": _decl("bool", False),
                "IRIS_TEST_COUNT": _decl("int", 3),
            },
        )
    ],
)


@pytest.mark.parametrize(
    ("name", "env", "expected"),
    [
        ("IRIS_TEST_ON_SWITCH", None, True),  # unset: the declared default (on)
        ("IRIS_TEST_ON_SWITCH", "  ", True),  # blank counts as unset
        ("IRIS_TEST_ON_SWITCH", "0", False),
        ("IRIS_TEST_ON_SWITCH", "off", False),
        ("IRIS_TEST_OFF_SWITCH", None, False),
        ("IRIS_TEST_OFF_SWITCH", "yes", True),
        ("IRIS_TEST_OFF_SWITCH", "1", True),
        ("IRIS_TEST_UNDECLARED", None, False),  # nobody declares it: off
        ("IRIS_TEST_UNDECLARED", "on", True),  # but a set value is read as set
        ("IRIS_TEST_COUNT", None, False),  # not an on/off setting
    ],
)
def test_bool_setting(
    monkeypatch: pytest.MonkeyPatch, name: str, env: str | None, expected: bool
) -> None:
    if env is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, env)
    assert bool_setting(name, _CATALOG) is expected
