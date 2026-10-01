"""A plugin's own expiry kinds: declared in its manifest, tuned in digest.yaml, read by key
(core/SDK boundary plan PR 5, email slice step 4). The core names none of them; these
tests install a throwaway plugin under ``$IRIS_HOME/plugins``."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.sdk.digest import expiry_days
from iris_harness.services.digest import expiry
from iris_harness.services.digest.expiry import (
    ExpiryPolicy,
    declared_expiry_kinds,
    expiry_view,
    parse_expiry,
    reset_declared_expiry_kinds,
)
from iris_harness.services.digest.settings import load_defaults


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "home"
    monkeypatch.setenv("IRIS_HOME", str(home))
    reset_declared_expiry_kinds()
    yield home
    reset_declared_expiry_kinds()


def _install(home: Path, name: str, expiry_block: str) -> None:
    plugin = home / "plugins" / name
    plugin.mkdir(parents=True)
    (plugin / "manifest.yaml").write_text(
        f"name: {name}\nprovides: []\nexpiry:\n{expiry_block}", encoding="utf-8"
    )


def _digest(config: Path, expiry_block: str) -> Path:
    config.mkdir(parents=True, exist_ok=True)
    (config / "digest.yaml").write_text(f"expiry:\n{expiry_block}", encoding="utf-8")
    return config


def test_a_declared_kind_reads_its_default_until_the_owner_sets_it(
    home: Path, tmp_path: Path
) -> None:
    _install(home, "demo", "  parcel_days:\n    default: 4\n    max: 30\n")
    assert expiry_days("parcel_days", _digest(tmp_path / "a", "  task_overdue_days: 3\n")) == 4
    assert expiry_days("parcel_days", _digest(tmp_path / "b", "  parcel_days: 9\n")) == 9


def test_the_owners_value_is_held_to_the_declared_range(
    home: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _install(home, "demo", "  parcel_days:\n    default: 4\n    max: 30\n")
    with pytest.raises(ValueError, match="between 0 and 30"):
        parse_expiry({"parcel_days": 31})
    with caplog.at_level(logging.WARNING):
        policy = load_defaults(_digest(tmp_path / "c", "  parcel_days: 31\n")).settings.expiry
    assert policy == ExpiryPolicy()  # warned, built-in policy: the default again
    assert "parcel_days" in caplog.text
    assert policy.days("parcel_days") == 4


def test_the_view_lists_declared_kinds_after_the_cores(home: Path) -> None:
    _install(home, "demo", "  parcel_days:\n    default: 4\n")
    view = expiry_view(parse_expiry({"parcel_days": 6}))
    assert list(view)[-1] == "parcel_days" and view["parcel_days"] == 6
    assert set(expiry.EXPIRY_RANGES) <= set(view)


# -- loud, never silent -----------------------------------------------------------------


def test_reading_an_undeclared_kind_raises(home: Path) -> None:
    """Mutation: a plugin whose manifest stops declaring its kind fails, not defaults."""
    with pytest.raises(KeyError, match="no installed plugin declares 'parcel_days'"):
        expiry_days("parcel_days")


def test_an_undeclared_key_in_digest_yaml_is_refused(home: Path) -> None:
    with pytest.raises(ValueError, match="'parcel_days' is not a key"):
        parse_expiry({"parcel_days": 3})


@pytest.mark.parametrize(
    "block",
    [
        "  task_overdue_days:\n    default: 3\n",  # the core's own key
        "  Parcel-Days:\n    default: 3\n",  # not a kind name
        "  parcel_days:\n    default: 40\n    max: 30\n",  # default out of its range
        "  parcel_days:\n    default: 3\n    unit: weeks\n",  # unknown field
    ],
)
def test_a_declaration_that_does_not_fit_is_skipped_and_refused_by_the_manifest(
    home: Path, block: str, caplog: pytest.LogCaptureFixture
) -> None:
    _install(home, "demo", block)
    with caplog.at_level(logging.WARNING):
        declared = declared_expiry_kinds()
    assert not {"task_overdue_days", "Parcel-Days", "parcel_days"} & set(declared)
    assert "expiry kind" in caplog.text
    import yaml

    raw = yaml.safe_load((home / "plugins" / "demo" / "manifest.yaml").read_text())
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(raw)


def test_two_plugins_declaring_one_kind_first_wins_loudly(
    home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _install(home, "alpha", "  parcel_days:\n    default: 4\n")
    _install(home, "beta", "  parcel_days:\n    default: 9\n")
    with caplog.at_level(logging.WARNING):
        declared = declared_expiry_kinds()
    assert declared["parcel_days"].default == 4
    assert "already declared by 'alpha'" in caplog.text


def test_the_manifest_accepts_a_declaration() -> None:
    manifest = PluginManifest.model_validate(
        {"name": "demo", "expiry": {"parcel_days": {"default": 4, "description": "parcels"}}}
    )
    assert manifest.expiry["parcel_days"].default == 4
    assert manifest.expiry["parcel_days"].max == 365
