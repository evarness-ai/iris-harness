"""``party``: a plugin's provenance, declared apart from ``trust`` (how it runs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.runtime.plugin_host.manifest import PluginManifest, load_manifest


def _write(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(f"name: demo\n{extra}", encoding="utf-8")
    return path


def test_party_defaults_to_first_party(tmp_path: Path) -> None:
    assert load_manifest(_write(tmp_path)).party == "first-party"


@pytest.mark.parametrize("party", ["first-party", "trusted-third-party", "untrusted"])
def test_each_declared_party_loads(tmp_path: Path, party: str) -> None:
    assert load_manifest(_write(tmp_path, f"party: {party}\n")).party == party


def test_an_unknown_party_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="party"):
        load_manifest(_write(tmp_path, "party: friendly\n"))


def test_party_is_independent_of_trust(tmp_path: Path) -> None:
    manifest = load_manifest(_write(tmp_path, "party: untrusted\ntrust: in-process\n"))
    assert (manifest.party, manifest.trust) == ("untrusted", "in-process")


def test_no_bundled_manifest_needs_a_party() -> None:
    assert PluginManifest(name="bundled").party == "first-party"
