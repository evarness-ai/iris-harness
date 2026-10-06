"""``party``: a plugin's provenance, declared apart from ``trust`` (how it runs)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from iris_harness.runtime.plugin_host.manifest import PluginManifest, load_manifest


def _write(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(f"name: demo\n{extra}", encoding="utf-8")
    return path


def test_party_defaults_to_untrusted(tmp_path: Path) -> None:
    """Fail closed (ADR-0134): a manifest that says nothing is not first-party."""
    assert load_manifest(_write(tmp_path)).party == "untrusted"
    assert PluginManifest(name="anon").party == "untrusted"


def test_an_explicit_first_party_round_trips(tmp_path: Path) -> None:
    manifest = load_manifest(_write(tmp_path, "party: first-party\n"))
    assert manifest.party == "first-party"
    assert PluginManifest.model_validate(manifest.model_dump()).party == "first-party"


@pytest.mark.parametrize("party", ["first-party", "trusted-third-party", "untrusted"])
def test_each_declared_party_loads(tmp_path: Path, party: str) -> None:
    assert load_manifest(_write(tmp_path, f"party: {party}\n")).party == party


def test_an_unknown_party_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="party"):
        load_manifest(_write(tmp_path, "party: friendly\n"))


def test_party_is_independent_of_trust(tmp_path: Path) -> None:
    manifest = load_manifest(_write(tmp_path, "party: untrusted\ntrust: in-process\n"))
    assert (manifest.party, manifest.trust) == ("untrusted", "in-process")


_ROOT = Path(__file__).resolve().parents[5]
_SHIPPED = sorted(
    [
        *(_ROOT / "src/iris_harness/plugins_builtin").glob("*/manifest.yaml"),
        *(_ROOT / "src/iris_personal/plugins").glob("*/manifest.yaml"),
    ]
)
_OUTSIDE_AUTHORS = sorted(
    [
        *(_ROOT / "examples").glob("*/manifest.yaml"),
        *(_ROOT / "src/iris_harness/cli/templates/plugin").glob("*/src/*/manifest.yaml"),
    ]
)


def test_the_shipped_manifests_are_found() -> None:
    assert len(_SHIPPED) >= 10 and len(_OUTSIDE_AUTHORS) >= 5


@pytest.mark.parametrize("path", _SHIPPED, ids=lambda p: p.parent.name)
def test_a_bundled_plugin_declares_first_party_explicitly(path: Path) -> None:
    assert "party: first-party" in path.read_text(encoding="utf-8")
    assert load_manifest(path).party == "first-party"


@pytest.mark.parametrize("path", _OUTSIDE_AUTHORS, ids=lambda p: str(p.relative_to(_ROOT)))
def test_examples_and_scaffolds_do_not_claim_first_party(path: Path) -> None:
    """What an outside author copies must not hand them the project's provenance."""
    manifest_text = path.read_text(encoding="utf-8")
    assert "party: first-party" not in manifest_text


def _every_plugin_manifest() -> list[Path]:
    found = [
        p
        for p in _ROOT.rglob("manifest.yaml")
        if not {".venv", "node_modules", "skills", ".git", "htmlcov"}
        & set(p.relative_to(_ROOT).parts)
    ]
    return sorted(found)


@pytest.mark.parametrize("path", _every_plugin_manifest(), ids=lambda p: str(p.relative_to(_ROOT)))
def test_every_plugin_manifest_in_the_repo_declares_party(path: Path) -> None:
    """Nothing the repo ships may trigger the loader's omitted-`party` notice."""
    # Raw YAML: a scaffold's name is a `__tmpl_` placeholder that load_manifest refuses.
    assert "party" in yaml.safe_load(path.read_text(encoding="utf-8")), f"{path} omits `party`"


def test_the_manifest_sweep_found_the_shipped_ones() -> None:
    assert len(_every_plugin_manifest()) >= len(_SHIPPED) + len(_OUTSIDE_AUTHORS)


@pytest.mark.parametrize("path", _OUTSIDE_AUTHORS, ids=lambda p: str(p.relative_to(_ROOT)))
def test_examples_and_scaffolds_say_untrusted_explicitly(path: Path) -> None:
    assert yaml.safe_load(path.read_text(encoding="utf-8")).get("party") == "untrusted"
