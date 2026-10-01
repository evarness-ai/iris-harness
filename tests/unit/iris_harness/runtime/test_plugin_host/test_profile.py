"""Profile layering: shipped → home overlay → env, with provenance."""

from __future__ import annotations

from pathlib import Path

from iris_harness.foundation.paths import default_config_dir
from iris_harness.runtime.plugin_host.profile import (
    DEFAULT_PLUGINS,
    _installed_plugins,
    _read_layer,
    list_profiles,
    load_profile,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_missing_shipped_profile_falls_back_to_builtin_default(tmp_path: Path) -> None:
    prof = load_profile(tmp_path, "nothing", home_dir=tmp_path / "home", env={})
    assert prof.name == "nothing"
    # The fallback is "what this installation actually ships": the built-ins plus every
    # installed entry-point plugin (M6.1b -- the six domains on a machine that has them).
    # Assert against the discovery, not a frozen list, and never against domain names:
    # the core knows none of them.
    installed = _installed_plugins()
    assert set(DEFAULT_PLUGINS) <= set(installed)
    assert [p.name for p in prof.plugins] == list(installed)
    assert prof.provenance == dict.fromkeys(installed, "built-in default")
    assert "missing" in prof.layers[0]


def test_shipped_profile_and_listing(tmp_path: Path) -> None:
    _write(
        tmp_path / "profiles" / "default.yaml",
        "name: default\ndescription: d\nplugins:\n  - name: system\n  - name: extra\n",
    )
    _write(tmp_path / "profiles" / "minimal.yaml", "plugins: [{name: system}]\n")
    assert list_profiles(tmp_path) == ["default", "minimal"]
    prof = load_profile(tmp_path, home_dir=tmp_path / "home", env={})
    assert prof.name == "default" and prof.description == "d"
    assert [p.name for p in prof.enabled_plugins()] == ["system", "extra"]
    assert prof.provenance == {"system": "shipped", "extra": "shipped"}


def test_home_overlay_merges_by_name_and_replaces_order(tmp_path: Path) -> None:
    _write(
        tmp_path / "profiles" / "default.yaml",
        "plugins:\n  - name: system\n  - name: extra\nintercept_order: [a, b]\n",
    )
    home = tmp_path / "home"
    _write(
        home / "profile.yaml",
        "plugins:\n  - name: extra\n    enabled: false\n  - name: mine\n    trust: mcp\n"
        "intercept_order: [z]\n",
    )
    prof = load_profile(tmp_path, home_dir=home, env={})
    by_name = {p.name: p for p in prof.plugins}
    assert by_name["extra"].enabled is False
    assert by_name["mine"].trust == "mcp"
    assert by_name["system"].enabled is True
    assert prof.intercept_order == ["z"]
    assert prof.provenance == {"system": "shipped", "extra": "home", "mine": "home"}
    assert any(layer.startswith("home:") for layer in prof.layers)


def test_env_selects_profile_and_toggles_plugins(tmp_path: Path) -> None:
    _write(tmp_path / "profiles" / "alt.yaml", "plugins:\n  - name: system\n  - name: extra\n")
    env = {
        "IRIS_PROFILE": "alt",
        "IRIS_PLUGINS_DISABLE": "extra",
        "IRIS_PLUGINS_ENABLE": "bonus",
    }
    prof = load_profile(tmp_path, home_dir=tmp_path / "home", env=env)
    assert prof.name == "alt"
    by_name = {p.name: p for p in prof.plugins}
    assert by_name["extra"].enabled is False
    assert by_name["bonus"].enabled is True
    assert prof.provenance["extra"] == "env" and prof.provenance["bonus"] == "env"
    assert prof.as_dict()["plugins"][1]["set_by"] == "env"


def test_invalid_layer_is_skipped_not_fatal(tmp_path: Path) -> None:
    _write(tmp_path / "profiles" / "default.yaml", "plugins:\n  - name: system\n    bogus: 1\n")
    prof = load_profile(tmp_path, home_dir=tmp_path / "home", env={})
    # invalid shipped layer → treated as missing → what this installation ships
    assert [p.name for p in prof.plugins] == list(_installed_plugins())


# -- which profile an unnamed run picks (prefer_when_installed) ----------------------


def _home_plugin(home: Path, name: str, *, requires: str = "") -> None:
    """A plugin installed under ``$IRIS_HOME/plugins``: found by the same discovery."""
    body = f"name: {name}\nentrypoint: plugin:setup\n"
    if requires:
        body += f"requires:\n  packages: [{requires}]\n"
    _write(home / "plugins" / name / "manifest.yaml", body)
    _write(home / "plugins" / name / "plugin.py", "def setup(api):\n    pass\n")


def _preferring_config(root: Path) -> Path:
    config = root / "config"
    _write(
        config / "profiles" / "default.yaml",
        "name: default\nplugins: [{name: system}]\nprefer_when_installed: [mail]\n",
    )
    _write(
        config / "profiles" / "mail.yaml",
        "name: mail\nplugins: [{name: system}, {name: mailbox}]\n",
    )
    return config


def test_unnamed_run_takes_the_preferred_profile_when_its_plugins_are_installed(
    tmp_path: Path,
) -> None:
    config = _preferring_config(tmp_path)
    home = tmp_path / "home"
    _home_plugin(home, "mailbox")
    prof = load_profile(config, home_dir=home, env={})
    assert prof.name == "mail"
    assert [p.name for p in prof.plugins] == ["system", "mailbox"]
    assert prof.layers[0].startswith("selected: mail")


def test_unnamed_run_stays_default_when_a_plugin_is_missing(tmp_path: Path) -> None:
    config = _preferring_config(tmp_path)
    prof = load_profile(config, home_dir=tmp_path / "home", env={})
    assert prof.name == "default"
    assert not any(layer.startswith("selected:") for layer in prof.layers)


def test_unnamed_run_stays_default_when_a_plugin_lacks_its_packages(tmp_path: Path) -> None:
    """The one-distribution case: the plugin ships, the extra it needs is not installed."""
    config = _preferring_config(tmp_path)
    home = tmp_path / "home"
    _home_plugin(home, "mailbox", requires="iris-no-such-package-for-tests")
    assert load_profile(config, home_dir=home, env={}).name == "default"


def test_a_named_profile_is_taken_as_named(tmp_path: Path) -> None:
    config = _preferring_config(tmp_path)
    home = tmp_path / "home"
    _home_plugin(home, "mailbox")
    assert load_profile(config, home_dir=home, env={"IRIS_PROFILE": "default"}).name == "default"
    assert load_profile(config, "default", home_dir=home, env={}).name == "default"


def test_the_home_overlay_still_applies_to_the_preferred_profile(tmp_path: Path) -> None:
    config = _preferring_config(tmp_path)
    home = tmp_path / "home"
    _home_plugin(home, "mailbox")
    _write(home / "profile.yaml", "plugins: [{name: mailbox, enabled: false}]\n")
    prof = load_profile(config, home_dir=home, env={})
    assert prof.name == "mail"
    assert [p.name for p in prof.enabled_plugins()] == ["system"]


def test_shipped_default_prefers_only_profiles_that_exist() -> None:
    config = default_config_dir()
    default = _read_layer(config / "profiles" / "default.yaml")
    assert default is not None
    assert default.prefer_when_installed == ["email"]
    for name in default.prefer_when_installed:
        assert _read_layer(config / "profiles" / f"{name}.yaml") is not None, name


def test_the_email_profile_is_default_plus_the_email_plugins() -> None:
    """The release-1 product (OSS plan R1/R2), read from the YAML alone: no plugin is
    looked up, so this holds on a core-only install too. The mailbox providers are
    optional -- whichever is installed mounts -- and at least one is required."""
    config = default_config_dir()
    default = load_profile(config, "default", env={})
    email = _read_layer(config / "profiles" / "email.yaml")
    assert email is not None
    assert [p.name for p in email.plugins] == [p.name for p in default.plugins] + [
        "gmail",
        "imap",
        "email_workflows",
    ]
    assert {p.name for p in email.plugins if p.optional} == {"gmail", "imap"}
    assert email.requires_one_of == [["gmail", "imap"]]


# --- optional rows and requires_one_of (owner decision 2026-09-30) -------------------


def _providers_config(root: Path) -> Path:
    config = root / "config"
    _write(
        config / "profiles" / "default.yaml",
        "name: default\nplugins: [{name: system}]\nprefer_when_installed: [mail]\n",
    )
    _write(
        config / "profiles" / "mail.yaml",
        "name: mail\nplugins:\n  - {name: system}\n  - {name: cloudbox, optional: true}\n"
        "  - {name: plainbox, optional: true}\n  - {name: workflows}\n"
        "requires_one_of: [[cloudbox, plainbox]]\n",
    )
    return config


def test_an_optional_plugin_that_is_not_installed_is_left_out_not_failed(
    tmp_path: Path,
) -> None:
    config = _providers_config(tmp_path)
    home = tmp_path / "home"
    _home_plugin(home, "workflows")
    _home_plugin(home, "plainbox")
    _home_plugin(home, "cloudbox", requires="iris-no-such-package-for-tests")
    prof = load_profile(config, "mail", home_dir=home, env={})
    assert [p.name for p in prof.plugins] == ["system", "plainbox", "workflows"]
    assert any(line.startswith("optional: cloudbox") for line in prof.layers)
    assert "cloudbox" not in prof.provenance


def test_unnamed_run_takes_the_profile_with_any_one_provider(tmp_path: Path) -> None:
    """The IMAP-only install: no Google packages, still the email assistant."""
    config = _providers_config(tmp_path)
    home = tmp_path / "home"
    _home_plugin(home, "workflows")
    _home_plugin(home, "plainbox")
    prof = load_profile(config, home_dir=home, env={})
    assert prof.name == "mail"
    assert [p.name for p in prof.plugins] == ["system", "plainbox", "workflows"]


def test_unnamed_run_stays_default_with_no_provider_or_no_workflows(tmp_path: Path) -> None:
    config = _providers_config(tmp_path)
    home = tmp_path / "home"
    _home_plugin(home, "workflows")  # no provider at all
    assert load_profile(config, home_dir=home, env={}).name == "default"
    other = tmp_path / "other"
    _home_plugin(other, "plainbox")  # a provider, but not the required workflows
    assert load_profile(config, home_dir=other, env={}).name == "default"


def test_env_enable_makes_an_optional_plugin_required(tmp_path: Path) -> None:
    config = _providers_config(tmp_path)
    home = tmp_path / "home"
    _home_plugin(home, "workflows")
    prof = load_profile(config, "mail", home_dir=home, env={"IRIS_PLUGINS_ENABLE": "cloudbox"})
    cloud = next(p for p in prof.plugins if p.name == "cloudbox")
    assert cloud.optional is False  # asked for by name: a missing one shows as failed
