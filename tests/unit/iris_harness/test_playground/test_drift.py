"""Drift detection tests — declared config vs runtime-registered."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from iris_harness.playground.drift import build_drift_report


def _write_config(root: Path) -> None:
    skills = root / "skills" / "finance" / "finance-query"
    skills.mkdir(parents=True)
    (skills / "manifest.yaml").write_text("name: finance-query\n", encoding="utf-8")
    auto = root / "skills" / "auto" / "ghost"
    auto.mkdir(parents=True)
    (auto / "manifest.yaml").write_text("name: ghost-skill\n", encoding="utf-8")
    (root / "llm_tiers.yaml").write_text("tiers:\n  tier1: {}\n  tier2: {}\n", encoding="utf-8")
    (root / "channels.yaml").write_text(
        "channels:\n  - type: console\n  - type: telegram\n", encoding="utf-8"
    )


def _runtime(*, skills: list[str], tiers: list[str], channels: list[str]) -> SimpleNamespace:
    registry = SimpleNamespace(
        list_packages=lambda: tuple(
            SimpleNamespace(manifest=SimpleNamespace(name=n)) for n in skills
        )
    )
    router = SimpleNamespace(_tiers={t: object() for t in tiers})
    gateway = SimpleNamespace(channels=lambda: sorted(channels))
    return SimpleNamespace(skill_registry=registry, tier_router=router, channels=gateway)


def test_in_sync_reports_ok(tmp_path: Path) -> None:
    _write_config(tmp_path)
    runtime = _runtime(
        skills=["finance-query"], tiers=["tier1", "tier2"], channels=["console", "telegram"]
    )
    report = build_drift_report(runtime, config_dir=tmp_path)
    assert report.ok
    skills = next(s for s in report.surfaces if s.surface == "skills")
    assert skills.in_sync == ("finance-query",)


def test_auto_quarantine_skills_are_not_declared(tmp_path: Path) -> None:
    # config/skills/auto/* is the crystallizer quarantine — the registry skips
    # it, so drift must too (else it would always show as registered_only).
    _write_config(tmp_path)
    runtime = _runtime(
        skills=["finance-query"], tiers=["tier1", "tier2"], channels=["console", "telegram"]
    )
    report = build_drift_report(runtime, config_dir=tmp_path)
    skills = next(s for s in report.surfaces if s.surface == "skills")
    assert "ghost-skill" not in skills.declared_only


def test_declared_only_when_not_registered(tmp_path: Path) -> None:
    _write_config(tmp_path)
    runtime = _runtime(skills=[], tiers=["tier1"], channels=["console"])
    report = build_drift_report(runtime, config_dir=tmp_path)
    assert not report.ok
    skills = next(s for s in report.surfaces if s.surface == "skills")
    tiers = next(s for s in report.surfaces if s.surface == "llm_tiers")
    channels = next(s for s in report.surfaces if s.surface == "channels")
    assert skills.declared_only == ("finance-query",)
    assert tiers.declared_only == ("tier2",)
    assert channels.declared_only == ("telegram",)


def test_registered_only_when_no_config(tmp_path: Path) -> None:
    _write_config(tmp_path)
    runtime = _runtime(
        skills=["finance-query", "surprise"],
        tiers=["tier1", "tier2"],
        channels=["console", "telegram"],
    )
    report = build_drift_report(runtime, config_dir=tmp_path)
    skills = next(s for s in report.surfaces if s.surface == "skills")
    assert skills.registered_only == ("surprise",)
    assert not report.ok


def test_to_dict_shape(tmp_path: Path) -> None:
    _write_config(tmp_path)
    runtime = _runtime(
        skills=["finance-query"], tiers=["tier1", "tier2"], channels=["console", "telegram"]
    )
    d = build_drift_report(runtime, config_dir=tmp_path).to_dict()
    assert d["ok"] is True
    assert {s["surface"] for s in d["surfaces"]} == {
        "skills",
        "llm_tiers",
        "channels",
        "intercepts",
        "plugin_tools",  # ADR-0110
        "plugin_uses",  # plugin-capabilities §4: a grant naming no registered tool
        "plugin_capabilities",  # §2: a declared capability never provided
        "capability_uses",  # §2: a capability used that nothing provides
    }


def test_intercept_with_missing_handler_is_declared_only(tmp_path: Path) -> None:
    _write_config(tmp_path)
    runtime = _runtime(
        skills=["finance-query"], tiers=["tier1", "tier2"], channels=["console", "telegram"]
    )
    # Declared chain: one wired handler, one whose method is absent.
    runtime._handle_time_date_turn = lambda *a, **k: None
    runtime.intercept_chain = (
        SimpleNamespace(name="time_date", handler="_handle_time_date_turn"),
        SimpleNamespace(name="ghost", handler="_handle_ghost_turn"),
    )
    report = build_drift_report(runtime, config_dir=tmp_path)
    intercepts = next(s for s in report.surfaces if s.surface == "intercepts")
    assert intercepts.declared_only == ("ghost",)  # handler method does not exist
    assert intercepts.in_sync == ("time_date",)
    assert not report.ok


def test_plugin_tools_surface_compares_manifest_declarations_with_registrations(
    tmp_path: Path,
) -> None:
    """ADR-0110: a declared-but-unregistered tool is drift; so is a registered-but-undeclared
    one (register_tool refuses those, so it can only appear via a bare PluginRecord)."""
    _write_config(tmp_path)
    records = {
        "p": SimpleNamespace(
            manifest=SimpleNamespace(tools={"declared_ok": {}, "declared_missing": {}})
        )
    }
    plugin_registry = SimpleNamespace(
        _plugins=records,
        tools=lambda: [SimpleNamespace(name="declared_ok"), SimpleNamespace(name="stray")],
        intercept=lambda name: None,
    )
    runtime = _runtime(skills=[], tiers=["tier1"], channels=["console"])
    runtime.plugin_registry = plugin_registry
    report = build_drift_report(runtime, config_dir=tmp_path)
    surface = next(s for s in report.surfaces if s.surface == "plugin_tools")
    assert surface.declared_only == ("declared_missing",)
    assert surface.registered_only == ("stray",)
    assert surface.in_sync == ("declared_ok",)
