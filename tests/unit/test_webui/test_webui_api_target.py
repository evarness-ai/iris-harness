"""The local web UI can watch another harness, and must say when it does.

The UI points at a remote harness with IRIS_API_URL. Reading another harness's numbers
as this machine's would make every screen quietly wrong, so the switch always comes
with a visible badge.

These are structural checks: webui has no JS test runner, and the wiring is what
breaks silently (a renamed env var, a badge that stops being mounted).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VITE = ROOT / "webui" / "vite.config.ts"
APP = ROOT / "webui" / "src" / "App.tsx"
BADGE = ROOT / "webui" / "src" / "components" / "TargetBadge.tsx"


def test_the_proxy_target_can_be_pointed_at_another_harness() -> None:
    config = VITE.read_text(encoding="utf-8")

    assert "process.env.IRIS_API_URL ?? `http://127.0.0.1:${apiPort}`" in config
    assert "VITE_IRIS_TARGET_LABEL" in config, "the UI is told which harness it shows"


def test_the_local_default_is_unchanged() -> None:
    """Without the flag this is the ordinary local UI, badge included: nothing shows."""
    config = VITE.read_text(encoding="utf-8")

    assert 'process.env.IRIS_API_LABEL ?? (process.env.IRIS_API_URL ? target : "")' in config
    assert "if (!label) return null;" in BADGE.read_text(encoding="utf-8")


def test_the_badge_prefers_the_servers_own_label() -> None:
    """The console in the server image is built once for every deployment, so the build
    cannot name the harness. The server does (``deployment_label`` in /capabilities,
    IRIS_DEPLOYMENT_LABEL); the build-time label is the fallback, never the override."""
    badge = BADGE.read_text(encoding="utf-8")
    client = (ROOT / "webui" / "src" / "lib" / "control.ts").read_text(encoding="utf-8")

    assert "useCapabilities().data?.deployment_label" in badge
    assert "const label = runtimeLabel || buildLabel;" in badge
    assert "import.meta.env.VITE_IRIS_TARGET_LABEL" in badge, "a build-time label still works"
    assert "deployment_label?: string;" in client


def test_the_badge_is_mounted_in_the_shell() -> None:
    app = APP.read_text(encoding="utf-8")

    assert "TargetBadge" in app
    assert app.index("<TargetBadge />") < app.index("<HealthBanner />"), "above the fold"
