"""The console can be added to the Home Screen (mobile-cloud-ui plan, track 2b PR 8).

This is a prerequisite, not polish: iOS delivers Web Push only to a home-screen
web app, and only through a service worker. PR 9's push handler has nowhere to
live until this holds.

Structural checks against the source tree. The browser half is covered by
``webui/tests/viewport.spec.ts``; what breaks silently here is the wiring
between three separate places that have to agree — ``webui/public/`` (what the
build ships), ``static_ui.ROOT_FILES`` (what the server will serve) and
``index.html`` (what the browser is told to look for). Any one of them drifting
means "Add to Home Screen" quietly stops offering an app.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from iris_harness.server.iris_api.static_ui import ROOT_FILES

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT / "webui" / "public"
INDEX = ROOT / "webui" / "index.html"
MANIFEST = PUBLIC / "manifest.webmanifest"


def test_the_build_ships_every_file_the_server_will_serve() -> None:
    """``webui/public`` is copied verbatim to the build root by Vite.

    A name on the server's allowlist with no file behind it is a 404 the
    server promises not to produce.
    """
    missing = sorted(name for name in ROOT_FILES if not (PUBLIC / name).is_file())
    assert not missing, f"on static_ui.ROOT_FILES but absent from webui/public: {missing}"


def test_the_server_will_serve_every_file_the_build_ships() -> None:
    """And the other direction: a file in ``public/`` the server never serves
    is dead weight that looks like it works until someone requests it."""
    unserved = sorted(p.name for p in PUBLIC.iterdir() if p.name not in ROOT_FILES)
    assert not unserved, f"in webui/public but absent from static_ui.ROOT_FILES: {unserved}"


@pytest.fixture()
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_the_manifest_is_valid_json(manifest: dict) -> None:
    assert manifest["name"]
    assert manifest["short_name"]


def test_display_is_standalone(manifest: dict) -> None:
    """The one field that decides whether iOS treats this as an app.

    WebKit delivers Web Push only to a home-screen web app, and a manifest is
    only a home-screen web app when `display` is standalone or fullscreen. Any
    other value and PR 9 cannot work, however correct its code is.
    """
    assert manifest["display"] in {"standalone", "fullscreen"}


def test_the_scope_covers_the_whole_app(manifest: dict) -> None:
    # A narrower scope would send in-app navigations back out to Safari.
    assert manifest["scope"] == "/"


def test_the_start_url_is_inside_the_scope(manifest: dict) -> None:
    assert manifest["start_url"].startswith(manifest["scope"])


def test_every_manifest_icon_exists_and_is_declared(manifest: dict) -> None:
    for icon in manifest["icons"]:
        name = icon["src"].lstrip("/")
        assert (PUBLIC / name).is_file(), f"manifest points at a missing icon: {name}"
        assert name in ROOT_FILES, f"manifest icon {name} is not served by static_ui"


def test_there_is_a_maskable_icon(manifest: dict) -> None:
    # Without one, Android crops the square icon into a circle and clips it.
    purposes = {p for icon in manifest["icons"] for p in icon.get("purpose", "any").split()}
    assert "maskable" in purposes


def test_the_theme_colour_matches_the_console(manifest: dict) -> None:
    # tokens.css .dark --bg: 12 15 20. A mismatched theme colour shows as a
    # band of the wrong colour above the app on a phone.
    assert manifest["background_color"].lower() == "#0c0f14"
    assert manifest["theme_color"].lower() == "#0c0f14"


def test_index_links_the_manifest_and_the_ios_icon() -> None:
    html = INDEX.read_text(encoding="utf-8")
    assert 'rel="manifest"' in html, "no manifest link: the browser never sees it"
    # iOS reads apple-touch-icon in preference to the manifest's icons.
    assert 'rel="apple-touch-icon"' in html


def test_the_viewport_covers_the_notch() -> None:
    # `viewport-fit=cover` is what makes env(safe-area-inset-*) resolve to the
    # real insets; without it the bottom tab bar ignores the home indicator.
    html = INDEX.read_text(encoding="utf-8")
    assert "viewport-fit=cover" in html


def test_the_worker_is_registered_in_production_only() -> None:
    main = (ROOT / "webui" / "src" / "main.tsx").read_text(encoding="utf-8")
    assert "navigator.serviceWorker" in main and '"/sw.js"' in main
    # In development Vite serves public/ too, and a worker surviving HMR
    # reloads is a confusing thing to debug for no gain.
    assert "import.meta.env.PROD" in main


def _sw_code() -> str:
    """The worker with its comments stripped.

    The comments explain at length why the shell is not cached, and naming the
    thing you are avoiding should not fail a test that looks for it.
    """
    source = (PUBLIC / "sw.js").read_text(encoding="utf-8")
    without_block = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", without_block, flags=re.MULTILINE)


def test_the_worker_does_not_cache_the_app_shell() -> None:
    """The failure this avoids is subtle and total.

    A cache-first shell eventually serves an index.html naming content-hashed
    assets that the current build no longer contains, and the app is a white
    screen until someone clears site data. Navigations go to the network; only
    the offline page is cached.
    """
    sw = _sw_code()
    assert "index.html" not in sw
    assert "offline.html" in sw


def test_the_worker_leaves_credentialed_requests_alone() -> None:
    # Data calls carry the HttpOnly pairing cookie; a worker that served any of
    # them from a cache would be handing back another session's answer.
    assert 'mode !== "navigate"' in _sw_code()


# ── Web Push (track 2b PR 9) ───────────────────────────────────────────────


def test_the_worker_handles_push_and_taps() -> None:
    """On iOS a push is delivered to a service worker or not at all."""
    sw = _sw_code()
    assert '"push"' in sw, "no push handler: iOS has nowhere to deliver"
    assert '"notificationclick"' in sw, "a notification nobody can tap is half a feature"


def test_the_worker_always_shows_something_for_a_push() -> None:
    # iOS revokes the push permission of a web app that receives a push and
    # displays no notification, so the handler must not have a silent path.
    sw = _sw_code()
    assert "showNotification" in sw
    assert "Something needs you" in sw, "no fallback body for an unparseable payload"


def test_the_worker_tags_each_reminder_and_renotifies_on_request() -> None:
    # Reminders arrive with their own tag (`reminder:<id>`): the worker must use the
    # payload's tag, or two reminders collapse into one banner. A snoozed reminder
    # returns under the same tag and asks to alert again.
    sw = _sw_code()
    assert "tag: payload.tag ||" in sw
    assert "payload.renotify" in sw
