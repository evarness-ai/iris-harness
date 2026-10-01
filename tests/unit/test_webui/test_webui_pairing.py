"""A browser pairs itself, and an unpaired one is sent to /pair — never to mock data.

Served by the API (no Vite proxy injecting the secret) the console authenticates with an
HttpOnly ``iris_device`` cookie (ADR-0117). That only holds if every data call goes through
one wrapper that turns a 401 into the pairing screen, and if no token ever reaches JS.

These are structural checks: webui has no JS test runner, and these are the lines that
break silently (one new raw ``fetch`` and that screen shows "API unavailable" — or canned
mock traces — to an unpaired phone instead of asking it to pair).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "webui" / "src"
HTTP = SRC / "lib" / "http.ts"
CLIENT = SRC / "lib" / "client.ts"
ROUTES = SRC / "routes.tsx"
MAIN = SRC / "main.tsx"
PAIR = SRC / "screens" / "Pair.tsx"

# `fetch(` as a call — not apiFetch(, refetch(, prefetch(, or a `.fetch(` method.
RAW_FETCH = re.compile(r"(?<![A-Za-z0-9_.])fetch\(")


def _sources() -> list[Path]:
    return sorted(p for p in SRC.rglob("*") if p.suffix in {".ts", ".tsx"})


def _group_routes(group_id: str) -> set[str]:
    """The routes the core declares in one nav group (config/webui/nav.yaml, OSS plan R17)."""
    nav = yaml.safe_load((ROOT / "config" / "webui" / "nav.yaml").read_text(encoding="utf-8"))
    return {s["route"] for s in nav["screens"] if s["group"] == group_id and s.get("nav", True)}


def test_every_data_call_goes_through_the_wrapper() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in _sources()
        if path != HTTP
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if RAW_FETCH.search(line)
    ]

    assert offenders == [], f"raw fetch( outside lib/http.ts: {offenders}"
    assert len(RAW_FETCH.findall(HTTP.read_text(encoding="utf-8"))) == 1, "one door"


def test_a_401_goes_to_the_pairing_screen_once() -> None:
    http = HTTP.read_text(encoding="utf-8")

    assert "res.status === 401 && pathOf(input) !== CLAIM_PATH" in http, "the claim is exempt"
    assert "if (redirecting || pathname === PAIR_PATH) return;" in http, "once, not from /pair"
    assert "else window.location.assign(to);" in http, "works with no router registered"
    assert "onUnauthorized(" in MAIN.read_text(encoding="utf-8"), "the router is registered"


def test_next_is_a_same_origin_path() -> None:
    """``/pair?next=`` is attacker-writable: ``//host`` and ``/\\host`` leave the origin."""
    http = HTTP.read_text(encoding="utf-8")
    guard = 'if (!raw || !raw.startsWith("/") || raw.startsWith("//") || raw.includes("\\\\"))'

    assert guard in http
    pair = PAIR.read_text(encoding="utf-8")
    assert 'safeNext(params.get("next"))' in pair, "the screen only navigates to a guarded next"
    assert "navigate(next," in pair
    assert "safeNext(pathname + search)" in http, "and the wrapper only writes a guarded one"


def test_the_pair_route_sits_outside_the_shell() -> None:
    """Inside AppLayout the shell's own queries would 401 on the pairing screen."""
    routes = ROUTES.read_text(encoding="utf-8")

    assert 'path: "pair"' in routes
    assert routes.index('path: "pair"') < routes.index("element: <AppLayout />")


def test_devices_sits_beside_governance() -> None:
    routes = ROUTES.read_text(encoding="utf-8")
    group = _group_routes("system")

    assert "/governance" in group
    assert "/devices" in group, "Devices is in the group that holds Governance"
    assert 'path: "devices"' in routes, "and the route is mounted"


def test_no_token_ever_reaches_browser_storage() -> None:
    """The cookie is HttpOnly on purpose. Storage calls exist (theme, nav state, chat
    session id); none of them may carry a credential."""
    storage = re.compile(r"(localStorage|sessionStorage|document\.cookie)")
    secretish = re.compile(r"token|iris_device|irisd_|secret|bearer", re.IGNORECASE)
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in _sources()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if storage.search(line)
        and secretish.search(line)
        and not line.lstrip().startswith(("*", "/"))
    ]

    assert offenders == []
    assert "Authorization" not in HTTP.read_text(encoding="utf-8"), "the cookie is the credential"
    assert 'credentials: "same-origin"' in HTTP.read_text(encoding="utf-8")


def test_the_console_carries_no_mock_data() -> None:
    """A 401 or an unreachable API is an error the screen shows, never canned data.

    The console once fell back to mock traces and sessions when the API was down (and,
    for sessions, when it was empty), so an unpaired phone or a fresh install showed
    someone else's turns. There is no mock data to fall back to now; keep it that way.
    """
    assert not (SRC / "mock").exists(), "mock data belongs under webui/tests, not src/"
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in _sources()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"MOCK_|['\"]mock['\"]|mock data", line)
    ]
    assert offenders == []
