"""The IRIS API serving the built web console (mobile-cloud-ui plan, track 1 PR 1).

What these pin, in the order the risk runs:

- the layer never opens the API: without a credential, only files from the build
  come back, and a forged ``Sec-Fetch-Mode`` on anything but a ``GET`` changes nothing;
- the navigate rule is the one Vite applies in development, so ``/health`` is the
  console in a browser tab and JSON to a ``fetch``;
- with no build on disk the app is the app it was before this module existed.

A dummy runtime is passed so no real one is built.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api import static_ui
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.health import service
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthSnapshot, HealthState

_INDEX_HTML = '<!DOCTYPE html><html><body><div id="root"></div></body></html>'
_ASSET_JS = 'console.log("iris console");'
_ASSET_CSS = "body{margin:0}"
_OUTSIDE = "not part of the build"

_NAVIGATE = {"Sec-Fetch-Mode": "navigate"}


def _app() -> FastAPI:
    return create_app(runtime=SimpleNamespace(), auto_start_runtime=False)


@pytest.fixture
def dist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal production build, with a file beside it that must never be served."""
    build = tmp_path / "dist"
    (build / "assets").mkdir(parents=True)
    (build / "index.html").write_text(_INDEX_HTML, encoding="utf-8")
    (build / "assets" / "index-abc123.js").write_text(_ASSET_JS, encoding="utf-8")
    (build / "assets" / "index-abc123.css").write_text(_ASSET_CSS, encoding="utf-8")
    # Root-level build output (track 2b PR 8).
    (build / "manifest.webmanifest").write_text('{"name":"IRIS Console"}', encoding="utf-8")
    (build / "sw.js").write_text("self.addEventListener('install', () => {});", encoding="utf-8")
    (build / "offline.html").write_text("<!doctype html><title>offline</title>", encoding="utf-8")
    (build / "icon-192.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "outside.txt").write_text(_OUTSIDE, encoding="utf-8")
    monkeypatch.setenv(static_ui.WEBUI_DIST_ENV, str(build))
    return build


@pytest.fixture
def anonymous(dist: Path) -> Iterator[TestClient]:
    """A browser that has not authenticated: no bearer token on any request."""
    with TestClient(_app()) as client:
        yield client


@pytest.fixture
def authed(dist: Path) -> Iterator[TestClient]:
    with TestClient(_app(), headers=auth_headers()) as client:
        yield client


@pytest.fixture
def no_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The variable points at a directory that holds no build."""
    monkeypatch.setenv(static_ui.WEBUI_DIST_ENV, str(tmp_path / "absent"))
    with TestClient(_app()) as client:
        yield client


# ── the navigate rule ────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", ["/", "/health", "/routines", "/governance/devices", "/chat"])
def test_a_browser_navigation_gets_the_shell_without_a_token(
    anonymous: TestClient, path: str
) -> None:
    resp = anonymous.get(path, headers=_NAVIGATE)
    assert resp.status_code == 200
    assert resp.text == _INDEX_HTML
    assert resp.headers["content-type"].startswith("text/html")


def test_the_shell_is_revalidated_on_every_load(anonymous: TestClient) -> None:
    # index.html names the current content-hashed assets; a cached copy would
    # keep pointing a browser at a build that has been replaced.
    resp = anonymous.get("/", headers=_NAVIGATE)
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_a_fetch_to_a_path_the_spa_also_owns_reaches_the_api(authed: TestClient) -> None:
    # /health is both a console screen and the System Health snapshot route.
    service.store_snapshot(
        HealthSnapshot(
            checks=(HealthCheck("governor", CheckKind.SERVICE, HealthState.GREEN, "ok"),),
            sampled_at="2026-09-20T00:00:00+00:00",
        )
    )
    resp = authed.get("/health")
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["checks"][0]["target"] == "governor"


@pytest.mark.parametrize("mode", ["cors", "no-cors", "same-origin", "websocket", "Navigate", ""])
def test_only_the_exact_navigate_mode_gets_the_shell(anonymous: TestClient, mode: str) -> None:
    # Browsers send the lowercase token; anything else is a data call.
    resp = anonymous.get("/capabilities", headers={"Sec-Fetch-Mode": mode})
    assert resp.status_code == 401
    assert resp.json() == {"detail": "missing or invalid bearer token"}


# ── the layer never opens the API ────────────────────────────────────────────


def test_a_data_call_without_a_token_is_still_refused(anonymous: TestClient) -> None:
    resp = anonymous.get("/capabilities")
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def test_a_forged_navigate_header_on_another_method_still_meets_auth(
    anonymous: TestClient, method: str
) -> None:
    resp = anonymous.request(method, "/capabilities", headers=_NAVIGATE)
    assert resp.status_code == 401
    assert _INDEX_HTML not in resp.text


def test_a_forged_navigate_post_with_a_token_reaches_the_router(authed: TestClient) -> None:
    # 405 is the router answering: /capabilities exists and is GET-only.
    resp = authed.post("/capabilities", headers=_NAVIGATE)
    assert resp.status_code == 405


# ── /assets ──────────────────────────────────────────────────────────────────


def test_an_asset_is_served_without_a_token_and_cached_forever(anonymous: TestClient) -> None:
    resp = anonymous.get("/assets/index-abc123.js")
    assert resp.status_code == 200
    assert resp.text == _ASSET_JS
    assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert resp.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    ("name", "content_type"),
    [("index-abc123.js", "text/javascript"), ("index-abc123.css", "text/css")],
)
def test_assets_carry_the_type_a_browser_demands(
    anonymous: TestClient, name: str, content_type: str
) -> None:
    # A module script with the wrong MIME type is refused outright, and under
    # nosniff so is a stylesheet: a wrong type here is a blank console.
    resp = anonymous.get(f"/assets/{name}")
    assert resp.headers["content-type"].startswith(content_type)


def test_an_asset_opened_in_a_tab_is_the_asset_not_the_shell(anonymous: TestClient) -> None:
    resp = anonymous.get("/assets/index-abc123.js", headers=_NAVIGATE)
    assert resp.text == _ASSET_JS


def test_a_missing_asset_is_404_not_the_shell(anonymous: TestClient) -> None:
    # The shell here would be parsed as JavaScript and fail somewhere confusing.
    resp = anonymous.get("/assets/gone-000000.js", headers=_NAVIGATE)
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Not Found"}


def test_the_assets_directory_itself_is_not_a_file(anonymous: TestClient) -> None:
    assert anonymous.get("/assets/").status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/assets/../../outside.txt",
        "/assets/%2e%2e/%2e%2e/outside.txt",
        "/assets/..%2f..%2foutside.txt",
        "/assets/%2e%2e/index.html",
    ],
)
def test_a_path_cannot_climb_out_of_the_assets_directory(anonymous: TestClient, path: str) -> None:
    resp = anonymous.get(path)
    assert resp.status_code in {401, 404}
    assert _OUTSIDE not in resp.text
    assert _INDEX_HTML not in resp.text


def test_asset_lookup_rejects_traversal_whatever_the_client_normalised(dist: Path) -> None:
    # HTTP clients collapse ``..`` before sending, so the guard is also pinned
    # directly, on the string a hostile client could put on the wire.
    assets_root = (dist / "assets").resolve()
    assert static_ui._asset_file(assets_root, "/assets/../../outside.txt") is None
    assert static_ui._asset_file(assets_root, "/assets/../index.html") is None
    assert static_ui._asset_file(assets_root, "/assets//etc/passwd") is None
    found = static_ui._asset_file(assets_root, "/assets/index-abc123.js")
    assert found == assets_root / "index-abc123.js"


def test_a_symlink_out_of_the_build_is_not_followed(dist: Path, anonymous: TestClient) -> None:
    (dist / "assets" / "leak.txt").symlink_to(dist.parent / "outside.txt")
    resp = anonymous.get("/assets/leak.txt")
    assert resp.status_code == 404
    assert _OUTSIDE not in resp.text


# ── no build: the app is unchanged ───────────────────────────────────────────


def test_without_a_build_a_navigation_meets_auth_like_any_request(no_build: TestClient) -> None:
    resp = no_build.get("/capabilities", headers=_NAVIGATE)
    assert resp.status_code == 401
    assert resp.json() == {"detail": "missing or invalid bearer token"}


def test_without_a_build_assets_are_not_a_route(no_build: TestClient) -> None:
    assert no_build.get("/assets/index-abc123.js").status_code == 401


def test_without_a_build_no_middleware_is_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(static_ui.WEBUI_DIST_ENV, str(tmp_path / "absent"))
    assert _middleware_names(_app()) == [
        "_ingress_log",
        "_host_guard",
        "_bearer_auth",
        "_control_writes_guard",
        "_upload_size_guard",
    ]


def test_a_directory_without_an_index_is_not_a_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "assets").mkdir()
    monkeypatch.setenv(static_ui.WEBUI_DIST_ENV, str(tmp_path))
    assert static_ui.resolve_webui_dist() is None
    assert static_ui.install_static_ui(FastAPI()) is False


@pytest.mark.parametrize("value", [None, "", "   "])
def test_an_unset_or_blank_variable_means_the_image_then_the_package(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(static_ui.WEBUI_DIST_ENV, raising=False)
    else:
        monkeypatch.setenv(static_ui.WEBUI_DIST_ENV, value)
    assert static_ui.webui_dist_candidates() == [
        Path("/app/webui/dist"),
        static_ui.PACKAGED_WEBUI_DIST,
    ]


# ── where the build is found (OSS plan R6) ───────────────────────────────────


def _build_at(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "index.html").write_text(_INDEX_HTML, encoding="utf-8")
    return path


@pytest.fixture
def places(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """The three places, all empty and all under tmp_path: no test reads the real ones."""
    found = {
        "env": tmp_path / "env",
        "image": tmp_path / "image",
        "packaged": tmp_path / "packaged",
    }
    monkeypatch.delenv(static_ui.WEBUI_DIST_ENV, raising=False)
    monkeypatch.setattr(static_ui, "DEFAULT_WEBUI_DIST", str(found["image"]))
    monkeypatch.setattr(static_ui, "PACKAGED_WEBUI_DIST", found["packaged"])
    return found


def test_the_variable_wins_over_the_image_and_the_package(
    places: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("env", "image", "packaged"):
        _build_at(places[name])
    monkeypatch.setenv(static_ui.WEBUI_DIST_ENV, str(places["env"]))
    assert static_ui.resolve_webui_dist() == places["env"].resolve()


def test_the_image_wins_over_the_package(places: dict[str, Path]) -> None:
    _build_at(places["image"])
    _build_at(places["packaged"])
    assert static_ui.resolve_webui_dist() == places["image"].resolve()


def test_the_packaged_build_serves_when_nothing_else_does(places: dict[str, Path]) -> None:
    _build_at(places["packaged"])
    assert static_ui.resolve_webui_dist() == places["packaged"].resolve()


def test_an_image_directory_without_an_index_falls_through_to_the_package(
    places: dict[str, Path],
) -> None:
    (places["image"] / "assets").mkdir(parents=True)
    _build_at(places["packaged"])
    assert static_ui.resolve_webui_dist() == places["packaged"].resolve()


def test_no_build_anywhere_is_none(places: dict[str, Path]) -> None:
    assert static_ui.resolve_webui_dist() is None
    assert static_ui.install_static_ui(FastAPI()) is False


def test_an_explicit_variable_is_never_second_guessed(
    places: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The operator named a directory: no build there means no UI, not some other build.
    _build_at(places["image"])
    _build_at(places["packaged"])
    monkeypatch.setenv(static_ui.WEBUI_DIST_ENV, str(places["env"]))
    assert static_ui.resolve_webui_dist() is None


def test_the_packaged_build_is_found_beside_the_module_not_from_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    module_dir = Path(static_ui.__file__).resolve().parent
    assert static_ui.PACKAGED_WEBUI_DIST == module_dir / "webui_dist"
    assert static_ui.PACKAGED_WEBUI_DIST.is_absolute()


def test_the_packaged_build_is_served_end_to_end(places: dict[str, Path]) -> None:
    _build_at(places["packaged"])
    with TestClient(_app()) as client:
        response = client.get("/health", headers=_NAVIGATE)
    assert response.status_code == 200
    assert response.text == _INDEX_HTML


# ── middleware order ─────────────────────────────────────────────────────────


def _middleware_names(app: FastAPI) -> list[str]:
    """Outermost first: Starlette puts the last-registered middleware at the front."""
    return [m.kwargs["dispatch"].__name__ for m in app.user_middleware]


def test_the_static_layer_sits_between_the_ingress_log_and_auth(dist: Path) -> None:
    # ingress log -> static UI -> Host guard -> bearer auth -> write guard: the shell
    # loads without a token, and its requests are still in the ingress trail. The Host
    # guard is installed with bearer auth and runs right before it. The upload size guard
    # runs last, after auth and before any route parses a body.
    assert _middleware_names(_app()) == [
        "_ingress_log",
        "_static_ui",
        "_host_guard",
        "_bearer_auth",
        "_control_writes_guard",
        "_upload_size_guard",
    ]


# ── root-level build output (track 2b PR 8) ────────────────────────────────
#
# The manifest, the service worker, the icons and the offline page have to be
# served from the origin root. None is a navigation and none is under
# /assets/, so before this rule every one of them fell through to the API and
# came back 404 — which would have been a silent "Add to Home Screen does not
# work", and no Web Push in PR 9.


def test_the_manifest_is_served_with_its_own_media_type(anonymous: TestClient) -> None:
    resp = anonymous.get("/manifest.webmanifest")
    assert resp.status_code == 200
    # A manifest served as JSON or octet-stream is ignored by some browsers.
    assert resp.headers["content-type"].startswith("application/manifest+json")


def test_the_service_worker_is_served_from_the_root(anonymous: TestClient) -> None:
    """Scope is the whole point.

    A worker only controls paths at or below where it is served, so one served
    from /assets/ could never handle a navigation to /chat — and on iOS that
    means no Web Push at all.
    """
    resp = anonymous.get("/sw.js")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/javascript")


def test_the_service_worker_is_never_cached(anonymous: TestClient) -> None:
    # Unhashed name: a cached worker would outlive the deploy that replaced it,
    # and PR 9 puts the push handler in here.
    assert anonymous.get("/sw.js").headers["cache-control"] == "no-cache"
    assert anonymous.get("/manifest.webmanifest").headers["cache-control"] == "no-cache"


def test_root_files_need_no_credential(anonymous: TestClient) -> None:
    # The browser fetches the manifest and the worker before any pairing, and
    # they are public build output carrying no data.
    for path in ("/manifest.webmanifest", "/sw.js", "/offline.html", "/icon-192.png"):
        assert anonymous.get(path).status_code == 200, path


def test_the_allowlist_is_exactly_the_allowlist(anonymous: TestClient) -> None:
    """Adding a file to webui/public must not silently add a route.

    The build directory holds whatever `public/` held; only the named files are
    reachable, so a stray note or a source map dropped in there stays private.
    """
    assert anonymous.get("/index.html").status_code != 200
    assert anonymous.get("/outside.txt").status_code != 200


def test_a_root_file_absent_from_the_build_is_not_invented(anonymous: TestClient) -> None:
    # icon-512.png is on the allowlist but not in this fixture's build.
    resp = anonymous.get("/icon-512.png")
    assert resp.status_code != 200


def test_the_allowlist_cannot_be_walked_out_of(anonymous: TestClient) -> None:
    # The names carry no slash, so there is no traversal to attempt — but a
    # future edit that added one should fail here rather than in production.
    for name in static_ui.ROOT_FILES:
        assert "/" not in name and ".." not in name


def test_root_files_are_untouched_when_there_is_no_build(no_build: TestClient) -> None:
    # Same promise the rest of this module makes: no build, no behaviour.
    assert no_build.get("/manifest.webmanifest").status_code != 200


def test_a_refused_host_is_not_served_the_shell(dist: Path) -> None:
    """DNS rebinding (#671 leftover): an unknown name gets the guard's 400, not the shell."""
    with TestClient(_app(), base_url="http://evil.example") as client:
        page = client.get("/", headers=_NAVIGATE)
        asset = client.get("/assets/index-abc123.js")
    assert page.status_code == 400 and _INDEX_HTML not in page.text
    assert asset.status_code == 400 and _ASSET_JS not in asset.text


def test_an_allowed_host_still_gets_the_shell(anonymous: TestClient) -> None:
    assert _INDEX_HTML in anonymous.get("/", headers=_NAVIGATE).text
