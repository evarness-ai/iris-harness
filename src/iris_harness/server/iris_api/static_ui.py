"""Serving the built web console from the IRIS API.

The console is a single-page app. In development Vite serves it and proxies the
data calls; a deployment has no Vite, so the API serves the production build
(``webui/dist``) itself. This module is that serving layer and nothing more: it
knows about files on disk, not about the runtime, auth or any route.

Three rules. The first two are the ones ``webui/vite.config.ts`` applies in
development:

- ``GET /assets/*`` is a file from the build. Vite content-hashes every name in
  there, so the response is cacheable forever.
- A ``GET`` carrying ``Sec-Fetch-Mode: navigate`` is a top-level browser
  navigation (address bar, deep link, refresh) and gets ``index.html``, whatever
  the path. Several SPA routes share a name with a data route (``/health``,
  ``/routines``, ...), and browsers send that header ONLY on navigations, never on
  ``fetch``/XHR. ``curl``, the CLI and the channel gateway never send it at all, so
  every data call reaches the API exactly as before.

- A short allowlist of files that must live at the ORIGIN ROOT: the manifest,
  the service worker, the icons and the offline page (track 2b PR 8). None is a
  navigation and none is under ``/assets/``, so without this rule every one of
  them would fall through to the API and 404. A service worker in particular
  only controls the paths below where it is served, so ``/sw.js`` has to be at
  the root to cover the whole app. The list is explicit rather than "any file in
  the build" so that adding a file to ``webui/public`` is never accidentally
  adding a route.

The shell and its assets are public build output and carry no data, so they are
served ahead of the bearer check (but never for a Host the server refuses); every
data call still needs a credential.
Nothing here can widen that: the only responses this layer produces are files
from the build directory.

Where the build is looked for, first match wins (``resolve_webui_dist``):

1. ``IRIS_WEBUI_DIST`` when it is set. An explicit setting is the operator's answer,
   so a directory there without an ``index.html`` means "no UI", never a fallback.
2. ``/app/webui/dist``, where the server image puts the build.
3. The copy packaged inside this package (``webui_dist/`` beside this module): the
   release wheel carries it (OSS plan R6, ``scripts/bundle_webui.py``), so a
   ``pip install`` serves the console with no checkout and no Node. It is found from
   this file's location, never from the working directory.

Off by default in effect: with no build on disk nothing is installed and the app
behaves exactly as it did without this module.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iris_harness.server.auth import host_header_ok, routed_path

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

WEBUI_DIST_ENV = "IRIS_WEBUI_DIST"
# Where the server image puts the build (mobile-cloud-ui plan, track 1 PR 4).
DEFAULT_WEBUI_DIST = "/app/webui/dist"
# The build the release wheel carries (OSS plan R6). Gitignored: only the release
# build (scripts/bundle_webui.py) puts it here, and pyproject's `include` ships it.
PACKAGED_WEBUI_DIST = Path(__file__).resolve().parent / "webui_dist"

_INDEX_FILE = "index.html"
_ASSETS_DIR = "assets"
_ASSETS_PREFIX = f"/{_ASSETS_DIR}/"

# Asset names are content-hashed, so a name never changes meaning. The shell is
# the one file that does, and it is what points at the current asset names: it
# must be revalidated on every load or a browser keeps a build that is gone.
_ASSET_CACHE_CONTROL = "public, max-age=31536000, immutable"
_INDEX_CACHE_CONTROL = "no-cache"
_NOSNIFF = {"X-Content-Type-Options": "nosniff"}

# Root-level build output (track 2b PR 8). Unhashed names, so like the shell
# they must be revalidated: a stale service worker would outlive the deploy
# that replaced it, and a stale manifest would keep an old name on the Home
# Screen. The icons are content-stable but cheap, and correctness beats a few
# saved bytes on a once-per-install fetch.
_ROOT_FILES: dict[str, str] = {
    "manifest.webmanifest": "application/manifest+json",
    "sw.js": "text/javascript",
    "offline.html": "text/html; charset=utf-8",
    "apple-touch-icon.png": "image/png",
    "icon-192.png": "image/png",
    "icon-512.png": "image/png",
    "icon-512-maskable.png": "image/png",
}


def webui_dist_candidates() -> list[Path]:
    """The directories a build is looked for in, in order (see the module docstring).

    ``IRIS_WEBUI_DIST``, when set, is the only candidate; otherwise the image location
    and then the packaged copy.
    """
    raw = os.environ.get(WEBUI_DIST_ENV, "").strip()
    if raw:
        return [Path(raw).expanduser()]
    return [Path(DEFAULT_WEBUI_DIST), PACKAGED_WEBUI_DIST]


def resolve_webui_dist() -> Path | None:
    """Return the build directory to serve, or ``None`` when there is no build.

    A directory without an ``index.html`` is not a build, so it counts as absent.
    """
    for dist in webui_dist_candidates():
        if (dist / _INDEX_FILE).is_file():
            return dist.resolve()
    return None


def _asset_file(assets_root: Path, url_path: str) -> Path | None:
    """Map ``/assets/<name>`` to a file inside ``assets_root``, or ``None``.

    The resolved path must stay under the assets directory, which rejects
    ``..`` segments and symlinks that point outside the build alike.
    """
    candidate = (assets_root / url_path[len(_ASSETS_PREFIX) :]).resolve()
    if not candidate.is_relative_to(assets_root) or not candidate.is_file():
        return None
    return candidate


def install_static_ui(app: FastAPI) -> bool:
    """Register the web-console middleware; return whether it was installed.

    Starlette runs the middleware registered LAST first. Call this after
    ``install_bearer_auth`` and before the ingress-log middleware, so requests
    flow ingress-log -> static UI -> bearer-auth: the shell loads without a
    token, and every request still lands in the ingress trail.

    The build directory is resolved once, here. No build means nothing is
    registered at all, so the app is untouched.
    """
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse

    dist = resolve_webui_dist()
    if dist is None:
        looked = ", ".join(str(p) for p in webui_dist_candidates())
        logger.info("web UI: no build at %s; not serving it", looked)
        return False

    index_file = dist / _INDEX_FILE
    assets_root = dist / _ASSETS_DIR

    @app.middleware("http")
    async def _static_ui(request: Request, call_next: Any) -> Any:
        # A refused Host (DNS rebinding, #671) is not served the shell: pass it on to
        # the bearer middleware's Host guard, which answers 400 and logs the name.
        if request.method != "GET" or not host_header_ok(request.headers.get("host")):
            return await call_next(request)
        path = routed_path(request)
        if path.startswith(_ASSETS_PREFIX):
            asset = _asset_file(assets_root, path)
            if asset is None:
                return JSONResponse(status_code=404, content={"detail": "Not Found"})
            return FileResponse(asset, headers={"Cache-Control": _ASSET_CACHE_CONTROL, **_NOSNIFF})
        root_file = _ROOT_FILES.get(path.lstrip("/"))
        if root_file is not None and "/" not in path.lstrip("/"):
            candidate = dist / path.lstrip("/")
            if candidate.is_file():
                return FileResponse(
                    candidate,
                    media_type=root_file,
                    headers={"Cache-Control": _INDEX_CACHE_CONTROL, **_NOSNIFF},
                )
        if request.headers.get("sec-fetch-mode") == "navigate":
            return FileResponse(
                index_file, headers={"Cache-Control": _INDEX_CACHE_CONTROL, **_NOSNIFF}
            )
        return await call_next(request)

    logger.info("web UI: serving the build at %s", dist)
    return True


__all__ = [
    "DEFAULT_WEBUI_DIST",
    "PACKAGED_WEBUI_DIST",
    "ROOT_FILES",
    "WEBUI_DIST_ENV",
    "install_static_ui",
    "resolve_webui_dist",
    "webui_dist_candidates",
]

#: Public alias — the deploy test asserts the build ships every one of these.
ROOT_FILES = _ROOT_FILES
