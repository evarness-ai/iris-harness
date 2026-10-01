"""Capture the README's web-console screenshots from a real demo run (issue #20).

Real data only: every pixel comes from ``iris email demo``'s synthetic mailbox and the
audit rows, sessions and traces that run and the API's first-chat welcome write. Nothing
is mocked. The steps:

1. ``iris email demo`` into a fresh demo home in a temporary directory (never
   ``~/.iris-demo``, never ``~/.iris``);
2. the Governor and the IRIS API on that home (the services ``docker compose up``
   runs, minus Ollama), in the demo's own environment (the scripted fake model, the
   demo's vault key, warm-up off) under the ``email`` profile, with a throwaway
   ``IRIS_AUTH_SECRET``; the API serves the built console from ``webui/dist``;
3. ``POST /chat/welcome``, so Sessions and Call trace hold the welcome turn;
4. a pairing code from ``POST /api/v1/devices/pair/start`` (what ``iris device pair``
   does), which the browser claims, as the console's Pair screen does;
5. ``webui/screenshots/readme.spec.ts`` (Playwright, Chromium) captures Governance, the
   welcome turn's Call trace, Inbox and Setup at 1280x800 in light and dark;
6. each PNG is re-encoded as WebP into ``docs/assets/screenshots/``.

Not part of any test suite; run it on demand, from the repository root, after
``npm ci`` in ``webui/``::

    PYTHONPATH=src poetry run python scripts/readme_screenshots.py
    PYTHONPATH=src poetry run python scripts/readme_screenshots.py --skip-build --keep

``--skip-build`` reuses an existing ``webui/dist``; ``--keep`` leaves the temporary demo
home and the raw PNGs behind and prints where they are.

The API process gets the demo's environment, not the caller's: the caller's ``IRIS_*``
settings and credential-looking variables are dropped (as ``iris email demo`` drops
them), and every key of a repository-root ``.env`` is set empty, because the API loads
that file at import and would otherwise fill in the owner's flags and keys.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WEBUI = ROOT / "webui"
OUT = ROOT / "docs" / "assets" / "screenshots"
SPEC_CONFIG = "playwright.screenshots.config.ts"
WEBP_QUALITY = 82
MAX_BYTES = 300_000


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(base: str, path: str, secret: str, body: Any = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(  # noqa: S310 -- our own loopback API
        f"{base}{path}",
        data=data,
        method="GET" if body is None else "POST",
        headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read())


def _wait_healthy(base: str, proc: subprocess.Popen[bytes], timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{base} exited with {proc.returncode} before it was up")
        try:
            with urllib.request.urlopen(f"{base}/healthz", timeout=2):  # noqa: S310
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"{base} was not healthy after {timeout:.0f}s")


def _service_environment(home: Path, port: int, governor_port: int, secret: str) -> dict[str, str]:
    """The demo's environment for the services, with the repository ``.env`` neutralised."""
    from iris_personal.plugins.email_workflows.demo.home import demo_environment

    env = demo_environment(home, os.environ)
    dotenv = ROOT / ".env"
    if dotenv.is_file():
        from dotenv import dotenv_values

        for key in dotenv_values(dotenv):
            env.setdefault(key, "")
    env.update(
        {
            "IRIS_AUTH_SECRET": secret,
            "IRIS_WEBUI_DIST": str(WEBUI / "dist"),
            "IRIS_API_HOST": "127.0.0.1",
            "IRIS_API_PORT": str(port),
            "IRIS_GOVERNOR_PORT": str(governor_port),
            "IRIS_GOVERNOR_AUDIT_DB": str(home / "data" / "governor-audit.db"),
            # The demo run uses `minimal`; the console shows the email assistant.
            "IRIS_PROFILE": "email",
        }
    )
    return env


def _to_webp(raw_dir: Path, out_dir: Path) -> list[tuple[Path, int]]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, int]] = []
    for png in sorted(raw_dir.glob("*.png")):
        target = out_dir / f"{png.stem}.webp"
        with Image.open(png) as image:
            image.convert("RGB").save(target, "WEBP", quality=WEBP_QUALITY, method=6)
        size = target.stat().st_size
        if size > MAX_BYTES:
            raise RuntimeError(f"{target.name} is {size} bytes, over the {MAX_BYTES} budget")
        written.append((target, size))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="where the WebP files go")
    parser.add_argument("--skip-build", action="store_true", help="reuse webui/dist")
    parser.add_argument("--keep", action="store_true", help="keep the temporary demo home")
    args = parser.parse_args(argv)

    if not (WEBUI / "node_modules").is_dir():
        print("readme_screenshots: run `npm ci` in webui/ first", file=sys.stderr)
        return 2
    if not args.skip_build:
        subprocess.run(["npm", "run", "build"], cwd=WEBUI, check=True)  # noqa: S607
    if not (WEBUI / "dist" / "index.html").is_file():
        print("readme_screenshots: webui/dist has no build", file=sys.stderr)
        return 2

    # A short, neutral path: Governance prints the audit ledger's location.
    tmp_root = "/tmp" if Path("/tmp").is_dir() else None  # noqa: S108
    scratch = Path(tempfile.mkdtemp(prefix="iris-readme-", dir=tmp_root))
    home = scratch / "demo-home"
    raw = scratch / "raw"
    raw.mkdir()
    port, governor_port = _free_port(), _free_port()
    base = f"http://127.0.0.1:{port}"
    secret = secrets.token_hex(32)
    services: list[subprocess.Popen[bytes]] = []
    try:
        # 1. The demo run, in its own child process and environment.
        demo_env = {k: v for k, v in os.environ.items() if k != "IRIS_DEMO_HOME"}
        subprocess.run(  # noqa: S603 -- this interpreter, fixed arguments
            [sys.executable, "-m", "iris_harness.main", "email", "demo", "--home", str(home)],
            env=demo_env,
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        # 2. The Governor and the API on the demo home.
        env = _service_environment(home, port, governor_port, secret)
        governor_app = "iris_harness.server.governor.main:app"
        for name, command, url in (
            (
                "governor",
                [
                    "-m",
                    "uvicorn",
                    governor_app,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(governor_port),
                ],
                f"http://127.0.0.1:{governor_port}",
            ),
            ("iris_api", ["-m", "iris_harness.main", "serve"], base),
        ):
            log = (scratch / f"{name}.log").open("wb")
            services.append(
                subprocess.Popen(  # noqa: S603
                    [sys.executable, *command], env=env, cwd=home, stdout=log, stderr=log
                )
            )
            _wait_healthy(url, services[-1])

        # 3. The first-chat welcome: the turn Sessions and Call trace show.
        welcome = _request(base, "/chat/welcome", secret, {"channel": "console"})
        session_id = welcome["session_id"]
        traces = _request(base, "/api/traces", secret)
        mine = [t for t in traces if t.get("session_id") == session_id]
        if not mine:
            raise RuntimeError(f"the welcome turn (session {session_id}) left no trace")
        trace_id = mine[0]["trace_id"]

        # 4. A one-time pairing code for the browser.
        code = _request(base, "/api/v1/devices/pair/start", secret, {"scope": "control"})["code"]

        # 5. The captures.
        subprocess.run(  # noqa: S603 -- fixed arguments
            ["npx", "playwright", "test", "-c", SPEC_CONFIG],  # noqa: S607
            cwd=WEBUI,
            check=True,
            env={
                **os.environ,
                "IRIS_SHOTS_BASE_URL": base,
                "IRIS_SHOTS_PAIR_CODE": code,
                "IRIS_SHOTS_TRACE_ID": trace_id,
                "IRIS_SHOTS_OUT": str(raw),
            },
        )

        # 6. WebP, under the size budget.
        for path, size in _to_webp(raw, args.out):
            print(f"{path}  {size // 1024} KB")
    finally:
        for proc in reversed(services):
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        if args.keep:
            print(f"kept: {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
