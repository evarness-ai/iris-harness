"""``iris serve`` -- run the IRIS API (``iris_harness.server.iris_api.main:app``).

The scheduled jobs (the email sweep and judge) and the web console run inside the IRIS
API, so an install needs one command to start it. This is that command: uvicorn in the
foreground, on the host and port ``scripts/start_iris.sh`` binds it to
(``IRIS_API_HOST``, default loopback; ``IRIS_API_PORT``, default 8003).

What the API enforces (``iris_harness.foundation.auth``): every route except the
``/healthz`` probe needs ``Authorization: Bearer $IRIS_AUTH_SECRET`` or a paired
device's token, and with the secret unset it refuses them all (503). It serves plain
HTTP, though, so off loopback the token and the answers cross the network unencrypted.
That is what the non-loopback warning says.
"""

from __future__ import annotations

import ipaddress
from typing import Annotated

import typer
from rich.markup import escape

from iris_harness.foundation.console import console

APP = "iris_harness.server.iris_api.main:app"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8003
HOST_ENV = "IRIS_API_HOST"
PORT_ENV = "IRIS_API_PORT"


def is_loopback(host: str) -> bool:
    """Whether ``host`` only accepts connections from this machine."""
    name = host.strip().strip("[]").lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def exposure_warning(host: str, port: int) -> str | None:
    """The warning for a bind other machines can reach, or ``None`` on loopback."""
    if is_loopback(host):
        return None
    return (
        f"The IRIS API will listen on {host}:{port}, which other machines can reach. "
        "Every route except /healthz needs the IRIS_AUTH_SECRET bearer token or a paired "
        "device's token, but the API serves plain HTTP: the token and your data cross the "
        "network unencrypted, and anyone who reaches the port can try the secret. Bind to "
        "127.0.0.1 unless the network is private or a TLS proxy fronts the API. Clients "
        "that call it by a DNS name need that name in IRIS_PUBLIC_URL or IRIS_ALLOWED_HOSTS."
    )


def cmd_serve(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            envvar=HOST_ENV,
            show_envvar=False,
            help=f"Address to bind (env {HOST_ENV}). Anything but loopback is reachable "
            "from other machines, and the command warns.",
        ),
    ] = DEFAULT_HOST,
    port: Annotated[
        int,
        typer.Option(
            "--port",
            envvar=PORT_ENV,
            show_envvar=False,
            min=1,
            max=65535,
            help=f"Port to listen on (env {PORT_ENV}).",
        ),
    ] = DEFAULT_PORT,
) -> None:
    """Run the IRIS API in the foreground: the scheduled jobs and the web console."""
    import uvicorn

    from iris_harness.foundation.auth import expected_secret

    warning = exposure_warning(host, port)
    if warning:
        console.print(f"[bold yellow]warning:[/bold yellow] {escape(warning)}")
    if expected_secret() is None:
        console.print(
            "[bold yellow]warning:[/bold yellow] IRIS_AUTH_SECRET is not set, so the API "
            "refuses every request except /healthz (503). Set it to a long random value "
            "before starting.",
        )
    console.print(f"  [dim]IRIS API on {escape(host)}:{port} -- Ctrl-C to stop[/dim]")
    uvicorn.run(APP, host=host, port=port)
