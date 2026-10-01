"""CLI for MCP server signing (``iris mcp keygen|sign|verify``) — sub-phase 6b.3.

Operator tooling to produce + manage the signatures the bridge gate verifies.
These commands are attached to the existing ``iris mcp`` group (which already
hosts ``mcp serve``) via :func:`register_signing_commands`; ``mcp_signing_app``
is a standalone app used only for isolated testing.

- ``iris mcp keygen --id <key_id>``  Ed25519 keypair; private key to a 0600 file
  the agent can't read, public key into the trust store.
- ``iris mcp sign <server> --key <key_id>``  write signature/signed_by into
  mcp-servers.yaml.
- ``iris mcp verify [server]``  print verdicts; non-zero exit if any unverified.

See ``docs/architecture/mcp-server-signing.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, cast

import typer
import yaml
from rich.console import Console
from rich.table import Table

from iris_harness.foundation.paths import config_path

console = Console()

_DEFAULT_SERVERS = config_path("coding-agent", "mcp-servers.yaml")
_DEFAULT_TRUST = config_path("governance", "mcp-trust.yaml")


def _keys_dir() -> Path:
    base = os.environ.get("IRIS_MCP_KEYS_DIR")
    return Path(base) if base else Path.home() / ".config" / "iris" / "mcp-keys"


def cmd_keygen(
    key_id: Annotated[str, typer.Option("--id", help="Trust-store key id.")],
    trust_store: Annotated[Path, typer.Option("--trust-store", help="Trust store path.")] = (
        _DEFAULT_TRUST
    ),
    private_key_out: Annotated[
        Path | None, typer.Option("--private-key-out", help="Where to write the private key.")
    ] = None,
    comment: Annotated[str, typer.Option("--comment", help="Trust-store comment.")] = "",
) -> None:
    """Generate an Ed25519 keypair; add the public key to the trust store."""
    from iris_harness.kernel.governance.mcp_signing import generate_keypair

    private_b64, public_b64 = generate_keypair()
    key_path = private_key_out or (_keys_dir() / f"{key_id}.key")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(private_b64 + "\n", encoding="utf-8")
    os.chmod(key_path, 0o600)

    doc = _load_yaml(trust_store) if trust_store.exists() else {"version": 1, "keys": []}
    keys = cast(list[dict[str, Any]], doc.get("keys") or [])
    if any(str(k.get("key_id")) == key_id for k in keys):
        console.print(f"[red]key_id {key_id!r} already exists in {trust_store}[/red]")
        raise typer.Exit(1)
    keys.append({"key_id": key_id, "public_key": public_b64, "comment": comment})
    doc["keys"] = keys
    _dump_yaml(trust_store, doc)

    console.print(f"[green]Generated key {key_id!r}[/green]")
    console.print(f"  private key: {key_path} [yellow](keep secret; do not commit)[/yellow]")
    console.print(f"  public key added to: {trust_store}")


def cmd_sign(
    server_name: Annotated[str, typer.Argument(help="Server name in mcp-servers.yaml.")],
    key_id: Annotated[str, typer.Option("--key", help="Trust-store key id to sign with.")],
    servers: Annotated[Path, typer.Option("--servers", help="mcp-servers.yaml path.")] = (
        _DEFAULT_SERVERS
    ),
    private_key: Annotated[
        Path | None, typer.Option("--private-key", help="Private key file (default: keys dir).")
    ] = None,
) -> None:
    """Sign one server's canonical spec; write signature/signed_by into the config."""
    from iris_harness.kernel.governance.mcp_signing import sign
    from iris_harness.tools.mcp_bridge import build_server_spec, load_mcp_bridge_config

    key_path = private_key or (_keys_dir() / f"{key_id}.key")
    if not key_path.exists():
        console.print(f"[red]private key not found: {key_path}[/red]")
        raise typer.Exit(1)
    private_b64 = key_path.read_text(encoding="utf-8").strip()

    config = load_mcp_bridge_config(Path.cwd(), config_path=servers)
    server = next((s for s in config.servers if s.name == server_name), None)
    if server is None:
        console.print(f"[red]server {server_name!r} not found in {servers}[/red]")
        raise typer.Exit(1)

    signature = sign(private_b64, build_server_spec(server).canonical_bytes())

    doc = _load_yaml(servers)
    for entry in cast(list[dict[str, Any]], doc.get("servers") or []):
        if str(entry.get("name")) == server_name:
            entry["signature"] = signature
            entry["signed_by"] = key_id
            break
    _dump_yaml(servers, doc)
    console.print(f"[green]Signed {server_name!r} with {key_id!r}[/green] → {servers}")
    console.print("[yellow]note: comments in the YAML are not preserved on write[/yellow]")


def cmd_verify(
    server_name: Annotated[
        str | None, typer.Argument(help="Server to verify (default: all).")
    ] = None,
    servers: Annotated[Path, typer.Option("--servers", help="mcp-servers.yaml path.")] = (
        _DEFAULT_SERVERS
    ),
    trust_store: Annotated[Path, typer.Option("--trust-store", help="Trust store path.")] = (
        _DEFAULT_TRUST
    ),
) -> None:
    """Verify server signatures; non-zero exit if any are not verified."""
    from iris_harness.kernel.governance.mcp_signing import TrustStore, verify_server_signature
    from iris_harness.tools.mcp_bridge import build_server_spec, load_mcp_bridge_config

    config = load_mcp_bridge_config(Path.cwd(), config_path=servers)
    store = TrustStore.from_yaml(trust_store)
    targets = [s for s in config.servers if server_name is None or s.name == server_name]
    if not targets:
        console.print(f"[red]no matching server in {servers}[/red]")
        raise typer.Exit(1)

    table = Table(title="MCP signature verification")
    table.add_column("server")
    table.add_column("status")
    table.add_column("signer")
    table.add_column("reason")
    any_unverified = False
    for server in targets:
        verdict = verify_server_signature(
            spec=build_server_spec(server),
            signature=server.signature,
            signed_by=server.signed_by,
            trust_store=store,
        )
        color = "green" if verdict.is_trusted else "red"
        if not verdict.is_trusted:
            any_unverified = True
        table.add_row(
            server.name,
            f"[{color}]{verdict.status}[/{color}]",
            verdict.key_id or "-",
            verdict.reason,
        )
    console.print(table)
    if any_unverified:
        raise typer.Exit(1)


def register_signing_commands(app: typer.Typer) -> None:
    """Attach keygen/sign/verify to an existing ``mcp`` Typer group."""
    app.command("keygen")(cmd_keygen)
    app.command("sign")(cmd_sign)
    app.command("verify")(cmd_verify)


#: Standalone app for isolated CLI tests (production attaches to the real group).
mcp_signing_app = typer.Typer(name="mcp", no_args_is_help=True)
register_signing_commands(mcp_signing_app)


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"{path} must decode to a mapping")
    return raw


def _dump_yaml(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
