# 08 · MCP, both ways

The Model Context Protocol lets an AI client call tools a server exposes. IRIS can be
the server (Claude Desktop, an IDE or another agent calls IRIS's tools) and, on the
other side, the client of external MCP servers -- behind an allowlist and signatures.

| File | What it is |
|---|---|
| `mcp_client.py` | A minimal MCP stdio client (standard library only). |
| `weather_server.py` | A stand-in external MCP server with one tool, `forecast`. |
| `mcp-servers.yaml` | The allowlist entry for it: enabled, stdio command, which persona may call which tool. |
| `test_mcp.py` | Both halves, offline, every `iris` command in a temporary home. |

## Run it

```bash
pytest examples/08-mcp -q
```

Expected output: `8 passed` in about 32 s (16 s with `--no-cov`).

### Serve: IRIS's tools to an MCP client

First the owner lists the client, in `$IRIS_HOME/mcp-serve.yaml` (the shipped
`config/governance/mcp-serve.yaml` lists only `stdio`):

```yaml
clients:
  example:
    local: false   # true: it runs on this machine, and may receive personal results
```

then

```bash
iris mcp serve --client example --tool system_health --tool trash_email --skill system-status
```

serves tools over stdio (newline-delimited JSON-RPC). Point any MCP client at that
command -- for Claude Desktop, an `mcpServers` entry with `"command": "iris"` and those
`"args"`. What is served:

- **the registered tools** -- the ones plugins register with `api.register_tool`, each
  with its manifest declaration. `--tool` names them, whatever they do. With none, only
  the *contained* reads are served: a `read` tool whose arguments go nowhere (no
  `sends_to`), whose output no third party wrote (not `content: external`) and that
  runs no code (not `executes_code`). So `research`, `code_exec` and the email reads
  are served only when named;
- **skill reads** -- `--skill` adds a skill package's tools whose manifest route is a
  read (`system/read`); none by default. A skill cannot declare a write's confirmation
  or approval, so a skill write is not served.

Every call is a governed call, exactly like one the model makes in a chat:
`PRE_TOOL_USE` (caller policy, tool policy, approvals, the credential broker), the call,
`POST_TOOL_USE` (a withheld result comes back as an error), and the audit rows -- as the
caller `mcp:<client>`. `<client>` is the `--client` name, which must be one the owner
listed; what the client says about itself at `initialize` is logged and decides nothing.
A call that needs the owner's answer -- a destructive tool, a pinned write, a write that
asks first -- is refused: an MCP client cannot give that answer, and a call queued to
run later would run out of its sight. With no vault master key nothing is served (an
audit row cannot be keyed).

Where a result may go: a result labelled `personal` reaches only a client the owner
declared `local: true`; a `secret` one reaches no client, local or not (a local client
is still a program IRIS does not govern). One connection is one label: once a result
carried personal data, every later result of the session is withheld from a client
that is not local.

The tests drive it with `mcp_client.py`: the handshake, the tool list, two calls and an
unknown tool; the `PRE_TOOL_USE` / `POST_TOOL_USE` rows of a call, as `mcp:example`;
`trash_email` refused, with its `deny` row and no `POST_TOOL_USE`; a client the owner
did not list, refused; what is served with no `--tool`; and the refusal with no vault
key.

### Consume: an external server, allowlisted and signed

`mcp-servers.yaml` declares the external server and what may be called on it; the
governance kernel's `mcp_allowlist` check refuses a call to a server or tool that is not
listed, and `mcp_signing` checks that the entry was signed by a key in the trust store,
so an edited command (a swapped binary, a new argument) no longer runs as trusted:

```bash
iris mcp keygen --id me                  # Ed25519 key; public half into the trust store
iris mcp sign weather --key me           # signature + signer written into mcp-servers.yaml
iris mcp verify                          # verdict per server; non-zero exit if any fails
```

The test runs those three against a copy of `mcp-servers.yaml` and a temporary trust
store, then edits the signed entry's command and shows `verify` rejecting it. It also
checks `weather_server.py` answers a real MCP client.

## What the stable tier does not cover yet (L2 findings)

- **A served tool has a free-form schema.** A registered tool declares its effect but no
  argument schema, so `tools/list` offers `{"query": string}` plus any other key (the
  argument every tool reads); a skill tool keeps its own schema.
- **Consumed tools do not reach the chat loop.** The client side (`MCPBridge`, with the
  allowlist and signing checks) is internal and is used by the coding agent, which is
  not in release 1; no stable API mounts an external server's tools into the governed
  loop, and a `trust: mcp` plugin manifest is recognised but not loaded yet. The
  example therefore stops at the allowlist and the signature: what an owner configures
  before IRIS may consume a server.
- The server config lives under `config/coding-agent/mcp-servers.yaml`
  (`IRIS_MCP_SERVERS_CONFIG_PATH` overrides it), for the same reason.

## Try changing

Show that the signature covers the arguments, not only the program. In
`test_a_signed_server_verifies_and_a_changed_one_does_not`, instead of pointing the
entry at `other.py`, add an argument to it. `iris mcp sign` rewrites the file, so match
the list item as it writes it:

```python
    servers.write_text(
        servers.read_text(encoding="utf-8").replace(
            "- weather_server.py", "- weather_server.py\n  - --debug"
        ),
        encoding="utf-8",
    )
```

The test still passes: `iris mcp verify` rejects the entry ("signature does not match
server spec (config or binary changed)"). The same holds for a changed `env`. The
signature covers what runs; which persona may call which tool is the allowlist's job,
checked by `mcp_allowlist` on every call.

## Next

This is the last example. Back to the [index](../README.md), or start your own plugin
from the scaffold: [Write a plugin](../../docs/guides/write-a-plugin.md). What a plugin
may import is the [stable API](../../docs/reference/stable-api.md).
