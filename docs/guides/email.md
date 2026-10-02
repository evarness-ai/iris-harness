# The email assistant

The email assistant is three plugins on the harness: the mail providers (`gmail`,
`imap`) and `email_workflows` (the sweep, the judge, categories, the digest, setup and
the demo). An install with the email plugins runs the `email` profile with no setting
at all; `iris --dump-config` shows the profile in effect and every plugin it mounts.

Start with [Connect your mailbox](../getting-started/connect-your-mailbox.md). This
page is what comes after.

## Run the IRIS API

The scheduled jobs (the email sweep that fetches new mail, and the judge that sorts it
on the local model) run inside the IRIS API service, which also serves the web
console. Set a long random shared secret first: every API call needs it, except the
`/healthz` probe. Write it to a file once, rather than re-running the generator in
every shell -- `openssl rand -hex 32` makes a *different* value each time it runs, so
exporting its output straight in a second terminal does not reproduce the first
shell's secret, and the API answers with a 401 that gives no sign why:

```bash
openssl rand -hex 32 > .auth-secret
export IRIS_AUTH_SECRET="$(cat .auth-secret)"
iris serve
```

Use the same two commands (`export IRIS_AUTH_SECRET="$(cat .auth-secret)"`, then
whatever you're running) in every other shell that talks to this API -- `iris device
pair`, `iris status`, a second terminal open to the same install.

`iris serve` runs the API in the foreground on `127.0.0.1:8003`; `--port` (or
`IRIS_API_PORT`) moves it. From a clone of the repository, the same command is
`poetry run iris serve`. [Run with Docker](docker.md) keeps it up as a service, and an
installed wheel or a container image ships the web console already built.

**From a source checkout, build the console first.** A bare `git clone` + `poetry
install` has no console at all -- `iris serve` answers `/chat` and friends, but
`http://localhost:8003` shows nothing -- until you build it and point `iris serve` at
the result:

```bash
cd webui && npm install && npm run build && cd ..
export IRIS_WEBUI_DIST="$PWD/webui/dist"
iris serve
```

The API listens on `127.0.0.1` only unless you pass `--host` (or set `IRIS_API_HOST`).
It serves plain HTTP, so any other address puts the shared secret and your mail on the
network unencrypted; `iris serve` warns when you do.

Open the console at `http://localhost:8003` and pair the browser: in another shell with
the same `IRIS_AUTH_SECRET` (`export IRIS_AUTH_SECRET="$(cat .auth-secret)"`, not a
fresh `openssl rand`), `iris device pair` prints a one-time code to type there.

**A "Governor not reachable" banner is expected** if this is all you started: the
Governor is a *separate* HTTP frontend on port 8080 for the same governance kernel
`iris serve` already runs in-process, so nothing above needs it. Start it too, with
the same secret, only if you want the warning gone or want the full stack the README
describes:

```bash
export IRIS_AUTH_SECRET="$(cat .auth-secret)"
iris serve &   # or: poetry run iris serve &, from a source checkout
uvicorn iris_harness.server.governor.main:app --port 8080
```

## Day to day

| What | How |
|---|---|
| Ask about your mail | `iris -p "what's in my inbox?"`, or plain `iris` for the REPL, or the console's Chat |
| See how the judge sorted recent mail | `iris email judgments` (`--bucket "Needs reply"`, `--limit 20`) |
| Judge new mail now, instead of waiting for the schedule | `iris email judge` |
| Search the stored mail | `iris email search "invoice"` |
| Correct a classification | `iris email recategorize --account <id> --message-id <id> --to <category>` |
| Approve, check or take back mailbox writes | `iris email writes approve --account <id>`, `writes status`, `writes revoke` |
| Where setup stands | `iris email setup --status` |
| What is waiting for you | `iris actions`, or the console's Actions screen |
| What every call did | the console's Governance and Call trace screens; `iris audit --help` |

Each command's `--help` lists its options. Only `iris doctor`, `iris serve`,
`iris email demo`, `iris email setup` and `iris email writes approve` are part of the
[stable CLI](../reference/cli.md); the rest may change in a 0.x release.

## Your own categories

The judge's buckets (Bill, Event, Needs reply, FYI, Unsure) and the categories setup
discovers are configuration, not code. The
[custom-categories example](https://github.com/evarness-ai/iris-harness/tree/main/examples/03-custom-categories)
adds your own and shows them live in the assistant.

## Privacy, by construction

- Email is classified `personal`, so the governance kernel lets it reach local models
  only. A cloud model never sees your mail, whatever you configure for other work.
- Nothing changes in the mailbox without the account's write approval. IRIS never
  archives, marks read or deletes on its own; a move to Trash is a governed write like
  a label, and needs the same approval.
- Credentials (app passwords, OAuth tokens) live in the encrypted vault, under the
  master key `iris doctor` checks.
- Every model call, tool call and answer leaves an audit row. The
  [proof bundle](../reference/proof-bundle.md) checks these invariants offline.
