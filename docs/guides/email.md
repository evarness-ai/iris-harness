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
`/healthz` probe.

```bash
export IRIS_AUTH_SECRET="$(openssl rand -hex 32)"
iris serve
```

`iris serve` runs the API in the foreground on `127.0.0.1:8003`; `--port` (or
`IRIS_API_PORT`) moves it. From a clone of the repository, the same command is
`poetry run iris serve`. [Run with Docker](docker.md) keeps it up as a service.

The API listens on `127.0.0.1` only unless you pass `--host` (or set `IRIS_API_HOST`).
It serves plain HTTP, so any other address puts the shared secret and your mail on the
network unencrypted; `iris serve` warns when you do.

Open the console at `http://localhost:8003` and pair the browser: in another shell with
the same `IRIS_AUTH_SECRET`, `iris device pair` prints a one-time code to type there.

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
