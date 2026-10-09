# Install

IRIS installs as one command-line tool, `iris`, with the email assistant included.
These three pages take you from nothing to your own inbox: install and check the
machine (this page), [try the demo](demo.md), then
[connect your mailbox](connect-your-mailbox.md). Every command on them runs in CI
on a fresh machine, on every change.

## What you need

- **macOS on Apple Silicon, or Linux** (x86_64 or arm64). On Windows, use WSL2.
- **16 GB of RAM** for real use. 8 GB runs the demo only: it uses a scripted model.
- **Python 3.12 or 3.13.** `uv` fetches one for you if you have neither.
- **[Ollama](https://ollama.com/download)** for the local model. The demo does not
  need it, so you can install it later.

## Install the tool

With [uv](https://docs.astral.sh/uv/):

<!-- ci: install -->
```bash
uv tool install "iris-harness[email]"
```

Or with [pipx](https://pipx.pypa.io/):

<!-- ci: install -->
```bash
pipx install "iris-harness[email]"
```

The `[email]` extra adds the Gmail client libraries. Without it IRIS is still the email
assistant over IMAP. Either way you get one command, `iris`, in an isolated
environment of its own. To upgrade later, run `uv tool upgrade iris-harness` (or
`pipx upgrade iris-harness`).

## Check the machine: `iris doctor`

<!-- ci: exit 0,1 -->
```bash
iris doctor
```

`iris doctor` checks what IRIS needs here: the Python version, the platform, memory,
disk space, a writable `IRIS_HOME` (default `~/.iris`), Ollama, the starter model and
the vault master key. Each check prints a one-line fix when it fails. The report ends
with a verdict, which is also the exit code:

| Verdict | Exit code | What it means |
|---|---|---|
| Ready. | 0 | Everything real use needs is in place. |
| Ready for the demo only. | 1 | The demo runs; real use still needs Ollama, the starter model, a vault master key, or 16 GB of RAM. |
| Not ready. | 2 | Fix the failing checks first (the Python version, native Windows, under 8 GB, or an unwritable `IRIS_HOME`). |

Right after installing, "demo only" is normal. The next page runs the demo.

### The safe fixes: `iris doctor --fix`

<!-- ci: skip downloads the starter model (about 4.7 GB) from Ollama -->
```bash
iris doctor --fix
```

`--fix` applies only two kinds of fix, and asks before each: it pulls a missing starter
model through Ollama, and it creates a vault master key if there is none. It never
overwrites a stored key, since your secrets are encrypted under it. Add `--yes` to
skip the questions. `--json` prints the report as JSON for scripts, and applies no
fixes.

**The vault master key.** Every governed tool call needs it, so without it IRIS
refuses them. On macOS, `--fix` stores the key in the Keychain. On Linux with no
keyring (WSL2, a headless server), it prints an `export IRIS_VAULT_MASTER_KEY=...`
line instead. Put that line in your shell profile, and keep a copy somewhere safe,
because the vault cannot be read without the key.

**The email judge needs its own model, separately.** `--fix` pulls only the general
starter model above; the email assistant's judge is pinned to a specific, smaller one
(`config/llm_tiers.yaml`'s `email_judge` tier) that `iris doctor` does not check or
pull. Before [connecting your mailbox](connect-your-mailbox.md), also run:

<!-- ci: skip downloads the email judge's model (about 2.5 GB) from Ollama -->
```bash
ollama pull qwen3.5:4b-q4_K_M
```

Without it, `iris email setup`'s "Classify" step blocks with "no local email_judge
model tier is configured" — the demo is unaffected, since it runs on a scripted model.

## Guided setup: `iris setup`

The steps above can also run as one guided wizard: `iris setup` walks through the
preflight, the auth secret, and the optional add-ons (the background services, a
Telegram pairing, email) in order, resuming where you left off if you stop partway.
`iris setup --status` shows progress; `iris setup --reset` starts over.

## Next

[Try the demo](demo.md): a synthetic mailbox, end to end, in a few seconds.
