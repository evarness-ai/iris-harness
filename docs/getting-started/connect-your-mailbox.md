# Connect your mailbox

`iris email setup` takes your own mailbox through the same steps the demo ran, one at a
time, and stops wherever it needs your decision. It is resumable: run it again and it
picks up where it stopped.

Before you start, run `iris doctor` until it says "Ready." (see [Install](install.md)):
your mail is personal data, so IRIS reads and sorts it on the local model only.

## 1. Connect an account

Setup never asks for a password or runs a login itself. Each mail provider plugin owns
its login, and setup takes over once the account is connected. With no account
connected yet, setup says how to connect one:

```bash
iris email setup
```

### IMAP with an app password

Most providers (iCloud, Yahoo, Fastmail, Gmail too) let you create an **app password**
for IMAP in your account's security settings. Then:

<!-- ci: skip needs your mailbox and its app password -->
```bash
iris auth imap login --user you@example.com
```

IRIS asks for the app password (hidden), logs in once to prove it works, and only then
stores the account in the vault. For common providers it knows the server; otherwise
pass `--host imap.example.com` (and `--port` or `--security starttls` if your server
needs them). `--folder` picks the folder IRIS syncs (default `INBOX`), and
`--password-stdin` reads the password from standard input, for scripts.

### Gmail with OAuth

Gmail through its API is the full-fidelity option. It needs a one-time Google Cloud
OAuth client of your own (about ten minutes: a project, the Gmail API, a consent
screen with you as the test user, and a "Desktop app" client whose JSON you save as
`~/.iris/workspace/credentials/google_oauth_client.json`). The
[Gmail setup guide](https://github.com/evarness-ai/iris-harness/blob/main/docs/usage-guides/gmail-auth.md)
walks through it. Then:

<!-- ci: skip opens a browser for Google's consent screen -->
```bash
iris auth gmail login --user you@gmail.com
```

## 2. Run setup

<!-- ci: skip needs the account connected above -->
```bash
iris email setup --account imap:you@example.com
```

With only one connected account you can leave out `--account`. The account id is the
provider and the address: `imap:you@example.com` or `gmail:you@gmail.com`.

Setup runs these steps, in order:

1. **Connect the mailbox**, and check the vault master key.
2. **Fetch recent mail**: the newest 500 messages from the last 90 days at most.
3. **Discover categories** in your mail (skipped with fewer than 40 emails).
4. **Review the categories**: accept all, or pick the ones you want.
5. **Classify**: the judge reads new mail on the local model, then a kNN classifier
   files the rest.
6. **Label preview and approval.** How many emails each `IRIS/*` label would go on, with
   sample subjects, and one question: may IRIS change this mailbox? The default is no.
   Nothing touches your mailbox before this step, and IRIS never archives, never marks
   mail read and never deletes on its own.
7. **First digest.**
8. **Review queue**: the emails the judge was unsure of wait for you in the Action
   Center.
9. **Keep it current**: the scheduled sweep starts taking this account.
10. **What IRIS just did**: what was fetched, judged and labelled, and the audit rows
    it wrote.

Setup exits 0 when it is done, 3 when it stopped to wait (for your decision, or for
something outside setup; run it again to go on) and 2 on bad input.

For scripts, `--yes` takes every default without asking, except mailbox writes:
add `--approve-writes` or `--decline-writes` to answer that one too. `--status` shows
where setup stands and changes nothing, and `--restart` starts the account's setup
over (fetched mail, judgments, categories and approvals stay).

## 3. Approve, check or revoke mailbox writes

Setup's step 6 records your answer. You can also give or take back the approval any
time, per account:

```bash
iris email writes status
```

<!-- ci: skip needs your connected account -->
```bash
iris email writes approve --account imap:you@example.com
iris email writes revoke --account imap:you@example.com
```

Without an approval, every write (a label, a move to Trash, a restore) is refused, and
the refusal names the command above.

## Start the IRIS API

The scheduled sweep and the judge run inside the IRIS API service, which also serves
the web console. Start it with a long random shared secret (every API call needs it);
it runs in the foreground, on `127.0.0.1:8003`, until you stop it:

<!-- ci: skip runs the API in the foreground until stopped -->
```bash
export IRIS_AUTH_SECRET="$(openssl rand -hex 32)"
iris serve
```

## Next

- [The email assistant](../guides/email.md): running the API, the web console, and
  what to use day to day.
- Build on IRIS: [Write a plugin](../guides/write-a-plugin.md).
