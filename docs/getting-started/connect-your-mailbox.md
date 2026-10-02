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

IRIS logs in with a password over TLS (`--security ssl`, port 993, by default) and
checks the server's certificate against the system's trusted certificates. It does not
do OAuth over IMAP, so a provider that accepts only OAuth for IMAP cannot be connected
this way.

#### Provider notes

Settings as each provider's own help pages give them, checked on 2026-10-01. Providers
change these; if a login fails, check the linked page first.

| Provider | Server | Port, security | App password |
|---|---|---|---|
| iCloud Mail | `imap.mail.me.com` (preset for `icloud.com`, `me.com`, `mac.com`) | 993, SSL | Required; needs two-factor authentication on the Apple Account |
| Fastmail | `imap.fastmail.com` (preset for `fastmail.com`) | 993, SSL (not STARTTLS) | Required; your normal password does not work |
| Yahoo Mail | `imap.mail.yahoo.com` (preset for `yahoo.com`) | 993, SSL | Required |
| Gmail | `imap.gmail.com` (preset) | 993, SSL | Needs 2-Step Verification; not offered for work or school accounts, Advanced Protection, or security-key-only 2SV |
| Outlook.com, Microsoft 365 | `outlook.office365.com` | 993, SSL | **Not supported**: both require OAuth for IMAP (below) |
| Proton Mail | Proton Mail Bridge on `127.0.0.1` | 1143, STARTTLS (Bridge default) | The password Bridge generates (below) |

- **iCloud Mail.** Create the password at account.apple.com: Sign-In and Security, then
  App-Specific Passwords
  ([Apple](https://support.apple.com/en-us/102654)). Apple gives the IMAP user name as
  the part of your address before the `@`
  ([Apple](https://support.apple.com/en-us/102525)); if the full address is refused,
  pass that name:

  <!-- ci: skip needs an iCloud mailbox and its app-specific password -->
  ```bash
  iris auth imap login --user you@icloud.com --username you
  ```

- **Fastmail.** Settings, then Privacy & Security, then "Manage app passwords and
  access", then New app password; the default access, Mail, Contacts & Calendars,
  includes IMAP ([Fastmail: app passwords](https://www.fastmail.help/hc/en-us/articles/360058752854),
  [server names and ports](https://www.fastmail.help/hc/en-us/articles/1500000278342-Server-names-and-ports)).
  An address on your own domain needs `--host imap.fastmail.com`.
- **Yahoo Mail.** Account Security, then under External connections, Create app
  password ([Yahoo: app passwords](https://help.yahoo.com/kb/SLN15241.html),
  [IMAP settings](https://help.yahoo.com/kb/SLN4075.html)). An app password stays valid
  when you change your account password, until you delete it.
- **Gmail.** App passwords exist only with 2-Step Verification on, and not for the
  account types in the table ([Google](https://support.google.com/accounts/answer/185833)).
  Without one, use Gmail with OAuth below.
- **Outlook.com and Microsoft 365.** Outlook.com requires OAuth 2 ("Modern Auth") for
  IMAP ([Microsoft](https://support.microsoft.com/en-us/office/pop-imap-and-smtp-settings-for-outlook-com-d088b986-291d-42b8-9564-9c414e2aa040)),
  and Exchange Online has turned off Basic authentication, password logins over IMAP
  included, in every Microsoft 365 tenant; no admin can turn it back on, and app
  passwords stop working with it
  ([Microsoft Learn](https://learn.microsoft.com/en-us/exchange/clients-and-mobile-in-exchange-online/deprecation-of-basic-authentication-exchange-online)).
  IRIS has no OAuth login for IMAP yet, so neither can be connected today.
- **Proton Mail.** Proton Mail has no IMAP server of its own; Proton Mail Bridge, a
  desktop app on a paid plan, serves your mailbox on `127.0.0.1`, IMAP on port 1143 by
  default ([Proton: IMAP setup](https://proton.me/support/imap-smtp-and-pop3-setup),
  [ports](https://proton.me/support/port-already-occupied-error)). Log in with the
  password Bridge shows under Mailbox details, not your Proton password
  ([Proton](https://proton.me/support/invalid-password-error-setting-email-client)).
  Bridge encrypts that local connection (STARTTLS or SSL, its "Connection mode"
  setting) with a self-signed certificate it generates
  ([Proton](https://proton.me/support/comprehensive-guide-to-bridge-settings)), which
  IRIS's certificate check does not trust out of the box. `--security plain` is
  accepted for a loopback host only, for a bridge that allows it. Connecting IRIS to
  Bridge has not been tested yet.

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
the web console:

<!-- ci: skip runs the API in the foreground until stopped -->
```bash
export IRIS_AUTH_SECRET="$(openssl rand -hex 32)"
iris serve
```

[The email assistant](../guides/email.md#run-the-iris-api) covers the rest: the one
shared secret, used the same way in every shell that talks to this API (`iris device
pair` included, not just this one), building the web console from a source checkout,
and the separate, optional Governor service.

## Next

- [The email assistant](../guides/email.md): running the API, the web console, and
  what to use day to day.
- Build on IRIS: [Write a plugin](../guides/write-a-plugin.md).
