# Gmail authentication — one-time setup

IRIS reads your Gmail via Google's standard OAuth 2.0 *installed-app
flow*. Tokens live in the macOS Keychain (per ADR-0003); only the
non-secret OAuth client config sits on disk.

The user-facing CLI commands are:

```bash
iris auth gmail login --user <your-address@gmail.com>
iris auth gmail status
iris auth gmail logout --user <your-address@gmail.com>
```

`login` requires a one-time Google Cloud setup — about ten minutes. The
rest of this doc walks through it.

## Why you have to do this manually

Google's OAuth model requires every IRIS install to use its own OAuth
2.0 *client* (a `client_id` + `client_secret` pair), not a shared one
you'd get from us. The setup is identical to what `gcloud`,
`google-cloud-sdk`, or any installed Google client tool needs.

You do it **once per machine**. Subsequent `iris auth gmail login`
runs reuse the same client to authorize additional addresses.

## Setup (≈10 minutes)

### 1. Create a Google Cloud project

1. Visit [console.cloud.google.com](https://console.cloud.google.com).
2. Top-bar dropdown → **"New Project"**. Name it anything (e.g. `iris-personal`).
3. Make sure your new project is selected in the top-bar dropdown for the rest of these steps.

### 2. Enable the Gmail API

1. Left nav → **"APIs & Services" → "Library"**.
2. Search **"Gmail API"**. Open it. Click **"Enable"**.

### 3. Configure the OAuth consent screen

1. Left nav → **"APIs & Services" → "OAuth consent screen"**.
2. Choose **"External"** user type (the only option for personal Gmail accounts that aren't part of a Google Workspace).
3. Fill the minimum required fields:
   - **App name**: `IRIS` (or your preference)
   - **User support email**: your address
   - **Developer contact**: your address
4. **Scopes**: click "Add or Remove Scopes" and select **`.../auth/gmail.modify`**. It lets IRIS read your mail and move messages to Trash and back (ADR-0118). It does **not** allow permanent deletion, which needs full mail access that IRIS never requests. (Before 2026-09-21 this was `gmail.readonly`; see "Upgrading from read-only" below.)
5. **Test users**: add your own Gmail address. (External + Test mode means only listed test users can authorize — fine for personal use; no Google verification needed.)
6. Save.

### 4. Create OAuth 2.0 credentials (Desktop app type)

1. Left nav → **"APIs & Services" → "Credentials"**.
2. **"+ Create Credentials" → "OAuth client ID"**.
3. **Application type**: `Desktop app`.
4. **Name**: `IRIS desktop client` (or your preference).
5. Click **"Create"**.
6. In the resulting dialog, click **"Download JSON"**. Save the file.

### 5. Place the client secrets file where IRIS expects it

**Run these commands one at a time** — do not paste them as a single
block. Some Markdown viewers strip newlines on copy, which collapses
the four commands into one `mkdir` invocation that then fails on the
`mv` (mkdir treats the trailing arguments as additional directories
to create and chokes on the existing Downloads file).

Create the credentials directory:

```bash
mkdir -p ~/.iris/workspace/credentials
```

Move the downloaded file to the canonical name (your filename will
include a long client-id hash):

```bash
mv ~/Downloads/client_secret_*.json ~/.iris/workspace/credentials/google_oauth_client.json
```

Lock down read/write permissions to your user only:

```bash
chmod 600 ~/.iris/workspace/credentials/google_oauth_client.json
```

Verify the file landed at the expected path with the right mode:

```bash
ls -la ~/.iris/workspace/credentials/google_oauth_client.json
```

You should see a `-rw-------` line at
`~/.iris/workspace/credentials/google_oauth_client.json`. That path is
the default IRIS looks for; pass `--client-secrets PATH` to
`iris auth gmail login` if you want it elsewhere.

### 6. Run the IRIS login

```bash
iris auth gmail login --user your-address@gmail.com
```

IRIS opens your default browser to Google's auth screen. Pick the
account (must match `--user` — IRIS rejects mismatches), grant
read access, and the callback returns to a transient localhost
port. Tokens land in the Keychain; the `email_accounts` row is
created in `data/iris.db`.

On macOS the first Keychain write triggers a system prompt
("`iris` wants to access your keychain"). Choose **"Always Allow"**
so background routines don't re-prompt.

## Verify

```bash
iris auth gmail status
```

You should see your address with `✓`, an expiry timestamp, and
`refresh token: yes`.

## Multiple addresses

Repeat steps 5 (path overrides if you like) and 6 with each address:

```bash
iris auth gmail login --user user.in@example.com
iris auth gmail login --user user.us@example.com
```

The same OAuth client (step 4) handles both. Each address gets its
own Keychain entry, own `email_accounts` row, and (later) its own
sweep cursor.

## Upgrading from read-only (so IRIS can trash mail)

A token granted before 2026-09-21 is read-only. It keeps reading mail, but `trash_email`
answers "connected read-only" until you grant `gmail.modify`. Do this **in this order**:

1. Pull the version with this change onto the Mac: a login from older code asks for
   read-only again.
2. In the consent screen's Scopes (step 3.4 above), add `.../auth/gmail.modify`.
3. On the Mac, once per address:
   ```bash
   iris auth gmail login --user you@gmail.com
   ```
   Google shows the new permission ("read, compose and send, and permanently delete"
   is **not** in the list; "manage" is). Allow it.
4. If IRIS also runs on a server, its keyring holds its own copy of the token: reconnect
   the account there from the web app (below), or run the same login on it, then restart
   its API.
5. Check it: ask IRIS to trash one email. The approval card lists it by subject; approve,
   then say "undo that" and it comes back.

## Reconnecting from the web app (the server runs the OAuth flow)

On a harness you reach over the network, a revoked Gmail, Calendar or Drive token can be
fixed from the phone:
Health shows **Reconnect** on the red row, and **Settings > Connections** lists every
Google account with Reconnect / Connect / Add a Google account. The server stores the new
token in its own keyring; the phone never sees it. The login above keeps working as the
fallback.

One-time setup, in the same Google Cloud project:

1. **Stop the weekly expiry.** APIs & Services > OAuth consent screen > Publishing
   status: if it says "Testing", choose "Publish app". Testing-mode refresh tokens expire
   after 7 days. For personal use Google shows an "unverified app" warning at login;
   click through it.
2. **Create a Web client.** Credentials > Create credentials > OAuth client ID > **Web
   application**. Authorised redirect URI:
   `https://<your-host>.ts.net/api/v1/connections/google/callback` (the harness's
   `IRIS_PUBLIC_URL` + that path; Settings > Connections shows the exact value).
3. **Give it to the server.** Download the client JSON and add it in Settings >
   Connections. It is stored in the server's keyring and never shown again. The Desktop
   client above keeps working for the Mac.
4. `IRIS_PUBLIC_URL` must be set in the server's environment
   (`IRIS_PUBLIC_URL=https://<your-host>.ts.net`); without it Reconnect refuses to start.

Safety rules the flow ships with: a random `state` and a PKCE verifier per start, each
link usable once for 10 minutes; the account Google approved must be the one being
reconnected (a different one saves nothing); the same scopes as the Mac login and no
others; no token in any response, redirect or log line; every start and outcome in the
governance audit ledger. Code: `src/iris_personal/connections/google.py`.

## Troubleshooting

**"OAuth client secrets not found at …"** — step 5 wasn't done, or the file is at a non-default path. Re-run with `--client-secrets <path>`. If you pasted step 5 as a single block and saw `mkdir: …client_secret_…json: File exists`, the newlines got stripped on copy — the `mv` never ran. Run each step-5 command on its own line.

**"OAuth authorized 'other@gmail.com' but CLI requested 'user@gmail.com'"** — you picked a different account in the browser than you passed to `--user`. Re-run with the matching address, or sign out of the other account in the browser first.

**"Access blocked: This app's request is invalid"** — the OAuth consent screen (step 3) wasn't completed, or your address isn't on the **test users** list. Go back to step 3.

**Token revoked / expired refresh token** — happens if you revoke access from your Google Account's security settings or after 6 months of inactivity. Run `iris auth gmail logout --user <addr>` then `iris auth gmail login --user <addr>` again.

## What gets stored where

| What | Where | Format |
|---|---|---|
| OAuth `client_id` + `client_secret` | `~/.iris/workspace/credentials/google_oauth_client.json` | plain JSON (chmod 600) |
| OAuth access + refresh tokens | macOS Keychain (`iris-gmail` service, account = email address) | JSON blob matching `google.oauth2.credentials.Credentials.to_json()` |
| Account registration row | `data/iris.db.email_accounts` | row with `provider='gmail'`, `address`, `active=1` |

Nothing in this flow involves IRIS sending data to a third party other
than Google itself for the OAuth handshake.

## Related

- ADR-0003 — Keychain over vault (credential storage rationale).
- ADR-0011 — Keychain credentials module implementation shape.
- ADR-0016 — Email accounts registry shape.
- `src/iris_harness/memory/identity/gmail_oauth.py` — the implementation.
