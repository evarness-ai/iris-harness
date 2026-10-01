# 04 · A routine delivered to Telegram

A morning note that goes out at 07:00 on weekdays, to the owner's Telegram.

What it shows:

- **a scheduled job** -- `api.register_heartbeat(name, handler, schedule="0 7 * * 1-5")`;
  the harness's scheduler runs it (and records the run), the owner can fire it by hand,
  and a routine can name it;
- **delivery through the gateway** -- the job hands a `ChannelMessage` to
  `api.services.channels.broadcast`. It never talks to Telegram: the channel does, so
  the same job reaches web push or any other channel the owner has;
- **the routine's choice of channel** -- a routine that fires the job passes
  `params["channel"]`; without one the note goes to the owner's default channel;
- **the content is data** -- `note.yaml` holds the schedule and the lines.

| File | What it is |
|---|---|
| `morning_note.py` | The plugin: the heartbeat and its handler. |
| `note.yaml` | The schedule and the note's lines. |
| `manifest.yaml` | The plugin's manifest (`provides: [heartbeat]`). |
| `test_morning_note.py` | Fires the job in a real IRIS; Telegram is faked at the HTTP layer. |

## Run it

```bash
pytest examples/04-routine-to-telegram -q
```

Expected output: `2 passed` in about 20 s (5 s with `--no-cov`). The test mounts the plugin beside a
Telegram channel whose HTTP transport is `httpx.MockTransport` -- the real
`TelegramConnector` builds the Bot API request, nothing leaves the machine -- then
fires the job with `api.services.heartbeats.trigger_by_name("morning_note")`:

- one `sendMessage` to the configured chat, text `Good morning. It is <today>.` and the
  lines from `note.yaml`; the run is `success`, no model was called;
- when Telegram answers `502`, the run is `failed` and says why.

## Use it in your IRIS

1. Give the Telegram channel (the `telegram_channel` plugin, in the `default` and
   `email` profiles) your bot: set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`, which
   `config/channels.yaml` reads. `telegram` is the default channel there.
2. Mount the plugin:

   ```bash
   cp -r examples/04-routine-to-telegram ~/.iris/plugins/morning_note
   ```

   ```yaml
   # ~/.iris/profile.yaml
   plugins:
     - name: morning_note
   ```

3. It now runs on its schedule. To run it on another one, or to another channel, ask
   IRIS for a routine in chat -- "every weekday at 7, send me the morning note on
   Telegram" -- and approve it; the routine engine fires `morning_note` with that
   channel.

## What the stable tier does not cover yet

A routine is created through chat (or the web Routines screen) and approved by the
owner; there is no stable API to create one from code, or to fire a job with a
routine's `params`. The test therefore fires the job the way the scheduler does, and the
routine path is described here rather than exercised.

## Try changing

The note's content is data. Add a line to `note.yaml`:

```yaml
lines:
  - Stand-up at 9:30.
  - Water the plants.
  - Bins go out tonight.
```

and assert it arrived, in `test_the_note_is_scheduled_and_delivered_to_telegram`:

```python
        assert message["text"].endswith("- Bins go out tonight.")
```

Both tests pass: the faked Bot API received the new line, and the job still made no
model call.

## Next

[`05-your-own-agent`](../05-your-own-agent/): a domain agent on the governed loop, with
its own tools. Background: [Write a plugin](../../docs/guides/write-a-plugin.md) (the
registration kinds, heartbeats among them).
