# 02 · A governed tool

A noticeboard plugin with three tools the model can call. The manifest declares how
each is governed; the code only does the work.

| Tool | Declaration | What governance does |
|---|---|---|
| `list_notes` | `effect: read`, `content: external` | Runs freely; other people wrote the notes, so every result is scanned for injected instructions before the model sees it. |
| `pin_note` | `effect: write`, `confirm: once` | A change. In chat the model asks the owner once before making it; called from code, with nobody to ask, it is held for the owner's approval. |
| `remove_note` | `effect: destructive` | Nothing is removed until the owner approves the exact call. The approval card is filled in by the plugin's `describe` ("Remove 1 note / Marcus: Dentist moved the check-up to 14:30."). |

Every call goes through the same governed runner, whoever makes it: `PRE_TOOL_USE`
checks, the approval rules, `POST_TOOL_USE` checks, and an audit row for each check.
A tool the plugin registers without declaring it is refused.

| File | What it is |
|---|---|
| `noticeboard.py` | The plugin: three tools, a `validate` check and the approval card's `describe`. |
| `manifest.yaml` | The declarations under `tools:`. |
| `test_noticeboard.py` | Drives the tools through real chat turns on a scripted model, offline. |

## Run it

```bash
pytest examples/02-governed-tool -q
```

Expected output: `4 passed` in about ten seconds. The tests show:

- **read** -- "What is on the noticeboard?" calls `list_notes`; the ledger has
  `pre_tool_use` and `post_tool_use` rows naming the tool, every check `allow`;
- **approval-gated write** -- "Remove the dentist note, please." stops with
  "... needs your approval, so nothing has changed yet"; the note is still there and
  one approval is pending with its card. `h.respond_to_approval(id, approve=True)` (what
  the Approve button does) resumes the run: the call runs, governed again, and the model
  answers "The dentist note is gone.";
- **rejection** -- rejecting it runs nothing, and the resumed run says so;
- **plugin code** -- `api.tools.call("pin_note", ...)` is held for approval, then runs
  once approved.

The scripted model (`SCRIPT` in the test) is plain data: each rule matches what the
loop sends and answers as a model would -- JSON for the router, an `Action:` to call a
tool, a `Final Answer:` to finish.

## Use it in your IRIS

```bash
cp -r examples/02-governed-tool ~/.iris/plugins/noticeboard
```

```yaml
# ~/.iris/profile.yaml
plugins:
  - name: noticeboard
```

Then ask `iris -p "Remove the dentist note"` and approve it with `iris approvals`, on
the web Governance screen or on Telegram.
