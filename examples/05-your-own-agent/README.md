# 05 · Your own agent

A to-do agent. The plugin brings the domain -- its tools and a deterministic fallback --
and the harness brings the agent: `api.register_loop_intent("planner", fallback=...)`
hands every turn the router classifies as `planner` to the one governed loop the
harness builds (every model call audited at `PRE_LLM_CALL`, every tool call through
the governed runner, approvals, the curator), over the tools the plugin registered.

What it shows:

- **a loop intent** -- planner turns run on the governed loop with `list_todos` (read)
  and `add_todo` (write, `confirm: never`: the owner asked for it in this very turn);
- **read-first** -- `read_first_intents: [planner]` in the manifest: a planner answer
  must come from a read, so the loop refuses an answer that never looked at the list;
- **the fallback** -- when the loop cannot give a grounded answer (the model guesses
  twice, errors or answers nothing), the plugin's deterministic `fallback(task)` answers
  instead. It is the floor under the model.

| File | What it is |
|---|---|
| `todo_agent.py` | The plugin: two tools, the fallback, `register_loop_intent`. |
| `manifest.yaml` | Tool declarations and `read_first_intents`. |
| `test_todo_agent.py` | Three chat turns on a scripted model, offline. |

## Run it

```bash
pytest examples/05-your-own-agent -q
```

Expected output: `3 passed` in about ten seconds:

- "What is on my to-do list?" -- routed to `planner`, answered by the loop after
  calling `list_todos` (`result.agent == "planner"`);
- "Add buy oat milk to my to-do list." -- `add_todo` runs, then the loop reads the list
  back before it answers;
- a model that answers without reading, twice, is overruled: the fallback's list is
  the answer.

## A limit you will hit: the router's intents are fixed

The router classifies a message into one of a fixed set of intents (`communication`,
`calendar`, `files`, `planner`, `finance`, `search`, ...), each mapped to an agent
lane. `register_loop_intent` can claim a lane the mounted profile leaves free -- this
example takes `planner`, as the email plugin takes the `email` lane -- but a plugin
cannot yet add a new intent ("recipes") or the words that route to it: those live in
the owner's `config/intent_keywords.yaml` and in the router's own list. A plugin-defined
intent is planned work (OSS plan L2 finding).

## Use it in your IRIS

```bash
cp -r examples/05-your-own-agent ~/.iris/plugins/todo_agent
```

```yaml
# ~/.iris/profile.yaml
plugins:
  - name: todo_agent
```

Then `iris -p "What is on my to-do list?"`. (Do not mount it beside the `planner`
plugin of the personal-assistant profile: both would claim the same lane.)
