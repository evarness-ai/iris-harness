# 01 · A deterministic handler

A plugin that answers "when is the library open?" from a YAML file, with no model
call. It is registered with `api.register_intercept`; the docs call these
**deterministic handlers**.

What it shows:

- a handler sees each message before the model and returns `None` to pass it on;
- its answer is built with `api.services.deterministic_reply`, so it is governed like
  a generated one: the model-free response check runs on it and the audit ledger gets a
  `pre_response` row marked `deterministic: true`, with the handler's name;
- the data (`hours.yaml`) lives beside the code, not in it.

| File | What it is |
|---|---|
| `opening_hours.py` | The plugin: `setup(api)` registers the handler. |
| `hours.yaml` | The opening hours it answers with. |
| `manifest.yaml` | The plugin's manifest (`provides: [intercept]`). |
| `test_opening_hours.py` | Runs the plugin in a real governed IRIS, offline. |

## Run it

From the repository root:

```bash
pytest examples/01-deterministic-handler -q
```

Expected output: `2 passed` in a few seconds. The tests build a real IRIS in a
throwaway home, on the scripted fake model, with the network refused, and check:

- "When is the library open on Saturday?" is answered from `hours.yaml`, no model is
  called, and the answer's audit row says `deterministic=True`, `handler="opening_hours"`;
- "Tell me a joke about libraries." is not claimed and goes to the model.

## Use it in your IRIS

Home plugins need no packaging. Copy the directory, then list the plugin in
`~/.iris/profile.yaml` (merged over the shipped profile):

```bash
mkdir -p ~/.iris/plugins
cp -r examples/01-deterministic-handler ~/.iris/plugins/opening_hours
```

```yaml
# ~/.iris/profile.yaml
plugins:
  - name: opening_hours
```

```bash
iris --dump-config                   # opening_hours shows up under home:...
iris -p "When is the library open?"
```

Imports: only `iris_harness.sdk` in the plugin, `iris_harness.testing` in its test.
