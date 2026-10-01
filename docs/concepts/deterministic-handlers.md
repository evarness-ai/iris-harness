# Deterministic handlers

A deterministic handler answers a turn without a model: it parses the message, calls
typed code, and returns a grounded answer. In IRIS it is a first-class primitive, and
it is governed exactly like a generated answer.

## Why

When an action is grounded and parseable, a model's tool choice is the weakest link.
Measured on one such action in IRIS (configuring a scheduled brief from a plain
request): a 7B local model called the right tool 25% of the time; a 35B model, 0%; a
judge that read only the output accepted the made-up confirmation. A deterministic
handler that parsed the request and called the function did it every time (5 of 5).
Removing the model from that decision moved the number from 25% to 100%; a bigger
model moved it the wrong way.

So a plugin can claim the turns it can answer exactly, and leave the rest to the model.

## How

```python
from iris_harness.sdk import PluginAPI


def setup(api: PluginAPI) -> None:
    reply = api.services.deterministic_reply

    def opening_hours(message, *, session_id, span=None):
        if "library" not in message.lower() or "open" not in message.lower():
            return None  # not ours: the next handler, then the model, gets the turn
        return reply(
            message=message,
            session_id=session_id,
            response="The library is open 9:00 to 17:00, Monday to Friday.",
            metadata={"handler": "opening_hours"},
            span=span,
        )

    api.register_intercept("opening_hours", opening_hours, trace_text="opening hours")
```

Handlers see each message in order (`config/intercepts.yaml` first, then plugins in
registration order; a profile's `intercept_order` moves names to the front). The first
one that returns a reply answers the turn; returning `None` passes it on. The
[deterministic-handler example](https://github.com/evarness-ai/iris-harness/tree/main/examples/01-deterministic-handler)
is a complete plugin with its test.

## Governed like a generated answer

Every turn runs through one pipeline, and the stages every answer needs run whether a
model or a handler produced it:

- **Before the handler.** The message passes the turn's input screen (`pre_turn`):
  classification and, when enabled, the input safety screen. A handler never sees a
  message the screen refused.
- **After the handler.** Its answer passes the same model-free response checks as a
  generated answer (`pre_response`: credentials, identity secrets, internal details).
  A blocked answer is replaced with the refusal text.
- **On the record.** Each check is audited with `deterministic: true` and the handler's
  name, and the answer reaches the session log through the one recording path. The
  log also records which handler answered (`handler.end`) and what the response check
  decided (`guard.end`), so the turn is listed in Sessions and drawn in Call trace:
  request, handler, response check, answer, with the audit rows beside them.

A turn the system opens has no message to read, so it is answered by an **opener**: a
deterministic handler declared under `openers:` in `config/intercepts.yaml` and run by
name (`IrisRuntime.open_turn`). It skips the input screen, since nothing arrived, and
gets the same response check, audit row and session log as any handler's answer. The
first-chat welcome is the one opener today.

Declare `guard_output=True` when the answer repeats text someone else wrote (an email
subject, a sender, a headline): the model-based output guard then runs on your
handler's answers too, when the operator has turned it on.

This design came from a red-team finding: a deterministic answer used to end the turn
before any response check ran, so it skipped every guard. Now no path does.
