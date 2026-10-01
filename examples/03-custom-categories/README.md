# 03 · Your own email categories

The email judge sorts every new email into a bucket -- shipped as **Needs reply**,
**Bill**, **Event**, **FYI** and **Unsure** -- and those buckets are what the digest,
the Inbox, the labels and chat corrections speak in. They are data: the judge reads
`judge.yaml`, and your override replaces its keys.

`judge.yaml` here renames the buckets to your words (**Waiting on me**, **Money**,
**Dates**, **Reading**), gives each its mailbox label and the definition the judge
model is told, and teaches chat your words for them ("the carpool email is money").

| Key | What it changes |
|---|---|
| `buckets.<key>.name` | What IRIS calls the bucket everywhere you read it. |
| `buckets.<key>.label` | The mailbox label written for it (after you approve mailbox writes). |
| `buckets.<key>.definition` | What the judge model is told the bucket means. |
| `bucket_words` | Your words for each bucket in chat and on the Unsure card. |

Rules: the file goes to `<config dir>/email/judge.yaml`; each top-level key **replaces**
the shipped one (no deep merge), so `buckets:` lists every bucket and keeps `unsure`.
The shipped file, with every other key (the prompt, `unsure_below`, the sender rules),
is `src/iris_personal/plugins/email_workflows/judge.yaml`.

## Run it

```bash
pytest examples/03-custom-categories -q
```

Expected output: `2 passed` in about 23 s (7 s with `--no-cov`). The test copies IRIS's config directory,
adds this `judge.yaml` as `email/judge.yaml`, and runs the `email` profile on it with a
scripted model:

- **your categories** -- "the carpool email is money" is a correction in your words:
  the judge's correction handler (`email_rebucket`) answers it without a model call
  ("I couldn't find a judged email like "carpool" ...": the mailbox is empty), and its
  audit row says `deterministic=True, handler="email_rebucket"`;
- **the shipped categories** -- the same sentence is not a correction there, and goes on
  to the email assistant.

## Use it on your mailbox

From a checkout, put the file at `config/email/judge.yaml`; with `IRIS_CONFIG_DIR` set,
at `$IRIS_CONFIG_DIR/email/judge.yaml`. New mail is judged into your buckets from then
on; `iris email judgments` lists the verdicts by your names.

## What does not work yet (L2 findings)

- **The demo ignores it.** `iris email demo` runs on the shipped config on purpose (so
  the owner's settings never leak into the demo home), so you cannot try your
  categories on the synthetic mailbox.
- **No per-user override location for an install.** Installed from a wheel, the config
  directory is inside the package; the only way to override `email/judge.yaml` is an
  `IRIS_CONFIG_DIR` that holds a complete copy of the config.
- **No stable way to judge mail in a test.** Judging needs a connected account for the
  sweep; connecting one is not in the stable API yet (`provider_api` has no account
  call), so this example shows your categories taking effect in the running assistant,
  not a mailbox sorted into them.

## Try changing

Teach chat one more word of yours. In `judge.yaml`, add `receipt` to the `bill` words:

```yaml
  bill: [receipt, money, bill, statement, invoice, payment]
```

and in `test_custom_categories.py` set `CORRECTION = "the carpool email is a receipt"`.
Both tests still pass: with your file the sentence is a correction, answered by
`email_rebucket` with no model call; with the shipped file it is not.

## Next

[`04-routine-to-telegram`](../04-routine-to-telegram/): a job that runs on a schedule
and reaches the owner through a channel. Background:
[The email assistant](../../docs/guides/email.md) ("Your own categories").
