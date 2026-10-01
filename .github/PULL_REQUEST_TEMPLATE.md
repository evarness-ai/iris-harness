<!-- Title: <type>(<scope>): <summary>  (feat, fix, docs, style, refactor, test, chore,
ci, perf, build). The PR is squash-merged and the title becomes the commit message. -->

## Why

<!-- The problem this solves. Link the issue (Fixes #123) if there is one. -->

## What changed

## How it was tested

<!-- Commands you ran and their results. -->

## Stable-tier impact

<!-- None, additive, or breaking. A breaking change to iris_harness.sdk, iris_harness.testing,
the PluginAPI kinds, the manifest schema, the entry-point group or the quickstart CLI needs a
deprecation (docs/reference/stable-api.md). -->

## Checklist

- [ ] Every commit is signed off (`git commit -s`, DCO)
- [ ] `scripts/ci_local.sh --fast` passes
- [ ] New behaviour has tests; a fix has the test that would have caught it
- [ ] Tool and model calls still go through the governed runner and the audit ledger
- [ ] No secrets, tokens or personal data in the diff
- [ ] No AI attribution (`Co-Authored-By` trailers for tools, "generated with" footers)
