# Contributing to IRIS

IRIS is a governed agent harness for small local models. Most contributions are
**plugins**; changes to the core are welcome too, but they carry the governance rules
below. Coding agents working on the repo: read [`AGENTS.md`](AGENTS.md) as well.

## Where things go

- **Bugs and concrete, actionable changes:** GitHub Issues, using one of the forms.
- **Questions, ideas, "is this a good plugin?":** GitHub Discussions.
- **Security issues:** never in public. See [`SECURITY.md`](SECURITY.md).

## Dev setup

Python 3.12 or 3.13 (`.python-version` pins 3.12), [Poetry](https://python-poetry.org/) 2.x.

```bash
poetry install                 # add `-E ml` for the torch-backed embedders
poetry run iris --help
```

Or run `./install.sh`, which does the `poetry install` above and hands off to
`iris setup` — a guided wizard for the auth secret and the optional add-ons
(background services, Telegram pairing, email).

## Start with a plugin

A plugin is a package with a `manifest.yaml` and a `setup(api)` that registers what it
adds. Scaffold one, with a passing test, by kind:

```bash
poetry run iris plugin new my-plugin --kind tool   # or channel, mail-provider, research-provider, skill
```

The contract and the registration kinds are in
[`docs/architecture/plugin-contract.md`](docs/architecture/plugin-contract.md); runnable
examples are in [`examples/`](examples/README.md). A plugin imports only the **stable
tier** ([`docs/reference/stable-api.md`](docs/reference/stable-api.md)):
`iris_harness.sdk`, `iris_harness.testing`, the `PluginAPI` kinds, the manifest schema and
the `iris_harness.plugins` entry-point group. Everything else is internal and may change in
any release. Hold your plugin to the tier with `iris_harness.testing.check_stable_imports`.

## Stable tier and deprecation

A change that removes or renames a stable name, kind, manifest key or command, or changes
its meaning, is breaking. It keeps the old name working for at least one minor release,
warns with `DeprecationWarning` naming the replacement, and is listed in the release notes.
Adding is never breaking. `tests/unit/test_stable_tier.py` pins the tier; update
`tests/fixtures/stable_tier/names.txt` only with a deprecation in the same PR.

## Governance non-negotiables

A contribution must keep all of these; review rejects a PR that weakens one.

- **Every tool call goes through the governed tool runner** (`GovernedToolRunner` in
  `src/iris_harness/agent/tool_runner.py`), and every model call through the governance
  kernel (`src/iris_harness/kernel/governance/`). `tests/security/test_no_bypass.py` fails
  a new direct call site.
- **Every model call, tool call and answer leaves an audit row.** Do not add a path that
  answers without one, including deterministic handlers.
- **Personal data stays local.** Anything that could egress is gated by the egress layer
  and ships off by default behind an opt-in flag.
- **Import contracts hold** (`poetry run lint-imports`, `[tool.importlinter]` in
  `pyproject.toml`): the core is layered, reference plugins import only the SDK, no plugin
  imports another. A new edge is a design question; do not add it to `ignore_imports`.
- **User state resolves `IRIS_HOME`** (`iris_harness.foundation.paths`), never
  `Path.home()` directly, so tests cannot write into a real profile.

## Tests and the local gate

```bash
export IRIS_AUTH_SECRET="test-secret-for-testing" IRIS_DISABLE_WARMUP=1
poetry run pytest tests/unit/<the path you changed>
scripts/ci_local.sh --fast     # ruff, black, mypy, import contracts, changed-scope tests
scripts/ci_local.sh            # the full gate (full suite, playground smoke, secret scan)
```

Hosted CI runs the **changed-scope** tests on Python 3.12 for every pull request and push
to `main` (`scripts/changed_tests.sh`, which never escalates to the whole suite there: when a
core file such as `pyproject.toml` changes it runs `-m smoke` plus the mapped tests and says
so in the job summary). The entire suite on 3.12 and 3.13 runs on demand:
`gh workflow run ci -f full=true`, or Actions, ci, Run workflow, full.

Tests mirror the source tree under `tests/unit/`. Async tests run with
`asyncio_mode = "auto"`: never add `@pytest.mark.asyncio`. Tests never reach a model or the
network; use the scripted fake model in `iris_harness.testing`. New behaviour ships with
tests; a bug fix ships with the test that would have caught it.

Style: Black (line length 100), Ruff, MyPy strict. No emojis in code, docs or commits.

## Commits, PRs and sign-off

- **DCO, no CLA.** Every commit carries a `Signed-off-by:` line certifying the
  [Developer Certificate of Origin](https://developercert.org/): commit with `git commit -s`.
  A GitHub no-reply address is fine. The `dco` check fails an unsigned commit.
- **Titles:** `<type>(<scope>): <summary>`, type one of `feat`, `fix`, `docs`, `style`,
  `refactor`, `test`, `chore`, `ci`, `perf`, `build`; scope optional. The PR title is linted
  the same way, because it becomes the commit message.
- **Squash merge only.** One PR, one commit on `main`. Keep PRs focused.
- One CODEOWNER review is required; `src/iris_harness/kernel/governance/` always needs one.
- No AI attribution in commits, PRs or credits: no `Co-Authored-By` trailers for tools, no
  "generated with" footers. You are the author of what you submit.

## Reporting a bug

Use the bug form: what you ran, what you expected, what happened, your version and model
backend, and the relevant log excerpt with secrets and personal data removed.
