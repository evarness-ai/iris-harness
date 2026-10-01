"""Foundation — the bottom layer: nothing here may import a layer above it.

Persistence (SQLite helpers), observability (session logs, traces, host pressure),
the process event bus, the static reference data, and the path resolution every store
shares. What unites them is that they carry no policy: they are how the harness writes
a row, records an event, publishes a signal, or finds the directory to do it in.

The layer order is enforced by the import-linter ``layers`` contract (OSS plan M6,
decision 6). An upward import from here is the worst kind the contract can catch --
foundation is what everything else is built on.
"""
