"""Reference plugins shipped with the harness.

Each subpackage is a complete plugin: ``manifest.yaml`` + ``plugin.py`` exposing
``setup(api)``. They use only :class:`iris_harness.sdk.PluginAPI` — never
runtime internals — so they double as the worked examples for plugin authors
and as the proof that the contract is sufficient (OSS plan release gate 2).
"""
