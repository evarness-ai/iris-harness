"""Sandboxed code execution as a reference plugin (OSS plan M4.6, decision 9).

The sandbox itself (``iris_harness.tools.sandbox``) is core — it is a governed
capability with an egress proxy and a runtime policy, and the eval sandbox uses it
too. What lives here is the *agent* that drives it: the bounded LLM ↔ sandbox loop
that turns "make me a PDF of this" into shell commands and an artifact.
"""
