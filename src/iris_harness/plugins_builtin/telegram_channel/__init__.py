"""Telegram as a channel plugin (OSS plan M4.5).

The gateway and the console sink are core; a *surface* is not. Nothing is
re-exported here — the connector itself stays in ``iris_harness.services.channels``,
because the approvals router builds one directly from env when no gateway exists.
"""
