"""The domain plugins, registered as ``iris_harness.plugins`` entry points.

Each mounts through ``PluginAPI`` exactly as a third-party plugin does: intercepts,
intent handlers, tools, heartbeats, CLI commands and pending-action providers. They
may import their own domain library and the harness's public surface; they may not
import the harness's composition root (release gate 2).
"""
