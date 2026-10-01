"""The web UI as a channel plugin (OSS plan M4.5, decision 10).

The web UI was the one chat surface with no way to *receive*: it sent
``channel: "web"`` on every request and nothing answered to that name, so a
morning brief or a finished Activity could only go to Telegram or stdout. This
plugin registers a real ``web`` connector, which makes the web UI a peer of
Telegram rather than a privileged built-in half-surface.
"""
