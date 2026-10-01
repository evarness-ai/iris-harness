"""Channel Gateway service (port 8006).

See ``project-iris-prd/07-proactive-autonomy.md`` §7.4 for the design.
The MVP runs a WebSocket ``/ws`` endpoint for browser-style clients and,
when ``TELEGRAM_BOT_TOKEN`` is configured, a Telegram Bot API long-poller.
Both paths route inbound text to ``POST /chat/stream`` on the IRIS API.
Single-instance only.
"""
