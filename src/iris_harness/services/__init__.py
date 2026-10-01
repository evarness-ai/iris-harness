"""The harness's long-lived services (OSS plan M6.2 layer 5).

Everything that runs *for* the agent rather than *as* the agent: health,
routines, learning, notifications, system status, missions, activities, RAG,
tasks, channels, heartbeats and research (the search-provider chain). They sit
above memory and below the agent, and the layers contract holds them to one row --
what imports what inside this package is this package's business.
"""
