"""IRIS API service package (port 8003): ``main`` builds the FastAPI app.

A regular package, not a namespace one, so import-linter's graph includes it and
its contracts (the harness never imports ``iris_personal`` or ``iris_code``) apply.
"""
