"""Out-of-process evaluator service (story 12.gov-3.9 / design §9.3).

A thin FastAPI app exposing the evaluator registry over HTTP. The
agent process opens the evaluator DB read-only at the filesystem
layer (chmod 0o400/0o440), and only this service — running as a
separate UNIX user — has write privileges. That isolation is what
keeps a compromised agent from influencing its own oversight.
"""
