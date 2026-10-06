# Process topology: why four processes

IRIS ships four FastAPI processes under `src/iris_harness/server/`. This document
classifies each against one reason-to-isolate and says what each is on a desktop and in
a future service deployment. It is an evaluation only (L4.5 Harness Hardening,
Workstream D, issue #83); nothing is collapsed here.

A process earns its own boundary only for one of these reasons:

| Reason | What it means |
|---|---|
| Security boundary | A different trust domain: a different OS user, volume or secret, so compromising one side does not hand over the other. |
| Failure isolation | A crash or stall on one side must not take the other down. |
| Independent network boundary | It is the part that faces a different network (internet, LAN) than the rest. |
| Hardware constraint | It must run on different hardware (GPU host, always-on box). |
| None | Keep in process. |

All claims below were checked against the code, not against other docs.

## Classification

| Process | Port | Primary reason | Desktop | Future service |
|---|---|---|---|---|
| IRIS API | 8003 | Hub; not an isolation question (see below) | Keep | Keep; scale as the one stateful node |
| Governor | 8080 | **None** today | Candidate to fold in | Keep only if a remote caller appears |
| Evaluator | 8090 | Security boundary, intended and not yet enforced; opt-in | Keep in process (default) | Keep out of process, once the boundary is real |
| Channel Gateway | 8006 | Independent network boundary, plus failure isolation | Optional (needs a bot token) | Keep |

No process is isolated for a hardware reason. The hardware-bound services (Ollama,
LM Studio, the LLM proxy) are reached over HTTP but are not among the four.

### IRIS API (`server/iris_api/main.py`)

The composition root: it builds the runtime, memory, vault and the embedded governance
kernel, and serves the web UI, CLI and device clients. The other three exist to be kept
away from it or to feed it, so the question "why is it separate" has no meaning. Its own
boundary is the network one: loopback by default, with `docker-compose.yml` publishing it
as `127.0.0.1:8003` only, bearer auth and the DNS-rebinding host guard on every route
(`server/auth.py`).

### Governor (`server/governor/main.py`): reason is "none"

- The HTTP service wraps `IRISGovernorService.guard(route, payload)`, a route-level
  rate-limit and policy check. Its only route is `POST /guard/{route}`.
- Nothing in the repository calls that route. `grep` finds `/guard/` only in the service
  itself and its test. `IRIS_GOVERNOR_BASE_URL` is set in `docker-compose.yml` and
  `.env.example` but read by no code.
- The one consumer of `IRISGovernorService`, the MCP bridge, builds its own embedded
  instance (`server/iris_api/main.py`, `tools/mcp_bridge.py`), as does the kernel path.
  So the running API never crosses the process boundary to enforce anything.
- It is not a security boundary: same image, same `IRIS_AUTH_SECRET`, same `iris-data`
  volume and same user as the API. A compromised API process has an embedded copy and the
  same files, so the separate process protects nothing.
- It is not failure isolation: compose makes `iris-api` wait for governor health
  (`depends_on: service_healthy`), which adds a startup dependency and no resilience,
  since no request waits on it.
- It is not a network boundary: it is bound to loopback, like the API.

Note that `unified-governance-layer.md` section 4.1 says the HTTP service exists for
"cross-service calls (Dashboard, Background Worker, Channel Gateway, future Voice
service)". None of those call it today, and it wraps `IRISGovernorService`, not the
`GovernanceKernel`.

**Finding:** the governor's only reason is "none". Per the milestone rule this is the one
process the classification flags as a collapse candidate. This document does not collapse
it. It becomes justified if a caller outside the API process needs route guarding (the
original Voice/Dashboard idea) and, to be a security boundary rather than a convenience,
it also gets its own OS user and audit volume.

### Evaluator (`server/evaluator/main.py`): intended security boundary, opt-in

- Out of process only when `IRIS_GOVERNANCE_EVALUATOR_MODE=remote`
  (`kernel/governance/wiring.py`). The default is the in-process registry, so on a default
  install the separate process serves nothing; the health check only probes it in remote
  mode (`services/health/checks.py`).
- The design intent (`unified-governance-layer.md` sections 9.3 and 4.1) is a boundary the
  agent cannot influence: separate user, read-only store, owned policy. The code does not
  yet deliver that. The service holds an in-memory registry with no database of its own,
  `scripts/start_iris.sh` starts it as the same user with the same bearer secret, and
  `tests/security/test_evaluator_isolation.py` asserts only a `chmod`'d file and the
  read-only SQLite helper, not a separate user.
- As failure isolation it is negative: if the sidecar is unreachable, a local-tier run
  fails open and a cloud-tier run fails closed (`evaluator/client.py`). Remote mode adds a
  way to degrade safety checks.

**Finding:** not "none", because the boundary is a declared goal and the mode is off by
default. But it is not a boundary today. Keep it opt-in and in process on the desktop. In a
service deployment, run it out of process only together with the separate user and
read-only store the design calls for; otherwise remote mode buys latency and a fail-open
path.

### Channel Gateway (`server/channel_gateway/main.py`): network boundary

- It is the only process that faces outward: an inbound WebSocket (`/ws`, bearer or
  `?token=`, with the query token redacted from logs) and the Telegram Bot API long-poll.
  It keeps no session state and proxies every frame to the API's `/chat/stream`.
- It has an in-process alternative: with `IRIS_CHANNEL_GATEWAY_TELEGRAM_ENABLED=0` the
  runtime starts its own Telegram poller (`runtime/channel_wiring.py`). So the split is a
  choice, which is why failure isolation is part of the reason: a stalled or 409-conflicted
  long-poll, or a flood of WebSocket clients, then cannot take chat down.
- On the desktop nothing asks for it until `TELEGRAM_BOT_TOKEN` is set
  (`_channel_gateway_requested`), and the compose stack does not run it.

**Finding:** justified by the network boundary and by failure isolation. Keep it for the
service deployment, where internet-facing ingress should not share a process with the
runtime and vault.

## Summary

- Desktop: the API alone is enough. The evaluator stays in process, the gateway runs only
  with a bot token, and the governor is the one process with no reason.
- Service: API plus gateway are the real split. Evaluator and governor stay out of
  process only when they get an actual boundary (separate user and store, or a real remote
  caller).
- Follow-up for the maintainers, not done here: decide whether to fold the governor into
  the API, or give it a caller. Either way, correct section 4.1 of
  `unified-governance-layer.md` and drop or wire the unused `IRIS_GOVERNOR_BASE_URL`.
