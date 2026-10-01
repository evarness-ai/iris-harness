# Run with Docker

Docker Compose runs IRIS as a set of services that stay up: the IRIS API (with the web
console and the scheduled email jobs), the Governor, and Ollama. It builds the image
from a clone of the repository.

```bash
git clone https://github.com/evarness-ai/iris-harness.git
cd iris-harness
export IRIS_AUTH_SECRET="$(openssl rand -hex 32)"   # or put it in .env
docker compose up        # or: make up
```

| Service | Port | Role |
|---|---|---|
| `ollama` | 11434 | The local model runtime. |
| `ollama-init` | | One shot: pulls the tier models, then exits. |
| `governor` | 8080 | The HTTP front of the governance kernel: policy, rate limits, audit. |
| `iris-api` | 8003 | Chat, streaming chat, and the web console. |

Every port binds to `127.0.0.1`. `iris-api` waits for the Governor to be healthy and for
the models to finish pulling, so the first start is slow (about 6 GB of model weights)
and later ones are fast.

## The one required variable

Compose refuses to start without `IRIS_AUTH_SECRET`, and every API call except the
`/healthz` probe must carry it as a bearer token:

```bash
curl -s localhost:8003/chat -H "authorization: Bearer $IRIS_AUTH_SECRET" \
  -H 'content-type: application/json' \
  -d '{"message": "what time is it?", "session_id": "demo"}'
```

Everything else defaults to local-only: the model endpoints point at the bundled
Ollama. A `.env` beside `docker-compose.yml` (see `.env.example`) is where credentials
and overrides go; it is loaded when present.

No trace UI runs in the stack. To see OpenTelemetry traces, run a backend of your own and
set `OTEL_EXPORTER_OTLP_ENDPOINT` in `.env` (from a container, the host is
`host.docker.internal`); see [Send traces to your backend](tracing.md).

## On a Mac: Metal-accelerated models

Ollama in a container runs on the CPU. For Metal speed, run Ollama natively
(`ollama serve`, with the tier models pulled) and start IRIS against it:

```bash
make up-host
```

## Lifecycle

```bash
make logs                # follow iris-api and the Governor
make down                # stop and remove the stack; the volumes stay
docker compose down -v   # also drop the data and model volumes
```

## Models

`ollama-init` pulls the models `config/llm_tiers.yaml` routes to. To change them, edit
both that file and the `ollama-init` command in `docker-compose.yml`, or pull by hand:

```bash
docker compose exec ollama ollama pull <model>
```
