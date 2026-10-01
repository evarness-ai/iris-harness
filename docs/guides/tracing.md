# Send traces to your backend

IRIS records every turn in its own session logs, and the web console's Call trace and
Sessions screens read those. You need nothing else to see what a turn did.

For span-level detail (each pipeline stage, each model call, each outbound HTTP call)
IRIS emits OpenTelemetry traces and sends them over OTLP to a backend you run: Arize
Phoenix, Jaeger, Grafana Tempo, Honeycomb, an OpenTelemetry Collector, or anything else
that accepts OTLP. IRIS bundles no trace UI.

With no endpoint set, nothing is exported, and `/healthz` reports
`"tracing_enabled": false` with no error.

## The settings

IRIS reads the standard OpenTelemetry variables, the way the OpenTelemetry SDK reads
them:

| Variable | What it does |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Base URL of the backend. For OTLP/HTTP, `/v1/traces` is appended. Setting it turns export on. |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Full traces URL, used as given. Wins over the base URL. |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` (default, port 4318) or `grpc` (port 4317). `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` overrides it for traces. |
| `OTEL_EXPORTER_OTLP_HEADERS` | Headers to send, such as an API key: `key1=value1,key2=value2`. |
| `OTEL_SERVICE_NAME` | The service name traces are filed under. Default: `iris`. `OTEL_RESOURCE_ATTRIBUTES` works too. |
| `OTEL_SDK_DISABLED=true` or `OTEL_TRACES_EXPORTER=none` | Turns export off even with an endpoint set. |

The exporter also reads the standard timeout, compression and TLS certificate variables
(`OTEL_EXPORTER_OTLP_TIMEOUT`, `OTEL_EXPORTER_OTLP_COMPRESSION`,
`OTEL_EXPORTER_OTLP_CERTIFICATE` and their `_TRACES_` forms).

Two IRIS settings choose which libraries get automatic client spans while export is on.
Both default to on:

| Variable | What it does |
|---|---|
| `IRIS_OTEL_LANGCHAIN_ENABLED` | LangChain spans (OpenInference attributes: model, messages, tokens). |
| `IRIS_OTEL_HTTPX_ENABLED` | A span for every outbound `httpx` call. |

The IRIS API traces when an endpoint is set. The channel gateway also needs
`IRIS_CHANNEL_GATEWAY_TRACING_ENABLED=1`.

Spans are batched before they are sent. Each batch sent is logged as one `iris.egress`
line naming the host, like every other outbound call.

!!! warning "Traces carry your conversations"
    LangChain spans include prompts and model replies. A backend on this machine keeps
    them here; a hosted backend receives them. Point the endpoint at a remote service
    only if you mean to send it that data.

Put the variables in `.env` (see `.env.example`) or export them before `iris serve`.
Under Docker Compose, `.env` is passed to the containers. From inside a container, the
host is `host.docker.internal`, not `127.0.0.1`.

## Phoenix

[Arize Phoenix](https://github.com/Arize-ai/phoenix) runs as its own process. Its UI and
its OTLP/HTTP receiver share port 6006:

```bash
pip install arize-phoenix          # in an environment of its own, not IRIS's
phoenix serve                      # UI at http://127.0.0.1:6006
```

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:6006
iris serve
```

Phoenix files traces under a project. IRIS sets the `openinference.project.name`
resource attribute to the service name, so they land in the `iris` project.

## Jaeger

The all-in-one image accepts OTLP on 4317 (gRPC) and 4318 (HTTP):

```bash
docker run --rm --name jaeger \
  -p 127.0.0.1:16686:16686 -p 127.0.0.1:4317:4317 -p 127.0.0.1:4318:4318 \
  jaegertracing/all-in-one:latest
```

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
iris serve
```

Open <http://127.0.0.1:16686> and pick the `iris` service. For gRPC instead, set
`OTEL_EXPORTER_OTLP_PROTOCOL=grpc` and use port 4317.

## Any backend, through an OpenTelemetry Collector

A collector receives OTLP from IRIS and forwards it wherever you like: Grafana Tempo,
Honeycomb, a vendor, or several at once. A minimal `collector.yaml` that receives OTLP
and forwards to Tempo:

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
      grpc:
        endpoint: 0.0.0.0:4317
processors:
  batch: {}
exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/tempo]
```

```bash
docker run --rm -p 127.0.0.1:4318:4318 -p 127.0.0.1:4317:4317 \
  -v "$PWD/collector.yaml:/etc/otelcol-contrib/config.yaml" \
  otel/opentelemetry-collector-contrib:latest
```

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
iris serve
```

A hosted backend that speaks OTLP directly needs no collector. Honeycomb, for example:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=<your API key>"
```

## Check it

```bash
curl -s localhost:8003/healthz
```

`tracing_enabled` is `true`, `otlp_endpoint` shows where spans go (without any
credentials), `otel_targets` lists the instrumented libraries, and
`observability_error` is `null`. Send a chat message and the trace appears in your
backend within a few seconds.
