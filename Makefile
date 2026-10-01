# IRIS — developer convenience targets.
#
# The experiment* targets drive the PRIVATE iris-evals behavioural harness
# (IRIS-as-SUT, exp-006) against a live IRIS, end to end, from a cold start.
# Full runbook: ../agent-workbench/iris-evals/RUNNING.md
#
#   make experiment         cold start, fully dockerized (IRIS + containerized Ollama)
#   make experiment-host    IRIS in docker -> your native Metal Ollama (faithful tiers)
#   make experiment-native  harness only, against an IRIS you already started
#
# ! macOS: `experiment` and `experiment-host` build the iris:latest Linux image,
#   whose Dockerfile compiles macOS-native deps (pyobjc/pyaudio) that cannot build
#   in the Linux container, so both fail at image build (iris-evals/PREREQUISITES.md
#   gap #7). On a Mac, start IRIS natively and use `make experiment-native`.
#
# Override any of these on the command line, e.g.:
#   make experiment MODELS="llama3.2:3b" IRIS_EVALS=/path/to/iris-evals

COMPOSE      ?= docker compose
IRIS_API_URL ?= http://localhost:8003
IRIS_EVALS   ?= ../agent-workbench/iris-evals
# Models pulled into the containerized Ollama; keep in sync with config/llm_tiers.yaml
# (router=llama3.2:3b, tier1=granite4:latest, tier2=qwen2.5:7b-instruct).
MODELS       ?= llama3.2:3b granite4:latest qwen2.5:7b-instruct

.PHONY: help up up-host up-local down logs wait pull-models run-harness \
        experiment experiment-host experiment-native

help:
	@echo "IRIS make targets:"
	@echo "  make up                 one-command local stack: build + Ollama + models + Governor + IRIS + Phoenix"
	@echo "  make down | logs"
	@echo "  make experiment         cold start (all-docker): IRIS + Ollama up, pull models, run harness"
	@echo "  make experiment-host    IRIS in docker -> native Metal Ollama, then run harness (faithful)"
	@echo "  make experiment-native  harness only, against an already-running IRIS  [use this on macOS]"
	@echo "  NOTE: experiment / experiment-host fail to build on macOS (native deps); see iris-evals/PREREQUISITES.md gap #7"
	@echo "  make up-local | up-host | down | logs | wait | pull-models | run-harness"
	@echo "  vars: IRIS_API_URL=$(IRIS_API_URL)  IRIS_EVALS=$(IRIS_EVALS)  MODELS='$(MODELS)'"

# --- one command: full local stack (Ollama + models + Governor + IRIS + Phoenix) ---
up:
	$(COMPOSE) up -d --build
	@$(MAKE) wait
	@echo ""
	@echo "  IRIS API:   $(IRIS_API_URL)   (POST /chat)"
	@echo "  Phoenix UI: http://localhost:6006"
	@echo "  Governor:   http://localhost:8080"

# --- bring IRIS up (pick one) ---
# Everything in docker, including Ollama (portable; CPU-only on a Mac).
up-local:
	$(COMPOSE) up -d

# IRIS in docker, talking to the host's native (Metal) Ollama. --no-deps so the
# containerized Ollama + model puller aren't dragged in by iris-api's depends_on.
up-host:
	$(COMPOSE) -f docker-compose.yml -f docker-compose.hostllm.yml up -d --no-deps governor iris-api

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100 iris-api governor

# --- helpers ---
# Block until IRIS reports its runtime is built (cold model load can take a couple
# minutes). /healthz is the open probe; /health is data and needs a token (ADR-0117).
wait:
	@echo "waiting for IRIS at $(IRIS_API_URL)/healthz ..."
	@for i in $$(seq 1 60); do \
	  curl -sf $(IRIS_API_URL)/healthz 2>/dev/null | grep -q '"runtime_ready": *true' && { echo "  IRIS up"; exit 0; }; \
	  sleep 3; \
	done; echo "  timed out; check 'make logs'"; exit 1

# Pull tier models into the containerized Ollama (only needed for 'up-local').
pull-models:
	@for m in $(MODELS); do echo "pulling $$m ..."; $(COMPOSE) exec -T ollama ollama pull $$m; done

# Run probe + all four batteries; proofs land in $(IRIS_EVALS)/docs/.../results/.
run-harness:
	IRIS_API_URL=$(IRIS_API_URL) $(IRIS_EVALS)/scripts/run-experiment.sh

# --- one-command experiments (cold start) ---
experiment: up-local wait pull-models run-harness
	@echo "done — proofs in $(IRIS_EVALS)/docs/exp-006-iris-agent-e2e-behavioural/results/"

experiment-host: up-host wait run-harness
	@echo "done — proofs in $(IRIS_EVALS)/docs/exp-006-iris-agent-e2e-behavioural/results/"

experiment-native: wait run-harness
	@echo "done — proofs in $(IRIS_EVALS)/docs/exp-006-iris-agent-e2e-behavioural/results/"
