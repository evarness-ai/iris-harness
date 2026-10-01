# ---- web console build stage ---------------------------------------------
# The console is static files once built, so Node exists in this stage only and
# never reaches the runtime image. The output lands at /app/webui/dist below, the
# default of IRIS_WEBUI_DIST (src/iris_harness/server/iris_api/static_ui.py), so
# the API serves the console with nothing to configure.
#
# Base images are pinned by the multi-arch index digest, so a rebuild of the same commit
# gets the same OS layers (the tag alone moves under us). To take upstream security
# fixes, bump the digest deliberately:
#   docker buildx imagetools inspect node:22-slim | sed -n 3p
#   docker buildx imagetools inspect python:3.12-slim | sed -n 3p
FROM node:22-slim@sha256:43ac6c60b8f89723f746e8a92ce91abd5017e627ce1ddfe4238355d3a30b772c AS webui

WORKDIR /webui

# Dependencies first, so this layer is cached until the lockfile changes.
COPY webui/package.json webui/package-lock.json ./
RUN npm ci

# `webui/dist/` and `webui/node_modules/` are in .dockerignore: a stale build on the
# build machine never reaches the image, only what this stage builds does.
COPY webui/ ./
RUN npm run build


# ---- build stage --------------------------------------------------------
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# poetry.lock is lock-version 2.1, which Poetry 1.x cannot read.
ENV POETRY_VERSION=2.3.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

# Optional extras from pyproject, space-separated (e.g. "ml"). Empty = the lean core,
# which is what docker-compose.yml and the server image build.
ARG POETRY_EXTRAS=""

# Install deps first so this layer is cached until lockfile changes
COPY pyproject.toml poetry.lock ./
RUN poetry install --without dev --no-root ${POETRY_EXTRAS:+--extras "$POETRY_EXTRAS"}

# Copy source and register iris package entry points
COPY . .
RUN poetry install --without dev ${POETRY_EXTRAS:+--extras "$POETRY_EXTRAS"}

# Extra packages for one deployment's image, outside the lock. A server with no OS
# keychain passes `keyrings.alt` here: the OAuth tokens and vault keys live in the
# keyring, so a VM needs a file-backed one.
ARG EXTRA_PIP_PACKAGES=""
RUN if [ -n "$EXTRA_PIP_PACKAGES" ]; then \
        .venv/bin/python -m pip install --no-cache-dir $EXTRA_PIP_PACKAGES; \
    fi


# ---- runtime stage ------------------------------------------------------
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app /app
# .dockerignore filters the build context only, not a copy between stages.
COPY --from=webui /webui/dist /app/webui/dist

# No PYTHONPATH: since M6.2 layer 10 the FastAPI apps are `iris_harness.server.*`,
# installed into the venv with the rest of the package, so `uvicorn` finds them the
# same way it finds anything else that was pip-installed.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8003 8080
