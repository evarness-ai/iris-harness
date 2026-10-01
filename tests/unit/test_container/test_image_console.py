"""The image carries the web console, built in the image, never a stale local build.

Mobile + cloud-native UI plan, track 1 PR 4. The promises live in the Dockerfile and
.dockerignore, which no other test loads, and each fails silently when it breaks: the
image builds the console itself and puts it where the API looks for it, so a deployment
serves the UI with nothing to configure, and no env file reaches an image layer.

Structural checks: a Docker build is not available to the unit suite.
"""

from __future__ import annotations

import re
from pathlib import Path

from iris_harness.server.iris_api.static_ui import DEFAULT_WEBUI_DIST

ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"


def _instructions(dockerfile: str) -> list[str]:
    """Dockerfile instructions, comments dropped and ``\\`` continuations joined."""
    joined = re.sub(r"\\\n", " ", dockerfile)
    lines = (line.strip() for line in joined.splitlines())
    return [" ".join(line.split()) for line in lines if line and not line.startswith("#")]


def _stages(dockerfile: str) -> list[tuple[str, str, list[str]]]:
    """``(image, name, instructions)`` for every build stage, in order (image without digest)."""
    stages: list[tuple[str, str, list[str]]] = []
    for instruction in _instructions(dockerfile):
        match = re.fullmatch(r"FROM (\S+)(?: AS (\S+))?", instruction, flags=re.IGNORECASE)
        if match:
            # The base is pinned by digest (``node:22-slim@sha256:...``); stages are
            # told apart by the tag.
            image = match.group(1).split("@", 1)[0]
            stages.append((image, match.group(2) or "", []))
        elif stages:
            stages[-1][2].append(instruction)
    return stages


def _ignore_patterns() -> list[str]:
    lines = (line.strip() for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


# ── The image builds the console ────────────────────────────────────────────


def test_the_image_builds_the_console_in_a_node_stage() -> None:
    stages = _stages(DOCKERFILE.read_text(encoding="utf-8"))
    node_stages = [stage for stage in stages if stage[0] == "node:22-slim"]

    assert len(node_stages) == 1, "one node:22-slim stage builds webui/"
    _image, name, instructions = node_stages[0]
    assert name, "the stage is named so the runtime stage can copy from it"
    assert "RUN npm ci" in instructions, "a locked, reproducible install"
    assert "RUN npm run build" in instructions
    assert instructions.index("RUN npm ci") < instructions.index("RUN npm run build")
    assert any(i.startswith("COPY webui/") for i in instructions), "it builds webui/"


def test_the_runtime_image_gets_the_build_where_the_api_looks_for_it() -> None:
    stages = _stages(DOCKERFILE.read_text(encoding="utf-8"))
    (node_name,) = (name for image, name, _ in stages if image == "node:22-slim")
    runtime_image, _name, runtime = stages[-1]

    assert runtime_image.startswith("python:"), "Node never reaches the runtime image"
    copies = [i for i in runtime if i.startswith(f"COPY --from={node_name} ")]
    assert len(copies) == 1
    source, destination = copies[0].split()[2:]
    assert source.endswith("/dist")
    # The default of IRIS_WEBUI_DIST: the deployment sets nothing and gets the UI.
    assert destination.rstrip("/") == DEFAULT_WEBUI_DIST == "/app/webui/dist"


def test_the_console_copy_comes_after_the_app_copy() -> None:
    """``COPY --from=builder /app /app`` replaces /app wholesale; the console must land after."""
    _image, _name, runtime = _stages(DOCKERFILE.read_text(encoding="utf-8"))[-1]

    app_copy = next(i for i, line in enumerate(runtime) if line.endswith(" /app /app"))
    dist_copy = next(i for i, line in enumerate(runtime) if line.endswith(DEFAULT_WEBUI_DIST))
    assert app_copy < dist_copy


def test_a_stale_local_build_never_reaches_the_image() -> None:
    """``dist/`` is anchored at the context root, so it does not cover ``webui/dist``."""
    patterns = _ignore_patterns()

    assert "webui/dist/" in patterns
    assert "webui/node_modules/" in patterns


def test_no_secret_file_reaches_an_image_layer() -> None:
    patterns = _ignore_patterns()
    assert ".env" in patterns, "`COPY . .` must never pick up the env file"
    # A bare pattern is anchored at the context root; Vite would read `webui/.env*`.
    assert "**/.env" in patterns and "**/.env.*" in patterns
    for instruction in _instructions(DOCKERFILE.read_text(encoding="utf-8")):
        if instruction.startswith(("COPY ", "ADD ")):
            assert ".env" not in instruction, instruction
