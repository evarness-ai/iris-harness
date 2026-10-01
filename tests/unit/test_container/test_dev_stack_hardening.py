"""The local Docker stack: no Linux capabilities, loopback-only ports, a pinned image.

Owner's decision (2026-09-26): the containers keep running as root and are hardened
around that. Each promise below lives in docker-compose.yml or the Dockerfile, which no
other test loads, and each one fails silently: a service added without the hardening
block, or a port published on every interface, still starts and passes every health
check.

Structural checks only: the unit suite has no Docker.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEV_COMPOSE = ROOT / "docker-compose.yml"
DOCKERFILE = ROOT / "Dockerfile"

NO_NEW_PRIVILEGES = "no-new-privileges:true"


def _services(path: Path) -> dict[str, dict[str, Any]]:
    services: dict[str, dict[str, Any]] = yaml.safe_load(path.read_text(encoding="utf-8"))[
        "services"
    ]
    return services


def test_every_service_drops_all_capabilities_and_cannot_gain_privileges() -> None:
    for name, spec in _services(DEV_COMPOSE).items():
        assert spec.get("cap_drop") == ["ALL"], f"{name} keeps default caps"
        assert NO_NEW_PRIVILEGES in spec.get("security_opt", []), name
        assert spec.get("privileged") is not True, name


def test_no_service_adds_a_capability_back() -> None:
    for name, spec in _services(DEV_COMPOSE).items():
        assert not spec.get("cap_add"), f"{name}: {spec.get('cap_add')}"


def test_nothing_mounts_the_docker_socket() -> None:
    """A container holding the socket is root on the host, whatever its capabilities."""
    for name, spec in _services(DEV_COMPOSE).items():
        for volume in spec.get("volumes", []):
            assert "docker.sock" not in str(volume), name


def test_dev_compose_publishes_every_port_on_loopback_only() -> None:
    published = [
        (name, str(port))
        for name, spec in _services(DEV_COMPOSE).items()
        for port in spec.get("ports", [])
    ]
    assert published, "the dev stack publishes the API, the governor and Ollama"
    for name, port in published:
        assert re.fullmatch(r"127\.0\.0\.1:\d+:\d+", port), f"{name} publishes {port!r}"


def test_the_base_images_are_pinned_by_digest() -> None:
    bases = re.findall(r"^FROM\s+(\S+)", DOCKERFILE.read_text(encoding="utf-8"), re.MULTILINE)
    assert bases
    for base in bases:
        assert re.search(r"@sha256:[0-9a-f]{64}$", base), f"{base} is not pinned"


def test_the_image_keeps_root() -> None:
    """Owner's decision: no USER line (state volumes are root-owned)."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert not re.search(r"^USER\s", text, re.MULTILINE)
