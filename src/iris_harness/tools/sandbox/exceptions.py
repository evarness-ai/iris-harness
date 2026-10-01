"""Exceptions raised by the sandbox subsystem."""

from __future__ import annotations


class SandboxError(Exception):
    """Base class for sandbox failures."""


class DockerUnavailableError(SandboxError):
    """Raised when the Docker CLI is missing or the daemon is unreachable."""


class SandboxImageMissingError(SandboxError):
    """Raised when the sandbox image hasn't been built yet."""


class SandboxTimeoutError(SandboxError):
    """Raised when a command exceeds its allotted timeout."""
