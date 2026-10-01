"""Small shared protocols for the channel gateway service."""

from __future__ import annotations

from typing import Protocol


class TelegramRuntimeProtocol(Protocol):
    @property
    def running(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self, *, join_timeout: float = 5.0) -> None: ...
