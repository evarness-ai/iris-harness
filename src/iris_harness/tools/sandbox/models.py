"""Result and config dataclasses for the sandbox subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExecResult:
    """Outcome of a single ``run_shell`` invocation in the sandbox."""

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self, *, max_chars: int = 4000) -> str:
        """Compact text rendering safe to feed back to an LLM."""
        parts: list[str] = [f"exit_code={self.exit_code}  duration={self.duration_ms:.0f}ms"]
        if self.timed_out:
            parts.append("(timed out)")
        if self.artifacts:
            parts.append("artifacts: " + ", ".join(self.artifacts))
        if self.stdout:
            parts.append("stdout:\n" + _trim(self.stdout, max_chars))
        if self.stderr:
            parts.append("stderr:\n" + _trim(self.stderr, max_chars))
        return "\n".join(parts)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 200]
    tail = text[-200:]
    return f"{head}\n... [truncated {len(text) - limit + 200} chars] ...\n{tail}"
