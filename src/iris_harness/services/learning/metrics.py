from __future__ import annotations

from typing import Any


class LearningMetricsCollector:
    """
    Collects and records metrics related to learning processes, such as execution outcomes,
    latency, and success/failure signals.
    """

    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []

    def record_metric(self, outcome: str, latency: float, success: bool) -> None:
        """Records a metric with the given outcome, latency, and success status."""
        self.metrics.append({"outcome": outcome, "latency": latency, "success": success})

    def get_metrics(self) -> list[dict[str, Any]]:
        """Retrieves all recorded metrics."""
        return self.metrics

    def clear_metrics(self) -> None:
        """Clears all recorded metrics."""
        self.metrics.clear()
