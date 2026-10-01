"""
Heartbeat System Implementation

This module implements the heartbeat system as described in the PRD section 7.2.
"""

import logging
import time

logger = logging.getLogger(__name__)


class Heartbeat:
    """
    A class to represent the heartbeat system.

    Attributes:
    - interval (int): The interval in seconds between heartbeats.
    - last_beat (Optional[float]): The timestamp of the last heartbeat.
    """

    def __init__(self, interval: int) -> None:
        """
        Initialize the Heartbeat system with a specified interval.

        Args:
        - interval (int): The interval in seconds between heartbeats.
        """
        self.interval = interval
        self.last_beat: float | None = None

    def beat(self) -> None:
        """
        Record a heartbeat event and log it at debug level.
        """
        self.last_beat = time.time()
        logger.debug("heartbeat recorded at %s", self.last_beat)

    def time_since_last_beat(self) -> float | None:
        """
        Calculate the time elapsed since the last heartbeat.

        Returns:
        - Optional[float]: Time in seconds since the last heartbeat, or None if no heartbeat has occurred.
        """
        if self.last_beat is None:
            return None
        return time.time() - self.last_beat

    def is_heartbeat_due(self) -> bool:
        """
        Check if a heartbeat is due based on the interval.

        Returns:
        - bool: True if a heartbeat is due, False otherwise.
        """
        if self.last_beat is None:
            return True
        return (time.time() - self.last_beat) >= self.interval


# Example usage
if __name__ == "__main__":
    heartbeat = Heartbeat(interval=5)
    while True:
        if heartbeat.is_heartbeat_due():
            heartbeat.beat()
        time.sleep(1)
