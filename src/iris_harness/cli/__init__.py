"""IRIS CLI package — presentation layer for the IRIS assistant."""

from iris_harness.llm.providers import ProviderManager, ProviderProfile

from .render import console
from .session import Session, SessionManager

__all__ = ["Session", "SessionManager", "ProviderManager", "ProviderProfile", "console"]
