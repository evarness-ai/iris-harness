"""Runtime topic constants and typed payloads for cross-domain events.

Empty at MVP. Phase 1+ producers (email-triage, finance-statements,
calendar-sync, tasks) will populate this module with their topic name
constants and frozen-dataclass payloads as those subsystems land.

Convention:
  - Topic names follow ``<domain>.<verb>`` (e.g. ``email.classified``,
    ``finance.statement_ingested``, ``task.due_soon``).
  - Payload classes are frozen dataclasses named in CamelCase plus a
    ``Payload`` suffix (``EmailClassifiedPayload``).
  - Subsystem-private topics (like the coding agent's ``tool_call.*``,
    ``stage.*``, ``persona.*``, ``pipeline.*``) live with their producer
    in that subsystem's own ``events`` module — not here.

See ADR-Q9 (canonical doc §3.4).
"""

from __future__ import annotations
