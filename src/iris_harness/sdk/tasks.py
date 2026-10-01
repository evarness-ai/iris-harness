"""The owner's tasks: the one store every agent raises work into.

A plugin that turns something it saw into work for the owner writes a `Task` to the
`TaskStore` (``data/tasks.db``), tagged with its `SourceKind` and, when the owner can
act on it in one tap, a `TaskAction` whose `ActionCard` explains why. A `WaitFor` parks
a task until a reply, an event or the owner resolves it. `short_title` is the one
trim every surface uses, so a task reads the same in the digest, the Action Center
and a reminder. Subscribe to `TASK_COMPLETED` (a process-bus topic) to react when the
owner finishes one.
"""

from __future__ import annotations

from iris_harness.services.tasks.brief_view import short_title
from iris_harness.services.tasks.events import TASK_COMPLETED
from iris_harness.services.tasks.models import (
    ActionCard,
    ActionChoice,
    ActionEvidence,
    ActionFact,
    ActionOptions,
    ActionOptionValue,
    SourceKind,
    Task,
    TaskAction,
    WaitFor,
)
from iris_harness.services.tasks.store import TaskStore

__all__ = [
    "TASK_COMPLETED",
    "ActionCard",
    "ActionChoice",
    "ActionEvidence",
    "ActionFact",
    "ActionOptionValue",
    "ActionOptions",
    "SourceKind",
    "Task",
    "TaskAction",
    "TaskStore",
    "WaitFor",
    "short_title",
]
