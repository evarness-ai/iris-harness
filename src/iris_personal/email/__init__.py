"""Email ACCESS — the capability every agent reaches for (OSS plan M3).

Reading the mailbox is core, the way file search/read/write is: the planner's daily
plan, the finance ingest path and the ReAct email tools all query ``EmailStore``, and
the sweep heartbeat is what keeps it current.

What is *not* here any more is the workflows built on top of that access — triage,
category discovery, kNN-gate measurement, holdout labelling, followup detection and the
email→wiki translator. They live in ``iris_personal.plugins.email_workflows``.

``subscribe_email_triage`` used to be re-exported from this package, which made the core
import the classifier; that re-export is gone, and the plugin owns the subscription.

Since M5.7 track A the mailbox itself is a plugin too: ``gmail_fetch``,
``gmail_attachments`` and Gmail OAuth live in ``plugins_builtin.gmail``, behind the
``MailProvider`` interface and registry in :mod:`providers`. The sweep here is the
core mechanism that drives whichever provider is registered.
"""

from .store import EmailStore
from .sweep import EmailSweepHandler, build_email_sweep_handler

__all__ = [
    "EmailStore",
    "EmailSweepHandler",
    "build_email_sweep_handler",
]
