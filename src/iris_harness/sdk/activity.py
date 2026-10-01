"""Whether the owner is chatting right now, for background work that should wait.

A plugin's background job (an embedding pass, a batch of model calls) competes with a
chat turn for the same local model. ``chat_in_progress()`` is true while a turn runs
and for a short quiet period after, so the job can yield instead of slowing the
answer the owner is waiting for.
"""

from __future__ import annotations

from iris_harness.foundation.activity import chat_in_progress

__all__ = ["chat_in_progress"]
