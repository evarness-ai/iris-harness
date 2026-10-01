"""What each ``HarnessServices`` handle promises a plugin.

``HarnessServices`` hands a plugin the harness's tier router, agent executor,
heartbeat scheduler, channel gateway, event bus, lesson store, continuation
registry and skill registry. These Protocols say which methods of each a plugin may
call -- only the ones plugins do call today -- so an author codes against a stated
surface rather than whatever the harness object happens to expose, and a test fake
has to supply only those methods.

Re-exports: the Protocols are defined beside ``HarnessServices`` in the runtime,
because the composition root types its fields with them and may not import the
layer above it.
"""

from __future__ import annotations

from iris_harness.runtime.harness_services import (
    AgentExecutorService,
    ChannelService,
    ContinuationService,
    EventBusService,
    HeartbeatService,
    LessonService,
    SkillRegistryService,
    TierRouterService,
)

__all__ = [
    "AgentExecutorService",
    "ChannelService",
    "ContinuationService",
    "EventBusService",
    "HeartbeatService",
    "LessonService",
    "SkillRegistryService",
    "TierRouterService",
]
