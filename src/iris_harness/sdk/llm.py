"""Reaching the harness's governed model access.

A plugin never builds an LLM client by hand: `tier_router.get_llm_config(intent)`
is the one place that decides which tier a call runs on, and a client built
around it is governed by construction. The error formatter is here for the same
reason -- a plugin that catches a provider failure should show the user what the
harness would have shown, not its own wording.

`make_narrative_llm_call(tier_router, intent)` is the one-line way to turn facts into
a sentence on the intent's governed tier. `embed_corpus` embeds texts locally
(`EMBED_MODEL_DEFAULT`, no egress). `TierConfig` is what `get_llm_config` returns,
`governance_tier_for_intent` maps an intent's tier to the governor's tier label for a
plugin that runs its own governed loop, and `provider_root_url` is a provider's
server root for a reachability probe that must not load a model.

`CodingLLMClient.invoke_json(system_prompt=, user_prompt=, schema=)` is the governed
structured call (a classifier, a judge): the reply is one JSON object fitting the
schema, returned as a `JsonReply`. It raises `LLMUnreachable` when the model server is
down (stop the batch and try later) and `LLMBadReply` when it answers twice with no
JSON object (skip that item). Only on a provider that `supports_json_schema` (Ollama,
or the scripted fake).

`FORCED_PROVIDER_ENV` (``IRIS_LLM_PROVIDER``) puts every tier on one declared provider;
with `FAKE_PROVIDER` it runs everything on the scripted fake model, whose script is the
YAML file `FAKE_MODEL_SCRIPT_ENV` names. A plugin that ships a demo selects the fake
through these, never by importing it; a test uses ``iris_harness.testing``.
"""

from __future__ import annotations

from iris_harness.llm.client import (
    CodingLLMClient,
    CodingLLMConfig,
    JsonReply,
    LLMBadReply,
    LLMUnreachable,
    supports_json_schema,
)
from iris_harness.llm.embeddings import EMBED_MODEL_DEFAULT, embed_corpus
from iris_harness.llm.errors import friendly_llm_error
from iris_harness.llm.fake import FAKE_PROVIDER
from iris_harness.llm.fake import SCRIPT_ENV as FAKE_MODEL_SCRIPT_ENV
from iris_harness.llm.narrate import make_narrative_llm_call
from iris_harness.llm.tier_router import (
    FORCED_PROVIDER_ENV,
    ModelTier,
    TierConfig,
    TierRouter,
    governance_tier_for_intent,
    model_tier_for,
    provider_root_url,
)

__all__ = [
    "EMBED_MODEL_DEFAULT",
    "FAKE_MODEL_SCRIPT_ENV",
    "FAKE_PROVIDER",
    "FORCED_PROVIDER_ENV",
    "CodingLLMClient",
    "CodingLLMConfig",
    "JsonReply",
    "LLMBadReply",
    "LLMUnreachable",
    "ModelTier",
    "TierConfig",
    "TierRouter",
    "embed_corpus",
    "friendly_llm_error",
    "governance_tier_for_intent",
    "make_narrative_llm_call",
    "model_tier_for",
    "provider_root_url",
    "supports_json_schema",
]
