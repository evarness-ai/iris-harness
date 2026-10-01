"""__tmpl_title: one tool on IRIS's governed ReAct loop.

``setup(api)`` registers the tool. ``manifest.yaml`` beside this file declares it under
``tools:`` with its effect -- a tool the manifest does not declare is refused when the
plugin mounts (ADR-0110). The loop runs every call through the governance kernel
(``PRE_TOOL_USE``, the approval rules, ``POST_TOOL_USE``) and writes an audit row for it.

Replace :func:`count_words` with what your tool does, and keep the manifest honest: a
tool that changes something on the owner's behalf is ``effect: write``; one that
returns text a third party wrote (a web page, an email) is ``content: external``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from iris_harness.sdk import PluginAPI

TOOL = "__tmpl_tool"
DESCRIPTION = 'Count the words, sentences and characters in a piece of text. Args: {"text": str}.'

_WORD = re.compile(r"\b\w+\b")
_SENTENCE_END = re.compile(r"[.!?]+")


def count_words(text: str) -> dict[str, int]:
    """The tool's work: plain Python, testable without IRIS."""
    sentences = [part for part in _SENTENCE_END.split(text) if part.strip()]
    return {
        "words": len(_WORD.findall(text)),
        "sentences": len(sentences),
        "characters": len(text),
    }


def run(args: dict[str, Any]) -> str:
    """The loop calls this with the model's arguments; what it returns is the observation
    the model reads. Return a readable error instead of raising on bad arguments."""
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        return 'error: pass the text to count, as {"text": "..."}'
    return json.dumps(count_words(text))


def setup(api: PluginAPI) -> None:
    api.register_tool(TOOL, DESCRIPTION, run)
