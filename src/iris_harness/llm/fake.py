"""The scripted fake model: a transport that answers from a script instead of a server.

``provider: fake`` is a provider like ``ollama`` or ``lmstudio``, declared ``runs: local``
in the ``providers:`` block of ``llm_tiers.yaml``. It replaces only the *transport*: the
chat model :class:`~iris_harness.llm.client.CodingLLMClient` builds for the call. Every
thing around it -- the governance hooks and their audit rows, the egress log, the span,
token accounting, JSON parsing and its retry -- runs exactly as for a real model, so a
test or a demo on the fake exercises the same governed path a real model does.

``IRIS_LLM_PROVIDER=fake`` puts every tier the router serves on it
(:meth:`TierRouter.load_from_yaml`). The script comes from the process's installed
script (:func:`install_script`, what ``iris_harness.testing.use_fake_model`` does) or,
failing that, the YAML file ``IRIS_FAKE_MODEL_SCRIPT`` names.

A script is data: an ordered list of rules, each a ``match`` (regexes over parts of the
request) and a ``reply``; the first rule that matches answers. Nothing about any domain
is in this module.

.. code-block:: yaml

    rules:
      - name: bill
        match:
          system: "You sort ONE email"          # re.search over the system prompt
          user: 'Amount due: \\$(?P<amount>[0-9.]+)'
        reply:
          json: {bucket: bill, confidence: 0.93, min_due: "{amount}"}
      - name: pick a tool
        match: {tool: search_inbox}             # a tool bound to the call, by name
        reply:
          tool_calls: [{name: search_inbox, arguments: {query: "{amount}"}}]
    default:                                    # optional; without it, no match raises
      content: "I don't know."

Match keys: ``system`` (the system messages), ``user`` (the last user message),
``prompt`` (every message, joined), ``model``, ``tool`` (the name of a bound tool) and
``json`` (``true``: the call asked for a JSON-schema reply). A rule with several keys
needs all of them. A reply is ``content`` (text; a ReAct ``Action:`` block is text too),
``json`` (an object, sent as its JSON text) or ``tool_calls`` (native calls). ``{name}``
in a reply is the named regex group of that name; a string that is exactly ``"{name}"``
becomes a number when the captured text is one.

With a JSON schema on the call (``invoke_json``), the fake plays the constrained decoder
a real server is: a required property the scripted object leaves out is filled with
``null`` when the schema allows null, and is an error when it does not.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from langchain_core.messages import AIMessage, AIMessageChunk

FAKE_PROVIDER = "fake"
SCRIPT_ENV = "IRIS_FAKE_MODEL_SCRIPT"
# The model name the fake reports in its response metadata.
FAKE_MODEL_NAME = "scripted"

_MATCH_KEYS = frozenset({"system", "user", "prompt", "model", "tool", "json"})
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class FakeModelError(ValueError):
    """The script cannot answer this call: no rule matched and there is no default, the
    script is malformed, or a reply does not fit the call's JSON schema."""


@dataclass(frozen=True)
class Reply:
    """What a rule answers: text, one JSON object, or native tool calls."""

    content: str | None = None
    json: Mapping[str, Any] | None = None
    tool_calls: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Reply:
        unknown = set(raw) - {"content", "json", "tool_calls"}
        if unknown:
            raise FakeModelError(f"reply has unknown keys {sorted(unknown)}")
        payload = raw.get("json")
        if payload is not None and not isinstance(payload, Mapping):
            raise FakeModelError("reply json must be an object")
        calls = raw.get("tool_calls") or ()
        if not isinstance(calls, Sequence) or not all(isinstance(c, Mapping) for c in calls):
            raise FakeModelError("reply tool_calls must be a list of {name, arguments}")
        for call in calls:
            if not isinstance(call.get("name"), str) or not call["name"]:
                raise FakeModelError("every scripted tool call needs a name")
        content = raw.get("content")
        if content is not None and not isinstance(content, str):
            raise FakeModelError("reply content must be text")
        if content is None and payload is None and not calls:
            raise FakeModelError("a reply needs content, json or tool_calls")
        return cls(content=content, json=payload, tool_calls=tuple(calls))


@dataclass(frozen=True)
class Rule:
    """One ``match`` -> ``reply`` pair of a script."""

    name: str
    reply: Reply
    patterns: Mapping[str, re.Pattern[str]] = field(default_factory=dict)
    wants_json: bool | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], index: int) -> Rule:
        match = raw.get("match") or {}
        if not isinstance(match, Mapping):
            raise FakeModelError(f"rule {index}: match must be a mapping")
        unknown = set(match) - _MATCH_KEYS
        if unknown:
            raise FakeModelError(f"rule {index}: unknown match keys {sorted(unknown)}")
        reply = raw.get("reply")
        if not isinstance(reply, Mapping):
            raise FakeModelError(f"rule {index}: a rule needs a reply")
        patterns: dict[str, re.Pattern[str]] = {}
        for key, value in match.items():
            if key == "json":
                continue
            try:
                patterns[str(key)] = re.compile(str(value))
            except re.error as exc:
                raise FakeModelError(f"rule {index}: bad {key} regex: {exc}") from exc
        wants_json = match.get("json")
        return cls(
            name=str(raw.get("name") or f"rule {index}"),
            reply=Reply.from_mapping(reply),
            patterns=patterns,
            wants_json=None if wants_json is None else bool(wants_json),
        )


@dataclass(frozen=True)
class Request:
    """The parts of one call a rule can match."""

    system: str
    user: str
    prompt: str
    model: str
    tools: tuple[str, ...]
    json_schema: Mapping[str, Any] | None


@dataclass(frozen=True)
class Script:
    """An ordered list of rules and an optional default reply."""

    rules: tuple[Rule, ...]
    default: Reply | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Script:
        rules = raw.get("rules") or []
        if not isinstance(rules, list):
            raise FakeModelError("script rules must be a list")
        default = raw.get("default")
        return cls(
            rules=tuple(Rule.from_mapping(r, i) for i, r in enumerate(rules, start=1)),
            default=Reply.from_mapping(default) if isinstance(default, Mapping) else None,
        )

    @classmethod
    def load(cls, path: Path | str) -> Script:
        """A script from a YAML file."""
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise FakeModelError(f"cannot read fake-model script {path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise FakeModelError(f"fake-model script {path} must be a mapping")
        return cls.from_mapping(raw)

    def answer(self, request: Request) -> tuple[str, Reply, dict[str, str]]:
        """``(rule name, reply, captured groups)`` for the first matching rule."""
        for rule in self.rules:
            groups = _match(rule, request)
            if groups is not None:
                return rule.name, rule.reply, groups
        if self.default is not None:
            return "default", self.default, {}
        raise FakeModelError(
            "no fake-model rule matches this call and the script has no default; "
            f"user message starts {request.user[:160]!r}"
        )


def _match(rule: Rule, request: Request) -> dict[str, str] | None:
    if rule.wants_json is not None and rule.wants_json != (request.json_schema is not None):
        return None
    groups: dict[str, str] = {}
    for key, pattern in rule.patterns.items():
        if key == "tool":
            hit = next((m for m in (pattern.search(t) for t in request.tools) if m), None)
        else:
            hit = pattern.search(str(getattr(request, key)))
        if hit is None:
            return None
        groups.update({k: v for k, v in hit.groupdict().items() if v is not None})
    return groups


@dataclass(frozen=True)
class FakeCall:
    """One call the fake answered (the transcript a test asserts on)."""

    model: str
    rule: str
    system: str
    user: str
    tools: tuple[str, ...]
    json_schema: bool
    reply: str


_lock = Lock()
_installed: Script | None = None
_transcript: list[FakeCall] = []
# (path, mtime) -> script: the env-named file is parsed once per change.
_file_cache: dict[str, tuple[float, Script]] = {}


def install_script(script: Script | None) -> None:
    """Make ``script`` the process's fake-model script (``None`` removes it)."""
    global _installed
    with _lock:
        _installed = script


def active_script() -> Script:
    """The installed script, else the one ``IRIS_FAKE_MODEL_SCRIPT`` names."""
    with _lock:
        if _installed is not None:
            return _installed
    raw = os.environ.get(SCRIPT_ENV, "").strip()
    if not raw:
        raise FakeModelError(
            f"provider 'fake' has no script: install one (iris_harness.testing) or set {SCRIPT_ENV}"
        )
    path = Path(raw).expanduser()
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        raise FakeModelError(f"{SCRIPT_ENV}={raw!r}: {exc}") from exc
    with _lock:
        cached = _file_cache.get(str(path))
        if cached is not None and cached[0] == mtime:
            return cached[1]
    script = Script.load(path)
    with _lock:
        _file_cache[str(path)] = (mtime, script)
    return script


def transcript() -> tuple[FakeCall, ...]:
    """Every call the fake answered in this process, oldest first."""
    with _lock:
        return tuple(_transcript)


def reset_transcript() -> None:
    with _lock:
        _transcript.clear()


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", "")) if isinstance(p, Mapping) else str(p) for p in content
        )
    return str(content or "")


def _fill(text: str, groups: Mapping[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda m: groups.get(m.group(1), m.group(0)), text)


def _number(text: str) -> int | float | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def _fill_value(value: Any, groups: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        whole = _PLACEHOLDER.fullmatch(value)
        if whole is not None and whole.group(1) in groups:
            number = _number(groups[whole.group(1)])
            return number if number is not None else groups[whole.group(1)]
        return _fill(value, groups)
    if isinstance(value, Mapping):
        return {str(k): _fill_value(v, groups) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_value(v, groups) for v in value]
    return value


def _allows_null(schema: Any) -> bool:
    if not isinstance(schema, Mapping):
        return False
    kind = schema.get("type")
    return kind == "null" or (isinstance(kind, list) and "null" in kind)


def _constrain(data: dict[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    """What a schema-constrained decoder returns for ``data``: required keys present."""
    properties = schema.get("properties") or {}
    out = dict(data)
    for name in schema.get("required") or ():
        if name in out:
            continue
        if _allows_null(properties.get(name)):
            out[name] = None
        else:
            raise FakeModelError(f"scripted JSON reply lacks required property {name!r}")
    return out


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ScriptedChatModel:
    """The chat model ``provider: fake`` builds: ``invoke``, ``stream``, ``bind_tools``."""

    def __init__(
        self,
        *,
        model: str,
        script: Script | None = None,
        json_schema: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.model = model
        self._script = script
        self._json_schema = json_schema
        self._tools = tuple(tools)

    def bind_tools(self, tools: Sequence[Mapping[str, Any]]) -> ScriptedChatModel:
        return ScriptedChatModel(
            model=self.model, script=self._script, json_schema=self._json_schema, tools=tools
        )

    def _tool_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for tool in self._tools:
            function = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
            name = function.get("name") if isinstance(function, Mapping) else None
            if isinstance(name, str):
                names.append(name)
        return tuple(names)

    def _answer(self, messages: Sequence[Any]) -> tuple[str, str, list[dict[str, Any]], int]:
        """``(content, rule, tool calls, prompt tokens)`` for one call."""
        system: list[str] = []
        users: list[str] = []
        everything: list[str] = []
        for message in messages:
            role = message.get("role") if isinstance(message, Mapping) else None
            content = _text(message.get("content") if isinstance(message, Mapping) else message)
            everything.append(content)
            if role == "system":
                system.append(content)
            elif role == "user":
                users.append(content)
        request = Request(
            system="\n".join(system),
            user=users[-1] if users else "",
            prompt="\n".join(everything),
            model=self.model,
            tools=self._tool_names(),
            json_schema=self._json_schema,
        )
        script = self._script or active_script()
        rule, reply, groups = script.answer(request)
        calls = [
            {
                "name": str(call["name"]),
                "args": _fill_value(dict(call.get("arguments") or {}), groups),
                "id": f"fake-call-{index}",
                "type": "tool_call",
            }
            for index, call in enumerate(reply.tool_calls, start=1)
        ]
        if reply.json is not None:
            data = _fill_value(dict(reply.json), groups)
            if self._json_schema is not None:
                data = _constrain(data, self._json_schema)
            content = json.dumps(data, sort_keys=True)
        else:
            content = _fill(reply.content or "", groups)
        with _lock:
            _transcript.append(
                FakeCall(
                    model=self.model,
                    rule=rule,
                    system=request.system,
                    user=request.user,
                    tools=request.tools,
                    json_schema=self._json_schema is not None,
                    reply=content if not calls else json.dumps(calls, sort_keys=True),
                )
            )
        return content, rule, calls, _tokens(request.prompt)

    def invoke(self, messages: Sequence[Any], **_: Any) -> AIMessage:
        content, rule, calls, prompt_tokens = self._answer(messages)
        completion = _tokens(content)
        return AIMessage(
            content=content,
            tool_calls=calls,
            response_metadata={"model_name": FAKE_MODEL_NAME, "rule": rule},
            usage_metadata={
                "input_tokens": prompt_tokens,
                "output_tokens": completion,
                "total_tokens": prompt_tokens + completion,
            },
        )

    def stream(self, messages: Sequence[Any], **_: Any) -> Iterator[AIMessageChunk]:
        content, _rule, _calls, prompt_tokens = self._answer(messages)
        pieces = re.findall(r"\S+\s*|\s+", content) or [""]
        for piece in pieces[:-1]:
            yield AIMessageChunk(content=piece)
        completion = _tokens(content)
        yield AIMessageChunk(
            content=pieces[-1],
            usage_metadata={
                "input_tokens": prompt_tokens,
                "output_tokens": completion,
                "total_tokens": prompt_tokens + completion,
            },
        )


def chat_model_factory(**kwargs: Any) -> ScriptedChatModel:
    """The ``provider: fake`` branch of the client's model factory. ``format`` is the
    call's JSON schema; every transport option a server would take is ignored."""
    schema = kwargs.get("format")
    return ScriptedChatModel(
        model=str(kwargs.get("model") or FAKE_MODEL_NAME),
        json_schema=schema if isinstance(schema, Mapping) else None,
    )


__all__ = [
    "FAKE_MODEL_NAME",
    "FAKE_PROVIDER",
    "SCRIPT_ENV",
    "FakeCall",
    "FakeModelError",
    "Reply",
    "Request",
    "Rule",
    "Script",
    "ScriptedChatModel",
    "active_script",
    "chat_model_factory",
    "install_script",
    "reset_transcript",
    "transcript",
]
