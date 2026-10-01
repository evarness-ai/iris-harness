"""Self-knowledge is generated from the registry, never hand-written.

SOUL.md carried a prose "Operational primer" naming llama3.2:3b and qwen2.5-coder:7b
as tier 1 and tier 2 while the configured models were granite4 and qwen2.5:7b — and
it rode in every prompt. These tests pin the generated replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iris_harness.runtime.capabilities import capability_line, capability_report


@dataclass
class _Tier:
    name: str
    model: str


@dataclass
class _Record:
    name: str
    status: str = "ok"
    registrations: list[Any] = field(default_factory=list)


@dataclass
class _Tool:
    name: str


class _Registry:
    def __init__(self, records: list[_Record], tools: list[_Tool] | None = None) -> None:
        self._records = records
        self._tools = tools or []

    def plugins(self) -> list[_Record]:
        return list(self._records)

    def tools(self) -> list[_Tool]:
        return list(self._tools)


class _Router:
    def __init__(self, mapping: dict[str, _Tier]) -> None:
        self._mapping = mapping

    def get_tier(self, intent: str) -> _Tier | None:
        return self._mapping.get(intent)


class _Runtime:
    def __init__(self, registry: _Registry | None, router: _Router | None) -> None:
        self.plugin_registry = registry
        self.tier_router = router


def _runtime() -> _Runtime:
    return _Runtime(
        _Registry(
            [
                _Record("email"),
                _Record("finance"),
                _Record("calendar"),
                _Record("broken", "failed"),
            ],
            [_Tool("search_inbox"), _Tool("finance_lookup")],
        ),
        _Router(
            {
                "general": _Tier("tier1", "granite4:latest"),
                "communication": _Tier("tier2", "qwen2.5:7b-instruct"),
            }
        ),
    )


def test_line_names_the_loaded_plugins_and_the_real_models() -> None:
    line = capability_line(_runtime(), ["memory_search", "ask_user"]) or ""

    assert "email" in line and "finance" in line and "calendar" in line
    assert "granite4:latest" in line and "qwen2.5:7b-instruct" in line
    assert "2 tools" in line
    assert 'iris_doc("CAPABILITIES")' in line


def test_a_failed_plugin_is_not_advertised() -> None:
    assert "broken" not in (capability_line(_runtime()) or "")


def test_line_stays_one_line_and_short() -> None:
    line = capability_line(_runtime()) or ""

    assert "\n" not in line
    assert len(line) < 400


def test_no_runtime_state_means_no_line() -> None:
    assert capability_line(_Runtime(None, None)) is None
    assert capability_line(None) is None


def test_a_broken_registry_does_not_break_the_turn() -> None:
    class _Boom:
        def plugins(self) -> list[_Record]:
            raise RuntimeError("registry down")

        def tools(self) -> list[_Tool]:
            return []

    assert capability_line(_Runtime(_Boom(), None)) is None  # type: ignore[arg-type]


def test_report_lists_plugins_models_and_tools() -> None:
    report = capability_report(_runtime(), ["memory_search"])

    assert "granite4:latest" in report
    assert "- email (ok)" in report
    assert "- broken (failed)" in report  # the report is diagnostic, unlike the line
    assert "search_inbox" in report
    assert "memory_search" in report
    assert "shortlisted per turn" in report
