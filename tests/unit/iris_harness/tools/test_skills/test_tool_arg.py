"""Tests for the ``ToolArg`` manifest model.

The structured ``args:`` block is the source of truth the routine-
authoring flow uses to decide which arguments to ask the user about.
The validators here defend the contract: a malformed manifest entry
must fail at load time, not surface as a confusing question later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.tools.skills.loader import load_skill_manifest
from iris_harness.tools.skills.models import ToolArg

REPO_ROOT = Path(__file__).resolve().parents[5]


def test_web_fetch_manifest_loads_args_block() -> None:
    """The canonical ``web-fetch`` manifest declares the new ``args:``
    block on its ``fetch_web_content`` tool; the loader parses it."""

    manifest = load_skill_manifest(REPO_ROOT / "config" / "skills" / "builtin" / "web-fetch")
    tool = manifest.tools[0]

    assert tool.name == "fetch_web_content"
    arg_names = [arg.name for arg in tool.args]
    assert arg_names == ["type", "category", "limit", "news_group"]

    type_arg, category_arg, limit_arg, group_arg = tool.args
    # The digest's news section (digest v5): optional, never asked for.
    assert (group_arg.type, group_arg.required) == ("string", False)
    assert type_arg.type == "enum"
    assert type_arg.options == ("git", "news", "stocks")
    assert category_arg.type == "enum"
    assert "git-repositories" in category_arg.options
    assert limit_arg.type == "int"
    assert limit_arg.required is False
    assert limit_arg.default == 10
    assert limit_arg.min == 1
    assert limit_arg.max == 50


def test_tool_arg_enum_requires_non_empty_options() -> None:
    with pytest.raises(ValueError, match="non-empty 'options' list"):
        ToolArg(name="x", description="d", type="enum")


def test_tool_arg_options_rejected_on_non_enum() -> None:
    with pytest.raises(ValueError, match="'options' is only valid when type='enum'"):
        ToolArg(name="x", description="d", type="string", options=("a", "b"))


def test_tool_arg_pattern_rejected_on_non_string() -> None:
    with pytest.raises(ValueError, match="'pattern' is only valid when type='string'"):
        ToolArg(name="x", description="d", type="int", pattern=r"^\d+$")


def test_tool_arg_min_max_rejected_on_non_numeric() -> None:
    with pytest.raises(ValueError, match="'min' is only valid"):
        ToolArg(name="x", description="d", type="string", min=1)
    with pytest.raises(ValueError, match="'max' is only valid"):
        ToolArg(name="x", description="d", type="bool", max=1)


def test_tool_arg_default_outside_options_rejected() -> None:
    with pytest.raises(ValueError, match="not in options"):
        ToolArg(
            name="x",
            description="d",
            type="enum",
            options=("a", "b"),
            default="c",
        )


def test_tool_arg_default_below_min_rejected() -> None:
    with pytest.raises(ValueError, match="below min"):
        ToolArg(name="x", description="d", type="int", min=10, default=5)


def test_tool_arg_default_above_max_rejected() -> None:
    with pytest.raises(ValueError, match="above max"):
        ToolArg(name="x", description="d", type="int", max=10, default=15)


def test_tool_arg_default_wrong_type_rejected() -> None:
    with pytest.raises(ValueError, match="default must be an int"):
        ToolArg(name="x", description="d", type="int", default="ten")


def test_tool_arg_invalid_regex_pattern_rejected() -> None:
    with pytest.raises(ValueError, match="not a valid regex"):
        ToolArg(name="x", description="d", type="string", pattern="([abc")


def test_tool_arg_min_greater_than_max_rejected() -> None:
    with pytest.raises(ValueError, match="must be <= 'max'"):
        ToolArg(name="x", description="d", type="int", min=10, max=5)


def test_tool_arg_default_matches_pattern() -> None:
    """A valid default that satisfies the declared pattern is accepted."""

    arg = ToolArg(
        name="symbol",
        description="Ticker.",
        type="string",
        pattern=r"^[A-Z]{1,5}$",
        default="AAPL",
    )
    assert arg.default == "AAPL"


def test_tool_arg_default_pattern_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="does not match pattern"):
        ToolArg(
            name="symbol",
            description="Ticker.",
            type="string",
            pattern=r"^[A-Z]+$",
            default="aapl",
        )


def test_tool_arg_bool_default_accepted() -> None:
    arg = ToolArg(name="enabled", description="d", type="bool", default=True)
    assert arg.default is True


def test_tool_arg_bool_non_bool_default_rejected() -> None:
    with pytest.raises(ValueError, match="default must be a bool"):
        ToolArg(name="enabled", description="d", type="bool", default="yes")
