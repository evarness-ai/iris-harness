"""The three env-flag semantics, pinned so they cannot drift apart again.

M6.3 found SIX copies of this logic in three variants that disagreed about
`FLAG=` (set but empty) and about `FLAG=banana`. The owner settled the first:
**an empty value is OFF, everywhere.** The second still separates the two
functions, which answer different questions on purpose.

These tests state what the behaviours are, so the next person to "just tidy this
up" has to change a test and say why.
"""

from __future__ import annotations

import pytest

from iris_harness.foundation.env import env_flag, env_flag_on, probe_host


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", " off ", "Off", "", "  "])
def test_the_off_values_are_false_to_both(monkeypatch, value: str) -> None:
    """The empty string is in this list now — that is the settled behaviour."""
    monkeypatch.setenv("IRIS_TEST_FLAG", value)
    assert env_flag("IRIS_TEST_FLAG", default=True) is False
    assert env_flag_on("IRIS_TEST_FLAG") is False


def test_unset_returns_the_default(monkeypatch) -> None:
    monkeypatch.delenv("IRIS_TEST_FLAG", raising=False)
    assert env_flag("IRIS_TEST_FLAG", default=True) is True
    assert env_flag("IRIS_TEST_FLAG", default=False) is False
    # env_flag_on has no default: absent is not "explicitly on"
    assert env_flag_on("IRIS_TEST_FLAG") is False


def test_empty_is_off_and_does_not_fall_back_to_the_default(monkeypatch) -> None:
    """`export FLAG=` — the divergence that turned a feature on in the runtime and
    off in the API, settled as OFF.

    And settled as *off*, not as *unset*: a blank value overrides a `default=True`
    rather than falling through to it, which is what an operator blanking a var
    means. Four flags moved onto this reading
    (IRIS_SYNC_MEMORY_BLOCKING, IRIS_RUNTIME_TELEGRAM_POLLER_ENABLED,
    IRIS_CHANNEL_GATEWAY_TELEGRAM_ENABLED, IRIS_CHANNEL_GATEWAY_TRACING_ENABLED).
    """
    monkeypatch.setenv("IRIS_TEST_FLAG", "")
    assert env_flag("IRIS_TEST_FLAG", default=False) is False
    assert env_flag("IRIS_TEST_FLAG", default=True) is False
    assert env_flag_on("IRIS_TEST_FLAG") is False


def test_an_unrecognised_value_still_separates_the_two(monkeypatch) -> None:
    """`FLAG=banana` — true to `env_flag`, false to `env_flag_on`.

    The remaining divergence, and deliberate: a typo'd value silently enables a
    feature under one reading and silently does nothing under the other, so the
    choice of function is a real decision at each call site.
    """
    monkeypatch.setenv("IRIS_TEST_FLAG", "banana")
    assert env_flag("IRIS_TEST_FLAG", default=False) is True
    assert env_flag_on("IRIS_TEST_FLAG") is False


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_the_allow_list_values_are_true_either_way(monkeypatch, value: str) -> None:
    monkeypatch.setenv("IRIS_TEST_FLAG", value)
    assert env_flag("IRIS_TEST_FLAG", default=False) is True
    assert env_flag_on("IRIS_TEST_FLAG") is True


def test_there_is_only_one_implementation_left() -> None:
    """The duplication was the defect; this is what stops it coming back.

    Four thin aliases remain (`_env_flag` in two server apps and two observability
    modules) and each delegates here. A new hand-rolled copy is how three variants
    happened in the first place.
    """
    import ast

    from iris_harness.foundation.paths import repo_root

    hand_rolled: list[str] = []
    for path in (repo_root() / "src").rglob("*.py"):
        if path.name == "env.py" and path.parent.name == "foundation":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if "env_flag" not in node.name:
                continue
            body = ast.dump(node)
            # a delegating alias mentions env_flag(...); a re-implementation reads os
            if "getenv" in body or "environ" in body:
                hand_rolled.append(f"{path.relative_to(repo_root())}:{node.name}")
    assert not hand_rolled, (
        "these re-read the environment instead of delegating to "
        "iris_harness.foundation.env:\n  " + "\n  ".join(hand_rolled)
    )


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0.0.0.0", "127.0.0.1"),  # noqa: S104 - the wildcard address under test
        ("::", "[::1]"),
        ("127.0.0.1", "127.0.0.1"),
        ("example.internal", "example.internal"),
    ],
)
def test_probe_host_translates_wildcard_binds_only(host: str, expected: str) -> None:
    """``iris serve --host 0.0.0.0`` binds every interface, but nothing answers a
    probe sent to 0.0.0.0 itself — a wildcard bind becomes its loopback
    equivalent; any other host is returned unchanged."""
    assert probe_host(host) == expected
