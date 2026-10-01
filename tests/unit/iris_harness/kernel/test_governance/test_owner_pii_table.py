"""The kind x guard action table and the decision function over it (ADR-0125, PR 3).

``config/governance/identity.yaml`` ``guards:`` holds the owner's decided cells
(2026-09-30). The loader types every column, so a guard cannot be given an action it
cannot take, and validates the table as a whole. ``decide`` turns a text into
``(span, kind, action)`` for one guard and audience.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from iris_harness.kernel.governance.identity_config import (
    GUARD_COLUMNS,
    GuardTable,
    IdentityConfig,
    guard_table,
    load_identity_config,
)
from iris_harness.kernel.governance.owner_identity import KINDS, declared, extract, merge
from iris_harness.kernel.governance.owner_pii import column_for, decide

SECRET = "CANARY_SOUL_SECRET_DIRECTIVE_d41f8a27"
LINK = "www.web3notes.example"
IDENTITY = merge(
    extract([f"key {SECRET} blog {LINK}"]),
    declared(
        {
            "name": ["Robin Example", "Robin"],
            "email": ["owner.canary@example.com"],
            "phone": ["+1 555 0100 0199"],
            "address": ["1 Example Street, Springfield"],
            "handle": ["robin-gh"],
        }
    ),
)
SAMPLES = {
    "secret": SECRET,
    "link": LINK,
    "name": "Robin Example",
    "email": "owner.canary@example.com",
    "phone": "+1 555 0100 0199",
    "address": "1 Example Street, Springfield",
    "handle": "@robin-gh",
}

# The owner's decided cells (ADR-0125 amendment 6), written out once more by hand.
DECIDED = {
    "secret": ("deny", "halt", "halt", "mask", "placeholder", "deny"),
    "link": ("deny", "pass", "pass", "pass", "pass", "deny"),
    "name": ("log", "pass", "mask", "pseudonym", "pass", "mask"),
    "email": ("deny", "pass", "mask", "pseudonym", "placeholder", "deny"),
    "phone": ("deny", "pass", "mask", "pseudonym", "placeholder", "deny"),
    "address": ("deny", "pass", "mask", "pseudonym", "placeholder", "deny"),
    "handle": ("log", "pass", "mask", "pseudonym", "pass", "mask"),
}
FIRST_NAME_ALONE = ("log", "pass", "pass", "pass", "pass", "mask")


@pytest.fixture(scope="module")
def table() -> GuardTable:
    loaded = guard_table()
    assert loaded is not None, "identity.yaml ships no guards table"
    return loaded


def _raw() -> dict[str, Any]:
    raw = yaml.safe_load(
        Path(__file__).parents[5].joinpath("config/governance/identity.yaml").read_text()
    )
    assert isinstance(raw, dict)
    return raw


def _load(tmp_path: Path, raw: dict[str, Any]) -> IdentityConfig:
    path = tmp_path / "identity.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return IdentityConfig.from_yaml(path)


# -- the shipped table -------------------------------------------------------------------


def test_the_shipped_table_is_the_owners_decision(table: GuardTable) -> None:
    for kind, cells in DECIDED.items():
        assert tuple(table.kinds[kind].of(c) for c in GUARD_COLUMNS) == cells, kind
    assert tuple(table.first_name_alone.of(c) for c in GUARD_COLUMNS) == FIRST_NAME_ALONE
    assert set(table.log_only_destinations.tools) == {"github_*", "mcp_*"}
    assert table.grantable() == {"name", "email", "phone", "address", "handle"}


def test_blog_and_website_are_links() -> None:
    kinds = load_identity_config().ontology_kinds
    assert kinds["blog"] == "link" and kinds["website"] == "link"


def test_the_repo_file_is_the_one_loaded(table: GuardTable) -> None:
    assert GuardTable.model_validate(_raw()["guards"]) == table


# -- validation --------------------------------------------------------------------------


def _mutated(path: list[str], value: Any) -> dict[str, Any]:
    raw = copy.deepcopy(_raw())
    node = raw["guards"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return raw


@pytest.mark.parametrize(
    ("path", "value", "why"),
    [
        (["kinds", "email", "egress"], "halt", "egress cannot halt"),
        (["kinds", "email", "tier3"], "mask", "tier3 only placeholder/pass"),
        (["kinds", "email", "capability"], "deny", "capability cannot deny"),
        (["kinds", "email", "answer_owner"], "pseudonym", "answers are not pseudonymised"),
        (["kinds", "secret", "capability"], "pseudonym", "secret"),
        (["kinds", "secret", "capability"], "pass", "secret"),
        (["kinds", "link", "capability"], "pseudonym", "cannot be a pseudonym"),
        (["kinds", "phone", "answer_owner"], "halt", "cannot halt an answer to the owner"),
        (["kinds", "name", "answer_owner"], "halt", "cannot halt an answer to the owner"),
        (["first_name_alone", "egress"], "deny", "never denies or halts"),
        (["first_name_alone", "answer_other"], "halt", "never denies or halts"),
        (["first_name_alone", "web_search"], "deny", "never denies or halts"),
        (["log_only_destinations", "kinds"], ["secret"], "cannot relax"),
        (["log_only_destinations", "kinds"], ["link"], "cannot relax"),
        (["kinds", "email", "shouting"], "deny", "extra"),
    ],
)
def test_a_table_that_breaks_a_rule_does_not_load(
    tmp_path: Path, path: list[str], value: Any, why: str
) -> None:
    with pytest.raises(ValueError):
        _load(tmp_path, _mutated(path, value))


def test_every_kind_needs_a_row(tmp_path: Path) -> None:
    raw = copy.deepcopy(_raw())
    del raw["guards"]["kinds"]["handle"]
    with pytest.raises(ValueError, match="no row for handle"):
        _load(tmp_path, raw)


def test_every_guard_needs_a_cell(tmp_path: Path) -> None:
    raw = copy.deepcopy(_raw())
    del raw["guards"]["kinds"]["email"]["web_search"]
    with pytest.raises(ValueError):
        _load(tmp_path, raw)


def test_a_file_without_guards_has_no_table(tmp_path: Path) -> None:
    assert _load(tmp_path, {"ontology_kinds": {"name": "name"}}).guards is None
    assert guard_table(tmp_path / "absent.yaml") is None


def test_the_table_is_reread_when_the_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "identity.yaml"
    raw = _raw()
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert guard_table(path) is not None
    path.write_text(yaml.safe_dump({"ontology_kinds": {}}) + "\n# changed\n", encoding="utf-8")
    assert guard_table(path) is None


# -- decide: every guard x kind x audience -----------------------------------------------


def _one(text: str, table: GuardTable, **kw: Any) -> tuple[str, str]:
    (d,) = decide(text, identity=IDENTITY, table=table, **kw)
    return d.kind, d.action


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("guard", "audience", "column"),
    [
        ("egress", "owner", 0),
        ("answer", "owner", 1),
        ("answer", "other", 2),
        ("capability", "owner", 3),
        ("tier3", "owner", 4),
        ("web_search", "owner", 5),
    ],
)
def test_decide_reads_the_cell_for_each_guard_kind_and_audience(
    table: GuardTable, kind: str, guard: Any, audience: Any, column: int
) -> None:
    text = f"note: {SAMPLES[kind]} ok"
    assert _one(text, table, guard=guard, audience=audience) == (kind, DECIDED[kind][column])


@pytest.mark.parametrize(
    ("guard", "audience", "action"),
    [
        ("egress", "owner", "log"),
        ("answer", "owner", "pass"),
        ("answer", "other", "pass"),
        ("capability", "owner", "pass"),
        ("tier3", "owner", "pass"),
        ("web_search", "owner", "mask"),
    ],
)
def test_a_first_name_alone_is_masked_only_in_web_search(
    table: GuardTable, guard: Any, audience: Any, action: str
) -> None:
    decisions = decide(
        "ask Robin later", guard=guard, audience=audience, identity=IDENTITY, table=table
    )
    assert [(d.kind, d.literal, d.action) for d in decisions] == [("name", "Robin", action)]
    assert not any(d.blocks for d in decisions)


def test_a_first_name_alone_never_blocks_anywhere(table: GuardTable) -> None:
    for guard, audience in (
        ("egress", "owner"),
        ("answer", "owner"),
        ("answer", "other"),
        ("capability", "owner"),
        ("tier3", "owner"),
        ("web_search", "owner"),
    ):
        decisions = decide("Robin", guard=guard, audience=audience, identity=IDENTITY, table=table)
        assert decisions and not any(d.blocks for d in decisions), guard


def test_the_full_name_is_not_a_first_name(table: GuardTable) -> None:
    (d,) = decide("Robin Example", guard="answer", audience="other", identity=IDENTITY, table=table)
    assert (d.literal, d.action) == ("Robin Example", "mask")


@pytest.mark.parametrize("tool", ["github_create_issue", "mcp_notes_write"])
def test_log_only_destinations_log_pii_but_still_deny_secrets(table: GuardTable, tool: str) -> None:
    text = f"owner.canary@example.com {SECRET} {LINK} +1 555 0100 0199"
    got = {
        d.kind: d.action
        for d in decide(text, guard="egress", identity=IDENTITY, table=table, destination=tool)
    }
    assert got == {"email": "log", "phone": "log", "secret": "deny", "link": "deny"}
    other = decide(text, guard="egress", identity=IDENTITY, table=table, destination="research")
    assert {d.kind: d.action for d in other}["email"] == "deny"


def test_the_stricter_action_keeps_an_overlapping_span(table: GuardTable) -> None:
    """A link that contains the owner's handle: a ``pass`` link never hides the handle."""
    identity = declared({"link": ["blog.example/robin-gh/2026"], "handle": ["robin-gh"]})
    text = "read blog.example/robin-gh/2026 today"

    def got(guard: Any, audience: Any = "owner") -> list[tuple[str, str]]:
        return [
            (d.kind, d.action)
            for d in decide(text, guard=guard, audience=audience, identity=identity, table=table)
        ]

    assert got("capability") == [("handle", "pseudonym")]  # link: pass
    assert got("answer", "other") == [("handle", "mask")]  # link: pass
    assert got("web_search") == [("link", "deny")]  # deny beats the handle's mask
    assert got("egress") == [("link", "deny")]  # deny beats the handle's log


def test_pass_occurrences_are_reported_for_audit(table: GuardTable) -> None:
    decisions = decide(
        "mail owner.canary@example.com", guard="answer", identity=IDENTITY, table=table
    )
    assert [(d.kind, d.action) for d in decisions] == [("email", "pass")]


def test_column_for() -> None:
    assert column_for("answer") == "answer_owner"
    assert column_for("answer", "other") == "answer_other"
    assert column_for("tier3", "other") == "tier3"
