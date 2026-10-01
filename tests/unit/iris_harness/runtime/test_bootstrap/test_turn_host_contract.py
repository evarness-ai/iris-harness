"""``TurnHost`` must declare exactly the surface the turn stages actually reach.

mypy already enforces one direction: a stage reaching an undeclared member fails the
type check, and ``run_turn(self, ...)`` in ``bootstrap`` is a structural assertion that
``IrisRuntime`` still supplies every declared member. Three mutations confirm it —
adding a 22nd member access, renaming a method off ``IrisRuntime``, and drifting a
declared signature all produce mypy errors.

What mypy cannot see is the *other* direction: a member that stays declared after the
last stage stopped calling it. A stale entry in a published interface is worse than no
interface, because it overstates the coupling the carve has to preserve. So this test
measures both sides independently — the stages' attribute accesses and the protocol's
body, both by AST — and requires them to be equal.

Both sides are read from source rather than from ``TurnHost.__protocol_attrs__``, which
is a CPython internal, and rather than from a hand-written list, which would be a third
thing to keep in sync.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

TURN = Path("src/iris_harness/runtime/turn")
HOST = TURN / "host.py"
STAGE_MODULES = sorted(TURN.glob("stages/*.py")) + [TURN / "pipeline.py"]


class _RuntimeAttributes(ast.NodeVisitor):
    """Every attribute read off a name bound to the stage's ``runtime`` parameter."""

    def __init__(self) -> None:
        self.hits: dict[str, list[int]] = defaultdict(list)
        self._names = {"runtime"}

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self._names:
            self._names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self._names:
            self.hits[node.attr].append(node.lineno)
        self.generic_visit(node)


def _reached() -> dict[str, list[str]]:
    """member -> ["module:line", ...] across every stage module and the pipeline."""
    out: dict[str, list[str]] = defaultdict(list)
    for path in STAGE_MODULES:
        visitor = _RuntimeAttributes()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        for attr, lines in visitor.hits.items():
            out[attr].extend(f"{path.stem}:{line}" for line in lines)
    return out


def _declared() -> set[str]:
    """The members in ``TurnHost``'s body — annotated attributes and methods."""
    tree = ast.parse(HOST.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TurnHost"
    )
    names: set[str] = set()
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
    return names


def test_turn_host_declares_everything_the_stages_reach() -> None:
    """A stage reaching a member TurnHost does not declare. mypy catches this too."""
    undeclared = {name: sites for name, sites in _reached().items() if name not in _declared()}
    assert not undeclared, (
        f"the turn stages reach {sorted(undeclared)} on their host, which TurnHost does "
        "not declare. Reaching further into IrisRuntime widens the pipeline's dependency "
        "on it — declare the member in runtime/turn/host.py so the seam stays readable, "
        f"or use something already on it. Sites: {undeclared}"
    )


def test_turn_host_declares_nothing_the_stages_stopped_using() -> None:
    """The direction mypy is blind to: a declaration outliving its last caller."""
    stale = sorted(_declared() - set(_reached()))
    assert not stale, (
        f"TurnHost declares {stale}, which no stage reaches any more. A stale entry "
        "overstates the coupling the IrisRuntime carve has to preserve — delete it."
    )


def test_the_measured_surface_is_the_size_the_plan_records() -> None:
    """21 members, 9 of them private. Pinned because the number is the finding.

    Slice 6 measured 21 members, 16 private. Track C retires them as it carves: slice 12
    replaced ``_evaluate_prior_turn_outcome`` with the public ``capture`` collaborator,
    slice 13 replaced ``_mission_autocreate_enabled`` and ``_propose_mission`` with
    ``mission_proposals``, slice 15 replaced ``_pre_intercept_activity_hint`` and
    ``_dispatch_intercepts`` with ``intercepts``, slice 16 replaced
    ``_format_recent_context`` and ``_build_memory_context`` with ``sessions``. The tiered
    learning work (ADR-0114 follow-up) added ``replies``, so the intercept stage can
    answer IRIS's own "should I remember this?" question deterministically instead of
    routing a bare "yes" to an agent that never saw the question. Deterministic-path
    parity added ``governance_kernel``, so the ``screen`` stage can fire the kernel's
    PRE_TURN hooks on every turn, and ``response_curator``, so the ``guard`` stage can run
    the model-free response check on a deterministic answer
    (docs/architecture/deterministic-path-parity.md).

    Not a limit — widening the seam is allowed. But the plan doc quotes this count as
    the reason the carve needs an interface first, so a change to it has to reach the
    doc in the same commit.
    """
    declared = _declared()
    private = {name for name in declared if name.startswith("_")}
    assert (len(declared), len(private)) == (21, 9), (
        f"the pipeline's host surface is now {len(declared)} members, {len(private)} "
        "private. Update this pin and the M5.7 slice 6 section of "
        "docs/architecture/OSS-PLUGIN-HARNESS-PLAN.md together."
    )


def test_iris_runtime_satisfies_the_protocol_at_runtime() -> None:
    """The structural check mypy does statically, asserted against the real class.

    mypy checks the ``run_turn(self, ...)`` call sites, which covers the sync and
    streaming paths. This is the same claim without a type checker in the loop: every
    declared member is actually present on ``IrisRuntime`` (or a mixin it inherits).
    """
    from iris_harness.runtime.bootstrap import IrisRuntime

    fields = {f.name for f in IrisRuntime.__dataclass_fields__.values()}
    missing = [
        name
        for name in sorted(_declared())
        if not hasattr(IrisRuntime, name) and name not in fields
    ]
    assert not missing, f"IrisRuntime no longer supplies {missing}, which TurnHost declares"
