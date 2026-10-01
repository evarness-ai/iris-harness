"""The changed-scope gate must never earn a green line it did not run (OSS plan M6, decision 9).

Two properties are asserted by running the real script with ``CHANGED_OVERRIDE``:

* a source path with no test home escalates to the full suite instead of reporting
  "nothing to run";
* every full-suite trigger regex still matches a tracked file, so a tree move that
  leaves a stale trigger behind fails here rather than silently ceasing to fire.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
# Every subprocess below runs our own literal script or `git` with a pinned PATH, hence
# the S603/S607 waivers (waived per file in pyproject, as for tests/integration).
SCRIPT = ROOT / "scripts" / "changed_tests.sh"
TRIGGERS = ROOT / "scripts" / "changed_tests_full_triggers.txt"


def _plan(*paths: str) -> str:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--print"],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "CHANGED_OVERRIDE": "\n".join(paths),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


# git's own variables stripped: a git hook (the pre-push gate runs this suite) exports
# GIT_DIR / GIT_INDEX_FILE, which would point the throwaway index below at the repo's.
_GIT_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _is_git_checkout() -> bool:
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        env=_GIT_ENV,
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "true"


@pytest.fixture(scope="module")
def tracked_files(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """What git tracks in this tree.

    In a checkout, ``git ls-files``. In a tree that is not one (the public export before
    its first commit, a source tarball), a throwaway repository indexes the tree in place
    -- its git dir is temporary, ``--work-tree`` points here, and ``add --intent-to-add``
    records paths without copying a byte -- so ``.gitignore`` decides what counts as
    tracked, exactly as it would on the first commit.
    """
    if _is_git_checkout():
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, env=_GIT_ENV, capture_output=True, text=True, check=True
        ).stdout
        return out.splitlines()
    git_dir = tmp_path_factory.mktemp("index") / ".git"
    git = ["git", f"--git-dir={git_dir}", f"--work-tree={ROOT}"]
    for step in (["init", "-q"], ["add", "--intent-to-add", "."]):
        subprocess.run([*git, *step], cwd=ROOT, env=_GIT_ENV, capture_output=True, check=True)
    out = subprocess.run(
        [*git, "ls-files"], cwd=ROOT, env=_GIT_ENV, capture_output=True, text=True, check=True
    ).stdout
    return out.splitlines()


def test_source_path_with_no_test_home_escalates_to_full_suite() -> None:
    plan = _plan("src/iris_harness/no_such_package/module.py")
    assert "full suite" in plan
    assert "nothing to run" not in plan


def test_a_root_dir_that_only_contains_layers_is_not_a_test_home() -> None:
    """tests/unit/iris_harness/ exists (M6.2 put foundation under it) but answers for
    nothing on its own: most of the core has no mirrored home yet, so a path that walks
    all the way up to the root must escalate rather than run the one layer that moved."""
    plan = _plan("src/iris_harness/no_such_package/module.py")
    assert "full suite" in plan
    assert "tests/unit/iris_harness\n" not in plan


@pytest.mark.parametrize(
    ("root", "module"),
    [
        ("memris", "src/memris/graph.py"),
        # The coding agent's root, where this tree carries it (not the public export).
        pytest.param(
            "iris_code",
            "src/iris_code/agent.py",
            marks=pytest.mark.skipif(
                not (ROOT / "tests" / "unit" / "iris_code").is_dir(),
                reason="tests/unit/iris_code is not in this tree",
            ),
        ),
    ],
)
def test_a_root_dir_with_its_own_tests_is_a_test_home(root: str, module: str) -> None:
    """tests/unit/memris/ (and tests/unit/iris_code/) hold test files directly, so each
    answers for its root."""
    plan = _plan(module)
    assert f"tests/unit/{root}" in plan
    assert "full suite" not in plan


def test_a_moved_layer_maps_to_its_mirrored_tests() -> None:
    plan = _plan("src/iris_harness/foundation/persistence/sqlite.py")
    assert "tests/unit/iris_harness/foundation" in plan
    assert "full suite" not in plan


def test_a_cli_command_module_maps_to_the_cli_tests() -> None:
    """src/iris_harness/cli/ mirrors to a `cli/` test dir that does not exist; the CLI
    tests live in `test_cli/`. Without the rule a new command module escalated too."""
    plan = _plan("src/iris_harness/cli/approvals.py")
    assert _targets(plan) == ["tests/unit/iris_harness/test_cli"]
    assert "full suite" not in plan


def test_the_cli_entry_point_maps_to_the_cli_tests() -> None:
    """src/iris_harness/main.py is the Typer wiring: its tests are the CLI tests.

    It mirrors to the container root (which answers for nothing), so without its own
    rule every new subcommand escalated the pre-push gate to the full suite.
    """
    plan = _plan("src/iris_harness/main.py")
    assert _targets(plan) == ["tests/unit/iris_harness/test_cli"]
    assert "full suite" not in plan


def test_mirror_rule_walks_up_to_the_nearest_test_directory(tmp_path: Path) -> None:
    # tests/unit/test_scripts exists, so a (hypothetical) src/test_scripts/deep/x.py maps
    # to it by the mirror rule alone — no flat-convention fallback is involved.
    plan = _plan("src/test_scripts/deep/nested/x.py")
    assert "tests/unit/test_scripts" in plan
    assert "full suite" not in plan


def test_the_flat_convention_has_no_subjects_left() -> None:
    """M6.3 folded `identity/` into `memory/`, and with it the last flat-tested package.

    This replaces `test_flat_convention_still_maps_until_the_move_lands`, whose own
    docstring said to delete it when that happened. The fallback still exists in the
    script and is still correct -- it just has nothing under `src/iris_harness/` to
    catch any more, and asserting that is what stops someone re-adding a flat test
    directory by accident.
    """
    flat = sorted(
        p.name
        for p in (ROOT / "tests" / "unit").iterdir()
        if p.is_dir() and p.name.startswith("test_")
    )
    # These mirror no source package: test_docs and test_scripts are tooling tests,
    # test_scenarios is end-to-end, test_container covers the Dockerfile and the local
    # compose stack, test_webui the console's wiring (no JS test runner), and test_deploy
    # covers deploy/ (the owner's server: compose, config overrides, migration scripts),
    # which stays private, so the public tree has no test_deploy. None is a Python
    # package. Everything that DOES mirror a package lives under
    # tests/unit/iris_harness/<layer>/.
    private = ["test_deploy"] if (ROOT / "deploy" / "compose.server.yml").is_file() else []
    public = ["test_container", "test_docs", "test_scenarios", "test_scripts", "test_webui"]
    assert flat == sorted(public + private), (
        f"unexpected flat test directories: {flat}. A package under src/iris_harness/ "
        "mirrors into tests/unit/iris_harness/<layer>/test_<pkg>/, not here."
    )


def test_a_docs_change_reaches_the_doc_gate() -> None:
    """Docs stopped being untested when tests/unit/test_docs landed: it asserts that
    every source path a current-state doc names exists. So a doc edit must reach a test,
    not report "nothing to run" -- which is what it did while that gate did not exist."""
    plan = _plan("docs/architecture/ARCHITECTURE.md")
    assert "tests/unit/test_docs" in plan
    assert "full suite" not in plan


def test_an_example_change_runs_that_example_and_the_stable_tier_check() -> None:
    """Each example carries its own test; the stable-tier contract holds all of them."""
    plan = _plan("examples/01-deterministic-handler/opening_hours.py")
    assert "examples/01-deterministic-handler" in plan
    assert "tests/unit/test_stable_tier.py" in plan
    assert "full suite" not in plan
    assert "tests/unit/test_stable_tier.py" in _plan("examples/README.md")


def test_a_non_doc_untested_file_still_runs_nothing() -> None:
    """The docs rule is a mapping, not a catch-all: a file in no test's scope and in no
    source root still reports honestly rather than being swept into the doc gate."""
    assert "nothing to run" in _plan(".editorconfig")


def test_the_image_the_console_and_the_server_kit_reach_their_tests() -> None:
    """None is a Python package, so each maps by path: the Dockerfile and the local
    stack to test_container, webui/ to test_webui, deploy/ to test_deploy (when this
    tree has it: the public one does not)."""
    assert "tests/unit/test_container" in _plan("Dockerfile")
    assert "tests/unit/test_container" in _plan("docker-compose.yml")
    assert "tests/unit/test_webui" in _plan("webui/src/lib/client.ts")
    if (ROOT / "tests" / "unit" / "test_deploy").is_dir():
        assert "tests/unit/test_deploy" in _plan("deploy/compose.server.yml")


def _trigger_lines() -> list[str]:
    return [
        line.strip()
        for line in TRIGGERS.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


@pytest.mark.parametrize("pattern", _trigger_lines())
def test_every_full_suite_trigger_matches_a_tracked_file(
    pattern: str, tracked_files: list[str]
) -> None:
    regex = re.compile(pattern)
    assert any(regex.search(path) for path in tracked_files), (
        f"trigger {pattern!r} matches no tracked file — a move left it stale, "
        "so the full suite would silently stop firing for that path"
    )


def test_trigger_fires_through_the_script() -> None:
    plan = _plan("src/iris_harness/runtime/bootstrap.py")
    assert "core file changed" in plan and "full suite" in plan


def _targets(plan: str) -> list[str]:
    return [line.strip() for line in plan.splitlines() if line.startswith("    ")]


def test_a_file_inside_a_selected_directory_is_not_passed_separately() -> None:
    """pytest 8.4 given `dir` + `dir/.../test_x.py` collects only the file (477 → 29).

    The directory already runs the file, so the file must not also be passed.
    """
    targets = _targets(
        _plan(
            "src/iris_harness/memory/graph.py",  # maps to tests/unit/iris_harness/memory
            "tests/unit/iris_harness/memory/test_facts/test_fact_keys.py",
        )
    )
    assert "tests/unit/iris_harness/memory" in targets
    assert "tests/unit/iris_harness/memory/test_facts/test_fact_keys.py" not in targets


def test_a_directory_inside_a_selected_directory_is_dropped_too() -> None:
    targets = _targets(
        _plan(
            "src/iris_harness/memory/graph.py",
            "tests/unit/iris_harness/memory/test_facts/conftest.py",
        )
    )
    assert [t for t in targets if t.startswith("tests/unit/iris_harness/memory")] == [
        "tests/unit/iris_harness/memory"
    ]


def _run_with_workers(workers: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "--print"],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "CHANGED_OVERRIDE": "docs/architecture/plugin-capabilities.md",
            "IRIS_PYTEST_WORKERS": workers,
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("workers", ["", "auto", "logical", "0", "4"])
def test_a_valid_worker_count_is_accepted(workers: str) -> None:
    assert _run_with_workers(workers).returncode == 0


@pytest.mark.parametrize("workers", ["-1", "four", "4 --lf"])
def test_an_invalid_worker_count_fails_loudly(workers: str) -> None:
    result = _run_with_workers(workers)
    assert result.returncode == 2
    assert "IRIS_PYTEST_WORKERS" in result.stderr
