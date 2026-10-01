"""Wire the loader + runner to a real runtime for the CLI and API surfaces.

This is the one place that knows how to build a runtime, so the runner stays
decoupled and testable. Suite-level env is applied BEFORE the runtime is built
so build-time flags (most IRIS_* intercept toggles) take effect; the runtime is
built with warmup disabled so scenarios never depend on a live Ollama for the
deterministic paths.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Any

from .loader import discover_suites, load_all_suites, load_suite
from .models import ScenarioSuite, SuiteResult
from .runner import PlaygroundRunner

__all__ = [
    "discover_suites",
    "load_all_suites",
    "load_suite",
    "run_suite",
    "run_suites",
]


# Safety floor for in-process scenario runs (same pair the eval preflight sets,
# ADR-0070): a scenario like "remind me tomorrow at 9am" must never write into
# the real Apple/Google calendar just because the suite ran on the dev's
# machine (the 2026-07-05 campaign did exactly that — 6 real events). A suite
# that genuinely wants live write-back opts back in via its own env block.
#
# Deliberately NOT neutralized: the dev .env. load_dotenv (override=False)
# fills any key the suite leaves unset, so suites that A/B a flag must force
# it explicitly in their env block. See
# docs/experiments/2026-07-05-deterministic-intercept-campaign.md.
_SAFETY_DEFAULTS = {
    "IRIS_DISABLE_EXTERNAL_WRITES": "1",
    "IRIS_CALENDAR_APPLE_WRITE": "0",
}


# The identity files a run can write (USER.md's auto-detected block above all) live
# under $IRIS_HOME; an isolated run points the loader's override slots at its own
# directory for its duration. $IRIS_HOME itself is left alone: the owner's profile
# overlay and home plugins still decide what the runtime mounts.
@contextmanager
def _isolated_memory() -> Iterator[dict[str, str]]:
    """A throwaway data directory and identity workspace; yields the env to apply."""
    from iris_harness.memory.identity import loader

    accessors = loader.IDENTITY_PATH_ACCESSORS
    with tempfile.TemporaryDirectory(prefix="iris-playground-") as tmp:
        root = Path(tmp)
        home = loader.iris_home()
        # Every target first: setting one slot moves the paths derived from it.
        moved = {name: root / "home" / path().relative_to(home) for name, path in accessors.items()}
        prior = {name: getattr(loader, name) for name in accessors}
        try:
            for name, path in moved.items():
                setattr(loader, name, path)
            yield {"IRIS_DATA_DIR": str(root / "data")}
        finally:
            for name, value in prior.items():
                setattr(loader, name, value)


@contextmanager
def _built_runtime(env: dict[str, str], *, isolated: bool = False) -> Iterator[Any]:
    """Build a runtime with *env* applied and warmup off; shut it down after."""
    from iris_harness.runtime import build_runtime  # heavy import, keep lazy

    sentinel = object()
    stack = ExitStack()
    isolation = stack.enter_context(_isolated_memory()) if isolated else {}
    overrides = {**_SAFETY_DEFAULTS, **env, **isolation, "IRIS_DISABLE_WARMUP": "1"}
    prior: dict[str, Any] = {k: os.environ.get(k, sentinel) for k in overrides}
    os.environ.update(overrides)
    runtime = None
    try:
        runtime = build_runtime()
        runtime.startup()
        yield runtime
    finally:
        if runtime is not None:
            with suppress(Exception):  # teardown must not mask scenario results
                runtime.shutdown()
        for key, old in prior.items():
            if old is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        stack.close()


def run_suite(suite: ScenarioSuite, *, runtime: Any | None = None) -> SuiteResult:
    """Run one suite. Builds a runtime (suite.env applied) unless one is given."""
    if runtime is not None:
        return PlaygroundRunner(runtime.chat).run_suite(suite)
    with _built_runtime(suite.env, isolated=suite.isolated) as built:
        return PlaygroundRunner(built.chat).run_suite(suite)


def run_suites(scenario_dir: Path | None = None) -> list[SuiteResult]:
    """Load and run every discoverable suite (one runtime per suite)."""
    return [run_suite(suite) for suite in load_all_suites(scenario_dir)]
