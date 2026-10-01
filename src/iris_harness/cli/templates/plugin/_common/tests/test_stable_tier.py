"""This plugin imports only the stable tier of IRIS (docs/reference/stable-api.md).

Everything else in IRIS may change in any release; a stable name changes only after a
deprecation cycle. ``check_stable_imports`` reads every import in the plugin and in its
tests and reports each one the tier does not cover.
"""

from __future__ import annotations

from pathlib import Path

from iris_harness.testing import check_stable_imports

ROOT = Path(__file__).resolve().parent.parent


def test_the_plugin_imports_only_the_stable_tier() -> None:
    violations = check_stable_imports([ROOT / "src", ROOT / "tests"])
    assert violations == [], "\n".join(str(v) for v in violations)
