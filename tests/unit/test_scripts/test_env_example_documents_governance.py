"""Every ``IRIS_GOVERNANCE_*`` the source reads must appear in ``.env.example``.

The drift this stops: the source read 29 governance variables and the example file
documented 12. Seventeen knobs on the safety layer existed with nothing pointing at
them — among them the cloud-judge opt-in, the side-effect ledger, and the cost cap.

Everything worked correctly unset, so nothing failed. That is exactly why it went
unnoticed for so long, and why a test is the only thing that keeps it fixed: the next
governance flag someone adds fails here until they write one line about it.

A commented-out line counts as documentation. The point is that an operator reading the
file learns the knob exists and what its default is, not that it ships switched on.
"""

from __future__ import annotations

import re

from iris_harness.foundation.paths import repo_root

_VAR = re.compile(r"IRIS_GOVERNANCE_[A-Z0-9_]+")


def _read_governance_vars() -> set[str]:
    """Governance variables named anywhere under ``src/``."""
    found: set[str] = set()
    for path in (repo_root() / "src").rglob("*.py"):
        found.update(_VAR.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return found


def _documented_governance_vars() -> set[str]:
    text = (repo_root() / ".env.example").read_text(encoding="utf-8")
    return set(_VAR.findall(text))


def test_every_governance_variable_the_source_reads_is_in_env_example() -> None:
    undocumented = sorted(_read_governance_vars() - _documented_governance_vars())
    assert not undocumented, (
        "These governance variables are read by the source but absent from .env.example.\n"
        "Add a commented line with the default and one sentence on what it does:\n  "
        + "\n  ".join(undocumented)
    )


def test_env_example_does_not_advertise_variables_nothing_reads() -> None:
    """The other direction: a documented knob that no longer exists is worse than an
    undocumented one, because an operator can set it and believe it took effect."""
    stale = sorted(_documented_governance_vars() - _read_governance_vars())
    assert not stale, (
        ".env.example documents governance variables no source file reads. "
        "Remove them, or the operator setting one will think it did something:\n  "
        + "\n  ".join(stale)
    )


def test_the_scan_is_not_vacuous() -> None:
    """A regex that stops matching, or a repo_root() that resolves somewhere empty,
    would make both assertions above pass by finding nothing. Guard the guard."""
    read = _read_governance_vars()
    documented = _documented_governance_vars()
    assert len(read) > 20, f"only {len(read)} variables found under src/ — scan looks broken"
    assert len(documented) > 20, f"only {len(documented)} found in .env.example"
    assert "IRIS_GOVERNANCE_ENABLED" in read and "IRIS_GOVERNANCE_ENABLED" in documented
