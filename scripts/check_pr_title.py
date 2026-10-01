#!/usr/bin/env python3
"""The ``pr-title`` check (OSS plan R11): a pull request's title is a valid commit
subject, ``<type>(<scope>): <summary>``, because squash-merging makes it one.

``type`` is one of the CONTRIBUTING.md list; ``(scope)`` is optional; ``!`` before the
colon marks a breaking change (release-drafter reads it). Usage::

    python scripts/check_pr_title.py "<title>"     # or the title in $PR_TITLE

Exit 0 when the title is valid, 1 otherwise. Standard library only.
"""

from __future__ import annotations

import os
import re
import sys

TYPES = ("feat", "fix", "docs", "style", "refactor", "test", "chore", "ci", "perf", "build")
PATTERN = re.compile(
    r"^(?P<type>" + "|".join(TYPES) + r")"
    # one scope, or several separated by commas: `fix(governance,observability)`
    r"(?:\((?P<scope>[a-z0-9][a-z0-9._/+-]*(?:, ?[a-z0-9][a-z0-9._/+-]*)*)\))?"
    r"(?P<breaking>!)?"
    r": (?P<summary>\S.*)$"
)


def problems(title: str) -> list[str]:
    """What is wrong with ``title``; empty when it is valid."""
    found = []
    if title != title.strip():
        found.append("leading or trailing whitespace")
    match = PATTERN.match(title.strip())
    if not match:
        found.append(
            "not `<type>(<scope>): <summary>` with type one of "
            + ", ".join(TYPES)
            + " (lower case; scope optional, lower case)"
        )
    return found


def main(argv: list[str]) -> int:
    title = argv[0] if argv else os.environ.get("PR_TITLE", "")
    found = problems(title)
    if found:
        print(f"pr-title: FAIL {title!r}")
        for problem in found:
            print(f"  - {problem}")
        print("  e.g. `fix(email): keep the digest order stable`")
        return 1
    print(f"pr-title: ok {title!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
