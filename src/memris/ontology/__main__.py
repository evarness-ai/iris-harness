"""``python -m memris.ontology <dir>`` — check an ontology directory.

Prints every issue, then a one-line summary. Exits 1 when there is an error, 0 otherwise
(warnings alone do not fail).
"""

from __future__ import annotations

import sys

from memris.ontology.loader import check_directory


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m memris.ontology <directory>", file=sys.stderr)
        return 2
    result, issues = check_directory(args[0])
    for issue in issues:
        print(issue)
    errors = sum(1 for i in issues if i.severity == "error")
    if result is not None:
        onto = result.ontology
        print(
            f"{onto.id} {onto.version}: {len(onto.classes)} classes, "
            f"{len(onto.relations)} relations, {len(onto.attributes)} attributes, "
            f"{len(onto.mappings)} mappings — {errors} error(s), {len(issues) - errors} warning(s)"
        )
    return 1 if errors or result is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
