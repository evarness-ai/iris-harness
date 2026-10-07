"""Is a domain a public suffix? (issue #175)

A wildcard host declaration over a public suffix (``*.co.uk``, ``*.com.au``) would let a plugin
reach every registrant under it, which is not what ``*.example.co.uk`` means. The rules are
Mozilla's Public Suffix List, ICANN section only, vendored as ``public_suffix_icann.dat`` and
refreshed by ``scripts/refresh_public_suffixes.py``. The PRIVATE section (hosting providers
such as ``github.io``) is deliberately left out: a plugin may legitimately declare
``*.github.io``-style families, and the declaration is the owner's to read.

The matching is the list's own algorithm for the question "is this name itself a public
suffix?": an exact rule, or a ``*.`` rule whose base is the name's parent, unless an exception
(``!``) rule names it. The default rule (an unlisted single label is a suffix) is applied too, so
every one-label name is a suffix. The data is read once, on first use.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).with_name("public_suffix_icann.dat")


@lru_cache(maxsize=1)
def _rules() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """``(exact, wildcard bases, exceptions)`` from the vendored list."""
    exact: set[str] = set()
    wildcard: set[str] = set()
    exceptions: set[str] = set()
    for line in DATA.read_text(encoding="utf-8").splitlines():
        rule = line.strip()
        if not rule or rule.startswith("//"):
            continue
        if rule.startswith("!"):
            exceptions.add(rule[1:])
        elif rule.startswith("*."):
            wildcard.add(rule[2:])
        else:
            exact.add(rule)
    return frozenset(exact), frozenset(wildcard), frozenset(exceptions)


def is_public_suffix(domain: str) -> bool:
    """Whether ``domain`` (lower-case ASCII, no trailing dot) is itself a public suffix."""
    name = domain.lower().strip(".")
    if not name:
        return False
    exact, wildcard, exceptions = _rules()
    if name in exceptions:
        return False  # ``!www.ck``: www.ck is a registrable name; the suffix is its parent
    if "." not in name:
        return True  # the default rule: any single label is a suffix
    if name in exact:
        return True
    return name.split(".", 1)[1] in wildcard


def rule_count() -> int:
    """How many rules were loaded (a sanity check for the vendored file)."""
    exact, wildcard, exceptions = _rules()
    return len(exact) + len(wildcard) + len(exceptions)


__all__ = ["DATA", "is_public_suffix", "rule_count"]
