#!/usr/bin/env python3
"""The ``licenses`` check (OSS plan R11, R19): every installed distribution's license is
on the allow list in ``scripts/license_policy.toml``.

Run it with the interpreter of the environment to check -- it reads that environment's
installed metadata, so it needs nothing but the standard library::

    <venv>/bin/python scripts/check_licenses.py            # check; exit 1 on a failure
    <venv>/bin/python scripts/check_licenses.py --list     # every package and its verdict

The public CI runs it on a fresh core install (``pip install .``, no extras), which is
what the R19 gate means by "the core install".

How a package's license is read, first match wins:

1. ``[deny]`` in the policy: fails, whatever the metadata says;
2. ``[packages.<name>]``: a hand-reviewed SPDX expression for a package whose metadata
   names none usable;
3. the ``License-Expression`` field (PEP 639);
4. the ``License`` field, as an alias from ``[aliases]`` or as an SPDX expression;
5. the ``License ::`` trove classifiers, mapped through ``[aliases]``; several are read
   as a choice (``OR``), the trove convention for dual licensing.

An SPDX expression passes when it is satisfiable from the allow list: ``A OR B`` needs
one of them, ``A AND B`` both, ``A WITH <exception>`` needs ``A``. A package with no
readable license fails: add an alias or a reviewed ``[packages]`` entry.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

DEFAULT_POLICY = Path(__file__).with_name("license_policy.toml")

_TOKEN = re.compile(r"\s*(\(|\)|[A-Za-z0-9.+\-:]+)")


class ExpressionError(ValueError):
    """Not an SPDX license expression."""


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match:
            raise ExpressionError(f"unexpected character at {pos}: {text!r}")
        out.append(match.group(1))
        pos = match.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
    if not out:
        raise ExpressionError("empty expression")
    return out


def satisfiable(expression: str, allowed: set[str]) -> bool:
    """Whether the SPDX ``expression`` is satisfiable from ``allowed`` (lower-cased ids).

    Raises ``ExpressionError`` when ``expression`` is not an SPDX expression (free text
    such as "MIT License" or "Dual License" is not one).
    """
    tokens = _tokens(expression)
    pos = 0

    def at(word: str) -> bool:
        return pos < len(tokens) and tokens[pos].upper() == word

    def take() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ExpressionError(f"expression ends early: {expression!r}")
        pos += 1
        return tokens[pos - 1]

    def is_operator(token: str) -> bool:
        return token.upper() in {"AND", "OR", "WITH"}

    def primary() -> bool:
        item = take()
        if item == "(":
            value = disjunction()
            if take() != ")":
                raise ExpressionError(f"unbalanced parentheses: {expression!r}")
            return value
        if item == ")" or is_operator(item):
            raise ExpressionError(f"unexpected {item!r}: {expression!r}")
        if at("WITH"):
            take()
            exception = take()
            if exception in {"(", ")"} or is_operator(exception):
                raise ExpressionError(f"WITH needs an exception id: {expression!r}")
        # An exception only grants more; the licence itself decides. A trailing `+`
        # (deprecated "or later") reads as the -or-later id.
        ident = item.lower()
        if ident.endswith("+"):
            ident = ident[:-1] + "-or-later"
        return ident in allowed

    def conjunction() -> bool:
        value = primary()
        while at("AND"):
            take()
            value = primary() and value
        return value

    def disjunction() -> bool:
        value = conjunction()
        while at("OR"):
            take()
            value = conjunction() or value
        return value

    result = disjunction()
    if pos != len(tokens):
        raise ExpressionError(f"trailing tokens in {expression!r}")
    return result


@dataclass(frozen=True)
class Verdict:
    name: str
    version: str
    license: str  # what was read, as shown
    source: str  # deny | reviewed | License-Expression | License | classifiers | none
    ok: bool


@dataclass(frozen=True)
class Policy:
    allowed: set[str]
    deny: dict[str, str]
    aliases: dict[str, str]
    packages: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> Policy:
        data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
        return cls(
            allowed={x.lower() for x in data.get("allow", {}).get("licenses", [])},
            deny={_norm(k): str(v) for k, v in data.get("deny", {}).items()},
            aliases={k.strip().lower(): v for k, v in data.get("aliases", {}).items()},
            packages={_norm(k): v["license"] for k, v in data.get("packages", {}).items()},
        )


def _norm(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _try(expression: str, policy: Policy) -> bool | None:
    try:
        return satisfiable(expression, policy.allowed)
    except ExpressionError:
        return None


def judge(metadata: Any, policy: Policy) -> Verdict:
    """The verdict for one distribution's metadata (an ``email.message.Message``)."""
    name = metadata["Name"]
    version = metadata["Version"] or "?"
    key = _norm(name)
    if key in policy.deny:
        return Verdict(name, version, policy.deny[key], "deny", False)
    if key in policy.packages:
        expr = policy.packages[key]
        return Verdict(name, version, expr, "reviewed", bool(_try(expr, policy)))

    expr = (metadata.get("License-Expression") or "").strip()
    if expr:
        ok = _try(expr, policy)
        return Verdict(name, version, expr, "License-Expression", bool(ok))

    raw = (metadata.get("License") or "").strip()
    first_line = raw.splitlines()[0].strip() if raw else ""
    # One line: an alias or an SPDX expression. Several: the licence text itself, whose
    # title line ("MIT License") is read only as an alias, never parsed.
    if first_line:
        single = len(raw.splitlines()) == 1
        mapped = policy.aliases.get(first_line.lower())
        ok = _try(mapped or first_line, policy) if (mapped or single) else None
        if ok is not None:
            return Verdict(name, version, mapped or first_line, "License", ok)

    mapped_classifiers = []
    for classifier in metadata.get_all("Classifier") or []:
        if not classifier.startswith("License ::"):
            continue
        leaf = classifier.split(" :: ")[-1].strip()
        if leaf.lower() in policy.aliases:
            mapped_classifiers.append(policy.aliases[leaf.lower()])
    if mapped_classifiers:
        expr = " OR ".join(sorted(set(mapped_classifiers)))
        return Verdict(name, version, expr, "classifiers", bool(_try(expr, policy)))

    shown = first_line[:60] or "(no license metadata)"
    return Verdict(name, version, shown, "none", False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--list", action="store_true", help="print every package's verdict")
    args = parser.parse_args(argv)

    policy = Policy.load(args.policy)
    seen: dict[str, Verdict] = {}
    for dist in md.distributions():
        verdict = judge(dist.metadata, policy)
        seen.setdefault(_norm(verdict.name), verdict)
    verdicts = sorted(seen.values(), key=lambda v: v.name.lower())
    failures = [v for v in verdicts if not v.ok]

    for v in verdicts if args.list else failures:
        mark = "ok  " if v.ok else "FAIL"
        print(f"{mark} {v.name} {v.version}: {v.license}  [{v.source}]")
    print(
        f"licenses: {len(verdicts)} distributions, {len(failures)} not on the allow list "
        f"({sys.executable})"
    )
    if failures:
        print(
            "licenses: FAIL -- each one needs a decision: drop the dependency, make it an "
            f"extra, or (after reading its terms) map it in {args.policy.name}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
