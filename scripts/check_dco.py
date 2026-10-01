#!/usr/bin/env python3
"""The ``dco`` check (OSS plan R10, R11): every commit a pull request adds is signed off
under the Developer Certificate of Origin (https://developercertificate.org).

A commit passes when its message carries a ``Signed-off-by: Name <email>`` trailer whose
email is the commit author's (``git commit -s`` writes exactly that). Merge commits are
skipped: they add no authored change of their own. The project squash-merges, so the
sign-offs travel into the squash commit's message.

Dependabot's commits are exempt (``EXEMPT_AUTHORS``): a bot cannot certify the DCO, and
its sign-off names support@github.com, never its author address. Like any DCO check this
one reads what a commit says about itself; the review is what verifies it.

Usage (run inside the repository, with both commits fetched)::

    python scripts/check_dco.py <base-sha> <head-sha>

Exit 0 when every commit in ``base..head`` passes, 1 otherwise (each failing commit is
listed with the fix), 2 on a usage or git error. Standard library only.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from email.utils import parseaddr

_TRAILER = "signed-off-by:"
EXEMPT_AUTHORS = frozenset({"49699333+dependabot[bot]@users.noreply.github.com"})
# Fields of one commit, NUL-separated; commits separated by \x1e (record separator).
_FORMAT = "%H%x00%an%x00%ae%x00%s%x00%B%x1e"


@dataclass(frozen=True)
class Commit:
    sha: str
    author_name: str
    author_email: str
    subject: str
    message: str

    def signoff_emails(self) -> list[str]:
        emails = []
        for line in self.message.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith(_TRAILER):
                _, email = parseaddr(stripped[len(_TRAILER) :].strip())
                if email:
                    emails.append(email.lower())
        return emails

    def signed_off(self) -> bool:
        if self.author_email.lower() in EXEMPT_AUTHORS:
            return True
        return self.author_email.lower() in self.signoff_emails()


def commits_between(base: str, head: str) -> list[Commit]:
    out = subprocess.run(  # noqa: S603 -- a fixed git argv, shas validated by git
        ["git", "log", "--no-merges", f"--format={_FORMAT}", f"{base}..{head}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    commits = []
    for record in out.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        sha, name, email, subject, message = record.split("\x00", 4)
        commits.append(Commit(sha, name, email, subject, message))
    return commits


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_dco.py <base-sha> <head-sha>", file=sys.stderr)
        return 2
    base, head = argv
    try:
        commits = commits_between(base, head)
    except subprocess.CalledProcessError as exc:
        print(f"dco: git log failed: {exc.stderr.strip()}", file=sys.stderr)
        return 2
    missing = [c for c in commits if not c.signed_off()]
    for c in missing:
        found = ", ".join(c.signoff_emails()) or "none"
        print(
            f"FAIL {c.sha[:12]} {c.subject!r}: no Signed-off-by for the author "
            f"<{c.author_email}> (sign-offs found: {found})"
        )
    print(f"dco: {len(commits)} commit(s) checked, {len(missing)} without a matching sign-off")
    if missing:
        print(
            "dco: sign off with `git commit -s` (one commit: `git commit --amend -s`; "
            "several: `git rebase --signoff <base>`), then force-push the branch. "
            "CONTRIBUTING.md explains the DCO.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
