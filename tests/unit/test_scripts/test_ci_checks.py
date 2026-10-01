"""The public CI's own check scripts (OSS plan R11): ``licenses``, ``dco`` and
``pr-title`` (``scripts/check_licenses.py``, ``check_dco.py``, ``check_pr_title.py``).

They ship in the public tree with the workflows that run them, and so does this test.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from email.message import Message
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered first: a dataclass resolves its annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


licenses = _load("check_licenses")
dco = _load("check_dco")
title = _load("check_pr_title")

POLICY = licenses.Policy.load(ROOT / "scripts" / "license_policy.toml")


# ---------------------------------------------------------------- licenses


@pytest.mark.parametrize(
    ("expression", "ok"),
    [
        ("MIT", True),
        ("Apache-2.0 OR BSD-3-Clause", True),
        ("MPL-2.0 AND (Apache-2.0 OR MIT)", True),
        ("BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0", True),
        ("MPL-1.1 OR GPL-2.0-only OR LGPL-2.1-or-later", True),  # one allowed option
        ("GPL-3.0-only", False),
        ("MIT AND GPL-3.0-only", False),  # every AND term must be allowed
        ("Elastic-2.0", False),
        ("Apache-2.0 WITH LLVM-exception", True),
        ("LGPL-2.1+", True),  # deprecated `+` reads as -or-later
    ],
)
def test_spdx_expressions(expression: str, ok: bool) -> None:
    assert licenses.satisfiable(expression, POLICY.allowed) is ok


@pytest.mark.parametrize("text", ["MIT License", "Dual License", "(MIT", "MIT OR", ""])
def test_free_text_is_not_an_expression(text: str) -> None:
    with pytest.raises(licenses.ExpressionError):
        licenses.satisfiable(text, POLICY.allowed)


def _meta(
    name: str, *, expr: str = "", lic: str = "", classifiers: tuple[str, ...] = ()
) -> Message:
    msg = Message()
    msg["Name"] = name
    msg["Version"] = "1.0"
    if expr:
        msg["License-Expression"] = expr
    if lic:
        msg["License"] = lic
    for classifier in classifiers:
        msg["Classifier"] = classifier
    return msg


def test_the_license_is_read_in_order() -> None:
    by_expr = licenses.judge(_meta("a", expr="MIT", lic="GPL-3.0-only"), POLICY)
    assert (by_expr.source, by_expr.ok) == ("License-Expression", True)

    alias = licenses.judge(_meta("b", lic="Apache License, Version 2.0"), POLICY)
    assert (alias.license, alias.source, alias.ok) == ("Apache-2.0", "License", True)

    # The full licence text in `License`: its title line is read as an alias only.
    text = licenses.judge(_meta("c", lic="MIT License\n\nCopyright (c) ..."), POLICY)
    assert (text.license, text.ok) == ("MIT", True)

    dual = licenses.judge(
        _meta(
            "d",
            lic="Dual License",
            classifiers=(
                "License :: OSI Approved :: BSD License",
                "License :: OSI Approved :: Apache Software License",
            ),
        ),
        POLICY,
    )
    assert (dual.source, dual.ok) == ("classifiers", True)

    unknown = licenses.judge(_meta("e", lic="BSD"), POLICY)
    assert (unknown.source, unknown.ok) == ("License", False)  # "BSD": which one?
    nothing = licenses.judge(_meta("f"), POLICY)
    assert (nothing.source, nothing.ok) == ("none", False)


def test_phoenix_is_denied_whatever_its_metadata_says() -> None:
    verdict = licenses.judge(_meta("arize_phoenix", expr="Apache-2.0"), POLICY)
    assert (verdict.source, verdict.ok) == ("deny", False)


def test_a_reviewed_entry_overrides_unusable_metadata() -> None:
    verdict = licenses.judge(
        _meta("pypdfium2", lic="BSD-3-Clause, Apache-2.0, dependency licenses"), POLICY
    )
    assert (verdict.source, verdict.ok) == ("reviewed", True)


def test_the_allow_list_has_no_strong_copyleft() -> None:
    assert not {x for x in POLICY.allowed if x.startswith(("gpl", "agpl"))}


# ---------------------------------------------------------------------- dco


def _git(repo: Path, *args: str) -> str:
    # A git hook (the pre-push gate runs the suite) exports GIT_DIR and friends: drop
    # them, or these commands act on the repository being pushed.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env |= {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit(repo: Path, message: str, email: str = "ada@example.com") -> str:
    (repo / "f.txt").write_text(message, encoding="utf-8")
    _git(repo, "add", "f.txt")
    _git(
        repo,
        "-c",
        "user.name=Ada",
        "-c",
        f"user.email={email}",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    return tmp_path


def _dco(repo: Path, base: str, head: str) -> int:
    """The script as the workflow runs it: a child process inside the repository."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_dco.py"), base, head],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    ).returncode


def test_dco_passes_when_every_commit_is_signed_off_by_its_author(repo: Path) -> None:
    base = _commit(repo, "init")
    _commit(repo, "feat: one\n\nSigned-off-by: Ada <ada@example.com>")
    head = _commit(repo, "fix: two\n\nSigned-off-by: Ada <ADA@example.com>")
    assert _dco(repo, base, head) == 0


def test_dco_fails_an_unsigned_or_mismatched_commit(repo: Path) -> None:
    base = _commit(repo, "init")
    head = _commit(repo, "feat: unsigned")
    assert _dco(repo, base, head) == 1
    head = _commit(repo, "feat: someone else\n\nSigned-off-by: Bob <bob@example.com>")
    assert _dco(repo, base, head) == 1


def test_dco_exempts_dependabot(repo: Path) -> None:
    base = _commit(repo, "init")
    head = _commit(
        repo,
        "build(deps): bump x\n\nSigned-off-by: dependabot[bot] <support@github.com>",
        email="49699333+dependabot[bot]@users.noreply.github.com",
    )
    assert _dco(repo, base, head) == 0


def test_dco_reports_a_git_error(repo: Path) -> None:
    _commit(repo, "init")
    assert _dco(repo, "0" * 40, "HEAD") == 2


def test_signoff_trailers_are_parsed_case_insensitively() -> None:
    commit = dco.Commit(
        "0" * 40,
        "Ada",
        "ada@example.com",
        "feat: x",
        "feat: x\n\nsigned-off-by: Ada L <Ada@Example.com>\nSigned-off-by: Bob <bob@x.org>",
    )
    assert commit.signoff_emails() == ["ada@example.com", "bob@x.org"]
    assert commit.signed_off()


# ----------------------------------------------------------------- pr-title


@pytest.mark.parametrize(
    "text",
    [
        "feat(email): add the IMAP provider",
        "fix: a crash on an empty inbox",
        "ci(release-1): public workflows",
        "fix(governance,observability): stamp session_id",
        "feat(api)!: drop the v0 routes",
        "build(deps): bump the actions group with 3 updates",
    ],
)
def test_valid_titles(text: str) -> None:
    assert title.problems(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "Add the IMAP provider",
        "feature(email): add it",
        "Feat(email): add it",
        "feat(Email): add it",
        "feat(email):add it",
        "feat(email): ",
        " fix: padded",
        "",
    ],
)
def test_invalid_titles(text: str) -> None:
    assert title.problems(text)
