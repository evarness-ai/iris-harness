"""``search_docs``: finds IRIS's own docs, and cannot reach identity files or secrets.

The corpus is an allow-list (``config/docs_search.yaml``). Every exclusion class the
owner named has a test that plants a canary where it would live and proves it never comes
back, including through ``..``, symlinks and absolute paths in the arguments; and a
mutation check shows the allow-list is what keeps a neighbouring directory out.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import yaml

from iris_harness.foundation.paths import data_dir, governance_config_dir, iris_home
from iris_harness.kernel.governance.file_scan import scan_text
from iris_harness.services.docs_search import corpus as corpus_mod
from iris_harness.services.docs_search import search as search_mod
from iris_harness.services.docs_search.corpus import (
    DocsSearchConfigError,
    build_corpus,
    load_config,
)
from iris_harness.services.docs_search.search import NOT_PRESENT, search_core_docs, search_docs_in

CANARY = "zqxcanaryword"
ROOTS = ["docs/architecture", "docs/concepts", "docs/guides", "docs/reference", "docs/usage-guides"]
# A real-shaped secret, built at runtime so no source file holds a contiguous PEM marker
# (the repo's history scan flags one). The kernel's own classifier must call it secret;
# the tests that use it assert that first.
_PEM = "PRIVATE " + "KEY"
SECRET_TEXT = (
    "-----"
    + "BEGIN RSA "
    + _PEM
    + "-----\nMIIEowIBAAKCAQEA"
    + "A" * 60
    + "\n-----"
    + "END RSA "
    + _PEM
    + "-----"
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in checkout with a small docs tree, and a config pointing at it."""
    base = tmp_path / "checkout"
    _write(
        base / "docs/architecture/iris-harness.md",
        "# Harness\n\nIntro text.\n\n## Tier routing\n\nThe router picks a model tier per intent "
        "and escalates when a tier cannot answer.\n\n## Governance hooks\n\nPreToolUse runs "
        "before every tool call.\n",
    )
    _write(
        base / "docs/concepts/plugins.md", "# Plugins\n\n## Manifest\n\nA plugin declares tools.\n"
    )
    _write(base / "docs/guides/setup.md", "# Setup\n\n## Install\n\nRun poetry install.\n")
    _write(
        base / "docs/reference/env.md",
        "# Env\n\n## Flags\n\nIRIS_DISABLE_WARMUP turns warmup off.\n",
    )
    _write(base / "docs/usage-guides/use.md", "# Use\n\n## Chat\n\nAsk the assistant.\n")
    _write_config(tmp_path, monkeypatch, ROOTS)
    return base


def _write_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, roots: list[str], **limits: int
) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir(exist_ok=True)
    body: dict[str, object] = {"roots": roots, "extensions": [".md"]}
    if limits:
        body["limits"] = limits
    (cfg / "docs_search.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    # This is the SHIPPED allow-list; an IRIS_CONFIG_DIR copy can only narrow it.
    monkeypatch.setattr(corpus_mod, "default_config_dir", lambda: cfg)
    monkeypatch.setattr(
        corpus_mod, "config_path", lambda name: Path(os.environ.get("IRIS_CONFIG_DIR", cfg)) / name
    )
    monkeypatch.delenv("IRIS_CONFIG_DIR", raising=False)


def _search(tree: Path, **args: object) -> str:
    return search_docs_in(args, base=tree)


@pytest.fixture()
def reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every file the tool reads, so a test can prove a path was never opened."""
    seen: list[str] = []
    original = os.open

    def spy(path: object, *a: object, **k: object) -> int:
        seen.append(os.path.realpath(os.fspath(path)))  # type: ignore[arg-type]
        return original(path, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", spy)
    return seen


# --------------------------------------------------------------------------- behaviour
def test_finds_the_section_and_returns_a_snippet_not_the_file(tree: Path) -> None:
    out = _search(tree, query="tier routing escalates")
    assert "docs/architecture/iris-harness.md > ## Tier routing (line " in out
    assert "escalates when a tier cannot answer" in out
    assert "Governance hooks" not in out  # a different section is not dumped
    assert "Intro text" not in out


def test_deterministic_and_ranked(tree: Path) -> None:
    first = _search(tree, query="tier routing tool")
    assert first == _search(tree, query="tier routing tool")
    assert "\n1. docs/architecture/iris-harness.md > ## Tier routing" in first
    assert first.index("## Tier routing") < first.index("## Governance hooks")


def test_section_filter_limits_to_matching_headings(tree: Path) -> None:
    out = _search(tree, query="tool tier", section="governance")
    assert "## Governance hooks" in out and "## Tier routing" not in out


def test_results_and_output_are_bounded(tree: Path, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for i in range(30):
        _write(tree / f"docs/guides/g{i}.md", f"# G{i}\n\n## Topic\n\n{'tier ' * 200}\n")
    out = _search(tree, query="tier", limit=1000)
    assert out.count("\n   ") <= 10  # max_results, whatever the caller asks
    assert len(out) <= 4000 + 200
    assert len(_search(tree, query="tier")) < len(out) + 1
    assert all(len(line) < 400 for line in out.splitlines())


def test_file_count_cap_is_enforced(tree: Path, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for i in range(10):
        _write(tree / f"docs/guides/g{i}.md", f"# G{i}\n\nbody {i}\n")
    _write_config(tmp_path, monkeypatch, ROOTS, max_files=3)
    corpus = build_corpus(load_config(), tree)
    assert len(corpus.docs) == 3 and corpus.truncated


def test_oversized_file_is_skipped(tree: Path, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write(tree / "docs/guides/huge.md", "# Huge\n\n## S\n\nbigwordzz " + "x" * 5000)
    _write_config(tmp_path, monkeypatch, ROOTS, max_file_bytes=1000)
    assert "no section matches" in _search(tree, query="bigwordzz")


def test_hostile_query_is_inert_text(tree: Path) -> None:
    out = _search(tree, query="tier " + "(" * 5000 + ".*" * 5000 + "[a-" * 100)
    assert out.startswith("search_docs:") and "## Tier routing" in out


def test_missing_query_is_a_clear_error(tree: Path) -> None:
    assert _search(tree, query="  ").startswith("Error: search_docs requires")
    assert _search(tree, query="!!! ???").startswith("Error: search_docs requires")


def test_docs_not_present_on_this_install(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write_config(tmp_path, monkeypatch, ROOTS)
    assert search_docs_in({"query": "tier"}, base=tmp_path / "empty") == NOT_PRESENT
    # a wheel install: no checkout, so no docs dir at all
    monkeypatch.setattr(corpus_mod, "checkout_docs_dir", lambda: None)
    assert search_core_docs({"query": "tier"}) == NOT_PRESENT


def test_the_real_checkout_docs_are_found() -> None:
    out = search_core_docs({"query": "tier routing"})
    assert out != NOT_PRESENT and "docs/architecture/iris-harness.md" in out


def test_shipped_allow_list_is_exactly_the_documented_roots(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("IRIS_CONFIG_DIR", raising=False)
    config = load_config()
    assert list(config.roots) == ROOTS
    assert config.extensions == frozenset({".md"})


# --------------------------------------------------------------------- exclusion classes
def _no_canary(out: str) -> None:
    # The header line echoes the query (so a query for the canary names it); no RESULT
    # line may carry it.
    assert not any(CANARY in line.lower() for line in out.splitlines()[1:])
    assert "\n1. " not in out or CANARY not in out.split("\n1. ", 1)[1].lower()


def _identity_files() -> list[Path]:
    home = iris_home()
    return [
        _write(
            home / "workspace/SOUL.md",
            f"---\nclassification: internal\n---\n# Soul\n{CANARY} soul\n",
        ),
        _write(
            home / "workspace/USER.md",
            f"---\nclassification: personal\n---\n# User\n{CANARY} user\n",
        ),
        _write(home / "workspace/AGENTS.md", f"# Agents\n{CANARY} agents\n"),
        _write(home / "identity/soul.md", f"# Soul\n{CANARY} legacy soul\n"),
        _write(home / "memory/user.md", f"# User\n{CANARY} legacy user\n"),
    ]


def test_identity_files_are_unreachable(tree: Path, reads: list[str]) -> None:
    files = _identity_files()
    for query in (CANARY, "soul user agents", "workspace SOUL.md", "identity soul legacy"):
        _no_canary(_search(tree, query=query))
    assert not {os.path.realpath(f) for f in files} & set(reads)


@pytest.mark.parametrize(
    "where",
    [
        lambda: governance_config_dir() / "vault/vault.db.md",
        lambda: governance_config_dir() / ".env",
        lambda: data_dir() / "email/mail.md",
        lambda: data_dir() / "finance/ledger.md",
        lambda: data_dir() / "memory/facts.md",
        lambda: iris_home() / "data/credentials.md",
        lambda: iris_home() / "tokens/token.md",
    ],
    ids=[
        "vault",
        "dotenv",
        "email-store",
        "finance-store",
        "memory-store",
        "credentials",
        "tokens",
    ],
)
def test_vault_env_credentials_and_stores_are_unreachable(tree: Path, reads: list[str], where) -> None:  # type: ignore[no-untyped-def]
    planted = _write(where(), f"# Secret\n\n{CANARY} here\n")
    _no_canary(_search(tree, query=CANARY))
    assert os.path.realpath(planted) not in reads


@pytest.mark.parametrize(
    "name", [".env", "credentials.json", "id_rsa", "server.pem", "token.txt", "cfg.yaml"]
)
def test_secret_shaped_files_inside_an_allowed_root_are_not_read(
    tree: Path, reads: list[str], name: str
) -> None:
    planted = _write(tree / "docs/architecture" / name, f"{CANARY}\n")
    _no_canary(_search(tree, query=CANARY))
    assert os.path.realpath(planted) not in reads


def test_a_root_that_overlaps_the_owners_home_is_refused(tree: Path, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    home = tree / "docs/privhome"
    monkeypatch.setenv("IRIS_HOME", str(home))
    _write(home / "workspace/SOUL.md", f"# Soul\n{CANARY}\n")
    _write(home / "data/notes.md", f"# Data\n{CANARY}\n")
    for root in ("docs/privhome", "docs/privhome/workspace", "docs/privhome/data"):
        _write_config(tmp_path, monkeypatch, [root])
        assert _search(tree, query=CANARY) == NOT_PRESENT  # the only root was refused
    # a root that contains the home is refused too
    monkeypatch.setenv("IRIS_HOME", str(tree / "docs/guides/home"))
    _write(tree / "docs/guides/home/workspace/SOUL.md", f"# Soul\n{CANARY}\n")
    _write_config(tmp_path, monkeypatch, ["docs/guides"])
    assert _search(tree, query=CANARY) == NOT_PRESENT


def test_file_symlink_out_of_the_root_is_not_followed(
    tree: Path, tmp_path: Path, reads: list[str]
) -> None:
    outside = _write(tmp_path / "outside/secret.md", f"# S\n\n{CANARY}\n")
    (tree / "docs/architecture/link.md").symlink_to(outside)
    _no_canary(_search(tree, query=CANARY))
    assert os.path.realpath(outside) not in reads


def test_directory_symlink_out_of_the_root_is_not_followed(
    tree: Path, tmp_path: Path, reads: list[str]
) -> None:
    outside = _write(tmp_path / "outside/dir/secret.md", f"# S\n\n{CANARY}\n")
    (tree / "docs/architecture/linked").symlink_to(outside.parent, target_is_directory=True)
    _no_canary(_search(tree, query=CANARY))
    assert os.path.realpath(outside) not in reads


def test_symlink_to_an_identity_file_is_not_followed(tree: Path, reads: list[str]) -> None:
    soul = _identity_files()[0]
    (tree / "docs/concepts/soul.md").symlink_to(soul)
    _no_canary(_search(tree, query=CANARY))
    assert os.path.realpath(soul) not in reads


def test_a_root_that_is_itself_a_symlink_is_refused(tree: Path, tmp_path: Path, monkeypatch, reads: list[str]) -> None:  # type: ignore[no-untyped-def]
    outside = _write(tmp_path / "outside2/secret.md", f"# S\n\n{CANARY}\n")
    (tree / "docs/elsewhere").symlink_to(outside.parent, target_is_directory=True)
    _write_config(tmp_path, monkeypatch, ["docs/elsewhere"])
    assert _search(tree, query=CANARY) == NOT_PRESENT
    assert os.path.realpath(outside) not in reads
    # and a symlink further up the root's path is refused as well
    (tree / "docs/guides").rename(tree / "docs/guides_real")
    (tree / "docs/guides").symlink_to(outside.parent, target_is_directory=True)
    _write_config(tmp_path, monkeypatch, ["docs/guides"])
    assert _search(tree, query=CANARY) == NOT_PRESENT


@pytest.mark.parametrize(
    "root",
    [
        "../outside",
        "/etc",
        "~/.iris/workspace",
        "docs/../..",
        "docs\\..\\x",
        ".",
        "",
        "/",
        "src",
        "config",
        "docs",
        "data/docs",
        "docs/../src",
        "/docs/guides",
    ],
)
def test_unsafe_roots_in_the_config_are_a_config_error(tree: Path, tmp_path, monkeypatch, root: str) -> None:  # type: ignore[no-untyped-def]
    _write_config(tmp_path, monkeypatch, [root])
    with pytest.raises(DocsSearchConfigError):
        load_config()
    assert _search(tree, query="tier").startswith("search_docs is unavailable:")


def test_missing_config_fails_closed(tree: Path, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "cfg" / "docs_search.yaml").write_text("roots: []\n", encoding="utf-8")
    assert _search(tree, query="tier").startswith("search_docs is unavailable:")


@pytest.mark.parametrize(
    "arg",
    [
        "../../.iris/workspace/SOUL.md",
        "../../../../etc/passwd",
        "~/.iris/workspace/USER.md",
        "$IRIS_HOME/workspace/AGENTS.md",
        "file:///etc/passwd",
        "..\\..\\.iris\\workspace\\SOUL.md",
        "docs/../../.iris/workspace/SOUL.md",
        "\x00../../SOUL.md",
    ],
)
def test_paths_in_query_and_section_args_are_text_never_paths(
    tree: Path, reads: list[str], arg: str
) -> None:
    files = _identity_files()
    absolute = [str(f) for f in files]
    for value in [arg, *absolute]:
        for out in (
            _search(tree, query=value),
            _search(tree, query="tier", section=value),
            _search(tree, query=value, section=value),
        ):
            _no_canary(out)
            assert out.startswith(("search_docs", "Error: search_docs"))
    assert not {os.path.realpath(f) for f in files} & set(reads)
    assert all(r.startswith(os.path.realpath(tree)) or "docs_search" in r for r in reads)


@pytest.mark.parametrize("classification", ["personal", "secret", "PERSONAL"])
def test_frontmatter_classified_documents_are_dropped(tree: Path, classification: str) -> None:
    _write(
        tree / "docs/architecture/notes.md",
        f"---\nclassification: {classification}\n---\n# N\n\n{CANARY}\n",
    )
    _no_canary(_search(tree, query=CANARY))
    assert build_corpus(load_config(), tree).dropped == 1


def test_a_document_that_classifies_secret_is_dropped_not_returned(tree: Path) -> None:
    assert scan_text(SECRET_TEXT).is_secret  # the premise: the kernel scan calls this secret
    _write(
        tree / "docs/guides/leaky.md",
        f"# Leaky\n\n## Keys\n\n{CANARY} and the key:\n{SECRET_TEXT}\n",
    )
    out = _search(tree, query=CANARY)
    _no_canary(out)
    assert "BEGIN RSA" not in out
    assert "Install" in _search(tree, query="poetry install")  # the other docs still answer
    assert build_corpus(load_config(), tree).dropped == 1


def test_returned_snippets_are_classified_again(tree: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real = search_mod.scan_text
    monkeypatch.setattr(
        search_mod, "scan_text", lambda t: real(SECRET_TEXT if "escalates" in t else t)
    )
    out = _search(tree, query="tier routing escalates")
    assert "escalates when a tier cannot answer" not in out


def test_the_scan_runs_on_changed_files_not_stale_cache(tree: Path) -> None:
    path = _write(tree / "docs/guides/live.md", f"# L\n\n## S\n\n{CANARY} fine\n")
    assert "\n1. " in _search(tree, query=CANARY)
    path.write_text(f"# L\n\n## S\n\n{CANARY}\n{SECRET_TEXT}\n", encoding="utf-8")
    os.utime(path, (path.stat().st_atime + 5, path.stat().st_mtime + 5))
    _no_canary(_search(tree, query=CANARY))


# ------------------------------------------------------------------- allow-list mutation
def test_mutation_the_allow_list_alone_keeps_a_neighbouring_directory_out(tree: Path, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write(tree / "docs/internal-notes/n.md", f"# N\n\n## Notes\n\n{CANARY}\n")
    _no_canary(_search(tree, query=CANARY))  # not allow-listed: absent
    # Mutate the allow-list: the same file is now found, so the test above is not vacuous
    # and the roots are the only thing excluding it.
    _write_config(tmp_path, monkeypatch, [*ROOTS, "docs/internal-notes"])
    assert "docs/internal-notes/n.md" in _search(tree, query=CANARY)
    # and removing a shipped root removes its documents
    _write_config(tmp_path, monkeypatch, [r for r in ROOTS if r != "docs/concepts"])
    assert "docs/concepts/plugins.md" not in _search(tree, query="plugin manifest")


# ------------------------------------------------- review findings (hostile input, parsing)
def test_heading_parser_is_linear_on_a_hostile_line() -> None:
    line = "# a" + " \t" * 2000 + "b #"
    start = time.perf_counter()
    sections = corpus_mod.split_sections(line + "\nbody\n")
    assert time.perf_counter() - start < 0.5
    assert sections and sections[0].heading.startswith("# a")


def test_a_400kb_hostile_document_is_split_and_searched_in_bounded_time(tree: Path) -> None:
    hostile = ("# a" + " \t" * 2000 + "b #\n") * 100  # ~400 KB of the worst heading line
    _write(tree / "docs/guides/hostile.md", hostile)
    start = time.perf_counter()
    out = _search(tree, query="tier routing")
    assert time.perf_counter() - start < 5
    assert out.startswith("search_docs:")


@pytest.mark.parametrize(
    "hostile",
    ["a." * 100000 + "@b.", "+1-" * 130000, "123-45-" * 57000],
    ids=["email-like", "phone-like", "ssn-like"],
)
def test_hostile_content_cannot_wedge_the_scan(tree: Path, hostile: str) -> None:
    """The kernel patterns are super-linear on these (19 s, >20 s, >20 s unchunked)."""
    _write(tree / "docs/guides/hostile.md", f"# H\n\n{hostile}\n")
    start = time.perf_counter()
    _search(tree, query="tier")
    assert time.perf_counter() - start < 5


def test_a_document_that_exhausts_the_scan_budget_is_withheld_and_cached(
    tree: Path, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    _write(tree / "docs/guides/slow.md", f"# S\n\n{CANARY}\n" + "x\n" * 5000)
    _write_config(tmp_path, monkeypatch, ROOTS, scan_budget_ms=1, scan_chunk_chars=100)
    calls: list[int] = []
    real = corpus_mod.scan_text

    def slow(text: str):  # type: ignore[no-untyped-def]
        calls.append(1)
        time.sleep(0.01)
        return real(text)

    monkeypatch.setattr(corpus_mod, "scan_text", slow)
    corpus_mod._CACHE.clear()
    _no_canary(_search(tree, query=CANARY))
    first = len(calls)
    _no_canary(_search(tree, query=CANARY))
    assert len(calls) - first < first  # the withheld document is not scanned again


def test_chunking_does_not_hide_a_secret(tree: Path) -> None:
    assert scan_text(SECRET_TEXT).is_secret
    padding = "ordinary words here\n" * 600
    _write(tree / "docs/guides/late.md", f"# L\n\n{padding}{CANARY}\n{SECRET_TEXT}\n{padding}")
    _no_canary(_search(tree, query=CANARY))
    assert build_corpus(load_config(), tree).dropped == 1


@pytest.mark.parametrize(
    "front",
    [
        "---\nclassification: secret\n---\n",
        "---\nClassification: secret\n---\n",
        '---\n"classification": "secret"\n---\n',
        "---\n classification: secret\n---\n",
        "---\n{classification: secret}\n---\n",
        "\ufeff---\nclassification: secret\n---\n",
        "\n\n---\nclassification: secret\n---\n",
        "---\nclassification: &a secret\n---\n",
        "---\nclassification: secret\n...\n",
        "---\nmeta:\n  Classification: PERSONAL\n---\n",
        "---\r\nclassification: secret\r\n---\r\n",
        "---\t\nclassification:\tsecret\n---\n",
        "---\nclassification: [unclosed\n---\n",  # unparseable: fail closed
        "---\n- a\n- list\n---\n",  # not a mapping: fail closed
        "---\nclassification: public\n",  # never closes: fail closed
    ],
    ids=[
        "plain",
        "capitalised",
        "quoted-key",
        "indented",
        "flow",
        "bom",
        "blank-line-before",
        "anchor",
        "dots-terminator",
        "nested-uppercase",
        "crlf",
        "tab",
        "unparseable",
        "not-a-mapping",
        "unclosed",
    ],
)
def test_every_frontmatter_variant_withholds_the_document(tree: Path, front: str) -> None:
    _write(tree / "docs/architecture/fm.md", f"{front}# F\n\n{CANARY}\n")
    _no_canary(_search(tree, query=CANARY))
    assert build_corpus(load_config(), tree).dropped == 1


def test_public_frontmatter_is_not_withheld(tree: Path) -> None:
    _write(
        tree / "docs/architecture/ok.md",
        f"---\nclassification: public\ntitle: T\n---\n# O\n\n{CANARY}\n",
    )
    assert "docs/architecture/ok.md" in _search(tree, query=CANARY)


# ------------------------------------------------------------------------ TOCTOU on read
def test_a_file_swapped_for_a_symlink_after_the_listing_is_not_read(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = _write(tmp_path / "outside/secret.md", f"# S\n\n{CANARY}\n")
    victim = tree / "docs/guides/setup.md"
    real_open = os.open
    swapped: list[bool] = []

    def swapping(path, *a, **k):  # type: ignore[no-untyped-def]
        if os.fspath(path) == str(victim) and not swapped:
            swapped.append(True)
            victim.unlink()
            victim.symlink_to(outside)  # between the listing and the open
        return real_open(path, *a, **k)

    monkeypatch.setattr(os, "open", swapping)
    _no_canary(_search(tree, query=CANARY))
    assert swapped


def test_a_descriptor_that_names_a_path_outside_the_root_is_refused(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory swapped for a symlink leaves a regular file at the end of the path;
    only the path the descriptor really names tells."""
    outside = _write(tmp_path / "outside/secret.md", f"# S\n\n{CANARY}\n")
    monkeypatch.setattr(corpus_mod, "_fd_path", lambda fd: os.path.realpath(outside))
    out = _search(tree, query="tier")
    assert out == NOT_PRESENT or "no section matches" in out
    assert build_corpus(load_config(), tree).docs == []


def test_unusable_descriptor_path_falls_back_to_samestat(tree: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(corpus_mod, "_fd_path", lambda fd: None)
    assert "## Tier routing" in _search(tree, query="tier routing")


# --------------------------------------------------------------- SDK surface and config
def test_the_public_function_takes_no_corpus_location(tree: Path) -> None:
    import inspect

    from iris_harness.sdk import docs as sdk_docs

    assert list(inspect.signature(sdk_docs.search_core_docs).parameters) == ["args"]
    assert list(inspect.signature(search_core_docs).parameters) == ["args"]
    with pytest.raises(TypeError):
        sdk_docs.search_core_docs({"query": "tier"}, base=tree)  # type: ignore[call-arg]


def _override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **body: object) -> None:
    ov = tmp_path / "override"
    ov.mkdir(exist_ok=True)
    (ov / "docs_search.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(ov))


def test_an_override_cannot_add_roots_outside_the_shipped_allow_list(
    tree: Path, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    _write(tree / "docs/internal-notes/n.md", f"# N\n\n## Notes\n\n{CANARY}\n")
    _override(tmp_path, monkeypatch, roots=[*ROOTS, "docs/internal-notes"])
    assert "docs/internal-notes" not in load_config().roots
    _no_canary(_search(tree, query=CANARY))


def test_an_override_can_only_narrow(tree: Path, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _override(
        tmp_path,
        monkeypatch,
        roots=["docs/concepts", "docs/internal-notes"],
        limits={"max_files": 2, "max_results": 99999},
    )
    config = load_config()
    assert config.roots == ("docs/concepts",)
    assert config.limits.max_files == 2
    assert config.limits.max_results == 10  # the larger override value does not widen it
    out = _search(tree, query="manifest tier")
    assert "docs/concepts/plugins.md" in out and "docs/architecture" not in out


def test_an_override_with_nothing_in_common_fails_closed(tree: Path, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _override(tmp_path, monkeypatch, roots=["docs/internal-notes"])
    assert _search(tree, query="tier").startswith("search_docs is unavailable:")


# ------------------------------------------------------------- round 2 review findings
def _alias_bomb(n: int) -> str:
    lines = ["---", "a0: &a0 [x]"]
    lines += [f"a{i}: &a{i} [" + ", ".join([f"*a{i - 1}"] * 9) + ", x]" for i in range(1, n + 1)]
    return "\n".join(lines) + "\n---\n"


@pytest.mark.parametrize("n", [9, 170])
def test_a_frontmatter_alias_bomb_is_withheld_fast(tree: Path, n: int) -> None:
    front = _alias_bomb(n)
    assert len(front) < 8000 or n == 170
    _write(tree / "docs/architecture/bomb.md", f"{front}# B\n\n{CANARY}\n")
    start = time.perf_counter()
    _no_canary(_search(tree, query=CANARY))
    assert time.perf_counter() - start < 0.5
    assert corpus_mod._frontmatter_withheld(f"{front}# B")


def test_a_deeply_shared_structure_without_anchors_is_walked_bounded() -> None:
    # No anchors, so the event stream passes; the walk is still bounded by its budget.
    deep = "---\n" + "a: " * 1 + "[" * 400 + "]" * 400 + "\n---\n"
    start = time.perf_counter()
    corpus_mod._frontmatter_withheld(deep)
    assert time.perf_counter() - start < 0.5


@pytest.mark.parametrize(
    "front",
    [
        "---\nclassification: [secret]\n---\n",
        "---\nclassification:\n  - secret\n---\n",
        "---\nclassification: [[secret]]\n---\n",
        "---\nclassification: {level: secret}\n---\n",
        "---\nclassification: !!set {secret}\n---\n",
        "---\nclassification: [public]\n---\n",  # non-scalar: cannot be read as the loader does
        "---\nmeta:\n  classification: [personal]\n---\n",
        "---\nclassification:   SECRET  \n---\n",
    ],
    ids=["list", "block-list", "nested-list", "mapping", "set", "list-public", "nested", "padded"],
)
def test_non_scalar_classification_values_withhold(tree: Path, front: str) -> None:
    _write(tree / "docs/architecture/fm.md", f"{front}# F\n\n{CANARY}\n")
    _no_canary(_search(tree, query=CANARY))
    assert build_corpus(load_config(), tree).dropped == 1


def _jwt(n: int) -> str:
    return (
        "eyJhbGciOiJIUzI1NiJ9."
        + "eyJzdWIiOiIx"
        + "A" * n
        + ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )


@pytest.mark.parametrize("size", [300, 600, 1500, 3000])
@pytest.mark.parametrize("offset", [3500, 3800, 4100, 4300, 7500, 7900, 8300])
def test_a_long_line_cannot_hide_a_jwt_across_a_chunk_cut(
    tree: Path, size: int, offset: int
) -> None:
    token = _jwt(size)
    assert scan_text(token).is_secret  # the premise
    _write(
        tree / "docs/guides/long.md",
        f"# L\n\n{CANARY} " + "word " * (offset // 5) + token + " tail " * 500 + "\n",
    )
    _no_canary(_search(tree, query=CANARY))
    assert build_corpus(load_config(), tree).dropped == 1


@pytest.mark.parametrize(
    "marker",
    ["eyJ" + "a" * 40, "sk-" + "a" * 30, "ghp_" + "a" * 30, "AKIA" + "A" * 16, "-----" + "BEGIN"],
)
def test_long_line_marker_precheck_withholds(tree: Path, marker: str) -> None:
    _write(tree / "docs/guides/m.md", f"# M\n\n{CANARY} " + "x " * 3000 + marker + " end\n")
    _no_canary(_search(tree, query=CANARY))


def test_long_line_marker_precheck_ignores_ordinary_words(tree: Path) -> None:
    prose = "the task-list and ask-me items " * 400  # contains "sk-" inside words
    _write(tree / "docs/guides/prose.md", f"# P\n\n{CANARY} {prose}\n")
    assert "docs/guides/prose.md" in _search(tree, query=CANARY)


def test_cuts_do_not_land_inside_a_token_run() -> None:
    token = "T" * 60
    for pad in range(3900, 4100, 7):  # the token straddles the 4000 mark at some pad
        line = "a " * (pad // 2) + token + " z" * 500
        assert any(token in c for c in corpus_mod._chunks(line, 4000)), pad


def test_every_answer_says_what_was_withheld_and_what_is_pending(tree: Path) -> None:
    _write(tree / "docs/guides/leaky.md", f"# L\n\n{SECRET_TEXT}\n")
    for query in ("tier routing", "nonexistentwordzzz"):
        out = _search(tree, query=query)
        assert "1 documents withheld (secret/personal/over scan budget), 0 not yet scanned" in out


def test_twelve_hostile_documents_never_block_a_call_and_converge(
    tree: Path, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    for i in range(12):
        _write(tree / f"docs/guides/h{i}.md", f"# H{i}\n\n" + "+1-" * 60000 + "\n")
    _write_config(tmp_path, monkeypatch, ROOTS, scan_total_ms=1000, scan_chunk_chars=1000)
    corpus_mod._CACHE.clear()
    corpus_mod._PROGRESS.clear()
    pending = []
    for _ in range(40):
        start = time.perf_counter()
        out = _search(tree, query="tier routing")
        assert time.perf_counter() - start < 5
        assert " not yet scanned" in out
        pending.append(
            int(out.rsplit("withheld (secret/personal/over scan budget), ", 1)[1].split()[0])
        )
        if pending[-1] == 0:
            break
    assert pending[0] > 0 and pending[-1] == 0, pending


def test_the_directory_swap_does_not_mislabel_a_document(tree: Path, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The name and cache key come from the path the descriptor names, not the listed one:
    when a listed file's descriptor really names another file of the root, the document is
    labelled as that file, never by the listing."""
    _write(tree / "docs/guides/other.md", "# Other\n\n## O\n\nother body\n")
    _write_config(tmp_path, monkeypatch, ["docs/guides"])
    named = os.path.realpath(tree / "docs/guides/setup.md")
    monkeypatch.setattr(corpus_mod, "_fd_path", lambda fd: named)
    names = [d.name for d in build_corpus(load_config(), tree).docs]
    assert names and set(names) == {"docs/guides/setup.md"}


def test_a_document_longer_than_one_calls_scan_time_still_finishes(
    tree: Path, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    _write(
        tree / "docs/guides/long.md", "# Long\n\n" + "\n".join(f"line {i} body" for i in range(300))
    )
    _write_config(tmp_path, monkeypatch, ROOTS, scan_total_ms=1, scan_chunk_chars=100)
    corpus_mod._CACHE.clear()
    corpus_mod._PROGRESS.clear()
    real = corpus_mod.scan_text

    def slow(text: str):  # type: ignore[no-untyped-def]
        time.sleep(0.003)  # more than the call's whole scan time: one chunk per call
        return real(text)

    monkeypatch.setattr(corpus_mod, "scan_text", slow)
    for _ in range(400):  # one chunk of progress per call at least
        if " 0 not yet scanned" in _search(tree, query="body"):
            break
    else:
        pytest.fail("the deferred document never finished scanning")
    assert "docs/guides/long.md" in _search(tree, query="body")


def test_a_hard_linked_file_is_refused(tree: Path, tmp_path: Path) -> None:
    outside = _write(tmp_path / "outside/secret.md", f"# S\n\n{CANARY}\n")
    os.link(outside, tree / "docs/guides/hard.md")
    _no_canary(_search(tree, query=CANARY))
    # a file whose other name is inside the tree is refused as well: nlink > 1
    os.link(tree / "docs/guides/setup.md", tree / "docs/guides/setup2.md")
    assert "docs/guides/setup" not in _search(tree, query="poetry install")
