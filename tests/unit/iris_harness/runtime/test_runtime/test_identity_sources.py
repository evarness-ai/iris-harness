"""The composition root's owner-identity sources (ADR-0125, PR 2).

``runtime/identity_redaction.py`` registers three sources at the kernel's seam: the
identity documents, the USER.md ``identity:`` block, and the owner's confirmed facts
mapped through ``config/governance/identity.yaml``. These tests read each one against real
files and a real memory store, and then prove what the sources change at the guards: only
the egress guard's links, by the owner's confirmed ``blog`` and ``website`` (PR 3); every
other guard's literal set is exactly the documents-only one.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governance.identity_config import IdentityConfig, load_identity_config
from iris_harness.kernel.governance.owner_identity import DECLARABLE_KINDS, extract
from iris_harness.kernel.governance.plugins.network_egress import NetworkEgress
from iris_harness.kernel.governance.plugins.response_safety import identity_literals
from iris_harness.memory.identity import loader
from iris_harness.memory.ontology import memory_ontology
from iris_harness.memory.store import MemoryStore, UserFact
from iris_harness.runtime import identity_redaction as root

NOW = datetime(2026, 9, 30, tzinfo=UTC)
SECRET = "CANARY_SOUL_SECRET_DIRECTIVE_d41f8a27"
LINK = "www.web3notes.example"

USER_MD = f"""---
classification: personal
load: always
egress: local-only
identity:
  names: [Robin Example, Robin]
  emails: [robin@mail.example]
  phones: ["+1 555 123 4567"]
  addresses: ["1 Example Street, Springfield"]
  handles: [robin-gh]
  never_match: [Robin, {LINK}, {SECRET}]
---

# User Profile

My key is {SECRET}; my blog is {LINK}.
"""


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    ws = tmp_path / ".iris" / "workspace"
    monkeypatch.setattr(loader, "SOUL_PATH", ws / "SOUL.md")
    monkeypatch.setattr(loader, "USER_MD_PATH", ws / "USER.md")
    monkeypatch.setattr(loader, "AGENTS_MD_PATH", ws / "AGENTS.md")
    monkeypatch.setattr(loader, "_LEGACY_SOUL_PATH", tmp_path / "legacy" / "soul.md")
    monkeypatch.setattr(loader, "_LEGACY_USER_MD_PATH", tmp_path / "legacy" / "user.md")
    monkeypatch.setattr(loader, "MEMORY_DIR", tmp_path / ".iris" / "memory")
    ws.mkdir(parents=True)
    return ws


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _fact(store: MemoryStore, key: str, value: str, *, confirmed: bool = True) -> None:
    store.upsert_user_fact(UserFact(key, value, 0.9, "test", NOW, NOW, 1, confirmed))


def _facts_source(store: MemoryStore) -> root.OwnerFacts:
    source = root.OwnerFacts()
    source.db_path = store.db_path
    return source


# -- the USER.md identity: block -------------------------------------------------------


def test_the_identity_block_is_read_by_kind(home: Path) -> None:
    (home / "USER.md").write_text(USER_MD, encoding="utf-8")
    literals = {k: list(v) for k, v in root.user_md_identity().items()}
    assert literals == {
        "name": ["Robin Example", "Robin"],
        "email": ["robin@mail.example"],
        "phone": ["+1 555 123 4567"],
        "address": ["1 Example Street, Springfield"],
        "handle": ["robin-gh"],
        "never_match": ["Robin", LINK, SECRET],
    }
    # load_user_md still strips the frontmatter: the block never reaches a prompt.
    body = loader.load_user_md()
    assert body is not None and "identity:" not in body and "Springfield" not in body


def test_no_block_or_no_file_is_empty(home: Path) -> None:
    assert loader.load_user_identity() == loader.UserIdentityBlock()
    (home / "USER.md").write_text("# User Profile\n", encoding="utf-8")
    assert loader.load_user_identity() == loader.UserIdentityBlock()


@pytest.mark.parametrize(
    "block",
    [
        "identity:\n  nmaes: [Robin]",  # a typo: extra keys are refused
        "identity:\n  names: Robin",  # not a list
        "identity: [Robin]",  # not a mapping
    ],
)
def test_a_bad_block_warns_and_reads_as_empty(
    home: Path, block: str, caplog: pytest.LogCaptureFixture
) -> None:
    (home / "USER.md").write_text(f"---\n{block}\n---\n\n# User Profile\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert loader.load_user_identity() == loader.UserIdentityBlock()
    assert "identity: block is invalid" in caplog.text


def test_the_user_md_fingerprint_moves_with_the_file(home: Path) -> None:
    before = root.user_md_fingerprint()
    (home / "USER.md").write_text(USER_MD, encoding="utf-8")
    after = root.user_md_fingerprint()
    assert before != after
    assert root.user_md_fingerprint() == after


def test_a_user_md_writer_invalidates_the_corpus(home: Path, owner_identity_seam: Any) -> None:
    seam = owner_identity_seam
    seam.set_owner_identity_clock(lambda: 0.0)  # no interval passes: only a write can tell
    seam.register_identity_text_provider(root.identity_documents)
    seam.register_owner_identity_source(root.USER_MD_SOURCE, root.user_md_identity)
    (home / "USER.md").write_text(USER_MD, encoding="utf-8")
    assert "robin@mail.example" in seam.owner_identity().of("email")
    loader.append_user_fact_to_md("phone", "+44 20 7946 0958", 0.9)
    assert "+44 20 7946 0958" in seam.owner_identity().of("phone")


# -- identity.yaml -----------------------------------------------------------------------


def test_every_ontology_kinds_key_is_an_ontology_attribute() -> None:
    config = load_identity_config()
    ontology = memory_ontology()
    assert config.ontology_kinds, "identity.yaml maps no attribute"
    assert {ontology.qualify(a) for a in config.ontology_kinds} <= set(ontology.attributes)
    assert set(config.ontology_kinds.values()) <= set(DECLARABLE_KINDS)


def test_identity_yaml_is_strict(tmp_path: Path) -> None:
    bad = tmp_path / "identity.yaml"
    bad.write_text("ontology_kind:\n  name: name\n", encoding="utf-8")
    with pytest.raises(ValueError):
        IdentityConfig.from_yaml(bad)
    bad.write_text("ontology_kinds:\n  blog: secret\n", encoding="utf-8")
    with pytest.raises(ValueError):
        IdentityConfig.from_yaml(bad)
    assert IdentityConfig.from_yaml(tmp_path / "absent.yaml") == IdentityConfig()


# -- the owner's confirmed facts ---------------------------------------------------------


def test_only_the_owners_confirmed_facts_count(store: MemoryStore) -> None:
    _fact(store, "name", "Robin Example")
    _fact(store, "email", "robin@mail.example")
    _fact(store, "github", "robin-gh")
    _fact(store, "city", "Springfield")  # not an identity attribute
    # Unconfirmed, however confident: the store once held name=ollama at 1.0.
    store.add_fact_proposal(key="phone", value="+1 555 999 0000", confidence=1.0, source="t")
    _fact(store, "nickname", "Ollama", confirmed=False)
    # A contact's name, confirmed: about Petra, never the owner.
    _fact(store, "spouse", "Petra")
    pid = store.add_fact_proposal(
        key="name",
        value="Petra Example",
        confidence=0.9,
        source="t",
        subject="Petra",
        subject_class="Person",
    )
    assert pid is not None and store.resolve_fact_proposal(pid, "approved")

    literals = {k: sorted(v) for k, v in _facts_source(store)().items()}
    assert literals == {
        "name": ["Robin Example"],
        "email": ["robin@mail.example"],
        "handle": ["robin-gh"],
    }


def test_confirming_a_fact_adds_it(store: MemoryStore) -> None:
    store.add_fact_proposal(key="phone", value="+1 555 999 0000", confidence=0.9, source="t")
    source = _facts_source(store)
    assert "phone" not in source()
    assert store.set_fact_confirmed("phone")
    assert source()["phone"] == ["+1 555 999 0000"]


def test_no_database_reads_as_nothing_and_creates_nothing(tmp_path: Path) -> None:
    source = root.OwnerFacts()
    source.db_path = tmp_path / "absent" / "memory.db"
    assert source() == {}
    assert not source.db_path.exists()
    assert source.fingerprint() == (str(source.db_path), None)


def test_the_facts_fingerprint_moves_on_another_connections_commit(store: MemoryStore) -> None:
    source = _facts_source(store)
    first = source.fingerprint()
    assert source.fingerprint() == first, "nothing written, nothing moved"
    other = MemoryStore(db_path=store.db_path)  # another writer, as another process is
    _fact(other, "name", "Robin Example")
    assert source.fingerprint() != first


def test_the_facts_fingerprint_moves_on_a_status_flip(store: MemoryStore) -> None:
    """Confirming rewrites a row in place: a row count would not move. data_version does."""
    store.add_fact_proposal(key="phone", value="+1 555 999 0000", confidence=0.9, source="t")
    source = _facts_source(store)
    before = source.fingerprint()
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM memris_statements").fetchone()
    MemoryStore(db_path=store.db_path).set_fact_confirmed("phone")
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM memris_statements").fetchone() == rows
    assert source.fingerprint() != before


def test_a_fact_write_in_process_invalidates_the_corpus(
    store: MemoryStore, owner_identity_seam: Any
) -> None:
    seam = owner_identity_seam
    seam.set_owner_identity_clock(lambda: 0.0)  # no interval passes: only a write can tell
    seam.register_identity_text_provider(list)
    source = _facts_source(store)
    seam.register_owner_identity_source(root.FACTS_SOURCE, source, fingerprint=source.fingerprint)
    assert seam.owner_identity().of("email") == frozenset()
    _fact(store, "email", "robin@mail.example")
    assert seam.owner_identity().of("email") == {"robin@mail.example"}
    store.delete_user_fact("email")
    assert seam.owner_identity().of("email") == frozenset()


def _frozen_facts_seam(seam: Any, store: MemoryStore) -> None:
    seam.set_owner_identity_clock(lambda: 0.0)  # no interval passes: only a write can tell
    seam.register_identity_text_provider(list)
    source = _facts_source(store)
    seam.register_owner_identity_source(root.FACTS_SOURCE, source, fingerprint=source.fingerprint)


def test_a_proposal_does_not_invalidate(store: MemoryStore, owner_identity_seam: Any) -> None:
    """A proposal is unconfirmed: it cannot change the corpus, so it must not cost a rebuild."""
    seam = owner_identity_seam
    _frozen_facts_seam(seam, store)
    first = seam.owner_identity()
    assert store.add_fact_proposal(key="phone", value="+1 555 999 0000", confidence=1.0, source="t")
    assert seam.owner_identity() is first


def test_every_write_that_can_change_the_confirmed_set_invalidates(
    store: MemoryStore, owner_identity_seam: Any
) -> None:
    seam = owner_identity_seam
    _frozen_facts_seam(seam, store)

    def changed() -> bool:
        before = seam.owner_identity()
        return seam.owner_identity() is not before

    corpus = seam.owner_identity()
    _fact(store, "email", "robin@mail.example")  # put
    assert seam.owner_identity() is not corpus
    assert seam.owner_identity().of("email") == {"robin@mail.example"}

    corpus = seam.owner_identity()
    assert store.set_fact_confirmed("email", False)  # set_confirmed
    assert seam.owner_identity() is not corpus
    assert seam.owner_identity().of("email") == frozenset()

    pid = store.add_fact_proposal(key="phone", value="+1 555 999 0000", confidence=0.9, source="t")
    assert pid is not None
    corpus = seam.owner_identity()
    assert store.resolve_fact_proposal(pid, "approved")  # resolve
    assert seam.owner_identity() is not corpus
    assert seam.owner_identity().of("phone") == {"+1 555 999 0000"}

    corpus = seam.owner_identity()
    assert store.delete_user_fact("phone")  # forget
    assert seam.owner_identity() is not corpus
    assert seam.owner_identity().of("phone") == frozenset()
    assert not changed()


def test_the_runtime_points_the_facts_source_at_its_store(
    store: MemoryStore, owner_identity_seam: Any
) -> None:
    saved = root._FACTS.db_path
    try:
        root.use_memory_db(store.db_path)
        assert root._FACTS.path() == store.db_path
    finally:
        root._FACTS.db_path = saved


def test_the_composition_root_registers_every_source() -> None:
    import importlib

    from iris_harness.kernel.governance.identity_redaction import owner_identity_sources

    importlib.reload(root)
    assert {"documents", root.USER_MD_SOURCE, root.FACTS_SOURCE} <= set(owner_identity_sources())


# -- what the sources change at the guards ------------------------------------------------

BLOG = "https://blog.robin.example/2026"
WEBSITE = "robin.example"


def _guard_sets() -> tuple[frozenset[str], frozenset[str]]:
    NetworkEgress._reset_identity_cache()
    return NetworkEgress._identity_secret_literals(), identity_literals()


def test_only_the_confirmed_blog_and_website_move_a_guard(
    home: Path, store: MemoryStore, owner_identity_seam: Any
) -> None:
    """Neutrality, but for the one intended change (ADR-0125 amendment 6).

    Every source is populated -- the identity block (names, an address, and a never_match
    naming the document's link and secret), the owner's confirmed facts, a plugin's account
    address. The egress guard now also refuses the owner's confirmed blog and website (they
    are ``link``); the response check's set is still what the documents alone gave it. A
    guard that starts asking for a new kind (PR 4, shadow first), or a never_match that
    reaches a kind a guard acts on, fails here until this test is changed on purpose.
    """
    seam = owner_identity_seam
    documents = [f"{USER_MD}\nwrite to robin@mail.example or +44 20 7946 0958"]
    (home / "USER.md").write_text(USER_MD, encoding="utf-8")
    _fact(store, "name", "Robin Example")
    _fact(store, "nickname", "Rob")
    _fact(store, "email", "robin.work@mail.example")
    _fact(store, "phone", "+1 555 123 4567")
    _fact(store, "github", "robin-gh")
    _fact(store, "linkedin", "robin-example-1234")
    _fact(store, "blog", BLOG)
    _fact(store, "website", WEBSITE)

    seam.register_identity_text_provider(lambda: documents)
    documents_only = _guard_sets()
    assert LINK in documents_only[0] and SECRET in documents_only[1]  # what never_match names
    expected = extract(documents)
    assert documents_only == (expected.of("secret", "link"), expected.of("secret"))

    seam.register_owner_identity_source(root.USER_MD_SOURCE, root.user_md_identity)
    source = _facts_source(store)
    seam.register_owner_identity_source(root.FACTS_SOURCE, source)
    seam.register_owner_identity_source(
        "plugin:mail", lambda: {"email": ["robin.mail@mail.example"]}, kinds=("email",)
    )
    corpus = seam.owner_identity()
    assert corpus is not None
    # The new sources did land in the corpus ...
    assert {"Robin Example", "Rob"} <= corpus.of("name")
    assert corpus.of("address") == {"1 Example Street, Springfield"}
    assert {"robin-gh", "robin-example-1234"} <= corpus.of("handle")
    assert "robin.mail@mail.example" in corpus.of("email")
    assert corpus.of("link") == documents_only[0] - {SECRET} | {BLOG, WEBSITE}
    # ... and only egress moved, by the confirmed blog and website.
    egress, response = _guard_sets()
    assert egress == documents_only[0] | {BLOG, WEBSITE}
    assert response == documents_only[1]
