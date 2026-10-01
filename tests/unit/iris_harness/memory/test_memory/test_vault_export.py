"""Exporting memory as a vault: linked, one-way, confirmed by default.

The wiki's connector sat as a "Phase 4" stub while the thing it was meant to sync filled
with 2,505 pages nobody read. What is worth exporting is the memory graph — and the
export stays one-way, because a vault edit flowing back would undo the confirmation gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_harness.memory import graph as graph_module
from iris_harness.memory.export import export_memory_vault
from iris_harness.memory.knowledge.connectors.obsidian import ObsidianConnector
from iris_harness.memory.store import MemoryStore, UserFact

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")


@pytest.fixture(autouse=True)
def _fresh_config() -> None:
    graph_module.reset_config_cache()


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _fact(key: str, value: str, confirmed: bool = True) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(
        key=key,
        value=value,
        confidence=0.9,
        source="test",
        first_seen=now,
        last_confirmed=now,
        confirmed=confirmed,
    )


@pytest.fixture
def seeded(store: MemoryStore) -> MemoryStore:
    store.upsert_user_fact(_fact("bank", "Northwind Bank"))
    store.upsert_user_fact(_fact("employer", "Quant Academy"))
    store.upsert_user_fact(_fact("name", "ollama", confirmed=False))
    store.save_conversation_turns("s1", [("user", "q"), ("assistant", "a")])
    store.save_conversation_summary(
        "s1", "Goal: the flat\nOpen items: none\nReferenced: Petra Sutton, Northwind Bank"
    )
    return store


class TestWhatLandsOnDisk:
    def test_a_note_per_node_plus_an_index(self, seeded: MemoryStore, tmp_path: Path) -> None:
        out = tmp_path / "vault"

        result = export_memory_vault(seeded, out)

        assert (out / "index.md").exists()
        assert (out / "entities" / "Northwind Bank.md").exists()
        assert result.notes == len(list(out.rglob("*.md"))) - 1  # index is not a node

    def test_notes_carry_frontmatter_and_wikilinks(
        self, seeded: MemoryStore, tmp_path: Path
    ) -> None:
        out = tmp_path / "vault"
        export_memory_vault(seeded, out)

        text = (out / "entities" / "Northwind Bank.md").read_text(encoding="utf-8")

        assert text.startswith("---\n")
        assert "kind: entity" in text
        assert "# Northwind Bank" in text
        assert "[[You]]" in text  # the edge back to the centre

    def test_the_index_lists_every_note(self, seeded: MemoryStore, tmp_path: Path) -> None:
        out = tmp_path / "vault"
        export_memory_vault(seeded, out)

        index = (out / "index.md").read_text(encoding="utf-8")

        assert "[[Northwind Bank]]" in index
        assert "[[Quant Academy]]" in index
        assert "read back" in index  # the one-way rule, stated in the vault itself

    def test_a_session_note_carries_its_summary(self, seeded: MemoryStore, tmp_path: Path) -> None:
        out = tmp_path / "vault"
        export_memory_vault(seeded, out)

        text = (out / "sessions" / "s1.md").read_text(encoding="utf-8")

        assert "Goal: the flat" in text


class TestConfirmedByDefault:
    def test_unconfirmed_items_are_left_out_and_counted(
        self, seeded: MemoryStore, tmp_path: Path
    ) -> None:
        out = tmp_path / "vault"

        result = export_memory_vault(seeded, out)

        assert not (out / "facts" / "name: ollama.md").exists()
        assert result.skipped_unconfirmed > 0
        assert "unconfirmed item(s) were left out" in (out / "index.md").read_text()

    def test_including_them_marks_them_loudly(self, seeded: MemoryStore, tmp_path: Path) -> None:
        out = tmp_path / "vault"

        export_memory_vault(seeded, out, include_unconfirmed=True)

        note = next(out.rglob("*ollama*.md"))
        text = note.read_text(encoding="utf-8")
        assert "confirmed: false" in text
        assert "Not confirmed" in text
        assert "not used in prompts" in text


class TestItIsASnapshot:
    def test_re_export_reflects_the_current_state(
        self, seeded: MemoryStore, tmp_path: Path
    ) -> None:
        out = tmp_path / "vault"
        export_memory_vault(seeded, out)
        assert (out / "entities" / "Northwind Bank.md").exists()

        # "Quant Academy" comes only from the employer fact; Northwind Bank is also named in
        # the session summary, so forgetting the fact would not remove that node.
        seeded.delete_user_fact("employer")
        result = export_memory_vault(seeded, out)

        index = (out / "index.md").read_text(encoding="utf-8")
        assert "[[Quant Academy]]" not in index  # gone from the view, gone from the index
        assert not (out / "entities" / "Quant Academy.md").exists()  # and off disk
        assert result.removed_stale == 1

    def test_an_empty_store_still_writes_a_usable_vault(
        self, store: MemoryStore, tmp_path: Path
    ) -> None:
        out = tmp_path / "vault"

        result = export_memory_vault(store, out)

        assert (out / "index.md").exists()
        assert result.notes >= 1  # "You"

    def test_a_name_with_slashes_does_not_escape_the_folder(
        self, store: MemoryStore, tmp_path: Path
    ) -> None:
        store.upsert_user_fact(_fact("employer", "Acme/Evil Corp: Ltd"))
        out = tmp_path / "vault"

        export_memory_vault(store, out)

        written = list((out / "entities").glob("*.md"))
        assert written and all(p.parent == out / "entities" for p in written)


class TestPruning:
    def test_a_note_the_owner_wrote_is_never_deleted(
        self, seeded: MemoryStore, tmp_path: Path
    ) -> None:
        out = tmp_path / "vault"
        export_memory_vault(seeded, out)
        mine = out / "my own note.md"
        mine.write_text("# My own note\n\nNothing to do with IRIS.\n", encoding="utf-8")

        export_memory_vault(seeded, out)

        assert mine.exists()


class TestTheConnector:
    def test_export_writes_the_vault(self, seeded: MemoryStore, tmp_path: Path) -> None:
        result = ObsidianConnector(tmp_path / "vault").export(seeded)

        assert result.pages_exported > 0
        assert result.success

    def test_importing_from_a_vault_is_refused_on_purpose(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError, match="deliberately unsupported"):
            ObsidianConnector(tmp_path).sync_from(tmp_path)

    def test_the_old_wiki_sync_points_at_the_new_export(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError, match="ADR-0114"):
            ObsidianConnector(tmp_path).sync_to(tmp_path)
