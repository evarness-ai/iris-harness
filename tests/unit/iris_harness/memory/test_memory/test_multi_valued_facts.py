"""A key that holds several values — two cards — is edited and forgotten one value at a time.

Found on the owner's own store (2026-09-19): the Memory page keyed rows by fact key, so
the two credit cards shared a React key, and Forget / Edit worked by key — forgetting
one card retracted both, and editing one ADDED a third. The recall index and USER.md
also keep a key once, and each write replaced the line with the last card only.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.memory import coherence
from iris_harness.memory.coordinator import FactCoordinator
from iris_harness.memory.identity import loader
from iris_harness.memory.store import MemoryStore
from iris_harness.server.iris_api.main import create_app

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")

CARDS = ("Discover", "Wingtip Bank Credit Card")


@pytest.fixture
def user_md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "workspace" / "USER.md"
    monkeypatch.setattr(loader, "USER_MD_PATH", path)
    return path


@pytest.fixture
def store(tmp_path: Path, user_md: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    coordinator = FactCoordinator(s, None)
    coordinator.record("employer", "US Cards", 0.9, "test", confirmed=True)
    for card in CARDS:
        coordinator.record("credit_card", card, 0.9, "test", confirmed=True)
    return s


def _cards(store: MemoryStore) -> list[str]:
    return sorted(f.value for f in store.fetch_all_user_facts() if f.key == "credit_card")


def _id(store: MemoryStore, value: str) -> str:
    fact = next(f for f in store.fetch_all_user_facts() if f.value == value)
    assert fact.statement_id is not None
    return fact.statement_id


def _md_line(user_md: Path, key: str) -> str | None:
    return next(
        (ln for ln in user_md.read_text(encoding="utf-8").splitlines() if f"**{key}**" in ln),
        None,
    )


def test_every_fact_carries_its_own_id(store: MemoryStore) -> None:
    ids = [f.statement_id for f in store.fetch_all_user_facts()]
    assert all(ids) and len(set(ids)) == 3


class TestForgetOne:
    def test_forgetting_one_card_keeps_the_other(self, store: MemoryStore, user_md: Path) -> None:
        FactCoordinator(store, None).forget("credit_card", statement_id=_id(store, "Discover"))

        assert _cards(store) == ["Wingtip Bank Credit Card"]
        line = _md_line(user_md, "credit_card")
        assert line is not None and "Wingtip" in line and "Discover" not in line

    def test_without_an_id_the_whole_key_goes_as_before(
        self, store: MemoryStore, user_md: Path
    ) -> None:
        FactCoordinator(store, None).forget("credit_card")

        assert _cards(store) == []
        assert _md_line(user_md, "credit_card") is None
        assert _md_line(user_md, "employer") is not None  # other keys untouched

    def test_an_id_of_another_key_forgets_nothing(self, store: MemoryStore) -> None:
        assert not FactCoordinator(store, None).forget(
            "credit_card", statement_id=_id(store, "US Cards")
        )
        assert _cards(store) == sorted(CARDS)


class TestEditOne:
    def test_editing_one_card_replaces_it_instead_of_adding_a_third(
        self, store: MemoryStore, user_md: Path
    ) -> None:
        FactCoordinator(store, None).correct(
            "credit_card", "Discover It", statement_id=_id(store, "Discover")
        )

        assert _cards(store) == ["Discover It", "Wingtip Bank Credit Card"]
        [edit] = [e for e in store.fetch_fact_history("credit_card") if e.reason == "correct"]
        assert (edit.old_value, edit.new_value) == ("Discover", "Discover It")
        line = _md_line(user_md, "credit_card")
        assert line is not None and "Discover It" in line and "Wingtip" in line

    def test_a_single_valued_key_is_superseded_as_before(self, store: MemoryStore) -> None:
        FactCoordinator(store, None).correct(
            "employer", "Infosys", statement_id=_id(store, "US Cards")
        )
        assert [f.value for f in store.fetch_all_user_facts() if f.key == "employer"] == ["Infosys"]


class TestDerivedHomes:
    def test_one_projection_per_key_lists_every_value(self, store: MemoryStore) -> None:
        projections = {p.key: p.value for p in store.fetch_fact_projections()}
        assert projections["credit_card"] == "Discover, Wingtip Bank Credit Card"
        assert projections["employer"] == "US Cards"

    def test_usermd_holds_both_cards_and_the_doctor_agrees(
        self, store: MemoryStore, user_md: Path
    ) -> None:
        line = _md_line(user_md, "credit_card")

        assert line is not None and "Discover" in line and "Wingtip" in line
        report = coherence.diagnose(store, None)
        assert report.md_value_mismatches == []


class TestTheApi:
    @pytest.fixture
    def client(self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
        monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
        runtime = SimpleNamespace(memory_store=store, semantic_index=None)
        with TestClient(
            create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
        ) as c:
            yield c

    def test_rows_have_ids_and_forget_and_edit_take_one(
        self, client: TestClient, store: MemoryStore
    ) -> None:
        rows = client.get("/memory/facts").json()["facts"]
        ids = {r["value"]: r["id"] for r in rows}
        assert len({r["id"] for r in rows}) == len(rows) == 3

        client.patch(
            "/memory/facts/credit_card", json={"value": "Discover It", "id": ids["Discover"]}
        )
        assert _cards(store) == ["Discover It", "Wingtip Bank Credit Card"]

        wingtip = ids["Wingtip Bank Credit Card"]
        forgot = client.post(f"/memory/facts/credit_card/forget?id={wingtip}")
        assert forgot.status_code == 200
        assert _cards(store) == ["Discover It"]
