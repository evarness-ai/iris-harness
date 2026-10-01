"""The Graphiti connector: a synthetic export into memris, with its report (memris PR 9).

The export is invented (no personal data) and follows Graphiti's documented data model;
see reader.py for the field names assumed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from iris_harness.plugins_builtin.graphiti_import import cli as graphiti_cli
from iris_harness.plugins_builtin.graphiti_import.reader import read_export
from iris_harness.sdk import PluginCLI, PluginManifest
from memris.graph import MemoryGraph
from memris.model import Statement
from memris.ontology import load_or_raise
from memris.store import InMemoryGraphStore

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ONTOLOGY_DIR = REPO / "config" / "memory"
PLUGIN_DIR = REPO / "src" / "iris_harness" / "plugins_builtin" / "graphiti_import"


def _t(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


@pytest.fixture
def export() -> dict:  # type: ignore[type-arg]
    return json.loads((HERE / "graphiti_export.json").read_text(encoding="utf-8"))


@pytest.fixture
def graph() -> MemoryGraph:
    return MemoryGraph(load_or_raise(ONTOLOGY_DIR), InMemoryGraphStore())


def _stmt(graph: MemoryGraph, edge_uuid: str) -> Statement:
    s = graph.store.get_statement(f"st_graphiti_{edge_uuid}")
    assert s is not None, edge_uuid
    return s


def test_an_edge_keeps_both_time_axes(graph: MemoryGraph, export: dict) -> None:  # type: ignore[type-arg]
    report = graphiti_cli.import_export(export, graph)
    old = _stmt(graph, "e-acme")
    assert (old.predicate, old.status) == ("mem:works_at", "proposed")
    assert old.recorded_at == _t("2023-01-05T10:00:00")  # Graphiti learned it
    assert (old.valid_from, old.valid_to) == (_t("2023-01-01T00:00:00"), _t("2024-06-01T00:00:00"))
    assert (old.source_episode, old.evidence, old.extractor) == (
        "ep-1",
        "Ana works at Acme Corp.",
        "graphiti",
    )
    notes = " | ".join(note for sid, note in report.notes if sid == "e-acme")
    assert "record-time end" in notes and "2 episodes" in notes


def test_expired_without_invalid_becomes_the_end_of_validity_and_says_so(graph: MemoryGraph, export: dict) -> None:  # type: ignore[type-arg]
    report = graphiti_cli.import_export(export, graph)
    leeds = _stmt(graph, "e-leeds")
    assert leeds.valid_to == _t("2024-01-01T00:00:00") and leeds.predicate == "mem:lives_in"
    assert "used as valid_to" in dict(report.notes)["e-leeds"]


def test_what_held_when_is_answerable_after_import(graph: MemoryGraph, export: dict) -> None:  # type: ignore[type-arg]
    graphiti_cli.import_export(export, graph, confirm=True)
    [ana] = graph.store.find_entities(label="Ana Lima")

    def employer(at: str) -> list[str]:
        found = graph.current(ana.id, "works_at", as_of=_t(at))
        return [graph.get_entity(s.object_id).label for s in found]  # type: ignore[arg-type, union-attr]

    assert employer("2023-06-01T00:00:00") == ["Acme Corp"]
    assert employer("2025-01-01T00:00:00") == ["Globex"]


def test_everything_not_written_is_in_the_report(graph: MemoryGraph, export: dict) -> None:  # type: ignore[type-arg]
    report = graphiti_cli.import_export(export, graph)
    skipped = dict(report.skipped)
    assert report.unmapped_types == {"graphiti_edge:ENJOYS": 1}  # a relation with no mapping
    assert "outside the range" in skipped["e-studies"]  # "Something" is no Topic
    assert "no edge" in skipped["n-lonely"]  # a node with nothing said about it
    assert "no mapping" in skipped["e-likes"]
    assert not report.lossless


def test_mentions_come_in_as_episode_statements(graph: MemoryGraph, export: dict) -> None:  # type: ignore[type-arg]
    graphiti_cli.import_export(export, graph)
    mention = _stmt(graph, "m-2")
    episode = graph.get_entity(mention.subject_id)
    assert mention.predicate == "mem:mentions"
    assert episode is not None and episode.class_ == "mem:Conversation"
    assert graph.get_entity(mention.object_id).label == "Globex"  # type: ignore[arg-type, union-attr]


def test_a_company_label_reads_as_an_organization(graph: MemoryGraph, export: dict) -> None:  # type: ignore[type-arg]
    graphiti_cli.import_export(export, graph)
    [globex] = graph.store.find_entities(label="Globex")
    assert globex.class_ == "mem:Organization"


def test_importing_twice_replaces_rather_than_duplicates(graph: MemoryGraph, export: dict) -> None:  # type: ignore[type-arg]
    first = graphiti_cli.import_export(export, graph)
    counts = (len(graph.store.statements()), len(graph.store.find_entities()))
    again = graphiti_cli.import_export(export, graph)
    assert again.already_present == first.statements
    assert (len(graph.store.statements()), len(graph.store.find_entities())) == counts


def test_a_malformed_timestamp_is_reported_not_raised(export: dict) -> None:  # type: ignore[type-arg]
    export["edges"][0]["valid_at"] = "last spring"
    read = read_export(export)
    assert "unreadable timestamp" in dict(read.skipped)["e-acme"]


def test_the_command_prints_the_report_and_fails_when_not_lossless(tmp_path: Path, export: dict) -> None:  # type: ignore[type-arg]
    root = typer.Typer()
    graphiti_cli.register(PluginCLI(root=root))
    path = tmp_path / "export.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    args = [
        "graphiti",
        "import",
        str(path),
        "--db",
        str(tmp_path / "m.db"),
        "--ontology-dir",
        str(ONTOLOGY_DIR),
    ]
    result = CliRunner().invoke(root, args)
    assert result.exit_code == 1, result.output
    assert "unmapped relation graphiti_edge:ENJOYS" in result.output
    assert "imported as proposals" in result.output

    lossless = {k: export[k] for k in ("nodes", "episodes")}
    lossless["nodes"] = [n for n in export["nodes"] if n["uuid"] in {"n-ana", "n-acme"}]
    lossless["edges"] = [export["edges"][0]]
    path.write_text(json.dumps(lossless), encoding="utf-8")
    assert CliRunner().invoke(root, args).exit_code == 0


def test_the_manifest_is_valid_and_contributes_only_a_command() -> None:
    import yaml

    manifest = PluginManifest.model_validate(
        yaml.safe_load((PLUGIN_DIR / "manifest.yaml").read_text())
    )
    assert (manifest.name, manifest.cli, manifest.provides) == (
        "graphiti_import",
        "cli:register",
        (),
    )


@pytest.mark.usefixtures("test_vocabulary")
def test_the_command_imports_with_the_plugins_vocabulary_too(
    tmp_path: Path, export: dict, monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    """memris PR 10: plugin vocabulary (the finance plugin's fin:; here the test
    vocabulary's tv:, installed the same way) is part of the ontology the importer runs
    with, so an entity stored under a plugin class — a bank — is recognised instead of
    looking like a class nobody declared."""
    from memris.graph import MemoryGraph

    seen: list[object] = []
    real_init = MemoryGraph.__init__

    def spy(self: MemoryGraph, ontology: object, *args: object, **kw: object) -> None:
        seen.append(ontology)
        real_init(self, ontology, *args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(MemoryGraph, "__init__", spy)
    root = typer.Typer()
    graphiti_cli.register(PluginCLI(root=root))
    path = tmp_path / "export.json"
    path.write_text(json.dumps(export), encoding="utf-8")

    CliRunner().invoke(root, ["graphiti", "import", str(path), "--db", str(tmp_path / "m.db")])

    assert seen and "tv:banks_with" in seen[0].relations  # type: ignore[attr-defined]
