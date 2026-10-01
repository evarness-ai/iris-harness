"""``python -m memris.interchange`` — move a memris SQLite store in and out as JSON-LD.

    python -m memris.interchange export <ontology_dir> <db> [--out file.jsonld]
    python -m memris.interchange import <ontology_dir> <db> <file.jsonld>

Import prints what it could not map and exits 1 if anything was skipped or unread.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from memris.graph import MemoryGraph
from memris.interchange.jsonld import export_document, import_document
from memris.ontology import load_or_raise
from memris.store import SQLiteGraphStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m memris.interchange")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="write the store as JSON-LD")
    exp.add_argument("ontology_dir")
    exp.add_argument("db")
    exp.add_argument("--out", help="file to write (default: stdout)")
    imp = sub.add_parser("import", help="read a JSON-LD document into the store")
    imp.add_argument("ontology_dir")
    imp.add_argument("db")
    imp.add_argument("file")
    args = parser.parse_args(argv)

    graph = MemoryGraph(load_or_raise(args.ontology_dir), SQLiteGraphStore(Path(args.db)))
    if args.command == "export":
        text = json.dumps(export_document(graph), indent=2, ensure_ascii=False)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    document = json.loads(Path(args.file).read_text(encoding="utf-8"))
    report = import_document(document, graph)
    print(f"imported {report.entities} entities, {report.statements} statements")
    for node_id, why in report.skipped:
        print(f"skipped {node_id}: {why}")
    for node_id, key in report.unmapped_fields:
        print(f"not read {node_id}: {key}")
    return 0 if report.lossless else 1


if __name__ == "__main__":
    sys.exit(main())
