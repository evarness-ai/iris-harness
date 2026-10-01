"""One-shot wiki entity compaction per ADR-0010.

Re-applies the tightened entity validators to existing
``data/wiki/entities/`` pages and moves rejected pages to
``data/wiki/_quarantine/entities/`` for later cleanup.

Only applies to ``entity_type ∈ {topic, person}``. Institutions and
concepts are left alone (their extraction paths are already precise).

Usage:
    poetry run python scripts/compact_wiki_entities.py            # dry-run
    poetry run python scripts/compact_wiki_entities.py --apply    # move files
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iris_harness.memory.knowledge.entity_extractor import (  # noqa: E402
    _is_likely_person_bigram,
    _is_valid_hint,
)
from iris_harness.memory.knowledge.page_manager import PageManager  # noqa: E402

WIKI_ROOT = REPO_ROOT / "data" / "wiki"
ENTITIES_DIR = WIKI_ROOT / "entities"
QUARANTINE_DIR = WIKI_ROOT / "_quarantine" / "entities"


@dataclass(frozen=True)
class Decision:
    slug: str
    entity_type: str
    title: str
    keep: bool
    reason: str


def classify(page) -> Decision:  # type: ignore[no-untyped-def]
    """Decide keep-or-quarantine for a single page using the new validators."""
    slug = page.slug
    entity_type = str(page.frontmatter.get("entity_type") or "").strip().lower()
    title = str(page.title or "").strip()

    # Out-of-scope types stay as-is (institution / concept paths are already precise).
    if entity_type not in ("topic", "person"):
        return Decision(slug, entity_type, title, True, "type out of scope")

    # Malformed/empty title → cannot safely decide; keep.
    if not title:
        return Decision(slug, entity_type, title, True, "no title")

    if entity_type == "topic":
        if _is_valid_hint(title):
            return Decision(slug, entity_type, title, True, "valid hint")
        return Decision(slug, entity_type, title, False, "invalid hint")

    # entity_type == "person"
    parts = title.split()
    if len(parts) != 2:
        # Not a clean _PERSON_RE bigram shape — likely LLM-extracted; keep.
        return Decision(slug, entity_type, title, True, "person, !=2 tokens (LLM-shaped)")
    if _is_likely_person_bigram(parts[0], parts[1]):
        return Decision(slug, entity_type, title, True, "valid person bigram")
    return Decision(slug, entity_type, title, False, "bigram-stopword person")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact wiki entity pages per ADR-0010.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually move files to quarantine (default: dry-run)",
    )
    args = parser.parse_args()

    if not ENTITIES_DIR.is_dir():
        print(f"no entities dir at {ENTITIES_DIR}", file=sys.stderr)
        return 1

    manager = PageManager(wiki_root=WIKI_ROOT)
    pages = [p for p in manager.load_all() if str(p.page_type) == "entity"]
    print(f"scanned {len(pages)} entity page(s) in {ENTITIES_DIR}")

    decisions = [classify(p) for p in pages]
    quarantine_list = [d for d in decisions if not d.keep]
    keep_list = [d for d in decisions if d.keep]

    print(f"\n=== KEEP ({len(keep_list)}) ===")
    for et, count in Counter(d.entity_type for d in keep_list).most_common():
        print(f"  {et or '<empty>'}: {count}")

    print(f"\n=== QUARANTINE ({len(quarantine_list)}) ===")
    for reason, count in Counter(d.reason for d in quarantine_list).most_common():
        print(f"  {reason}: {count}")

    print("\n=== SAMPLE QUARANTINE SLUGS (first 15) ===")
    for d in quarantine_list[:15]:
        print(f"  - {d.slug} (type={d.entity_type}, reason={d.reason})")
    if len(quarantine_list) > 15:
        print(f"  ... and {len(quarantine_list) - 15} more")

    if not args.apply:
        print("\n[dry-run] no files moved. Re-run with --apply to execute.")
        return 0

    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    for d in quarantine_list:
        src = manager.page_path(d.slug, "entity")  # type: ignore[arg-type]
        dst = QUARANTINE_DIR / f"{d.slug}.md"
        if src.exists():
            shutil.move(str(src), str(dst))
            moved += 1
    print(f"\nmoved {moved} page(s) to {QUARANTINE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
