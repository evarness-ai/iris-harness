"""Phase 1 re-spike — interactive labeling tool for user-defined categories.

Throwaway diagnostic. Delete after Phase 1 lands.

Reads ``data/spike/phase1_emails.jsonl`` (50 emails from the IMAP fetch),
prompts the user to assign one of five categories to each, and appends
to ``data/spike/phase1_emails_labeled.jsonl`` after every keystroke
(resumable — re-running skips already-labeled UIDs).

Categories (derived from the actual inbox sample):
  1) finance      — banks, insurance, investment, statements, txn notifications
  2) marketing    — shopping promos, sales, "last chance" offers, ad emails
  3) learning     — school (community-school), courses, workshops, GitHub
  4) news-digest  — Economic Times, Medium digests, general newsletters
  5) social       — LinkedIn / Facebook / community posts and notifications
  6) personal     — real conversations with humans, family, work threads

Controls:
  1-6  pick a category
  s    skip (exclude from accuracy calc)
  q    save progress and quit
  ?    re-show the current email + key legend

Usage:
    poetry run python scripts/phase1_spike_label.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "spike" / "phase1_emails.jsonl"
LABELED_PATH = REPO_ROOT / "data" / "spike" / "phase1_emails_labeled.jsonl"

CATEGORIES: dict[str, str] = {
    "1": "finance",
    "2": "marketing",
    "3": "learning",
    "4": "news-digest",
    "5": "social",
    "6": "personal",
}

LEGEND = (
    "  1) finance      2) marketing    3) learning\n"
    "  4) news-digest  5) social       6) personal\n"
    "  s) skip         q) save and quit         ?) show legend"
)


def _load_existing_labels() -> dict[str, str]:
    """Return {uid: user_category_or_<skipped>} from any prior labeled file."""
    if not LABELED_PATH.is_file():
        return {}
    out: dict[str, str] = {}
    with LABELED_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row.get("uid")
            if uid:
                out[uid] = row.get("user_category", "<skipped>")
    return out


def _append_label(row: dict, user_category: str | None) -> None:
    """Append a labeled row to the labels file (atomic, line-buffered)."""
    out = dict(row)
    out["user_category"] = user_category  # None means explicitly skipped
    LABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELED_PATH.open("a") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


def _render(row: dict, idx: int, total: int) -> None:
    """Print the current email + category legend."""
    print("\n" + "=" * 76)
    print(f"[{idx}/{total}]  uid={row.get('uid')}  gmail_category={row.get('gmail_category')!r}")
    print(f"  from:    {row.get('from_address') or row.get('from_domain') or '<unknown>'}")
    print(f"  subject: {row.get('subject') or '<empty>'}")
    snippet = (row.get("snippet") or "").strip()
    if snippet:
        print(f"  snippet: {snippet[:300]}")
    print()
    print(LEGEND)


def _prompt_choice() -> str:
    """Read a single line from stdin. Returns the raw choice string."""
    try:
        choice = input("  Your choice: ").strip().lower()
    except EOFError:
        return "q"
    return choice


def main() -> int:
    if not INPUT_PATH.is_file():
        print(f"error: missing input {INPUT_PATH}", file=sys.stderr)
        print("       run scripts/phase1_spike_imap_fetch.py first", file=sys.stderr)
        return 1

    with INPUT_PATH.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        print("error: input file empty", file=sys.stderr)
        return 2

    existing = _load_existing_labels()
    if existing:
        print(f"resuming — {len(existing)} of {len(rows)} already labeled")

    unlabeled = [r for r in rows if r.get("uid") not in existing]
    total = len(rows)
    labeled_count = len(existing)
    skipped_count = sum(1 for v in existing.values() if v == "<skipped>")

    if not unlabeled:
        print(f"all {total} emails already labeled. Nothing to do.")
        print(f"  labels file: {LABELED_PATH}")
        return 0

    print(f"\n{len(unlabeled)} emails remaining to label.")
    print("Single keystrokes only — choose 1-6, s to skip, q to save+quit, ? for legend.\n")

    for idx_offset, row in enumerate(unlabeled, start=labeled_count + 1):
        _render(row, idx_offset, total)
        while True:
            choice = _prompt_choice()
            if choice == "?":
                _render(row, idx_offset, total)
                continue
            if choice == "q":
                print(
                    f"\nsaved {labeled_count + idx_offset - labeled_count - 1} labels this session."
                )
                print(f"  labels file: {LABELED_PATH}")
                print("  re-run this script to resume.")
                return 0
            if choice == "s":
                _append_label(row, None)
                skipped_count += 1
                print("  → skipped")
                break
            if choice in CATEGORIES:
                category = CATEGORIES[choice]
                _append_label(row, category)
                print(f"  → {category}")
                break
            print(f"  unknown key {choice!r}; valid: 1-6, s, q, ?")

    print(f"\nDONE — labeled all {total} emails.")
    print(f"  labels file: {LABELED_PATH}")
    print(f"  skipped: {skipped_count}/{total}")
    print("\nNext: re-run the classifier against user labels:")
    print("  poetry run python scripts/phase1_spike_classify.py --ground-truth-field user_category")
    return 0


if __name__ == "__main__":
    sys.exit(main())
