"""Maintain wiki index.md catalog and append-only log.md."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .models import WikiPage


class WikiIndex:
    """Read and write the wiki index catalog and activity log."""

    INDEX_FILE = "index.md"
    LOG_FILE = "log.md"

    def __init__(self, wiki_root: Path) -> None:
        self.wiki_root = wiki_root

    @property
    def index_path(self) -> Path:
        return self.wiki_root / self.INDEX_FILE

    @property
    def log_path(self) -> Path:
        return self.wiki_root / self.LOG_FILE

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def rebuild(self, pages: list[WikiPage]) -> None:
        """Rebuild index.md from the current set of wiki pages."""
        lines = [
            "# IRIS Knowledge Wiki — Index\n",
            f"> Last rebuilt: {datetime.now(UTC).isoformat()}\n\n",
        ]
        by_type: dict[str, list[WikiPage]] = {}
        for page in sorted(pages, key=lambda p: p.slug):
            by_type.setdefault(page.page_type, []).append(page)

        for page_type in ("entity", "concept", "synthesis", "source"):
            group = by_type.get(page_type, [])
            if not group:
                continue
            lines.append(f"## {page_type.title()}s\n\n")
            for page in group:
                excerpt = page.body[:80].replace("\n", " ").strip()
                lines.append(f"- [[{page.slug}]] — {page.title}: {excerpt}\n")
            lines.append("\n")

        self.index_path.write_text("".join(lines), encoding="utf-8")

    def find_slug(self, query: str) -> str | None:
        """Return the slug for the best index match, or None."""
        if not self.index_path.exists():
            return None
        content = self.index_path.read_text(encoding="utf-8")
        query_lower = query.lower()
        best: tuple[int, str] | None = None
        for line in content.splitlines():
            if "[[" not in line:
                continue
            import re

            m = re.search(r"\[\[([^\]]+)\]\]", line)
            if not m:
                continue
            slug = m.group(1)
            score = sum(1 for token in query_lower.split() if token in line.lower())
            if score > 0 and (best is None or score > best[0]):
                best = (score, slug)
        return best[1] if best else None

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def append_log(self, event: str, detail: str = "") -> None:
        """Append one line to log.md."""
        ts = datetime.now(UTC).isoformat()
        entry = f"[{ts}] {event}"
        if detail:
            entry += f" — {detail}"
        entry += "\n"
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(entry)

    def recent_log_lines(self, n: int = 20) -> list[str]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return lines[-n:]
