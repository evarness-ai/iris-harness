"""File-backed wiki page CRUD with YAML frontmatter and wikilink support."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .models import PageType, WikiPage

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_SLUG_RE = re.compile(r"[^\w\-]")

_TYPE_DIRS: dict[PageType, str] = {
    "entity": "entities",
    "concept": "concepts",
    "source": "sources",
    "synthesis": "synthesis",
}


def slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower().strip()).strip("-")


class PageManager:
    """Create, read, update, and delete wiki pages as markdown files."""

    def __init__(self, wiki_root: Path) -> None:
        self.wiki_root = wiki_root
        self.wiki_root.mkdir(parents=True, exist_ok=True)
        for subdir in _TYPE_DIRS.values():
            (self.wiki_root / subdir).mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def page_path(self, slug: str, page_type: PageType) -> Path:
        return self.wiki_root / _TYPE_DIRS[page_type] / f"{slug}.md"

    def all_page_paths(self) -> list[Path]:
        return [p for p in self.wiki_root.rglob("*.md") if p.name not in ("index.md", "log.md")]

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, page: WikiPage, *, touch: bool = True) -> Path:
        if touch:
            page.last_updated = datetime.now(UTC)
        fm = dict(page.frontmatter)
        fm.setdefault("title", page.title)
        fm.setdefault("page_type", page.page_type)
        fm["last_updated"] = page.last_updated.isoformat()
        fm.setdefault("created", page.created.isoformat())
        text = f"---\n{yaml.dump(fm, sort_keys=True, allow_unicode=True)}---\n\n# {page.title}\n\n{page.body}"
        path = self.page_path(page.slug, page.page_type)
        path.write_text(text, encoding="utf-8")
        return path

    def load(self, slug: str, page_type: PageType) -> WikiPage | None:
        path = self.page_path(slug, page_type)
        if not path.exists():
            return None
        return self._parse(slug, page_type, path.read_text(encoding="utf-8"))

    def delete(self, slug: str, page_type: PageType) -> bool:
        path = self.page_path(slug, page_type)
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, slug: str, page_type: PageType) -> bool:
        return self.page_path(slug, page_type).exists()

    def load_all(self) -> list[WikiPage]:
        pages: list[WikiPage] = []
        for path in self.all_page_paths():
            page_type = _dir_to_type(path.parent.name)
            if page_type is None:
                continue
            slug = path.stem
            page = self._parse(slug, page_type, path.read_text(encoding="utf-8"))
            if page is not None:
                pages.append(page)
        return pages

    def append_section(self, slug: str, page_type: PageType, section_body: str) -> bool:
        """Append new content to an existing page's body."""
        page = self.load(slug, page_type)
        if page is None:
            return False
        page.body = page.body.rstrip() + "\n\n" + section_body.strip()
        page.frontmatter["source_count"] = page.source_count + 1
        self.save(page)
        return True

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, slug: str, page_type: PageType, text: str) -> WikiPage | None:
        fm: dict[str, object] = {}
        body = text
        m = _FRONTMATTER_RE.match(text)
        if m:
            try:
                fm = yaml.safe_load(m.group(1)) or {}
            except yaml.YAMLError:
                fm = {}
            body = text[m.end() :]

        # Strip the H1 title line so body contains only the content
        body_lines = body.split("\n")
        title = slug.replace("-", " ").title()
        if body_lines and body_lines[0].startswith("# "):
            title = body_lines[0][2:].strip()
            body = "\n".join(body_lines[1:]).strip()

        def _dt(key: str) -> datetime:
            val = fm.get(key)
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except ValueError:
                    pass
            return datetime.now(UTC)

        return WikiPage(
            slug=slug,
            page_type=page_type,
            title=str(fm.get("title", title)),
            body=body,
            frontmatter=fm,
            created=_dt("created"),
            last_updated=_dt("last_updated"),
        )


def _dir_to_type(dirname: str) -> PageType | None:
    reverse = {v: k for k, v in _TYPE_DIRS.items()}
    return reverse.get(dirname)
