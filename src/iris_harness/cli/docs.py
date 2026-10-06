"""``iris docs`` CLI — index your own documents for cited retrieval (RAG R0).

Point IRIS at a file or folder (e.g. an Obsidian vault); it indexes the
content for search and answers, leaving your files as the source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from iris_harness.cli.render import console, print_error
from iris_harness.services.rag.models import IngestResult

docs_app = typer.Typer(
    name="docs",
    help="Document RAG — index your markdown / PDFs / images for cited retrieval "
    "(connector, not a copy; images + scanned PDFs need OCR installed).",
    no_args_is_help=True,
)


def _store_and_index() -> tuple[object, object]:
    from iris_harness.services.rag.index import DocumentIndex
    from iris_harness.services.rag.store import DocumentStore

    store = DocumentStore()
    store.ensure_schema()
    return store, DocumentIndex()


def _report(result: IngestResult) -> None:
    """Print the ingest summary; a refused (secret) file is a warning and a non-zero exit."""
    if result.sources_denied:
        console.print(f"  [bold yellow]![/bold yellow]  {result.summary()}")
        raise typer.Exit(1)
    console.print(f"  [bold green]✓[/bold green]  {result.summary()}")


@docs_app.command("add")
def cmd_add(
    path: Annotated[str, typer.Argument(help="File or folder to index (e.g. an Obsidian vault).")],
    kind: Annotated[str, typer.Option("--kind", help="Source kind tag.")] = "folder",
) -> None:
    """Index a file or folder of markdown/text/PDF/image documents.

    PDFs use embedded text (per-page citations); image-only pages and image
    files are OCR'd when OCR is installed, otherwise skipped.
    """
    from iris_harness.services.rag.ingest import ingest_path
    from iris_harness.services.rag.ingest_source import current_ingest_source

    target = Path(path).expanduser()
    if not target.exists():
        print_error(f"no such path: {target}")
        raise typer.Exit(2)
    store, index = _store_and_index()
    result = ingest_path(
        target,
        store=store,  # type: ignore[arg-type]
        index=index,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        source=current_ingest_source(),
    )
    _report(result)


@docs_app.command("sync")
def cmd_sync() -> None:
    """Re-index every registered source (picks up external edits)."""
    from iris_harness.services.rag.ingest import sync_all
    from iris_harness.services.rag.ingest_source import current_ingest_source

    store, index = _store_and_index()
    result = sync_all(
        store=store,  # type: ignore[arg-type]
        index=index,  # type: ignore[arg-type]
        source=current_ingest_source(),
    )
    _report(result)


@docs_app.command("reindex")
def cmd_reindex(
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Rebuild even if the chunk store is empty (empties the index)."
        ),
    ] = False,
) -> None:
    """Rebuild the vector index from the stored chunks (after a lost or damaged index).

    Reads no source file: the chunk store is canonical, the vector index only mirrors it.
    `sync` cannot do this, since it skips files that have not changed.
    Refuses when the store is empty but the index is not, unless --force.
    """
    from iris_harness.services.rag.ingest import reindex_all

    store, index = _store_and_index()
    try:
        count = reindex_all(store=store, index=index, force=force)  # type: ignore[arg-type]
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from exc
    console.print(f"  [bold green]✓[/bold green]  rebuilt the vector index: {count} chunk(s)")


@docs_app.command("list")
def cmd_list() -> None:
    """List indexed document sources."""
    store, _ = _store_and_index()
    sources = store.list_sources()  # type: ignore[attr-defined]
    if not sources:
        console.print("  [yellow]no sources — add one with `iris docs add <path>`[/yellow]")
        return
    for s in sources:
        n = store.count_chunks(s.id)  # type: ignore[attr-defined]
        tags = f"  [magenta]#{' #'.join(s.tags)}[/magenta]" if s.tags else ""
        console.print(
            f"  [cyan]{s.title}[/cyan]  [dim]{s.path}  ({n} chunk(s), {s.kind})[/dim]{tags}"
        )


@docs_app.command("search")
def cmd_search(
    query: Annotated[str, typer.Argument(help="What to search for.")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max results.")] = 5,
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Scope to sources with this tag (repeatable).")
    ] = None,
) -> None:
    """Search indexed documents; prints matching chunks with citations."""
    from iris_harness.services.rag.retrieve import search_documents

    store, index = _store_and_index()
    hits = search_documents(
        query, store=store, index=index, limit=limit, tags=tuple(tag or ())  # type: ignore[arg-type]
    )
    if not hits:
        console.print("  [yellow]no matches[/yellow]")
        return
    for h in hits:
        snippet = h.text.replace("\n", " ")[:160]
        console.print(f"  [cyan]{h.citation.label()}[/cyan]  [dim]({h.score:.2f})[/dim]")
        console.print(f"    {snippet}")


@docs_app.command("ask")
def cmd_ask(
    question: Annotated[str, typer.Argument(help="Question to answer from your documents.")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Passages to ground on.")] = 5,
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Scope to sources with this tag (repeatable).")
    ] = None,
) -> None:
    """Answer a question grounded in your documents, with citations (NotebookLM-style)."""
    from iris_harness.services.rag.qa import answer_question, default_llm_call

    store, index = _store_and_index()
    result = answer_question(
        question,
        store=store,  # type: ignore[arg-type]
        index=index,  # type: ignore[arg-type]
        llm_call=default_llm_call(),
        limit=limit,
        tags=tuple(tag or ()),
    )
    tag_note = "" if result.grounded else "  [dim](passages — no synthesis model)[/dim]"
    console.print(f"  {result.answer}{tag_note}")
    if result.citations:
        console.print("  [dim]Sources:[/dim]")
        for i, c in enumerate(result.citations, start=1):
            console.print(f"    [cyan][{i}][/cyan] {c.label()}")


@docs_app.command("related")
def cmd_related(
    path: Annotated[str, typer.Argument(help="Indexed note path to show links/tags for.")],
) -> None:
    """Show a note's tags and outgoing [[wikilinks]] (the vault graph)."""
    from iris_harness.services.rag.ingest import _source_id

    store, _ = _store_and_index()
    sid = _source_id(Path(path).expanduser().resolve())
    source = store.get_source(sid)  # type: ignore[attr-defined]
    if source is None:
        console.print(f"  [yellow]not indexed: {path}[/yellow]")
        return
    console.print(f"  [cyan]{source.title}[/cyan]")
    console.print(f"  tags:  {', '.join(source.tags) if source.tags else '—'}")
    console.print(f"  links: {', '.join(source.links) if source.links else '—'}")


@docs_app.command("remove")
def cmd_remove(
    path: Annotated[str, typer.Argument(help="Source path to remove from the index.")],
) -> None:
    """Remove a source's chunks from the index (does not touch your file)."""
    from iris_harness.services.rag.ingest import _source_id
    from iris_harness.services.rag.ingest_source import current_ingest_source, report_removed

    store, index = _store_and_index()
    resolved = Path(path).expanduser().resolve()
    sid = _source_id(resolved)
    if store.get_source(sid) is None:  # type: ignore[attr-defined]
        console.print(f"  [yellow]not indexed: {path}[/yellow]")
        return
    store.delete_source(sid)  # type: ignore[attr-defined]
    index.delete_source(sid)  # type: ignore[attr-defined]
    report_removed(current_ingest_source(), resolved, source_id=sid, reason="removed")
    console.print(f"  [bold green]✓[/bold green]  removed [cyan]{path}[/cyan] from the index")


__all__ = ["docs_app"]
