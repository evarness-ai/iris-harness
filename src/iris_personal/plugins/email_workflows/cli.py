"""``iris email`` — the group this plugin publishes (OSS plan M3.3, M6.1b).

The *workflow* half: discovering categories, accepting them, classifying mail,
labelling a holdout, measuring the kNN gate, recording corrections, re-ingesting the
wiki and detecting followups. The *read* half — ``search`` and ``semantic-index`` —
was main.py's until M6.1b, when the email domain left the core tree (OSS plan M6,
decision 2): a core-only install has no ``iris email`` at all.

These arrived through the CLI seam (``manifest.yaml``'s ``cli:``), which runs at
``iris`` start-up with no runtime built and no ``setup`` called, so adding them costs
nothing on the ``--help`` path. That also means no ``HarnessServices``: a body needing
the runtime builds it itself, exactly as it did when it lived in ``main.py``.

Command bodies are unchanged from ``main.py``; only the app they attach to is passed
in rather than being a module global, so each ``@email_app.command`` decorator became
one line of :func:`register`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from iris_harness.sdk.cli import console, print_error
from iris_harness.sdk.config import config_dir, config_path
from iris_harness.sdk.config import workspace_dir as iris_workspace_dir
from iris_harness.sdk.persistence import data_path
from iris_personal.plugins.email_workflows.cli_setup import cmd_email_setup

if TYPE_CHECKING:
    from iris_harness.sdk import PluginCLI


def register(cli: PluginCLI) -> None:
    """Attach this plugin's commands to the harness's ``iris email`` group."""
    email_app = cli.group(
        "email",
        help=(
            "Email domain commands (Phase 1+). Operates on the locally-stored "
            "email.db; per-account opt-in."
        ),
    )
    email_app.command("search")(cmd_email_search)
    email_app.command("semantic-index")(cmd_email_semantic_index)
    email_app.command("repair-domains")(cmd_email_repair_domains)
    email_app.command("bootstrap-categories")(cmd_email_bootstrap_categories)
    email_app.command("accept-categories")(cmd_email_accept_categories)
    email_app.command("triage")(cmd_email_triage)
    email_app.command("triage-batch")(cmd_email_triage_batch)
    email_app.command("label-holdout")(cmd_email_label_holdout)
    email_app.command("recategorize")(cmd_email_recategorize)
    email_app.command("reingest-wiki")(cmd_email_reingest_wiki)
    email_app.command("knn-gate")(cmd_email_knn_gate)
    email_app.command("detect-followups")(cmd_email_detect_followups)
    email_app.command("judge")(cmd_email_judge)
    email_app.command("judgments")(cmd_email_judgments)
    email_app.command("demo")(cmd_email_demo)
    email_app.command("setup")(cmd_email_setup)
    email_app.add_typer(corrections_app, name="corrections")
    from .cli_writes import writes_app  # the mailbox-write gate (R4), any provider

    email_app.add_typer(writes_app, name="writes")


def _account_slug_for_path(account_id: str) -> str:
    """Translate ``gmail:user@gmail.com`` → ``gmail-user-at-gmail.com``."""
    return account_id.replace(":", "-").replace("@", "-at-")


def cmd_email_bootstrap_categories(
    account: Annotated[
        str,
        typer.Option(
            "--account",
            help="email_accounts.id slug, e.g. gmail:user@gmail.com",
        ),
    ],
    min_corpus: Annotated[
        int,
        typer.Option(
            "--min-corpus",
            help="Minimum corpus size before clustering. If the local "
            "email.db has fewer rows for the account, run a bulk fetch.",
        ),
    ] = 500,
    no_fetch_if_low: Annotated[
        bool,
        typer.Option(
            "--no-fetch-if-low",
            help="Skip the auto bulk fetch even if the corpus is below --min-corpus.",
        ),
    ] = False,
    llama_base: Annotated[
        str,
        typer.Option(
            "--llama-base",
            help="Local llama-server base URL. Marked transitional per ADR-0018 §5 — "
            "retires when Track 1H lands Tier-3-local in tier_router.",
        ),
    ] = "http://localhost:8090/v1",
    skip_naming: Annotated[
        bool,
        typer.Option(
            "--skip-naming",
            help="Skip the LLM naming pass; just write cluster proposals without names.",
        ),
    ] = False,
    workspace_dir: Annotated[
        Path | None,
        typer.Option(
            "--workspace-dir",
            help="Override workspace root (default: $IRIS_HOME/workspace).",
        ),
    ] = None,
) -> None:
    """Discover category candidates from this account's mail.

    Per ADR-0017 (dynamic categories) + ADR-0018 (CLI shape). Opt-in
    per account. Writes proposals to
    ``$IRIS_HOME/workspace/email/<slug>/proposals.jsonl`` for human review;
    persistence to ``data/iris.db.categories`` is Track 1F.
    """
    from iris_personal.plugins.email_workflows.discovery import (
        LlamaServerClient,
        bootstrap_categories,
    )

    ws_root = workspace_dir or iris_workspace_dir()
    out_dir = ws_root / "email" / _account_slug_for_path(account)
    out_path = out_dir / "proposals.jsonl"

    client = None if skip_naming else LlamaServerClient(base_url=llama_base)

    try:
        proposals = bootstrap_categories(
            account,
            min_corpus=min_corpus,
            fetch_if_low=not no_fetch_if_low,
            naming_client=client,
        )
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc
    except RuntimeError as exc:
        print_error(f"bootstrap failed: {exc}")
        raise typer.Exit(3) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in proposals:
            f.write(p.model_dump_json() + "\n")

    # Summary table
    from rich.table import Table

    table = Table(title=f"{len(proposals)} category candidates", show_lines=False)
    table.add_column("size", justify="right", style="bold")
    table.add_column("cohesion", justify="right")
    table.add_column("name")
    table.add_column("top domain", style="dim")
    for p in proposals:
        name = (
            f"{p.proposed_root or '-'}.{p.proposed_branch or '-'}.{p.proposed_leaf or '-'}"
            if p.proposed_root or p.proposed_branch or p.proposed_leaf
            else "[dim](unnamed)[/dim]"
        )
        top = p.top_domains[0][0] if p.top_domains else "-"
        table.add_row(str(p.size), f"{p.cohesion:.2f}", name, top)
    console.print(table)
    console.print(f"  [dim]wrote[/dim] {out_path}")


def cmd_email_accept_categories(
    account: Annotated[
        str,
        typer.Option(
            "--account",
            help="email_accounts.id slug, e.g. gmail:user@gmail.com",
        ),
    ],
    proposals_path: Annotated[
        Path | None,
        typer.Option(
            "--proposals",
            help="Override the proposals JSONL path (default: "
            "<workspace>/email/<account_slug>/proposals.jsonl).",
        ),
    ] = None,
    workspace_dir: Annotated[
        Path | None,
        typer.Option(
            "--workspace-dir",
            help="Override workspace root (default: $IRIS_HOME/workspace).",
        ),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option(
            "--db-path",
            help="Override iris.db location (default: iris.db in the IRIS data dir).",
        ),
    ] = None,
) -> None:
    """Persist accepted category proposals into ``data/iris.db.categories``.

    Reads the JSONL produced by ``iris email bootstrap-categories``,
    applies strict validation per ADR-0019 §5 (single bad row aborts
    with exit 4), and upserts via ``CategoryStore.upsert_if_new`` —
    idempotent on re-runs.
    """
    from iris_personal.email.category_store import Category, CategoryStore
    from iris_personal.plugins.email_workflows.discovery import category_fields

    ws_root = workspace_dir or iris_workspace_dir()
    path = proposals_path or (
        ws_root / "email" / _account_slug_for_path(account) / "proposals.jsonl"
    )

    if not path.exists():
        print_error(f"proposals file not found: {path}")
        raise typer.Exit(2)

    raw_rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            raw_rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print_error(f"line {i + 1}: invalid JSON ({exc})")
            raise typer.Exit(4) from exc

    if not raw_rows:
        print_error(f"proposals file is empty: {path}")
        raise typer.Exit(3)

    # Strict validation phase — collect ALL errors before deciding to abort,
    # so the user sees every fix needed in one pass.
    parsed: list[dict[str, Any]] = []
    errors: list[str] = []
    for raw in raw_rows:
        err, kwargs = category_fields(raw, account)
        if err is not None:
            errors.append(err)
        else:
            parsed.append(kwargs)

    if errors:
        print_error(f"{len(errors)} invalid proposal(s) — fix the JSONL and re-run:")
        for e in errors:
            console.print(f"  [red]·[/red] {e}")
        raise typer.Exit(4)

    # Collision detection — per ADR-0020 amendment (Path D).
    # Distinct clusters that resolved to the same (root, branch, leaf) would
    # silently overwrite each other on the path PK; surface them so the user
    # can disambiguate one of the leaves before any DB write.
    from collections import defaultdict

    path_to_clusters: dict[str, list[Any]] = defaultdict(list)
    for kwargs in parsed:
        cid = (kwargs.get("metadata") or {}).get("cluster_id")
        path_to_clusters[kwargs["path"]].append(cid)
    collisions = {p: cids for p, cids in path_to_clusters.items() if len(cids) > 1}

    if collisions:
        print_error(
            f"{len(collisions)} path collision(s) — distinct clusters got the "
            "same name. Edit the JSONL to disambiguate one of each pair "
            "(e.g. add a topic suffix to the leaf):"
        )
        for collision_path, cids in collisions.items():
            console.print(
                f"  [red]·[/red] [cyan]{collision_path}[/cyan] from clusters "
                + ", ".join(str(c) for c in cids)
            )
        raise typer.Exit(5)

    # Commit phase
    store = CategoryStore(db_path=db_path) if db_path else CategoryStore()
    store.ensure_schema()

    inserted = 0
    skipped = 0
    for kwargs in parsed:
        category = Category(**kwargs)
        if store.upsert_if_new(category):
            inserted += 1
        else:
            skipped += 1

    # Summary
    from rich.table import Table

    table = Table(title=f"{inserted} new / {skipped} unchanged", show_lines=False)
    table.add_column("path")
    table.add_column("cohesion", justify="right")
    table.add_column("size", justify="right", style="dim")
    for kw in parsed:
        coh = kw.get("cohesion")
        coh_str = f"{coh:.2f}" if coh is not None else "-"
        size = (kw.get("metadata") or {}).get("size_at_acceptance", "-")
        table.add_row(kw["path"], coh_str, str(size))
    console.print(table)
    console.print(
        f"  [dim]wrote to[/dim] [cyan]{store.db_path}[/cyan]  "
        f"[dim]({inserted} new, {skipped} already accepted)[/dim]"
    )


def cmd_email_triage(
    account: Annotated[
        str,
        typer.Option(
            "--account",
            help="email_accounts.id slug, e.g. gmail:user@gmail.com",
        ),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Cap on unclassified emails processed in this run.",
        ),
    ] = 20,
    use_llm: Annotated[
        bool,
        typer.Option(
            "--use-llm",
            help="Use the hybrid LLM picker on every email (heavy — see ADR-0022). "
            "Off by default → pure-kNN with confidence-gated queueing.",
        ),
    ] = False,
    workspace_dir: Annotated[
        Path | None,
        typer.Option(
            "--workspace-dir",
            help="Override workspace root (default: $IRIS_HOME/workspace).",
        ),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option(
            "--db-path",
            help="Override iris.db location (default: iris.db in the IRIS data dir).",
        ),
    ] = None,
    email_db_path: Annotated[
        Path | None,
        typer.Option(
            "--email-db",
            help="Override email.db location (default: email.db in the IRIS data dir).",
        ),
    ] = None,
    llama_base: Annotated[
        str,
        typer.Option(
            "--llama-base",
            help="Local llama-server base URL (used only when --use-llm).",
        ),
    ] = "http://localhost:8090/v1",
) -> None:
    """Classify unclassified emails on demand.

    Default per ADR-0022: pure kNN with the confidence gate. Ambiguous
    rows are queued (``triage_state='pending_review'``) for batch LLM
    review via ``iris email triage-batch``. Tier-3-local is NOT
    invoked unless ``--use-llm`` is passed.
    """
    from iris_personal.plugins.email_workflows.discovery import LlamaServerClient
    from iris_personal.plugins.email_workflows.triage import EmailTriageClassifier

    ws_root = workspace_dir or iris_workspace_dir()

    classifier = EmailTriageClassifier(
        workspace_dir=ws_root,
        db_path=db_path or data_path("iris.db"),
        email_db_path=email_db_path or data_path("email.db"),
        naming_client=LlamaServerClient(base_url=llama_base) if use_llm else None,
    )

    try:
        results = classifier.classify_unclassified(account, limit=limit, use_llm=use_llm)
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc

    if not results:
        console.print(
            f"  [yellow]no unclassified emails for {account}[/yellow]"
            "  [dim](inbox already triaged, or no mail fetched yet)[/dim]"
        )
        raise typer.Exit(3)

    classified = [r for r in results if r.category_path is not None]
    queued = [r for r in results if r.queued]
    failed = [r for r in results if r.category_path is None and not r.queued]

    from rich.table import Table

    title_parts = [f"{len(classified)} classified"]
    if queued:
        title_parts.append(f"{len(queued)} queued")
    if failed:
        title_parts.append(f"{len(failed)} soft-failed")
    table = Table(title=", ".join(title_parts), show_lines=False)
    table.add_column("message id", style="cyan")
    table.add_column("state")
    table.add_column("category", style="bold")
    table.add_column("confidence", justify="right")
    table.add_column("note", style="dim")
    for r in results:
        if r.queued:
            state = "[yellow]queued[/yellow]"
            path = "—"
        elif r.category_path is not None:
            state = "[green]ok[/green]"
            path = r.category_path
        else:
            state = "[red]error[/red]"
            path = "—"
        conf = f"{r.confidence:.2f}" if r.confidence is not None else "—"
        note = r.error or ""
        table.add_row(r.message_id, state, path, conf, note[:80])
    console.print(table)
    console.print(f"  [dim]wrote classifications to[/dim] [cyan]{classifier.email_db_path}[/cyan]")
    if queued:
        console.print(
            f"  [dim]{len(queued)} queued for batch — run[/dim] "
            f"[cyan]iris email triage-batch --account {account}[/cyan] "
            "[dim]to process with the LLM[/dim]"
        )


def cmd_email_triage_batch(
    account: Annotated[
        str,
        typer.Option(
            "--account",
            help="email_accounts.id slug, e.g. gmail:user@gmail.com",
        ),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Cap on pending_review emails processed in this run.",
        ),
    ] = 20,
    workspace_dir: Annotated[
        Path | None,
        typer.Option(
            "--workspace-dir",
            help="Override workspace root (default: $IRIS_HOME/workspace).",
        ),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option(
            "--db-path",
            help="Override iris.db location (default: iris.db in the IRIS data dir).",
        ),
    ] = None,
    email_db_path: Annotated[
        Path | None,
        typer.Option(
            "--email-db",
            help="Override email.db location (default: email.db in the IRIS data dir).",
        ),
    ] = None,
    llama_base: Annotated[
        str,
        typer.Option(
            "--llama-base",
            help="Local llama-server base URL (Tier 3 — heavy; see ADR-0022).",
        ),
    ] = "http://localhost:8090/v1",
) -> None:
    """Drain the pending_review queue with the LLM picker (ADR-0022 §4).

    Runs the hybrid kNN-then-LLM classifier over rows where
    ``triage_state='pending_review'`` — emails the pure-kNN gate
    flagged as ambiguous on first pass. Tier-3-local spins up only
    for this explicit invocation.
    """
    from iris_personal.plugins.email_workflows.discovery import LlamaServerClient
    from iris_personal.plugins.email_workflows.triage import EmailTriageClassifier

    ws_root = workspace_dir or iris_workspace_dir()

    classifier = EmailTriageClassifier(
        workspace_dir=ws_root,
        db_path=db_path or data_path("iris.db"),
        email_db_path=email_db_path or data_path("email.db"),
        naming_client=LlamaServerClient(base_url=llama_base),
    )

    try:
        results = classifier.classify_pending_review_batch(account, limit=limit)
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc

    if not results:
        console.print(
            f"  [yellow]no pending-review emails for {account}[/yellow]"
            "  [dim](queue is empty)[/dim]"
        )
        raise typer.Exit(3)

    classified = [r for r in results if r.category_path is not None]
    still_failed = [r for r in results if r.category_path is None]

    from rich.table import Table

    table = Table(
        title=f"{len(classified)} drained from queue, {len(still_failed)} still pending",
        show_lines=False,
    )
    table.add_column("message id", style="cyan")
    table.add_column("category", style="bold")
    table.add_column("confidence", justify="right")
    table.add_column("note", style="dim")
    for r in results:
        path = r.category_path or "—"
        conf = f"{r.confidence:.2f}" if r.confidence is not None else "—"
        note = r.error or ""
        table.add_row(r.message_id, path, conf, note[:80])
    console.print(table)
    console.print(f"  [dim]wrote classifications to[/dim] [cyan]{classifier.email_db_path}[/cyan]")


def cmd_email_label_holdout(
    account: Annotated[
        str,
        typer.Option(
            "--account",
            help="email_accounts.id slug, e.g. gmail:user@gmail.com",
        ),
    ],
    sample: Annotated[
        int,
        typer.Option(
            "--sample",
            help="How many unlabeled emails to present per session.",
        ),
    ] = 10,
    import_spike: Annotated[
        Path | None,
        typer.Option(
            "--import-spike",
            help="One-shot: import data/spike/phase1_emails_labeled.jsonl "
            "(flat schema) into the holdout JSONL. No interactive prompt.",
        ),
    ] = None,
    workspace_dir: Annotated[
        Path | None,
        typer.Option(
            "--workspace-dir",
            help="Override workspace root (default: $IRIS_HOME/workspace).",
        ),
    ] = None,
    email_db_path: Annotated[
        Path | None,
        typer.Option(
            "--email-db",
            help="Override email.db location (default: email.db in the IRIS data dir).",
        ),
    ] = None,
) -> None:
    """Capture user labels for the held-out evaluation set (ADR-0023).

    Two modes:
      * ``--import-spike <path>``: one-shot bootstrap from the Phase 1
        spike's labeled JSONL. No interactive prompt.
      * Default (interactive): sample N unlabeled emails from email.db,
        prompt for a label per email, write to
        ``<workspace>/email/<slug>/holdout-labels.jsonl``.
    """
    from iris_personal.email.store import EmailStore
    from iris_personal.plugins.email_workflows.holdout import (
        ALLOWED_HOLDOUT_LABELS,
        HoldoutLabel,
        append_label,
        holdout_path,
        import_from_spike,
        labeled_message_ids,
    )

    ws_root = workspace_dir or iris_workspace_dir()
    target_path = holdout_path(ws_root, account)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Spike-import mode ──────────────────────────────────────────
    if import_spike is not None:
        try:
            imported, skipped = import_from_spike(
                import_spike, account_id=account, target_path=target_path
            )
        except FileNotFoundError as exc:
            print_error(str(exc))
            raise typer.Exit(2) from exc
        console.print(
            f"  [bold green]✓[/bold green]  imported [cyan]{imported}[/cyan] labels "
            f"[dim]({skipped} skipped)[/dim] → {target_path}"
        )
        return

    # ─── Interactive mode ───────────────────────────────────────────
    store = EmailStore(db_path=email_db_path or data_path("email.db"))
    store.ensure_schema()

    already_labeled = labeled_message_ids(target_path, account)

    candidates = [
        m for m in store.list_recent(account, limit=sample * 20) if m.id not in already_labeled
    ]

    if not candidates:
        console.print(
            f"  [yellow]no new emails to label for {account}[/yellow]  "
            "[dim](everything in email.db is already labeled)[/dim]"
        )
        raise typer.Exit(3)

    candidates = candidates[:sample]
    console.print(
        f"  [dim]labeling[/dim] [cyan]{len(candidates)}[/cyan] "
        f"[dim]emails (already labeled: {len(already_labeled)})[/dim]"
    )

    # Numbered label menu — match ADR-0023's ALLOWED_HOLDOUT_LABELS order
    label_choices = list(ALLOWED_HOLDOUT_LABELS)
    legend = "  " + "    ".join(f"[bold]{i + 1}[/bold]) {c}" for i, c in enumerate(label_choices))

    saved = 0
    for idx, msg in enumerate(candidates, start=1):
        console.print()
        console.print(f"[dim]{'━' * 76}[/dim]")
        console.print(f"  [bold cyan][{idx}/{len(candidates)}][/bold cyan]  id={msg.id}")
        console.print(f"  [dim]from:[/dim]    {msg.from_address}")
        console.print(f"  [dim]subject:[/dim] {msg.subject}")
        if msg.snippet:
            console.print(f"  [dim]snippet:[/dim] {msg.snippet[:240]}")
        console.print()
        console.print(legend)
        console.print("  [dim]s) skip   q) save+quit[/dim]")
        choice = typer.prompt("  Your choice", default="s", show_default=False).strip().lower()

        if choice == "q":
            console.print(f"\n  saved {saved} labels; exiting. Re-run to resume.")
            break
        if choice == "s":
            console.print("  [dim]→ skipped[/dim]")
            continue
        try:
            idx_choice = int(choice) - 1
        except ValueError:
            console.print(f"  [yellow]unknown key {choice!r}; skipped[/yellow]")
            continue
        if not 0 <= idx_choice < len(label_choices):
            console.print(f"  [yellow]out of range {choice!r}; skipped[/yellow]")
            continue
        true_root = label_choices[idx_choice]
        label = HoldoutLabel(
            message_id=msg.id,
            account_id=account,
            true_root=true_root,
            from_address=msg.from_address,
            from_domain=msg.from_domain,
            subject=msg.subject,
            snippet=msg.snippet,
            received_at=msg.received_at,
        )
        append_label(target_path, label)
        saved += 1
        console.print(f"  [green]→ {true_root}[/green]")

    console.print()
    console.print(
        f"  [bold green]✓[/bold green]  saved [cyan]{saved}[/cyan] "
        f"labels → [cyan]{target_path}[/cyan]"
    )


def cmd_email_recategorize(
    account: Annotated[
        str,
        typer.Option(
            "--account",
            help="email_accounts.id slug, e.g. gmail:user@gmail.com",
        ),
    ],
    message_id: Annotated[
        str,
        typer.Option(
            "--message-id",
            help="Provider-native message id of the email being recategorized.",
        ),
    ],
    to_path: Annotated[
        str,
        typer.Option(
            "--to",
            help="Full category path, e.g. email/finance/banking/northwind-savings. "
            "Must already exist in the categories table.",
        ),
    ],
    reason: Annotated[
        str | None,
        typer.Option(
            "--reason",
            help="Optional free-form note saved into the correction history payload.",
        ),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option(
            "--db-path",
            help="Override iris.db location (default: iris.db in the IRIS data dir).",
        ),
    ] = None,
    email_db_path: Annotated[
        Path | None,
        typer.Option(
            "--email-db",
            help="Override email.db location (default: email.db in the IRIS data dir).",
        ),
    ] = None,
) -> None:
    """Record a user-classification correction (ADR-0024).

    Updates the email row's classified_category + writes a row to
    categories_history with ``source='user-classification-correction'``.
    The Track 1J audit CLI (``iris email corrections list``) and
    measurement flow (``iris email knn-gate --include-corrections``)
    read from this history.

    Confidence is set to 1.0 — the user's assertion is ground truth.
    """
    import sqlite3

    from iris_personal.email.category_store import CategoryStore
    from iris_personal.email.store import EmailStore

    iris_db = db_path or data_path("iris.db")
    email_db = email_db_path or data_path("email.db")

    # ─── Validate the target category exists + is active ────────────
    category_store = CategoryStore(db_path=iris_db)
    category_store.ensure_schema()
    target = category_store.get(to_path)
    if target is None or not target.active:
        print_error(f"category {to_path!r} not found or inactive in {iris_db}")
        raise typer.Exit(3)

    # ─── Validate the message exists + capture its current state ─────
    email_store = EmailStore(db_path=email_db)
    email_store.ensure_schema()
    message = email_store.get(message_id)
    if message is None:
        print_error(f"no email with id={message_id!r} in {email_db}")
        raise typer.Exit(2)

    # Pull the previous classification + classifier tag directly via SQL
    # (EmailMessage doesn't expose those columns yet).
    with sqlite3.connect(email_db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT classified_category, sensitivity FROM emails WHERE id = ?",
            (message_id,),
        ).fetchone()
    old_path = row["classified_category"] if row else None

    # ─── Commit: history first, then email row ──────────────────────
    category_store.record_correction(
        message_id=message_id,
        account_id=account,
        old_path=old_path,
        new_path=to_path,
        previous_classifier=None,  # we don't track the classifier tag per-row yet
        reason=reason,
    )
    email_store.mark_classified(
        message_id,
        category=to_path,
        confidence=1.0,
        sensitivity=target.sensitivity,
    )

    # ADR-0025 §6 — re-emit email.classified so the wiki-ingestion
    # subscriber follows user corrections. Subscribers downstream see
    # the same payload shape as the auto-fire path; classifier is
    # tagged to distinguish corrections from kNN/LLM classifications.
    from iris_harness.sdk.events import get_default_bus
    from iris_personal.email.events import EMAIL_CLASSIFIED, EmailClassifiedPayload

    get_default_bus().emit_sync(
        EMAIL_CLASSIFIED,
        EmailClassifiedPayload(
            id=message_id,
            account_id=account,
            category_path=to_path,
            confidence=1.0,
            classifier="user-classification-correction",
        ),
    )

    console.print(
        f"  [bold green]✓[/bold green]  {message_id} recategorized: "
        f"[dim]{old_path or '(unclassified)'}[/dim] → [cyan]{to_path}[/cyan]"
    )
    if reason:
        console.print(f"  [dim]reason: {reason}[/dim]")


def cmd_email_reingest_wiki(
    account: Annotated[
        str,
        typer.Option(
            "--account",
            help="email_accounts.id slug, e.g. gmail:user@gmail.com",
        ),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Cap on emails to re-emit (default 200).",
        ),
    ] = 200,
    since: Annotated[
        datetime | None,
        typer.Option(
            "--since",
            help="ISO timestamp; only re-emit emails classified after this point.",
            formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
        ),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option(
            "--db-path",
            help="Override iris.db location (default: iris.db in the IRIS data dir).",
        ),
    ] = None,
    email_db_path: Annotated[
        Path | None,
        typer.Option(
            "--email-db",
            help="Override email.db location (default: email.db in the IRIS data dir).",
        ),
    ] = None,
) -> None:
    """Backfill the wiki for already-classified emails (ADR-0025 §7).

    Walks ``emails`` for the account where ``classified_category IS NOT
    NULL`` and re-emits ``email.classified`` for each — the same event
    the auto-fire path produces, so the wiki subscriber treats backfill
    rows identically to live ones. Vendor-classified rows (a mailbox tab,
    not an IRIS verdict) are skipped: the live path never emits for them.

    Idempotent at the event level. Wiki pages may accumulate "Recent
    Activity" entries on each re-run; treated as audit trail.
    """
    import sqlite3

    from iris_harness.sdk.events import get_default_bus
    from iris_harness.sdk.memory import WikiEngine, subscribe_wiki_ingest_consumer
    from iris_personal.email.events import EMAIL_CLASSIFIED, EmailClassifiedPayload
    from iris_personal.email.store import EmailStore, visible_condition
    from iris_personal.plugins.email_workflows.wiki_ingestion import (
        subscribe_email_classified_to_wiki,
    )

    email_db = email_db_path or data_path("email.db")
    if not email_db.exists():
        print_error(f"email.db not found at {email_db}")
        raise typer.Exit(2)
    EmailStore(db_path=email_db).ensure_schema()  # classified_source may be new

    # Pull the rows we want to re-emit. classified_at carries an ISO
    # string in the schema; SQLite compares those lexicographically
    # — fine for our --since filter.
    sql = (
        "SELECT id, classified_category, classified_confidence "
        "FROM emails WHERE account_id = ? "
        "  AND classified_category IS NOT NULL"
        "  AND (classified_source IS NULL OR classified_source != 'vendor')"
    )
    params: list[Any] = [account]
    if since is not None:
        sql += " AND classified_at >= ?"
        params.append(since.isoformat())
    with sqlite3.connect(email_db) as conn:
        conn.row_factory = sqlite3.Row
        # A held message (not released by its plugin yet) is not re-emitted.
        sql += f" AND {visible_condition(conn)} ORDER BY classified_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        console.print(
            f"  [yellow]no classified emails for {account}[/yellow]  "
            "[dim](run `iris email triage` first)[/dim]"
        )
        raise typer.Exit(3)

    bus = get_default_bus()

    # The runtime auto-fire path wires these subscribers during
    # IrisRuntime.startup(). A one-shot CLI process doesn't run the
    # full runtime, so wire them here per ADR-0025 §7. WikiEngine
    # construction is cheap when SemanticIndex is None (no ChromaDB
    # bulk-index sync at startup); the per-ingest rebuild still
    # happens via WikiEngine.ingest as expected.
    wiki_root = data_path("wiki")
    wiki_root.mkdir(parents=True, exist_ok=True)
    # This command IS the explicit user action, so it still ingests while the
    # automatic feeds are off (IRIS_WIKI_INGEST unset).
    wiki = WikiEngine(wiki_root=wiki_root, ingest_enabled=True)
    subscribe_email_classified_to_wiki(bus=bus)
    subscribe_wiki_ingest_consumer(wiki, bus=bus)

    for row in rows:
        bus.emit_sync(
            EMAIL_CLASSIFIED,
            EmailClassifiedPayload(
                id=row["id"],
                account_id=account,
                category_path=row["classified_category"],
                confidence=float(row["classified_confidence"] or 0.0),
                classifier="reingest-wiki",
            ),
        )

    console.print(
        f"  [bold green]✓[/bold green]  re-emitted [cyan]{len(rows)}[/cyan] "
        f"email.classified event(s) for [cyan]{account}[/cyan]"
    )
    console.print(f"  [dim]wiki pages written to[/dim] [cyan]{wiki_root}[/cyan]")


corrections_app = typer.Typer(
    name="corrections",
    help="Audit user-classification corrections recorded via " "`iris email recategorize`.",
    no_args_is_help=True,
)


@corrections_app.command("list")
def cmd_email_corrections_list(
    account: Annotated[
        str,
        typer.Option(
            "--account",
            help="email_accounts.id slug, e.g. gmail:user@gmail.com",
        ),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Cap on the number of corrections returned.",
        ),
    ] = 50,
    since: Annotated[
        datetime | None,
        typer.Option(
            "--since",
            help="ISO timestamp; only return corrections edited after this point.",
            formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
        ),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option(
            "--db-path",
            help="Override iris.db location (default: iris.db in the IRIS data dir).",
        ),
    ] = None,
) -> None:
    """List user-classification corrections for the account.

    Read-only audit per ADR-0024. Returns rows from categories_history
    where ``source='user-classification-correction'`` for the given
    account, ordered most-recent-first.
    """
    from iris_personal.email.category_store import CategoryStore

    store = CategoryStore(db_path=db_path or data_path("iris.db"))
    store.ensure_schema()
    rows = store.list_corrections(account_id=account, since=since, limit=limit)

    if not rows:
        console.print(
            f"  [yellow]no corrections for {account}[/yellow]"
            "  [dim](run `iris email recategorize` to capture one)[/dim]"
        )
        raise typer.Exit(3)

    from rich.table import Table

    table = Table(title=f"{len(rows)} correction(s) for {account}", show_lines=False)
    table.add_column("edited_at", style="dim")
    table.add_column("message_id", style="cyan")
    table.add_column("from → to", style="bold")
    table.add_column("reason", style="dim")
    for r in rows:
        payload = r["payload"]
        edited_at = (r["edited_at"] or "")[:19]
        old = payload.get("old_path") or "(unclassified)"
        new = payload.get("new_path") or "—"
        reason = payload.get("reason") or ""
        table.add_row(edited_at, payload.get("message_id", "—"), f"{old} → {new}", reason[:80])
    console.print(table)


def cmd_email_knn_gate(
    account: Annotated[
        str,
        typer.Option(
            "--account",
            help="email_accounts.id slug, e.g. gmail:user@gmail.com",
        ),
    ],
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels",
            help="Override holdout-labels.jsonl path (default: "
            "<workspace>/email/<slug>/holdout-labels.jsonl).",
        ),
    ] = None,
    writeup_to: Annotated[
        Path | None,
        typer.Option(
            "--writeup-to",
            help="Write a markdown writeup of the measurement to this path.",
        ),
    ] = None,
    workspace_dir: Annotated[
        Path | None,
        typer.Option(
            "--workspace-dir",
            help="Override workspace root (default: $IRIS_HOME/workspace).",
        ),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option(
            "--db-path",
            help="Override iris.db location (default: iris.db in the IRIS data dir).",
        ),
    ] = None,
    email_db_path: Annotated[
        Path | None,
        typer.Option(
            "--email-db",
            help="Override email.db location (default: email.db in the IRIS data dir).",
        ),
    ] = None,
    include_corrections: Annotated[
        bool,
        typer.Option(
            "--include-corrections",
            help="Augment the holdout with user corrections from "
            "categories_history (ADR-0024). Each correction becomes "
            "a synthetic HoldoutLabel — lets the measurement grow "
            "without manual labeling.",
        ),
    ] = False,
) -> None:
    """Measure pure-kNN gate accuracy against the held-out labeled set.

    Per ADR-0023: sweeps (cos_min × margin_min), recommends a
    threshold subject to ≥50% gated rate, surfaces a confusion
    matrix at the recommended cell. Output is a Rich table to
    stdout plus (optional) markdown writeup.

    With ``--include-corrections`` (ADR-0024): user-classification
    corrections recorded via ``iris email recategorize`` join the
    holdout as additional ground-truth labels.
    """
    from iris_personal.plugins.email_workflows.knn_gate import KnnGateRunner, render_writeup

    ws_root = workspace_dir or iris_workspace_dir()
    runner = KnnGateRunner(
        workspace_dir=ws_root,
        db_path=db_path or data_path("iris.db"),
        email_db_path=email_db_path or data_path("email.db"),
    )

    try:
        report = runner.measure(
            account, labels_path=labels, include_corrections=include_corrections
        )
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc

    from rich.table import Table

    # ─── Label distribution ─────────────────────────────────────────
    dist_table = Table(title=f"Holdout distribution ({report.total_labels} labels)")
    dist_table.add_column("root", style="bold")
    dist_table.add_column("count", justify="right")
    for root, count in sorted(report.label_distribution.items(), key=lambda kv: -kv[1]):
        dist_table.add_row(root, str(count))
    console.print(dist_table)

    # ─── Sweep grid ─────────────────────────────────────────────────
    sweep_table = Table(title="Threshold sweep — gated accuracy / queue rate")
    sweep_table.add_column("cos_min", justify="right")
    sweep_table.add_column("margin_min", justify="right")
    sweep_table.add_column("gated", justify="right")
    sweep_table.add_column("accuracy", justify="right")
    sweep_table.add_column("queue", justify="right", style="dim")
    for cell in report.sweep:
        is_recommended = (
            cell.cos_min == report.recommended_cos_min
            and cell.margin_min == report.recommended_margin_min
        )
        style = "bold green" if is_recommended else None
        sweep_table.add_row(
            f"{cell.cos_min:.2f}",
            f"{cell.margin_min:.2f}",
            f"{cell.gated_count}/{cell.total}",
            f"{cell.gated_accuracy:.1%}",
            f"{cell.queue_rate:.1%}",
            style=style,
        )
    console.print(sweep_table)

    # ─── Recommendation ─────────────────────────────────────────────
    console.print()
    console.print(
        f"  [bold green]Recommended thresholds[/bold green]: "
        f"cos_min=[cyan]{report.recommended_cos_min:.2f}[/cyan]  "
        f"margin_min=[cyan]{report.recommended_margin_min:.2f}[/cyan]"
    )
    console.print(
        f"  [dim]→ {report.recommended_accuracy:.1%} accuracy on "
        f"{report.recommended_gated_count}/{report.total_labels} gated labels[/dim]"
    )
    if report.note:
        console.print(f"  [yellow]note: {report.note}[/yellow]")

    if writeup_to is not None:
        writeup_to.parent.mkdir(parents=True, exist_ok=True)
        writeup_to.write_text(render_writeup(report))
        console.print(f"  [dim]wrote markdown writeup → [/dim][cyan]{writeup_to}[/cyan]")


def cmd_email_detect_followups(
    account: Annotated[
        str,
        typer.Option(
            "--account",
            help="email_accounts.id slug, e.g. gmail:user@gmail.com",
        ),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Cap on classified emails scanned in this run.",
        ),
    ] = 50,
    since: Annotated[
        datetime | None,
        typer.Option(
            "--since",
            help="Only emails received on or after this date.",
            formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
        ),
    ] = None,
    confidence_threshold: Annotated[
        float,
        typer.Option(
            "--confidence",
            help="Skip detections below this confidence (default 0.6).",
        ),
    ] = 0.6,
    email_db_path: Annotated[
        Path | None,
        typer.Option("--email-db", help="Override email.db location."),
    ] = None,
    tasks_db_path: Annotated[
        Path | None,
        typer.Option("--tasks-db", help="Override tasks.db location."),
    ] = None,
    llama_base: Annotated[
        str,
        typer.Option(
            "--llama-base",
            help="Local llama-server base URL (Tier 3 local — heavy; see ADR-0022).",
        ),
    ] = "http://localhost:8090/v1",
) -> None:
    """Detect followups over recent classified emails (Tier 3 local).

    User-invoked per ADR-0022's "Tier 3 only on demand" stance —
    detection costs an LLM call per email. Auto-resolution of
    existing followups runs continuously as inbound mail arrives;
    detection is the part you batch.

    Skips emails without a ``thread_id`` (no anchor for auto-resolution)
    and emails whose thread already has a tracked followup.
    """
    from iris_harness.sdk.learning import SurfaceFeedbackStore
    from iris_harness.sdk.tasks import TaskStore
    from iris_personal.email.store import EmailStore
    from iris_personal.plugins.email_workflows.discovery import LlamaServerClient
    from iris_personal.plugins.email_workflows.followup import (
        FollowupDetector,
        detect_and_persist,
    )

    estore = EmailStore(db_path=email_db_path or data_path("email.db"))
    estore.ensure_schema()
    tstore = TaskStore(db_path=tasks_db_path or data_path("tasks.db"))
    tstore.ensure_schema()
    feedback = SurfaceFeedbackStore()
    feedback.ensure_schema()
    detector = FollowupDetector(client=LlamaServerClient(base_url=llama_base))

    candidates = [m for m in estore.list_recent(account, limit=limit, since=since) if m.thread_id]
    if not candidates:
        console.print(f"  [yellow]no thread-bearing emails for {account} in window[/yellow]")
        raise typer.Exit(3)

    outcomes = []
    for email in candidates:
        # category_path=None → detect_and_persist reads the email's own
        # classified_category; bulk-label + non-actionable-root + user
        # feedback gates run before the Tier 3 call (issue 0028).
        outcome = detect_and_persist(
            email,
            category_path=None,
            detector=detector,
            task_store=tstore,
            confidence_threshold=confidence_threshold,
            feedback_store=feedback,
        )
        outcomes.append(outcome)

    created = [o for o in outcomes if o.action == "created"]
    skipped = [o for o in outcomes if o.action.startswith("skipped")]

    from rich.table import Table

    if created:
        table = Table(
            title=f"{len(created)} followup(s) created, "
            f"{len(skipped)} skipped, {len(outcomes)} scanned",
            show_lines=False,
        )
        table.add_column("task id", style="cyan", no_wrap=True)
        table.add_column("email id", style="dim", no_wrap=True)
        table.add_column("rationale")
        for o in created:
            short = (o.task_id or "?")[:8]
            table.add_row(short, o.email_id, o.detail[:80])
        console.print(table)
    else:
        console.print(
            f"  [yellow]no new followups[/yellow] "
            f"[dim]({len(skipped)} skipped, {len(outcomes)} scanned)[/dim]"
        )

    if skipped:
        from collections import Counter

        kinds = Counter(o.action for o in skipped)
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        console.print(f"  [dim]skip breakdown:[/dim] {breakdown}")


# ---------------------------------------------------------------------------
# iris email search (Track 1M — ADR-0026; left main.py at M6.1b)
# ---------------------------------------------------------------------------


def cmd_email_search(
    query: Annotated[
        str,
        typer.Argument(
            help='FTS5 query. Supports phrase ("acme apparel"), boolean '
            "(northwind AND statement), prefix (stat*), column qualifier "
            "(from_address:northwind).",
        ),
    ],
    account: Annotated[
        str | None,
        typer.Option("--account", help="email_accounts.id slug; filter to one account."),
    ] = None,
    category: Annotated[
        str | None,
        typer.Option(
            "--category",
            help="Category path prefix; e.g. email/shopping/ matches all " "shopping leaves.",
        ),
    ] = None,
    since: Annotated[
        datetime | None,
        typer.Option(
            "--since",
            help="ISO timestamp; only emails received after this point.",
            formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Cap on results (default 20).",
        ),
    ] = 20,
    email_db_path: Annotated[
        Path | None,
        typer.Option(
            "--email-db",
            help="Override email.db location (default: email.db in the IRIS data dir).",
        ),
    ] = None,
) -> None:
    """Full-text search over the locally-stored email envelope.

    Per ADR-0026 — searches subject + from_address + snippet via SQLite
    FTS5. BM25 ranking with snippet highlighting. Filters compose at
    SQL level.
    """
    import sqlite3

    from iris_personal.email.store import EmailStore

    store = EmailStore(db_path=email_db_path or data_path("email.db"))
    store.ensure_schema()

    try:
        hits = store.search(
            query,
            account_id=account,
            category_prefix=category,
            since=since,
            limit=limit,
        )
    except sqlite3.OperationalError as exc:
        print_error(f"FTS5 query error: {exc}")
        raise typer.Exit(2) from exc

    if not hits:
        console.print(f"  [yellow]no matches for [bold]{query}[/bold][/yellow]")
        return  # exit 0 — empty result is valid

    from rich.table import Table

    table = Table(
        title=f"{len(hits)} match(es) for [bold]{query}[/bold]",
        show_lines=False,
    )
    table.add_column("received", style="dim", no_wrap=True)
    table.add_column("from", style="cyan", no_wrap=True)
    table.add_column("category", style="dim")
    table.add_column("subject + snippet")
    for hit in hits:
        received = hit.received_at.strftime("%Y-%m-%d")
        sender = hit.from_address[:32]
        cat = (hit.classified_category or "—").replace("email/", "")[:32]
        snippet = (hit.subject[:60] + " — " if hit.subject else "") + hit.snippet_highlighted
        # Rich tags for the FTS5 <mark>...</mark> highlights
        snippet = snippet.replace("<mark>", "[bold yellow]").replace("</mark>", "[/]")
        table.add_row(received, sender, cat, snippet)
    console.print(table)


# ---------------------------------------------------------------------------
# iris email repair-domains (2026-09-24 parser fix)
# ---------------------------------------------------------------------------


def cmd_email_repair_domains(
    apply: Annotated[
        bool, typer.Option("--apply", help="Write the fix (default: a dry run that lists it).")
    ] = False,
    email_db_path: Annotated[
        Path | None,
        typer.Option(
            "--email-db",
            help="Override email.db location (default: email.db in the IRIS data dir).",
        ),
    ] = None,
) -> None:
    """Recompute stored sender domains the old parser got wrong.

    '"alerts@bank.example" <alerts@bank.example>' was stored as 'bank.example"', so
    those emails never matched their institution. Dry run by default.
    """
    from iris_personal.email.store import EmailStore

    store = EmailStore(db_path=email_db_path or data_path("email.db"))
    store.ensure_schema()
    fixes = store.repair_from_domains(apply=apply)
    head = "Fixed" if apply else "Would fix (dry run; add --apply to write)"
    console.print(f"{head}: {len(fixes)} email(s).", markup=False)
    for stored, correct in sorted({(old, new) for _, old, new in fixes}):
        count = sum(1 for _, o, n in fixes if (o, n) == (stored, correct))
        console.print(f"- {stored!r} → {correct!r} ({count})", markup=False, highlight=False)


# ---------------------------------------------------------------------------
# iris email semantic-index (ADR-0071 slice 1; left main.py at M6.1b)
# ---------------------------------------------------------------------------


def cmd_email_semantic_index(
    account: Annotated[
        str | None,
        typer.Option("--account", help="email_accounts.id slug; index only this account."),
    ] = None,
    email_db_path: Annotated[
        Path | None,
        typer.Option(
            "--email-db",
            help="Override email.db location (default: email.db in the IRIS data dir).",
        ),
    ] = None,
    persist_dir: Annotated[
        Path | None,
        typer.Option(
            "--persist-dir",
            help="Vector-index location (default: email_semantic in the IRIS data dir).",
        ),
    ] = None,
) -> None:
    """Backfill the semantic email index (ADR-0071).

    Embeds stored messages (sender+subject+snippet, MiniLM) into a local Chroma
    vector store so the chat agent's ``semantic_search`` tool can retrieve by
    meaning. Stores ONLY vectors + ids + minimal metadata — no snippet/body text.
    Idempotent: re-run any time to pick up new mail. Enable retrieval with
    ``IRIS_EMAIL_SEMANTIC_SEARCH=1``.
    """
    from iris_personal.email.semantic_index import EmailSemanticIndex, backfill_semantic_index
    from iris_personal.email.store import EmailStore

    store = EmailStore(db_path=email_db_path or data_path("email.db"))
    store.ensure_schema()
    index = EmailSemanticIndex(persist_dir=persist_dir or data_path("email_semantic"))
    if not index.is_ready:
        print_error("semantic index unavailable — is chromadb installed?")
        raise typer.Exit(2)

    if account:
        indexed = index.index_messages(store.list_recent(account, limit=10_000))
    else:
        indexed = backfill_semantic_index(email_store=store, index=index)
    console.print(
        f"  [green]indexed {indexed} message(s)[/green]; "
        f"index now holds [bold]{index.count()}[/bold] vector(s)."
    )


# ---------------------------------------------------------------------------
# iris email judge / judgments (loop-proof PR 5)
# ---------------------------------------------------------------------------


def _figures_text(fields: dict[str, Any]) -> str:
    from iris_personal.plugins.email_workflows.judge import FIGURE_FIELDS

    parts = [f"{k}={fields[k]}" for k in FIGURE_FIELDS if k in fields]
    if "guess" in fields:
        parts.append(f"guess={fields['guess']}")
    if fields.get("read") and fields["read"] != "body":
        parts.append(f"read={fields['read']}")
    return " ".join(parts)


def _mount_gmail_for_body_reads() -> None:
    """The CLI runs with no runtime, so no plugin has mounted a mail provider: mount
    Gmail's if it is installed; without it the judge reads the stored snippets."""
    try:
        from iris_personal.email.providers import register_mail_provider
        from iris_personal.plugins.gmail.provider import GmailProvider

        register_mail_provider(GmailProvider())
    except Exception:  # noqa: BLE001 — no Gmail plugin: snippets only
        console.print("  [yellow]no mail provider; the judge reads stored snippets[/yellow]")


def cmd_email_judge(
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", min=1, help="Most emails to judge in this run."),
    ] = 30,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Judge and print; write no rows and no labels."),
    ] = False,
    waiting_only: Annotated[
        bool,
        typer.Option(
            "--waiting-only",
            help="Only the queued emails; by default the newest unjudged mail fills the rest.",
        ),
    ] = False,
    email_db_path: Annotated[
        Path | None,
        typer.Option(
            "--email-db",
            help="Override email.db location (default: email.db in the IRIS data dir).",
        ),
    ] = None,
    tiers_path: Annotated[
        Path | None,
        typer.Option(
            "--tiers",
            help="llm_tiers.yaml to read the email_judge tier from (default: the config dir's).",
        ),
    ] = None,
) -> None:
    """Judge email now on the local email_judge model and print each verdict."""
    tiers_path = tiers_path or config_path("llm_tiers.yaml")
    from rich.table import Table

    from iris_harness.sdk.llm import TierRouter
    from iris_personal.plugins.email_workflows.judge import llm_from_router
    from iris_personal.plugins.email_workflows.judge_config import JudgeConfig
    from iris_personal.plugins.email_workflows.judge_wiring import judge_and_release

    config = JudgeConfig.load(config_dir())
    llm = llm_from_router(TierRouter.load_from_yaml(tiers_path))
    if llm is None:
        print_error(f"no local email_judge tier in {tiers_path}")
        raise typer.Exit(2)
    _mount_gmail_for_body_reads()
    # Judged waiting mail is released as email.new_arrived on the process bus; with no
    # runtime here nothing listens, so triage and the index catch up on their own runs.
    report, note = judge_and_release(
        llm=llm,
        config_dir=config_dir(),
        db_path=email_db_path or data_path("email.db"),
        limit=limit,
        backfill=0 if waiting_only else limit,
        dry_run=dry_run,
    )
    if not report.enabled:
        print_error("the judge is off (IRIS_EMAIL_JUDGE); nothing judged")
        raise typer.Exit(3)

    title = ("DRY RUN — nothing written. " if dry_run else "") + report.summary()
    table = Table(title=title, show_lines=False)
    table.add_column("sender", style="cyan", max_width=32)
    table.add_column("subject", max_width=48)
    table.add_column("bucket", style="bold")
    table.add_column("conf", justify="right")
    table.add_column("figures", style="dim", max_width=40)
    table.add_column("ms", justify="right")
    for item in report.items:
        v = item.verdict
        table.add_row(
            item.sender,
            item.subject,
            config.name(v.bucket) + (" [red](error)[/red]" if v.error else ""),
            f"{v.confidence:.2f}" if v.confidence is not None else "—",
            _figures_text(v.fields) or (v.error[:40] if v.error else ""),
            str(v.latency_ms) if v.latency_ms is not None else "—",
        )
    console.print(table)
    timed = [i.verdict.latency_ms for i in report.items if i.verdict.latency_ms is not None]
    if timed:
        console.print(
            f"  [dim]mean {sum(timed) // len(timed)} ms per email, run {report.run_id}[/dim]"
        )
    if report.unreachable:
        console.print(
            f"  [yellow]the model was unreachable; {report.waiting} email(s) wait[/yellow] "
            f"[dim]({report.unreachable_error[:120]})[/dim]"
        )
    console.print(f"  [dim]{note}[/dim]")


def cmd_email_judgments(
    bucket: Annotated[
        str | None,
        typer.Option("--bucket", "-b", help="Only this (effective) bucket."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1)] = 50,
    email_db_path: Annotated[
        Path | None,
        typer.Option(
            "--email-db",
            help="Override email.db location (default: email.db in the IRIS data dir).",
        ),
    ] = None,
) -> None:
    """List the judge's recent verdicts, newest first (the owner's bucket when corrected)."""
    from rich.table import Table

    from iris_personal.email.store import EmailStore
    from iris_personal.plugins.email_workflows.judge_config import JudgeConfig
    from iris_personal.plugins.email_workflows.judgments import JudgmentStore

    email_store = EmailStore(db_path=email_db_path or data_path("email.db"))
    email_store.ensure_schema()
    store = JudgmentStore(db_path=email_store.db_path)
    store.ensure_schema()
    config = JudgeConfig.load(config_dir())
    rows = store.recent(bucket=bucket, limit=limit)
    table = Table(
        title=f"{len(rows)} judgment(s); {store.count_waiting()} waiting", show_lines=False
    )
    table.add_column("judged", style="dim")
    table.add_column("sender", style="cyan", max_width=32)
    table.add_column("subject", max_width=48)
    table.add_column("bucket", style="bold")
    table.add_column("conf", justify="right")
    table.add_column("figures", style="dim", max_width=40)
    table.add_column("ms", justify="right")
    for j in rows:
        message = email_store.get(j.message_id)
        shown = config.name(j.effective_bucket or "")
        if j.owner_bucket:
            shown += f" (owner; judge: {config.name(j.bucket or '')})"
        table.add_row(
            (j.judged_at or "")[:16],
            message.from_address if message else "—",
            message.subject if message else j.message_id,
            shown,
            f"{j.confidence:.2f}" if j.confidence is not None else "—",
            _figures_text(j.fields) or j.error[:40],
            str(j.latency_ms) if j.latency_ms is not None else "—",
        )
    console.print(table)


def cmd_email_demo(
    home: Annotated[
        Path | None,
        typer.Option(
            "--home",
            help="The demo's own IRIS home (default: $IRIS_DEMO_HOME, else ~/.iris-demo). "
            "Never your real profile.",
        ),
    ] = None,
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Delete the demo home first and start over."),
    ] = False,
) -> None:
    """Try IRIS on a synthetic mailbox: fetch, judge, first digest. Offline, no credentials.

    Runs in an isolated demo home with a scripted demo model (no model server needed),
    then prints the digest and what IRIS did, audit rows included. Re-running is safe.
    """
    import os
    import subprocess
    import sys

    from iris_personal.plugins.email_workflows.demo.home import (
        DemoHomeError,
        default_home,
        demo_environment,
        prepare_home,
    )

    try:
        demo_home = prepare_home(home or default_home(), reset=reset)
    except DemoHomeError as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc
    console.print(f"[dim]demo home: {demo_home}[/dim]")
    # A child process with an environment built from scratch: nothing of this process's
    # settings (the owner's .env among them) reaches the demo's stores.
    completed = subprocess.run(
        [sys.executable, "-m", "iris_personal.plugins.email_workflows.demo.run"],
        env=demo_environment(demo_home, os.environ),
        cwd=demo_home,
        check=False,
    )
    if completed.returncode != 0:
        raise typer.Exit(completed.returncode)
