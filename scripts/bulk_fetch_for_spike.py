"""One-shot bulk Gmail pull for the Phase 1 Track 1E corpus-discovery spike.

Resets the gmail sync cursor (optional) and runs ``fetch_new_emails`` with a
wide cold-start window so the spike has a meaningful corpus (~1-2k messages)
to embed and cluster.

Idempotent: ``upsert_many`` is keyed on the provider-native message id, so
re-running this script will not duplicate rows.

Usage:
    poetry run python scripts/bulk_fetch_for_spike.py \\
        --account gmail:user@gmail.com --max 2000 --days 180 --reset-cursor
"""

from __future__ import annotations

import argparse
import logging
import sqlite3

from iris_personal.email.store import EmailStore
from iris_personal.plugins.gmail.gmail_fetch import (
    GMAIL_HISTORY_CURSOR_KIND,
    GMAIL_PROVIDER,
    fetch_new_emails,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _delete_cursor(store: EmailStore, account_id: str) -> None:
    """EmailStore.set_cursor takes str — for a reset we do a raw DELETE."""
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "DELETE FROM sync_cursors " "WHERE provider = ? AND account_id = ? AND cursor_kind = ?",
            (GMAIL_PROVIDER, account_id, GMAIL_HISTORY_CURSOR_KIND),
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--account", required=True, help="email_accounts.id, e.g. gmail:user@gmail.com")
    p.add_argument("--max", type=int, default=2000, dest="max_messages")
    p.add_argument("--days", type=int, default=180, dest="cold_start_days")
    p.add_argument(
        "--reset-cursor",
        action="store_true",
        help="Delete the existing sync cursor first to force a cold-start fetch.",
    )
    args = p.parse_args()

    store = EmailStore()
    store.ensure_schema()

    if args.reset_cursor:
        _delete_cursor(store, args.account)
        logging.info("reset cursor for %s", args.account)

    result = fetch_new_emails(
        args.account,
        store=store,
        max_messages=args.max_messages,
        cold_start_days=args.cold_start_days,
    )
    logging.info(
        "fetched=%d  new_cursor=%s  fell_back_to_cold_start=%s",
        result.fetched,
        result.new_cursor,
        result.fell_back_to_cold_start,
    )


if __name__ == "__main__":
    main()
