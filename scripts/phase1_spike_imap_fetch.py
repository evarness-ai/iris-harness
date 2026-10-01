"""Phase 1 spike — fetch a balanced 50-email sample from Gmail via IMAP.

Throwaway diagnostic. Delete after Phase 1 lands.

Sampling strategy: 10 most-recent emails from each of Gmail's 5
categories (PERSONAL, SOCIAL, PROMOTIONS, UPDATES, FORUMS) for a
balanced 50-email evaluation set. See ADR-Q2/Phase-1 spike brief.

Auth: app password via IRIS_SPIKE_GMAIL_PASSWORD env var.

Usage:
    export IRIS_SPIKE_GMAIL_PASSWORD="<app-password>"
    poetry run python scripts/phase1_spike_imap_fetch.py --user user@example.com

Output: data/spike/phase1_emails.jsonl (one JSON object per email).
"""

from __future__ import annotations

import argparse
import email
import imaplib
import json
import os
import sys
from dataclasses import asdict, dataclass
from email.header import decode_header
from email.message import Message
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "spike"
OUTPUT_PATH = OUTPUT_DIR / "phase1_emails.jsonl"

# Gmail's 5 categories. Mapped to Gmail search syntax (X-GM-RAW).
# 'personal' = the Primary tab = category:primary in Gmail's search dialect.
CATEGORIES: dict[str, str] = {
    "personal": "category:primary",
    "social": "category:social",
    "promotions": "category:promotions",
    "updates": "category:updates",
    "forums": "category:forums",
}

PER_CATEGORY = 10
SNIPPET_LEN = 400


@dataclass(frozen=True)
class FetchedEmail:
    uid: str
    gmail_category: str  # the canonical category this row was sampled FOR
    gmail_labels: list[str]  # raw X-GM-LABELS as returned (for analysis)
    from_address: str
    from_domain: str
    subject: str
    snippet: str
    received_at: str  # ISO 8601 string, best-effort


def _decode(value: str | bytes | None) -> str:
    """Best-effort decode of a possibly-MIME-encoded header value."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            value = str(value)
    parts = []
    for fragment, charset in decode_header(value):
        if isinstance(fragment, bytes):
            try:
                parts.append(fragment.decode(charset or "utf-8", errors="replace"))
            except Exception:
                parts.append(fragment.decode("utf-8", errors="replace"))
        else:
            parts.append(fragment)
    return "".join(parts).strip()


def _from_domain(from_header: str) -> str:
    """Pull the bare domain out of a From header value."""
    if "@" not in from_header:
        return ""
    after_at = from_header.split("@", 1)[1]
    return after_at.split(">", 1)[0].split()[0].strip().lower()


def _snippet_from_msg(msg: Message, limit: int = SNIPPET_LEN) -> str:
    """Extract the first ``limit`` chars of plaintext body."""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        text = payload.decode(charset, errors="replace")
                    except Exception:
                        text = payload.decode("utf-8", errors="replace")
                    text = " ".join(text.split())  # collapse whitespace
                    if text:
                        return text[:limit]
        return ""
    payload = msg.get_payload(decode=True)
    if isinstance(payload, bytes):
        text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
        return " ".join(text.split())[:limit]
    return ""


def _parse_labels(raw: bytes | str) -> list[str]:
    """Parse the X-GM-LABELS response payload into a clean list of labels.

    Format is `(X-GM-LABELS ("Label1" "Label2"))` — parse the inner list.
    """
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    out: list[str] = []
    if "(" not in text:
        return out
    body = text[text.find("(", text.find("X-GM-LABELS")) :]
    # Strip outer paren
    body = body.lstrip("(").rstrip(")")
    # Naive tokenize: split on quoted regions and bare tokens
    cur = ""
    in_quote = False
    for ch in body:
        if ch == '"':
            if in_quote:
                if cur:
                    out.append(cur)
                cur = ""
                in_quote = False
            else:
                in_quote = True
        elif in_quote:
            cur += ch
        elif ch.isspace():
            if cur:
                out.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _search_category(conn: imaplib.IMAP4_SSL, gmail_query: str) -> list[bytes]:
    """Search using Gmail's X-GM-RAW extension; return matching UIDs."""
    typ, data = conn.uid("SEARCH", None, "X-GM-RAW", f'"{gmail_query}"')
    if typ != "OK" or not data or not data[0]:
        return []
    return data[0].split()


def _fetch_one(conn: imaplib.IMAP4_SSL, uid: bytes, category_tag: str) -> FetchedEmail | None:
    """Fetch a single email by UID with labels and headers."""
    typ, data = conn.uid(
        "FETCH", uid, "(X-GM-LABELS INTERNALDATE BODY.PEEK[HEADER] BODY.PEEK[TEXT])"
    )
    if typ != "OK" or not data:
        return None

    # Response: a list with alternating tuple/bytes elements. Flatten what we need.
    labels: list[str] = []
    internal_date = ""
    raw_header = b""
    raw_text = b""
    for item in data:
        if isinstance(item, tuple):
            head_meta, payload = item
            if not isinstance(head_meta, bytes):
                continue
            meta = head_meta.decode("utf-8", errors="replace")
            if "X-GM-LABELS" in meta and not labels:
                labels = _parse_labels(meta)
            if "INTERNALDATE" in meta and not internal_date:
                # Format: ... INTERNALDATE "25-May-2026 08:00:00 +0530" ...
                start = meta.find("INTERNALDATE")
                q1 = meta.find('"', start)
                q2 = meta.find('"', q1 + 1)
                if q1 != -1 and q2 != -1:
                    internal_date = meta[q1 + 1 : q2]
            if "HEADER" in meta:
                raw_header = payload if isinstance(payload, bytes) else b""
            if "TEXT" in meta:
                raw_text = payload if isinstance(payload, bytes) else b""

    if not raw_header:
        return None

    msg = email.message_from_bytes(raw_header + b"\r\n\r\n" + raw_text)
    from_h = _decode(msg.get("From", ""))
    subject = _decode(msg.get("Subject", ""))
    snippet = _snippet_from_msg(msg)
    return FetchedEmail(
        uid=uid.decode("utf-8", errors="replace"),
        gmail_category=category_tag,
        gmail_labels=labels,
        from_address=from_h,
        from_domain=_from_domain(from_h),
        subject=subject,
        snippet=snippet,
        received_at=internal_date,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Gmail address (IMAP username)")
    parser.add_argument(
        "--per-category", type=int, default=PER_CATEGORY, help="Emails per category"
    )
    parser.add_argument(
        "--mailbox", default='"[Gmail]/All Mail"', help="IMAP mailbox (default: All Mail)"
    )
    args = parser.parse_args()

    password = os.environ.get("IRIS_SPIKE_GMAIL_PASSWORD")
    if not password:
        print("error: set IRIS_SPIKE_GMAIL_PASSWORD env var (Gmail app password)", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"connecting to imap.gmail.com as {args.user} ...")
    conn = imaplib.IMAP4_SSL("imap.gmail.com", 993)
    try:
        conn.login(args.user, password)
        typ, _ = conn.select(args.mailbox)
        if typ != "OK":
            print(f"error: cannot select mailbox {args.mailbox}", file=sys.stderr)
            return 2

        results: list[FetchedEmail] = []
        for category_tag, gmail_query in CATEGORIES.items():
            print(f"\nsearching category={category_tag!r} via X-GM-RAW {gmail_query!r}")
            uids = _search_category(conn, gmail_query)
            if not uids:
                print(f"  no matches for {category_tag}")
                continue
            # Take the most recent N (UIDs are roughly ascending)
            picked = uids[-args.per_category :]
            print(f"  found {len(uids)} total, picking {len(picked)} most recent")
            for uid in picked:
                fetched = _fetch_one(conn, uid, category_tag)
                if fetched is None:
                    print(f"  skipped uid={uid!r} (fetch failed)")
                    continue
                results.append(fetched)
                print(f"  uid={fetched.uid} from={fetched.from_domain or '<?>'!r}")

        with OUTPUT_PATH.open("w") as f:
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        print(f"\nwrote {len(results)} emails to {OUTPUT_PATH}")
        return 0
    finally:
        try:
            conn.logout()
        except Exception:  # noqa: S110, BLE001 — best-effort cleanup
            pass


if __name__ == "__main__":
    sys.exit(main())
