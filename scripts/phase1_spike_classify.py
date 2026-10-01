"""Phase 1 spike — classify fetched emails via local llama-server, score
against Gmail's category labels.

Throwaway diagnostic. Delete after Phase 1 lands.

Reads ``data/spike/phase1_emails.jsonl`` (output of phase1_spike_imap_fetch.py),
classifies each row via the local llama-server's OpenAI-compatible
``/v1/chat/completions`` endpoint, writes:

  - ``data/spike/phase1_classifier_trace.jsonl``  (per-email trace)
  - ``docs/architecture/spikes/phase1_email_classification.md``  (summary)

Two passes by default: zero-shot then few-shot. Use ``--single-pass`` to
skip the second pass.

Acceptance bar (spike #2 per canonical doc §5):
  >= 85% agreement with Gmail labels.

Usage:
    poetry run python scripts/phase1_spike_classify.py
    poetry run python scripts/phase1_spike_classify.py --endpoint http://127.0.0.1:8090
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "spike" / "phase1_emails.jsonl"
LABELED_INPUT_PATH = REPO_ROOT / "data" / "spike" / "phase1_emails_labeled.jsonl"
TRACE_PATH = REPO_ROOT / "data" / "spike" / "phase1_classifier_trace.jsonl"
REPORT_PATH = REPO_ROOT / "docs" / "architecture" / "spikes" / "phase1_email_classification.md"

# Two category sets — one for the original Gmail-label run, one for the re-spike
# against user-defined categories. Selected by --ground-truth-field at runtime.
GMAIL_CATEGORIES = ("personal", "social", "promotions", "updates", "forums")
USER_CATEGORIES = ("finance", "marketing", "learning", "news-digest", "social", "personal")
ACCEPTANCE_BAR = 0.85

ZERO_SHOT_TEMPLATE_GMAIL = """You are an email classifier. Given the sender domain, subject, and a brief snippet, choose EXACTLY ONE category:

- personal: Mail from individuals you know personally (friends, family, colleagues)
- social: Notifications from social networks (LinkedIn, Twitter, Facebook, etc.)
- promotions: Marketing, offers, newsletters, ads
- updates: Confirmations, receipts, statements, system notices, transactional mail
- forums: Mailing lists, discussion groups, community messages

Respond ONLY with strict JSON of the form: {{"category": "<one>", "confidence": <0.0-1.0>}}

Email:
  from_domain: {from_domain}
  subject: {subject}
  snippet: {snippet}

JSON:"""

ZERO_SHOT_TEMPLATE_USER = """You are an email classifier. Given the sender domain, subject, and a brief snippet, choose EXACTLY ONE category:

- finance: Banks, insurance, investment, statements, transaction notifications
- marketing: Shopping promotions, sales, "last chance" offers, advertising emails
- learning: School communications, online courses, workshops, GitHub code notifications
- news-digest: Newsletter digests (Economic Times, Medium, general newsletters)
- social: LinkedIn / Facebook / community notifications, friend requests, comments, group posts
- personal: Real conversations with humans — family, friends, work threads, replies

Respond ONLY with strict JSON of the form: {{"category": "<one>", "confidence": <0.0-1.0>}}

Email:
  from_domain: {from_domain}
  subject: {subject}
  snippet: {snippet}

JSON:"""

FEW_SHOT_EXAMPLES_GMAIL = """Examples:

Email: from_domain=linkedin.com, subject="New connection request from John", snippet="Hi, I'd like to add you to my LinkedIn network."
JSON: {{"category": "social", "confidence": 0.95}}

Email: from_domain=chase.com, subject="Your January statement is ready", snippet="Your statement period ending Jan 31 is now available."
JSON: {{"category": "updates", "confidence": 0.92}}

Email: from_domain=spotify.com, subject="50% off Premium for 3 months", snippet="Upgrade your music experience with this limited offer."
JSON: {{"category": "promotions", "confidence": 0.97}}

Email: from_domain=gmail.com, subject="Re: dinner saturday", snippet="Yeah let's do 7pm at the new place on King St."
JSON: {{"category": "personal", "confidence": 0.94}}

Email: from_domain=python.org, subject="[python-dev] PEP 695 review", snippet="See the latest revision; please reply to thread by Friday."
JSON: {{"category": "forums", "confidence": 0.91}}

"""

FEW_SHOT_EXAMPLES_USER = """Examples:

Email: from_domain=northwindbank.test, subject="Your Fixed Deposit advice", snippet="Your FD will renew on 2026-06-15..."
JSON: {{"category": "finance", "confidence": 0.96}}

Email: from_domain=email.gapfactory.com, subject="GAP Sale ends Sunday — 50% off", snippet="Don't miss the biggest sale of the year..."
JSON: {{"category": "marketing", "confidence": 0.97}}

Email: from_domain=communityschool.example, subject="Grade 2 - May 2nd class updates", snippet="This week's homework + upcoming summer schedule..."
JSON: {{"category": "learning", "confidence": 0.93}}

Email: from_domain=economictimesnews.com, subject="Daily news digest — May 25", snippet="Top headlines: markets close higher; RBI hints..."
JSON: {{"category": "news-digest", "confidence": 0.95}}

Email: from_domain=linkedin.com, subject="John Smith wants to connect", snippet="Hi, I'd like to add you to my LinkedIn network."
JSON: {{"category": "social", "confidence": 0.95}}

Email: from_domain=gmail.com, subject="Re: dinner saturday", snippet="Yeah let's do 7pm at the new place on King St."
JSON: {{"category": "personal", "confidence": 0.94}}

"""


def _build_template(zero_shot: str, examples: str) -> str:
    return (
        examples
        + "Now classify this email:\n\nEmail:\n  from_domain: {from_domain}\n  subject: {subject}\n  snippet: {snippet}\n\nJSON:"
    )


@dataclass(frozen=True)
class Classification:
    uid: str
    pass_label: str  # 'zero-shot' or 'few-shot'
    raw_response: str
    predicted: str | None
    confidence: float | None
    ground_truth: str
    correct: bool


def _build_prompt(template: str, row: dict) -> str:
    return template.format(
        from_domain=row.get("from_domain") or "<unknown>",
        subject=(row.get("subject") or "")[:200],
        snippet=(row.get("snippet") or "")[:400],
    )


def _call_llm(endpoint: str, prompt: str, *, timeout: int = 120) -> tuple[str, float]:
    """POST to /v1/chat/completions; return (content, elapsed_s)."""
    body = json.dumps(
        {
            "model": "qwen3-30b-a3b",  # ignored by llama-server but required by OpenAI schema
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 80,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 — local-only spike endpoint
        f"{endpoint}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8", errors="replace")
    elapsed = time.monotonic() - t0
    parsed = json.loads(raw)
    content = parsed["choices"][0]["message"]["content"]
    return content, elapsed


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_decision(raw: str, allowed: tuple[str, ...]) -> tuple[str | None, float | None]:
    """Parse the JSON object the model emitted; return (category, confidence).

    Categories outside the ``allowed`` set are rejected (None).
    """
    match = _JSON_RE.search(raw)
    if not match:
        return None, None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None
    cat = obj.get("category")
    if isinstance(cat, str):
        cat = cat.strip().lower()
        if cat not in allowed:
            cat = None
    else:
        cat = None
    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)):
        conf = None
    return cat, conf


def _run_pass(
    pass_label: str,
    template: str,
    rows: list[dict],
    endpoint: str,
    ground_truth_field: str,
    allowed: tuple[str, ...],
) -> list[Classification]:
    print(f"\n=== {pass_label} pass ===")
    classifications: list[Classification] = []
    for i, row in enumerate(rows, 1):
        prompt = _build_prompt(template, row)
        try:
            content, elapsed = _call_llm(endpoint, prompt)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  [{i}/{len(rows)}] uid={row.get('uid')} → REQUEST FAILED: {e}")
            classifications.append(
                Classification(
                    uid=row.get("uid", ""),
                    pass_label=pass_label,
                    raw_response=f"<error: {e}>",
                    predicted=None,
                    confidence=None,
                    ground_truth=row.get(ground_truth_field, ""),
                    correct=False,
                )
            )
            continue
        predicted, confidence = _extract_decision(content, allowed)
        ground_truth = row.get(ground_truth_field, "")
        correct = predicted == ground_truth
        marker = "✓" if correct else "✗"
        print(
            f"  [{i}/{len(rows)}] {marker} uid={row.get('uid')} truth={ground_truth!r} "
            f"pred={predicted!r} conf={confidence} ({elapsed:.1f}s)"
        )
        classifications.append(
            Classification(
                uid=row.get("uid", ""),
                pass_label=pass_label,
                raw_response=content,
                predicted=predicted,
                confidence=confidence,
                ground_truth=ground_truth,
                correct=correct,
            )
        )
    return classifications


def _summarize(classifications: list[Classification], label: str, allowed: tuple[str, ...]) -> dict:
    total = len(classifications)
    correct = sum(1 for c in classifications if c.correct)
    accuracy = correct / total if total else 0.0

    per_cat_total: Counter[str] = Counter()
    per_cat_correct: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for c in classifications:
        per_cat_total[c.ground_truth] += 1
        if c.correct:
            per_cat_correct[c.ground_truth] += 1
        confusion[c.ground_truth][c.predicted or "<none>"] += 1

    return {
        "label": label,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_cat_total": dict(per_cat_total),
        "per_cat_correct": dict(per_cat_correct),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "categories": list(allowed),
    }


def _render_report(
    summary_zero: dict,
    summary_few: dict | None,
    classifications_zero: list[Classification],
    classifications_few: list[Classification] | None,
) -> str:
    def fmt_pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    def decision(acc: float) -> str:
        if acc >= ACCEPTANCE_BAR:
            return "**PASS** — meets the ≥85% acceptance bar."
        if acc >= 0.75:
            return "**MARGINAL** — between 75-85%. Phase 1 viable with stronger few-shot or cloud-Tier-3 escalation for unconfident cases."
        return "**FAIL** — below 75%. Rethink Phase 1 classification design before producer skills ship."

    def render_pass(s: dict, cls: list[Classification]) -> str:
        active_cats = s["categories"]
        lines = [f"## {s['label']} pass\n"]
        lines.append(
            f"Overall accuracy: **{s['correct']}/{s['total']} ({fmt_pct(s['accuracy'])})**\n"
        )
        lines.append(decision(s["accuracy"]) + "\n")
        lines.append("### Per-category breakdown\n")
        lines.append("| Category | Sample N | Correct | Recall |")
        lines.append("|---|---:|---:|---:|")
        for cat in active_cats:
            n = s["per_cat_total"].get(cat, 0)
            k = s["per_cat_correct"].get(cat, 0)
            r = (k / n) if n else 0.0
            lines.append(f"| {cat} | {n} | {k} | {fmt_pct(r)} |")
        lines.append("")
        lines.append("### Confusion matrix (rows=ground truth, cols=predicted)\n")
        cols = list(active_cats) + ["<none>"]
        lines.append("|  | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for cat in active_cats:
            row = s["confusion"].get(cat, {})
            counts = [str(row.get(c, 0)) for c in cols]
            lines.append(f"| **{cat}** | " + " | ".join(counts) + " |")
        lines.append("")
        wrong = [c for c in cls if not c.correct]
        if wrong:
            lines.append("### Worst disagreements (up to 5)\n")
            for c in wrong[:5]:
                lines.append(
                    f"- uid `{c.uid}` — truth `{c.ground_truth}`, model said `{c.predicted}` "
                    f"(conf={c.confidence}). Raw response:\n  ```\n  {c.raw_response.strip()[:400]}\n  ```"
                )
            lines.append("")
        return "\n".join(lines)

    cats = summary_zero["categories"]
    ground_truth_kind = "user-labeled categories" if "finance" in cats else "Gmail category labels"
    total_n = summary_zero["total"]
    out = ["# Phase 1 Spike — Email Classification (Tier-3-local)\n"]
    out.append("Date: 2026-05-25  ")
    out.append("Model: qwen3-30b-a3b (Q4_K_M, ~18GB), served via llama-server  ")
    out.append(f"Sample: {total_n} emails  ")
    out.append(f"Ground truth: {ground_truth_kind}  ")
    out.append(f"Categories: `{', '.join(cats)}`  ")
    out.append(f"Acceptance bar: ≥{int(ACCEPTANCE_BAR * 100)}% accuracy  ")
    out.append("")
    out.append(render_pass(summary_zero, classifications_zero))
    if summary_few is not None and classifications_few is not None:
        out.append(render_pass(summary_few, classifications_few))
        delta = summary_few["accuracy"] - summary_zero["accuracy"]
        out.append("## Marginal value of few-shot examples\n")
        out.append(f"Δaccuracy (few-shot − zero-shot): **{delta * 100:+.1f}pp**\n")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8090")
    parser.add_argument("--single-pass", action="store_true", help="zero-shot only")
    parser.add_argument(
        "--ground-truth-field",
        default="gmail_category",
        choices=("gmail_category", "user_category"),
        help="Which field on each row holds the ground-truth label. "
        "gmail_category = original Gmail labels (from data/spike/phase1_emails.jsonl); "
        "user_category = user-labeled categories (from data/spike/phase1_emails_labeled.jsonl).",
    )
    args = parser.parse_args()

    if args.ground_truth_field == "user_category":
        input_path = LABELED_INPUT_PATH
        allowed = USER_CATEGORIES
        zero_shot_template = ZERO_SHOT_TEMPLATE_USER
        few_shot_template = _build_template(ZERO_SHOT_TEMPLATE_USER, FEW_SHOT_EXAMPLES_USER)
    else:
        input_path = DEFAULT_INPUT_PATH
        allowed = GMAIL_CATEGORIES
        zero_shot_template = ZERO_SHOT_TEMPLATE_GMAIL
        few_shot_template = _build_template(ZERO_SHOT_TEMPLATE_GMAIL, FEW_SHOT_EXAMPLES_GMAIL)

    if not input_path.is_file():
        print(f"error: missing input {input_path}", file=sys.stderr)
        if args.ground_truth_field == "user_category":
            print("       run scripts/phase1_spike_label.py first to label emails", file=sys.stderr)
        else:
            print("       run scripts/phase1_spike_imap_fetch.py first", file=sys.stderr)
        return 1

    with input_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        print("error: input file empty", file=sys.stderr)
        return 2

    # Filter out rows where the user explicitly skipped labeling.
    if args.ground_truth_field == "user_category":
        before = len(rows)
        rows = [r for r in rows if r.get("user_category")]
        if before != len(rows):
            print(
                f"excluding {before - len(rows)} rows with no user label (skipped during labeling)"
            )

    print(f"loaded {len(rows)} emails from {input_path}")
    print(f"ground truth field: {args.ground_truth_field}")
    print(f"categories: {allowed}")
    print(f"llama-server endpoint: {args.endpoint}")

    zero = _run_pass(
        "zero-shot", zero_shot_template, rows, args.endpoint, args.ground_truth_field, allowed
    )
    few: list[Classification] | None = None
    if not args.single_pass:
        few = _run_pass(
            "few-shot", few_shot_template, rows, args.endpoint, args.ground_truth_field, allowed
        )

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_PATH.open("w") as f:
        for c in zero:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
        if few:
            for c in few:
                f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    print(f"\nwrote trace to {TRACE_PATH}")

    summary_zero = _summarize(zero, "Zero-shot", allowed)
    summary_few = _summarize(few, "Few-shot", allowed) if few else None
    report = _render_report(summary_zero, summary_few, zero, few)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"wrote report to {REPORT_PATH}")

    print(
        f"\nzero-shot accuracy: {summary_zero['correct']}/{summary_zero['total']} ({summary_zero['accuracy'] * 100:.1f}%)"
    )
    if summary_few:
        print(
            f"few-shot  accuracy: {summary_few['correct']}/{summary_few['total']} ({summary_few['accuracy'] * 100:.1f}%)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
