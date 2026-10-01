"""Compare embedding backends for the email kNN classifier: memory, speed, agreement.

Each backend runs in its own subprocess so RSS is attributable. Outputs the
embeddings to .npy so the parent computes cross-backend agreement and a
centroid-kNN accuracy on a synthetic, labelled email corpus (no personal data).

Usage:
    python scripts/embed_bench.py --out /tmp/embed_bench --backends st,onnx,m2v8,m2v32

Backends:
    st     sentence-transformers all-MiniLM-L6-v2 (torch)      -- what triage uses today
    onnx   chromadb DefaultEmbeddingFunction, same model via onnxruntime -- already in the image
    m2v8   model2vec potion-base-8M   (static token embeddings, numpy only)
    m2v32  model2vec potion-base-32M

Env: HF_HOME (sentence-transformers cache), CHROMA_CACHE (onnx cache root), M2V_DIR
(folder holding potion-base-8M and potion-base-32M).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------- corpus ----
CATEGORIES = {
    "Finance/Bank statements": (
        [
            "Your {bank} statement for {month} is ready",
            "{bank}: e-statement available",
            "Account statement {month} {year}",
            "{bank} monthly statement notification",
        ],
        ["statements@{bankdom}", "no-reply@{bankdom}", "alerts@{bankdom}"],
        [
            "Your statement for the period ending {month} {year} is now available. Log in to view your balance and transactions.",
            "The e-statement for account ending {last4} has been generated. Total credits and debits are listed inside.",
            "View your {month} statement online. Minimum payment due date is shown on page one.",
        ],
    ),
    "Finance/Card alerts": (
        [
            "Transaction alert: {amount} at {merchant}",
            "{bank} card used for {amount}",
            "Payment of {amount} authorised",
            "Alert: purchase {amount} {merchant}",
        ],
        ["alerts@{bankdom}", "cardservices@{bankdom}", "notify@{bankdom}"],
        [
            "A transaction of {amount} was made with your card ending {last4} at {merchant}. If this was not you, call us immediately.",
            "Your card ending {last4} was charged {amount} at {merchant} on {day}. Available limit updated.",
            "Purchase approved: {amount} at {merchant}. Reply STOP to opt out of alerts.",
        ],
    ),
    "Shopping/Receipts": (
        [
            "Your {shop} order #{order} has shipped",
            "Receipt for order {order}",
            "Thanks for your {shop} purchase",
            "Order {order} confirmed",
        ],
        ["orders@{shopdom}", "receipts@{shopdom}", "noreply@{shopdom}"],
        [
            "Thank you for shopping with {shop}. Order {order} totalling {amount} will arrive by {day}. Track your package here.",
            "Here is your receipt. Items: {item}. Total {amount}. Paid with card ending {last4}.",
            "Good news, your order {order} is on its way. Estimated delivery {day}.",
        ],
    ),
    "Bills/Utilities": (
        [
            "Your {utility} bill is due",
            "{utility} invoice for {month}",
            "Payment reminder: {utility} {amount}",
            "Autopay scheduled: {utility}",
        ],
        ["billing@{utildom}", "noreply@{utildom}", "service@{utildom}"],
        [
            "Your {utility} bill of {amount} for {month} is due on {day}. Pay online to avoid late fees.",
            "This is a reminder that {amount} will be drafted from your account on {day} for your {utility} service.",
            "Your {month} usage summary is attached. Amount due: {amount}.",
        ],
    ),
    "Newsletters/Tech": (
        [
            "{newsletter}: this week in AI",
            "Issue #{order}: {topic}",
            "{newsletter} weekly digest",
            "New post: {topic}",
        ],
        ["hello@{newsdom}", "newsletter@{newsdom}", "digest@{newsdom}"],
        [
            "This week we cover {topic}, a deep dive on {topic2}, and links worth your time. Read the full issue online.",
            "Top stories: {topic}. Plus an interview about {topic2}. Unsubscribe at any time.",
            "Our latest essay on {topic} is out. Also inside: tools, jobs and {topic2}.",
        ],
    ),
    "Work/Meetings": (
        [
            "Invitation: {meeting} @ {day}",
            "Updated: {meeting}",
            "Agenda for {meeting}",
            "Reminder: {meeting} tomorrow",
        ],
        ["{person}@{workdom}", "calendar@{workdom}", "{person2}@{workdom}"],
        [
            "You have been invited to {meeting} on {day} at 10:00. Agenda: status, blockers, next steps. Join via the link.",
            "{person} moved {meeting} to {day}. Please accept the updated invite and review the attached notes.",
            "Quick reminder about {meeting} tomorrow. Bring the numbers from last sprint and the risk list.",
        ],
    ),
    "Travel/Bookings": (
        [
            "Your booking confirmation {order}",
            "{airline} itinerary for {day}",
            "Hotel reservation confirmed",
            "Check-in opens for flight {order}",
        ],
        ["reservations@{traveldom}", "noreply@{traveldom}", "bookings@{traveldom}"],
        [
            "Your trip to {city} is confirmed. Departure {day}, booking reference {order}. Check in online 24 hours before.",
            "Reservation {order} at {hotel} in {city} for {day} is confirmed. Free cancellation until 48 hours before arrival.",
            "Check-in is now open for your {airline} flight to {city}. Seats and baggage can be managed online.",
        ],
    ),
    "Social/Notifications": (
        [
            "{person} mentioned you in a comment",
            "New connection request from {person}",
            "{person} liked your post",
            "You have {order} new notifications",
        ],
        ["notifications@{socialdom}", "noreply@{socialdom}", "updates@{socialdom}"],
        [
            "{person} commented on your post about {topic}. See the conversation and reply.",
            "{person} wants to connect with you. Accept or ignore this request in the app.",
            "You have new activity: {person} and {order} others reacted to your update.",
        ],
    ),
}

FILL = {
    "bank": ["Chase", "Northwind", "Wells Fargo", "Wingtip", "Capital One", "Adatum"],
    "bankdom": [
        "chase.com",
        "northwindbank.test",
        "wellsfargo.com",
        "wingtipbank.test",
        "capitalone.com",
        "adatumbank.test",
    ],
    "month": ["January", "March", "June", "August", "September", "November"],
    "year": ["2025", "2026"],
    "last4": ["4821", "0093", "7710", "5566", "1234"],
    "amount": ["$42.17", "$1,250.00", "Rs 3,499", "$18.99", "$780.40", "$6.50"],
    "merchant": ["Costco", "Shell", "Amazon", "Trader Joe's", "Uber", "Apple.com"],
    "shop": ["Amazon", "Best Buy", "Target", "Flipkart", "Etsy"],
    "shopdom": ["amazon.com", "bestbuy.com", "target.com", "flipkart.com", "etsy.com"],
    "order": ["113-4471", "88213", "A7Q2K9", "20931", "556-902"],
    "item": [
        "USB-C cable, notebook",
        "running shoes",
        "coffee beans x2",
        "desk lamp",
        "toddler socks",
    ],
    "day": ["Monday", "Tuesday", "Oct 3", "Sep 28", "Friday", "Nov 12"],
    "utility": ["Ameren electric", "Spire gas", "Xfinity internet", "water", "AT&T mobile"],
    "utildom": ["ameren.com", "spireenergy.com", "xfinity.com", "cityofballwin.gov", "att.com"],
    "newsletter": ["The Batch", "Import AI", "TLDR", "Latent Space", "Ben's Bites"],
    "newsdom": ["deeplearning.ai", "substack.com", "tldr.tech", "latent.space", "bensbites.co"],
    "topic": [
        "agent evals",
        "KV cache quantization",
        "open-weight models",
        "harness engineering",
        "prompt injection",
    ],
    "topic2": [
        "local inference",
        "governance kernels",
        "Rust tooling",
        "GPU pricing",
        "small models",
    ],
    "meeting": [
        "Sprint review",
        "Architecture sync",
        "1:1 with manager",
        "Security roadmap",
        "Vendor call",
    ],
    "person": ["Petra", "Daniel", "Meera", "Alex", "Sam"],
    "person2": ["ops", "hr", "pm"],
    "workdom": ["acme.com", "globex.io", "initech.net"],
    "airline": ["Delta", "United", "IndiGo", "Southwest"],
    "traveldom": ["delta.com", "united.com", "goindigo.in", "booking.com", "marriott.com"],
    "city": ["Chicago", "Chennai", "Denver", "Seattle", "Austin"],
    "hotel": ["Marriott Downtown", "Taj Coromandel", "Hyatt Place", "Holiday Inn"],
    "socialdom": ["linkedin.com", "facebookmail.com", "x.com", "reddit.com"],
}


def make_corpus(n_per_cat: int = 30, seed: int = 7) -> list[tuple[str, str]]:
    rng = random.Random(seed)  # noqa: S311 - a reproducible corpus, not crypto
    rows = []
    for path, (subjects, senders, bodies) in CATEGORIES.items():
        for _ in range(n_per_cat):
            fill = {k: rng.choice(v) for k, v in FILL.items()}
            subj = rng.choice(subjects).format(**fill)
            sender = rng.choice(senders).format(**fill)
            body = rng.choice(bodies).format(**fill)
            # same shape triage embeds: subject + from + from_domain + snippet
            text = f"Subject: {subj}\nFrom: {sender}\nDomain: {sender.split('@')[-1]}\n{body}"
            rows.append((path, text))
    rng.shuffle(rows)
    return rows


# --------------------------------------------------------------- backends ----
CHILD = r"""
import json, os, sys, time, numpy as np, psutil
backend, in_path, out_path = sys.argv[1:4]
texts = json.load(open(in_path))
p = psutil.Process()
rss = lambda: p.memory_info().rss / 2**20
r0 = rss(); t0 = time.perf_counter()
if backend == "st":
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    r_imp = rss()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    enc = lambda xs: model.encode(xs, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
elif backend == "onnx":
    from chromadb.utils import embedding_functions
    r_imp = rss()
    ef = embedding_functions.DefaultEmbeddingFunction()
    def enc(xs):
        v = np.asarray(ef(xs), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
elif backend.startswith("m2v"):
    from model2vec import StaticModel
    r_imp = rss()
    name = {"m2v8": "potion-base-8M", "m2v32": "potion-base-32M"}[backend]
    model = StaticModel.from_pretrained(os.path.join(os.environ["M2V_DIR"], name))
    def enc(xs):
        v = np.asarray(model.encode(xs), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
else:
    raise SystemExit("unknown backend")
t_load = time.perf_counter() - t0
enc(texts[:1]); r_first = rss(); t_first = time.perf_counter() - t0 - t_load
t1 = time.perf_counter(); vecs = enc(texts); t_batch = time.perf_counter() - t1
t2 = time.perf_counter()
for x in texts[:40]: enc([x])
t_single = (time.perf_counter() - t2) / 40
r_end = rss()
np.save(out_path, vecs)
print(json.dumps({"backend": backend, "rss_bare_mib": r0, "rss_after_import_mib": r_imp,
    "rss_after_first_embed_mib": r_first, "rss_end_mib": r_end, "load_s": t_load,
    "first_embed_s": t_first, "batch_n": len(texts), "batch_s": t_batch,
    "single_ms": t_single * 1000, "dim": int(vecs.shape[1])}))
"""


def run_backend(python: str, backend: str, texts_path: str, out_dir: str, env: dict) -> dict:
    out_npy = os.path.join(out_dir, f"{backend}.npy")
    # This interpreter running a fixed child script; no shell, no outside input.
    proc = subprocess.run(  # noqa: S603
        [python, "-c", CHILD, backend, texts_path, out_npy],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        return {"backend": backend, "error": proc.stderr[-2000:]}
    res = json.loads(lines[-1])
    res["npy"] = out_npy
    return res


def centroid_knn(train_v, train_y, test_v, test_y):
    labels = sorted(set(train_y))
    cents = []
    for lab in labels:
        c = train_v[[i for i, y in enumerate(train_y) if y == lab]].mean(axis=0)
        cents.append(c / np.linalg.norm(c))
    cents = np.stack(cents)
    sims = test_v @ cents.T
    pred = [labels[i] for i in sims.argmax(axis=1)]
    acc = float(np.mean([p == y for p, y in zip(pred, test_y, strict=True)]))
    top1 = sims.max(axis=1)
    return pred, acc, float(np.median(top1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "embed_bench"))
    ap.add_argument("--backends", default="st,onnx,m2v8,m2v32")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--n-per-cat", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = make_corpus(args.n_per_cat)
    texts = [t for _, t in rows]
    labels = [c for c, _ in rows]
    texts_path = os.path.join(args.out, "texts.json")
    json.dump(texts, open(texts_path, "w"))
    n = len(rows)
    split = n // 2
    print(
        f"corpus: {n} synthetic emails, {len(CATEGORIES)} categories; train {split} / test {n-split}\n"
    )

    results = {}
    for b in args.backends.split(","):
        res = run_backend(args.python, b, texts_path, args.out, {})
        results[b] = res
        if "error" in res:
            print(f"{b:6s} ERROR\n{res['error']}\n")
            continue
        v = np.load(res["npy"])
        pred, acc, med = centroid_knn(v[:split], labels[:split], v[split:], labels[split:])
        res.update({"knn_acc": acc, "median_top1_sim": med, "pred": pred})
        print(
            f"{b:6s} dim={res['dim']:4d} rss: bare {res['rss_bare_mib']:.0f} -> import {res['rss_after_import_mib']:.0f} "
            f"-> first embed {res['rss_after_first_embed_mib']:.0f} -> end {res['rss_end_mib']:.0f} MiB | "
            f"load {res['load_s']:.2f}s first {res['first_embed_s']:.2f}s | batch {n}: {res['batch_s']:.2f}s | "
            f"single {res['single_ms']:.1f} ms | centroid-kNN acc {acc:.3f} (median top1 sim {med:.2f})"
        )

    ok = [b for b in results if "error" not in results[b]]
    if "st" in ok:
        ref = np.load(results["st"]["npy"])
        print("\nagreement with sentence-transformers (the current backend):")
        for b in ok:
            v = np.load(results[b]["npy"])
            same_dim = v.shape[1] == ref.shape[1]
            cos = float(np.mean(np.sum(v * ref, axis=1))) if same_dim else float("nan")
            agree = float(
                np.mean(
                    [a == c for a, c in zip(results[b]["pred"], results["st"]["pred"], strict=True)]
                )
            )
            print(
                f"  {b:6s} mean cosine(same text) {cos:.4f}   kNN prediction agreement {agree:.3f}"
            )
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(
            {b: {k: v for k, v in r.items() if k != "pred"} for b, r in results.items()},
            fh,
            indent=1,
        )


if __name__ == "__main__":
    main()
