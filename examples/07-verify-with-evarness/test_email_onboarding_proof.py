"""The R14 proof for email onboarding, as a proof bundle exported and verified offline.

Part 1 runs ``iris email demo`` -- fetch, judge, the label preview, the approval of
mailbox writes for the demo's own account, the labels, the first digest -- as a child
process in a temporary home, exports its proof bundle with
``iris governance proof-bundle export`` and verifies it with ``... verify``.
Part 2 asks the email assistant a question in a governed harness and does the same from
Python (``iris_harness.testing.proof_bundle``). Part 3 tampers with a bundle three ways
and shows each one fail verification.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iris_harness.testing import harness
from iris_harness.testing.proof_bundle import (
    content_sha256,
    export_bundle,
    observations_from_session_logs,
    verify_bundle,
)


def child_env(tmp_path: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("IRIS_")}
    return env | {"HOME": str(tmp_path), "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring"}


def iris(*args: str, env: dict[str, str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- a fixed argv, our own interpreter
        [sys.executable, "-m", "iris_harness.main", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# -- 1. the onboarding run, exported and verified by the CLI ------------------------------


def test_the_onboarding_run_exports_a_bundle_that_verifies(tmp_path: Path) -> None:
    home = tmp_path / "demo"
    env = child_env(tmp_path)
    demo = iris("email", "demo", "--home", str(home), env=env, timeout=240)
    assert demo.returncode == 0, demo.stderr[-2000:]
    labelled = re.search(r"labelled (\d+) email", demo.stdout)
    assert labelled, "the demo reports the labels it wrote"

    # The labels the run wrote need no caller's word: the demo provider writes through
    # mailbox_write, so each write is a timed mailbox_write_performed row in the ledger.
    bundle_path = tmp_path / "bundle.json"
    export = iris(
        "governance", "proof-bundle", "export",
        "--db", str(home / "governance" / "audit.db"),
        "--session-logs", str(home / "logs"),
        "--subject", "email-onboarding",
        "--out", str(bundle_path),
        env=env,
    )  # fmt: skip
    assert export.returncode == 0, export.stdout + export.stderr

    verified = iris("governance", "proof-bundle", "verify", str(bundle_path), env=env)
    assert verified.returncode == 0, verified.stdout

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    calls = [row for row in bundle["ledger"] if row["hook_point"] == "pre_llm_call"]
    # The run judged real (synthetic) mail: its model calls carried personal data, and
    # every one stayed on the local tier.
    assert any(row["classification"] == "personal" for row in calls)
    assert {row["tier"] for row in calls if row["classification"] == "personal"} == {"tier_1"}
    # The approval behind the labels is in the bundle -- under a pseudonym, not the address.
    assert any(row["hook_point"] == "mailbox_writes" for row in bundle["ledger"])
    # ...and the writes it authorised, each with its time, so the order was checked.
    writes = bundle["observations"]["mailbox_writes"]
    assert writes and all(write["at"] for write in writes)
    assert sum(write["count"] for write in writes) >= int(labelled.group(1))
    assert "sam.rivera" not in bundle_path.read_text(encoding="utf-8")


# -- 2. a chat turn, exported and verified from Python ------------------------------------

QUESTION = "Which emails need a reply?"
SCRIPT: dict[str, Any] = {
    "rules": [
        {
            "name": "route to email",
            "match": {"system": "request router"},
            "reply": {"json": {"intent": "communication"}},
        },
        {
            "name": "answer from the search",
            "match": {"user": r"(?s)Observation:"},
            "reply": {"content": "Thought: Nothing found.\nFinal Answer: No email needs a reply."},
        },
        {
            "name": "search the inbox",
            "match": {"user": r"User: Which emails need a reply"},
            "reply": {
                "content": "Thought: Search.\nAction: search_inbox\n"
                'Action Input: {"query": "reply"}'
            },
        },
    ]
}


@pytest.fixture(scope="module")
def turn_bundle() -> Iterator[dict[str, Any]]:
    """The bundle of one email chat turn: its ledger rows, and the model calls and the
    answer its session log recorded."""
    with harness(profile="email", fake_model=SCRIPT) as h:
        assert h.plugin_loaded("email_workflows")
        result = h.chat_stream(QUESTION)
        assert result.answered, result.error
        assert result.agent == "email"
        observed = observations_from_session_logs(h.home / "logs")
        assert len(observed.model_calls) == len(h.model_calls()) > 0
        assert observed.answers
        yield export_bundle(h.audit_db, observations=observed, subject="email-chat-turn")


def test_an_email_turn_exports_a_bundle_that_verifies(turn_bundle: dict[str, Any]) -> None:
    assert verify_bundle(turn_bundle) == []


# -- 3. the checks bite -------------------------------------------------------------------


def _resealed(bundle: dict[str, Any]) -> dict[str, Any]:
    """A tampered bundle whose digest was recomputed: only the invariants can catch it."""
    bundle["content_sha256"] = content_sha256(bundle)
    return bundle


def test_a_tampered_bundle_fails_verification(turn_bundle: dict[str, Any]) -> None:
    # A private model call that went to the cloud tier.
    leak = copy.deepcopy(turn_bundle)
    call = next(row for row in leak["ledger"] if row["hook_point"] == "pre_llm_call")
    call |= {"classification": "personal", "tier": "tier_3", "decision": "allow"}
    assert [v.invariant for v in verify_bundle(_resealed(leak))] == ["no-private-to-cloud"]

    # A model call the ledger never saw.
    unaudited = copy.deepcopy(turn_bundle)
    unaudited["observations"]["model_calls"].append({"session_id": None, "tier": "tier3"})
    assert [v.invariant for v in verify_bundle(_resealed(unaudited))] == [
        "every-call-and-answer-audited"
    ]

    # A mailbox write with no approval row behind it (the turn approved none).
    written = copy.deepcopy(turn_bundle)
    written["observations"]["mailbox_writes"].append(
        {"account_ref": "acct-" + "0" * 24, "count": 3, "at": None}
    )
    assert [v.invariant for v in verify_bundle(_resealed(written))] == ["mailbox-write-approved"]

    # An edit that does not recompute the digest.
    edited = copy.deepcopy(turn_bundle)
    edited["ledger"].pop()
    assert "integrity" in [v.invariant for v in verify_bundle(edited)]
