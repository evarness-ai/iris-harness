"""``iris email setup`` over IMAP, end to end on the scripted fake model (launch issue #12).

The commands are the ones the ``email`` profile puts on ``iris`` (the plugins' own CLI
registration, not a hand-built app): ``iris auth imap login`` against the in-process
IMAP server, then ``iris email setup --yes`` from connect to the approval stop, then
``--approve-writes`` to the end. Setup's model work -- the judge, the category namer,
the digest's narration -- goes through ``cli_deps``'s tier router onto the scripted fake
(``iris_harness.testing.use_fake_model``); only the embedder is a hash, as in
``test_onboarding.py``, so no ~80 MB model loads.

Pinned: nothing reaches the mailbox before the approval (no write command on the
server, no keyword on any message, no mail marked read), the approval stop exits 3 with
a pending row, and the approved run labels the judged mail with IRIS keywords.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from iris_harness.foundation.paths import default_config_dir
from iris_harness.foundation.process_state import restore_process_state, snapshot_process_state
from iris_harness.llm import fake
from iris_harness.sdk import PluginCLI, register_plugin_commands
from iris_harness.testing import use_fake_model
from iris_personal.email.providers import clear_cli_mail_providers, clear_mail_providers
from iris_personal.email.write_approvals import WriteApprovalStore
from iris_personal.plugins.email_workflows import cli_setup
from iris_personal.plugins.email_workflows.onboarding import COMPLETE, Onboarding, OnboardingDeps

from .conftest import PASSWORD, USER, build_message
from .fake_imap_server import FakeImapServer, FakeMailbox

ACCOUNT = f"imap:{USER}"
# The IMAP verbs that change a mailbox, or open it read-write (SELECT; reads EXAMINE).
# Setup may send none of them before the approval.
WRITES = {"SELECT", "STORE", "COPY", "MOVE", "EXPUNGE", "CREATE", "APPEND", "DELETE", "RENAME"}

# Every model call setup makes, scripted. No default: a call nobody expected fails.
SCRIPT: dict[str, Any] = {
    "rules": [
        {
            "name": "judge / bill",
            "match": {"system": "You sort ONE email", "user": r"Amount due: \$\d+\.\d{2}"},
            "reply": {"json": {"bucket": "bill", "confidence": 0.93, "reason": "a bill"}},
        },
        {
            "name": "judge / fyi",
            "match": {"system": "You sort ONE email"},
            "reply": {"json": {"bucket": "fyi", "confidence": 0.9, "reason": "for info"}},
        },
        {
            "name": "namer",
            "match": {
                "system": "taxonomy classifier",
                "user": r"Top sender domains: (?P<brand>[a-z]+)\.example",
            },
            "reply": {
                "json": {
                    "root": "shopping",
                    "branch": "orders",
                    "leaf": "{brand}",
                    "rationale": "one sender",
                }
            },
        },
        {
            "name": "digest narration",
            "match": {"user": "quick read on their inbox"},
            "reply": {"content": "A quiet inbox: a few bills and some order updates."},
        },
    ]
}


def _embed(texts: list[str], model: str) -> np.ndarray[Any, Any]:
    """By sender domain, so each sender is one cluster (test_onboarding.py's hash)."""
    del model
    out = []
    for text in texts:
        domain = text.split("(", 1)[1].split(")", 1)[0] if "(" in text else text[:20]
        base = np.random.default_rng(int(hashlib.sha256(domain.encode()).hexdigest()[:8], 16))
        noise = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
        vec = base.normal(size=32) + 0.05 * noise.normal(size=32)
        out.append(vec / np.linalg.norm(vec))
    return np.asarray(out, dtype=np.float32)


def _seed(mailbox: FakeMailbox) -> None:
    """60 synthetic emails from three reserved-domain senders: enough to discover."""
    now = datetime.now(UTC)
    for n in range(20):
        when = now - timedelta(days=1 + n % 5)
        mailbox.add_message(
            build_message(
                subject=f"Your order {n} shipped",
                sender="Shop <orders@shop.example>",
                text=f"Order {n} is on its way.",
                message_id=f"<order-{n}@shop.example>",
            ),
            internaldate=when,
        )
        mailbox.add_message(
            build_message(
                subject=f"Statement {n} is ready",
                sender="Power Co <billing@power.example>",
                text=f"Amount due: ${40 + n}.00\nDue date: 2026-11-{n + 1:02d}",
                message_id=f"<bill-{n}@power.example>",
            ),
            internaldate=when,
        )
        mailbox.add_message(
            build_message(
                subject=f"Weekly notes {n}",
                sender="Notes <hello@notes.example>",
                text=f"This week's notes, issue {n}.",
                message_id=f"<notes-{n}@notes.example>",
            ),
            internaldate=when,
        )


@pytest.fixture
def server() -> Iterator[tuple[FakeImapServer, FakeMailbox]]:
    mailbox = FakeMailbox(users={USER: PASSWORD})
    _seed(mailbox)
    with FakeImapServer(mailbox) as srv:
        yield srv, mailbox


@pytest.fixture
def iris(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """``iris`` as the email profile builds it, in a home of its own, on the fake model."""
    from cryptography.fernet import Fernet

    from iris_harness.kernel.governance.approvals import store as approvals_store

    home = tmp_path / "home"
    (home / "data").mkdir(parents=True)
    monkeypatch.setenv("IRIS_HOME", str(home))
    monkeypatch.setenv("IRIS_DATA_DIR", str(home / "data"))
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(home / "governance" / "audit.db"))
    monkeypatch.setenv("IRIS_VAULT_MASTER_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("IRIS_PROFILE", "email")
    monkeypatch.setenv("IRIS_EMAIL_JUDGE", "1")
    monkeypatch.setenv("IRIS_EMAIL_JUDGE_LABELS", "1")
    monkeypatch.setenv("IRIS_EMAIL_SEMANTIC_SEARCH", "0")
    # The approval queue resolves its file at import, from the suite's home.
    monkeypatch.setattr(
        approvals_store, "DEFAULT_APPROVALS_DB_PATH", home / "governance" / "approvals.db"
    )
    real_deps = cli_setup.cli_deps
    monkeypatch.setattr(
        cli_setup,
        "cli_deps",
        lambda *, interactive: replace(real_deps(interactive=interactive), embedder=_embed),
    )

    snapshot = snapshot_process_state()
    clear_mail_providers()
    clear_cli_mail_providers()
    root = typer.Typer()
    added = register_plugin_commands(
        PluginCLI(root=root), config_dir=default_config_dir(), home_dir=home
    )
    assert {"imap", "email_workflows"} <= set(added), added
    runner = CliRunner()

    def invoke(*args: str, input: str | None = None) -> Any:
        return runner.invoke(root, list(args), input=input)

    try:
        with use_fake_model(SCRIPT):
            yield invoke
    finally:
        clear_mail_providers()
        clear_cli_mail_providers()
        restore_process_state(snapshot)


def _writes(mailbox: FakeMailbox) -> list[str]:
    return [c for c in mailbox.commands if c.split()[-1] in WRITES or c in WRITES]


def _keywords(mailbox: FakeMailbox) -> dict[int, set[str]]:
    inbox = mailbox.folders["INBOX"]
    return {
        uid: {f for f in mailbox.flags_of("INBOX", uid) if f.startswith("$")}
        for uid in inbox.uids()
    }


def test_iris_email_setup_over_imap_stops_at_the_approval_then_labels(
    iris: Any, server: tuple[FakeImapServer, FakeMailbox]
) -> None:
    srv, mailbox = server
    login = iris(
        "auth",
        "imap",
        "login",
        "--user",
        USER,
        "--host",
        srv.host,
        "--port",
        str(srv.port),
        "--security",
        "plain",
        "--password-stdin",
        input=PASSWORD + "\n",
    )
    assert login.exit_code == 0, login.output
    assert mailbox.logins == [USER]

    stopped = iris("email", "setup", "--account", ACCOUNT, "--yes")
    assert stopped.exit_code == cli_setup.EXIT_WAITING, stopped.output
    assert "Label preview and approval" in stopped.output

    machine = Onboarding(OnboardingDeps())
    state = machine.state(ACCOUNT)
    assert state is not None and state.step == "label_approval"
    assert state.results["fetch"]["fetched"] == 60
    assert state.results["review_categories"]["accepted"], state.results["discover"]
    assert state.results["classify"]["judged"] > 0
    # The model work ran on the fake, every call of it scripted.
    rules = {c.rule for c in fake.transcript()}
    assert {"judge / bill", "judge / fyi", "namer"} <= rules
    # Nothing touched the mailbox: no write verb, no keyword, nothing marked read.
    assert _writes(mailbox) == []
    assert not any(_keywords(mailbox).values())
    assert all(
        "\\Seen" not in mailbox.flags_of("INBOX", u) for u in mailbox.folders["INBOX"].uids()
    )
    assert WriteApprovalStore().get(ACCOUNT) is None

    done = iris("email", "setup", "--account", ACCOUNT, "--yes", "--approve-writes")
    assert done.exit_code == 0, done.output
    assert "What IRIS just did" in done.output
    state = machine.state(ACCOUNT)
    assert state is not None and state.step == COMPLETE
    written = state.results["label_approval"]["labels_written"]
    assert written > 0 and state.results["label_approval"]["labels_failed"] == 0
    assert WriteApprovalStore().get(ACCOUNT) is not None
    labelled = {uid: kws for uid, kws in _keywords(mailbox).items() if kws}
    assert len(labelled) == written
    assert "UID STORE" in mailbox.commands  # the labels went over the wire, as keywords
    assert all(
        "\\Seen" not in mailbox.flags_of("INBOX", u) for u in mailbox.folders["INBOX"].uids()
    )
    assert "digest narration" in {c.rule for c in fake.transcript()}
