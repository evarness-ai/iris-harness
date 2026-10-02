"""``iris email demo`` end to end on the scripted fake model (OSS plan L1 exit criterion).

Pinned:

* the corpus is exactly what its seeded generator writes, 200 synthetic emails over
  invented people and reserved domains, spread over every judge bucket;
* the demo's fake-model script and the corpus agree: every email the judge sees gets
  the bucket the generator meant;
* one run, with the network refused: fetch -> judge -> labels -> first digest, a digest
  of the expected shape, and -- the R14 invariant -- an audit row for every model call,
  the digest's narrated answer included, all on the local fake;
* the labels walk email setup's step 6 for the demo's own account: the preview, an
  audited approval scoped to that account alone, then the labels written -- and
  nothing outside the demo home is touched;
* a second run fetches, judges and approves nothing and shows the same digest;
* the demo refuses to approve writes outside a demo home;
* the run's environment is the demo home's alone: the owner's IRIS settings and
  credentials are dropped, and the vault master key is a throwaway in the home;
* the command itself, as a user types it, in a child process.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
from collections.abc import Iterator
from email.utils import parseaddr
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iris_harness.llm import fake
from iris_harness.sdk.audit import AuditLog, audit_db_path
from iris_harness.testing import no_network
from iris_personal.email.providers import clear_mail_providers
from iris_personal.email.write_approvals import AUDIT_HOOK, AUDIT_PLUGIN, WriteApprovalStore
from iris_personal.plugins.email_workflows.demo import corpus as corpus_module
from iris_personal.plugins.email_workflows.demo.home import (
    FAKE_SCRIPT,
    KEY_FILENAME,
    MARKER,
    DemoHomeError,
    demo_environment,
    prepare_home,
)
from iris_personal.plugins.email_workflows.demo.provider import (
    DEMO_ACCOUNT,
    DemoMailProvider,
    render_body,
)
from iris_personal.plugins.email_workflows.demo.run import (
    DemoEnvironmentError,
    approve_demo_writes,
    render_label_preview,
    render_summary,
    run_demo,
)

SRC = Path(corpus_module.__file__).resolve().parents[4]

# What the judge is meant to say for each kind the generator writes (promo, social and
# sent mail never reach it: their tab or label releases them unjudged).
EXPECTED_BUCKET = {
    "bill": "bill",
    "event": "event",
    "needs_reply": "needs_reply",
    "automated_ask": "needs_reply",  # the model's call; judge.yaml's person-only rule -> fyi
    "fyi": "fyi",
    "unsure": "unsure",
}
UNJUDGED = {"promo", "social", "sent"}


@pytest.fixture(autouse=True)
def _clean_registries() -> Iterator[None]:
    clear_mail_providers()
    fake.install_script(None)
    fake.reset_transcript()
    yield
    clear_mail_providers()
    fake.reset_transcript()


def _files_under(*roots: str | None) -> dict[str, tuple[int, int]]:
    """Every file under ``roots`` with its size and mtime (what "touched" would change)."""
    out: dict[str, tuple[int, int]] = {}
    for root in roots:
        if not root or not Path(root).exists():
            continue
        for path in Path(root).rglob("*"):
            if path.is_file():
                st = path.stat()
                out[str(path)] = (st.st_size, st.st_mtime_ns)
    return out


def _approval_rows() -> list[Any]:
    return [
        r
        for r in AuditLog(db_path=audit_db_path()).query()
        if r.plugin == AUDIT_PLUGIN and r.hook_point == AUDIT_HOOK
    ]


def _enter_demo_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """This process runs as ``iris email demo``'s child would: the demo environment, the
    demo home as its working directory."""
    home = prepare_home(tmp_path / "demo-home")
    for name in list(os.environ):
        if name.startswith("IRIS_"):
            monkeypatch.delenv(name)
    for name, value in demo_environment(home, {}).items():
        monkeypatch.setenv(name, value)
    # The kernel's approval queue resolves its file once, at import (from IRIS_HOME).
    # The child process `iris email demo` starts imports with the demo home already set;
    # this process imported it under the suite's home, so point it where the child's is.
    from iris_harness.kernel.governance.approvals import store as approvals_store

    monkeypatch.setattr(
        approvals_store, "DEFAULT_APPROVALS_DB_PATH", home / "governance" / "approvals.db"
    )
    monkeypatch.chdir(home)
    return home


# -- the corpus -----------------------------------------------------------------------


def test_the_checked_in_corpus_is_what_the_generator_writes() -> None:
    assert corpus_module.CORPUS_PATH.read_text(encoding="utf-8") == corpus_module.render(
        corpus_module.generate()
    ), "corpus.json is stale: run python -m iris_personal.plugins.email_workflows.demo.corpus"


def test_the_corpus_spreads_over_every_bucket() -> None:
    rows = corpus_module.load_corpus()
    assert len(rows) == 200
    assert corpus_module.kind_counts(rows) == {
        "automated_ask": 3,
        "bill": 16,
        "event": 12,
        "fyi": 66,
        "needs_reply": 22,
        "promo": 58,
        "sent": 5,
        "social": 13,
        "unsure": 5,
    }
    assert sum(1 for r in rows if r["attachments"]) >= 15
    threads = [t for t in {r["thread_id"] for r in rows} if not t.startswith("t-demo-")]
    assert len(threads) == 4  # dinner, rollout, birthday, boiler


def test_the_corpus_is_synthetic() -> None:
    """Invented people on example.* and invented companies on the reserved .test TLD."""
    for row in corpus_module.load_corpus():
        for raw in [row["from"], *row["to"], *row["cc"]]:
            domain = parseaddr(raw)[1].rsplit("@", 1)[-1]
            assert re.fullmatch(r"example\.(com|org|net)|[a-z0-9-]+\.test", domain), raw


def test_every_body_date_token_renders() -> None:
    from datetime import date

    for row in corpus_module.load_corpus():
        body = render_body(row["body"], date(2026, 9, 30))
        assert "{" not in body, (row["id"], body)


def test_the_script_and_the_corpus_agree() -> None:
    """Every email the judge sees gets the bucket the generator meant (the judge's own
    prompt and email message, through the script the demo runs on)."""
    from datetime import UTC, date, datetime

    from iris_harness.llm.fake import Request, Script
    from iris_personal.plugins.email_workflows.judge import build_prompt
    from iris_personal.plugins.email_workflows.judge_config import JudgeConfig

    config = JudgeConfig.load(None)
    script = Script.load(FAKE_SCRIPT)
    system = build_prompt(config, now=datetime(2026, 9, 30, tzinfo=UTC), tz=UTC)
    for row in corpus_module.load_corpus():
        if row["kind"] in UNJUDGED:
            continue
        body = render_body(row["body"], date(2026, 9, 30))
        user = f"From: {row['from']}\nSubject: {row['subject']}\n\n{body}"
        _rule, reply, _groups = script.answer(
            Request(system=system, user=user, prompt=user, model="m", tools=(), json_schema={})
        )
        assert reply.json is not None
        bucket = reply.json["bucket"]
        if float(reply.json["confidence"]) < config.unsure_below:
            bucket = "unsure"
        assert bucket == EXPECTED_BUCKET[row["kind"]], (row["id"], row["subject"], bucket)


# -- the provider ---------------------------------------------------------------------


def test_the_demo_mailbox_is_read_only(tmp_path: Path) -> None:
    provider = DemoMailProvider(state_path=tmp_path / "state.json")
    with pytest.raises(PermissionError):
        provider.trash_messages("demo:x@example.com", ["demo-0001"])
    with pytest.raises(PermissionError):
        provider.restore_messages("demo:x@example.com", ["demo-0001"])
    assert not hasattr(provider, "send_message")


def test_fetch_new_reports_progress_once(tmp_path: Path) -> None:
    from iris_personal.email.store import EmailStore

    provider = DemoMailProvider(state_path=tmp_path / "state.json")
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    calls: list[tuple[float, str]] = []

    result = provider.fetch_new(
        "demo:x@example.com",
        store=store,
        max_messages=5,
        progress=lambda f, m: calls.append((f, m)),
    )

    assert len(calls) == 1
    assert calls[0][0] == 1.0
    assert calls[0][1] == f"fetched {result.fetched}"


def test_labels_wait_for_the_mailbox_write_approval(tmp_path: Path) -> None:
    """R4: the demo mailbox is where the approval step is tried, so it keeps the rule."""
    approvals = WriteApprovalStore(db_path=tmp_path / "approvals.db")
    provider = DemoMailProvider(state_path=tmp_path / "state.json", write_approvals=approvals)
    ids = provider.ensure_labels("demo:x@example.com", ["IRIS/Bill"])
    with pytest.raises(PermissionError, match="no approval"):
        provider.modify_labels("demo:x@example.com", ["demo-0001"], [ids["IRIS/Bill"]], [])
    state = json.loads((tmp_path / "state.json").read_text())
    assert "labels" not in state


def test_labels_stay_in_the_demo_state(tmp_path: Path) -> None:
    approvals = WriteApprovalStore(db_path=tmp_path / "approvals.db")
    approvals.approve("demo:x@example.com", "test")
    provider = DemoMailProvider(state_path=tmp_path / "state.json", write_approvals=approvals)
    ids = provider.ensure_labels("demo:x@example.com", ["IRIS/Bill", "IRIS/FYI"])
    provider.modify_labels("demo:x@example.com", ["demo-0001"], [ids["IRIS/Bill"]], [])
    provider.modify_labels(
        "demo:x@example.com", ["demo-0001"], [ids["IRIS/FYI"]], [ids["IRIS/Bill"]]
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["labels"] == {"demo-0001": ["IRIS/FYI"]}


def test_demo_label_writes_leave_a_ledger_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R14: the demo's label writes are observed like any provider's; a refused write and
    a write that changes no message record nothing."""
    from iris_harness.sdk.audit import AuditLog
    from iris_personal.email.write_approvals import WRITE_HOOK

    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "audit.db"))
    approvals = WriteApprovalStore(db_path=tmp_path / "approvals.db")
    provider = DemoMailProvider(state_path=tmp_path / "state.json", write_approvals=approvals)
    ids = provider.ensure_labels("demo:x@example.com", ["IRIS/Bill"])
    with pytest.raises(PermissionError):
        provider.modify_labels("demo:x@example.com", ["demo-0001"], [ids["IRIS/Bill"]], [])
    approvals.approve("demo:x@example.com", "test")
    provider.modify_labels("demo:x@example.com", ["demo-0001", "demo-0002"], [ids["IRIS/Bill"]], [])
    provider.modify_labels("demo:x@example.com", [], [ids["IRIS/Bill"]], [])

    rows = [
        r for r in AuditLog(db_path=tmp_path / "audit.db").query() if r.hook_point == WRITE_HOOK
    ]
    assert [json.loads(r.payload_json) for r in rows] == [
        {"account": "demo:x@example.com", "op": "label", "count": 2}
    ]
    assert rows[0].reason == "mailbox write: label x2 on a demo account"


# -- one run, end to end --------------------------------------------------------------


def _llm_rows(since: str | None = None) -> list[Any]:
    return [
        r
        for r in AuditLog(db_path=audit_db_path()).query(since=since)
        if r.hook_point == "pre_llm_call"
    ]


def test_the_demo_runs_end_to_end_offline_and_audits_every_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The suite's own (non-demo) home and data dir: the demo must leave them alone.
    outside = (os.environ.get("IRIS_HOME"), os.environ.get("IRIS_DATA_DIR"))
    before = _files_under(*outside)
    home = _enter_demo_home(tmp_path, monkeypatch)
    with no_network() as attempted:
        report = run_demo()
    assert attempted == []
    assert report.network_attempts == 0

    # fetch -> judge: 200 fetched, the tabs and sent mail released unjudged.
    assert (report.fetched, report.stored) == (200, 200)
    assert report.released_unjudged == 58 + 13 + 5
    assert report.judged == 124
    assert report.buckets == {"Bill": 16, "Event": 12, "FYI": 69, "Needs reply": 22, "Unsure": 5}
    assert report.errors == 0

    # The first digest: narrated inbox, Needs reply, bills with figures, what's coming up.
    digest = report.digest
    assert "new today across 1 account (200 total stored)" in digest
    assert "are in the demo mailbox" in digest  # the narration, on the governed tier
    # 22 needs-reply verdicts; the owner already answered three in their threads.
    assert "## Needs reply (19)" in digest
    assert "Petra Raman: Re: Dinner next Saturday?" in digest
    assert re.search(r"## Bills due \(11\)\n- .+: \$\d+\.\d{2} due \d{4}-\d{2}-\d{2}", digest)
    assert "5 more bill email(s) need nothing" in digest
    assert "## Coming up (12)" in digest

    # R14: every model call has its audit rows -- one governed run per call, each on
    # the fake (declared runs: local, so tier_1). The narrated digest is one of them.
    calls = fake.transcript()
    assert len(calls) == report.judged + 1
    assert [c.rule for c in calls].count("digest narration") == 1
    rows = _llm_rows()
    assert len({r.run_id for r in rows}) == len(calls) == report.llm_calls
    assert all('"provider": "fake"' in r.payload_json for r in rows)
    assert {r.tier for r in rows} == {"tier_1"}
    assert report.providers == ("fake",)

    # Step 6 for the demo's own account: the preview setup shows, then the approval.
    preview = render_label_preview(report)
    assert "IRIS/Bill: 16 email(s) (e.g. " in preview
    assert report.labels_previewed > 0
    assert report.labels_written == report.labels_previewed
    assert (report.labels_failed, report.label_error) == (0, "")
    labelled = json.loads((home / "data" / "demo_mailbox.json").read_text())["labels"]
    assert sum(1 for names in labelled.values() if names) == report.labels_written
    # The approval: one audit row (allow, actor demo, this account), then the grant
    # naming it -- the #757/#758 pattern -- for the demo account and no other.
    (row,) = _approval_rows()
    assert (row.decision, row.agent_type) == ("allow", "email_demo")
    payload = json.loads(row.payload_json)
    assert (payload["account"], payload["actor"]) == (DEMO_ACCOUNT, "demo")
    grant = WriteApprovalStore().get(DEMO_ACCOUNT)
    assert grant is not None and grant.approval_ref == f"iris email demo [audit #{row.id}]"
    assert report.writes_approved_now and report.writes_approval_ref == grant.approval_ref
    with sqlite3.connect(WriteApprovalStore().db_path) as conn:
        approved = [r[0] for r in conn.execute("SELECT account_id FROM mailbox_write_approvals")]
    assert approved == [DEMO_ACCOUNT]

    # Everything landed in the demo home; nothing outside it was touched.
    assert audit_db_path() == home / "governance" / "audit.db"
    assert WriteApprovalStore().db_path == home / "data" / "mailbox_write_approvals.db"
    assert (home / "data" / "email.db").is_file()
    assert _files_under(*outside) == before
    summary = render_summary(report)
    assert "Model calls: 125" in summary and "Network connections attempted: 0" in summary
    assert f"labelled {report.labels_written} email(s) in the demo mailbox" in summary
    assert "iris email writes approve --account" in summary  # what a real mailbox needs


def test_a_second_run_fetches_and_judges_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enter_demo_home(tmp_path, monkeypatch)
    first = run_demo()
    clear_mail_providers()
    fake.reset_transcript()
    second = run_demo()
    assert (second.fetched, second.judged, second.stored) == (0, 0, 200)
    assert second.sections["needs_reply"] == first.sections["needs_reply"]
    assert second.sections["bills"] == first.sections["bills"]
    assert [c.rule for c in fake.transcript()] == ["digest narration"]
    assert "Nothing new in the demo mailbox" in render_summary(second)
    # Approved once: the re-run reuses it and records no second approval row.
    assert first.writes_approved_now and not second.writes_approved_now
    assert second.writes_approval_ref == first.writes_approval_ref
    assert (second.labels_previewed, second.labels_written) == (0, 0)
    assert len(_approval_rows()) == 1
    assert "used the demo's approval from its first run" in render_summary(second)


def test_the_demo_approves_writes_only_inside_its_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_personal.plugins.email_workflows.demo.run import _setup_machine

    home = _enter_demo_home(tmp_path, monkeypatch)
    machine = _setup_machine()
    # Not the demo home's environment (IRIS_HOME is someone else's).
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "owner-home"))
    with pytest.raises(DemoEnvironmentError):
        approve_demo_writes(machine, home)
    monkeypatch.setenv("IRIS_HOME", str(home))
    # A home without the demo's marker is not a demo home.
    (home / MARKER).unlink()
    with pytest.raises(DemoEnvironmentError):
        approve_demo_writes(machine, home)
    (home / MARKER).write_text("iris email demo\n")
    # A different directory than the process's demo home.
    with pytest.raises(DemoEnvironmentError):
        approve_demo_writes(machine, tmp_path / "elsewhere")
    assert WriteApprovalStore().get(DEMO_ACCOUNT) is None
    assert _approval_rows() == []


def test_the_run_refuses_outside_a_demo_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_DEMO_HOME", raising=False)
    with pytest.raises(DemoEnvironmentError):
        run_demo()


# -- the home and its environment -----------------------------------------------------


def test_the_environment_is_the_demo_homes_alone(tmp_path: Path) -> None:
    home = prepare_home(tmp_path / "h")
    owner = {
        "PATH": "/usr/bin",
        "HOME": "/Users/someone",
        "IRIS_TZ": "Europe/Lisbon",
        "IRIS_DATA_DIR": "/Users/someone/iris-data",
        "IRIS_GOVERNANCE_AUDIT_DB_PATH": "/Users/someone/audit.db",
        "IRIS_CONFIG_DIR": "/Users/someone/iris-config",
        "IRIS_EMAIL_JUDGE_LABELS": "1",
        "IRIS_VAULT_MASTER_KEY": "the-owners-key",
        "GITHUB_TOKEN": "x",
        "ANTHROPIC_API_KEY": "x",
        "PYTHONPATH": "src",
    }
    env = demo_environment(home, owner)
    assert env["IRIS_HOME"] == str(home)
    assert env["IRIS_DATA_DIR"] == str(home / "data")
    assert env["IRIS_GOVERNANCE_AUDIT_DB_PATH"] == str(home / "governance" / "audit.db")
    assert "IRIS_CONFIG_DIR" not in env  # the shipped config, not the owner's
    assert env["IRIS_TZ"] == "Europe/Lisbon"
    assert env["IRIS_EMAIL_JUDGE_LABELS"] == "1"  # the demo walks step 6, then labels
    assert env["IRIS_LLM_PROVIDER"] == "fake"
    assert "GITHUB_TOKEN" not in env and "ANTHROPIC_API_KEY" not in env
    assert env["IRIS_VAULT_MASTER_KEY"] != "the-owners-key"
    assert Path(env["PYTHONPATH"]).is_absolute()
    key_file = home / KEY_FILENAME
    assert key_file.read_text().strip() == env["IRIS_VAULT_MASTER_KEY"]
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert demo_environment(home, owner)["IRIS_VAULT_MASTER_KEY"] == env["IRIS_VAULT_MASTER_KEY"]


def test_the_home_is_never_a_directory_the_demo_did_not_make(tmp_path: Path) -> None:
    other = tmp_path / "someones-files"
    other.mkdir()
    (other / "notes.txt").write_text("keep me")
    with pytest.raises(DemoHomeError):
        prepare_home(other, reset=True)
    assert (other / "notes.txt").read_text() == "keep me"


def test_reset_starts_the_demo_home_over(tmp_path: Path) -> None:
    home = prepare_home(tmp_path / "h")
    (home / "data" / "email.db").write_text("old")
    prepare_home(home, reset=True)
    assert not (home / "data" / "email.db").exists()


# -- the command, as typed ------------------------------------------------------------


def test_iris_email_demo_prints_the_digest_and_what_iris_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import typer

    from iris_personal.plugins.email_workflows.cli import cmd_email_demo

    # The child imports this checkout's code, not whatever else is on the machine.
    monkeypatch.setenv("PYTHONPATH", str(SRC))
    app = typer.Typer()
    app.command()(cmd_email_demo)
    home = tmp_path / "demo"
    result = CliRunner().invoke(app, ["--home", str(home)])
    assert result.exit_code == 0, result.output
    assert (home / "governance" / "audit.db").is_file()
    assert (home / KEY_FILENAME).is_file()
