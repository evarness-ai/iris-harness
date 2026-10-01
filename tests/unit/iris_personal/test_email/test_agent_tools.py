"""Issue 0001 — the email ReAct tools (query-aware email chat). Tools run against
a seeded temp EmailStore; no network, no LLM (llm_call=None)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.agent.agentic_core import ToolSpec
from iris_personal.email.agent_tools import build_email_tools
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore

ACCT = "gmail:user@gmail.com"


def _msg(msg_id: str, subject: str, snippet: str = "", attachments=()) -> EmailMessage:
    return EmailMessage(
        id=msg_id,
        provider="gmail",  # type: ignore[arg-type]
        account_id=ACCT,
        thread_id=None,
        from_address="News <news@example.com>",
        to=("user@gmail.com",),
        subject=subject,
        received_at=datetime.now(UTC),
        snippet=snippet,
        attachments=attachments,
    )


@pytest.fixture(autouse=True)
def _gmail_provider_mounted() -> None:
    """These tools reach the mailbox through the provider registry (M5.7 track A);
    the real Gmail provider is registered so the ``gf.*`` patches below are what runs."""
    from iris_personal.email.providers import clear_mail_providers, register_mail_provider
    from iris_personal.plugins.gmail.provider import GmailProvider

    clear_mail_providers()
    register_mail_provider(GmailProvider())
    yield  # type: ignore[misc]
    clear_mail_providers()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_msg("m1", "Latest breakthroughs in AI research", "An AI article on transformers"))
    store.upsert(_msg("m2", "Your monthly bank statement", "Account balance summary"))
    store.upsert(_msg("m3", "AI weekly newsletter", "Curated AI articles"))
    store.mark_classified("m3", category="Newsletters/AI", confidence=0.9)
    return tmp_path


def _tools(data_dir: Path) -> dict[str, ToolSpec]:
    return {t.name: t for t in build_email_tools(data_dir=data_dir, llm_call=None)}


def test_tool_set_names() -> None:
    names = {t.name for t in build_email_tools(data_dir=Path("/nonexistent"), llm_call=None)}
    assert names == {
        "inbox_digest",
        "search_inbox",
        "list_by_category",
        "read_email",
        "find_attachment",
        "analyze_inbox",
        "trash_email",  # ADR-0118 step 5: destructive, approved per call
        "trash_category",  # ADR-0118 amendment: code picks the emails, no card
        "restore_email",
        "send_email",  # ADR-0118 amendment: a write approved per call
    }


def test_tool_set_unchanged_by_semantic_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ADR-0071 slice 2 collapsed semantic into search_inbox (hybrid) — the flag never
    # adds a separate tool, so the model never chooses lexical vs semantic.
    base = {
        "inbox_digest",
        "search_inbox",
        "list_by_category",
        "read_email",
        "find_attachment",
        "analyze_inbox",
        "trash_email",  # ADR-0118 step 5: destructive, approved per call
        "trash_category",
        "restore_email",
        "send_email",  # ADR-0118 amendment: a write approved per call
    }
    monkeypatch.setenv("IRIS_EMAIL_SEMANTIC_SEARCH", "1")
    monkeypatch.delenv("IRIS_TEST_NULL_EMBEDDINGS", raising=False)
    assert {t.name for t in build_email_tools(data_dir=tmp_path, llm_call=None)} == base


# ── read_email: on-demand full-body fetch (issue 0002 item C) ────────────────


def test_read_email_summarizes_the_body(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a summariser, read_email returns a SUMMARY (tool-side, single-shot) —
    grounded on the fetched body, which is NOT dumped to the user (issue 0002 C)."""
    import iris_personal.plugins.gmail.gmail_fetch as gf

    body = "Long body: a bootcamp on building multi-agent AI systems with LangGraph and RAG."
    monkeypatch.setattr(gf, "fetch_message_body", lambda _a, _m, **_k: body)
    captured: dict[str, str] = {}

    def _summ(prompt: str) -> str:
        captured["prompt"] = prompt
        return "It's about a multi-agent AI bootcamp using LangGraph."

    tools = {
        t.name: t for t in build_email_tools(data_dir=data_dir, llm_call=None, summarize_llm=_summ)
    }
    out = tools["read_email"].call({"query": "breakthroughs"})

    assert "Summary: It's about a multi-agent AI bootcamp using LangGraph." in out
    assert "Long body:" in captured["prompt"]  # the summariser is grounded on the body
    assert "Long body:" not in out  # raw body is NOT surfaced to the user


def test_read_email_falls_back_to_body_without_summariser(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris_personal.plugins.gmail.gmail_fetch as gf

    monkeypatch.setattr(
        gf, "fetch_message_body", lambda _a, _m, **_k: "Full body: AI workflows automate tasks."
    )
    out = _tools(data_dir)["read_email"].call(
        {"query": "breakthroughs"}
    )  # no summarize_llm + llm_call=None
    assert "Full body: AI workflows automate tasks." in out
    assert "Email:" in out  # header line present


def test_read_email_graceful_on_fetch_error(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris_personal.plugins.gmail.gmail_fetch as gf

    def _boom(_a: str, _m: str, **_k: object) -> str:
        raise RuntimeError("invalid_grant: Token has been revoked")

    monkeypatch.setattr(gf, "fetch_message_body", _boom)
    out = _tools(data_dir)["read_email"].call({"query": "breakthroughs"})
    assert "re-authentication" in out.lower()


def test_read_email_supports_message_id_lookup(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris_personal.plugins.gmail.gmail_fetch as gf

    monkeypatch.setattr(gf, "fetch_message_body", lambda *args, **_kwargs: "Body via id lookup")
    out = _tools(data_dir)["read_email"].call({"message_id": "m1"})
    assert "Latest breakthroughs in AI research" in out
    assert "Body via id lookup" in out


def test_read_email_no_match(data_dir: Path) -> None:
    out = _tools(data_dir)["read_email"].call({"query": "cryptocurrency mining rig"})
    assert "No email found" in out


def test_read_email_requires_query(data_dir: Path) -> None:
    out = _tools(data_dir)["read_email"].call({})
    assert "which email" in out.lower() or "subject/sender" in out.lower()


def test_read_email_asks_user_to_pick_when_ambiguous(data_dir: Path) -> None:
    out = _tools(data_dir)["read_email"].call({"query": "AI"})
    assert "multiple emails" in out.lower()
    assert "1." in out and "2." in out
    assert "which one" in out.lower() or "reply with the number" in out.lower()


def test_read_email_pick_reads_selected_match(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris_personal.plugins.gmail.gmail_fetch as gf

    def _body(_account_id: str, message_id: str, **_kwargs: object) -> str:
        if message_id == "m3":
            return "Body for m3: Weekly AI newsletter details."
        return "Body for m1: Breakthrough AI research details."

    monkeypatch.setattr(gf, "fetch_message_body", _body)
    out1 = _tools(data_dir)["read_email"].call({"query": "AI", "pick": 1})
    out2 = _tools(data_dir)["read_email"].call({"query": "AI", "pick": 2})
    assert "Body for m1" in out1 or "Body for m3" in out1
    assert "Body for m1" in out2 or "Body for m3" in out2
    assert ("Body for m1" in out1) != ("Body for m1" in out2)


def test_read_email_records_its_shortlist_as_the_sessions_choice(
    data_dir: Path, tmp_path: Path
) -> None:
    """ADR-0106 ``choice``: the ids go on record in the order they were shown, so a
    later "1" reads the email that was on screen instead of re-running the search."""
    from iris_harness.memory.state.continuations import ContinuationStore
    from iris_harness.runtime.continuations import ContinuationRegistry

    registry = ContinuationRegistry(store=ContinuationStore(db_path=tmp_path / "cp.db"))
    tools = {
        t.name: t
        for t in build_email_tools(
            data_dir=data_dir,
            llm_call=None,
            current_session_id=lambda: "web-1",
            continuations=registry,
        )
    }

    out = tools["read_email"].call({"query": "AI"})

    pending = registry.pending("web-1")
    assert pending is not None
    assert (pending.owner, pending.kind, pending.intent) == ("email", "choice", "communication")
    assert pending.question == out
    shown = [line.split("  ")[1] for line in out.splitlines() if line[:2] in {"1.", "2."}]
    recorded = [c["subject"] for c in pending.choices]
    assert recorded == shown  # same order as the numbers the user sees
    assert {c["message_id"] for c in pending.choices} == {"m1", "m3"}


def test_read_email_records_nothing_outside_a_session(data_dir: Path, tmp_path: Path) -> None:
    from iris_harness.memory.state.continuations import ContinuationStore
    from iris_harness.runtime.continuations import ContinuationRegistry

    registry = ContinuationRegistry(store=ContinuationStore(db_path=tmp_path / "cp.db"))
    tools = {
        t.name: t
        for t in build_email_tools(
            data_dir=data_dir, llm_call=None, current_session_id=lambda: "", continuations=registry
        )
    }

    assert "multiple emails" in tools["read_email"].call({"query": "AI"}).lower()
    assert registry.history("") == ()


# ── search_inbox: the reported case ─────────────────────────────────────────


def test_search_inbox_filters_to_topic(data_dir: Path) -> None:
    out = _tools(data_dir)["search_inbox"].call({"query": "AI"})
    assert "AI research" in out  # the AI email is found
    assert "bank statement" not in out  # the unrelated one is filtered out


def test_search_results_carry_each_emails_id(data_dir: Path) -> None:
    """So the model can name exact emails to trash_email (ADR-0118 step 5)."""
    out = _tools(data_dir)["search_inbox"].call({"query": "AI"})
    research = next(line for line in out.splitlines() if "AI research" in line)
    assert research.endswith("[id m1]")


def test_search_inbox_no_query_falls_back_to_digest(data_dir: Path) -> None:
    # "search my inbox" with no term means "show my inbox" — return the digest,
    # not an error the model would echo to the user.
    out = _tools(data_dir)["search_inbox"].call({})
    assert "requires" not in out.lower()
    assert out.strip()


def test_search_inbox_rejects_hallucinated_term_falls_back_to_digest(data_dir: Path) -> None:
    # Reported bug: "how is my inbox today?" came back as "matched 'finance'".
    # The model invented "finance" — it's not in the user's turn — so the tool
    # must fall back to the digest instead of running a misleading search.
    tools = {
        t.name: t
        for t in build_email_tools(
            data_dir=data_dir, llm_call=None, current_query=lambda: "how is my inbox today?"
        )
    }
    out = tools["search_inbox"].call({"query": "finance"})
    assert "matched 'finance'" not in out
    assert "No emails matched" not in out
    assert out.strip()  # a digest, not an error


def test_search_inbox_keeps_grounded_term(data_dir: Path) -> None:
    # When the user actually named the topic, the search runs normally.
    tools = {
        t.name: t
        for t in build_email_tools(
            data_dir=data_dir, llm_call=None, current_query=lambda: "any AI emails today?"
        )
    }
    out = tools["search_inbox"].call({"query": "AI"})
    assert "AI research" in out
    assert "bank statement" not in out


def test_search_inbox_infers_today_window_from_user_turn(tmp_path: Path) -> None:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    now = datetime.now(UTC)
    store.upsert(
        EmailMessage(
            id="today1",
            provider="gmail",  # type: ignore[arg-type]
            account_id=ACCT,
            thread_id=None,
            from_address="Alerts <alerts@example.com>",
            to=("user@gmail.com",),
            subject="Finance update for today",
            received_at=now,
            snippet="Today only finance summary",
            attachments=(),
        )
    )
    store.upsert(
        EmailMessage(
            id="old1",
            provider="gmail",  # type: ignore[arg-type]
            account_id=ACCT,
            thread_id=None,
            from_address="Alerts <alerts@example.com>",
            to=("user@gmail.com",),
            subject="Finance update from earlier this week",
            received_at=now - timedelta(days=3),
            snippet="Older finance summary",
            attachments=(),
        )
    )
    tools = {
        t.name: t
        for t in build_email_tools(
            data_dir=tmp_path, llm_call=None, current_query=lambda: "any finance emails today?"
        )
    }

    out = tools["search_inbox"].call({"query": "finance"})

    assert "Finance update for today" in out
    assert "Finance update from earlier this week" not in out


def test_search_inbox_no_match_is_honest(data_dir: Path) -> None:
    out = _tools(data_dir)["search_inbox"].call({"query": "cryptocurrency"})
    assert "No emails matched" in out


def test_search_inbox_broadens_on_no_exact_match(data_dir: Path) -> None:
    # "pending payment" ANDs both terms (FTS5) and finds nothing, but the bill
    # email only says "payment". The dead-end gate broadens to an OR and surfaces
    # it, labelled honestly as a related-terms match rather than an exact one.
    store = EmailStore(db_path=data_dir / "email.db")
    store.ensure_schema()
    store.upsert(_msg("m4", "Your credit card payment is due", "Minimum payment due soon"))
    out = _tools(data_dir)["search_inbox"].call({"query": "pending payment"})
    assert "No exact match for 'pending payment'" in out
    assert "payment is due" in out


def test_search_inbox_offers_next_steps_when_truly_empty(data_dir: Path) -> None:
    # No match even after broadening — the tool offers a path forward / invites a
    # clarification (the dead-end gate) instead of a flat "no results".
    out = _tools(data_dir)["search_inbox"].call({"query": "quantum cryptography zebra"})
    assert "No emails matched" in out
    assert "?" in out  # invites a next step / clarification


def test_search_inbox_category_filter(data_dir: Path) -> None:
    # query matches m1 + m3, but the category filter narrows to the classified one.
    out = _tools(data_dir)["search_inbox"].call({"query": "AI", "category": "Newsletters"})
    assert "AI weekly newsletter" in out
    assert "AI research" not in out


def test_search_inbox_auto_reads_statement_details_with_masking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris_personal.plugins.gmail.gmail_fetch as gf

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(
        EmailMessage(
            id="woodgrove1",
            provider="gmail",  # type: ignore[arg-type]
            account_id=ACCT,
            thread_id=None,
            from_address="alerts@alerts.woodgrovebank.test",
            to=("user@gmail.com",),
            subject="E-account statement for your Woodgrove account(s).",
            received_at=datetime.now(UTC),
            snippet="Your account statement is ready.",
            attachments=(),
        )
    )

    monkeypatch.setattr(
        gf,
        "fetch_message_body",
        lambda *args, **_kwargs: (
            "Account Number: 123456789012\n"
            "Available Balance: INR 45,678.90\n"
            "Closing Balance: INR 44,000.00"
        ),
    )

    tools = {
        t.name: t
        for t in build_email_tools(
            data_dir=tmp_path,
            llm_call=None,
            summarize_llm=lambda _p: (
                "Account Number: 123456789012. "
                "Available Balance: INR 45,678.90. "
                "Closing Balance: INR 44,000.00."
            ),
        )
    }

    out = tools["search_inbox"].call(
        {"query": "Woodgrove bank statement account details and balances"}
    )

    assert "Top statement details (auto-read):" in out
    assert "Account: ••9012" in out
    assert "Available balance: INR 45,678.90" in out
    assert "Closing balance: INR 44,000.00" in out
    assert "123456789012" not in out


def test_search_inbox_prefers_institution_sender_tokens(tmp_path: Path) -> None:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    now = datetime.now(UTC)
    store.upsert(
        EmailMessage(
            id="sbm1",
            provider="gmail",  # type: ignore[arg-type]
            account_id=ACCT,
            thread_id=None,
            from_address="info@sbmbank.co.in",
            to=("user@gmail.com",),
            subject="This is a scheduled downtime notification",
            received_at=now,
            snippet="Scheduled maintenance notification",
            attachments=(),
        )
    )
    store.upsert(
        EmailMessage(
            id="woodgrove1",
            provider="gmail",  # type: ignore[arg-type]
            account_id=ACCT,
            thread_id=None,
            from_address="statements@alerts.woodgrovebank.test",
            to=("user@gmail.com",),
            subject="E-account statement for your Woodgrove account(s).",
            received_at=now - timedelta(minutes=1),
            snippet="Your statement is available.",
            attachments=(),
        )
    )

    out = _tools(tmp_path)["search_inbox"].call(
        {
            "query": "Woodgrove bank statement account details and balances",
            "prefer_sender_tokens": ["woodgrove"],
        }
    )
    assert "E-account statement for your Woodgrove account(s)." in out
    lines = [line for line in out.splitlines() if line.startswith("- ")]
    assert lines
    assert "woodgrove" in lines[0].lower()


# ── list_by_category ────────────────────────────────────────────────────────


def test_list_by_category_prefix_match(data_dir: Path) -> None:
    out = _tools(data_dir)["list_by_category"].call({"category": "Newsletters"})
    assert "AI weekly newsletter" in out  # Newsletters/AI matched by prefix


def test_list_by_category_requires_category(data_dir: Path) -> None:
    out = _tools(data_dir)["list_by_category"].call({})
    assert "requires" in out.lower()


def test_list_by_category_empty_is_honest(data_dir: Path) -> None:
    out = _tools(data_dir)["list_by_category"].call({"category": "DoesNotExist"})
    # A miss names the categories that do exist, so the model can pick one.
    assert "No category matches 'DoesNotExist'" in out
    assert "AI (1)" in out


# ── inbox_digest still works (unchanged path) ───────────────────────────────


def test_inbox_digest_returns_text(data_dir: Path) -> None:
    out = _tools(data_dir)["inbox_digest"].call({})
    assert isinstance(out, str) and out.strip()


# ── find_attachment (issue 0024) ───────────────────────────────────────────


def test_find_attachment_locates_by_filename(tmp_path: Path) -> None:
    from iris_personal.email.contracts import EmailAttachment

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    att = EmailAttachment(
        filename="passport_copy.pdf",
        mime_type="application/pdf",
        size_bytes=204800,
        attachment_id="att1",
    )
    store.upsert(_msg("p1", "VFS Global Notification", "Your application", attachments=(att,)))
    store.upsert(_msg("n1", "AI newsletter", "no attachment here"))

    out = _tools(tmp_path)["find_attachment"].call({"query": "passport"})
    assert "passport_copy.pdf" in out
    assert "from News" in out and "200 KB" in out


def test_find_attachment_none_is_honest(tmp_path: Path) -> None:
    # Store has no attachment match; with no live Gmail creds in the test, the live
    # fallback degrades gracefully — either an honest "couldn't find" or a re-auth
    # nudge, never a crash or a fake "technical issue" (issue 0024).
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_msg("n1", "AI newsletter", "no attachments"))
    out = _tools(tmp_path)["find_attachment"].call({"query": "passport"})
    assert "couldn't find any email attachment" in out or "re-authentication" in out


def test_term_hit_alphanumeric_boundary() -> None:
    from iris_personal.email.agent_tools import _term_hit

    assert _term_hit("passport", "passport_copy.pdf")  # _/. are boundaries
    assert _term_hit("passport", "Fwd: passport scan")
    assert not _term_hit("you", "Your cover is active")  # not a substring match
    assert not _term_hit("can", "scanned document")


def test_find_attachment_full_sentence_filters_filler(tmp_path: Path) -> None:
    # "Can you get my passport copy from email?" must not match a "Your ..." subject
    # via the filler token "you"; only the distinctive "passport" counts.
    from iris_personal.email.contracts import EmailAttachment

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    noise = EmailAttachment(
        filename="welcome.pdf", mime_type="application/pdf", size_bytes=1000, attachment_id="a0"
    )
    pp = EmailAttachment(
        filename="Kutty Passport.pdf",
        mime_type="application/pdf",
        size_bytes=1000,
        attachment_id="a1",
    )
    store.upsert(_msg("m_noise", "Your cover is now active", attachments=(noise,)))
    store.upsert(_msg("m_pp", "Fwd: passport scan", attachments=(pp,)))

    out = _tools(tmp_path)["find_attachment"].call(
        {"query": "Can you get my passport copy from email?"}
    )
    assert "Kutty Passport.pdf" in out
    assert "welcome.pdf" not in out  # "you"/"copy" filler didn't drag in the noise email


# ── search-result feedback / downrank (issue 0006 close-the-loop, issue 0028 spine) ──


def _msg_from(msg_id: str, subject: str, from_address: str) -> EmailMessage:
    return EmailMessage(
        id=msg_id,
        provider="gmail",  # type: ignore[arg-type]
        account_id=ACCT,
        thread_id=None,
        from_address=from_address,
        to=("user@gmail.com",),
        subject=subject,
        received_at=datetime.now(UTC),
        snippet=subject,
    )


def test_email_search_dims_extracts_sender_domain() -> None:
    from iris_harness.services.learning.suppression import email_search_dims_from_sender

    assert email_search_dims_from_sender("Promo <deals@vendor.example>") == {
        "from_domain": "vendor.example"
    }
    assert email_search_dims_from_sender("vendor.example") == {"from_domain": "vendor.example"}


def test_search_inbox_downranks_suppressed_sender(tmp_path: Path) -> None:
    from iris_harness.services.learning.suppression import NOT_USEFUL, SurfaceFeedbackStore
    from iris_personal.email.feedback_keys import email_search_dims

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    # Two emails that both match "AI", from distinct (unique) senders.
    store.upsert(_msg_from("g1", "AI research weekly", "Good <ai@goodsrc-fbtest.example>"))
    store.upsert(_msg_from("s1", "AI news roundup", "Spam <ai@spammy-fbtest.example>"))

    tools = {t.name: t for t in build_email_tools(data_dir=tmp_path, llm_call=None)}

    # Baseline: both appear.
    out = tools["search_inbox"].call({"query": "AI"})
    assert "goodsrc-fbtest.example" in out and "spammy-fbtest.example" in out

    # The user marks the spammy sender "not what I meant" → suppress it.
    fb = SurfaceFeedbackStore()
    fb.ensure_schema()
    fb.record("email", "search_result", email_search_dims("spammy-fbtest.example"), NOT_USEFUL)

    tools2 = {t.name: t for t in build_email_tools(data_dir=tmp_path, llm_call=None)}
    out2 = tools2["search_inbox"].call({"query": "AI"})
    # Both still present (downrank, not hide) but the suppressed sender ranks last.
    assert "goodsrc-fbtest.example" in out2 and "spammy-fbtest.example" in out2
    assert out2.index("goodsrc-fbtest.example") < out2.index("spammy-fbtest.example")


def test_search_inbox_shows_feedback_hint(tmp_path: Path) -> None:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_msg_from("h1", "AI weekly", "News <n@hintsrc.example>"))
    tools = {t.name: t for t in build_email_tools(data_dir=tmp_path, llm_call=None)}
    out = tools["search_inbox"].call({"query": "AI"})
    assert "ignore <sender> in search" in out


def test_display_query_redacts_emails() -> None:
    """A reflected search query must not echo email addresses back to the user
    (privacy-first; 2026-07-06 red-team finding 3). Non-email terms are kept."""
    from iris_personal.email.agent_tools import _display_query

    assert _display_query("jordan.canary@example.test") == "<email>"
    assert (
        _display_query("invoice from alice@corp.co about march")
        == "invoice from <email> about march"
    )
    assert _display_query("quarterly statement") == "quarterly statement"
