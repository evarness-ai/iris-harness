"""The email judge (loop-proof PR 5): what it reads, what it records, when it stops."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from iris_harness.sdk.llm import JsonReply, LLMBadReply, LLMUnreachable
from iris_harness.services.heartbeat.models import HeartbeatStatus
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.events import EMAIL_NEW_ARRIVED, EmailNewArrivedPayload
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows import judge_wiring
from iris_personal.plugins.email_workflows.judge import (
    admit_swept_mail,
    build_prompt,
    build_schema,
    judge_one,
    run_judge,
    skip_reason,
)
from iris_personal.plugins.email_workflows.judge_config import (
    EMAIL_JUDGED,
    EmailJudgedPayload,
    JudgeConfig,
    judge_max_per_run,
)
from iris_personal.plugins.email_workflows.judgments import JUDGED, PROMO, WAITING, JudgmentStore

ACCOUNT = "gmail:owner@example.com"
NOW = datetime(2026, 9, 27, 3, 0, tzinfo=UTC)  # still Saturday Sep 26 in Chicago


@pytest.fixture(autouse=True)
def _judge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "IRIS_EMAIL_JUDGE",
        "IRIS_EMAIL_JUDGE_LABELS",
        "IRIS_EMAIL_JUDGE_MAX",
        "IRIS_EMAIL_JUDGE_UNSURE_BELOW",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IRIS_TZ", "America/Chicago")


@pytest.fixture
def config() -> JudgeConfig:
    return JudgeConfig.load()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "email.db"


@pytest.fixture
def emails(db: Path) -> EmailStore:
    s = EmailStore(db_path=db)
    s.ensure_schema()
    return s


@pytest.fixture
def store(db: Path, emails: EmailStore) -> JudgmentStore:
    s = JudgmentStore(db_path=db)
    s.ensure_schema()
    return s


def _email(
    emails: EmailStore,
    mid: str,
    *,
    sender: str = "person@example.com",
    subject: str = "Hello",
    labels: tuple[str, ...] = ("INBOX",),
    snippet: str = "stored snippet",
    minutes: int = 0,
    category: str | None = None,
) -> EmailMessage:
    message = EmailMessage(
        id=mid,
        provider="gmail",
        account_id=ACCOUNT,
        thread_id=f"t-{mid}",
        from_address=sender,
        subject=subject,
        snippet=snippet,
        labels=labels,
        received_at=datetime(2026, 9, 26, 12, tzinfo=UTC) + timedelta(minutes=minutes),
    )
    emails.upsert(message)
    if category:
        emails.mark_classified(mid, category=category, confidence=0.9)
        got = emails.get(mid)
        assert got is not None
        return got
    return message


@dataclass
class FakeLLM:
    """Answers per call; an Exception in the list is raised instead."""

    answers: list[Any]
    calls: list[tuple[str, str]]

    def __call__(self, system: str, user: str, schema: Mapping[str, Any]) -> JsonReply:
        self.calls.append((system, user))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return JsonReply(data=dict(answer), latency_ms=42, model="judge:4b")


def _llm(*answers: Any) -> FakeLLM:
    return FakeLLM(answers=list(answers), calls=[])


def _bill(conf: float = 0.93) -> dict[str, Any]:
    return {
        "bucket": "bill",
        "confidence": conf,
        "reason": "statement ready",
        "min_due": 35.0,
        "statement_balance": 812.4,
        "credit_balance": None,
        "due_date": "2026-10-14",
        "event_start": None,
        "reply_by": None,
    }


def _body(text: str = "the body") -> Callable[[EmailMessage, int], str]:
    return lambda message, max_chars: text


# -- the call ------------------------------------------------------------------------


def test_schema_enum_is_the_yaml_buckets(config: JudgeConfig) -> None:
    schema = build_schema(config)
    assert schema["properties"]["bucket"]["enum"] == list(config.keys)
    assert set(schema["required"]) >= {"bucket", "confidence", "min_due", "reply_by"}


def test_prompt_has_todays_date_in_the_owners_zone(config: JudgeConfig) -> None:
    prompt = build_prompt(config, now=NOW, tz=ZoneInfo("America/Chicago"))
    assert "Today is Saturday 2026-09-26 (America/Chicago)" in prompt
    for bucket in config.buckets:
        assert bucket.definition in prompt


def test_prompt_lists_the_next_seven_days_by_weekday(config: JudgeConfig) -> None:
    # "by Thursday" was dated a Friday when the model did the weekday arithmetic itself.
    prompt = build_prompt(config, now=NOW, tz=ZoneInfo("America/Chicago"))
    assert "Thursday 2026-10-01" in prompt and "Saturday 2026-10-03" in prompt
    assert "Saturday 2026-09-26," not in prompt.split("next days are:")[1].split(".")[0]


def test_an_automated_sender_never_needs_a_reply(config: JudgeConfig, emails: EmailStore) -> None:
    # Measured on 100 real emails: the model called "verify your account" and "we value
    # your feedback" needs_reply; only a real person's email can be.
    system = _email(emails, "a1", sender="Shop <no-reply@shop.example>", subject="Verify")
    person = _email(emails, "a2", sender="Petra <petra@example.com>", subject="dinner?")
    asks = {"bucket": "needs_reply", "confidence": 0.95, "reason": "asks"}
    v = judge_one(system, config=config, llm=_llm(asks), body="b", now=NOW)
    assert (v.bucket, v.fields["guess"], v.fields["rule"]) == (
        "fyi",
        "needs_reply",
        "automated sender",
    )
    assert (
        judge_one(person, config=config, llm=_llm(asks), body="b", now=NOW).bucket == "needs_reply"
    )
    # other buckets are not person-only
    bill = judge_one(system, config=config, llm=_llm(_bill()), body="b", now=NOW)
    assert bill.bucket == "bill"


def test_automated_sender_words_come_from_the_yaml(config: JudgeConfig) -> None:
    from iris_personal.plugins.email_workflows.judge import is_automated_sender

    assert config.person_only == ("needs_reply",) and config.person_only_else == "fyi"
    for address in ("alerts@bank.example", "Team <updates@x.example>", "feedback@m.example"):
        assert is_automated_sender(address, config)
    assert not is_automated_sender("Katie <kate@school.example>", config)


def test_own_and_junk_mail_is_not_judged(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    for i, label in enumerate(("SENT", "DRAFT", "SPAM", "TRASH")):
        message = _email(emails, f"s{i}", labels=(label,))
        assert skip_reason(message, config, store) == f"tab {label}"


def test_judge_one_reads_the_zone_from_iris_tz(config: JudgeConfig, emails: EmailStore) -> None:
    llm = _llm(_bill())
    judge_one(_email(emails, "m1"), config=config, llm=llm, body="b", now=NOW)
    assert "Saturday 2026-09-26" in llm.calls[0][0]


def test_judge_one_records_bucket_and_figures(config: JudgeConfig, emails: EmailStore) -> None:
    message = _email(emails, "m1", sender="bank@example.com", subject="Your statement")
    llm = _llm(_bill())
    verdict = judge_one(message, config=config, llm=llm, body="Minimum due $35", now=NOW)

    assert verdict.bucket == "bill"
    assert verdict.confidence == pytest.approx(0.93)
    assert verdict.fields == {
        "min_due": 35.0,
        "statement_balance": 812.4,
        "due_date": "2026-10-14",
        "read": "body",
    }
    assert (verdict.model, verdict.latency_ms) == ("judge:4b", 42)
    user = llm.calls[0][1]
    assert "bank@example.com" in user and "Your statement" in user and "Minimum due $35" in user


def test_low_confidence_is_unsure_and_keeps_the_guess(
    config: JudgeConfig, emails: EmailStore
) -> None:
    verdict = judge_one(
        _email(emails, "m1"), config=config, llm=_llm(_bill(conf=0.55)), body="b", now=NOW
    )
    assert verdict.bucket == "unsure"
    assert verdict.fields["guess"] == "bill"


def test_threshold_comes_from_the_setting(
    config: JudgeConfig, emails: EmailStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_EMAIL_JUDGE_UNSURE_BELOW", "0.5")
    verdict = judge_one(
        _email(emails, "m1"), config=config, llm=_llm(_bill(conf=0.55)), body="b", now=NOW
    )
    assert verdict.bucket == "bill"


def test_a_bucket_the_yaml_does_not_know_is_unsure(config: JudgeConfig, emails: EmailStore) -> None:
    verdict = judge_one(
        _email(emails, "m1"),
        config=config,
        llm=_llm({"bucket": "spam", "confidence": 0.99}),
        body="b",
        now=NOW,
    )
    assert (verdict.bucket, verdict.fields["guess"]) == ("unsure", "spam")


def test_hints_carry_the_owners_correction_for_this_sender(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _email(emails, "old", sender="agent@example.com", subject="About your inquiry")
    store.record("old", ACCOUNT, bucket="fyi", confidence=0.8)
    store.correct("old", "needs_reply", source="card")
    _email(emails, "new", sender="agent@example.com", subject="Following up", minutes=5)
    store.mark_waiting(ACCOUNT, ["new"])
    llm = _llm(_bill())

    run_judge(store, emails, config, llm=llm, fetch_body=_body(), now=NOW)

    system = llm.calls[0][0]
    assert 'The owner filed mail from this sender "About your inquiry" as needs_reply.' in system


def test_triage_category_is_passed_as_a_hint(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _email(emails, "m1", category="finance/banking")
    store.mark_waiting(ACCOUNT, ["m1"])
    llm = _llm(_bill())
    run_judge(store, emails, config, llm=llm, fetch_body=_body(), now=NOW)
    assert '"finance/banking"' in llm.calls[0][0]


# -- a run ---------------------------------------------------------------------------


def _queue(emails: EmailStore, store: JudgmentStore, n: int, start: int = 0) -> list[str]:
    ids = [f"m{i}" for i in range(start, start + n)]
    for i, mid in enumerate(ids):
        _email(emails, mid, minutes=start + i)
        store.mark_waiting(ACCOUNT, [mid])
    return ids


def test_a_run_records_rows_and_emits_email_judged(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _queue(emails, store, 1)
    events: list[tuple[str, Any]] = []
    report = run_judge(
        store,
        emails,
        config,
        llm=_llm(_bill()),
        fetch_body=_body(),
        emit=lambda t, p: events.append((t, p)),
        now=NOW,
    )
    row = store.get("m0")
    assert row is not None
    assert (row.status, row.bucket, row.model, row.tier, row.latency_ms) == (
        JUDGED,
        "bill",
        "judge:4b",
        "email_judge",
        42,
    )
    assert row.run_id == report.run_id and row.fields["min_due"] == 35.0
    assert events == [(EMAIL_JUDGED, EmailJudgedPayload("m0", ACCOUNT, "bill", 0.93))]
    assert (report.judged, report.counts, report.waiting) == (1, {"bill": 1}, 0)


def test_run_judge_reports_progress_per_candidate(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _queue(emails, store, 3)
    calls: list[tuple[float, str]] = []

    run_judge(
        store,
        emails,
        config,
        llm=_llm(_bill()),
        fetch_body=_body(),
        now=NOW,
        progress=lambda f, m: calls.append((f, m)),
    )

    assert calls == [
        (1 / 3, "judging 1/3"),
        (2 / 3, "judging 2/3"),
        (1.0, "judging 3/3"),
    ]


def test_over_the_cap_waits_and_waiting_drains_first_next_run(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_EMAIL_JUDGE_MAX", "3")
    _queue(emails, store, 5)
    first = run_judge(store, emails, config, llm=_llm(_bill()), fetch_body=_body(), now=NOW)
    assert (first.judged, first.waiting) == (3, 2)
    assert [j.message_id for j in store.waiting()] == ["m3", "m4"]

    _queue(emails, store, 2, start=10)  # newer mail lands before the next run
    llm = _llm(_bill())
    second = run_judge(store, emails, config, llm=llm, fetch_body=_body(), now=NOW)
    assert [i.message_id for i in second.items] == ["m3", "m4", "m10"]
    assert [j.message_id for j in store.waiting()] == ["m11"]


def test_unreachable_stops_the_run_and_leaves_the_rest_waiting(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _queue(emails, store, 4)
    llm = _llm(_bill(), LLMUnreachable("connection refused"), _bill())
    report = run_judge(store, emails, config, llm=llm, fetch_body=_body(), now=NOW)

    assert report.unreachable and report.judged == 1
    assert len(llm.calls) == 2  # stopped at once: no retry, no other model
    assert [j.message_id for j in store.waiting()] == ["m1", "m2", "m3"]
    assert report.waiting == 3


def test_one_failure_is_unsure_with_the_error_and_the_run_goes_on(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _queue(emails, store, 3)
    llm = _llm(_bill(), LLMBadReply("no JSON"), _bill())
    report = run_judge(store, emails, config, llm=llm, fetch_body=_body(), now=NOW)

    bad = store.get("m1")
    assert bad is not None and bad.bucket == "unsure" and "no JSON" in bad.error
    assert [store.get(m).bucket for m in ("m0", "m2")] == ["bill", "bill"]  # type: ignore[union-attr]
    assert (report.judged, report.errors, report.unsure) == (3, 1, 1)


def test_a_failed_body_read_falls_back_to_the_snippet(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _queue(emails, store, 1)

    def broken(message: EmailMessage, max_chars: int) -> str:
        raise RuntimeError("token expired")

    llm = _llm(_bill())
    run_judge(store, emails, config, llm=llm, fetch_body=broken, now=NOW)
    assert "stored snippet" in llm.calls[0][1]
    assert store.get("m0").fields["read"] == "snippet"  # type: ignore[union-attr]


def test_the_body_is_capped_at_body_chars(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _queue(emails, store, 1)
    asked: list[int] = []

    def fetch(message: EmailMessage, max_chars: int) -> str:
        asked.append(max_chars)
        return "x" * 10_000

    llm = _llm(_bill())
    run_judge(store, emails, config, llm=llm, fetch_body=fetch, now=NOW)
    assert asked == [config.body_chars]
    assert "x" * (config.body_chars + 1) not in llm.calls[0][1]


def test_skip_tabs_and_promo_senders_are_released_not_queued(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _email(emails, "promo-tab", labels=("INBOX", "CATEGORY_PROMOTIONS"))
    _email(emails, "social", labels=("CATEGORY_SOCIAL",))
    _email(emails, "updates", labels=("INBOX", "CATEGORY_UPDATES"))
    _email(emails, "was-promo", sender="shop@example.com")
    store.record("was-promo", ACCOUNT, bucket="fyi", confidence=0.9)
    store.correct("was-promo", PROMO, source="chat")
    _email(emails, "shop-again", sender="shop@example.com", minutes=3)

    admitted = admit_swept_mail(
        store, emails, config, ACCOUNT, ["promo-tab", "social", "updates", "shop-again"], now=NOW
    )
    assert admitted.waiting == ("updates",)
    assert admitted.released == ("promo-tab", "social", "shop-again")
    assert [j.message_id for j in store.waiting()] == ["updates"]
    assert store.get("promo-tab") is None and store.get("shop-again") is None


def test_old_mail_from_a_cold_start_is_released_not_hidden(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    assert config.queue_hours == 72
    _email(emails, "recent")
    _email(emails, "last-month", minutes=-60 * 24 * 20)
    admitted = admit_swept_mail(store, emails, config, ACCOUNT, ["recent", "last-month"], now=NOW)
    assert (admitted.waiting, admitted.released) == (("recent",), ("last-month",))


def test_already_judged_mail_is_released_again_not_requeued(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _email(emails, "m1")
    store.record("m1", ACCOUNT, bucket="fyi", confidence=0.9)
    admitted = admit_swept_mail(store, emails, config, ACCOUNT, ["m1"], now=NOW)
    assert admitted.released == ("m1",) and store.count_waiting() == 0


def test_a_sender_marked_promo_after_queueing_is_closed_not_judged(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _email(emails, "queued", sender="shop@example.com")
    store.mark_waiting(ACCOUNT, ["queued"])
    _email(emails, "marked", sender="shop@example.com", minutes=1)
    store.record("marked", ACCOUNT, bucket="fyi", confidence=0.9)
    store.correct("marked", PROMO, source="chat")
    llm = _llm(_bill())

    report = run_judge(store, emails, config, llm=llm, fetch_body=_body(), now=NOW)
    assert llm.calls == [] and report.skipped == 1
    row = store.get("queued")
    assert row is not None and row.bucket == PROMO and store.count_waiting() == 0
    assert report.released == {ACCOUNT: ["queued"]}


def test_the_default_cap_is_100(monkeypatch: pytest.MonkeyPatch) -> None:
    assert judge_max_per_run() == 100
    monkeypatch.setenv("IRIS_EMAIL_JUDGE_MAX", "7")
    assert judge_max_per_run() == 7


def test_dry_run_writes_nothing(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _queue(emails, store, 1)
    _email(emails, "fresh", minutes=30)
    events: list[Any] = []
    report = run_judge(
        store,
        emails,
        config,
        llm=_llm(_bill()),
        fetch_body=_body(),
        emit=lambda t, p: events.append(t),
        backfill=5,
        dry_run=True,
        now=NOW,
    )
    assert [i.message_id for i in report.items] == ["m0", "fresh"]
    assert events == [] and store.get("fresh") is None and report.released == {}
    assert store.get("m0").status == WAITING  # type: ignore[union-attr]


def test_backfill_takes_the_newest_unjudged_mail_and_never_re_releases_it(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _queue(emails, store, 1)
    _email(emails, "old", minutes=10)
    _email(emails, "newest", minutes=20)
    _email(emails, "promo", minutes=30, labels=("CATEGORY_PROMOTIONS",))
    report = run_judge(
        store, emails, config, llm=_llm(_bill()), fetch_body=_body(), backfill=5, limit=2, now=NOW
    )
    assert [i.message_id for i in report.items] == ["m0", "newest"]
    assert store.get("newest").status == JUDGED  # type: ignore[union-attr]
    assert report.released == {ACCOUNT: ["m0"]}  # "newest" was never hidden


def test_no_local_tier_judges_nothing(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    _queue(emails, store, 1)
    report = run_judge(store, emails, config, llm=None, fetch_body=_body(), now=NOW)
    assert report.no_model and report.waiting == 1


def test_a_cloud_tier_is_refused() -> None:
    from iris_harness.sdk.llm import CodingLLMConfig
    from iris_personal.plugins.email_workflows.judge import make_tier_llm

    config = CodingLLMConfig(provider="lmstudio", model="m", tier_name="email_judge")
    with pytest.raises(ValueError, match="ollama"):
        make_tier_llm(config)


# -- the judge's model call is the governed client's (issue #31, owner decision B) --------

_TIERS = Path(__file__).resolve().parents[5] / "config" / "llm_tiers.yaml"


def _router(tiers: Path = _TIERS) -> Any:
    from iris_harness.llm.tier_router import TierRouter

    return TierRouter.load_from_yaml(tiers, settings=None)


def test_the_router_hands_the_judge_its_pinned_tier_governed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iris_personal.plugins.email_workflows import judge

    built: list[tuple[Any, str]] = []

    class _Client:
        def __init__(self, config: Any, *, governance_agent_type: str) -> None:
            built.append((config, governance_agent_type))

        def invoke_json(self, *, system_prompt: str, user_prompt: str, schema: Any) -> JsonReply:
            return JsonReply(data={"bucket": "bill"}, latency_ms=1, model="judge:4b")

    monkeypatch.setattr(judge, "CodingLLMClient", _Client)
    llm = judge.llm_from_router(_router())
    assert llm is not None
    assert llm("sys", "usr", {"type": "object"}).data == {"bucket": "bill"}
    [(config, agent_type)] = built
    # The finance sweep's precedent: one governed call per email, agent type "chat".
    assert agent_type == "chat"
    assert (config.tier_name, config.provider) == ("email_judge", "ollama")


def test_an_intent_the_router_does_not_resolve_to_the_tier_judges_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iris_harness.sdk.llm import CodingLLMConfig
    from iris_personal.plugins.email_workflows import judge

    router = _router()
    fallback = CodingLLMConfig(provider="ollama", model="other", tier_name="fallback")
    monkeypatch.setattr(router, "get_llm_config", lambda _intent: fallback)
    assert judge.llm_from_router(router) is None


def test_the_unreachable_stop_holds_through_the_governed_client(
    config: JudgeConfig, emails: EmailStore, store: JudgmentStore
) -> None:
    """A down Mac on the real client (a connection error inside invoke_turn) still
    stops the run with every unjudged email kept waiting."""
    import httpx

    from iris_harness.llm.arbiter import OllamaCircuitBreaker
    from iris_harness.sdk.llm import CodingLLMClient, CodingLLMConfig
    from iris_personal.plugins.email_workflows.judge import make_tier_llm

    class _Down:
        def invoke(self, *_a: Any, **_k: Any) -> Any:
            raise httpx.ConnectError("refused")

    client = CodingLLMClient(
        CodingLLMConfig(provider="ollama", model="judge:4b", tier_name="email_judge"),
        model_factory=lambda **_k: _Down(),
        governance_handled_upstream=True,
        circuit_breaker=OllamaCircuitBreaker(),
    )
    llm = make_tier_llm(client.config, llm_client=client)
    _queue(emails, store, 3)
    report = run_judge(store, emails, config, llm=llm, fetch_body=_body(), now=NOW)
    assert report.unreachable and report.judged == 0 and report.waiting == 3


# -- wiring: queue on sweep, release on judge --------------------------------------------


def _new_arrived(events: list[tuple[str, Any]]) -> list[tuple[str, tuple[str, ...]]]:
    return [(p.account_id, p.new_message_ids) for t, p in events if t == EMAIL_NEW_ARRIVED]


def test_the_queue_releases_skipped_mail_and_hides_the_rest(
    db: Path, emails: EmailStore, store: JudgmentStore
) -> None:
    _email(emails, "promo-tab", labels=("CATEGORY_PROMOTIONS",))
    _email(emails, "person")
    events: list[tuple[str, Any]] = []
    handler = judge_wiring.build_queue_handler(
        db_path=db, emit=lambda t, p: events.append((t, p)), clock=lambda: NOW
    )
    handler(EmailNewArrivedPayload(ACCOUNT, ("promo-tab", "person"), 2, False))

    assert _new_arrived(events) == [(ACCOUNT, ("promo-tab",))]
    assert [j.message_id for j in store.waiting()] == ["person"]


def test_the_queue_emits_nothing_when_everything_waits(
    db: Path, emails: EmailStore, store: JudgmentStore
) -> None:
    _email(emails, "person")
    events: list[tuple[str, Any]] = []
    handler = judge_wiring.build_queue_handler(
        db_path=db, emit=lambda t, p: events.append((t, p)), clock=lambda: NOW
    )
    handler(EmailNewArrivedPayload(ACCOUNT, ("person",), 1, False))
    assert events == []


def test_judge_off_judges_nothing_but_new_mail_still_waits_hidden(
    config: JudgeConfig,
    db: Path,
    emails: EmailStore,
    store: JudgmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_EMAIL_JUDGE", "0")
    _email(emails, "m1")
    events: list[tuple[str, Any]] = []
    handler = judge_wiring.build_queue_handler(
        db_path=db, emit=lambda t, p: events.append((t, p)), clock=lambda: NOW
    )
    handler(EmailNewArrivedPayload(ACCOUNT, ("m1",), 1, False))
    llm = _llm(_bill())

    report, _ = judge_wiring.judge_and_release(
        llm=llm, db_path=db, emit=lambda t, p: events.append((t, p)), fetch_body=_body()
    )
    assert not report.enabled and llm.calls == [] and events == []
    row = store.get("m1")
    assert row is not None and row.status == WAITING and report.waiting == 1


def test_the_judge_job_releases_judged_mail_per_account_only(
    db: Path, emails: EmailStore, store: JudgmentStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_EMAIL_JUDGE_LABELS", "0")
    _queue(emails, store, 3)
    other = "gmail:other@example.com"
    emails.upsert(_email(emails, "x1", minutes=9).model_copy(update={"account_id": other}))
    store.mark_waiting(other, ["x1"])
    events: list[tuple[str, Any]] = []
    # m0 judged, m1 judged, then the Mac goes away: m2 and x1 stay waiting, unreleased.
    llm = _llm(_bill(), _bill(), LLMUnreachable("refused"))

    report, note = judge_wiring.judge_and_release(
        llm=llm, db_path=db, emit=lambda t, p: events.append((t, p)), fetch_body=_body(), now=NOW
    )
    assert _new_arrived(events) == [(ACCOUNT, ("m0", "m1"))]
    assert [t for t, _ in events].count(EMAIL_JUDGED) == 2
    assert {j.message_id for j in store.waiting()} == {"m2", "x1"}
    assert report.unreachable and "released 2" in note


def test_the_judge_job_runs_the_label_step_only_when_labels_are_on(
    db: Path, emails: EmailStore, store: JudgmentStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _queue(emails, store, 1)
    synced: list[Any] = []

    def fake_sync(s: JudgmentStore, c: JudgeConfig, providers: Mapping[str, Any]) -> str:
        synced.append(s.get("m0").status)  # type: ignore[union-attr]
        return "1 labelled"

    from iris_personal.plugins.email_workflows import judge_labels

    monkeypatch.setattr(judge_labels, "sync_labels", fake_sync)
    _, note = judge_wiring.judge_and_release(
        llm=_llm(_bill()), db_path=db, emit=lambda t, p: None, fetch_body=_body()
    )
    assert synced == [JUDGED]
    assert note.startswith("judge: judged 1") and note.endswith("labels: 1 labelled")

    monkeypatch.setenv("IRIS_EMAIL_JUDGE_LABELS", "0")
    synced.clear()
    _queue(emails, store, 1, start=5)
    judge_wiring.judge_and_release(
        llm=_llm(_bill()), db_path=db, emit=lambda t, p: None, fetch_body=_body()
    )
    assert synced == []


def test_a_judge_failure_is_a_failed_run_not_a_crash() -> None:
    def boom() -> Any:
        raise RuntimeError("db locked")

    run = judge_wiring.EmailJudgeJob(judge=boom)(None)
    assert run.status is HeartbeatStatus.FAILED and "db locked" in run.error


def test_an_unreachable_mac_with_nothing_judged_is_a_skipped_run() -> None:
    from iris_personal.plugins.email_workflows.judge import JudgeRunReport

    report = JudgeRunReport(run_id="r", unreachable=True, unreachable_error="refused", waiting=4)
    run = judge_wiring.EmailJudgeJob(judge=lambda: (report, "judge: ..."))(None)
    assert run.status is HeartbeatStatus.SKIPPED and run.error == "refused"
    # the structured result the health check (job_watch.judge_reachable) reads
    assert run.result == {
        "judged": 0,
        "waiting": 4,
        "oldest_waiting": None,
        "unsure": 0,
        "errors": 0,
        "unreachable": True,
    }


def test_the_judge_run_records_when_the_oldest_waiting_email_was_queued(
    db: Path, emails: EmailStore, store: JudgmentStore, config: JudgeConfig
) -> None:
    store.mark_waiting(ACCOUNT, ["w1", "w2"])
    report, _ = judge_wiring.judge_and_release(db_path=db, llm=None, emit=lambda *_: None)
    oldest = store.waiting(limit=1)[0].created_at
    assert report.oldest_waiting == oldest and report.result()["oldest_waiting"] == oldest
