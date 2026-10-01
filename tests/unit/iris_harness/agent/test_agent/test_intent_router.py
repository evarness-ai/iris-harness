"""Behavioral tests for the intent router."""

from __future__ import annotations

import pytest

from iris_harness.agent.intent_router import (
    HybridClassifier,
    IntentResult,
    IntentRouter,
    KeywordClassifier,
    KeywordFirstClassifier,
    LLMRouterClassifier,
)


def test_keyword_classifier_email_intent() -> None:
    clf = KeywordClassifier()
    result = clf.classify("check my inbox for new emails")

    assert result.intent == "communication"
    assert result.agent_type == "email"
    assert result.confidence >= 0.8


def test_keyword_classifier_gmail_email_extraction_intent() -> None:
    clf = KeywordClassifier()
    result = clf.classify("connect to my gmail and extract emails for a morning brief")

    assert result.intent == "communication"
    assert result.agent_type == "email"


def test_keyword_classifier_finance_networth_intent() -> None:
    clf = KeywordClassifier()
    # The live misroute: this previously fell through to the web-search system
    # handler (search/general) and stalled instead of reaching the finance agent.
    result = clf.classify("what's my total networth as of today?")

    assert result.intent == "finance"
    assert result.agent_type == "finance"
    assert result.confidence >= 0.8


@pytest.mark.parametrize(
    "query",
    [
        "how's my portfolio doing?",
        "what are my holdings?",
        "how much money do I have?",
        "check my financial statements",
        "what's my net worth",
        "show me my account balance",
        "how much did I spend last month?",
        "what's my spending this year",
        "show me my expenses",
        # issue 0009 — finance-account asks that name EMAIL as the source must still
        # reach finance (they previously fell through to the email agent).
        "based on the finance statements from my email or CAS statement, "
        "can you list my total bank account details?",
        "list my bank accounts",
        "what are my account details",
        "show me my demat accounts from the CAS",
        # issue 0014 — any "balance" ask routes to finance deterministically (was
        # falling back to the LLM, which sometimes mislabeled it as email).
        "what's my credit card balance",
        "what's my Northwind balance",
        "what's my balance",
        # issue 0015 — "do you have my <type> statements?" is a finance-document
        # inventory ask, not an email ask; must reach finance, not the email agent.
        "do you have my credit card statements?",
        "what statements do you have",
        "list my statements",
        # dues / bills vocabulary — the finance domain's own words. Previously
        # absent from the keyword anchor, so these fell to the small-LLM fallback
        # and sometimes misrouted (the "dues -> net-worth/system summary" report).
        "any dues that are pending?",
        "any due's pending?",
        "do I have any pending payments",
        "what bills are due",
        "anything overdue",
        "what do I owe",
    ],
)
def test_keyword_classifier_finance_phrasings(query: str) -> None:
    clf = KeywordClassifier()
    result = clf.classify(query)

    assert result.intent == "finance"
    assert result.agent_type == "finance"


@pytest.mark.parametrize(
    "query",
    [
        "any finance emails?",  # topic=finance but it's an inbox ask → email
        "show me my bank statement emails",
        "summarize the email from my bank",
        "any credit card emails?",  # issue 0014 — no "balance", so stays email
        "any statement emails?",  # issue 0015 — an inbox ask, not a doc inventory
    ],
)
def test_finance_broadening_does_not_steal_email_asks(query: str) -> None:
    # The issue-0009 finance broadening must not grab inbox queries that merely
    # mention finance/bank — those stay communication (email).
    result = KeywordClassifier().classify(query)
    assert result.intent == "communication"
    assert result.agent_type == "email"


@pytest.mark.parametrize(
    "query",
    [
        # issue 0019 — FX/exchange-rate is a web lookup, not the user's own finances.
        "What is the conversion rate from USD to INR?",
        "USD to INR rate",
        "exchange rate today",
        "convert 100 dollars to rupees",
        # issue 0019 — read/summarize a blog/URL is a web-fetch, not email/finance.
        "Can you read one of the blogs on my blog site and give me a summary?",
        "summarize https://web3notes.example/post",
        "open www.example.com and summarize it",
    ],
)
def test_web_and_fx_route_to_search(query: str) -> None:
    result = KeywordClassifier().classify(query)
    assert result.intent == "search", query
    assert result.agent_type == "system", query


@pytest.mark.parametrize(
    "query, agent",
    [
        ("read my email", "email"),  # "read" + email noun stays communication
        ("summarize my inbox", "email"),
        ("summarize my bank statement", "finance"),  # statement → finance, not web
    ],
)
def test_web_read_rule_does_not_steal_email_or_finance(query: str, agent: str) -> None:
    result = KeywordClassifier().classify(query)
    assert result.agent_type == agent, query


@pytest.mark.parametrize(
    "query",
    [
        "list the files in my catalog",
        "show me the file catalog",
        "what files do you manage",
        "my documents",
        "organize my downloads folder",
        "list my file organize plans",
        "check my file vault",
        "find my folder",
    ],
)
def test_keyword_classifier_files_route_to_filemanager(query: str) -> None:
    # Plural "files" + "catalog"/"documents" used to miss the narrow \bfile\b
    # rule and fall to the LLM (which mislabelled them email) — the live
    # FileManager misroute. They now deterministically reach the FileManager.
    result = KeywordClassifier().classify(query)
    assert result.intent == "files", query
    assert result.agent_type == "filemanager", query


@pytest.mark.parametrize(
    ("query", "agent"),
    [
        ("check my inbox for new emails", "email"),
        ("reply to the last message", "email"),
        ("what's my net worth", "finance"),
        ("list my bank accounts", "finance"),
        ("do I have any dues pending", "finance"),
    ],
)
def test_files_broadening_does_not_steal_email_or_finance(query: str, agent: str) -> None:
    # The catalog/files broadening must not poach email or finance asks (they
    # classify earlier in the rule order).
    result = KeywordClassifier().classify(query)
    assert result.agent_type == agent, query


def test_keyword_classifier_coding_intent() -> None:
    clf = KeywordClassifier()
    result = clf.classify("write a function to sort a list")

    assert result.intent == "coding"
    assert result.agent_type == "coding_agent"


def test_keyword_classifier_multi_step_detection() -> None:
    clf = KeywordClassifier()
    result = clf.classify("check my emails and then update the calendar")

    assert result.is_multi_step


def test_keyword_classifier_unknown_query_returns_general() -> None:
    clf = KeywordClassifier()
    result = clf.classify("gxkqz blargh florp")

    assert result.intent == "general"
    assert result.confidence < 0.8


def test_hybrid_classifier_uses_keyword_when_confident() -> None:
    llm_called = [False]

    def llm(prompt: str) -> str:
        llm_called[0] = True
        return "INTENT: coding\nAGENT: coding_agent\nMULTI_STEP: no"

    clf = HybridClassifier(llm_call=llm)
    result = clf.classify("send an email to Alice")

    assert result.agent_type == "email"
    assert not llm_called[0]


def test_intent_router_dispatches_to_registered_handler() -> None:
    router = IntentRouter(classifier=KeywordClassifier())
    outputs: list[str] = []
    router.register_agent("email", lambda q: outputs.append(q) or "email handled")

    result, handler = router.route("check my inbox")

    assert result.agent_type == "email"
    assert handler is not None


def test_intent_router_returns_none_handler_for_unregistered_agent() -> None:
    router = IntentRouter(classifier=KeywordClassifier())
    _, handler = router.route("gxkqz blargh")

    assert handler is None


# ---------------------------------------------------------------------------
# LLMRouterClassifier
# ---------------------------------------------------------------------------


def test_llm_router_parses_valid_json_response() -> None:
    # The router LLM now returns only the intent; agent + multi-step are derived.
    def invoke(_system: str, _user: str) -> str:
        return '{"intent": "coding"}'

    clf = LLMRouterClassifier(invoke=invoke)
    result = clf.classify("can you add a footer to the CLI?")

    assert result.intent == "coding"
    assert result.agent_type == "coding_agent"  # derived from _INTENT_TO_AGENT
    assert result.confidence == 0.9
    assert not result.is_multi_step


def test_llm_router_strips_markdown_fences() -> None:
    def invoke(_system: str, _user: str) -> str:
        return "```json\n" '{"intent": "search"}\n' "```"

    clf = LLMRouterClassifier(invoke=invoke)
    result = clf.classify("look up python decorators")

    assert result.intent == "search"
    assert result.agent_type == "system"  # web lookups go to the web-capable handler


def test_llm_router_falls_back_on_parse_error() -> None:
    fallback_called = [False]

    class _RecordingFallback:
        def classify(self, query: str, *, context: str | None = None):  # type: ignore[no-untyped-def]
            fallback_called[0] = True
            return KeywordClassifier().classify(query)

    def invoke(_system: str, _user: str) -> str:
        return "I think this is about emails maybe?"

    clf = LLMRouterClassifier(invoke=invoke, keyword_fallback=_RecordingFallback())
    result = clf.classify("send Alice a note")

    assert fallback_called[0]
    assert result.agent_type == "email"


def test_llm_router_falls_back_on_invocation_exception() -> None:
    def invoke(_system: str, _user: str) -> str:
        raise RuntimeError("ollama unreachable")

    clf = LLMRouterClassifier(invoke=invoke)
    result = clf.classify("write a function to sort a list")

    # Falls through to KeywordClassifier which classifies coding correctly.
    assert result.intent == "coding"
    assert result.agent_type == "coding_agent"


def test_llm_router_rejects_unknown_intent_label() -> None:
    def invoke(_system: str, _user: str) -> str:
        return '{"intent": "wizardry", "agent_type": "system", "is_multi_step": false}'

    clf = LLMRouterClassifier(invoke=invoke)
    result = clf.classify("write a python script")

    # Invalid intent → keyword fallback → 'coding'.
    assert result.intent == "coding"


def test_llm_router_does_not_route_chat_with_word_time_to_system() -> None:
    """Regression: keyword classifier mis-routed 'all the time' phrasing to system."""

    def invoke(_system: str, _user: str) -> str:
        return (
            '{"intent": "coding", "agent_type": "coding_agent", '
            '"is_multi_step": false, "confidence": 0.9}'
        )

    clf = LLMRouterClassifier(invoke=invoke)
    result = clf.classify(
        "can you add an autolist of commands instead of showing them all the time?"
    )

    assert result.intent == "coding"
    assert result.agent_type == "coding_agent"


def test_llm_router_derives_agent_from_intent_ignoring_model_agent() -> None:
    """The agent is derived from the intent in Python; a wrong ``agent_type`` the
    model emits is ignored (no more valid-but-mismatched misroutes)."""

    def invoke(_system: str, _user: str) -> str:
        return '{"intent": "code_exec", "agent_type": "rag"}'  # stray agent_type ignored

    clf = LLMRouterClassifier(invoke=invoke)
    result = clf.classify("create a pdf of the top 10 AI news today")

    assert result.intent == "code_exec"
    assert result.agent_type == "code_exec"


def test_keyword_first_uses_keyword_when_confident() -> None:
    """High-confidence keyword match short-circuits the secondary classifier."""

    class _Boom:
        def classify(
            self, _query: str, *, context: str | None = None
        ) -> IntentResult:  # pragma: no cover
            raise AssertionError("secondary classifier should not be called")

    clf = KeywordFirstClassifier(_Boom())
    result = clf.classify("create a pdf of the top 10 AI news today")

    assert result.intent == "code_exec"
    assert result.agent_type == "code_exec"


def test_keyword_first_defers_to_secondary_when_keyword_uncertain() -> None:
    """Queries with no keyword match should fall through to the secondary."""

    class _Stub:
        def classify(self, query: str, *, context: str | None = None) -> IntentResult:
            return IntentResult(
                intent="general",
                agent_type="system",
                confidence=0.95,
                raw_query=query,
            )

    clf = KeywordFirstClassifier(_Stub())
    result = clf.classify("hmm interesting thought")

    assert result.confidence == 0.95
    assert result.agent_type == "system"


# ---------------------------------------------------------------------------
# profile_query — added 2026-05-19 because Tier 1 models hallucinate stack
# details when answering identity questions; route to Tier 2 for fidelity.
# ---------------------------------------------------------------------------


def test_keyword_classifier_routes_what_do_you_know_about_me_to_profile_query() -> None:
    clf = KeywordClassifier()
    result = clf.classify("What do you know about me?")
    assert result.intent == "profile_query"
    assert result.agent_type == "system"


def test_do_you_know_my_x_routes_to_profile_query() -> None:
    # issue 0020: "do you know my <thing>" is a recall of a stored self-fact, not a
    # web search. Calendar nouns still route to the calendar agent.
    clf = KeywordClassifier()
    for q in ("Do you know my blog site?", "Do you know my blog?", "do you know about my blogs?"):
        assert clf.classify(q).intent == "profile_query", q
    assert clf.classify("do you know my schedule?").intent == "calendar"


def test_keyword_classifier_routes_whats_my_name_to_profile_query() -> None:
    clf = KeywordClassifier()
    assert clf.classify("what's my name?").intent == "profile_query"
    assert clf.classify("who am I").intent == "profile_query"
    assert clf.classify("what's my role").intent == "profile_query"
    assert clf.classify("what's my profile").intent == "profile_query"


def test_keyword_classifier_routes_stack_questions_to_profile_query() -> None:
    clf = KeywordClassifier()
    assert clf.classify("what languages do I prefer?").intent == "profile_query"
    assert clf.classify("what do I use for coding?").intent == "profile_query"
    assert clf.classify("do I use Rust?").intent == "profile_query"
    assert clf.classify("what's my stack").intent == "profile_query"
    assert clf.classify("name three programming languages I use daily").intent == "profile_query"
    assert clf.classify("list the cloud platforms I use").intent == "profile_query"


def test_keyword_classifier_does_not_misroute_neighbour_phrases() -> None:
    """Profile rule must not eat unrelated queries that just mention 'me'
    or 'I' in passing."""

    clf = KeywordClassifier()
    assert clf.classify("tell me how to write a REST endpoint").intent != "profile_query"
    assert clf.classify("show me the latest AI news").intent != "profile_query"
    assert clf.classify("send me an email summary").intent == "communication"
    assert clf.classify("schedule a meeting for me").intent == "calendar"


def test_keyword_classifier_routes_values_style_mission_to_profile_query() -> None:
    """2026-05-20 follow-up #3: profile_query regex broadened to cover
    values / style / mission / priorities / ambitions / personality."""

    clf = KeywordClassifier()
    assert clf.classify("What are my values?").intent == "profile_query"
    assert clf.classify("What is my style?").intent == "profile_query"
    assert clf.classify("What are my priorities?").intent == "profile_query"
    assert clf.classify("What is my mission?").intent == "profile_query"
    assert clf.classify("What's my personality?").intent == "profile_query"
    assert clf.classify("Do I value privacy?").intent == "profile_query"


def test_keyword_classifier_routes_what_kind_of_work_to_profile_query() -> None:
    """'What kind of <noun> am/do I ...' shape — a common way users
    ask about their own activities."""

    clf = KeywordClassifier()
    assert clf.classify("What kind of work am I doing?").intent == "profile_query"
    assert clf.classify("What kind of projects do I work on?").intent == "profile_query"
    assert clf.classify("What kind of code do I write?").intent == "profile_query"


def test_keyword_classifier_preserves_advice_form_general_intent() -> None:
    """Ambiguous-with-advice phrases ('how do I approach X', 'what are
    the best values for Y') must still route to general so the advice
    rules below the profile rule fire correctly."""

    clf = KeywordClassifier()
    assert clf.classify("How do I approach this CLI command?").intent == "general"
    assert clf.classify("What are the best values for tier1?").intent == "general"
    assert clf.classify("How should I structure this service?").intent == "general"


def test_keyword_classifier_recurring_standup_routes_to_calendar() -> None:
    """Phase 2 scenario harness regression (case cal-05): a recurring
    standup request matched no calendar keyword and fell through to the
    general fallback."""
    clf = KeywordClassifier()
    result = clf.classify("set up a recurring standup every weekday at 9")

    assert result.intent == "calendar"
    assert result.agent_type == "calendar"


# ─── exp-005 trap-family regressions (routing-rule-fixes pass) ──────────────
# Each case is a confident keyword error measured in exp-005 spike 0001/0002
# (agentic-evals docs/experiments/exp-005-router-shootout). The golden-set
# case id is noted; the scenario harness grades the full set.


@pytest.mark.parametrize(
    ("utterance", "intent", "agent"),
    [
        # adv-01: literal-response meta request, not an email op (Phase 1 live misroute)
        ("Please reply with just the single word: hello", "general", "system"),
        # adv-02: reminder verb governs over the email topic
        ("remind me to reply to John's email", "calendar", "calendar"),
        # adv-07: personal item lookup, not web search
        ("find my passport scan", "files", "filemanager"),
        # adv-08: first action governs (meeting room booking)
        ("book a meeting room and email the team", "calendar", "calendar"),
        # adv-09: possessive time-of-event query, not clock time
        ("what time is my flight?", "calendar", "calendar"),
        # adv-10: decision advice that merely mentions weather
        ("weather looks bad, should I cancel my trip?", "general", "system"),
        # adv-18: "who am I <gerund>" is an event query, not a profile query
        ("who am I meeting tomorrow?", "calendar", "calendar"),
        # adv-20: bug-fixing is coding even when the subject is an email parser
        ("find and fix the bug in my email parser", "coding", "coding_agent"),
        # adv-21: calendar action first, invites second
        ("update my calendar and send invites", "calendar", "calendar"),
    ],
)
def test_keyword_classifier_trap_families(utterance: str, intent: str, agent: str) -> None:
    result = KeywordClassifier().classify(utterance)
    assert (result.intent, result.agent_type) == (
        intent,
        agent,
    ), f"{utterance!r} -> {result.intent}/{result.agent_type}"


def test_keyword_classifier_trap_fixes_do_not_regress_core_routes() -> None:
    """Spot-check neighbors of the changed rules."""
    clf = KeywordClassifier()
    # who-am-i lookahead must not break the plain profile query
    assert clf.classify("who am i?").intent == "profile_query"
    # clock time still routes to system
    assert clf.classify("what time is it?").intent == "system"
    # plain email ops still route to email despite calendar moving up
    assert clf.classify("forward the invoice email to my accountant").intent == "communication"
    assert clf.classify("send a follow up to the recruiter").intent == "communication"
    # web search unaffected by the find-my files rule
    assert clf.classify("find out who won the f1 race yesterday").intent == "search"


def test_router_json_parser_strips_think_blocks() -> None:
    """exp-005 production gap: thinking models wrap output in <think> blocks;
    the parser must strip them (terminated or budget-truncated) before JSON
    extraction, else every reply silently falls back to keywords."""
    from iris_harness.agent.intent_router import _parse_router_json

    valid = '{"intent": "calendar", "agent_type": "calendar", "is_multi_step": false, "confidence": 0.9}'
    parsed = _parse_router_json(f"<think>hmm, scheduling...</think>\n{valid}")
    assert parsed is not None and parsed["intent"] == "calendar"

    # Unterminated think block (output budget exhausted): no JSON to find.
    assert _parse_router_json("<think>thinking forever, no json emitted") is None

    # A JSON-looking object INSIDE the think block must not be parsed.
    leaky = '<think>maybe {"intent": "weather", "agent_type": "system"}?</think>' + valid
    parsed = _parse_router_json(leaky)
    assert parsed is not None and parsed["intent"] == "calendar"


def test_keyword_search_routes_to_web_capable_system() -> None:
    """Web/factual lookups must reach the system handler (research), not rag."""
    result = KeywordClassifier().classify("search the web for the AAPL stock price")
    assert result.intent == "search"
    assert result.agent_type == "system"


def test_llm_router_search_routes_to_system() -> None:
    def invoke(_system: str, _user: str) -> str:
        return '{"intent": "search"}'

    result = LLMRouterClassifier(invoke=invoke).classify("what is apple stock worth?")
    assert result.intent == "search"
    assert result.agent_type == "system"


def test_keyword_demotes_ambiguous_general_help_to_defer() -> None:
    """general/help keyword matches score below the KeywordFirst threshold so the
    LLM router gets a say; precise domain intents stay authoritative."""
    clf = KeywordClassifier()
    assert clf.classify("explain how this works").confidence < 0.8  # help
    assert clf.classify("what is the best way to do X").confidence < 0.8  # general advice
    assert clf.classify("send Alice an email").confidence >= 0.8  # domain stays high


# ── issue 0002: context-aware routing + clarify ─────────────────────────────


def test_keyword_first_defers_to_secondary_when_context_present() -> None:
    """A follow-up (context present) must reach the context-aware secondary even
    if the isolated message would match a keyword rule."""
    seen: dict[str, str | None] = {"ctx": None}

    class _Secondary:
        def classify(self, query: str, *, context: str | None = None) -> IntentResult:
            seen["ctx"] = context
            return IntentResult(intent="communication", agent_type="email", confidence=0.9)

    clf = KeywordFirstClassifier(_Secondary())
    result = clf.classify(
        "what is the summary?",
        context="User: any AI emails?\nAssistant: yes, an CourseHub one about AI workflows",
    )

    assert result.agent_type == "email"  # secondary used, not the keyword guess
    assert seen["ctx"] is not None and "CourseHub" in seen["ctx"]


def test_keyword_first_uses_keyword_on_first_turn_no_context() -> None:
    class _Boom:
        def classify(self, _query: str, *, context: str | None = None) -> IntentResult:
            raise AssertionError("secondary must not be called on a confident first turn")

    clf = KeywordFirstClassifier(_Boom())
    result = clf.classify("create a pdf of the top 10 AI news today")  # no context
    assert result.agent_type == "code_exec"


def test_llm_router_passes_context_into_user_message() -> None:
    seen: dict[str, str] = {}

    def invoke(_system: str, user: str) -> str:
        seen["user"] = user
        return '{"intent": "communication"}'

    clf = LLMRouterClassifier(invoke=invoke)
    clf.classify("what is the summary?", context="Assistant: you got an AI email from CourseHub")

    assert "Conversation so far" in seen["user"]
    assert "CourseHub" in seen["user"]
    assert "what is the summary?" in seen["user"]


def test_llm_router_clarify_maps_to_clarify_agent() -> None:
    clf = LLMRouterClassifier(invoke=lambda _s, _u: '{"intent": "clarify"}')
    result = clf.classify("hmm")
    assert result.intent == "clarify"
    assert result.agent_type == "clarify"


def test_keyword_first_keeps_confident_keyword_even_with_context() -> None:
    """A confident keyword match wins even on a follow-up — keyword-clear turns
    like 'any finance emails?' must not be handed to a weak router LLM (issue 0002
    regression: context must NOT force a deferral when the keyword is sure)."""

    class _Boom:
        def classify(self, _query: str, *, context: str | None = None) -> IntentResult:
            raise AssertionError("secondary must not be called for a confident keyword")

    clf = KeywordFirstClassifier(_Boom())
    result = clf.classify("any finance emails today?", context="User: hi\nAssistant: hello there")
    assert result.agent_type == "email"


def test_looks_like_question_or_command_guards_fact_extraction() -> None:
    # issue 0021: questions/commands must not be mined for durable facts.
    from iris_harness.runtime.turn_capture import _looks_like_question_or_command as q

    for m in [
        "Do you know my blog site?",
        "what's apple stock worth",
        "show my balance",
        "summarize my inbox",
        "how is my day",
        "any meeting tomorrow?",
    ]:
        assert q(m), m
    for m in ["I write blogs at www.web3notes.example", "my name is Robin", "I work at Acme"]:
        assert not q(m), m


def test_email_attachment_intent_detector() -> None:
    # issue 0024: attachment/document asks route deterministically to find_attachment.
    from iris_personal.plugins.email_workflows.agent import (
        _is_email_attachment_intent as att,
    )

    for m in [
        "Can you get my passport copy from email?",
        "find the invoice pdf",
        "where is my boarding pass",
        "download my resume",
        "send me my offer letter",
    ]:
        assert att(m), m
    for m in ["how is my inbox today?", "any AI emails?", "summarize my inbox"]:
        assert not att(m), m


# ── ADR-0103: generic action escalation ─────────────────────────────────────


@pytest.mark.parametrize(
    "message",
    [
        "can you update my daily briefing to add the outstanding dues next week?",
        "remind me on this day when I get this email",
        "create a notification when I receive this email",
        "configure my morning brief to include weather",
        "set up a reminder for my rent",
        "add a recurring transfer to savings",
        "turn off email notifications at night",
    ],
)
def test_is_action_request_matches_generic_imperatives(message: str) -> None:
    from iris_harness.agent.intent_router import is_action_request

    assert is_action_request(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "what are my insurance dues",
        "show my portfolio",
        "what is my net worth",
        "how is my day",
        "tell me my outstanding dues",
        "any AI emails?",
    ],
)
def test_is_action_request_ignores_reads(message: str) -> None:
    from iris_harness.agent.intent_router import is_action_request

    assert is_action_request(message) is False


def test_promote_weak_tier_action_to_action_intent() -> None:
    # A weak-tier classification of an imperative is promoted to `action` (→ tier2).
    from iris_harness.agent.intent_router import promote_to_action_intent

    for weak in ("general", "system", "help"):
        base = IntentResult(intent=weak, agent_type="system", confidence=0.5, raw_query="x")
        promoted = promote_to_action_intent(base, "configure my briefing to add dues")
        assert promoted.intent == "action"
        assert promoted.agent_type == "system"


def test_promote_leaves_domain_and_reads_untouched() -> None:
    from iris_harness.agent.intent_router import promote_to_action_intent

    # Domain actions already route to a capable tier — don't relabel.
    finance = IntentResult(intent="finance", agent_type="finance", confidence=0.85, raw_query="x")
    assert promote_to_action_intent(finance, "update my finance tracker").intent == "finance"
    # A read on a weak tier is not an action.
    read = IntentResult(intent="general", agent_type="system", confidence=0.5, raw_query="x")
    assert promote_to_action_intent(read, "what are my dues").intent == "general"


def test_action_intent_maps_to_tier2() -> None:
    # The whole point: a promoted action starts on the capable instruct tier.
    from pathlib import Path

    from iris_harness.llm.tier_router import TierRouter

    tr = TierRouter.load_from_yaml(Path("config/llm_tiers.yaml"))
    assert tr.get_tier("action").name == tr.get_tier("finance").name  # both tier2 "Advanced"
    assert tr.get_tier("action").name != tr.get_tier("general").name  # not the weak tier1


# --- multi-step cues live in config/multi_step.yaml (ADR-0111) -------------------------


def test_shipped_cues_include_the_phrases_the_acceptance_query_used() -> None:
    from pathlib import Path

    from iris_harness.agent.intent_router import (
        compile_multi_step_pattern,
        load_multi_step_cues,
    )

    root = Path(__file__).resolve().parents[5]
    cues = load_multi_step_cues(root / "config" / "multi_step.yaml")
    assert "after this" in cues and "for each" in cues and "and then" in cues
    pattern = compile_multi_step_pattern(cues)
    assert pattern is not None
    assert pattern.search("find the dues, after this schedule a reminder for each due")
    assert pattern.search("first find it then read it")  # "first ... then" spans words
    assert not pattern.search("what do i owe")


def test_missing_or_empty_cue_file_means_no_turn_is_compound(tmp_path) -> None:
    from iris_harness.agent.intent_router import (
        compile_multi_step_pattern,
        load_multi_step_cues,
    )

    assert load_multi_step_cues(tmp_path / "absent.yaml") == ()
    (tmp_path / "empty.yaml").write_text("cues: []\n")
    assert compile_multi_step_pattern(load_multi_step_cues(tmp_path / "empty.yaml")) is None
    (tmp_path / "bad.yaml").write_text("cues: not-a-list\n")
    assert load_multi_step_cues(tmp_path / "bad.yaml") == ()


def test_a_temp_config_dir_does_not_poison_the_shipped_cues(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Regression (found by the release-1 export's strict pass, an xdist-order flake): the
    compiled cues were cached once per process, so a test that pointed IRIS_CONFIG_DIR at
    a temp dir with no multi_step.yaml left every later turn in that worker non-compound.
    """
    from iris_harness.agent.intent_router import is_multi_step_query

    query = "check my email and then summarize my inbox"
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(tmp_path))
    assert is_multi_step_query(query) is False  # that dir has no cues: honest degradation
    monkeypatch.delenv("IRIS_CONFIG_DIR")
    assert is_multi_step_query(query) is True  # the shipped config is read again


def test_all_three_classifiers_share_the_detector(monkeypatch) -> None:
    from iris_harness.agent import intent_router as ir

    monkeypatch.setattr(ir, "is_multi_step_query", lambda q: "zzz" in q)
    assert ir.KeywordClassifier().classify("send zzz mail").is_multi_step is True
    assert ir.KeywordClassifier().classify("send mail").is_multi_step is False


def test_llm_classifier_failure_is_logged_and_falls_back_to_keywords(caplog) -> None:  # type: ignore[no-untyped-def]
    """A dead model used to look like ordinary keyword routing."""
    import logging

    from iris_harness.agent.intent_router import KeywordClassifier, LLMClassifier

    def dead_model(_prompt: str) -> str:
        raise ConnectionError("model server down")

    query = "what's on my calendar tomorrow, my secret plan"
    with caplog.at_level(logging.DEBUG, logger="iris_harness.agent.intent_router"):
        result = LLMClassifier(dead_model).classify(query)

    assert result == KeywordClassifier().classify(query)
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == "iris_harness.agent.intent_router" and r.levelno == logging.WARNING
    ]
    assert warnings == [
        "intent router: LLM classify failed (ConnectionError); using the keyword router"
    ]
    assert all("secret plan" not in w for w in warnings)


# --- keyword pre-rules live in config/intent_keywords.yaml -----------------------------


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        # 2026-09-27: this reached finance on the word "expense".
        ("Mark the expense report task as done", "planner"),
        ("add a task to renew my passport by Friday", "planner"),
        ("what's left on my to-do list", "planner"),
        # A turn that names a reminder, email or code keeps its own rule.
        ("remind me to finish the report task at 6pm", "calendar"),
        ("any emails about the passport task?", "communication"),
        ("write a task scheduler script", "coding"),
        # No task word: the built-in rules, unchanged.
        ("how much did I spend on travel expenses", "finance"),
    ],
)
def test_shipped_pre_rules_route_the_owners_tasks(message: str, intent: str) -> None:
    assert KeywordClassifier().classify(message).intent == intent


def test_pre_rules_load_from_yaml_and_skip_bad_entries(tmp_path) -> None:
    from iris_harness.agent.intent_router import _pre_rule_intent, load_keyword_pre_rules

    path = tmp_path / "intent_keywords.yaml"
    path.write_text(
        "pre_rules:\n"
        "  - {intent: planner, any: [chore], unless: [calendar]}\n"
        "  - {intent: not-an-intent, any: [x]}\n"
        "  - {intent: finance, any: []}\n"
    )
    rules = load_keyword_pre_rules(path)
    assert [r.intent for r in rules] == ["planner"]
    assert _pre_rule_intent("mark the laundry chore done", rules) == "planner"
    assert _pre_rule_intent("put the chore on my calendar", rules) is None
    assert _pre_rule_intent("chores", rules) is None  # word-bounded


def test_missing_or_malformed_pre_rules_file_means_none(tmp_path) -> None:
    from iris_harness.agent.intent_router import load_keyword_pre_rules

    assert load_keyword_pre_rules(tmp_path / "absent.yaml") == ()
    (tmp_path / "bad.yaml").write_text("pre_rules: not-a-list\n")
    assert load_keyword_pre_rules(tmp_path / "bad.yaml") == ()


# --- the regex rules live in config/intent_keywords.yaml (rules:) -----------------------


def test_the_shipped_rules_load_in_order_and_the_router_has_no_list_of_its_own() -> None:
    import inspect
    from pathlib import Path

    from iris_harness.agent import intent_router
    from iris_harness.agent.intent_router import load_keyword_rules

    root = Path(__file__).resolve().parents[5]
    rules = load_keyword_rules(root / "config" / "intent_keywords.yaml")
    assert rules[0].intent == "profile_query"  # before help/general ("what" is in both)
    assert rules[-1].intent == "search"
    order = [r.intent for r in rules]
    # Order is behaviour: calendar before planner, finance before communication.
    assert order.index("calendar") < order.index("planner") < order.index("finance")
    assert order.index("finance") < order.index("communication")
    # The keyword vocabulary is YAML (owner rule 2026-09-13), not a Python list.
    assert "_KEYWORD_RULES:" not in inspect.getsource(intent_router)


def _rules_file(tmp_path, body: str):  # type: ignore[no-untyped-def]
    path = tmp_path / "intent_keywords.yaml"
    path.write_text(body)
    return path


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("rules: []\n", "non-empty list"),
        ("rules:\n  - {intent: nope, agent: system, pattern: [x]}\n", "unknown intent 'nope'"),
        ("rules:\n  - {intent: search, agent: nope, pattern: [x]}\n", "unknown agent 'nope'"),
        ("rules:\n  - {intent: search, agent: system, pattern: ['(']}\n", "bad regex"),
        ("rules:\n  - {intent: search, agent: system}\n", "'pattern' must be a list"),
        ("rules: [\n", "cannot read keyword rules"),
    ],
)
def test_a_broken_rules_file_is_refused_naming_the_rule(tmp_path, body: str, message: str) -> None:
    from iris_harness.agent.intent_router import load_keyword_rules

    with pytest.raises(ValueError, match=message):
        load_keyword_rules(_rules_file(tmp_path, body))


def test_fragments_join_with_nothing_between(tmp_path) -> None:
    from iris_harness.agent.intent_router import load_keyword_rules

    path = _rules_file(
        tmp_path,
        "rules:\n  - intent: weather\n    agent: system\n    pattern: ['\\bsun', 'ny\\b']\n",
    )
    [rule] = load_keyword_rules(path)
    assert rule.pattern.search("is it sunny") and not rule.pattern.search("sun ny")


def test_a_config_dir_without_the_file_uses_the_shipped_rules(tmp_path, monkeypatch) -> None:
    from iris_harness.agent import intent_router as ir

    monkeypatch.setenv("IRIS_CONFIG_DIR", str(tmp_path))
    ir._keyword_rules.cache_clear()
    ir._keyword_pre_rules.cache_clear()
    try:
        assert KeywordClassifier().classify("what's my net worth").intent == "finance"
        assert KeywordClassifier().classify("mark the expense report task done").intent == "planner"
    finally:
        ir._keyword_rules.cache_clear()
        ir._keyword_pre_rules.cache_clear()


def test_a_broken_override_falls_back_to_the_shipped_rules(tmp_path, monkeypatch, caplog) -> None:
    import logging

    from iris_harness.agent import intent_router as ir

    _rules_file(tmp_path, "rules:\n  - {intent: nope, agent: system, pattern: [x]}\n")
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(tmp_path))
    ir._keyword_rules.cache_clear()
    try:
        with caplog.at_level(logging.ERROR):
            assert KeywordClassifier().classify("what's my net worth").intent == "finance"
        assert "refused; using the shipped" in caplog.text
    finally:
        ir._keyword_rules.cache_clear()
