"""Tests for the `skill_brief` heartbeat handler."""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from iris_harness.runtime.handlers.skill_brief import (
    SlotContext,
    _build_skill_brief_handler,
    _build_tool_index,
    render_brief_package,
    render_brief_result,
    render_layout,
    resolve_slot,
)
from iris_harness.services.channels import ChannelGateway, ChannelMessage
from iris_harness.services.channels.connectors.console import ConsoleConnector
from iris_harness.services.channels.models import DeliveryReceipt, DeliveryStatus
from iris_harness.services.digests import shared_digest_store
from iris_harness.services.heartbeat import HeartbeatDefinition, HeartbeatStatus
from iris_harness.tools.skills.models import (
    BriefLiteralSlot,
    BriefSpec,
    BriefToolSlot,
    SkillManifest,
    SkillPackage,
    SkillRequirements,
    SkillToolManifest,
)


class _NoArgs(BaseModel):
    pass


class _FakeReminderTool(BaseTool):
    name: str = "list_due_reminders"
    description: str = "fake tool returning two reminders"
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> list[dict]:
        return [
            {"task": "Buy milk", "time": "9am"},
            {"task": "Call dentist", "time": "10am"},
        ]

    async def _arun(self) -> list[dict]:
        return self._run()


class _EmptyTool(BaseTool):
    name: str = "list_due_reminders"
    description: str = "fake tool returning empty list"
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> list[dict]:
        return []

    async def _arun(self) -> list[dict]:
        return []


def _fake_package(*, skill_name: str, tool_name: str, tool_class: type[BaseTool]) -> SkillPackage:
    manifest = SkillManifest(
        name=skill_name,
        version="0.1.0",
        description="fake",
        author="iris",
        license="Apache-2.0",
        tools=(
            SkillToolManifest(
                name=tool_name,
                description="fake",
                governor_route="system/read",
            ),
        ),
        requires=SkillRequirements(),
    )
    return SkillPackage(
        manifest=manifest,
        skill_dir=Path("/fake/skill"),
        tools_module_path=Path("/fake/skill/tools.py"),
        tool_classes=(tool_class,),
    )


def _fake_brief_package(*, brief: BriefSpec, name: str = "sample-brief") -> SkillPackage:
    manifest = SkillManifest(
        name=name,
        version="0.1.0",
        description="brief skill",
        author="iris",
        license="Apache-2.0",
        kind="brief",
        brief=brief,
    )
    return SkillPackage(
        manifest=manifest,
        skill_dir=Path("/fake/brief"),
        tools_module_path=Path("/fake/brief/tools.py"),
        tool_classes=(),
    )


def _fake_registry(packages: list[SkillPackage]) -> SimpleNamespace:
    def list_packages(*, agent_name=None, only_loadable: bool = False):
        return tuple(p for p in packages if (not only_loadable) or p.is_loadable)

    return SimpleNamespace(list_packages=list_packages)


def test_render_layout_substitutes_placeholders() -> None:
    output = render_layout("Hi {{name}}, today is {{day}}.", {"name": "Iris", "day": "Wed"})
    assert output == "Hi Iris, today is Wed."


def test_render_layout_raises_on_missing_slot() -> None:
    with pytest.raises(KeyError, match="ghost"):
        render_layout("{{ghost}}", {})


def test_resolve_literal_slot_supports_strftime_placeholders() -> None:
    from datetime import datetime

    ctx = SlotContext(now=datetime(2026, 5, 14, 9, 0), tool_index={})
    slot = BriefLiteralSlot(kind="literal", value="{today:%A}")
    assert resolve_slot(slot, ctx, uses=()) == "Thursday"


def test_resolve_tool_slot_invokes_tool_and_formats_bullets() -> None:
    from datetime import datetime

    package = _fake_package(
        skill_name="calendar-reminders",
        tool_name="list_due_reminders",
        tool_class=_FakeReminderTool,
    )
    registry = _fake_registry([package])
    ctx = SlotContext(now=datetime(2026, 5, 14), tool_index=_build_tool_index(registry))
    slot = BriefToolSlot(
        kind="tool",
        skill="calendar-reminders",
        tool="list_due_reminders",
        args={},
        format="bullets",
        empty="none",
    )
    assert resolve_slot(slot, ctx, uses=("calendar-reminders",)) == "- Buy milk\n- Call dentist"


def test_resolve_tool_slot_applies_item_template() -> None:
    from datetime import datetime

    package = _fake_package(
        skill_name="calendar-reminders",
        tool_name="list_due_reminders",
        tool_class=_FakeReminderTool,
    )
    registry = _fake_registry([package])
    ctx = SlotContext(now=datetime(2026, 5, 14), tool_index=_build_tool_index(registry))
    slot = BriefToolSlot(
        kind="tool",
        skill="calendar-reminders",
        tool="list_due_reminders",
        args={},
        format="bullets",
        empty="none",
        item_template="{task} @ {time}",
    )
    assert (
        resolve_slot(slot, ctx, uses=("calendar-reminders",))
        == "- Buy milk @ 9am\n- Call dentist @ 10am"
    )


def test_resolve_tool_slot_uses_empty_fallback_for_empty_result() -> None:
    from datetime import datetime

    package = _fake_package(
        skill_name="calendar-reminders",
        tool_name="list_due_reminders",
        tool_class=_EmptyTool,
    )
    registry = _fake_registry([package])
    ctx = SlotContext(now=datetime(2026, 5, 14), tool_index=_build_tool_index(registry))
    slot = BriefToolSlot(
        kind="tool",
        skill="calendar-reminders",
        tool="list_due_reminders",
        format="bullets",
        empty="No reminders due.",
    )
    assert resolve_slot(slot, ctx, uses=("calendar-reminders",)) == "No reminders due."


def test_resolve_tool_slot_rejects_skill_outside_uses() -> None:
    from datetime import datetime

    ctx = SlotContext(now=datetime(2026, 5, 14), tool_index={})
    slot = BriefToolSlot(kind="tool", skill="other", tool="t")
    with pytest.raises(ValueError, match="not in brief.uses"):
        resolve_slot(slot, ctx, uses=("allowed",))


def test_handler_renders_and_dispatches_to_channel() -> None:
    stream = StringIO()
    gateway = ChannelGateway()
    gateway.register(ConsoleConnector(name="console", stream=stream))

    brief = BriefSpec(
        subject="Test",
        uses=("calendar-reminders",),
        layout="Reminders:\n{{reminders}}",
        slots={
            "reminders": BriefToolSlot(
                kind="tool",
                skill="calendar-reminders",
                tool="list_due_reminders",
                format="bullets",
                empty="none",
            ),
        },
    )
    registry = _fake_registry(
        [
            _fake_brief_package(brief=brief),
            _fake_package(
                skill_name="calendar-reminders",
                tool_name="list_due_reminders",
                tool_class=_FakeReminderTool,
            ),
        ]
    )
    runtime = SimpleNamespace(skill_registry=registry, channels=gateway, default_channel="console")

    handler = _build_skill_brief_handler(runtime)
    run = handler(
        HeartbeatDefinition(
            name="skill_brief",
            handler="skill_brief",
            schedule="manual",
            params={"channel": "console", "skill_id": "sample-brief"},
        )
    )

    assert run.status is HeartbeatStatus.SUCCESS
    assert "Reminders:" in run.output
    assert "- Buy milk" in run.output
    assert "[console]" in stream.getvalue()
    assert "Buy milk" in stream.getvalue()


# ---------------------------------------------------------------------------
# Phase 2: per-routine section selection + content-style at render
# ---------------------------------------------------------------------------


_STOCKS_INVOKED = {"count": 0}


class _FakeStocksTool(BaseTool):
    name: str = "fetch_stocks"
    description: str = "fake tool returning three stocks"
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> list[dict]:
        _STOCKS_INVOKED["count"] += 1
        return [
            {"sym": "AAA"},
            {"sym": "BBB"},
            {"sym": "CCC"},
        ]

    async def _arun(self) -> list[dict]:
        return self._run()


def _multi_section_setup() -> SimpleNamespace:
    """A brief with a greeting + two tool sections (reminders, stocks)."""
    _STOCKS_INVOKED["count"] = 0
    brief = BriefSpec(
        subject="Multi",
        uses=("calendar-reminders", "finance"),
        layout=(
            "Good morning. {{date}}.\n\n" "## Reminders\n{{reminders}}\n\n" "## Stocks\n{{stocks}}"
        ),
        slots={
            "date": BriefLiteralSlot(kind="literal", value="{today:%A}"),
            "reminders": BriefToolSlot(
                kind="tool",
                skill="calendar-reminders",
                tool="list_due_reminders",
                format="bullets",
                empty="none",
                item_template="{task}",
            ),
            "stocks": BriefToolSlot(
                kind="tool",
                skill="finance",
                tool="fetch_stocks",
                format="bullets",
                empty="none",
                item_template="{sym}",
            ),
        },
    )
    registry = _fake_registry(
        [
            _fake_brief_package(brief=brief, name="multi-brief"),
            _fake_package(
                skill_name="calendar-reminders",
                tool_name="list_due_reminders",
                tool_class=_FakeReminderTool,
            ),
            _fake_package(
                skill_name="finance",
                tool_name="fetch_stocks",
                tool_class=_FakeStocksTool,
            ),
        ]
    )
    return SimpleNamespace(
        registry=registry,
        package=next(p for p in registry.list_packages() if p.manifest.name == "multi-brief"),
    )


def test_render_full_brief_when_no_section_keys() -> None:
    setup = _multi_section_setup()
    body = render_brief_package(setup.package, setup.registry)
    assert "## Reminders" in body
    assert "## Stocks" in body
    assert "- AAA" in body
    assert "- Buy milk" in body


def test_section_keys_filters_to_selected_sections() -> None:
    setup = _multi_section_setup()
    body = render_brief_package(setup.package, setup.registry, section_keys=("stocks",))
    assert "## Stocks" in body
    assert "- AAA" in body
    # unselected reminders section pruned, and its tool never invoked
    assert "## Reminders" not in body
    assert "Buy milk" not in body
    # greeting / literal block survives
    assert "Good morning." in body


def test_unselected_section_tool_is_not_invoked() -> None:
    setup = _multi_section_setup()
    render_brief_package(setup.package, setup.registry, section_keys=("reminders",))
    # stocks not selected -> its tool must not run (no wasted fetch / side-effects)
    assert _STOCKS_INVOKED["count"] == 0


def test_content_lines_per_item_caps_bullets_per_section() -> None:
    setup = _multi_section_setup()
    body = render_brief_package(
        setup.package,
        setup.registry,
        section_keys=("stocks",),
        content_lines_per_item=2,
    )
    assert "- AAA" in body
    assert "- BBB" in body
    assert "- CCC" not in body  # capped at 2


def test_section_order_reorders_kept_sections() -> None:
    setup = _multi_section_setup()
    body = render_brief_package(
        setup.package,
        setup.registry,
        section_keys=("reminders", "stocks"),
        section_order=("stocks", "reminders"),
    )
    assert body.index("## Stocks") < body.index("## Reminders")


def test_header_and_footer_wrap_body() -> None:
    setup = _multi_section_setup()
    body = render_brief_package(
        setup.package,
        setup.registry,
        section_keys=("stocks",),
        header="HELLO HEADER",
        footer="BYE FOOTER",
    )
    assert body.startswith("HELLO HEADER")
    assert body.rstrip().endswith("BYE FOOTER")


def test_handler_honors_source_preferences_param() -> None:
    setup = _multi_section_setup()
    gateway = ChannelGateway()
    gateway.register(ConsoleConnector(name="console", stream=StringIO()))
    runtime = SimpleNamespace(
        skill_registry=setup.registry, channels=gateway, default_channel="console"
    )
    handler = _build_skill_brief_handler(runtime)
    run = handler(
        HeartbeatDefinition(
            name="skill_brief",
            handler="skill_brief",
            schedule="manual",
            params={
                "channel": "console",
                "skill_id": "multi-brief",
                "source_preferences": ["stocks"],
                "content_lines_per_item": 1,
            },
        )
    )
    assert run.status is HeartbeatStatus.SUCCESS
    assert "## Stocks" in run.output
    assert "## Reminders" not in run.output
    assert "- AAA" in run.output
    assert "- BBB" not in run.output  # capped at 1


def test_handler_fails_when_skill_id_missing() -> None:
    runtime = SimpleNamespace(
        skill_registry=_fake_registry([]), channels=ChannelGateway(), default_channel="console"
    )
    handler = _build_skill_brief_handler(runtime)
    run = handler(HeartbeatDefinition(name="skill_brief", handler="skill_brief", schedule="manual"))
    assert run.status is HeartbeatStatus.FAILED
    assert "skill_id" in (run.error or "")


def test_handler_fails_when_skill_id_unknown() -> None:
    runtime = SimpleNamespace(
        skill_registry=_fake_registry([]), channels=ChannelGateway(), default_channel="console"
    )
    handler = _build_skill_brief_handler(runtime)
    run = handler(
        HeartbeatDefinition(
            name="skill_brief",
            handler="skill_brief",
            schedule="manual",
            params={"skill_id": "no-such-skill"},
        )
    )
    assert run.status is HeartbeatStatus.FAILED
    assert "no-such-skill" in (run.error or "")


# ---------------------------------------------------------------------------
# Loop-proof PR 2: partial render, stored copy, three renderings
# ---------------------------------------------------------------------------


class _TimeoutStocksTool(BaseTool):
    name: str = "fetch_stocks"
    description: str = "fake tool that times out"
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> list[dict]:
        raise TimeoutError("research engine timeout after 20 s")

    async def _arun(self) -> list[dict]:
        return self._run()


class _BoomReminderTool(BaseTool):
    name: str = "list_due_reminders"
    description: str = "fake tool that raises with no message"
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> list[dict]:
        raise RuntimeError()

    async def _arun(self) -> list[dict]:
        return self._run()


class _FocusTool(BaseTool):
    name: str = "list_due_reminders"
    description: str = "fake Focus lines carrying the 👎 action link"
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> list[str]:
        return [
            "Mo Harbor School · Saturday class",
            "Fabrikam · Travel smarter [👎](iris:not-useful/offers%40fabrikam.test)",
        ]

    async def _arun(self) -> list[str]:
        return self._run()


class _ManyItemsTool(BaseTool):
    name: str = "fetch_stocks"
    description: str = "fake tool returning enough lines to need several Telegram messages"
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> list[str]:
        return [f"STOCK{i:03d} " + "x" * 80 for i in range(120)]

    async def _arun(self) -> list[str]:
        return self._run()


def _setup_with(
    reminders: type[BaseTool] = _FakeReminderTool, stocks: type[BaseTool] = _FakeStocksTool
) -> SimpleNamespace:
    """The multi-section brief, with either tool swapped for a variant."""
    setup = _multi_section_setup()
    registry = _fake_registry(
        [
            setup.package,
            _fake_package(
                skill_name="calendar-reminders",
                tool_name="list_due_reminders",
                tool_class=reminders,
            ),
            _fake_package(skill_name="finance", tool_name="fetch_stocks", tool_class=stocks),
        ]
    )
    return SimpleNamespace(registry=registry, package=setup.package)


class _RecordingConnector:
    """A channel that remembers every message, in order."""

    def __init__(self, name: str, status: DeliveryStatus = DeliveryStatus.SENT) -> None:
        self.name = name
        self.status = status
        self.sent: list[ChannelMessage] = []

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        self.sent.append(message)
        error = "no browser has subscribed" if self.status is DeliveryStatus.SKIPPED else ""
        return DeliveryReceipt(channel=self.name, status=self.status, error=error)

    def healthy(self) -> bool:
        return True


@pytest.fixture()
def digest_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stored digests go to a throwaway data dir, never the developer's own."""
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    return tmp_path


def _run(runtime: SimpleNamespace, channel: str, **params: object):
    handler = _build_skill_brief_handler(runtime)
    return handler(
        HeartbeatDefinition(
            name="morning-digest",
            handler="skill_brief",
            schedule="manual",
            params={"channel": channel, "skill_id": "multi-brief", **params},
        )
    )


def test_a_failing_section_is_named_and_the_rest_still_renders() -> None:
    setup = _setup_with(stocks=_TimeoutStocksTool)
    rendered = render_brief_result(
        setup.package, setup.registry, section_keys=("reminders", "stocks"), footer="loop: ok"
    )
    assert "- Buy milk" in rendered.body
    assert "## Stocks" not in rendered.body
    assert (
        "⚠ couldn't build: Stocks (research engine timeout after 20 s) "
        "— everything else is current." in rendered.body
    )
    # The failure line sits after the sections and before the footer.
    assert rendered.body.index("⚠ couldn't build") > rendered.body.index("- Buy milk")
    assert rendered.body.rstrip().endswith("loop: ok")
    assert [(f.name, f.title) for f in rendered.failed] == [("stocks", "Stocks")]


def test_the_full_brief_path_is_partial_too() -> None:
    setup = _setup_with(stocks=_TimeoutStocksTool)
    body = render_brief_package(setup.package, setup.registry)
    assert "- Buy milk" in body
    assert "⚠ couldn't build: Stocks" in body


def test_failures_with_different_reasons_are_each_named() -> None:
    setup = _setup_with(reminders=_BoomReminderTool, stocks=_TimeoutStocksTool)
    rendered = render_brief_result(
        setup.package, setup.registry, section_keys=("reminders", "stocks")
    )
    assert "Good morning." in rendered.body
    assert (
        "⚠ couldn't build: Reminders (RuntimeError); Stocks (research engine timeout after 20 s)"
        in rendered.body
    )
    assert rendered.counts == ()


def test_counts_follow_the_rendered_list_sections_in_order() -> None:
    setup = _setup_with()
    rendered = render_brief_result(
        setup.package,
        setup.registry,
        section_keys=("reminders", "stocks"),
        section_order=("stocks", "reminders"),
    )
    assert rendered.counts == (("Stocks", 3), ("Reminders", 2))


class _BillsTool(BaseTool):
    name: str = "brief_bills_due"
    description: str = "fake bills: one owed, one card in credit"
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> list[dict]:
        return [
            {"name": "Discover Card", "amount": "USD 35.00", "owed": "yes"},
            {"name": "Wingtip", "amount": "INR 3,150.40 credit — nothing to pay", "owed": ""},
        ]

    async def _arun(self) -> list[dict]:
        return self._run()


def _bills_brief(count_if: str | None) -> SimpleNamespace:
    brief = BriefSpec(
        subject="Bills",
        uses=("finance-brief",),
        layout="## Bills due\n{{bills_due}}",
        slots={
            "bills_due": BriefToolSlot(
                kind="tool",
                skill="finance-brief",
                tool="brief_bills_due",
                format="bullets",
                item_template="{name}: {amount}",
                count_if=count_if,
                empty="No bills due.",
            )
        },
    )
    package = _fake_brief_package(brief=brief)
    registry = _fake_registry(
        [
            package,
            _fake_package(
                skill_name="finance-brief", tool_name="brief_bills_due", tool_class=_BillsTool
            ),
        ]
    )
    return SimpleNamespace(package=package, registry=registry)


def test_count_if_counts_only_the_items_that_count_but_renders_them_all() -> None:
    """A card in credit shows in the section but is not a bill in the push ("Bills 1")."""
    setup = _bills_brief("owed")
    rendered = render_brief_result(setup.package, setup.registry)
    assert "- Discover Card: USD 35.00" in rendered.body
    assert "- Wingtip: INR 3,150.40 credit — nothing to pay" in rendered.body
    assert rendered.counts == (("Bills due", 1),)
    (section,) = rendered.sections
    assert (section.items, section.empty) == (1, False)


def test_without_count_if_every_bullet_counts() -> None:
    setup = _bills_brief(None)
    rendered = render_brief_result(setup.package, setup.registry)
    assert rendered.counts == (("Bills due", 2),)


def test_a_section_of_only_credits_still_shows_but_counts_nothing() -> None:
    class _CreditOnly(_BillsTool):
        def _run(self) -> list[dict]:
            return [{"name": "Wingtip", "amount": "INR 5 credit", "owed": ""}]

    setup = _bills_brief("owed")
    registry = _fake_registry(
        [
            setup.package,
            _fake_package(
                skill_name="finance-brief", tool_name="brief_bills_due", tool_class=_CreditOnly
            ),
        ]
    )
    rendered = render_brief_result(setup.package, registry)
    (section,) = rendered.sections
    assert (section.items, section.empty) == (0, False)
    assert "- Wingtip: INR 5 credit" in rendered.body


def test_handler_delivers_a_partial_digest_and_records_the_failure(digest_dir: Path) -> None:
    setup = _setup_with(stocks=_TimeoutStocksTool)
    gateway = ChannelGateway()
    console = _RecordingConnector("console")
    gateway.register(console)
    runtime = SimpleNamespace(
        skill_registry=setup.registry, channels=gateway, default_channel="console"
    )
    run = _run(runtime, "console", source_preferences=["reminders", "stocks"])

    assert run.status is HeartbeatStatus.SUCCESS
    assert "partial: ⚠ couldn't build: Stocks" in run.error
    assert len(console.sent) == 1 and "Buy milk" in console.sent[0].body

    stored = shared_digest_store().latest()
    assert stored is not None
    assert stored.body == run.output
    assert stored.heartbeat == "morning-digest" and stored.skill_id == "multi-brief"
    assert [f.name for f in stored.failed_sections] == ["stocks"]
    assert console.sent[0].metadata["digest_id"] == stored.id
    assert (digest_dir / "digests.db").exists()


def test_telegram_gets_the_digest_in_chunks_sent_in_order(digest_dir: Path) -> None:
    setup = _setup_with(stocks=_ManyItemsTool)
    gateway = ChannelGateway()
    telegram = _RecordingConnector("telegram")
    gateway.register(telegram)
    runtime = SimpleNamespace(
        skill_registry=setup.registry, channels=gateway, default_channel="telegram"
    )
    run = _run(runtime, "telegram", source_preferences=["reminders", "stocks"])

    assert run.status is HeartbeatStatus.SUCCESS
    assert len(telegram.sent) > 1
    assert all(len(m.body) <= 4000 for m in telegram.sent)
    assert all(m.metadata["parse_mode"] == "HTML" for m in telegram.sent)
    joined = "\n".join(m.body for m in telegram.sent)
    numbers = [int(n) for n in re.findall(r"STOCK(\d{3})", joined)]
    assert numbers == list(range(120))


def test_web_push_gets_a_headline_that_links_to_the_stored_copy(digest_dir: Path) -> None:
    setup = _setup_with(stocks=_TimeoutStocksTool)
    gateway = ChannelGateway()
    push = _RecordingConnector("web_push")
    gateway.register(push)
    runtime = SimpleNamespace(
        skill_registry=setup.registry, channels=gateway, default_channel="web_push"
    )
    run = _run(runtime, "web_push", source_preferences=["reminders", "stocks"])

    assert run.status is HeartbeatStatus.SUCCESS
    [message] = push.sent
    assert message.body == "Reminders: 2\n⚠ partial: Stocks failed"
    assert message.subject == "Multi"
    stored = shared_digest_store().latest()
    assert stored is not None
    assert message.metadata["url"] == f"/digest/{stored.id}"


def test_channel_all_skips_the_console_and_tolerates_a_push_nobody_subscribed_to(
    digest_dir: Path,
) -> None:
    setup = _setup_with()
    gateway = ChannelGateway()
    console = _RecordingConnector("console")
    telegram = _RecordingConnector("telegram")
    push = _RecordingConnector("web_push", status=DeliveryStatus.SKIPPED)
    for connector in (console, telegram, push):
        gateway.register(connector)
    runtime = SimpleNamespace(
        skill_registry=setup.registry, channels=gateway, default_channel="console"
    )
    run = _run(runtime, "all", source_preferences=["reminders", "stocks"])

    assert console.sent == []  # D5: `all` never includes the console
    assert len(telegram.sent) == 1
    assert len(push.sent) == 1
    assert run.status is HeartbeatStatus.SUCCESS
    assert "web_push: skipped" in run.error


def test_channel_all_fails_when_a_real_channel_fails(digest_dir: Path) -> None:
    setup = _setup_with()
    gateway = ChannelGateway()
    gateway.register(_RecordingConnector("telegram", status=DeliveryStatus.FAILED))
    gateway.register(_RecordingConnector("web_push"))
    runtime = SimpleNamespace(
        skill_registry=setup.registry, channels=gateway, default_channel="telegram"
    )
    run = _run(runtime, "all", source_preferences=["reminders"])
    assert run.status is HeartbeatStatus.FAILED
    assert run.error.startswith("telegram:")


def test_action_links_stay_in_the_stored_copy_and_leave_every_channel(digest_dir: Path) -> None:
    setup = _setup_with(reminders=_FocusTool)
    gateway = ChannelGateway()
    stream = StringIO()
    gateway.register(ConsoleConnector(name="console", stream=stream))
    telegram = _RecordingConnector("telegram")
    gateway.register(telegram)
    runtime = SimpleNamespace(
        skill_registry=setup.registry, channels=gateway, default_channel="console"
    )
    _run(runtime, "console", source_preferences=["reminders"])
    _run(runtime, "telegram", source_preferences=["reminders"])

    stored = shared_digest_store().latest()
    assert stored is not None
    assert "[👎](iris:not-useful/offers%40fabrikam.test)" in stored.body
    assert "iris:" not in stream.getvalue() and "👎" not in stream.getvalue()
    assert all("iris:" not in m.body and "👎" not in m.body for m in telegram.sent)
    assert "Fabrikam · Travel smarter" in telegram.sent[0].body


class _TextFocusTool(BaseTool):
    name: str = "email_focus"
    description: str = "fake text slot that writes its own heading and bullets"
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        return "## Focus — personal, newest 5\n- Mo Harbor School · Saturday\n- Northwind Bank · RM call"

    async def _arun(self) -> str:
        return self._run()


class _LearnedTool(BaseTool):
    name: str = "learned_yesterday"
    description: str = "fake footer line"
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        return "learned yesterday: nothing"

    async def _arun(self) -> str:
        return self._run()


def _footer_setup(stocks: type[BaseTool] = _FakeStocksTool) -> SimpleNamespace:
    """A brief whose last slot is a declared footer (the digest's learned-yesterday line)."""
    brief = BriefSpec(
        subject="Digest",
        uses=("finance", "email-triage", "iris-core"),
        layout="Good morning.\n\n## Stocks\n{{stocks}}\n\n{{focus}}\n\n{{learned}}",
        footer_slots=("learned",),
        slots={
            "stocks": BriefToolSlot(
                kind="tool",
                skill="finance",
                tool="fetch_stocks",
                format="bullets",
                empty="none",
                item_template="{sym}",
            ),
            "focus": BriefToolSlot(
                kind="tool", skill="email-triage", tool="email_focus", format="text", empty="none"
            ),
            "learned": BriefToolSlot(
                kind="tool",
                skill="iris-core",
                tool="learned_yesterday",
                format="text",
                empty="learned yesterday: nothing",
            ),
        },
    )
    registry = _fake_registry(
        [
            _fake_brief_package(brief=brief, name="footer-brief"),
            _fake_package(skill_name="finance", tool_name="fetch_stocks", tool_class=stocks),
            _fake_package(
                skill_name="email-triage", tool_name="email_focus", tool_class=_TextFocusTool
            ),
            _fake_package(
                skill_name="iris-core", tool_name="learned_yesterday", tool_class=_LearnedTool
            ),
        ]
    )
    package = next(p for p in registry.list_packages() if p.manifest.name == "footer-brief")
    return SimpleNamespace(registry=registry, package=package)


def test_the_failure_line_goes_before_the_footer_slot() -> None:
    setup = _footer_setup(stocks=_TimeoutStocksTool)
    body = render_brief_result(
        setup.package, setup.registry, section_keys=("stocks", "focus", "learned")
    ).body
    assert body.index("- Northwind Bank") < body.index("⚠ couldn't build: Stocks")
    assert body.index("⚠ couldn't build") < body.index("learned yesterday: nothing")
    assert body.rstrip().endswith("learned yesterday: nothing")


def test_the_footer_slot_stays_last_whatever_the_section_order() -> None:
    setup = _footer_setup()
    body = render_brief_result(
        setup.package,
        setup.registry,
        section_keys=("stocks", "focus", "learned"),
        section_order=("learned", "focus", "stocks"),
    ).body
    assert body.rstrip().endswith("learned yesterday: nothing")


def test_a_text_section_with_bullets_counts_toward_the_push_headline() -> None:
    setup = _footer_setup()
    rendered = render_brief_result(
        setup.package, setup.registry, section_keys=("stocks", "focus", "learned")
    )
    # Focus writes its own heading (no layout heading), so it is named by slot;
    # the footer renders no bullets and is not counted.
    assert rendered.counts == (("Stocks", 3), ("Focus", 2))


def test_footer_slots_must_name_real_slots() -> None:
    with pytest.raises(ValueError, match="footer_slots"):
        BriefSpec(
            subject="x",
            layout="{{a}}",
            slots={"a": BriefLiteralSlot(kind="literal", value="a")},
            footer_slots=("nope",),
        )


# ---------------------------------------------------------------------------
# Digest v5: structured sections, and the grouped rendering per channel
# ---------------------------------------------------------------------------


def test_render_exposes_the_sections_in_render_order() -> None:
    setup = _footer_setup(stocks=_TimeoutStocksTool)
    rendered = render_brief_result(
        setup.package, setup.registry, section_keys=("stocks", "focus", "learned")
    )
    assert [(s.name, s.footer) for s in rendered.sections] == [
        ("stocks", False),
        ("focus", False),
        ("learned", True),
    ]
    stocks, focus, learned = rendered.sections
    assert stocks.failed == "research engine timeout after 20 s" and stocks.title == "Stocks"
    # A slot that writes its own heading is titled by it; the text is what follows.
    assert focus.title == "Focus — personal, newest 5"
    assert focus.text == "- Mo Harbor School · Saturday\n- Northwind Bank · RM call"
    assert (focus.items, focus.empty) == (2, False)
    assert learned.empty and learned.text == "learned yesterday: nothing"
    assert rendered.greeting == "Good morning."
    assert rendered.closing == "learned yesterday: nothing"


def test_a_section_that_renders_its_empty_text_is_empty() -> None:
    setup = _setup_with(reminders=_EmptyTool)
    rendered = render_brief_result(
        setup.package, setup.registry, section_keys=("reminders", "stocks")
    )
    reminders = next(s for s in rendered.sections if s.name == "reminders")
    assert reminders.empty and reminders.items == 0
    stocks = next(s for s in rendered.sections if s.name == "stocks")
    assert not stocks.empty and stocks.items == 3


def _digest_settings_with_groups() -> object:
    from iris_harness.services.digest.settings import DigestGroup, DigestSettings

    return DigestSettings(
        groups=(
            DigestGroup(
                id="inbox", title="Inbox", icon="📬", sections=("focus",), push="{focus} in Focus"
            ),
            DigestGroup(id="markets", title="Markets", icon="📈", sections=("stocks",)),
        ),
        channels={
            "telegram": {"markets": "expandable_card", "buttons": ["full_digest", "settings"]},
            "web_push": {"lines": 3, "exclude_groups": ["markets"]},
        },
        footer_sections=("learned",),
    )


@pytest.fixture()
def grouped_runtime(
    digest_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SimpleNamespace, dict[str, _RecordingConnector]]:
    from iris_harness.runtime.handlers import skill_brief

    monkeypatch.setattr(
        skill_brief, "_digest_settings", lambda runtime: _digest_settings_with_groups()
    )
    monkeypatch.delenv("IRIS_PUBLIC_URL", raising=False)
    setup = _footer_setup()
    gateway = ChannelGateway()
    connectors = {name: _RecordingConnector(name) for name in ("console", "telegram", "web_push")}
    for connector in connectors.values():
        gateway.register(connector)
    runtime = SimpleNamespace(
        skill_registry=setup.registry, channels=gateway, default_channel="console"
    )
    return runtime, connectors


def _run_digest(runtime: SimpleNamespace, channel: str = "all", **params: object):
    handler = _build_skill_brief_handler(runtime)
    return handler(
        HeartbeatDefinition(
            name="morning-digest",
            handler="skill_brief",
            schedule="manual",
            params={
                "channel": channel,
                "skill_id": "footer-brief",
                "source_preferences": ["stocks", "focus", "learned"],
                "digest": True,
                **params,
            },
        )
    )


def test_the_digest_is_stored_grouped(grouped_runtime) -> None:
    runtime, _ = grouped_runtime
    run = _run_digest(runtime)
    assert run.status is HeartbeatStatus.SUCCESS
    stored = shared_digest_store().latest()
    assert stored is not None and stored.body == run.output
    assert stored.body == (
        "Good morning.\n\n"
        "## 📬 Inbox\n\n"
        "### Focus — personal, newest 5\n- Mo Harbor School · Saturday\n- Northwind Bank · RM call\n\n"
        "## 📈 Markets\n\n"
        "### Stocks\n- AAA\n- BBB\n- CCC\n\n"
        "---\n\n"
        "learned yesterday: nothing"
    )


def test_telegram_gets_one_message_per_group_and_no_buttons_without_a_base_url(
    grouped_runtime,
) -> None:
    runtime, connectors = grouped_runtime
    _run_digest(runtime, "telegram")
    bodies = [m.body for m in connectors["telegram"].sent]
    assert bodies == [
        "<b>Good morning.</b>",
        "<b>📬 INBOX</b>\n<blockquote><b>Focus — personal, newest 5</b>\n"
        "• Mo Harbor School · Saturday\n• Northwind Bank · RM call</blockquote>",
        "<b>📈 MARKETS</b>\n<blockquote expandable><b>Stocks</b>\n"
        "• AAA\n• BBB\n• CCC</blockquote>"
        "\n\n<i>learned yesterday: nothing</i>",
    ]
    assert all(m.metadata["parse_mode"] == "HTML" for m in connectors["telegram"].sent)
    assert all("inline_keyboard" not in m.metadata for m in connectors["telegram"].sent)


def test_the_last_telegram_message_links_the_stored_copy_and_settings(
    grouped_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, connectors = grouped_runtime
    monkeypatch.setenv("IRIS_PUBLIC_URL", "https://iris.example/")
    _run_digest(runtime, "telegram")
    stored = shared_digest_store().latest()
    assert stored is not None
    *first, last = connectors["telegram"].sent
    assert last.metadata["inline_keyboard"] == [
        [
            {"text": "📄 Full digest", "url": f"https://iris.example/digest/{stored.id}"},
            {"text": "⚙️ Digest settings", "url": "https://iris.example/settings#digest"},
        ]
    ]
    assert all("inline_keyboard" not in m.metadata for m in first)


def test_a_base_url_that_is_not_http_gives_no_buttons(
    grouped_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, connectors = grouped_runtime
    monkeypatch.setenv("IRIS_PUBLIC_URL", "iris.example")
    _run_digest(runtime, "telegram")
    assert all("inline_keyboard" not in m.metadata for m in connectors["telegram"].sent)


def test_push_gets_the_group_count_lines(grouped_runtime) -> None:
    runtime, connectors = grouped_runtime
    _run_digest(runtime, "web_push")
    [message] = connectors["web_push"].sent
    assert message.body == "📬 2 in Focus"
    stored = shared_digest_store().latest()
    assert stored is not None and message.metadata["url"] == f"/digest/{stored.id}"


def test_a_partial_grouped_digest_names_the_failure_everywhere(
    digest_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.runtime.handlers import skill_brief

    monkeypatch.setattr(
        skill_brief, "_digest_settings", lambda runtime: _digest_settings_with_groups()
    )
    setup = _footer_setup(stocks=_TimeoutStocksTool)
    gateway = ChannelGateway()
    telegram, push = _RecordingConnector("telegram"), _RecordingConnector("web_push")
    gateway.register(telegram)
    gateway.register(push)
    runtime = SimpleNamespace(skill_registry=setup.registry, channels=gateway, default_channel="x")
    run = _run_digest(runtime)
    markets = run.output[run.output.index("## 📈 Markets") :]
    assert markets.startswith(
        "## 📈 Markets\n\n⚠ couldn't build: Stocks (research engine timeout after 20 s)"
    )
    assert any("<b>⚠ Stocks</b>" in m.body for m in telegram.sent)
    assert push.sent[0].body == "📬 2 in Focus\n⚠ partial: Stocks failed"
    assert "partial: ⚠ couldn't build: Stocks" in run.error


def test_a_brief_that_is_not_the_digest_stays_flat(grouped_runtime) -> None:
    runtime, connectors = grouped_runtime
    run = _run_digest(runtime, "telegram", digest=False)
    assert "## Stocks" in run.output and "## 📈" not in run.output
    assert "<blockquote" not in "".join(m.body for m in connectors["telegram"].sent)


def test_a_digest_with_no_groups_configured_stays_flat(
    grouped_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.runtime.handlers import skill_brief
    from iris_harness.services.digest.settings import DigestSettings

    monkeypatch.setattr(skill_brief, "_digest_settings", lambda runtime: DigestSettings())
    runtime, _ = grouped_runtime
    run = _run_digest(runtime, "console")
    assert "## Stocks" in run.output


def test_a_news_group_is_titled_by_the_digest_settings(
    grouped_runtime: tuple[SimpleNamespace, dict[str, _RecordingConnector]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest's generic heading gives way to Settings -> Digest's title."""
    from dataclasses import replace as dc_replace

    from iris_harness.runtime.handlers import skill_brief

    settings = dc_replace(
        _digest_settings_with_groups(),  # type: ignore[type-var]
        news_groups={"stocks": {"title": "Local — {news_local_area}", "topics": ("x",)}},
        news_local_area="St. Louis",
    )
    monkeypatch.setattr(skill_brief, "_digest_settings", lambda runtime: settings)
    runtime, connectors = grouped_runtime

    run = _run_digest(runtime, channel="telegram")

    assert "### Local — St. Louis" in (run.output or "")
    sent = "\n".join(m.body for m in connectors["telegram"].sent)
    assert "<b>Local — St. Louis</b>" in sent


# ---------------------------------------------------------------------------
# A slot whose skill is not installed is absent, not failed
# ---------------------------------------------------------------------------


def _registry_without(skill_name: str, setup: SimpleNamespace) -> SimpleNamespace:
    """The multi-section brief's registry with one skill not installed at all."""
    return _fake_registry(
        [p for p in setup.registry.list_packages() if p.manifest.name != skill_name]
    )


@pytest.mark.parametrize("section_keys", [None, ("reminders", "stocks")])
def test_a_section_whose_skill_is_not_installed_is_left_out_not_failed(
    section_keys: tuple[str, ...] | None,
) -> None:
    setup = _setup_with()
    registry = _registry_without("finance", setup)

    rendered = render_brief_result(setup.package, registry, section_keys=section_keys)

    assert "- Buy milk" in rendered.body
    assert "## Stocks" not in rendered.body
    assert "couldn't build" not in rendered.body
    assert rendered.failed == ()
    assert [s.name for s in rendered.sections] == ["reminders"]
    assert rendered.counts == (("Reminders", 2),)
    assert _STOCKS_INVOKED["count"] == 0


def test_an_installed_skill_that_cannot_serve_its_slot_still_fails() -> None:
    """Installed-but-broken is loud: a package that failed to load keeps the failure."""
    setup = _setup_with()
    packages = setup.registry.list_packages()
    broken = packages[2].model_copy(update={"missing_prerequisites": ("FINANCE_API_KEY",)})
    assert broken.manifest.name == "finance" and not broken.is_loadable
    registry = _fake_registry([*packages[:2], broken])

    rendered = render_brief_result(setup.package, registry)

    assert [(f.name, f.reason) for f in rendered.failed] == [
        ("stocks", "tool not found: finance.fetch_stocks")
    ]
    assert "⚠ couldn't build: Stocks" in rendered.body


def test_a_tool_that_raises_in_an_installed_skill_still_fails_beside_an_absent_one() -> None:
    setup = _setup_with(reminders=_BoomReminderTool)
    registry = _registry_without("finance", setup)

    rendered = render_brief_result(setup.package, registry)

    assert [f.name for f in rendered.failed] == ["reminders"]
    assert "⚠ couldn't build: Reminders (RuntimeError)" in rendered.body
    assert "Stocks" not in rendered.body


class _AnyArgs(BaseModel):
    model_config = {"extra": "allow"}


def _empty_tool(tool_name: str) -> type[BaseTool]:
    """A tool named ``tool_name`` that accepts any args and returns nothing."""

    def _run(self: BaseTool, **_kwargs: object) -> list[dict]:
        return []

    return type(
        f"_Empty_{tool_name}",
        (BaseTool,),
        {
            "__module__": __name__,
            "__qualname__": f"_Empty_{tool_name}",
            "__annotations__": {"name": str, "description": str, "args_schema": type[BaseModel]},
            "name": tool_name,
            "description": "empty",
            "args_schema": _AnyArgs,
            "_run": _run,
        },
    )


_REPO_ROOT = Path(__file__).resolve().parents[5]


def _real_morning_briefing(installed: set[str] | None) -> tuple[SkillPackage, SimpleNamespace]:
    """The shipped morning-briefing brief over fake skills: only ``installed`` exist
    (``None``: every skill the brief uses).

    Every installed skill carries every tool the brief's slots ask of it, each
    returning nothing, so every section it serves renders its ``empty`` text.
    """
    from iris_harness.tools.skills.loader import load_skill_manifest

    manifest = load_skill_manifest(
        _REPO_ROOT / "config" / "skills" / "builtin" / "morning-briefing"
    )
    assert manifest.brief is not None
    brief = SkillPackage(
        manifest=manifest,
        skill_dir=Path("/fake/brief"),
        tools_module_path=Path("/fake/brief/tools.py"),
    )
    tools: dict[str, list[str]] = {}
    for slot in manifest.brief.slots.values():
        if isinstance(slot, BriefToolSlot) and slot.tool not in tools.setdefault(slot.skill, []):
            tools[slot.skill].append(slot.tool)
    packages = [brief]
    for skill, names in tools.items():
        if installed is not None and skill not in installed:
            continue
        packages.append(
            SkillPackage(
                manifest=SkillManifest(
                    name=skill,
                    version="0.1.0",
                    description="fake",
                    author="iris",
                    license="Apache-2.0",
                    tools=tuple(
                        SkillToolManifest(name=n, description="fake", governor_route="system/read")
                        for n in names
                    ),
                    requires=SkillRequirements(),
                ),
                skill_dir=Path(f"/fake/{skill}"),
                tools_module_path=Path(f"/fake/{skill}/tools.py"),
                tool_classes=tuple(_empty_tool(n) for n in names),
            )
        )
    return brief, _fake_registry(packages)


def test_the_morning_digest_with_every_skill_installed_renders_every_section() -> None:
    """The private build's shape: every skill the brief uses is installed, so nothing is
    left out -- every tool section of the layout renders, in layout order."""
    package, registry = _real_morning_briefing(None)
    brief = package.manifest.brief
    assert brief is not None
    tool_slots = [
        name
        for name in dict.fromkeys(re.findall(r"{{\s*([A-Za-z_]\w*)\s*}}", brief.layout))
        if isinstance(brief.slots.get(name), BriefToolSlot)
    ]

    rendered = render_brief_result(package, registry)

    assert rendered.failed == ()
    assert [s.name for s in rendered.sections] == tool_slots
    assert "## Today's plan\nNothing scheduled today." in rendered.body
    assert "## Bills due\nNo bills due." in rendered.body
    assert "## Portfolio\nNo holdings tracked yet." in rendered.body


def test_the_morning_digest_without_the_domain_skills_leaves_them_out_silently() -> None:
    """The public build's shape: only the core's own skills are installed."""
    core = {"iris-core", "iris-tasks", "web-fetch"}
    package, registry = _real_morning_briefing(core)
    brief = package.manifest.brief
    assert brief is not None

    rendered = render_brief_result(package, registry)

    assert rendered.failed == ()
    assert "couldn't build" not in rendered.body
    kept = [s.name for s in rendered.sections]
    skills = {
        name: slot.skill for name, slot in brief.slots.items() if isinstance(slot, BriefToolSlot)
    }
    assert kept and {skills[name] for name in kept} == core
    for heading in ("## Today's plan", "## Bills due", "## Inbox summary", "## Portfolio"):
        assert heading not in rendered.body
    assert "## Open tasks\nNo open tasks." in rendered.body
    assert rendered.body.rstrip().endswith("learned yesterday: nothing")
