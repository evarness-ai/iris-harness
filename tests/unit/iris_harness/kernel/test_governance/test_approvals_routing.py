"""``ChannelRouter.select`` — deliver where the request came from.

The ``channel`` column has been on every approval row since Phase 3, and until the
evaluator started stamping it there was nothing to put in it but the ``"cli"`` default.
So ``select`` could only guess, and under the API server it guessed wrong twice over:
``sys.stdin.isatty()`` is false there, so an approval raised by a *web* turn went to
Telegram if a bot happened to be configured and to the server's stderr otherwise. The
one person guaranteed not to be told was the one watching the browser.
"""

from __future__ import annotations

from pathlib import Path

from iris_harness.kernel.governance.approvals.queue import ApprovalQueue
from iris_harness.kernel.governance.approvals.router import ChannelRouter
from iris_harness.kernel.governance.approvals.store import ApprovalRow, ApprovalStore
from iris_harness.services.channels.approval_delivery import TelegramApprovalChannel


def _row(channel: str) -> ApprovalRow:
    return ApprovalRow(
        approval_id="a1",
        run_id="run-1",
        checkpoint_id="run-1:1",
        signal="goal_drift",
        context_summary="drifted",
        requested_at="2026-09-12T16:17:08+00:00",
        channel=channel,
        status="pending",
        responded_at=None,
        response_actor=None,
        timeout_at="2026-09-12T16:27:08+00:00",
        policy_on_timeout="fail_closed",
    )


def _router(
    tmp_path: Path, *, interactive: bool | None = False, telegram: bool = False
) -> ChannelRouter:
    queue = ApprovalQueue(store=ApprovalStore(db_path=tmp_path / "approvals.db"))
    remote = (
        TelegramApprovalChannel(connector=object(), chat_id="@t")  # type: ignore[arg-type]
        if telegram
        else None
    )
    return ChannelRouter(queue=queue, force_interactive=interactive, remote=remote)


# ── the origin wins ───────────────────────────────────────────────────────────


def test_a_web_approval_goes_to_the_web(tmp_path: Path) -> None:
    assert _router(tmp_path).select(_row("web")).name == "web"


def test_a_web_approval_goes_to_the_web_even_on_a_tty(tmp_path: Path) -> None:
    """The regression that matters: a halt raised in the browser must not be answered
    by a Y/n on whatever terminal happens to be attached to the server."""
    assert _router(tmp_path, interactive=True).select(_row("web")).name == "web"


def test_a_web_approval_goes_to_the_web_even_with_telegram_configured(
    tmp_path: Path,
) -> None:
    assert _router(tmp_path, telegram=True).select(_row("web")).name == "web"


def test_a_telegram_approval_goes_to_telegram(tmp_path: Path) -> None:
    assert _router(tmp_path, telegram=True).select(_row("telegram")).name == "telegram"


def test_a_telegram_approval_falls_back_when_no_bot_is_configured(tmp_path: Path) -> None:
    """Naming a channel that is not set up must not black-hole the request."""
    assert _router(tmp_path, telegram=False).select(_row("telegram")).name == "stderr"


# ── the old chain still answers for rows that name nothing useful ─────────────


def test_an_interactive_cli_turn_still_prompts(tmp_path: Path) -> None:
    assert _router(tmp_path, interactive=True).select(_row("cli")).name == "cli"


def test_a_non_interactive_cli_turn_falls_back_to_stderr(tmp_path: Path) -> None:
    assert _router(tmp_path, interactive=False).select(_row("cli")).name == "stderr"


def test_a_non_interactive_cli_turn_prefers_telegram_when_configured(
    tmp_path: Path,
) -> None:
    assert _router(tmp_path, interactive=False, telegram=True).select(_row("cli")).name == (
        "telegram"
    )


def test_an_unknown_or_empty_channel_uses_the_fallback_chain(tmp_path: Path) -> None:
    for channel in ("", "voice", "  "):
        assert _router(tmp_path, interactive=False).select(_row(channel)).name == "stderr"


def test_the_channel_is_matched_case_insensitively(tmp_path: Path) -> None:
    assert _router(tmp_path).select(_row("WEB")).name == "web"


# ── the web channel itself ────────────────────────────────────────────────────


def test_the_web_channel_never_raises(tmp_path: Path) -> None:
    """Delivery is best-effort by contract; the queue is the source of truth, and for a
    polling UI the row's existence *is* the delivery."""
    from iris_harness.kernel.governance.approvals.channels import WebChannel

    channel = WebChannel()
    assert channel.is_configured() is True
    assert channel.notify(_row("web")) is None


# ── announcing a timeout goes to the same surface as the request ──────────────


def test_a_timeout_is_announced_on_the_channel_that_was_asked(tmp_path: Path) -> None:
    """Not reusing `notify`: the request message asks for a decision that can no longer
    be made, and on Telegram it offers /approve commands that would now be refused."""
    calls: list[str] = []

    class _Recording:
        name = "web"

        def notify(self, approval: ApprovalRow) -> None:
            calls.append("notify")

        def notify_timeout(self, approval: ApprovalRow) -> None:
            calls.append("notify_timeout")

    router = _router(tmp_path)
    router._web = _Recording()  # type: ignore[assignment]

    router.notify_timeout(_row("web"))

    assert calls == ["notify_timeout"]


def test_a_channel_without_one_is_skipped_rather_than_breaking_the_sweep(
    tmp_path: Path,
) -> None:
    """`notify_timeout` is newer than the protocol; an older channel should still be
    able to deliver requests."""

    class _RequestOnly:
        name = "web"

        def notify(self, approval: ApprovalRow) -> None:
            raise AssertionError("a timeout must never be sent as a fresh request")

    router = _router(tmp_path)
    router._web = _RequestOnly()  # type: ignore[assignment]

    router.notify_timeout(_row("web"))  # must not raise


def test_a_channel_that_raises_does_not_break_the_sweep(tmp_path: Path) -> None:
    class _Broken:
        name = "web"

        def notify(self, approval: ApprovalRow) -> None: ...

        def notify_timeout(self, approval: ApprovalRow) -> None:
            raise RuntimeError("telegram is down")

    router = _router(tmp_path)
    router._web = _Broken()  # type: ignore[assignment]

    router.notify_timeout(_row("web"))  # best-effort by contract


def test_every_in_tree_channel_can_announce_a_timeout(tmp_path: Path) -> None:
    """Including the stderr fallback, which is what a headless CLI run actually hits.

    `_remote` resolves the injected transport (or the registered one) rather than a
    field, since M6.2 moved Telegram delivery out of the kernel.
    """
    router = _router(tmp_path, telegram=True)
    for channel in (router._cli, router._remote, router._web, router._fallback):
        assert callable(getattr(channel, "notify_timeout", None)), channel.name
        channel.notify_timeout(_row(channel.name))  # must not raise


# ── a web approval also goes off-box (2026-09-21) ─────────────────────────────────


class _Rec:
    def __init__(self, name: str, calls: list[str], *, configured: bool = True) -> None:
        self.name, self._calls, self._configured = name, calls, configured

    def is_configured(self) -> bool:
        return self._configured

    def notify(self, approval: ApprovalRow) -> None:
        self._calls.append(f"{self.name}:notify")

    def notify_timeout(self, approval: ApprovalRow) -> None:
        self._calls.append(f"{self.name}:timeout")


def _recording_router(
    tmp_path: Path, calls: list[str], *, configured: bool = True
) -> ChannelRouter:
    queue = ApprovalQueue(store=ApprovalStore(db_path=tmp_path / "approvals.db"))
    router = ChannelRouter(
        queue=queue,
        force_interactive=False,
        remote=_Rec("telegram", calls, configured=configured),  # type: ignore[arg-type]
    )
    router._web = _Rec("web", calls)  # type: ignore[assignment]
    return router


def test_a_web_approval_is_also_sent_to_telegram(tmp_path: Path) -> None:
    """The web channel reaches the owner only while the app is open; the owner who
    switched apps looked on Telegram, as the halt message told them to, and found
    nothing."""
    calls: list[str] = []
    _recording_router(tmp_path, calls).notify(_row("web"))
    assert calls == ["web:notify", "telegram:notify"]


def test_its_timeout_is_announced_on_both(tmp_path: Path) -> None:
    calls: list[str] = []
    _recording_router(tmp_path, calls).notify_timeout(_row("web"))
    assert calls == ["web:timeout", "telegram:timeout"]


def test_no_copy_without_a_bot(tmp_path: Path) -> None:
    calls: list[str] = []
    _recording_router(tmp_path, calls, configured=False).notify(_row("web"))
    assert calls == ["web:notify"]


def test_a_telegram_approval_is_not_sent_twice(tmp_path: Path) -> None:
    calls: list[str] = []
    _recording_router(tmp_path, calls).notify(_row("telegram"))
    assert calls == ["telegram:notify"]


def test_a_failing_copy_does_not_break_the_request(tmp_path: Path) -> None:
    calls: list[str] = []
    router = _recording_router(tmp_path, calls)

    def boom(approval: ApprovalRow) -> None:
        raise RuntimeError("telegram down")

    router._remote_override.notify = boom  # type: ignore[method-assign,union-attr]
    router.notify(_row("web"))
    assert calls == ["web:notify"]
