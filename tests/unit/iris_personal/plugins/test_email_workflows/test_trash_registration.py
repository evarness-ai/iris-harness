"""The email plugin registers trash_email with its describe (ADR-0118 step 5).

Without describe the approval card falls back to raw ids — the owner would be asked
to approve mail they cannot recognise. The manifest says it is destructive, with
restore_email as the undo."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import iris_personal.plugins.email_workflows as email_workflows
from iris_harness.runtime.plugin_host.manifest import load_manifest
from iris_personal.plugins.email_workflows import tools


class _API:
    def __init__(self, data_dir: Path) -> None:
        self.services = SimpleNamespace(
            tier_router=None,
            data_dir=data_dir,
            current_query=lambda: "",
            current_session_id=lambda: "",
            continuations=None,
        )
        self.registered: dict[str, dict[str, Any]] = {}

    def register_tool(self, name: str, description: str, call: Any, **kwargs: Any) -> None:
        self.registered[name] = kwargs


def test_trash_email_is_registered_with_its_describe(tmp_path: Path) -> None:
    api = _API(tmp_path)
    tools.register(api)  # type: ignore[arg-type]
    assert api.registered["trash_email"]["describe"] is not None
    assert api.registered["search_inbox"]["describe"] is None  # read tools have none
    # And with its validate, so a call naming ids not in the mail never becomes a card.
    assert api.registered["trash_email"]["validate"] is not None


def test_the_manifest_declares_trash_destructive_with_its_undo() -> None:
    manifest = load_manifest(Path(email_workflows.__file__).parent / "manifest.yaml")
    trash, restore = manifest.tools["trash_email"], manifest.tools["restore_email"]
    assert (trash.effect, trash.confirm_mode) == ("destructive", "approval")
    assert (trash.undo, trash.undo_window_days) == ("restore_email", 30)
    assert (restore.effect, restore.confirm_mode) == ("write", "never")


def test_every_read_tool_declares_its_output_external() -> None:
    """Each read returns text the senders wrote (subjects, snippets, bodies, attachment
    names), so the retrieved-content injection guard must scan it: it scans by the
    tool's declaration (``content: external``), never by name."""
    manifest = load_manifest(Path(email_workflows.__file__).parent / "manifest.yaml")
    reads = {name for name, decl in manifest.tools.items() if decl.effect == "read"}
    assert reads == {
        "search_inbox",
        "read_email",
        "find_attachment",
        "inbox_digest",
        "list_by_category",
        "analyze_inbox",
    }
    assert {manifest.tools[name].content for name in reads} == {"external"}
