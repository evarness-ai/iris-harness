"""Governed tools: a shared noticeboard the model can read, pin and clear.

Each tool is declared in ``manifest.yaml``, and the declaration -- not the code --
decides how it is governed:

* ``list_notes`` -- ``effect: read``, ``content: external``: other people write the
  notes, so every result is marked untrusted and tripwire-scanned before the model sees it;
* ``pin_note`` -- ``effect: write``, ``confirm: once``: a change, asked about once;
* ``remove_note`` -- ``effect: destructive``: nothing is removed until the owner approves
  the exact call on an approval card, which ``describe`` fills in from the board.

Every call, whoever makes it (the model in a chat turn, or plugin code through
``api.tools``), goes through the same governed runner: ``PRE_TOOL_USE`` checks, the
approval rules, ``POST_TOOL_USE`` checks, and a row in the audit ledger for each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.types import ToolDescription


@dataclass
class Note:
    note_id: str
    author: str
    text: str
    pinned: bool = False


@dataclass
class Board:
    """The board's notes. A real plugin keeps them in its own store."""

    notes: dict[str, Note] = field(default_factory=dict)

    @classmethod
    def sample(cls) -> Board:
        rows = [
            Note("n1", "Petra", "Bake sale on Friday, bring a tray."),
            Note("n2", "Marcus", "Dentist moved the check-up to 14:30."),
            Note("n3", "Lena", "The printer on floor 2 is fixed."),
        ]
        return cls({note.note_id: note for note in rows})


def _note_id(args: dict[str, Any]) -> str:
    return str(args.get("note_id", "")).strip()


def setup(api: PluginAPI, board: Board | None = None) -> None:
    board = board if board is not None else Board.sample()

    def list_notes(args: dict[str, Any]) -> str:
        if not board.notes:
            return "The board is empty."
        return "\n".join(
            f"{n.note_id} ({n.author}){' [pinned]' if n.pinned else ''}: {n.text}"
            for n in board.notes.values()
        )

    def pin_note(args: dict[str, Any]) -> str:
        note = board.notes[_note_id(args)]
        note.pinned = True
        return f"Pinned {note.note_id}: {note.text}"

    def remove_note(args: dict[str, Any]) -> str:
        note = board.notes.pop(_note_id(args))
        return f"Removed {note.note_id}: {note.text}"

    def known_note(args: dict[str, Any]) -> str | None:
        """Checked before any approval is queued: the model sees why a call cannot run."""
        if _note_id(args) not in board.notes:
            return f"There is no note {_note_id(args)!r}; call list_notes for the ids."
        return None

    def describe_removal(args: dict[str, Any]) -> ToolDescription:
        """The approval card, in the owner's words rather than an id."""
        note = board.notes[_note_id(args)]
        return ToolDescription(title="Remove 1 note", lines=(f"{note.author}: {note.text}",))

    api.register_tool(
        "list_notes",
        "List the notes on the shared noticeboard, with their ids.",
        list_notes,
    )
    api.register_tool(
        "pin_note",
        'Pin a note to the top of the board. Args: {"note_id": str}.',
        pin_note,
        validate=known_note,
    )
    api.register_tool(
        "remove_note",
        'Remove a note from the board for good. Args: {"note_id": str}.',
        remove_note,
        describe=describe_removal,
        validate=known_note,
    )
