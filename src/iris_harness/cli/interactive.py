"""Arrow-key interactive UI primitives for the IRIS CLI.

Thin wrappers around prompt_toolkit dialog shortcuts that apply a consistent
IRIS dark theme. All dialogs block until the user makes a choice.
"""

from __future__ import annotations

from prompt_toolkit.styles import Style

_DIALOG_STYLE = Style.from_dict(
    {
        "dialog": "bg:#0f0f1a",
        "dialog.body": "bg:#0f0f1a fg:#c8c8d8",
        "dialog shadow": "bg:#000000",
        "dialog frame.label": "fg:#4169e1 bold",
        "button": "bg:#1a2a4a fg:#c8c8d8",
        "button.focused": "bg:#4169e1 fg:#ffffff bold",
        "button.arrow": "fg:#4169e1",
        "radio-list": "bg:#0f0f1a",
        "radio": "fg:#555577",
        "radio-selected": "fg:#00ced1 bold",
        "radio-checked": "fg:#4169e1",
        "text-area": "bg:#1a1a2e fg:#c8c8d8",
        "text-area.prompt": "fg:#4169e1",
    }
)


def pick_from_list(
    title: str,
    options: list[tuple[str, str]],
    current: str | None = None,
) -> str | None:
    """Arrow-key radiolist dialog.

    Args:
        title:   Dialog title shown in the frame border.
        options: List of (value, display_label) pairs.
        current: Value to pre-select; defaults to the first option.

    Returns the selected value, or None if the user cancelled with Escape.
    """
    from prompt_toolkit.shortcuts import radiolist_dialog

    return radiolist_dialog(
        title=title,
        text="↑ ↓ navigate  ·  Enter confirm  ·  Escape cancel",
        values=options,
        default=current,
        style=_DIALOG_STYLE,
    ).run()


def confirm(question: str, default: bool = True) -> bool:
    """Yes / No confirmation dialog.

    Returns True if the user chose Yes, False otherwise (including Escape).
    """
    from prompt_toolkit.shortcuts import yes_no_dialog

    result = yes_no_dialog(
        title="Confirm",
        text=question,
        yes_text="  Yes  ",
        no_text="  No   ",
        style=_DIALOG_STYLE,
    ).run()
    return bool(result)


def prompt_text(label: str, default: str = "") -> str | None:
    """Single-line text input dialog.

    Returns the entered string, or None if the user cancelled with Escape.
    """
    from prompt_toolkit.shortcuts import input_dialog

    return input_dialog(
        title=label,
        text="Enter value  ·  Escape to cancel:",
        default=default,
        style=_DIALOG_STYLE,
    ).run()
