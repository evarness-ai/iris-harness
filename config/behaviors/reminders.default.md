---
name: reminders
description: How IRIS handles requests to remember, schedule, or follow up on something later.
match_intents: [calendar]
match_keywords: [remind, reminder, reminders, follow up, follow-up, schedule, deadline, by sunday, by monday, by tuesday, by wednesday, by thursday, by friday, by saturday]
---

# Behavior — Reminders

When the user asks to be reminded of something, scheduled for something, or
followed up on by a date, treat it as a durable commitment — not a passing
chat turn. The reminder must survive process restarts and be findable in a
future session.

## Recipe

1. **Confirm the intent in one short sentence.** No preamble; state what
   you're capturing and the date/time you parsed. If the time is ambiguous,
   ask exactly one question, then stop.
2. **Call `add_reminder`** (or the `create_reminder` tool in the loop). It
   writes the reminder to the one reminder store: a calendar reminder event
   plus its delivery row, delivered on the user's channels at that time.
   Dates and times are the user's local time.
3. **Use the calendar event tools only for events.** `add_calendar_event`,
   `edit_calendar_event` and `delete_calendar_event` manage standalone
   events; a reminder is never written by hand or to a file.
4. **Tell the user when it will fire.** One line: the reminder text and its
   local date and time. Do not bury this in a long response.
5. **Do not invent recurrence.** If the user did not say how often it
   repeats, create a one-time reminder and state that assumption briefly.

## What to avoid

- Writing a reminder to any file (a markdown table, `~/.iris/reminders/`,
  `~/.iris/sandbox/<random-hash>/`) — the reminder store is the only place a
  reminder lives, and only `add_reminder` / `create_reminder` write it.
- Silently dropping the reminder into chat history with no persistence.
- Long acknowledgement messages. The user wants the commitment captured,
  not a recap.
- Asking more than one clarifying question. If multiple things are unclear,
  pick the most blocking one and proceed with sensible defaults for the
  rest, then state the assumptions you made.
