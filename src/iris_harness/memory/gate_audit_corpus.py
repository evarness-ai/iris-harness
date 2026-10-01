"""A built-in, synthetic, labelled corpus for the memory garbage-in/out audit.

Every entry is invented (placeholder names/domains, never real user data) and labelled
``keep`` (a real durable fact that *should* be admitted + recalled) or ``junk`` (pollution
that *should* be blocked). The set deliberately exercises each capture gate's reject path
AND the defence-in-depth seam: a few junk facts are structurally fine (they pass all three
gates) but carry low confidence, so only the recall filter can catch them.

Pass a custom corpus (JSON list of the same fields) to the CLI to audit real labelled data;
this default makes the harness and its tests run out of the box.
"""

from __future__ import annotations

from iris_harness.memory.gate_audit import FactCandidate

# fmt: off
DEFAULT_CORPUS: list[FactCandidate] = [
    # ---- keep: real durable facts that should pass the gates + reach the prompt ----
    FactCandidate("name", "Anita Rao", "my name is anita rao", 0.92, "keep"),
    FactCandidate("blog", "web3notes.example", "i blog at web3notes.example", 0.85, "keep"),
    FactCandidate("city", "Springfield", "i live in springfield", 0.88, "keep"),
    FactCandidate("employer", "Quant Academy", "i work at quant academy", 0.80, "keep"),
    FactCandidate("language", "Spanish", "i am learning spanish", 0.70, "keep"),
    FactCandidate("instrument", "piano", "i play the piano", 0.55, "keep"),  # uncertain band
    FactCandidate("sport", "tennis", "i play tennis sometimes", 0.38, "keep"),  # just above drop
    # keep that the (conservative) grounding gate OVER-blocks: the user said "nyc",
    # the extractor normalised to "New York City", whose tokens aren't in the message.
    FactCandidate("city", "New York City", "i moved to nyc last year", 0.80, "keep"),
    # origin is its own fact, not a second residence (memris plan, ADR-0115)
    FactCandidate("hometown", "Chennai", "i grew up in chennai", 0.85, "keep"),
    FactCandidate("nationality", "Indian", "my nationality is indian", 0.90, "keep"),

    # ---- junk blocked by the PLAUSIBILITY gate (sentence / first-person) ----
    FactCandidate("note", "i really need to reorganize my whole schedule this week",
                  "i really need to reorganize my whole schedule this week", 0.60, "junk"),
    FactCandidate("hobby", "i love hiking on weekends", "i love hiking on weekends", 0.50, "junk"),

    # ---- junk blocked by the DURABILITY gate (ephemeral key / greeting / clock / relative) ----
    FactCandidate("greeting", "hello", "hello there", 0.60, "junk"),
    FactCandidate("reminder_time", "5pm", "remind me at 5pm", 0.70, "junk"),
    FactCandidate("mood", "happy", "i feel happy", 0.50, "junk"),
    FactCandidate("day", "tomorrow", "let's meet tomorrow", 0.60, "junk"),

    # ---- junk blocked by the GROUNDING gate (value not in the message — hallucinated) ----
    FactCandidate("profession", "teacher", "show me my daily brief", 0.60, "junk"),
    FactCandidate("stock", "apple", "what is the weather today", 0.50, "junk"),

    # ---- junk that PASSES all gates but is low-confidence pollution (recall filter's job) ----
    FactCandidate("snack", "chips", "i grabbed some chips", 0.20, "junk"),
    FactCandidate("color", "teal", "the teal one looks nice", 0.25, "junk"),
    FactCandidate("gadget", "speaker", "i bought a speaker", 0.30, "junk"),
    # slips the recall filter too (conf >= 0.35) — the leak the threshold sweep targets:
    FactCandidate("impulse", "lamp", "i bought a lamp", 0.42, "junk"),

    # ---- junk taken from the REAL store (2026-09-17), with the message shape that
    # produced it. These are why confidence stopped being treated as consent: every
    # one of them was stored, several at 0.9-1.0, from content the user was merely
    # discussing or configuring. ----
    # the SUBJECT of the conversation, mistaken for a property of the user:
    FactCandidate("topic", "murder plot", "i read about the murder plot case", 0.60, "junk"),
    FactCandidate("source", "CBS News", "i saw it on CBS News", 0.90, "junk"),
    FactCandidate("number", "4", "i need 4 of them", 1.00, "junk"),
    FactCandidate("error", "it shows error", "i ran it and it shows error", 0.60, "junk"),
    FactCandidate("folder", "one folder", "i put them in one folder", 0.60, "junk"),
    FactCandidate("file_extension", "png", "i saved my chart as png", 1.00, "junk"),
    # mined from pasted/third-party content — no first person of the user's own:
    FactCandidate("employer", "Department of Justice",
                  "the Department of Justice said the charges were filed", 0.90, "junk"),
    FactCandidate("alleged wrongdoing", "Russian intelligence agents",
                  "the report names Russian intelligence agents", 0.60, "junk"),
    # the hardest one: an allowed key, a first-person message, a value present in it.
    # The store really held `name=ollama` at 1.00. No capture gate can tell this from
    # a real name — owner confirmation is what keeps it out of a prompt, which is the
    # whole reason confirmation exists.
    FactCandidate("name", "ollama", "i am running ollama for local models", 1.00, "junk"),
    # a tool/config value read as an identity fact:
    FactCandidate("provider", "ollama", "i set the provider to ollama", 1.00, "junk"),
    FactCandidate("environment_variable", ".env file contents",
                  "i checked my .env file contents", 0.60, "junk"),
    # fragments: an allowed key with a value that says nothing:
    FactCandidate("insurance", "dues", "i asked about my insurance dues", 0.60, "junk"),
    FactCandidate("credit_cards", "cards", "i have some cards", 0.30, "junk"),

    # ---- keep: the real facts from the same store that SHOULD survive ----
    FactCandidate("profession", "solutions architect",
                  "i work as a solutions architect", 0.90, "keep"),
    FactCandidate("country", "India", "i live in India", 0.90, "keep"),
    FactCandidate("research_area", "graph-based ontological systems",
                  "my research area is graph-based ontological systems", 0.90, "keep"),
    FactCandidate("interest", "cybersecurity", "i am interested in cybersecurity", 0.60, "keep"),
]
# fmt: on


__all__ = ["DEFAULT_CORPUS"]
