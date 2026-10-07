-- Side-effect ledger — records irreversible tool calls for safe resume.
-- Created by PostToolUseLedgerHook at PostToolUse, or -- for a high-risk call -- by
-- PreToolUseLedgerHook before the call runs and settled at PostToolUse; consumed by
-- `iris run resume`.
CREATE TABLE IF NOT EXISTS side_effect_ledger (
    side_effect_id   TEXT PRIMARY KEY,          -- <run_id>:<step_id>:<tool_call_id> (or a UUID)
    run_id           TEXT NOT NULL,
    step_id          INTEGER NOT NULL,
    tool             TEXT NOT NULL,             -- tool name that caused the side effect
    verification_probe TEXT NOT NULL,           -- probe name ('' = none: resume treats it as ambiguous)
    probe_metadata   TEXT NOT NULL DEFAULT '{}',-- JSON dict passed to the probe at resume time
    status           TEXT NOT NULL DEFAULT 'pending', -- pending | completed | not_completed | ambiguous | error
    completed_at     TEXT,                      -- ISO-8601 when status set to completed
    error            TEXT,                      -- exception class name if status=error (never its message)
    -- Identity (issue #134, stage 3). Nullable: a row written before the database was
    -- migrated has none. The key above stays <run>:<step>:<call_id>, so lookups by run work.
    call_id          TEXT,
    parent_call_id   TEXT,
    attempt          INTEGER,
    replay_of        TEXT,
    record_id        TEXT
);

-- Every transition of a row (pending, then settled), appended and never changed: the row above
-- is mutated in place, so on its own it is the only history there is. Identifiers and a status
-- only (no arguments, no result, no error message).
CREATE TABLE IF NOT EXISTS side_effect_events (
    event_id         TEXT PRIMARY KEY,          -- ULID, minted by the store
    side_effect_id   TEXT NOT NULL,
    status           TEXT NOT NULL,
    ts               TEXT NOT NULL,
    call_id          TEXT
);

CREATE INDEX IF NOT EXISTS idx_side_effect_events_key
    ON side_effect_events (side_effect_id, ts);

CREATE INDEX IF NOT EXISTS idx_side_effect_run_status
    ON side_effect_ledger (run_id, status);
