-- Side-effect ledger — records irreversible tool calls for safe resume.
-- Created by PostToolUseLedgerHook at PostToolUse; consumed by `iris run resume`.
CREATE TABLE IF NOT EXISTS side_effect_ledger (
    side_effect_id   TEXT PRIMARY KEY,          -- <run_id>:<step_id>:<tool_call_id> (or a UUID)
    run_id           TEXT NOT NULL,
    step_id          INTEGER NOT NULL,
    tool             TEXT NOT NULL,             -- tool name that caused the side effect
    verification_probe TEXT NOT NULL,           -- probe name ('' = none: resume treats it as ambiguous)
    probe_metadata   TEXT NOT NULL DEFAULT '{}',-- JSON dict passed to the probe at resume time
    status           TEXT NOT NULL DEFAULT 'pending', -- pending | completed | not_completed | ambiguous | error
    completed_at     TEXT,                      -- ISO-8601 when status set to completed
    error            TEXT                       -- error message if status=error
);

CREATE INDEX IF NOT EXISTS idx_side_effect_run_status
    ON side_effect_ledger (run_id, status);
