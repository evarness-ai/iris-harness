CREATE TABLE IF NOT EXISTS approval_queue (
    approval_id       TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL,
    checkpoint_id     TEXT,
    signal            TEXT NOT NULL,
    context_summary   TEXT NOT NULL,
    requested_at      TEXT NOT NULL,
    channel           TEXT NOT NULL DEFAULT 'cli',
    status            TEXT NOT NULL DEFAULT 'pending',
    responded_at      TEXT,
    response_actor    TEXT,
    timeout_at        TEXT NOT NULL,
    policy_on_timeout TEXT NOT NULL DEFAULT 'fail_closed',
    session_id        TEXT,
    items_json        TEXT,
    card_json         TEXT,
    caller            TEXT,
    executed_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_approval_pending ON approval_queue(status, timeout_at);
