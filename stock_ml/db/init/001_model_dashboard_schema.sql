create table if not exists models (
    id text primary key,
    name text not null,
    version text not null,
    status text not null,
    strategy text not null,
    feature_set text not null,
    entry_model text not null,
    exit_model text not null,
    market text not null,
    priority integer not null default 0,
    visible_in_dashboard integer not null default 0,
    created_at text not null,
    updated_at text not null,
    retired_at text,
    retired_reason text
);

create table if not exists runs (
    id text primary key,
    model_id text not null references models(id),
    run_name text not null,
    config_hash text not null,
    config_path text,
    resolved_config_path text,
    status text not null,
    created_at text not null,
    completed_at text
);

create index if not exists idx_runs_model_id
    on runs(model_id);

create table if not exists artifacts (
    id text primary key,
    run_id text not null references runs(id),
    kind text not null,
    path text not null unique,
    size_bytes integer not null default 0,
    checksum text,
    created_at text not null,
    deleted_at text
);

create index if not exists idx_artifacts_run_id
    on artifacts(run_id);

create table if not exists metrics_snapshots (
    id text primary key,
    run_id text not null references runs(id),
    wr real not null,
    pf real not null,
    total_pnl real not null,
    max_drawdown real not null,
    sharpe real not null,
    mdd_per_symbol real not null,
    yearly_consistency real not null,
    composite_score real not null,
    captured_at text not null
);

create index if not exists idx_metrics_snapshots_run_id
    on metrics_snapshots(run_id);

create table if not exists leaderboard_rows (
    run_id text primary key references runs(id),
    model_id text not null references models(id),
    bundle text not null,
    run_name text not null,
    market text not null,
    market_family text not null,
    timeframe text not null,
    generated_at text not null,
    composite_score real not null,
    score_mode text not null,
    visible_in_dashboard integer not null default 0,
    superseded integer not null default 0,
    row_json text not null,
    created_at text not null,
    updated_at text not null
);

create index if not exists idx_leaderboard_rows_market
    on leaderboard_rows(market, market_family, timeframe, visible_in_dashboard, superseded, composite_score);

create table if not exists audit_log (
    id text primary key,
    entity_type text not null,
    entity_id text not null,
    action text not null,
    actor text not null default 'system',
    reason text,
    payload_json text,
    created_at text not null
);

create index if not exists idx_audit_log_entity
    on audit_log(entity_type, entity_id);
