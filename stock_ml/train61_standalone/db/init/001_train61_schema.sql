create table if not exists instruments (
    id bigserial primary key,
    provider_symbol_id bigint,
    symbol text not null unique,
    name text,
    exchange text,
    asset_type text,
    data_provider text,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists market_bars (
    id bigserial primary key,
    symbol text not null,
    symbol_id bigint,
    timeframe text not null,
    timestamp timestamptz not null,
    open numeric(20, 6) not null,
    high numeric(20, 6) not null,
    low numeric(20, 6) not null,
    close numeric(20, 6) not null,
    volume bigint not null default 0,
    traded_value numeric(30, 6),
    provider text not null,
    provider_bar_id bigint,
    provider_created_at timestamptz,
    ingested_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique(symbol, timeframe, timestamp, provider)
);

create index if not exists idx_market_bars_symbol_tf_time
    on market_bars(symbol, timeframe, timestamp desc);

create index if not exists idx_market_bars_time
    on market_bars(timestamp desc);

create table if not exists data_versions (
    symbol text not null,
    timeframe text not null,
    provider text not null,
    latest_timestamp timestamptz,
    latest_close numeric(20, 6),
    row_count bigint not null default 0,
    version_hash text not null,
    updated_at timestamptz not null default now(),
    primary key(symbol, timeframe, provider)
);

create table if not exists ingestion_runs (
    id bigserial primary key,
    provider text not null,
    timeframe text not null,
    mode text not null,
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    status text not null default 'running',
    symbol_count int not null default 0,
    inserted_count int not null default 0,
    updated_count int not null default 0,
    error_count int not null default 0,
    error text
);

create table if not exists data_quality_checks (
    id bigserial primary key,
    ingestion_run_id bigint references ingestion_runs(id),
    symbol text,
    timeframe text,
    timestamp timestamptz,
    check_type text not null,
    severity text not null,
    message text,
    payload jsonb,
    created_at timestamptz not null default now()
);

create table if not exists model_artifacts (
    model_id text primary key,
    model_version text not null,
    status text not null,
    artifact_path text not null,
    config_path text,
    config_hash text,
    feature_set text,
    trained_until timestamptz,
    created_at timestamptz not null default now(),
    promoted_at timestamptz
);

create table if not exists predictions (
    id bigserial primary key,
    model_id text not null,
    model_version text not null,
    symbol text not null,
    timeframe text not null,
    bar_time timestamptz not null,
    data_version_hash text not null,
    y_pred int,
    y_pred_exit int,
    buy_proba numeric(10, 6),
    payload jsonb,
    created_at timestamptz not null default now(),
    unique(model_id, model_version, symbol, timeframe, bar_time, data_version_hash)
);

create index if not exists idx_predictions_symbol_time
    on predictions(model_id, symbol, timeframe, bar_time desc);

create table if not exists signals (
    id bigserial primary key,
    model_id text not null,
    model_version text not null,
    symbol text not null,
    timeframe text not null,
    bar_time timestamptz not null,
    data_version_hash text not null,
    action text not null,
    reason text,
    confidence numeric(10, 6),
    payload jsonb,
    created_at timestamptz not null default now(),
    unique(model_id, model_version, symbol, timeframe, bar_time, data_version_hash)
);

create index if not exists idx_signals_symbol_time
    on signals(model_id, symbol, timeframe, bar_time desc);

create table if not exists trades (
    id bigserial primary key,
    model_id text not null,
    model_version text not null,
    symbol text not null,
    timeframe text not null,
    entry_date timestamptz,
    exit_date timestamptz,
    entry_price numeric(20, 6),
    exit_price numeric(20, 6),
    pnl_pct numeric(12, 6),
    holding_days int,
    entry_reason text,
    exit_reason text,
    is_open boolean not null default false,
    data_version_hash text,
    payload jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_trades_model_symbol_entry
    on trades(model_id, symbol, entry_date desc);

create table if not exists model_positions (
    model_id text not null,
    model_version text not null default 'live',
    symbol text not null,
    timeframe text not null default '1D',
    status text not null,
    entry_date timestamptz,
    entry_price numeric(20, 6),
    latest_bar_date timestamptz,
    latest_close numeric(20, 6),
    unrealized_pnl_pct numeric(12, 6),
    position_size numeric(12, 6),
    pending_action text,
    pending_signal_bar timestamptz,
    data_version_hash text,
    source text,
    payload jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key(model_id, symbol, timeframe)
);

create index if not exists idx_model_positions_status
    on model_positions(model_id, status, symbol);

create table if not exists replay_runs (
    id bigserial primary key,
    model_id text not null,
    model_version text not null,
    timeframe text not null,
    start_time timestamptz not null,
    end_time timestamptz not null,
    status text not null default 'running',
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    summary jsonb
);

create table if not exists replay_steps (
    id bigserial primary key,
    replay_run_id bigint not null references replay_runs(id),
    symbol text not null,
    bar_time timestamptz not null,
    data_version_hash text,
    y_pred int,
    y_pred_exit int,
    action text,
    position_state jsonb,
    signal_payload jsonb,
    created_at timestamptz not null default now(),
    unique(replay_run_id, symbol, bar_time)
);

create index if not exists idx_replay_steps_run_symbol_time
    on replay_steps(replay_run_id, symbol, bar_time);
