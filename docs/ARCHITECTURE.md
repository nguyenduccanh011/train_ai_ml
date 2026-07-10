# Stock ML Architecture — Phase 0-2 Complete

**Updated:** 2026-05-31 (Phase 1-7 refactoring completed)

---

## System Overview

Stock ML is a professional ML trading system with three core components:

1. **Data Layer** — DuckDB-based OHLCV storage (replaced CSV directories)
2. **Feature Pipeline** — Vectorized feature computation with intelligent caching
3. **Experiment Engine** — YAML-driven config, walk-forward backtesting, leaderboard ranking

```
OHLCV Data              Feature Pipeline           Trading Simulation         Results
─────────────────────────────────────────────────────────────────────────────────────
  
  vn_stock.duckdb       ┌─────────────────┐       ┌──────────────────┐       
  (DuckDB)              │ DataLoader      │       │  Backtest Engine │       
       ↓                │ load_many()     │       │ walk-forward fit │       trades.csv
       │                │ load_date_range │──────→│ signal genesis   │──────→ signals.csv
  1000+ symbols         │ (SQL-based)     │       │ hard_stop/max_hold       stats.json
  1M+ bars              └─────────────────┘       │ exit logic       │
                                │                 └──────────────────┘
                                ↓
                        ┌─────────────────┐       ┌──────────────────┐
                        │ FeatureCache    │       │ Leaderboard      │
                        │ (cache manager) │       │ DB + Dashboard   │
                        │ load/save       │       │ (SQLite)         │
                        └─────────────────┘       └──────────────────┘
                                │
                                ↓
                        ┌─────────────────┐
                        │ apply_features()│
                        │ 8+ technical    │
                        │ indicators      │
                        └─────────────────┘
```

---

## Phase 1-7 Deliverables

### Phase 1: Critical Bug Fixes ✅

| Bug | Fix | Impact |
|-----|-----|--------|
| API returns empty list | Switch runs.py to DB query (LeaderboardRunRepository) | Production data consistency |
| leading_v3 NaN features | Fix column name mismatches (ret_20d, macd_hist, ret_1d) | Prevents silent NaN in training |
| Config data_source_dir ignored | Add field to ExperimentConfig, parse from YAML | Enables data root override in YAML |
| Deprecated pandas fillna | Replace fillna(method="bfill") → bfill() | Pandas 2.x compatibility |
| CSV missing fieldnames | Add direction, experiment_group, variant_type to aggregator | Complete leaderboard export |
| Duplicate trade constraint | Add UniqueConstraint(run_id, symbol, entry_date) | Prevents duplicate trade records |

### Phase 2: DuckDB Migration ✅

**Old:** 1582 CSV directories under `portable_data/vn_stock_ai_dataset/all_symbols/symbol=*/`
**New:** `portable_data/vn_stock.duckdb` with table schema:

```sql
CREATE TABLE ohlcv (
    symbol      VARCHAR NOT NULL,
    timeframe   VARCHAR DEFAULT '1D',
    date        DATE    NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE,
    traded_value DOUBLE,
    PRIMARY KEY (symbol, timeframe, date)
);
```

**Files:**
- `stock_ml/src/data/duckdb_loader.py` — DuckDBLoader class (SQL-based queries)
- `stock_ml/scripts/migrate_csv_to_duckdb.py` — One-time migration script
- `stock_ml/src/data/loader.py` — Factory function `get_loader()` (auto-detect .duckdb vs CSV)

**Benefits:**
- ✅ Single 100MB file vs 1582 directories
- ✅ O(1) symbol lookup vs filesystem traversal
- ✅ Efficient date-range queries via SQL
- ✅ Backward compatible (factory pattern)

### Phase 3: Feature Cache Integration ✅

**Before:** FeatureCacheManager built but never called → features recomputed every run
**After:** Cache automatically loaded/saved in pipeline

```python
# In run_experiment():
cache = FeatureCacheManager(cache_dir)
cache_key = cache.build_signature(data_root, symbols, feature_set)
feat = cache.load(cache_key)
if feat is None:
    feat = apply_features(raw, feature_set)
    cache.save(cache_key, feat)
```

- Cache key: SHA-1(data_root + sorted_symbols + feature_set + code_mtime)
- Multiple runs with same symbols/features share cache
- 10-50x speedup for repeated experiments

### Phase 4: Universe/Symbol Config ✅

**Before:** Symbols hardcoded in CLI `--symbols VNM,SOS,FPT`
**After:** Defined in YAML with flexible resolution

```yaml
# Example: experiment config
universe:
  mode: explicit           # explicit | group | file | market_default
  explicit_list:
    - ACB
    - VCB
    - HPG
```

**Files:**
- `stock_ml/config/universes/vn_top60.yaml` — 60 largest VN stocks
- `stock_ml/config/universes/vn_bank_sector.yaml` — 15 banking stocks
- `stock_ml/src/utils/config_loader.py:get_pipeline_symbols()` — Resolution logic

**Resolution Priority:**
1. `universe.explicit_list` (direct symbol array)
2. `universe.group` (market profile lookup, e.g., "bank")
3. `universe.file` (load from external YAML)
4. Market default (all symbols for market)

### Phase 5: DB Schema Completion ✅

**New Tables:**

1. **jobs** — async task persistence
   ```sql
   CREATE TABLE jobs (
       id          TEXT PRIMARY KEY,
       type        TEXT,                -- train, retrain, gc_sweep
       status      TEXT DEFAULT 'pending',  -- pending/running/done/failed
       run_id      TEXT REFERENCES leaderboard_runs(run_id),
       created_at  TIMESTAMP,
       started_at  TIMESTAMP,
       finished_at TIMESTAMP,
       error       TEXT,
       result      JSON
   );
   ```

2. **run_yearly_stats** — per-year performance breakdown
   ```sql
   CREATE TABLE run_yearly_stats (
       id          INTEGER PRIMARY KEY,
       run_id      TEXT NOT NULL,
       year        INTEGER,
       trades      INTEGER,
       win_rate    DOUBLE,
       total_pnl   DOUBLE,
       max_drawdown DOUBLE,
       UNIQUE (run_id, year)
   );
   ```

3. **run_symbol_stats** — per-symbol performance breakdown
   ```sql
   CREATE TABLE run_symbol_stats (
       id          INTEGER PRIMARY KEY,
       run_id      TEXT NOT NULL,
       symbol      TEXT,
       trades      INTEGER,
       win_rate    DOUBLE,
       total_pnl   DOUBLE,
       UNIQUE (run_id, symbol)
   );
   ```

**Files:**
- `stock_ml/db/models/job.py` — JobModel ORM
- `stock_ml/db/models/yearly_stat.py` — RunYearlyStatModel ORM
- `stock_ml/db/models/symbol_stat.py` — RunSymbolStatModel ORM
- `stock_ml/db/migrations/versions/0004_add_jobs_stats_tables.py` — Alembic migration

### Phase 6: API Unification ✅

**Before:** Dual data sources
- `leaderboard.py` reads from DB ✓
- `runs.py` reads from JSON file ✗

**After:** Single source of truth (SQLite DB)

**New Endpoints:**
- `GET /runs/{run_id}/trades` — Query run_trades table
- `GET /runs/{run_id}/yearly-stats` — Query run_yearly_stats table
- All routes use LeaderboardRunRepository

### Phase 7: Test Coverage ✅

**File:** `stock_ml/tests/api/test_api_main.py`

**Coverage:**
- Health check endpoint
- Leaderboard CRUD operations
- Run state management
- Error handling (404 for missing runs)
- New trades/yearly-stats endpoints
- Cache statistics

### Phase 1b.12: Universe Tracking System (3-Phase) ✅

**Problem:** Backtests don't record which universe/symbols were used → Cannot reproducibly re-run or audit fairness

**Solution:** Three-phase system to preserve universe history, track leaderboard usage, and compute fair comparisons

#### Phase 1: Live Universe Management
- **Status:** ✅ Complete
- **Components:**
  - `UniverseSetModel` table: current state + version counter
  - `UniverseSymbolModel` table: symbols per universe
  - Database: auto-increment `version` on symbol changes
  - API: CRUD endpoints in `/api/v1/universes`

#### Phase 2: Version Snapshots (✅ Implemented)
- **New Table:** `UniverseVersionModel` (immutable snapshots)
  ```sql
  CREATE TABLE universe_versions (
      id INTEGER PRIMARY KEY,
      universe_id INTEGER REFERENCES universe_sets(id),
      version INTEGER NOT NULL,
      symbols_json TEXT NOT NULL,         -- immutable copy of symbol list
      symbol_count INTEGER,
      created_at TIMESTAMP,
      UNIQUE(universe_id, version)
  );
  ```
- **Snapshots saved automatically when:**
  - `add_symbols()` — Adding new symbols → version bumps, snapshot created
  - `remove_symbol()` — Deleting symbol → version bumps, snapshot created
  - `replace_symbols()` — Full replacement → version bumps, snapshot created
- **Query:** `repo.get_version_snapshot(slug="vn_top60", version=2)` → returns list of symbols used at v2
- **Files:** `stock_ml/db/models/universe_version.py`, `0009_add_universe_versions.py` migration

#### Phase 3: Backfill & Fairness (✅ Implemented)
- **Fairness Key Enhancement:**
  - `fairness_group_key` now includes `universe_slug` when not `None`
  - Old runs: `universe_slug=NULL` → key unchanged (backward compat)
  - New runs: `universe_slug="vn_top60"` → key includes it → different fairness group
  - Ensures runs from different universes don't incorrectly compare as "same fairness group"

- **Leaderboard Integration:**
  - `LeaderboardRun` table has: `universe_slug TEXT, universe_version INT`
  - Pipeline automatically populates on experiment run:
    1. Config YAML: `universe: {mode: db, slug: vn_top60}`
    2. `run_experiments.py`: injects `resolved_version` from DB
    3. `experiment.py`: writes `universe_slug/version` to summary artifact
    4. `loader.py`: reads from summary → populate `LeaderboardRow`
  - API: `/leaderboard` endpoint returns `universe_tracked: bool` per run

- **Backfill Script:**
  - File: `stock_ml/scripts/backfill_universe_info.py`
  - Purpose: Update existing runs (universe_slug=NULL) with universe info
  - Method: Parse config YAML, query DB for current version, update leaderboard_runs
  - Safety: Adds metadata note `[backfilled_universe: slug@vN]` for audit trail

#### Universe-Leaderboard Flow Diagram
```
Experiment YAML
├─ universe: {mode: db, slug: vn_top60}
└─ ↓ run_experiments.py injects version
   
   run_experiments.py
   ├─ UniverseRepository.get_by_slug("vn_top60") → version=2
   └─ cfg.universe["resolved_version"] = 2
   
   experiment.py (run_experiment)
   ├─ Summary artifact:
   │  ├─ "universe_slug": "vn_top60"
   │  └─ "universe_version": 2
   └─ ↓
   
   loader.py (run_dir_to_row)
   ├─ Reads summary → extracts universe_slug/version
   ├─ Computes fairness_group_key (includes universe_slug)
   └─ Returns LeaderboardRow with universe fields
   
   Database
   └─ leaderboard_runs
      ├─ universe_slug: "vn_top60"
      ├─ universe_version: 2
      ├─ fairness_group_key: sha1({..., "universe_slug": "vn_top60"})
      └─ ↓ Can reconstruct: symbols = universe_versions.get(slug="vn_top60", v=2)
```

---

## Data Layer Architecture

### Storage

```
portable_data/
  vn_stock.duckdb           ← Main OHLCV (DuckDB)
  derivatives.duckdb        ← Futures OHLCV
  crypto.duckdb             ← Crypto OHLCV

cache/
  features/
    {sha1_key}.parquet      ← Precomputed features
    index.json              ← Cache metadata

results/
  leaderboard.db            ← SQLite: runs, trades, jobs, stats
```

### DataLoader Factory

```python
from stock_ml.src.data.loader import get_loader

# Auto-detect loader type
loader = get_loader("portable_data/vn_stock.duckdb")  # → DuckDBLoader
loader = get_loader("portable_data/vn_stock_ai_dataset")  # → DataLoader (CSV fallback)

# Interface (same for both):
symbols = loader.list_symbols()
df = loader.load_symbol("ACB", "2020-01-01", "2024-12-31")
dfs = loader.load_many(["ACB", "VCB"], "2020-01-01", "2024-12-31")
```

---

## Feature Pipeline

> ⚠️ **REDESIGN PLANNED**: `apply_features()` + Python registry mô tả dưới đây đang được thay bằng **Expression DSL + per-feature Feature Store** (định nghĩa chuẩn hoá trong DB, giá trị materialize 1 lần/feature). Phần dưới phản ánh hệ thống *hiện tại* cho tới khi cutover. Xem [Feature Store + DSL Design](FEATURE_STORE_DSL_DESIGN.md).

### Computation Flow

```
Raw OHLCV
    ↓
FeatureCacheManager.load(cache_key)  ← Checks cache
    ↓ (if miss)
apply_features(raw, feature_set)
    ├─ add_sma(periods=[5,10,20])
    ├─ add_rsi(periods=[14])
    ├─ add_macd()
    ├─ add_atr()
    ├─ add_obv()
    ├─ add_momentum()
    ├─ add_sector_relative(sectors)
    └─ add_cross_sectional()
    ↓
FeatureCacheManager.save(cache_key, features)
    ↓
Features → Model Input
```

### Cache Manager

**Key Building:**
```python
cache_key = hashlib.sha1(
    json.dumps({
        "data_root": data_root,
        "symbols": sorted(symbols),
        "feature_set": feature_config,
        "code_mtime": os.path.getmtime("leading_v3.py")
    }).encode()
).hexdigest()
```

**Hit Rate:** Typically 50-70% in variant generation (same symbols, different models)

---

## Experiment Configuration

### YAML Structure

```yaml
# Complete experiment config

name: "alpha_gate_v1_refined"
description: "Entry/exit refinement with new universe"

# Data configuration
data:
  source_dir: portable_data/vn_stock.duckdb  # NEW: can override CLI
  
# Symbol set configuration
universe:
  mode: explicit                    # explicit | group | file | market_default
  explicit_list:
    - ACB
    - VCB
    - HPG

# Model configuration
entry_model:
  type: lightgbm
  params:
    learning_rate: 0.05
    max_depth: 8

exit_model:
  type: hardstop
  params:
    pct: 2.0

# Feature configuration
features:
  name: leading_v3
  params:
    sma_periods: [5, 10, 20]
    rsi_period: 14

# Target configuration
target:
  type: forward_return
  params:
    forward_days: 5
    threshold_up: 0.02
    threshold_down: -0.01

# Training configuration
training:
  train_years: 4
  test_years: 1
  gap_days: 25

# Direction support (long/short/both)
direction: both  # long | short | both
```

---

## API Routes

### Leaderboard

- `GET /api/v1/leaderboard` — List all runs
- `GET /api/v1/leaderboard?market=vn_stock&limit=10` — Filter by market
- `GET /api/v1/leaderboard/top/5` — Top N models

### Runs

- `GET /api/v1/runs` — List all runs
- `GET /api/v1/runs/{run_id}/state` — Get run lifecycle state
- `PATCH /api/v1/runs/{run_id}/state` — Change state (trained/pinned/retired)
- `GET /api/v1/runs/{run_id}/trades` — Query all trades for run (NEW)
- `GET /api/v1/runs/{run_id}/yearly-stats` — Query yearly breakdown (NEW)
- `POST /api/v1/runs/{run_id}/retrain` — Spawn async retrain
- `DELETE /api/v1/runs/{run_id}` — Delete run
- `DELETE /api/v1/runs/{run_id}/cache` — Quarantine cache

### Jobs

- `GET /api/v1/jobs` — List all jobs
- `GET /api/v1/jobs/{job_id}` — Get job status
- `POST /api/v1/jobs/{job_id}/cancel` — Cancel job

### Cache

- `GET /api/v1/cache/stats` — Cache storage statistics

---

## Key Design Decisions

### DuckDB Over CSV

- ✅ Single file instead of 1582 directories
- ✅ O(1) symbol/date lookups via SQL
- ✅ Atomic backups
- ✅ Zero data loss from half-written CSVs
- ❌ Requires Python 3.10+ (acceptable)

### Feature Cache with Fingerprinting

- ✅ 10-50x speedup for variant generation
- ✅ Automatic invalidation on code change (mtime check)
- ❌ Requires careful dependency tracking
- **TODO**: Wire into DuckDB loader (currently CSV-only)

### Universe Config in YAML

- ✅ Single source of truth (no CLI symbol duplication)
- ✅ 4 resolution modes (explicit, group, file, market_default)
- ✅ Variant generator can swap universes
- ❌ Adds one more layer to config parsing

### DB as Source of Truth

- ✅ All API routes read from DB
- ✅ Reproducible run queries
- ✅ Audit trail (created_at, updated_at)
- ❌ Requires schema migrations (Alembic)

---

## Future Work

### Phase 8: Dashboard UI Updates
- Field name mismatches (max_loss vs max_dd)
- Symbol stats visualization
- Job monitoring panel

### Phase 9: Live Trading Simulator
- Real-time signal generation
- Daily execution simulation
- Live PnL tracking

### Phase 10: Leading_v4 Feature Set
- 15+ advanced technical indicators
- ML-derived features
- Market microstructure signals

---

## Validation

### Quick Start

```bash
# Verify DuckDB migration
python -c "
import duckdb
conn = duckdb.connect('portable_data/vn_stock.duckdb')
print(conn.execute('SELECT COUNT(*) FROM ohlcv').fetchall())
print(conn.execute('SELECT DISTINCT symbol FROM ohlcv LIMIT 10').fetchall())
"

# Verify feature cache
python -m pytest stock_ml/tests/cache/ -v

# Verify API endpoints
python -m pytest stock_ml/tests/api/ -v
```

### Production Checklist

- [ ] DuckDB migration complete (1000+ symbols)
- [ ] Feature cache hit rate > 50% on variant runs
- [ ] All API routes read from DB (no JSON fallback)
- [ ] Test coverage > 70% for critical paths
- [ ] DB migrations applied (alembic upgrade head)
- [ ] Leaderboard rebuilds successfully
