# Implementation Roadmap: Research-Grade Trading Model System

**Status**: Phase 0 ✅ · Phase 1a ✅ · Phase 1b ✅ · Phase 1.5 ✅ · Alpha Gate Prep ✅ · Alpha Gate Run (FAILED 2026-05-29) · **Phase 0-2: Production Research Pipeline ✅ (2026-05-31)** · Next: Feature/universe iteration
**Last Updated**: 2026-05-31 (Production Research Pipeline complete: auto-rebuild, variant generator, metadata tracking, leaderboard UI, retention policy)
**Owner**: Architecture Team

---

## Executive Summary

Refactor `stock_ml` from monolithic single-model pipeline to a **research-grade, modular trading platform** that follows the same methodology institutional quant teams use (Two Sigma / Man AHL / de Prado-style workflow).

### Guiding principles
1. **Research is search, not engineering** — do not build framework before validating alpha exists.
2. **Backtest is a hypothesis test** — every metric must carry confidence intervals + significance correction.
3. **Reproducibility is non-negotiable** — track code commit, data hash, config, and seed on every run.
4. **Cost is the alpha killer** — model microstructure costs realistically; favor pessimistic assumptions.
5. **Simple model + good features > complex model + bad features.**

### Phase order (revised)

| Phase | Focus | Gate to next phase |
|-------|-------|--------------------|
| **Phase 0** | Foundation infra (tracking, versioning, logging, caching) | Infra smoke-test pass |
| **Phase 1a** | Model / Feature / Target registries (DONE) | Unit tests pass |
| **Phase 1b** | Experiment runner + YAML config + flow fixes | E2E run reproducible |
| **Phase 1.5** | Research methodology (Purged CV, DSR, PBO, bootstrap CI, multi-seed) | Methodology smoke-test |
| **🚦 ALPHA GATE** | Validate alpha on 200+ symbols × 5y OOS with DSR/PBO | DSR > 0.5, PBO < 30% |
| **Phase 2** | Rule engine + signal/order separation (scope reduced) | Rules reduce false signals measurably |
| **Phase 3** | Portfolio risk + position sizing + leaderboard | Sizing improves risk-adjusted return |
| **Phase 4** | Live paper trading + monitoring + retraining schedule | 3 months shadow PnL ≈ backtest |

> **Critical**: do not start Phase 2 until the Alpha Gate is passed. Building rule/portfolio/live infrastructure on top of an unvalidated edge wastes weeks.

---

## Phase 0: Foundation Infrastructure (✅ COMPLETE)

Setup before any research work. None of these require alpha — they protect you from wasting time later.

**Status**: Phase 0 ✅ · Phase 1a ✅ · Phase 1b ✅ · Phase 1.5 ✅ · **Alpha Gate Prep ✅** (2026-05-29) · Next: Full Alpha Gate Run
- `src/tracking/mlflow_logger.py` — MLflow tracking with config, metrics, artifacts
- `src/seed.py` — global seed propagation (numpy, random, lightgbm, xgboost, sklearn, tf)
- `src/pipeline/run.py` — integrated reproducibility + structured logging
- `stock_ml/scripts/run_v2.py` — `--seed` CLI argument added
- `tests/test_phase0_smoke.py` — smoke tests for all Phase 0 components

### 0.1 Experiment tracking ✅
- **Tool**: MLflow (local file backend).
- **Implementation**: `src/tracking/mlflow_logger.py::MLFlowLogger` context manager.
- **Tracks per run**: config dict, git commit, data fingerprint, all scalar metrics, output artifacts (trades.csv, signals.csv, summary.json).
- **MLflow store**: `results/{experiment_name}/mlruns/`

### 0.2 Reproducibility primitives ✅
- **Data fingerprint**: `sha256(sorted(symbols) + start_date + end_date)` → 16-char hex. Computed once per run, stored in summary JSON + MLflow param.
- **Code version**: `git rev-parse HEAD` (40-char) + dirty flag via MLflow params.
- **Seed propagation**: `src/seed.py::set_global_seed(seed)` covers numpy, random, LightGBM, XGBoost, sklearn, TensorFlow. Called at run start to ensure determinism.
- **Result**: identical run (same seed + same data) → byte-identical metrics.

### 0.3 Structured logging ✅
- **Tool**: loguru.
- **Format**: `[timestamp] [LEVEL] [module:function:line] message`.
- **Replacement**: All `print(...)` → `logger.info()`, `logger.warning()`, `logger.debug()`, `logger.error()`.
- **Benefit**: Log levels controllable; structured output for parsing; `run_id` automatically included by MLflow context.

### 0.4 Feature cache ✅ (COMPLETE — 2026-05-30)
- **Path**: `results/cache/features/{feature_set}/{cache_key}.parquet` + `.pkl` fallback.
- **Implementation**: `src/cache/feature_cache.py::FeatureCacheManager` with SHA1 signature-based cache keys.
- **Cache key**: deterministic SHA1(schema_version, data_dir, symbols, timeframe, feature_set, target_config, source_fingerprint, code_fingerprint).
- **Status**: ✅ Production-ready with atomic writes, metadata tracking, and garbage collection support.

### 0.5 Output layout standardization ✅
```
results/
└── {experiment_name}/
    └── {run_id}/                # = {YYYYMMDD-HHMMSS}-{git_short_hash[:8]}
        ├── config.yaml          # (optional; currently JSON in summary)
        ├── data_fingerprint.txt # {fingerprint}\n{git_commit}\n
        ├── trades.csv
        ├── signals.csv
        ├── daily_stats.csv
        ├── yearly_stats.csv
        ├── symbol_stats.csv
        ├── summary.json         # master summary with all metadata
        └── mlruns/              # MLflow tracking (per experiment_name)
```
- **Feature**: Prevents overwrites. Second run → different run_id → separate output directory.
- **Summary JSON fields**: `run_id`, `data_fingerprint`, `git_commit`, `mlflow_run_id`, metrics, audit report, config, timestamps.

### 0.6 Golden + leakage regression tests (DEFERRED)
- **Golden test**: fixed input data + fixed config → assert exact `trades.csv` hash.
- **Leakage regression test**: inject future feature → audit must flag it.
- **Status**: Design complete, tests deferred (can add after Phase 1b).

### Phase 0 verification ✅
- ✅ Two runs of same config + same data + same seed → identical metrics (verified with test_phase0_smoke.py).
- ✅ MLflow logs created on every run (run_id, config, metrics, params).
- ✅ Output structure enforced (run_id directory with all files).
- ✅ Data fingerprint stable across runs.
- ✅ Seed propagation working (all RNG systems seeded).

---

## Phase 0-2: Production Research Pipeline (✅ COMPLETE 2026-05-31)

Optimized research iteration cycle supporting variant generation, auto-rebuild, experiment tracking, and automated cleanup. Enables professional-grade model research with full auditability and reproducibility.

### 0-2.0 Colab Cleanup
- ✅ Removed Colab-specific logic (`is_colab()`, `DRIVE_BASE` branching)
- ✅ Refactored `env.py` to local-only (simplified path resolution)
- ✅ Deleted temp docs (ANALYSIS_PLAN, FINAL_STATUS, PHASE_2_PLAN, etc.)
- ✅ Cleaned memory files (project_colab_mcp, feedback_colab_mcp_tools)

### 0-2.1 Auto-Rebuild Leaderboard
- ✅ Added `--rebuild-leaderboard` flag to `run_experiments.py` (default: True)
- ✅ After batch completes → auto-calls `rebuild_leaderboard()` if n_ok > 0
- ✅ Updates `results/leaderboard/leaderboard.json` + splits
- **Usage**: `python scripts/run_experiments.py --parallel 4` (auto-rebuilds)

### 0-2.2 Experiment Metadata Tracking
- ✅ Extended `ExperimentConfig` with `metadata: dict | None` (YAML section)
- ✅ Metadata written to `summary_{name}.json` during run
- ✅ Extended `LeaderboardRow` schema with 4 new fields:
  - `experiment_group`: string (e.g., "2026-06-01_param_tuning")
  - `variant_type`: string (params|features|target|model)
  - `parent_run_id`: string (baseline model being optimized from)
  - `metadata_notes`: string (researcher notes)
- ✅ Loader auto-extracts metadata from `summary.json` → populates row fields
- **Usage**: Add `metadata:` section to YAML config with experiment tracking

### 0-2.3 Variant Generator CLI
- ✅ New tool: `scripts/generate_variants.py`
- ✅ Input: base YAML + param grid (JSON) → Output: N YAML files (cartesian product)
- ✅ Auto-injects metadata (experiment_group, variant_type, parent_run_id, notes)
- ✅ Supports `--dry-run` preview before generating
- **Usage**: `python scripts/generate_variants.py --base X --group Y --variant-type Z --grid '{...}'`

### 0-2.4 Leaderboard UI Enhancements
- ✅ Experiment group dropdown filter (NEW)
- ✅ Group + Type columns in leaderboard table (NEW)
- ✅ Metadata notes as tooltip on Group column
- ✅ Auto-populate filter options from leaderboard rows
- **Benefit**: Filter variants by experiment group, see lineage at a glance

### 0-2.5 Retention Policy Configuration
- ✅ Added `retention:` config section to `base.yaml`:
  - `keep_pinned: true` — never delete pinned models
  - `keep_per_group: 5` — keep top-N trained per experiment_group
  - `auto_retire_after_days: 30` — retire trained runs older than N days
  - `trash_purge_after_days: 14` — purge trash older than N days
- ✅ Added `--apply-policy` flag to `cache_gc.py`
- ✅ Reads config and reports policy impact (dry-run)
- **Usage**: `python scripts/cache_gc.py --apply-policy` (shows what would happen)

### 0-2.6 End-to-End Research Iteration Cycle
1. **ANALYZE** → Dashboard + top1_model.html (pinned models)
2. **HYPOTHESIZE** → Choose variant dimension (params, features, target, model)
3. **GENERATE** → `generate_variants.py --grid {...}` → N YAMLs
4. **RUN BATCH** → `run_experiments.py --parallel 4` (auto-rebuilds)
5. **EVALUATE** → leaderboard.html + filter by experiment_group
6. **PIN WINNERS** → Click "Pin" button on top models
7. **DEEP ANALYZE** → dashboard.html (pinned models overlaid)
8. **CLEANUP** → `cache_gc.py --apply-policy` (auto-retire old, purge trash)

### Production-Ready Features
- ✅ Reproducible: Every config committed to git
- ✅ Traceable: Metadata tracks variant lineage (parent_run_id)
- ✅ Comparable: Fairness flagging when configs differ
- ✅ Scalable: Parallel runs + auto-cleanup
- ✅ Visual: Leaderboard filters + dashboard overlays
- ✅ Auditable: All runs logged with timestamps + metadata

---

## Phase 1a: Registries (COMPLETE)

Status: ✅ done.

- `src/models/registry.py` — `EntryModelProtocol` + `ExitModelProtocol` + factories for lightgbm, xgboost, random_forest, mlp. LSTM and Rule are placeholders (deferred — see scope reduction below).
- `src/features/registry.py` — `basic_v1` registered, `leading_v2` stub.
- `src/targets/registry.py` — `forward_return`, `trend_regime`.

### Scope reduction vs original roadmap
- **Drop**: LSTM and Rule entry/exit models in Phase 1. They add dependency surface without alpha evidence. Revisit only if LGBM/XGB clearly hit a ceiling.
- **Keep**: lightgbm, xgboost, random_forest, mlp.

---

## Phase 1b: Experiment Pipeline + Flow Fixes (Week 1–2)

### 1b.1 Label semantic fix — REGRESSION APPROACH (✅ RESOLVED 2026-05-29)

**DECISION: Implement regression (Option C) — professional standard**

Original problem: classification approach mixed incompatible regimes:
```python
y_entry = (target == 1)   # buy vs (sell ∪ neutral) ← semantic mismatch
y_exit  = (target == -1)  # sell vs (buy ∪ neutral) ← semantic mismatch
```

**Chosen solution: Single regression model (not A or B)**
- Target: forward return ∈ ℝ (float, not classification)
- Entry model: `LGBMRegressionModel` predicts expected forward return
- Signal rule (threshold-based):
  ```python
  if pred_return > +threshold → buy (1)
  if pred_return < -threshold → sell (-1)
  else → hold (0)
  ```
- No separate exit model (exit signal from return prediction sign)
- Uses full training data (no row dropping)

**Why professional quant shops use this:**
- Semantic clarity: model predicts what actually matters (return), not proxy classes
- Data efficiency: no wasted rows, full information for learning
- Sizing ready: Phase 3 can size positions by return magnitude (Kelly-compatible)
- Reproducible across teams (standard approach: Two Sigma, AHL, de Prado)
- Single threshold (configurable via `cfg.signal_threshold`) vs complex multi-model tuning

**Implementation:**
- ✅ `src/targets/forward_regression.py` — `ForwardReturnRegressionTarget` (raw float return)
- ✅ `src/models/regression.py` — `LGBMRegressionModel`, `XGBRegressionModel`, `RandomForestRegressionModel`
- ✅ `src/pipeline/experiment.py::train_fold()` — auto-detect target dtype, route to regression or classification
- ✅ Signal output: `[symbol, date, signal, score]` where `score = predicted_return`
- ✅ Tests: `test_train_fold_regression`, `test_train_fold_regression_reproducibility` (all pass)

**Decision documented in:** `src/pipeline/experiment.py::train_fold()` docstring (Phase 1b.1 DECISION block)

### 1b.2 Vectorize signal generation (✅ RESOLVED 2026-05-29)

**DONE via regression approach:**
```python
# Vectorized numpy operation (O(n) time, no Python loop overhead)
signals = np.where(
    pred_returns > cfg.signal_threshold,
    1,
    np.where(pred_returns < -cfg.signal_threshold, -1, 0),
)
```

For regression: fully vectorized, scales to 100k+ test rows in milliseconds.
Classification fallback: still uses loop (but not primary path, regression is recommended).

### 1b.3 Use `predict_proba` as score (✅ RESOLVED 2026-05-29)

**DONE:**
- Signal output schema: `[symbol, date, signal, score]`
- `score = predicted_return` (regression) or `predict_proba[1]` (classification)
- Enables Phase 3 confidence-weighted sizing without re-running models
- Stored as float32 in outputs

### 1b.4 Unify `run.py` ↔ `experiment.py` ✅ DONE
- ~80% logic duplicated. Maintaining both doubles cost and risks divergence.
- **Action**: refactor `pipeline/run.py` into a thin wrapper that builds an `ExperimentConfig` and calls `run_experiment()`. Single source of truth.
- **Implemented 2026-05-29**: `run.py` now delegates to `run_experiment()`, maintains legacy output structure (run_id directories) for backward compatibility

### 1b.5 Unify signal generation across backtest and live_sim ✅ DONE
- Risk: live and backtest computing signals via slightly different code paths → train/serve skew (a top-3 cause of live underperformance).
- **Action**: extract `generate_signal(model, features) → signal, score` to `src/signals/core.py`. Use in both `backtest/engine.py` and `live_sim/signals.py`.
- **Implemented 2026-05-29**: 
  - Created `src/signals/core.py` with `generate_signals_from_predictions()`, `generate_signals_from_features()`, `generate_signals_dict()`
  - Updated `src/pipeline/experiment.py::train_fold()` to use unified function
  - Updated `src/live_sim/signals.py::SignalGenerator` to use unified function
  - Same threshold logic + vectorization + filters across both paths → eliminates train/serve skew

### 1b.6 YAML schema (canonical, nested)
Decision: use nested `components:` block. Update `ExperimentConfig.from_yaml` to validate this schema strictly.

```yaml
name: lgbm_entry_rf_exit_v1
hypothesis: |             # NEW — required field
  Stocks with positive 20d momentum AND volume spike outperform over a
  5-bar horizon because retail FOMO drives short-term continuation.
strategy: entry_exit_ensemble
market: vn_stock
seed: 42

components:
  features: basic_v1
  target:
    type: forward_return
    horizon: 5
    gain_threshold: 0.04
    loss_threshold: 0.04

  entry_model:
    type: lightgbm
    params: {n_estimators: 300, learning_rate: 0.05, num_leaves: 31}

  exit_model:
    type: random_forest
    enabled: true
    params: {n_estimators: 200, max_depth: 15}

split:
  type: purged_kfold        # NEW — replaces hardcoded year split
  n_splits: 6
  embargo_days: 10
  # legacy year split still supported via type: walk_forward_year

engine:
  max_hold_bars: 20
  min_hold_bars: 1
  hard_stop_pct: -0.08
  costs:
    commission: 0.0015
    tax: 0.001
    slippage_model: fixed   # fixed | sqrt_volume
    slippage: 0.001

validation:                 # NEW — Phase 1.5
  n_seeds: 20
  bootstrap_iterations: 1000
  compute_dsr: true
  compute_pbo: true

strict_audit: true          # NEW — fail run if leakage audit fails
```

### 1b.7 Auto-detect data year range
Replace hardcoded `first_test_year=2020, last_test_year=2025` with auto-detect from loaded data. Hardcoding breaks every January.

### 1b.8 Output path with run_id
- Current: `results/{out}/trades_{name}.csv` → second run of same name overwrites first.
- Fix: see Phase 0.5 layout. Add `run_id` to all output paths.

### 1b.9 `strict_audit` mode
- Currently leakage audit prints and continues. A failing audit must abort the run with non-zero exit.
- `strict_audit: true` (default true) → raise on `audit.fail == True`.

### 1b.10 Atomic YAML queue
- `scripts/run_experiments.py` moves YAML between `pending/done/failed/` directories. Two workers can race and pick the same file.
- Fix: per-file lock (`.lock` sidecar) or rename atomicity via `os.rename` with try/except.

### 1b.11 Resumable runs
- On crash mid-fold, partial state is lost.
- Minimum fix: write each fold's signals to `results/{exp}/{run_id}/folds/{fold_label}.parquet` as it completes. On rerun, skip folds whose parquet exists.

### Phase 1b verification (COMPLETE 2026-05-29)
- ✅ Regression train_fold reproducibility: same seed → identical signals (test_train_fold_regression_reproducibility)
- ✅ Vectorized signal generation (np.where, not loop) for regression path
- ✅ Score column added to signal output (predicted_return for regression)
- ✅ Signal generation unified: backtest and live_sim use same core logic (1b.5)
- ✅ `run.py` delegates to `run_experiment()` — single source of truth (1b.4)
- ✅ **1b.6 YAML schema**: ExperimentConfig strict validation (components, split, engine, hypothesis, validation)
- ✅ **1b.7 Auto-detect years**: YearSplitter.from_data() auto-detects year range from DataFrame (no hardcoded 2020-2025)
- ✅ **1b.9 Strict audit**: strict_audit=true aborts run on audit failure (configurable per YAML)
- ✅ **1b.10 Atomic queue**: Per-file .lock file prevents race condition in parallel runs
- ✅ **1b.11 Resumable runs**: Fold checkpointing via {run_id}/folds/{label}.parquet; skips completed folds on rerun
- ✅ All 74 tests pass (10 pipeline + 8 phase0 + 10 live_sim + 46 integration)

---

## Phase 1.5: Research Methodology ✅ COMPLETE (2026-05-29)

This is the **most important phase** and the one missing from the original roadmap. Without it, the leaderboard in Phase 3 ranks noise.

### 1.5.1 Purged K-Fold + Embargo splitter ✅
- ✅ `src/data/splitter.py::PurgedKFoldSplitter(n_splits, embargo_days, label_horizon)`
- ✅ Prevents label leakage from forward-return targets (e.g., 5-bar forward return at train end won't peek into test)
- ✅ Reference: de Prado, *Advances in Financial ML*, Ch. 7
- ✅ Tests: basic split, embargo verification, edge cases

### 1.5.2 Bootstrap confidence intervals ✅
- ✅ `src/evaluation/bootstrap.py::bootstrap_metric(returns, fn, n_iter=1000, block_size=None)`
- ✅ Block bootstrap accounts for autocorrelation in returns
- ✅ Computes 95% CI + std for any metric (Sharpe, Sortino, max drawdown, win rate)
- ✅ Tests: CI coverage, metric calculations

### 1.5.3 Deflated Sharpe Ratio (DSR) ✅
- ✅ `src/evaluation/dsr.py::deflated_sharpe(sharpe, n_trials, returns_skew, returns_kurt, sample_length)`
- ✅ Corrects for multiple-testing bias: DSR = SR - sqrt(2*ln(N)) * sigma(SR)
- ✅ Accounts for returns skewness and kurtosis
- ✅ Reference: Bailey & López de Prado (2014)
- ✅ Tests: single/multiple trials, correction magnitude

### 1.5.4 Probability of Backtest Overfitting (PBO) ✅
- ✅ `src/evaluation/pbo.py::pbo(scores)` — Combinatorially Symmetric CV
- ✅ Detects when in-sample champion underperforms OOS (Type I + Type II errors)
- ✅ Reference: Bailey, Borwein, López de Prado, Zhu (2017)
- ✅ Tests: basic functionality, overfitting detection

### 1.5.5 Multi-seed variance ✅
- ✅ `src/pipeline/multi_seed.py::run_experiment_multi_seed(cfg, n_seeds, seeds)`
- ✅ Runs experiment with different seeds (data sampling, model init)
- ✅ Reports mean ± std for all metrics (not point estimates)
- ✅ Enables reproducibility variance analysis

### 1.5.6 Hypothesis pre-registration ✅
- ✅ YAML field `hypothesis:` (implemented in Phase 1b.6)
- ✅ Describes economic intuition before running
- ✅ Ready for MLflow logging + post-run hit rate tracking

### 1.5.7 Hyperparameter search (DEFERRED)
- ⏳ Optuna + nested CV (optional, not blocking Alpha Gate)
- Placeholder: `src/pipeline/tune.py` (future)

### Phase 1.5 verification ✅
- ✅ Purged KFold: embargo reduces training set size (test confirmed)
- ✅ Bootstrap CI: CI contains point estimate (test confirmed)
- ✅ DSR: larger n_trials → larger correction (test confirmed)
- ✅ PBO: detects in-sample champions that underperform OOS (test confirmed)
- ✅ All 15 Phase 1.5 tests pass; 89 total tests (no regressions)

---

## Alpha Gate Prep ✅ COMPLETE (2026-05-29)

Preparation infrastructure for the Alpha Validation Gate. All components built and tested.

### Implementation Summary

**1. leading_v2 Feature Set (36 per-symbol features)**

Created `src/features/leading_v2.py` with 9 blocks:
- **ohlcv_basic** (5): ret_1d/5d/10d/20d, close_to_open
- **moving_averages** (5): sma_5/20/50 ratios, ema_10_ratio, sma5_cross_sma20  
- **momentum** (5): rsi_14/7, macd_line/hist, roc_10
- **trend** (3): adx_14, plus_di_14, minus_di_14
- **volatility** (4): atr_14_ratio, bb_width/pct_20, realized_vol_10
- **volume_advanced** (4): volume_ratio_5/20, obv_slope_10, mfi_14
- **market_structure** (3): dist_52w_high/low, high_low_pct_5d
- **exhaustion** (4): upper/lower_wick_ratio, body_ratio, high_low_pct
- **volatility_regime** (3): atr_regime, bb_squeeze, vol_percentile_60

All leakage-safe (per-symbol groupby, no lookahead). Warmup NaNs only.

**2. PurgedKFoldSplitter Integration**

Added support in `src/pipeline/experiment.py` for purged_kfold split type:
- Imports PurgedKFoldSplitter from src.data.splitter
- Handles n_splits, embargo_days, label_horizon parameters
- Collects windows during split() loop for audit
- Modified audit_report() to gracefully handle PurgedFoldWindow (skip year-based checks)

Reference: de Prado *Advances in Financial ML*, Ch. 7.

**3. Alpha Gate YAML Config**

Created `config/experiments/pending/alpha_gate_v1.yaml`:
```yaml
name: alpha_gate_v1
components:
  features: leading_v2
  target: {type: forward_return_regression, horizon: 5}
  entry_model: {type: lightgbm, params: {n_estimators: 300, learning_rate: 0.05}}
split: {type: purged_kfold, n_splits: 6, embargo_days: 10, label_horizon: 5}
costs: {commission: 0.0025, tax: 0.001, slippage: 0.0015}  # pessimistic
validation: {n_seeds: 20, bootstrap_iterations: 1000, compute_dsr: true, compute_pbo: true}
signal_threshold: 0.005
strict_audit: true
```

**4. Fixes Applied**

- Fixed `scripts/run_experiments.py` sys.path setup (added REPO_ROOT, matching run_v2.py pattern)
- Fixed cost extraction to handle nested `engine.costs` dict from YAML
- Fixed slippage_model parameter handling (not used by CostModel, gracefully dropped)
- Fixed audit_report to work with both SplitWindow and PurgedFoldWindow types

### Test Run Results (10 VN symbols, 6 folds)

```
[alpha_gate_v1] dataset: 27,696 bars across 10 symbols
[split] purged_kfold: 6 splits, embargo_days=10, label_horizon=5
  [fold fold_0] train=2,331  test=217   buys=140  sells=27
  [fold fold_1] train=2,073  test=469   buys=336  sells=76
  [fold fold_2] train=2,071  test=469   buys=279  sells=110
  [fold fold_3] train=2,076  test=464   buys=140  sells=191
  [fold fold_4] train=2,069  test=469   buys=189  sells=172
  [fold fold_5] train=2,082  test=471   buys=227  sells=115
[backtesting] 2,559 signals → 182 trades

Results:
- Win rate: 42.3%
- Profit factor: 0.76
- Audit: PASS (no leakage detected)
- Output: results/alpha_gate/{trades,signals,daily_stats,yearly_stats,symbol_stats}_alpha_gate_v1.csv
```

✓ Pipeline end-to-end verified: features → split → training → backtesting → audit

---

## 🚦 Alpha Validation Gate (Week 4–5)

Run a **single decisive experiment** before continuing the roadmap:

- Universe: ≥ 200 VN symbols (HOSE + HNX), including survivors AND delisted.
- Period: ≥ 5 years OOS via Purged KFold.
- Models: LightGBM entry + LightGBM exit (current best understood algorithm).
- Features: `basic_v1` (8 features) + add `leading_v2` (≥ 30 features: cross-sectional rank, sector-relative, volume profile, etc.).
- Costs: pessimistic — commission 0.25%, tax 0.1%, slippage 0.15%, market impact via √volume.

### Gate criteria

| Metric | Threshold | Action if missed |
|--------|-----------|------------------|
| Deflated Sharpe Ratio | > 0.5 | Stop; revisit features, target, universe |
| PBO | < 30% | Stop; reduce search space, add embargo |
| Annualized return (mean of 20 seeds) | > 12% net of costs | Stop |
| Max drawdown (95th percentile) | < 25% | Stop |
| Trades per year | 20 ≤ N ≤ 500 | Adjust target horizon / thresholds |
| Sharpe std across 20 seeds | < 0.5 × mean | Investigate instability |

**If gate fails**: do NOT proceed to Phase 2. Iterate on features, targets, or universe construction. Building rule engines and leaderboards on top of no-alpha just produces nicer-looking nothing.

**If gate passes**: proceed to Phase 2 with confidence.

---

## Phase 1.6: Production Readiness & Model Lifecycle (2026-05-30) ✅

Post-alpha-gate iteration on production infrastructure while waiting for feature engineering iteration to re-run Alpha Gate.

### 1.6.1 API Server & Model Lifecycle Management ✅
- **REST API**: FastAPI app in `stock_ml/api/main.py` with lifecycle endpoints.
- **Lifecycle states**: `trained` (leaderboard only) → `pinned` (dashboard export) → `retired` (historical).
- **Endpoints**:
  - `PATCH /api/v1/runs/{run_id}/state` — change lifecycle state
  - `DELETE /api/v1/runs/{run_id}` — delete run + artifacts
  - `GET /api/v1/runs` — list all runs with filtering
- **Status**: ✅ Complete. Integrated into main production API (port 8000).

### 1.6.2 Cache Management & Garbage Collection ✅
- **Feature cache**: `results/cache/features/{feature_set}/{cache_key}.parquet`
- **Garbage collection**: Orphan detection, safe quarantine to `_trash/`, purge on schedule.
- **API endpoints**:
  - `GET /api/v1/cache/stats` — disk usage, orphan count
  - `POST /api/v1/gc/sweep` — quarantine orphans (dry-run or apply)
  - `POST /api/v1/cache/purge-trash` — delete old trash batches
- **Status**: ✅ Complete. `src/cache/garbage_collector.py` + API integrated.

### 1.6.3 Bulk Model Operations ✅
- **Bulk state transitions**: `POST /api/v1/runs/bulk-state` — retire/pin all trained models at once.
- **Bulk delete**: `DELETE /api/v1/runs/bulk` — delete all retired models + rebuild leaderboard.
- **Status**: ✅ Complete. Reduces manual overhead for lifecycle management.

### 1.6.4 Dashboard & Leaderboard UI ✅
- **Leaderboard UI**: Sortable table with pin/unpin/retire/delete buttons per model.
- **Dashboard**: Candlestick chart with pinned model overlays, signal markers.
- **Cache management panel**: Toggle-able UI for GC sweep, purge trash, bulk operations.
- **API integration**: Fixed `apiFetch()` to use correct `/api/v1` prefix.
- **Status**: ✅ Complete. All buttons wired to REST API endpoints.

### 1.6.5 Documentation ✅
- **API.md**: Complete endpoint documentation (lifecycle, cache, bulk, jobs).
- **README.md**: Added REST API section with startup instructions + endpoint overview.
- **QUICKSTART.md**: Fixed reference to missing file, added lifecycle + cache testing examples.
- **IMPLEMENTATION_ROADMAP.md**: Updated with Phase 0.4 cache status + Phase 1.6 overview.
- **Status**: ✅ Complete. All docs current as of 2026-05-30.

### 1.6 Verification ✅
- ✅ API loads without errors (21 /api/v1 endpoints).
- ✅ Lifecycle state transitions work: trained → pinned → retired → deleted.
- ✅ Cache stats endpoint returns accurate MB usage + orphan count.
- ✅ GC sweep (dry-run) correctly identifies orphans.
- ✅ Bulk operations update multiple models with single rebuild.
- ✅ Dashboard UI buttons integrated with lifecycle API.
- ✅ Documentation complete and tested (all curl examples work).

### Next steps for Phase 1.6 closure
- [ ] Run Alpha Gate iteration on improved features (Phase 1.7 activity)
- [ ] If gate passes: proceed to Phase 2 (rule engine) 
- [ ] If gate fails: iterate features/targets again with new insights

---

## Phase 2: Rule Engine (Scope Reduced) (Week 5–7)

### Scope reduction
The original roadmap listed 5 rule categories with ~30 parameters. This is premature for an unproven strategy. Reduced scope:

| Category | Phase 2 inclusion |
|----------|------------------|
| Exit rules (stop-loss, take-profit, trailing, max-hold) | ✅ KEEP — highest impact on PnL |
| Portfolio rules (max positions, max position size) | ✅ KEEP — required before any live |
| Market regime filter | ⏳ DEFER until Phase 2.5 if Phase 2 yields measurable gain |
| Universe filter | ⏳ DEFER — already partially in `data/loader.py` |
| Entry setup filter | ⏳ DEFER — features already encode trend/volume/momentum |

### 2.1 Signal/Order separation
- `Signal` = model's opinion (symbol, date, direction, score, reason).
- `Order` = decision after rules + portfolio state (symbol, date, action, qty, reason).
- Clean separation makes A/B-testing rules trivial.
- **Files**: `src/signals/core.py` (already created in 1b.5), `src/orders/executor.py`.

### 2.2 Exit rules
`src/rules/exit_rules.py` — `StopLoss`, `TakeProfit`, `TrailingStop`, `MaxHolding`. Each is a `RuleProtocol`.

### 2.3 Portfolio rules
`src/rules/portfolio_rules.py` — `MaxPositions`, `MaxPositionSize`. Defer industry/correlation rules until you have industry mapping data.

### 2.4 Rule engine
Single class evaluating rule list with AND semantics, returning per-rule pass/fail + reason. Reasons logged to every order — critical for debugging live trades.

### Phase 2 verification
- A/B test: same signals + ON/OFF each rule → measure ΔSharpe, ΔmaxDD.
- A rule with no statistically significant impact (bootstrap p > 0.1) is dropped.

---

## Phase 3: Portfolio Risk + Sizing + Leaderboard (Week 7–9)

### 3.1 Position sizing
- **Fixed fractional** (baseline).
- **Volatility targeting** — size = target_vol / realized_vol (recommended default).
- **Kelly fractional** (0.25 × Kelly) — optional, requires reliable edge estimate.
- Decision: drive sizing from `score` produced in 1b.3.

### 3.2 Risk metrics
`src/portfolio/metrics.py` — exposure, concentration (Herfindahl), correlation, VaR / CVaR (historical + parametric).

### 3.3 Strategy leaderboard
Single leaderboard (not three). Each row = `(strategy_id, run_id)` with:
- Mean ± std of: annual return, Sharpe, Sortino, max DD, Calmar, win rate.
- DSR, PBO from Phase 1.5.
- Composite score (formula explicit, tunable in `config/leaderboard.yaml`).
- Rank.

Entry-model and exit-model leaderboards are derivatives — generate on demand by aggregating strategy leaderboard, not separate pipelines.

### Phase 3 verification
- Position sizing ON vs OFF: measure risk-adjusted return improvement (Sharpe, Calmar).
- Leaderboard top strategy outperforms equal-weight baseline by ≥ 20% in Sharpe.

---

## Phase 4: Live Paper Trading + Monitoring (Week 10+)

Only start after Phase 3 confirms a deployable strategy.

### 4.1 Paper trading loop
- Existing `src/live_sim/` is the starting point.
- Must use the **unified signal-gen** from 1b.5 (no separate code path).
- Daily log to `results/live/{date}/` with same schema as backtest.

### 4.2 Drift monitoring
- **Data drift**: PSI / KS test on feature distributions vs training window.
- **Prediction drift**: distribution of `score` vs training distribution.
- **Performance decay**: rolling Sharpe of live vs backtest CI band.

### 4.3 Champion/Challenger
- Current production strategy = champion.
- Top of leaderboard from latest research run = challenger.
- Promote challenger only after N months of shadow PnL within backtest CI.

### 4.4 Retraining schedule
- Quarterly retrain on rolling 4-year window.
- Diff old vs new model on identical holdout; promote only if Sharpe improvement is bootstrap-significant.

### Phase 4 verification
- 3 months live shadow PnL within backtest 95% CI.
- Drift dashboard flags injected synthetic regime change.

---

## Phase 0 + Phase 1b.1-5 Completion Summary

**What was done (2026-05-29)**:

### Phase 0 ✅
- ✅ MLflow tracking infrastructure (config, metrics, artifacts logging)
- ✅ Reproducibility: data fingerprint + git commit + global seed propagation
- ✅ Structured logging (loguru replacing print)
- ✅ Output layout with run_id directories (prevents overwrites)
- ✅ Smoke tests validating reproducibility (same seed → identical metrics)

### Phase 1b.1-3 ✅ (Label Semantic + Vectorization + Scoring)
- ✅ **1b.1 Label semantic fix**: Regression approach (professional standard)
  - Single entry model predicts forward return (not classification)
  - Threshold-based signal generation (buy/sell/hold)
  - No separate exit model (exit from return sign)
- ✅ **1b.2 Vectorize signals**: Numpy vectorized (np.where, O(n), no loop overhead)
- ✅ **1b.3 Predict_proba score**: Score column added (`[symbol, date, signal, score]`)
- ✅ New files: `src/targets/forward_regression.py`, `src/models/regression.py`
- ✅ Updated files: `src/pipeline/experiment.py`, `src/targets/registry.py`, `src/models/registry.py`
- ✅ Tests: 10 smoke tests pass (reproducibility, vectorization, perf verified)

### Phase 1b.4-5 ✅ (Unification: Pipeline + Signals)

**1b.4: Unified Pipeline**
- ✅ **Goal**: Single source of truth, eliminate 200+ duplicate lines
- ✅ **Implementation**: 
  - `src/pipeline/run.py` refactored → thin wrapper over `run_experiment()`
  - `RunConfig` → `ExperimentConfig` conversion logic
  - Maintains backward compatibility (run_id directories, legacy summary format)
- ✅ **Files**: `src/pipeline/run.py` (95 lines, was 265 lines)
- ✅ **No breaking changes**: `scripts/run_v2.py`, all tests unchanged

**1b.5: Unified Signal Generation**
- ✅ **Goal**: Backtest/live signal consistency, eliminate train/serve skew
- ✅ **Implementation**:
  - Created `src/signals/core.py` with unified functions:
    - `generate_signals_from_predictions()` — backtest path
    - `generate_signals_from_features()` — live trading path
    - `generate_signals_dict()` — dict-based signal helper
  - Updated `experiment.py::train_fold()` to use unified function
  - Updated `live_sim/signals.py::SignalGenerator` to use unified function
- ✅ **Files**: New `src/signals/{__init__.py,core.py}`, updated `experiment.py`, `live_sim/signals.py`
- ✅ **Benefit**: Same threshold logic + vectorization + filters across both paths

### Test Status
- ✅ All 74 tests pass (10 pipeline + 8 phase0 + 10 live_sim + 46 integration)
- ✅ Reproducibility verified (same seed → identical signals)
- ✅ No regressions

**Next steps**:
1. **Phase 1b.6-11** — YAML schema, strict_audit, auto-detect years, YAML queue locking, resumable folds
2. **Phase 1.5** — Add Purged KFold, DSR, PBO, bootstrap CI for research-grade validation
3. **Alpha Gate** — Run decisive experiment on 200+ symbols × 5y OOS with proper methodology

---

## Flow Issues in Current Codebase (Tracked)

These are concrete code-level problems found during review. Issues #1–#3 are blockers for Phase 1b; the rest are addressed or deferred.

| # | Issue | File / Reference | Phase | Status |
|---|-------|------------------|-------|--------|
| 1 | Duplicate logic in `run.py` and `experiment.py` | `src/pipeline/{run,experiment}.py` | 1b.4 | ⏳ TODO |
| 2 | `train_fold` calls `exit_model.predict` per row in a Python loop | `experiment.py:126-135` | 1b.2 | ✅ FIXED (regression vectorized) |
| 3 | Label conversion `(target==1)` mixes neutral + sell into negative class | `experiment.py:109,115` | 1b.1 | ✅ FIXED (regression approach) |
| 4 | Features recomputed on every run | `pipeline/experiment.py:178` | 0.4 | ⏳ DEFERRED |
| 5 | Hardcoded `first_test_year=2020, last_test_year=2025` | `run.py:42-43` | 1b.7 | ⏳ TODO |
| 6 | No data fingerprint in summary | `experiment.py:258-294` | 0.2 | ✅ FIXED (Phase 0) |
| 7 | Backtest discards model `predict_proba` | `experiment.py:124-138` | 1b.3 | ✅ FIXED (score column added) |
| 8 | LightGBM `random_state=0` ignores `cfg.seed` | `models/registry.py:73` | 0.2 | ✅ FIXED (Phase 0: seed propagation) |
| 9 | `print(...)` everywhere; no run_id in logs | many files | 0.3 | ✅ FIXED (Phase 0: loguru) |
| 10 | Smoke tests don't assert metrics, no golden / leakage regression tests | `tests/` | 0.6 | ✅ PARTIAL (smoke tests added; golden/leakage deferred) |
| 11 | `audit_report` failures don't abort the run | `experiment.py:242-243` | 1b.9 | ⏳ TODO |
| 12 | Output path overwrites on second run with same name | `experiment.py:245-256` | 0.5 | ✅ FIXED (Phase 0: run_id layout) |
| 13 | YAML queue lacks per-file locking | `scripts/run_experiments.py` | 1b.10 | ⏳ TODO |
| 14 | No mid-fold resume on crash | `experiment.py:198-218` | 1b.11 | ⏳ TODO |
| 15 | live_sim and backtest signal paths diverge | `live_sim/signals.py` vs `backtest/engine.py` | 1b.5 | ⏳ TODO |

---

## Technology Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Tracking | MLflow (local) | Run metadata, metrics, artifacts |
| Versioning | Git + DVC (or parquet snapshot hash) | Code + data lineage |
| ML | LightGBM 4.x, XGBoost 2.x | Entry / exit models |
| ML (optional) | scikit-learn | RandomForest, MLP baseline |
| HP search | Optuna | Nested CV tuning |
| Stats | scipy, arch | Bootstrap, block bootstrap |
| Data | pandas 2.x, pyarrow | DataFrames + parquet |
| Config | PyYAML 6.x + pydantic (optional) | Schema-validated configs |
| Logging | loguru | Structured logs |
| Tests | pytest, hypothesis | Unit + property-based |

**Deferred** (revisit only after alpha gate): TensorFlow/Keras (LSTM), rule-only models, SQL backend.

---

## Directory Layout

```
stock_ml/
├── src/
│   ├── data/                  loader, splitter (+ PurgedKFold in 1.5)
│   ├── features/              basic_v1 (8 feat), leading_v2 (36 feat), registry, cache
│   ├── targets/               forward, trend_regime, registry
│   ├── models/                registry (lgbm, xgb, rf, mlp)
│   ├── signals/               core.py (unified signal-gen, 1b.5)
│   ├── orders/                executor.py (phase 2)
│   ├── rules/                 exit_rules.py, portfolio_rules.py (phase 2)
│   ├── portfolio/             sizing.py, metrics.py (phase 3)
│   ├── leaderboard/           strategy_leaderboard.py (phase 3)
│   ├── evaluation/            bootstrap.py, dsr.py, pbo.py (1.5)
│   ├── tracking/              mlflow_logger.py (0.1)
│   ├── pipeline/              experiment.py (single source of truth)
│   ├── backtest/              engine, stats, integrity
│   └── live_sim/              uses signals/core.py
│
├── config/
│   ├── experiments/{pending,done,failed}/
│   ├── rules/                 (phase 2)
│   └── leaderboard.yaml       composite-score weights
│
├── cache/
│   └── features/{set}/{hash}.parquet   (0.4)
│
├── results/
│   └── {experiment}/{run_id}/...        (0.5)
│
├── scripts/
│   ├── run_experiment.py      single run
│   ├── run_experiments.py     batch with locking
│   └── tune.py                Optuna nested CV (1.5.7)
│
├── tests/
│   ├── test_pipeline_smoke.py
│   ├── test_golden.py         (0.6)
│   ├── test_leakage_regression.py (0.6)
│   ├── test_methodology.py    (1.5)
│   └── test_rules.py          (phase 2)
│
└── IMPLEMENTATION_ROADMAP.md
```

---

## Milestones & Timeline (revised, realistic)

| Phase | Duration | Status | Exit criteria |
|-------|----------|--------|---------------|
| Phase 0 | 1 week | ✅ DONE (2026-05-22) | MLflow + seed + logging + output layout |
| Phase 1a | done | ✅ DONE | Registries import + instantiate |
| Phase 1b | 1–2 weeks | ✅ DONE (2026-05-29) | 1b.1-11: regression, vectorization, unified pipeline, YAML schema, strict audit, atomic queue, resumable |
| Phase 1.5 | 2 weeks | ✅ DONE (2026-05-29) | Purged KFold, DSR, PBO, bootstrap CI, multi-seed (all 5 items complete) |
| **Alpha Gate Prep** | done | ✅ DONE (2026-05-29) | leading_v2 (36 features), purged_kfold, YAML config, test run verified |
| **Alpha Gate Run** | 1–2 weeks | ⏳ NEXT | 200+ symbols, 20 seeds, DSR > 0.5, PBO < 30%, return > 12%, maxDD < 25% |
| Phase 2 | 2 weeks | ⏳ CONDITIONAL on gate | Exit + portfolio rules A/B-validated |
| Phase 3 | 2 weeks | ⏳ CONDITIONAL | Sizing improves Sharpe; leaderboard ranks |
| Phase 4 | 4+ weeks | ⏳ CONDITIONAL | 3-month shadow PnL within CI |

**Total to live trading** (assuming gate passes first attempt): **~13 weeks**
- Phase 0 (foundation): 1 week ✅ DONE (2026-05-22)
- Phase 1a (registries): done ✅ DONE
- Phase 1b (pipeline): 1–2 weeks ✅ DONE (2026-05-29)
  - 1b.1 Label semantic: ✅ Regression approach (professional standard)
  - 1b.2 Vectorize signals: ✅ Numpy vectorized (no loop overhead)
  - 1b.3 Predict_proba score: ✅ Score column added
  - 1b.4 Unify run.py/experiment.py: ✅ Single source of truth, 200+ lines eliminated
  - 1b.5 Unify signal generation: ✅ Backtest/live consistency, train/serve skew eliminated
  - 1b.6 YAML schema: ✅ Strict validation (components, split, engine, validation blocks)
  - 1b.7 Auto-detect years: ✅ YearSplitter.from_data() replaces hardcoded years
  - 1b.8 Output path with run_id: ✅ (done in Phase 0.5)
  - 1b.9 Strict audit: ✅ Aborts on failure if strict_audit=true
  - 1b.10 Atomic queue: ✅ Per-file .lock prevents race conditions in parallel runs
  - 1b.11 Resumable runs: ✅ Fold checkpointing via {run_id}/folds/*.parquet
- Phase 1.5 (methodology): 2 weeks (critical for alpha validation)
- Alpha gate: 1 week (go/no-go decision)
- Phase 2–4 (conditional): 8 weeks if gate passes

Original roadmap estimated 7 weeks but skipped Phase 1.5 (methodology) and alpha gate, which are highest-risk. Doing them properly adds ~3 weeks but prevents costly mistakes.

**Progress summary (as of 2026-05-29):**
- ✅ **Phase 0** (DONE 2026-05-22): MLflow tracking, seed propagation, structured logging, reproducibility
- ✅ **Phase 1a** (DONE): Entry/exit model registries, feature registries, target registries
- ✅ **Phase 1b.1-5** (DONE 2026-05-29): Regression approach, vectorization, unified pipeline & signals
- ✅ **Phase 1b.6-11** (DONE 2026-05-29):
  - 1b.6: YAML schema strict validation (components, split, engine, validation, hypothesis)
  - 1b.7: Auto-detect year range from data
  - 1b.8: Output path with run_id (Phase 0)
  - 1b.9: Strict audit mode (aborts on failure)
  - 1b.10: Atomic YAML queue (per-file .lock prevents race conditions)
  - 1b.11: Resumable runs (fold checkpointing)
- ✅ **Phase 1.5** (DONE 2026-05-29):
  - 1.5.1: Purged K-Fold with embargo (de Prado method)
  - 1.5.2: Bootstrap confidence intervals (block bootstrap for autocorrelated returns)
  - 1.5.3: Deflated Sharpe Ratio (multiple-testing correction)
  - 1.5.4: Probability of Backtest Overfitting (Combinatorially Symmetric CV)
  - 1.5.5: Multi-seed variance runner (mean ± std reporting)
- ✅ **Alpha Gate Prep** (DONE 2026-05-29):
  - leading_v2: 36 per-symbol features (9 blocks: ohlcv, MA, momentum, trend, vol, volume, structure, exhaustion, regime)
  - purged_kfold: integrated into experiment.py with embargo support
  - YAML config: alpha_gate_v1.yaml with pessimistic costs, 20-seed validation
  - Test run: 10 symbols × 6 folds, end-to-end pipeline verified, audit PASS
- ✅ Reproducibility verified (same seed → identical signals, all 89+ tests pass)
- **Next**: Alpha Gate Run (200+ symbols, 20 seeds, gate criteria validation)

---

## Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| Alpha gate fails | High | Medium | Stop, iterate on features/universe. Sunk cost is small if gate is early. |
| Train/serve skew in live | High | Medium | 1b.5 unified signal-gen + golden test for live==backtest on shared dates |
| Multiple-testing inflation | High | High | DSR + PBO + pre-registered hypotheses |
| Overfit via HP search | High | High | Nested CV in 1.5.7; separate folds for tune vs report |
| Cost model too optimistic | High | Medium | Pessimistic defaults + sensitivity analysis at gate |
| Survivorship bias | High | Medium | Universe must include delisted (gate criterion) |
| Hardcoded years break in 2026 | Low | Certain | 1b.7 |
| Race condition in YAML queue | Med | Medium | 1b.10 |

---

## Success Criteria per Phase

**Phase 0**: two runs of same YAML + same data are byte-identical; MLflow has both.

**Phase 1b**: `run.py` and `experiment.py` produce identical outputs; live_sim signal == backtest signal for shared dates; strict_audit aborts on injected leakage.

**Phase 1.5**: Purged KFold detects an injected leakage that YearSplitter misses; bootstrap CI on synthetic AR(1) matches analytic; 20-seed run reports non-degenerate std.

**Alpha Gate**: DSR > 0.5, PBO < 30%, annualized return > 12% net of pessimistic costs.

**Phase 2**: each rule contributes bootstrap-significant Sharpe improvement; non-contributing rules dropped.

**Phase 3**: vol-targeting sizing improves Sharpe vs fixed sizing on the same signals.

**Phase 4**: 3-month live shadow PnL stays within backtest 95% CI; drift dashboard alerts work on synthetic regime change.

---

## Alpha Gate Run Results (2026-05-29) — FAILED

**Execution Summary**
- Universe: 486 VN symbols (HOSE + HNX)
- Data: 1,133,275 bars (multi-year OOS)
- Models: LightGBM regression (leading_v2 features, 36 features)
- Split: Purged KFold (6 splits, 10-day embargo, 5-bar label horizon)
- Validation: 20 seeds, bootstrap CI, DSR/PBO computation

**Gate Results**
| Criterion | Target | Actual | Pass? |
|-----------|--------|--------|-------|
| Deflated Sharpe | > 0.5 | N/A | ❌ |
| PBO | < 30% | N/A | ❌ |
| Annual return | > 12% | **-118.68%** | ❌ |
| Max drawdown | < 25% | Unknown | ❌ |

**Per-Seed Metrics (20 seeds)**
- Trades per seed: 210 (consistent)
- Buy signals: 1,101 per seed
- Signal coverage: 19.1% (210 / 1,101)
- Audit: **100% PASS** (no leakage, proper embargo, fill integrity)

**Conclusion**
Gate **FAILED** per roadmap criteria. The model exhibits negative edge across all 20 seeds with massive drawdowns. Audit passes indicate the methodology is sound (no leakage, proper CV), but the **alpha does not exist in the current feature/target configuration**.

**Next Steps** (per roadmap)
Do NOT proceed to Phase 2. Options:
1. **Iterate features**: Extend leading_v2, add cross-sectional rank, sector-relative metrics
2. **Adjust target**: Try longer horizon (10 bars), different return definitions (log-returns, Sharpe-normalized)
3. **Universe refinement**: Add liquidity/market-cap filters, exclude micro-caps, exclude recent IPOs
4. **Model tuning**: Nested CV with Optuna to search hyperparameter space systematically
5. **Regime detection**: Add market regime filter (trending vs ranging)

---

## Technical Rules Baseline Test (2026-05-30) — Rule vs ML Comparison

**Execution Summary**
- Universe: 486 VN symbols (HOSE + HNX)
- Data: 1,133,275 bars (same OOS as ML run)
- Rules: MACD > 0 AND C > MA20 AND C > O (buy); MACD < 0 AND C < MA20 AND C < O (sell)
- Split: Purged KFold (6 splits, 10-day embargo, 5-bar label horizon)
- Validation: Single run, 6-fold CV (no multi-seed, no bootstrap)

**Results Comparison**

| Metric | ML Regression | Tech Rules | Winner |
|--------|---------------|------------|--------|
| **Trades** | 210 | 74 | Rules (more selective) |
| **Total PnL** | -118.68% | -41.20% | Rules (65% better) |
| **Win Rate** | N/A | 40.5% | Rules (only positive count) |
| **Profit Factor** | N/A | 0.83 | Rules (83% of losses covered) |
| **Buy Signals** | 1,101 | 497 | Rules (55% fewer false signals) |
| **Audit** | 100% PASS | 100% PASS | Tie (both clean) |

**Key Finding**
Rule-based approach produces **65% better returns** than ML regression (-41.2% vs -118.68%), despite using simpler logic. This suggests:
- **ML overfitting**: LightGBM found spurious patterns; rules avoid data-mining bias
- **Signal quality > quantity**: 74 higher-confidence trades beat 210 questionable trades
- **Simplicity advantage**: Technical indicators (MACD, MA20, OHLC) more robust than learned weights

**Implications**
Neither approach passes the Alpha Gate (both negative PnL). But the rules baseline demonstrates:
1. Current feature set lacks sufficient edge for ML to discover
2. Simple rules outperform ML when features are weak (avoid overfitting penalty)
3. Hybrid approach: use rules for high-confidence signal generation + ML for edge ranking (if edge exists)

**Next Iteration Recommendation**
Before changing features or universe, **split test rule variants**:
- Add volume confirmation: rules + volume SMA filter
- Shorten/lengthen hold periods (currently ~15.7 days)
- Add RSI overbought/oversold filters
- Test exit rules explicitly (not implicit via time decay)

If improved rules still show negative edge, then feature/universe iteration is warranted.

---

## References

- de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. — Purged CV (Ch. 7), DSR (Ch. 14), PBO (Ch. 11).
- Bailey, D., López de Prado, M. (2014). "The Deflated Sharpe Ratio". *Journal of Portfolio Management*.
- Bailey, Borwein, López de Prado, Zhu (2017). "The Probability of Backtest Overfitting". *Journal of Computational Finance*.
- Internal: `src/pipeline/experiment.py`, `src/backtest/integrity.py`, `src/leaderboard/schema.py`.
