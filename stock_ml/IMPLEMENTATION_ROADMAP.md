# Implementation Roadmap: Research-Grade Trading Model System

**Status**: Phase 1a complete · Phase 0 + 1b in progress
**Last Updated**: 2026-05-29
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

## Phase 0: Foundation Infrastructure (Week 0–1)

Setup before any research work. None of these require alpha — they protect you from wasting time later.

### 0.1 Experiment tracking
- **Tool**: MLflow (local file backend) or Aim.
- **Track per run**: config YAML, git commit hash, data fingerprint, all metrics, output artifacts.
- **Deliverable**: `src/tracking/mlflow_logger.py` + integration in `run_experiment()`.

### 0.2 Reproducibility primitives
- **Data fingerprint**: `sha256(sorted(symbols) + date_range + last_modified_ts)` written to every summary JSON.
- **Code version**: auto-log `git rev-parse HEAD` and dirty flag.
- **Seed propagation**: centralized `set_global_seed(seed)` covering numpy, random, lightgbm, xgboost, sklearn, tf. Replace hardcoded `random_state=0` in model wrappers.

### 0.3 Structured logging
- Replace `print(...)` with loguru / std logging.
- Format: `[{run_id}] [{exp_name}] [{level}] message`.
- Required when running parallel experiments.

### 0.4 Feature cache (mini feature store)
- **Path**: `cache/features/{feature_set}/{data_hash}.parquet`.
- **Key**: `(feature_set_name, data_fingerprint)`.
- **Behavior**: compute once, reuse across experiments using the same data + feature set.

### 0.5 Output layout standardization
```
results/
└── {experiment_name}/
    └── {run_id}/                # = {YYYYMMDD-HHMMSS}-{git_short_hash}
        ├── config.yaml          # frozen copy of input config
        ├── data_fingerprint.txt
        ├── trades.csv
        ├── signals.csv
        ├── stats/
        │   ├── aggregate.json
        │   ├── daily.csv
        │   ├── yearly.csv
        │   └── symbol.csv
        ├── audit.json           # leakage audit (machine-readable)
        ├── metrics.json         # all metrics with CIs
        └── mlflow_run_id.txt
```

### 0.6 Golden + leakage regression tests
- **Golden test**: fixed input data + fixed config → assert exact `trades.csv` hash. Catches accidental behavior changes.
- **Leakage regression test**: inject a future-looking feature → audit must flag it (`fail=True`). Catches silent leakage regressions.

### Phase 0 verification
- Two runs of same config + same data → identical metrics.
- MLflow logs created on every run.
- Feature cache hit on 2nd run of same `(feature_set, data)`.
- Golden test passes; leakage regression fails as expected when future-feature injected.

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

### 1b.1 Label semantic fix (BLOCKER)
Current `train_fold` converts `target ∈ {-1, 0, 1}` to binary as:
```python
y_entry = (target == 1)   # buy vs (sell ∪ neutral)
y_exit  = (target == -1)  # sell vs (buy ∪ neutral)
```
This is semantically wrong: the negative class mixes two incompatible regimes, hurting learnability and inflating noise.

**Required fix** — choose ONE and document:
- **Option A (recommended)**: drop `target == -1` rows when training entry model; drop `target == +1` rows when training exit model. Clean binary signal.
- **Option B**: switch entry model to multi-class `{sell, neutral, buy}` and use `predict_proba(buy)` for ranking.

Decision must be recorded in `src/pipeline/experiment.py` docstring + roadmap.

### 1b.2 Vectorize signal generation (BLOCKER)
Current `train_fold` runs `exit_model.predict(X_test[idx:idx+1])` inside a Python `for` loop over test rows → O(N) Python overhead. On 100k test rows this is minutes wasted.

**Fix**:
```python
entry_pred = entry_model.predict(X_test)
exit_pred  = exit_model.predict(X_test) if exit_model else np.zeros_like(entry_pred)
signals = np.where(entry_pred == 1, 1, np.where(exit_pred == 1, -1, 0))
```

### 1b.3 Use `predict_proba` as score
- Backtest currently only sees binary signals.
- Change `signals_df` schema to `[symbol, date, signal, score]` where `score = predict_proba(buy)` ∈ [0, 1].
- This enables confidence-weighted sizing in Phase 3 without re-running models.

### 1b.4 Unify `run.py` ↔ `experiment.py`
- ~80% logic duplicated. Maintaining both doubles cost and risks divergence.
- **Action**: refactor `pipeline/run.py` into a thin wrapper that builds an `ExperimentConfig` and calls `run_experiment()`. Single source of truth.

### 1b.5 Unify signal generation across backtest and live_sim
- Risk: live and backtest computing signals via slightly different code paths → train/serve skew (a top-3 cause of live underperformance).
- **Action**: extract `generate_signal(model, features) → signal, score` to `src/signals/core.py`. Use in both `backtest/engine.py` and `live_sim/signals.py`.

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

### Phase 1b verification
- Same YAML run twice → byte-identical `trades.csv` (deterministic).
- `run.py` and `experiment.py` produce identical outputs for matching config.
- `live_sim` signal for a fixed date matches backtest signal for the same date.
- Vectorized `train_fold` ≥ 10× faster than loop version.
- Strict audit aborts when fed leaked features.

---

## Phase 1.5: Research Methodology (Week 2–4)

This is the **most important phase** and the one missing from the original roadmap. Without it, the leaderboard in Phase 3 ranks noise.

### 1.5.1 Purged K-Fold + Embargo splitter
- Reference: de Prado, *Advances in Financial ML*, Ch. 7.
- **Why**: walk-forward by year leaks via label horizon (a 5-bar forward-return target at the end of train overlaps the first 5 bars of test).
- **Deliverable**: `src/data/splitter.py::PurgedKFoldSplitter(n_splits, embargo_days, label_horizon)`.
- Keep `YearSplitter` for legacy comparison; switch default in YAML to `purged_kfold`.

### 1.5.2 Bootstrap confidence intervals
- Every aggregate metric (Sharpe, total return, win rate, profit factor) gets a 95% CI from block bootstrap (block size ≈ avg holding period).
- **Deliverable**: `src/evaluation/bootstrap.py::bootstrap_metric(returns, fn, n_iter=1000, block_size=20)`.

### 1.5.3 Deflated Sharpe Ratio (DSR)
- Adjusts Sharpe for multiple testing across the experiment population.
- **Why**: if you try 50 YAMLs and pick the best Sharpe, naïve Sharpe is biased upward by ~ √(2 ln N) std deviations.
- **Deliverable**: `src/evaluation/dsr.py::deflated_sharpe(sharpe, n_trials, returns_skew, returns_kurt, sample_length)`.

### 1.5.4 Probability of Backtest Overfitting (PBO)
- Reference: Bailey, Borwein, López de Prado, Zhu (2017).
- **Why**: quantifies how often the in-sample best model becomes out-of-sample below-median.
- **Deliverable**: `src/evaluation/pbo.py` — Combinatorially Symmetric CV variant.

### 1.5.5 Multi-seed variance
- Every experiment runs `n_seeds: 20` times with different seeds (data sampling, model init).
- Report **mean ± std** for all metrics, not point estimates.
- Reject any strategy whose Sharpe std > 0.5 × its mean.

### 1.5.6 Hypothesis pre-registration
- YAML field `hypothesis:` (required) describes the economic intuition before running.
- Log to MLflow. Track post-run hit rate of pre-registered hypotheses over time. A team consistently below 30% is data-mining.

### 1.5.7 Hyperparameter search
- Use **Optuna** with **nested CV** (outer = Purged KFold for OOS, inner = Purged KFold for HP tuning).
- Do NOT use the same folds for HP tuning and reporting.
- **Deliverable**: `src/pipeline/tune.py::tune_experiment(cfg, n_trials=100)`.

### Phase 1.5 verification
- Purged KFold on a known-leakage dataset → audit catches the embargo violation.
- Bootstrap CI for a synthetic AR(1) return series matches analytic CI within 5%.
- DSR on a known overfit example reproduces published numerical value.
- 20-seed run reports non-trivial std (not all seeds collapse to the same number).

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

## Flow Issues in Current Codebase (Tracked)

These are concrete code-level problems found during review. Issues #1–#3 are blockers for Phase 1.5; the rest are scheduled into Phase 0 / 1b.

| # | Issue | File / Reference | Phase |
|---|-------|------------------|-------|
| 1 | Duplicate logic in `run.py` and `experiment.py` | `src/pipeline/{run,experiment}.py` | 1b.4 |
| 2 | `train_fold` calls `exit_model.predict` per row in a Python loop | `experiment.py:126-135` | 1b.2 |
| 3 | Label conversion `(target==1)` mixes neutral + sell into negative class | `experiment.py:109,115` | 1b.1 |
| 4 | Features recomputed on every run | `pipeline/experiment.py:178` | 0.4 |
| 5 | Hardcoded `first_test_year=2020, last_test_year=2025` | `run.py:42-43` | 1b.7 |
| 6 | No data fingerprint in summary | `experiment.py:258-294` | 0.2 |
| 7 | Backtest discards model `predict_proba` | `experiment.py:124-138` | 1b.3 |
| 8 | LightGBM `random_state=0` ignores `cfg.seed` | `models/registry.py:73` | 0.2 |
| 9 | `print(...)` everywhere; no run_id in logs | many files | 0.3 |
| 10 | Smoke tests don't assert metrics, no golden / leakage regression tests | `tests/` | 0.6 |
| 11 | `audit_report` failures don't abort the run | `experiment.py:242-243` | 1b.9 |
| 12 | Output path overwrites on second run with same name | `experiment.py:245-256` | 0.5 / 1b.8 |
| 13 | YAML queue lacks per-file locking | `scripts/run_experiments.py` | 1b.10 |
| 14 | No mid-fold resume on crash | `experiment.py:198-218` | 1b.11 |
| 15 | live_sim and backtest signal paths diverge | `live_sim/signals.py` vs `backtest/engine.py` | 1b.5 |

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
│   ├── features/              basic, registry, cache
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
| Phase 0 | 1 week | 🔨 IN PROGRESS | Infra smoke tests pass |
| Phase 1a | done | ✅ DONE | Registries import + instantiate |
| Phase 1b | 1–2 weeks | ⏳ NEXT | Deterministic E2E + unified signal path |
| Phase 1.5 | 2 weeks | ⏳ PLANNED | Methodology smoke tests |
| **Alpha Gate** | 1 week | ⏳ PLANNED | DSR > 0.5, PBO < 30% |
| Phase 2 | 2 weeks | ⏳ CONDITIONAL on gate | Exit + portfolio rules A/B-validated |
| Phase 3 | 2 weeks | ⏳ CONDITIONAL | Sizing improves Sharpe; leaderboard ranks |
| Phase 4 | 4+ weeks | ⏳ CONDITIONAL | 3-month shadow PnL within CI |

Total to live trading (assuming gate passes first attempt): **~13 weeks**. Original roadmap estimated 7+ weeks but skipped methodology and alpha validation, which are the highest-risk steps.

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

## References

- de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. — Purged CV (Ch. 7), DSR (Ch. 14), PBO (Ch. 11).
- Bailey, D., López de Prado, M. (2014). "The Deflated Sharpe Ratio". *Journal of Portfolio Management*.
- Bailey, Borwein, López de Prado, Zhu (2017). "The Probability of Backtest Overfitting". *Journal of Computational Finance*.
- Internal: `src/pipeline/experiment.py`, `src/backtest/integrity.py`, `src/leaderboard/schema.py`.
