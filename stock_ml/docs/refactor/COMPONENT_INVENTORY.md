# Component Inventory — Refactor V3 Phase 1

**Status:** Drafted 2026-05-01  
**Scope:** Audit current component boundaries for the Signal / Strategy / Execution split in [REFACTOR_V3_PLAN.md](REFACTOR_V3_PLAN.md).

## Strategy modules

### Entry signals / filters

| Module | Classification | Main component | Current behavior | Migration notes |
|---|---|---|---|---|
| [rule_signal.py](../../src/components/fusion/strategies/rule_signal.py) | `entry_signal`, `exit_rule` | `RuleSignalEntry`, `RuleSignalExit` | Rule-only MACD/MA/candle entry and matching rule exit. | Split into `strategy.entry_rules.rule_signal` and `strategy.exit_rules.rule_signal_exit`. |
| [entry/v19_entry_cascade.py](../../src/components/fusion/strategies/entry/v19_entry_cascade.py) | `entry_filter` + `entry_signal` | `V19EntryCascade` | Replays the v19_3 entry cascade, including rule/ML gating and legacy counters. | Decompose later into named entry filters; keep as compatibility rule while parity is still required. |

### Hold rules

| Module | Classification | Main component | Current behavior | Migration notes |
|---|---|---|---|---|
| [hold/long_horizon_carry.py](../../src/components/fusion/strategies/hold/long_horizon_carry.py) | `hold_rule` | `LongHorizonCarry` | Keeps a position through normal exits while long-horizon trend remains intact. | Maps to `strategy.hold_rules.long_horizon_carry`. |
| [hold/v19_signal_hold_guard.py](../../src/components/fusion/strategies/hold/v19_signal_hold_guard.py) | `hold_rule` | `V19SignalHoldGuard` | Guards signal-driven exits using confirm/carry/quality checks. | Maps to `strategy.hold_rules.signal_hold_guard`; still version-specific. |
| [core/min_hold_protection.py](../../src/components/fusion/strategies/core/min_hold_protection.py) | `hold_rule` | `MinHoldProtection` | Blocks soft exits before a configured minimum hold period. | Current file is in `core`, but target layer is `strategy.hold_rules`. |

### Exit rules

| Module | Classification | Main component | Current behavior | Migration notes |
|---|---|---|---|---|
| [core/hard_stop.py](../../src/components/fusion/strategies/core/hard_stop.py) | `exit_rule` | `HardStopExit` | Exits when cumulative return breaches hard stop. | Maps to `strategy.exit_rules.hard_stop`. |
| [core/signal_hard_cap.py](../../src/components/fusion/strategies/core/signal_hard_cap.py) | `exit_rule` | `SignalHardCapExit` | Exits when signal-relative return breaches hard cap. | Maps to `strategy.exit_rules.signal_hard_cap`. |
| [core/fast_exit_loss.py](../../src/components/fusion/strategies/core/fast_exit_loss.py) | `exit_rule` | `FastExitLossLegacy` | Legacy v19_3 fast loss exit. | Version-specific; migrate after parity tests isolate it. |
| [core/atr_stop.py](../../src/components/fusion/strategies/core/atr_stop.py) | `exit_rule` | `AtrStopLoss` | ATR-based stop loss. | Maps to `strategy.exit_rules.atr_stop`. |
| [core/peak_protect_dist.py](../../src/components/fusion/strategies/core/peak_protect_dist.py) | `exit_rule` | `PeakProtectDist` | Protects first peak after large profit with distribution/volume conditions. | Maps to `strategy.exit_rules.peak_protect_dist`. |
| [core/peak_protect_ema.py](../../src/components/fusion/strategies/core/peak_protect_ema.py) | `exit_rule` | `PeakProtectEma8Streak` | Protects profit after EMA8 weakness streak. | Maps to `strategy.exit_rules.peak_protect_ema`. |
| [core/adaptive_trailing.py](../../src/components/fusion/strategies/core/adaptive_trailing.py) | `exit_rule` | `AdaptiveTrailing` | Tiered trailing stop based on max profit and trend strength. | Maps to `strategy.exit_rules.adaptive_trailing`. |
| [core/zombie_exit.py](../../src/components/fusion/strategies/core/zombie_exit.py) | `exit_rule` | `ZombieExit` | Exits stale low-return positions outside strong trend. | Maps to `strategy.exit_rules.zombie_exit`. |
| [core/v22_fast_exit.py](../../src/components/fusion/strategies/core/v22_fast_exit.py) | `exit_rule` | `V22FastExit` | v22 fast-loss exit with profile/volatility confirmations. | Rename by behavior after v22 parity is protected. |
| [core/ma_cross_hybrid_exit.py](../../src/components/fusion/strategies/core/ma_cross_hybrid_exit.py) | `exit_rule` | `MaCrossHybridExit` | Hybrid MA-cross exit with strong-uptrend/profit gating. | Maps to `strategy.exit_rules.ma_cross_hybrid`. |
| [core/profit_lock.py](../../src/components/fusion/strategies/core/profit_lock.py) | `exit_rule` | `ProfitLock` | Locks profit when max profit was high but current return weakens. | Maps to `strategy.exit_rules.profit_lock`. |
| [core/v22_hard_cap.py](../../src/components/fusion/strategies/core/v22_hard_cap.py) | `exit_rule` | `V22HardCap` | v22 hard-cap exit with adaptive floor/multiplier logic. | Rename by behavior after v22 parity is protected. |
| [core/model_b_exit.py](../../src/components/fusion/strategies/core/model_b_exit.py) | `exit_rule` consuming signal | `ModelBExit` | Exits when Model B predicts sell after minimum hold. | `signals.exit_model` owns prediction; `strategy.exit_rules.model_b_exit` owns the decision rule. |
| [core/early_loss_cut.py](../../src/components/fusion/strategies/core/early_loss_cut.py) | `exit_rule` | `EarlyLossCutExit` | Cuts early positions when PnL is below threshold within max hold days. | Maps to `strategy.exit_rules.early_loss_cut`. |
| [core/hap_preempt.py](../../src/components/fusion/strategies/core/hap_preempt.py) | `exit_rule` | `HapPreemptExit` | Preempts hard-after-profit style drawdown from a prior profit trigger. | Maps to `strategy.exit_rules.hap_preempt`. |

## Champion runners

| Runner | Backing implementation | Quirks to preserve during migration |
|---|---|---|
| [rule_runner.py](../../src/components/runners/rule_runner.py) | Local rule baseline loop | No ML signal; can remain a special runner as allowed by the V3 plan. |
| [v19_3_runner.py](../../src/components/runners/v19_3_runner.py) | Legacy v19_3 path | Own entry/hold/exit stack predates current champion YAML; uses `mods`, prediction cache and legacy trade dataframe. |
| [v22_runner.py](../../src/components/runners/v22_runner.py) | v34 lineage wrapper with v22 defaults | Loads `mods`/`params` from model config, can enable Model B exit, preserves v22 trade dataframe shape. |
| [v32_runner.py](../../src/components/runners/v32_runner.py) | `run_lineage` + `experiments.run_v32_final.backtest_v32` | Registry overrides target default to `leading_v3`/early-wave 5% gain/4% loss; legacy experiment function is still the execution engine. |
| [v34_runner.py](../../src/components/runners/v34_runner.py) | v34 lineage/cache helpers | Central bridge between prediction cache and legacy experiment backtests. |
| [v35b_runner.py](../../src/components/runners/v35b_runner.py) | `run_lineage` + `experiments.run_v34_final.backtest_v35b` | Many flat `params` mix entry filters, hold guards, exits, HAP and symbol tuning. |
| [v37a_runner.py](../../src/components/runners/v37a_runner.py) | `run_lineage` + `experiments.run_v37a.backtest_v37a` | Uses v37a entry reason and legacy v37a execution semantics. |
| [v37a_exit_runner.py](../../src/components/runners/v37a_exit_runner.py) | `run_lineage` + `experiments.run_v37a.backtest_v37a` | Same backing backtest as v37a but separate entry reason/config; current champion has dual target and exit model enabled. |
| [v37d_runner.py](../../src/components/runners/v37d_runner.py) | `run_lineage` + `experiments.run_v37d.backtest_v37d` | Legacy v37d execution is still external to the component layers. |
| [v39d_runner.py](../../src/components/runners/v39d_runner.py) | `run_lineage` + `experiments.run_v39d.backtest_v39d` | Per-version exit behavior remains embedded in legacy experiment backtest. |
| [v42_a_runner.py](../../src/components/runners/v42_a_runner.py) | `run_lineage` + `experiments.run_v42.backtest_v42` | v42 semantics are delegated to legacy experiment code, not declarative strategy config. |
| [runner_registry.py](../../src/components/runners/runner_registry.py) | Runner dispatch metadata | Registry currently maps version keys to legacy experiment backtest functions and entry reasons. |
| [_lineage_v34.py](../../src/components/runners/_lineage_v34.py) | Shared lineage bridge | Builds entry/exit predictions, merges `mods`/`params`, injects costs, then calls the selected legacy backtest. |

## Champion YAML key mapping

| Current key/pattern | Target layer | Notes |
|---|---|---|
| `name` | metadata | Experiment identity, not part of signal/strategy/execution. |
| `strategy` | strategy metadata | Currently names a runner/backtest family; in V3 it should not carry execution implementation by itself. |
| `runner` | execution | Current imperative execution entrypoint; eventually replaced by `execution.backtester`. |
| `components.features` | signals | Feature set for signal generation. |
| `components.target.*` | signals | Label/target definition for entry model training. |
| `components.entry_model.*` | signals | Entry model backend/device/extras. |
| `components.exit_model.enabled/type/forward_window/loss_threshold/extras` | signals + strategy | Model definition belongs to `signals.exit_model`; acting on its sell prediction belongs to an exit rule such as `strategy.exit_rules.model_b_exit`. |
| `split.*` | execution / training protocol | Walk-forward timing controls prediction generation and backtest windows. |
| `mods.a`-`mods.j` | strategy, mostly ambiguous | Opaque legacy switches; must be replaced by named entry/hold/exit rules when each champion is migrated. |
| `fusion.entry` | strategy.entry_rules | Current declarative entry rule list. |
| `fusion.hold` | strategy.hold_rules | Current declarative hold rule list. |
| `fusion.active_exit`, `fusion.force_exit` | strategy.exit_rules | Current declarative exit rule lists. |
| `params.v22_fast_exit_*`, `params.v28_early_loss_cut*`, `params.v30_signal_exit_defer`, `params.v31_short_hold_exit_filter` | strategy.exit_rules | Exit decision thresholds mixed into flat params. |
| `params.v22_hard_cap_*`, `params.v27_hardcap_two_step`, `params.v31_hardcap_after_profit`, `params.v32_hap_preempt`, `params.v31_hap_*`, `params.v32_hap_*` | strategy.exit_rules | Hard-cap/HAP behavior should become named exit rules. |
| `params.patch_long_horizon`, `params.v27_trend_persistence_hold` | strategy.hold_rules | Hold-through-exit behavior. |
| `params.v26_relaxed_entry`, `params.v26_skip_choppy`, `params.v28_early_wave_filter`, `params.v35_rule_override`, `params.v35_single_bar_signal`, `params.v35_skip_price_proximity` | strategy.entry_rules | Entry filtering/override behavior. |
| `params.patch_rule_ensemble`, `params.v26_strong_rule_ensemble`, `params.v27_rule_priority` | strategy.entry_rules / strategy arbitration | Rule-vs-ML arbitration; split by actual call site during Phase 3. |
| `params.patch_symbol_tuning`, `params.v27_dynamic_score5_penalty`, cooldown params | strategy / execution ambiguous | Symbol/profile and cooldown behavior may affect entries, exits and sizing; needs call-site audit per champion. |
| cost keys passed by runner (`initial_capital`, `commission`, `tax`) | execution | Currently function kwargs injected into `params`; target should be `execution.capital` and `execution.costs`. |

## Current champion configs

| YAML | Signals | Strategy keys | Execution keys | Ambiguities |
|---|---|---|---|---|
| [v22.yaml](../../config/experiments/champions/v22.yaml) | `features=leading_v2`, `target=trend_regime`, LightGBM entry, null exit model | `mods`, `fusion.*`, v22 fast-exit/hard-cap params | `runner=components.runners.run_v22` | `mods` letters and flat v22 params need named rules. |
| [v22_exit_b.yaml](../../config/experiments/champions/v22_exit_b.yaml) | `features=leading_v2`, `target=early_wave`, LightGBM entry, enabled LightGBM exit model | Same v22 `mods`, `fusion.*`, fast-exit/hard-cap params | `runner=src.components.runners.v22_runner`, `split.*` | Exit model config mixes signal definition with exit-rule policy. |
| [v35b.yaml](../../config/experiments/champions/v35b.yaml) | `features=leading_v4`, `target=early_wave`, LightGBM entry, null exit model | Heavy `mods`, `fusion.*`, and many versioned params for entry/hold/exit | `runner=components.runners.run_v35b` | Most params are behavior names but still flat and version-prefixed. |
| [v37a_exit.yaml](../../config/experiments/champions/v37a_exit.yaml) | `features=leading_v4`, `target=early_wave_dual`, LightGBM entry, enabled LightGBM exit model | `mods` only; no `fusion`/`params` in current file | `runner=src.components.runners.v37a_exit_runner`, `split.*` | Behavior depends on runner/legacy backtest defaults not visible in YAML. |

## Phase 2 migration implications

- Add new typed sections beside current fields: `signals`, `strategy`, and `execution`.
- Keep an in-memory adapter from `components`/`mods`/`params`/`fusion` to the new schema until champion parity is verified.
- Treat Model B as two concepts: `signals.exit_model` for prediction and `strategy.exit_rules.model_b_exit` for action.
- Do not delete or rename legacy runner fields until the golden tests pass under the new schema.
