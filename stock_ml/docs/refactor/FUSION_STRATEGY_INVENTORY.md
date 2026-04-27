# Fusion Strategy Inventory — Phase 2.1 audit

Phân loại toàn bộ flag fusion từ legacy code vào 4 layer của kiến trúc mới.
Output này là input cho Phase 2.2 (base interface) và 2.3+ (port từng strategy).

## Tổng quan

- **80 flag** xuất hiện trong `config/models.yaml` (qua 61 entries).
- **~45 flag ẩn** trong `src/backtest/engine.py` (v29_*, v30_*, v33_*, v38_*, v39_*) — xuất hiện qua `cfg.get(...)` mặc định, có thể bật từ YAML.
- **3 entry path đặc biệt** không qua flag: `quick_reentry`, `consolidation_breakout`, `vshape_bypass` (luôn-on trong engine).
- **1 entry path ẩn**: `v28_wave_acceleration_entry` (entry path nhưng nằm chung file flag exit của V28).
- **Rule baseline** (`compare_rule_vs_model.backtest_rule`): 1 strategy độc lập (MACD_hist>0 + Close>MA20 + Close>Open) — không qua engine; đã port ở Phase 2.3a với exact golden parity.

## Định nghĩa 4 layer

Theo `ARCHITECTURE.md` §4.6:

| Layer | Lifecycle | Quyết định |
|-------|-----------|------------|
| `pre_entry` | Trước khi cân nhắc vào lệnh | Block / cho phép entry signal |
| `entry` | Tại bar có entry signal | Decide có vào không + position size |
| `hold` | Trong khi giữ vị thế | Modify hold duration / cooldown / re-arm |
| `exit_override` | Tại bar có exit signal hoặc protective trigger | Override / defer / confirm exit |

> **Lưu ý ordering**: `exit_override` chạy theo PRIORITY trong engine cũ (xem §3 bên dưới). Phase 2 phải giữ exact ordering để regression match.

---

## 1. Mapping flag → fusion strategy

### 1.1 Layer `pre_entry` (filter / block entry)

| Flag | Strategy class (đề xuất) | Trigger condition | Versions |
|------|---------------------------|--------------------|----------|
| `mod_g` (Bear Regime Defense) | `BearRegimeDefenseFilter` | `sma20<sma50 + close<sma50 + ret_60d<-10%` | tất cả v22+ |
| `mod_j` (Anti-chop) | `AntiChopFilter` | `\|sma20/sma50-1\|<2% + \|ret_20d\|<6% + bb<0.45 + trend=weak` | tất cả v22+ |
| `patch_noise_filter` | `NoiseFilter` | `trend=weak + entry_score<3 + ret_5d>3%` | v22, v23+, V32 |
| `patch_rule_ensemble` | `RuleEnsembleEntry` (entry) | rule_signal=1 và ML=0 → vào với size 0.30 | v22+ — *đã đẩy sang `entry` vì thay đổi quyết định vào lệnh* |
| `v26_skip_choppy` | `SkipChoppyFilter` | regime_cfg.choppy_regime=True | V26+ champions |
| `v26_relaxed_entry` | `RelaxedRuleEntry` (entry) | `trend=strong + rule_consecutive>=3` lỏng tiêu chí entry | V26 |
| `v26_strong_rule_ensemble` | `StrongRuleEnsembleEntry` (entry) | `rule_consecutive>=3 + trend≥moderate` → vào size 0.35 | V26+, V27, V32 |
| `v27_selective_choppy` | `SelectiveChoppyFilter` | choppy + thiếu vol/quality → block | V27+ |
| `v27_rule_priority` | `RulePrioritySymbolEntry` (entry) | `sym in RULE_PRIORITY_SYMBOLS + rule_consecutive>=2 + trend≥moderate` | V27+ |
| `v28_early_wave_filter` | `EarlyWaveMaturityFilter` | `days_since_low_10>7 + ret_5d>8%` HOẶC `wave_exhausted` → block | V28+ |
| `v28_crash_guard` | `CrashGuardFilter` | `ret_20d<-12%` → block | V28+ |
| `v28_wave_acceleration_entry` | `WaveAccelerationEntry` (entry) | `ret_2d>0.015 + ret_3d>0.02 + ret_5d<0.05 + ret_acceleration>0.005` → vào size 0.40 | V28+ |
| `v29_breakout_strength_entry` | `BreakoutStrengthEntry` (entry) | clean break recent_high_10 + dry-then-spike vol → vào size 0.45 | V29+ champions |
| `v29_relstrength_filter` | `RelativeStrengthFilter` | `trend≠strong + ret_20d<v29_rs_ret20_threshold` | V29+ |
| `v30_peak_proximity_filter` | `PeakProximityFilter` | gần đỉnh 20d + đã rally → block | V30+ |
| `v30_rally_extension_filter` | `RallyExtensionFilter` | `ret_10d>v30_rally10_hard_block` HOẶC `ret_20d>v30_rally20_hard_block` | V30+ |
| `v30_pullback_only_entry` | `PullbackOnlyFilter` | `(high_5d-close)/high_5d < v30_pullback_min_pct` | V30+ |
| `v31_peak_chasing_guard` | `PeakChasingGuard` | `ret_5d>thresh + dist_sma20*100>thresh` → skip/half | V31+ |
| `v31_profile_sizing` | `ProfileSizingAdjuster` (entry) | nhân `position_size *= profile_mult` | V31+ |
| `v33_recovery_peak_filter` | `RecoveryPeakFilter` | `ret_10d>thresh + dist_sma20>thresh + trend≠strong` | V33+ |
| `v38d_fomo_filter` | `AntiFomoFilter` | `ret_5d>thresh OR dist_sma20>thresh` (trừ breakout/vshape) | V38d+ |
| `v22_cooldown_after_loss` (constant) | `CooldownAfterLoss` | block entry trong N bar sau loss | tất cả |
| `v35_skip_price_proximity` | `PriceProximityBypass` | bỏ qua check `\|close/last_exit_price-1\|<3%` khi rule trigger | V35+ |

### 1.2 Layer `entry` (quyết định vào lệnh + sizing)

| Flag / behavior | Strategy class (đề xuất) | Mô tả | Versions |
|-----------------|---------------------------|-------|----------|
| **default ML-only** | `MlOnlyEntry` | `pred==1 → enter` (canonical) | tất cả ML |
| `quick_reentry` (always-on) | `QuickReentryAfterTrailing` | sau trailing_stop trong N bar + trend≥moderate + macd>0 + close>sma20 | tất cả |
| `consolidation_breakout` (always-on) | `ConsolidationBreakoutEntry` | tight range + break + vol_ok | tất cả |
| `mod_e` `secondary_breakout` | `SecondaryBreakoutEntry` | breakout 5d trong uptrend macro | v19+ |
| `mod_a` `vshape_bypass` | `VShapeReboundEntry` | drop_from_peak + bullish bar + oversold + vol_ok | v19+ |
| `mod_f` `bo_quality_ok` | `BreakoutQualityFilter` | macd_hist>0 + close>open + vol>1.5×avg | v19+ |
| `v35_single_bar_signal` | `SingleBarSignalEntry` | bỏ điều kiện `prev_pred=1` khi `rule_trigger_now` | V35+ |
| `v35_rule_override` | `RuleOverrideMinScore` | rule_trigger + entry_score≥min → override các filter | V35+ |
| `v35_hybrid_entry` | `HybridRuleEntry` | ML=0 nhưng rule_trigger → vào với size `v35_hybrid_size` | V35+ |
| `v26_min_position` | `MinPositionThreshold` | block khi `position_size<0.28` | V26+ |
| `v26_score5_penalty` | `Score5Penalty` | `entry_score==5 → size *= 0.75` | V26+ |
| `v27_dynamic_score5_penalty` | `DynamicScore5Penalty` | nhân nhiều multiplier theo sym/trend/bb/rule_consec | V27+ |
| `v30_rally_position_scaling` | `RallyPositionScaling` | tier1: size×0.70, tier2: skip | V30+ |
| (always-on) entry alpha gate | `EntryAlphaGate` | tổng hợp `wp/dp/rs/vs/hl/score` → min_score check | tất cả ML |
| (always-on) ATR-aware sizing | `AtrAwareSizing` | giảm size khi atr_ratio cao | tất cả |

> **Ghi chú**: `patch_rule_ensemble` / `v26_strong_rule_ensemble` / `v27_rule_priority` / `v28_wave_acceleration_entry` / `v29_breakout_strength_entry` được liệt kê ở §1.1 nhưng phần code chạy ở `entry` layer (quyết định mới). Thiết kế Phase 2: giữ chúng ở **`entry` layer với priority cao hơn `MlOnlyEntry`** để map đúng semantics.

### 1.3 Layer `hold` (modify hold duration / cooldown / re-arm state)

| Flag | Strategy class (đề xuất) | Trigger | Versions |
|------|---------------------------|---------|----------|
| `v26_extended_hold` | `ExtendedHoldStrong` | trend=strong + cum_ret>5% → MIN_HOLD=12 | V26+ |
| `v27_trend_persistence_hold` | `TrendPersistenceHold` | strong + cum_ret>3% + close≥sma20 + macd_hist>-0.01 → MIN_HOLD=10 | V27+ |
| `v35_relax_cooldown` | `RelaxedCooldown` | dùng `v35_cooldown_after_big_loss/loss` thay 5/3 mặc định | V35+ |
| `v35_cooldown_after_big_loss` | param của `RelaxedCooldown` | giá trị 1/2 (V37a profile dispatch) | V35+ |
| `v35_cooldown_after_loss` | param của `RelaxedCooldown` | giá trị 0/1 | V35+ |
| `v30_signal_exit_defer` | `SignalExitDeferFlat` | sau exit signal: defer N bar nếu cum_ret≥thresh + trend≥moderate | V30+ |
| `v31_adaptive_defer` | `AdaptiveExitDefer` | tăng/reset counter theo ema8 + cum_ret | V31+ |
| `v33_signal_confirm_exit` | `SignalConfirmExit` | yêu cầu 2 bar liên tiếp signal → mới cho exit | V33+ |
| `v39a_signal_exit_min_hold` | `SignalExitMinHold` | block exit signal trước hold_days đủ | V39a+ |
| `v39a_rule_confirm_exit` | `RuleConfirmExit` | exit signal cần MACD<0 AND Close<MA20 | V39a2+ |
| `v39g_rule_confirm_min_maxprofit` | param của `RuleConfirmExit` | chỉ defer khi `price_max_profit≥thresh` | V39g+ |
| `v31_short_hold_exit_filter` | `ShortHoldExitFilter` | block exit signal khi `hold<min + cum_ret>min_pnl` | V31+ |
| `v33_rsi_oversold_block` | `RsiOversoldExitBlock` | block exit signal/hap_preempt khi RSI<thresh + cur_ret>-8% | V33+ |
| `v32_signal_weak_exit` | `WeakOversoldPassthrough` | KHÔNG block signal exit khi trend=weak + dist_sma20<thresh | V32+ |
| `v30_momentum_hold_override` | `MomentumHoldOverride` | block exit signal khi `cum_ret≥min + RSI<max + close>sma20+ema8 + macd>0` | V30+ |
| `v39d_rule_exit_symbols` | `PerSymbolRuleExit` | với sym trong set: thay signal exit bằng rule (close<sma20 OR macd_hist≤0) | V39d+ |
| `mod_h` confirmed exit (always-on) | `ConfirmedSignalExitScoring` | tính `bearish_score` vs `score_threshold` | tất cả |
| `mod_i` trend carry (always-on) | `TrendCarryOverride` | save exit khi cum_ret>3% + max_profit>6% + trend ok + close≥0.99×sma20 | tất cả |

> **Ghi chú**: nhiều flag "*_signal_exit_*" có nature là *modifier* của exit, nhưng theo design ARCHITECTURE.md (`hold` layer = modify hold duration), tôi đặt chúng vào `hold` layer. Phase 2.2 cần đặt rules: nếu fusion strategy ở hold layer trả về `modify_hold` → engine sẽ flip `new_position` từ 0 về 1. Đây chính là pattern engine cũ đang làm (block exit by re-asserting position).

### 1.4 Layer `exit_override` (force exit / hard cap / trail)

Order trong code (PRIORITY rất quan trọng — Phase 2 phải giữ exact):

| Priority | Flag | Strategy class | Trigger | Versions |
|----------|------|----------------|---------|----------|
| 0 | `HARD_STOP` (constant 0.08) | `HardStopExit` | cum_ret≤-8% | tất cả |
| 1 | (model_b path) | `MlExitModel` | `y_pred_exit[i-1]==1 + hold≥MODEL_B_MIN_HOLD` | exit_model versions; hiện preserve trained-but-dropped behavior để match golden — xem EXIT_MODEL_BUG.md |
| 2 | `v38b_stall_exit` | `StallExit` | hold≥min + max_profit<thresh + cur_ret≤thresh | V38b+ |
| 3 | `v38c_ha_exit` | `HeikinAshiBearExit` | hold≥min + cur_ret<thresh + (ha_bear OR ha_late+body_shrink) | V38c+ |
| 4 | `v38d_copilot_exit` | `RuleCopilotExit` | hold≥min + max_profit≥thresh + rule_off + cur<0.6×max | V38d+ |
| 5 | `v28_early_loss_cut` | `EarlyLossCut` | hold≤early_days + cur_ret≤thresh | V28+ |
| 6 | `v32_hap_preempt` | `HapPreempt` | max_profit≥trigger + cur_ret≤floor (+ optional v33_consec_drop, v39b min_hold/trigger) | V32+ |
| 7a | `v30_atr_aware_hardcap` | `AtrAwareHardCap` | thay thế fixed cap bằng `mult×atr_ratio_now` | V30+ |
| 7b | `v30_hardcap_two_step_v2` | `TwoStepHardCap` | step1: halve size, step2: full exit | V30+ |
| 7c | `v22_mode` (+ `v22_adaptive_hard_cap`) | `V22HardCap` | adaptive theo profile (high_beta vs std) hoặc fixed -12% | v22 |
| 7d | `patch_smart_hardcap` (+ `v26_wider_hardcap`, `v27_hardcap_two_step`, `patch_adaptive_hardcap`) | `SmartHardCapByTrend` | per-trend cap với confirm bars | v23+, V26+, V27+ |
| 7e | (default) `hard_cap_*` constants | `FlatHardCapByTrend` | per-trend cap không confirm | v23 baseline |
| 8 | `v30_regime_aware_hardcap` | `RegimeAwareHardCap` (post-check) | choppy_now → cap chặt hơn | V30+ |
| 9 | `fast_exit_*` (always-on) | `FastExitLoss` | per-trend threshold + hold conditions; có variant `v22_mode` smart skip | tất cả |
| 10 | ATR stop (always-on) | `AtrStopLoss` | `cum_ret≤-atr_stop` | tất cả |
| 11 | `v29_adaptive_peak_lock` | `AdaptivePeakLock` | `max_profit≥trigger → floor=keep×max_profit` | V29+ |
| 12 | `v31_hardcap_after_profit` | `HardCapAfterProfit` | max_profit≥5% → exit khi cur_ret≤floor | V31+ |
| 13 | `v32_weak_oversold_exit` | `TrendWeakOversoldExit` | trend=weak + hold≥min + dist_sma20<thresh + profit>0 | V32+ |
| 14 | `v32_dynamic_hc_dist` | `DynamicHardCapByDist` | dist_sma20<-8% → exit ở -7% | V32+ |
| 15 | `v32_profit_ratchet` | `ProfitRatchetExit` | max_profit≥trigger → floor=keep×max_profit | V32+ |
| 16 | `v33_trailing_ratchet` | `MultiTierTrailingRatchet` | 3 tier (12%/20%/35%) với keep tăng dần | V33+ |
| 17 | `v33_trend_rev_exit` | `TrendReversalExit` | max_profit≥thresh + close<ema8 (2d) + RSI<thresh | V33+ |
| 18 | `v29_profit_safety_net` | `ProfitSafetyNet` | max_profit≥trigger → never let cur_ret<0 | V29+ |
| 19 | `v29_hardcap_after_peak` | `HardCapAfterPeak` | max_profit≥trigger → exit khi cur_ret≤floor | V29+ |
| 20 | `v29_reversal_after_peak` | `ReversalAfterPeak` | max_profit≥trigger + ret_2d≤thresh | V29+ |
| 21 | `v29_atr_velocity_exit` | `AtrVelocityExit` | 2-day drop ≥ k×atr14 sau profit≥min | V29+ |
| 22 | `v30_chandelier_trail` | `ChandelierTrail` | close < `max_price - mult×atr14` sau profit trigger | V30+ |
| 23 | `mod_b` peak protect (variants) | `PeakProtectByMaxProfit` (+ variants `v22_mode`/`patch_pp_restore`/`patch_pp_2of3`/default) | max_profit≥0.20-0.25 + price<sma10 + heavy_vol + bearish_candle | tất cả |
| 24 | EMA8 peak protect (always-on) | `PeakProtectEma8Streak` | max_profit≥15% + 2d below ema8 + cur<0.75×max | tất cả |
| 25 | `v28_cycle_peak_exit` | `CyclePeakExit` | max_profit≥8% + ret_3d<-2% + cur<0.7×max + (heavy_vol OR ret_3d<-4%) | V28+ |
| 26 | hybrid_exit (always-on, conditional strong+profit) | `MaCrossHybridExit` | strong + cum_ret>5% + max_profit>8% + macd_bearish + close<sma20 | tất cả |
| 27 | adaptive trailing (always-on) | `AdaptiveTrailing` | giveback ≥ trail_pct theo max_profit tier | tất cả |
| 27.5 | `v29_tighter_trail_high_profit` | (modifier of AdaptiveTrailing) | tighten trail_pct khi max_profit≥high_profit_trigger | V29+ |
| 28 | Profit lock (always-on) | `ProfitLock` | max_profit≥0.12 + cum_ret<0.06 + trend≠strong | tất cả |
| 28.5 | `regime_cfg.disable_profit_lock_in_strong` | (modifier of ProfitLock) | bỏ profit_lock khi trend=strong | `patch_symbol_tuning` |
| 29 | Zombie exit (always-on) | `ZombieExit` | hold≥14 + cum_ret<0.01 + trend≠strong | tất cả |
| 30 | `patch_long_horizon` | `LongHorizonCarry` | save exit khi `ret_60d>30% + sma20>50>100 + days_above_sma50≥20 + cum_ret>15%` | v22+ |
| 31 | min_hold protection (constant MIN_HOLD=6) | `MinHoldProtection` | block exit khi hold<MIN_HOLD (trừ stop_loss/hard_stop/peak_protect/...) | tất cả |

### 1.5 Constants / hyperparams (không phải flag fusion, là param)

Đi kèm các strategy ở §1.4 — sẽ trở thành `params` trong YAML mới:

```
fast_exit_strong, fast_exit_moderate, fast_exit_weak, fast_exit_hb_buffer,
peak_protect_strong_threshold, peak_protect_normal_threshold,
hard_cap_weak, hard_cap_moderate_mult, hard_cap_moderate_floor,
hard_cap_strong_mult, hard_cap_strong_floor,
time_decay_bars, time_decay_mult,
v22_hard_cap_floor, v22_hard_cap_floor_hb, v22_hard_cap_mult_std, v22_hard_cap_mult_hb,
v22_fast_exit_threshold_std, v22_fast_exit_threshold_hb,
v22_fast_exit_skip_strong, v22_fast_exit_vol_confirm,
v26_hardcap_confirm_strong, v32_hap_pre_trigger, v32_hap_pre_floor,
v32_woe_dist_thresh, v32_woe_min_profit, v32_woe_hold_min,
v32_dhc_dist_thresh, v32_dhc_tight_cap, v32_pr_trigger, v32_pr_keep,
v32_swe_dist_thresh, v33_tr_*, v33_tre_*, v33_rpf_*, v33_hcd_*, v33_rob_*, v33_sce_*,
v28_early_loss_cut_threshold, v28_early_loss_cut_days,
v29_apl_trigger, v29_apl_keep, v29_atr_velocity_k, v29_atr_velocity_min_profit,
v29_high_profit_trigger, v29_high_profit_trail,
v29_reversal_peak_trigger, v29_reversal_ret2_threshold,
v29_rs_ret20_threshold, v29_peak_lock_high_beta_only,
v29_profit_safety_trigger, v29_hardcap_after_peak_trigger, v29_hardcap_after_peak_floor,
v30_peak_prox_dist_threshold, v30_peak_prox_rally10_min,
v30_rally10_hard_block, v30_rally20_hard_block, v30_pullback_min_pct,
v30_rps_tier1_rally, v30_rps_tier2_rally,
v30_sed_defer_bars, v30_sed_min_cum_ret, v30_mho_min_profit, v30_mho_rsi_max,
v30_chand_atr_mult, v30_chand_profit_trigger,
v30_atr_hc_mult, v30_atr_hc_floor, v30_atr_hc_ceiling,
v30_hc2_step1_loss, v30_hc2_step2_loss,
v30_rah_choppy_cap, v30_rah_trending_cap,
v31_pcg_*, v31_ad_*, v31_hap_*, v31_ps_*, v31_shef_*, v31_enriched_log,
v35_hybrid_size, v35_rule_override_min_score,
v38b_*, v38c_*, v38d_*,
v39a_signal_exit_min_hold, v39b_hap_min_hold, v39b_hap_trigger,
model_b_min_hold,
```

`model_b_min_hold`/`MlExitModel` chưa active trong parity baseline hiện tại. Phase 2.4 phải giữ `y_pred_exit` trained-but-dropped cho v37a_exit/v42_a để match golden; Model B fix là work riêng sau khi champion parity hoàn tất.

---

## 2. Mapping ngược: champion → fusion stack

11 champion + ML/non-ML, mỗi version cần stack nào (high-level — chi tiết params trong models.yaml):

### 2.1 v22 — `experiments/run_v22_final.backtest_v22` ✅ verified 2026-04-29
- Legacy engine: `backtest_unified` với `v22_mode=True`.
- Port thực tế dùng dedicated runner [src/components/runners/v22_runner.py](../../src/components/runners/v22_runner.py) để giữ exact parity thay vì tách YAML-executable stack ngay trong Phase 2.3c.
- [config/experiments/champions/v22.yaml](../../config/experiments/champions/v22.yaml) là declarative spec cho Phase 3, mirror runner order.
- Mapping đã verify:
  - Entry: `V19EntryCascade` giữ nguyên shared entry state (`MlOnlyEntry`, `mod_a/e/f`, ATR sizing, filters) để tránh lệch parity.
  - Force exit: `HardStopExit`, `V22HardCap`.
  - Active exit: `V22FastExit`, `AtrStopLoss`, `PeakProtectDist`, `PeakProtectEma8Streak`, `MaCrossHybridExit`, `AdaptiveTrailing`, `ProfitLock`, `ZombieExit`.
  - Hold/save exit: `MinHoldProtection`, `V19SignalHoldGuard`, `LongHorizonCarry`.
- Regression: [tests/regression/test_v22_parity.py](../../tests/regression/test_v22_parity.py) pass exact golden 1784 trades; [tests/components/fusion/test_v22_registry.py](../../tests/components/fusion/test_v22_registry.py) locks YAML registry/order/defaults.

### 2.2 v32 — `experiments/run_v32_final.backtest_v32` ✅ verified 2026-04-29
- Engine: `backtest_unified` (delegates to v31 chain)
- pre_entry: V22 stack + V26-V31 cumulative (skip_choppy, selective_choppy, peak_chasing_guard, profile_sizing, ...)
- entry: `MlOnlyEntry` + RuleEnsemble variants
- hold: V26 extended_hold, V27 persistence_hold, V30 sed/mho, V31 adaptive_defer, V31 shef
- exit_override: + `HapPreempt`, `TrendWeakOversoldExit`, `DynamicHardCapByDist`, `ProfitRatchetExit`
- V32-E (`signal_weak_exit`) — **không** trong v32 base config (gắn ở variants)
- Port thực tế dùng parity-first wrapper [src/components/runners/v32_runner.py](../../src/components/runners/v32_runner.py), reuse shared V34-lineage cache helper trong [src/components/runners/v34_runner.py](../../src/components/runners/v34_runner.py).
- Preserve legacy trained-but-dropped exit-model behavior: cache train exit model theo meta/config nhưng runner không truyền `y_pred_exit` vào `backtest_v32`.
- Regression: [tests/regression/test_v32_parity.py](../../tests/regression/test_v32_parity.py) pass exact golden 1347 trades.

### 2.3 v34 — `experiments/run_v34_final.backtest_v34` ✅ verified 2026-04-29
- = backtest_v32 (chỉ khác YAML params + feature_set=leading_v4)
- Stack: V32 stack — không thêm flag mới.
- Port thực tế dùng parity-first wrapper [src/components/runners/v34_runner.py](../../src/components/runners/v34_runner.py) quanh legacy `backtest_v34`, vì full V26-V32 cumulative stack chưa tách thành YAML-executable strategies trong Phase 2.4a.
- Preserve legacy trained-but-dropped exit-model behavior: prediction cache train exit model theo meta/config nhưng runner không truyền `y_pred_exit` vào `backtest_v34`, khớp `run_pipeline._run_backtest_from_cache` signature filtering.
- Regression: [tests/regression/test_v34_parity.py](../../tests/regression/test_v34_parity.py) pass exact golden 1323 trades.

### 2.4 v35b — `experiments/run_v34_final.backtest_v35b` ✅ verified 2026-04-29
- = backtest_v32 (chỉ khác YAML params + V35 entry-layer flags)
- Bật: `v35_relax_cooldown`, `v35_skip_price_proximity`, `v35_single_bar_signal`, `v35_rule_override`; `v35_hybrid_entry` tồn tại trong engine nhưng không bật trong `v35b` config hiện tại.
- Port thực tế dùng parity-first wrapper [src/components/runners/v35b_runner.py](../../src/components/runners/v35b_runner.py), reuse shared V34-lineage cache helper trong [src/components/runners/v34_runner.py](../../src/components/runners/v34_runner.py).
- Preserve legacy trained-but-dropped exit-model behavior: cache train exit model theo meta/config nhưng runner không truyền `y_pred_exit` vào `backtest_v35b`; regression khóa `model_b_exit == 0`.
- Regression: [tests/regression/test_v35b_parity.py](../../tests/regression/test_v35b_parity.py) pass exact golden 1381 trades.

### 2.5 v37a — `experiments/run_v37a.backtest_v37a`
- Wrapper áp **per-profile dispatch**: với `profile in {bank, defensive, balanced}` → bật `V37A_RELAX_FLAGS`
- Còn lại: gọi backtest_v32
- Mới ở Phase 2: **profile_overrides** trong YAML thay cho hardcoded set

### 2.6 v37a_exit — same as v37a + exit_model train
- Stack identical. exit_model trained nhưng `model_b_exit` count = 0 (bug)

### 2.7 v37d — `experiments/run_v37d.backtest_v37d`
- = backtest_v32, chỉ khác model class (GRU thay LightGBM)
- Stack: V32 stack

### 2.8 v39d — `experiments/run_v39d.backtest_v39d`
- = backtest_v32 + bật V39 flags: `v39a_signal_exit_min_hold=35`, `v39b_hap_min_hold=15`, `v39b_hap_trigger=0.08`, `v39d_rule_exit_symbols={12 sym}`
- Stack mới: `SignalExitMinHold` (hold), `PerSymbolRuleExit` (hold), HapPreempt với min_hold/trigger override

### 2.9 v42_a — `experiments/run_v42.backtest_v42`
- = backtest_v37a (dispatch + V32 stack) + target=early_wave_dual fw=15
- exit_model trained nhưng dropped

### 2.10 v19_3 — `src/strategies/legacy.backtest_v19_3` ✅ verified 2026-04-29
- **Tách riêng engine** — không qua `backtest_unified`. Dùng inline indicators + variable position sizing 0.25–1.0.
- Stack tương đương rút gọn của V22 baseline, KHÔNG có V26+ modifiers.
- Port thực tế dùng dedicated runner [src/components/runners/v19_3_runner.py](../../src/components/runners/v19_3_runner.py) thay vì `SimpleLongBacktester`, vì cần quản lý equity-scaled returns, cooldown, max_profit state và dangling `end` trade.
- Mapping đã verify:
  - `V19EntryCascade`: gộp quick reentry, consolidation breakout, secondary breakout, V-shape bypass, ML signal entry, filters và sizing để giữ parity shared state.
  - `V19SignalHoldGuard`: gộp confirmed signal exit, confirm bars, strong uptrend carry và trend carry override.
  - 11 core strategies ở `strategies/core/`: hard stop, signal hard cap, fast exit loss, ATR stop, peak protect dist/EMA, hybrid exit, adaptive trailing, profit lock, zombie exit, min-hold protection.
- Regression: [tests/regression/test_v19_3_parity.py](../../tests/regression/test_v19_3_parity.py) pass exact golden 1910 trades sau CSV-normalize float formatting.

### 2.11 rule — `compare_rule_vs_model.backtest_rule` ✅ verified 2026-04-29
- Stack độc lập: `RuleSignalEntry` (MACD_hist>0 + Close>MA20 + Close>Open) + `RuleSignalExit` (đối xứng) ở [src/components/fusion/strategies/rule_signal.py](../../src/components/fusion/strategies/rule_signal.py).
- Không qua engine_unified, không có hard_cap/trailing/peak_protect.
- Runner [src/components/runners/rule_runner.py](../../src/components/runners/rule_runner.py) mirror `_run_rule_backtest_fair()` với inline MA20/MACD compute.
- Regression: [tests/regression/test_rule_parity.py](../../tests/regression/test_rule_parity.py) pass exact golden 2585 trades.

---

## 3. Exit-priority ordering (must-preserve)

Engine cũ chạy exit theo `if/elif` chain. Phase 2 fusion stack PHẢI giữ exact ordering này (test bằng golden hash).

```
HARD_STOP
  → MlExitModel (model_b)
    → StallExit (v38b)
      → HeikinAshiBearExit (v38c)
        → RuleCopilotExit (v38d)
          → EarlyLossCut (v28)
            → HapPreempt (v32 + v33 + v39b)
              → AtrAwareHardCap (v30) | TwoStepHardCap (v30) | V22HardCap | SmartHardCapByTrend | FlatHardCapByTrend
                → RegimeAwareHardCap (v30, post-check)
                  → FastExitLoss (always-on, có v22 smart variant)
                    → AtrStopLoss
                      → AdaptivePeakLock (v29) — *parallel branch*
                        → HardCapAfterProfit (v31) — *parallel*
                          → TrendWeakOversoldExit (v32) — *parallel*
                            → DynamicHardCapByDist (v32) — *parallel*
                              → ProfitRatchetExit (v32) — *parallel*
                                → MultiTierTrailingRatchet (v33) — *parallel*
                                  → TrendReversalExit (v33) — *parallel*
                                    → ProfitSafetyNet (v29) — *parallel*
                                      → HardCapAfterPeak (v29) — *parallel*
                                        → ReversalAfterPeak (v29) — *parallel*
                                          → AtrVelocityExit (v29) — *parallel*
                                            → ChandelierTrail (v30) — *parallel*
                                              → PeakProtectByMaxProfit (mod_b variants) — *parallel*
                                                → PeakProtectEma8Streak — *parallel*
                                                  → CyclePeakExit (v28) — *parallel*
                                                    → MaCrossHybridExit — *parallel*
                                                      → AdaptiveTrailing — *parallel*
                                                        → ProfitLock — *parallel*
                                                          → ZombieExit — *parallel*
                                                            → ExtendedHold/PersistenceHold (modifier)
                                                              → MinHoldProtection
                                                                → SignalExitMinHold (v39a)
                                                                  → RuleConfirmExit (v39a)
                                                                    → ShortHoldExitFilter (v31)
                                                                      → WeakOversoldPassthrough (v32 — un-blocks)
                                                                        → RsiOversoldExitBlock (v33)
                                                                          → SignalConfirmExit (v33)
                                                                            → MomentumHoldOverride (v30)
                                                                              → SignalExitDeferFlat (v30)
                                                                                → AdaptiveExitDefer (v31)
                                                                                  → PerSymbolRuleExit (v39d)
                                                                                    → ConfirmedSignalExitScoring (mod_h)
                                                                                      → ConsecutiveExitSignals
                                                                                        → TrendCarrySaved (mod_i + always-on)
                                                                                          → LongHorizonCarry (patch_long_horizon)
```

Ghi chú "parallel": code dùng `if new_position==1 and ...` đứng cạnh nhau, mỗi check có thể flip new_position=0 độc lập. Order code-as-written PHẢI giữ vì counter `counters[]` ghi nhận theo trigger đầu tiên (test sẽ bắt sai nếu hoán đổi).

---

## 4. Hidden flags không trong models.yaml (phải đọc engine.py)

Các flag sau dùng `cfg.get(name, default)` trong engine — chỉ bật khi có entry trong YAML, nhưng nhiều entry **không khai báo** mà dùng default:

```
V29: tất cả v29_*  (15+ flag)
V30: tất cả v30_*  (25+ flag)
V31: v31_pcg_*, v31_ad_*, v31_hap_*, v31_ps_*, v31_shef_*, v31_enriched_log
V32: v32_signal_weak_exit và các param liên quan (v32_swe_*)
V33: v33_trailing_ratchet, v33_trend_rev_exit, v33_recovery_peak_filter, v33_hap_consec_drop, v33_rsi_oversold_block, v33_signal_confirm_exit (+ params)
V38: tất cả v38_*
V39: v39a_*, v39b_*, v39d_*, v39g_*
model_b_min_hold
```

Phase 2 phải **explicit** trong YAML — không có default ngầm. Validation rule: flag chưa khai báo → error.

---

## 5. Chốt 4 layer: tổng kết count

| Layer | # strategies (estimate) |
|-------|------------------------:|
| pre_entry | ~22 (filters) |
| entry | ~15 (decision + sizing) |
| hold | ~20 (defer + cooldown + extend + block-exit) |
| exit_override | ~35 (hard caps + trails + peak protects + force-exits) |
| **Tổng** | **~92 strategy class** |

So với plan ARCHITECTURE.md (`30+ fusion strategies`): thực tế gần 100. Các strategies "always-on" (mod_a-j defaults, ATR sizing, AdaptiveTrailing, ProfitLock, ZombieExit, MinHoldProtection, ConfirmedSignalExitScoring) chiếm ~15-20 strategy "core" — luôn trong stack.

---

## 6. Roadmap port Phase 2.3+ (đề xuất)

Theo độ phức tạp tăng dần — khớp REFACTOR_ROADMAP §2.4:

1. **rule** (Phase 2.3a) ✅ DONE 2026-04-29: standalone, exact golden 2585 trades.
2. **v19_3** (Phase 2.3b) ✅ DONE 2026-04-29: minimal stack, exact golden 1910 trades.
3. **v22** (Phase 2.3c) ✅ DONE 2026-04-29: full V22 parity runner, exact golden 1784 trades.
4. **v34 = v32 stack** (Phase 2.4a) ✅ DONE 2026-04-29: parity-first wrapper, exact golden 1323 trades.
5. **v35b** (Phase 2.4b) ✅ DONE 2026-04-29: V35 entry-layer flags, exact golden 1381 trades.
6. **v32** standalone (Phase 2.4c) ✅ DONE 2026-04-29: parity-first wrapper, exact golden 1347 trades.
7. **v37a** (Phase 2.4d, 1 ngày): + per-profile dispatch.
8. **v39d** (Phase 2.4e, 1 ngày): + V39 hold modifiers + per-symbol rule exit.
9. **v37a_exit / v42_a** (Phase 2.4f, 0.5 ngày): dual target + preserve trained-but-dropped exit-model behavior for parity.
10. **v37d** (Phase 2.4g, 0.5 ngày): chỉ swap entry model (đã có ở Phase 1.4).

> **Risk**: per-profile dispatch v37a và per-symbol set v39d là hai điểm dễ lệch parity nhất. `MlExitModel` fix không làm trong Phase 2.4; chỉ preserve golden behavior trước.

---

## 7. Open issues cho Phase 2.2 (base interface)

1. **Layer `hold` semantics** ✅ resolved 2026-04-29: giữ 4 layer; thêm `FusionActionType="keep_position"`. Hold layer trả `keep_position` để block toàn bộ `exit_override` ở bar đó (test `test_keep_position_blocks_exit_override`).
2. **State carry across bars** ✅ resolved 2026-04-29: `Position.strategy_state: dict[str, Any]` — strategy mutate trực tiếp qua `ctx.position.strategy_state[key]` (test `test_position_state_carry_via_strategy_state`). Convention key: namespace prefix `"<strategy_name>:<field>"` để tránh collision (sẽ enforce ở Phase 2.3+).
3. **Per-symbol set** (v39d, RULE_PRIORITY_SYMBOLS, SCORE5_RISKY_SYMBOLS): defer Phase 2.4f (port v39d). Convention dự kiến: YAML truyền `list[str]` → orchestrator bind vào `ctx.config[strategy_name]["symbols"]` dưới dạng `frozenset` để lookup `O(1)`.
4. **Always-on vs flag-on** ✅ resolved 2026-04-29: registry hỗ trợ flag `always_on=True`. Orchestrator (Phase 3) sẽ auto-prepend `list_always_on()` vào stack trước khi load YAML overrides.
5. **Counter semantics** ✅ resolved 2026-04-29: strategy đặt `metadata={"counter": "<key>"}` trong `FusionResult` → `FusionStack` gom vào `StackOutcome.counters: Counter`. Naming convention: `n_<strategy_snake_case>` để khớp `counters[]` engine cũ (sẽ verify ở Phase 2.3 v22 port).

### Quyết định bổ sung (Phase 2.2)

- **Lifecycle ordering**: `pre_entry → entry → hold → exit_override`. Pre_entry/entry chỉ chạy khi `position is None`; hold/exit_override chỉ khi đang giữ vị thế. Pre_entry còn skip nếu `entry_signal == 0` (engine cũ không evaluate filter khi không có signal).
- **Sizing**: strategy `entry` truyền `metadata["size"]: float` để override size mặc định 1.0. Mặc định nếu không khai báo.
- **`enter` ngắn gọn**: lần đầu `action="enter"` ở entry layer thắng — không tích lũy candidate. Khớp pattern engine cũ `if rule_signal: return entry`.

---

## Phụ lục — toàn bộ 80 flag trong models.yaml

```
mod_a, mod_b, mod_c, mod_d, mod_e, mod_f, mod_g, mod_h, mod_i, mod_j,
fast_exit_strong, fast_exit_moderate, fast_exit_weak, fast_exit_hb_buffer,
peak_protect_strong_threshold, peak_protect_normal_threshold,
hard_cap_weak, hard_cap_moderate_mult, hard_cap_strong_mult,
hard_cap_strong_floor, hard_cap_moderate_floor,
time_decay_bars, time_decay_mult,
v22_mode, v22_adaptive_hard_cap, v22_hard_cap_floor, v22_hard_cap_floor_hb,
v22_hard_cap_mult_std, v22_hard_cap_mult_hb,
v22_fast_exit_threshold_std, v22_fast_exit_threshold_hb,
v22_fast_exit_skip_strong, v22_fast_exit_vol_confirm,
patch_smart_hardcap, patch_pp_restore, patch_long_horizon, patch_symbol_tuning,
patch_rule_ensemble, patch_noise_filter, patch_adaptive_hardcap, patch_pp_2of3,
v26_wider_hardcap, v26_relaxed_entry, v26_skip_choppy, v26_extended_hold,
v26_strong_rule_ensemble, v26_min_position, v26_score5_penalty, v26_hardcap_confirm_strong,
v27_selective_choppy, v27_hardcap_two_step, v27_rule_priority,
v27_dynamic_score5_penalty, v27_trend_persistence_hold,
v28_early_wave_filter, v28_crash_guard, v28_wave_acceleration_entry,
v28_early_loss_cut, v28_cycle_peak_exit, v28_early_loss_cut_threshold, v28_early_loss_cut_days,
v32_hap_preempt, v32_hap_pre_trigger, v32_hap_pre_floor,
v32_weak_oversold_exit, v32_woe_dist_thresh, v32_woe_min_profit, v32_woe_hold_min,
v32_dynamic_hc_dist, v32_dhc_dist_thresh, v32_dhc_tight_cap,
v32_profit_ratchet, v32_pr_trigger, v32_pr_keep,
v35_relax_cooldown, v35_cooldown_after_big_loss, v35_cooldown_after_loss,
v35_skip_price_proximity, v35_single_bar_signal,
v35_rule_override, v35_rule_override_min_score, v35_hybrid_entry, v35_hybrid_size
```
