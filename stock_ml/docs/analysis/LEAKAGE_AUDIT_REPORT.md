# Leakage Audit Report - Stock ML Pipeline

**Date**: 2026-05-18  
**Auditor**: Claude Opus 4.7  
**Scope**: Full audit of top 1 leaderboard model and feature engineering pipeline

---

## Executive Summary

**CRITICAL LEAKAGE FOUND** in `_market_structure()` pivot detection affecting top 1 model.

- **Model**: `v22_exit_ablation_round25` (config_hash: `5be8a6ecd7fb`)
- **Leaderboard metrics**: WR 78.2%, PF 15.16, Total PnL 15,977, Composite 662.1
- **Feature set**: `leading_v2` (includes `_market_structure`)
- **Impact**: Metrics are **inflated** due to forward-looking pivot features
- **Status**: **FIX APPLIED** to `src/features/engine.py` and `src/components/features/blocks/market_structure.py`

---

## Findings

### ✅ No Leakage (Verified)

| Component | Status | Notes |
|-----------|--------|-------|
| Target shift logic | ✅ OK | `target[i] = target_raw[i+1]` correctly predicts future |
| Backtest execution | ✅ OK | `pred = y_pred[i-1]` at bar `i`, no look-ahead |
| `_leading_signals` | ✅ OK | All `rolling()` backward-looking |
| `_exhaustion_signals` | ✅ OK | Uses `[i-N:i]` slices, no future data |
| `_volatility_regime` | ✅ OK | `rolling()` and `rank(pct=True)` backward |
| `_multi_timeframe` | ✅ OK | Weekly aggregations use past data only |
| `_accumulation_features` | ✅ OK | All features backward-looking |
| `_heikin_ashi_features` | ✅ OK | HA computed from current/past OHLC |
| `_relative_strength` | ✅ OK | Cross-sectional at same timestamp, no leak |
| `_liquidity_features` | ✅ OK | All `rolling()` backward |

### 🚨 CRITICAL: Pivot Point Leakage

**Location**: `src/features/engine.py:442-527` (original), `src/components/features/blocks/market_structure.py:20-95`

**Problem**:
```python
# ORIGINAL (LEAKY) CODE:
for i in range(order, n - order):
    if all(h[i] >= h[i - j] for j in range(1, order + 1)) and all(
        h[i] >= h[i + j] for j in range(1, min(order + 1, n - i))  # ← FORWARD!
    ):
        ph[i] = 1.0
```

**Issue**: `pivot_high[i] = 1` requires `high[i] >= high[i+1], high[i+2], ..., high[i+order]`

At bar `i`, features include `pivot_high[i]` which depends on **future** prices `high[i+1..i+order]`.

**Affected features**:
- `pivot_high_3`, `pivot_high_5`, `pivot_high_7`
- `pivot_low_3`, `pivot_low_5`, `pivot_low_7`
- `dist_to_last_swing_high`, `dist_to_last_swing_low` (derived from pivots)
- `bos_up`, `bos_down`, `choch` (break of structure, derived from pivots)

**Impact on top 1 model**:
- Model learned: "when `pivot_high[i]=1`, entry is good"
- But `pivot_high[i]=1` is only known **after** seeing `order` future bars
- Live trading cannot use this signal at bar `i`
- **Metrics are inflated** — actual live performance will be significantly lower

---

## Fix Applied

**New logic**: Pivot at bar `i-order` is confirmed and set in `pivot_high[i]` (after seeing `order` bars ahead).

```python
# FIXED CODE:
for i in range(2 * order, n):
    pivot_idx = i - order
    if all(h[pivot_idx] >= h[pivot_idx - j] for j in range(1, order + 1)) and all(
        h[pivot_idx] >= h[pivot_idx + j] for j in range(1, order + 1)
    ):
        ph[i] = 1.0  # Confirmed at bar i, refers to bar i-order
```

**Verification**:
- Pivots only appear starting from bar `2*order` (e.g., bar 6 for order=3)
- Features at bar `i` use `pivot_high[i]` which refers to bar `i-order` (already in the past)
- No forward-looking data

**Files modified**:
1. `src/features/engine.py` lines 442-527
2. `src/components/features/blocks/market_structure.py` lines 20-95

---

## Recommendations

### Immediate Actions

1. **Clear feature cache** (optional — cache key auto-updates due to code hash)
   ```bash
   rm -rf results/cache/features/
   ```

2. **Retrain top 1 model** with fixed features:
   ```bash
   cd /path/to/train_ai_ml
   python -m stock_ml run-matrix config/experiments/matrix/v22_exit_ablation_round25_leakage_fix_test.yaml
   ```

3. **Compare metrics** before/after fix:
   - Expected: WR drops from 78.2% to ~65-70%
   - Expected: PF drops from 15.16 to ~5-8
   - Expected: Total PnL drops proportionally

### Long-term Actions

1. **Add leakage tests** to CI/CD:
   - Check all features use only `rolling()`, `shift(positive)`, or `[i-N:i]` slices
   - Flag any `[i:i+N]`, `shift(-N)`, or forward indexing

2. **Audit other models** using `leading_v2`, `leading_v3`, `leading_v4`:
   - All use `_market_structure()` → all affected by pivot leakage
   - Retrain after fix

3. **Review leaderboard**:
   - Mark pre-fix models as "leaky"
   - Re-rank after retraining with fix

---

## Technical Details

### Pivot Detection Logic

**Original (leaky)**:
- At bar `i`: check if `high[i]` is local maximum in window `[i-order, i+order]`
- Uses future data `high[i+1..i+order]` to label bar `i`

**Fixed (no leak)**:
- At bar `i`: check if `high[i-order]` was local maximum in window `[i-2*order, i]`
- Only uses past data `high[i-order-order..i]` to label bar `i`
- Pivot confirmation delayed by `order` bars (acceptable tradeoff)

### Test Results

Synthetic data test (100 bars):
```
pivot_high_3: first pivot at bar 12 (refers to bar 9), requires bar >= 6 ✓
pivot_high_5: first pivot at bar 14 (refers to bar 9), requires bar >= 10 ✓
pivot_high_7: first pivot at bar 16 (refers to bar 9), requires bar >= 14 ✓

Pivots in early bars (< 2*order): 0 for all orders ✓
```

---

## Appendix: Audit Methodology

1. **Target generation**: Verified `shift(-1)` convention aligns with backtest `pred[i-1]`
2. **Backtest execution**: Verified entry at bar `i` uses prediction from bar `i-1`
3. **Feature computation**: Line-by-line review of all feature blocks
4. **Forward-looking patterns**: Searched for `shift(-N)`, `[i+N]`, `future`, `lookahead`
5. **Pivot detection**: Detailed analysis with synthetic data examples

---

## Conclusion

The pivot leakage in `_market_structure()` is **severe** and affects all models using `leading_v2+` feature sets. The fix has been applied and verified. Retraining is required to obtain realistic performance metrics.

**Next steps**: Retrain top 1 model and compare metrics to quantify leakage impact.
