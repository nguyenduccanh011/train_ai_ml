# Phase Guard Experiment Notes

## Baseline

Best current baseline:

```text
feature_set: leading_v2
entry_model: random_forest
exit_model: lightgbm
strategy: v22
target: early_wave
window: 2020-2025
score: 742.7
total_pnl: 10322.88
win_rate: 58.05
mdd_per_symbol: 16.50
trades: 1504
```

## Negative experiments

### 1. Full phase-aware features

Tested feature block:

```text
days_since_ma20_cross_up
return_since_ma20_cross_up
consecutive_up_closes
days_since_vol_expansion
overextension_score
```

Result on `early_wave`:

```text
leading_v5 + random_forest + lightgbm_exit
score: 724.0
total_pnl: 8123.97
win_rate: 54.53
mdd_per_symbol: 19.20
trades: 1335
```

Outcome: worse than baseline.

### 2. Lightweight phase features

Tested feature block:

```text
consecutive_up_closes
days_since_vol_expansion
```

Result on `early_wave`:

```text
leading_v2_phase_light + random_forest + lightgbm_exit
score: 724.0
total_pnl: 8123.97
win_rate: 54.53
mdd_per_symbol: 19.20
trades: 1335
```

Outcome: also worse than baseline, identical to full phase-aware result.

## Interpretation

`early_wave` labels pre-breakout accumulation setups. Features that describe trend maturity or recent continuation can act as anti-signals for this target and cause the entry model to filter out profitable early-wave setups.

Do not add phase/overextension features directly into the `early_wave` entry model unless the target definition is also changed.

## Next experiment direction

Keep the baseline entry model unchanged:

```text
leading_v2 + random_forest + lightgbm_exit + early_wave
```

Then test phase logic outside the model as a strategy guard:

1. Position-size guard only:
   - reduce size when entry is late or overextended.
   - do not reject the trade.

2. Entry reject guard only for extreme cases:
   - reject only if price is far above MA20 and the signal is not a fresh breakout.

3. Exit guard:
   - if price was overextended and then loses MA20 on a large red candle, exit earlier.

Recommended first test:

```text
v22 baseline + position-size guard
```

Rationale: safer than rejecting entries, because previous feature tests suggest many filtered trades were actually good early-wave winners.

## Acceptance target

Promote only if the new guard beats baseline on the same fair setup:

```text
score > 742.7
total_pnl close to or above 10322.88
mdd_per_symbol <= 16.50
win_rate >= 58.05
```
