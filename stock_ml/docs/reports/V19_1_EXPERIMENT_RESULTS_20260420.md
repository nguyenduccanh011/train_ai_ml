# V19.1 Experiment Results (2026-04-20)

## Scope
- Base: `run_v19_compare.py`
- New: `run_v19_1_compare.py`
- Objective: continue v19 with risk tuning for HPG/VND using:
  - lower `size_mult` in momentum/choppy/high-ATR states
  - stricter `exit_score_threshold` in volatile states
  - targeted symbol guard for `HPG`, `VND`

## Aggregate (14 symbols, walk-forward 2020-2025)
- V18: `+1831.5%` | PF `2.74` | MaxLoss `-21.7%` | 422 trades
- V19: `+1864.2%` | PF `2.64` | MaxLoss `-26.2%` | 427 trades
- V19.1: `+1866.8%` | PF `2.68` | MaxLoss `-28.6%` | 419 trades
- Rule: `+1982.6%` | PF `2.21` | MaxLoss `-27.2%` | 585 trades

Delta:
- `V19.1 - V19 = +2.6%`
- `V19.1 - Rule = -115.8%`

## Target symbols check (trade-level)
### HPG
- V19: total `+157.4%`, PF `3.39`, max_loss `-12.3%`
- V19.1: total `+155.9%`, PF `3.65`, max_loss `-9.7%`
- Result: drawdown improved, return near flat.

### VND
- V19: total `+253.6%`, PF `2.52`, max_loss `-26.2%`
- V19.1: total `+253.6%`, PF `2.72`, max_loss `-28.6%`
- Result: PF improved but max loss worsened; risk objective not met for VND.

## Notable side effects
- Strong upside: `AAV` improved heavily (`+260.2% -> +323.1%`, PF `3.26 -> 4.63`).
- Tradeoff: `AAS`, `SSI`, `VIC` softened vs V19.

## Files
- `run_v19_1_compare.py`
- `results/v19_1_compare_20260420_full.txt`
