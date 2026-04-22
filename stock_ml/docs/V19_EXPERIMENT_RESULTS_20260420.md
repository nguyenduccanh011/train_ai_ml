# V19 Experiment Results (2026-04-20)

## Configuration
- Baseline source: `run_v18_compare.py`
- New experiment file: `run_v19_compare.py`
- Priorities implemented:
  1. Entry alpha and position-size policy separated.
  2. Signal exit quality scoring.
  3. Symbol-regime adapter (bank/high-beta/momentum/defensive + choppy detection).

## Aggregate Backtest (14 symbols, 2020-2025 walk-forward)
- V18: `+1831.5%` | `422` trades | PF `2.74` | MaxLoss `-21.7%`
- V19: `+1864.2%` | `427` trades | PF `2.64` | MaxLoss `-26.2%`
- Rule: `+1982.6%` | `585` trades | PF `2.21` | MaxLoss `-27.2%`

Delta:
- `V19 - V18 = +32.8%`
- `V19 - Rule = -118.4%`

## Key Symbol Deltas (V19 - V18)
- Positive: `BID +32.2%`, `MBB +32.0%`, `VIC +20.9%`, `TCB +12.4%`, `AAV +7.5%`, `DGC +6.3%`.
- Negative: `HPG -26.4%`, `VND -23.5%`, `SSI -7.7%`, `VNM -6.4%`, `FPT -6.3%`.

## TCB Focus
- V18: `+23.1%`, PF `1.26`, signal PnL `-39.5%`
- V19: `+35.5%`, PF `1.45`, signal PnL `-26.1%`
- Rule: `+37.6%`, PF `1.29`

=> V19 closed most of the TCB gap to Rule (`-2.0%` remaining).

## Output files
- `results/v19_compare_20260420_full.txt`
- `run_v19_compare.py`
