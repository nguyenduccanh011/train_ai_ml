# bot_shallow_* — quality-gated shallow fill: implementation notes (2026-07-09)

## Knob (engine.py only, gated default-off)
- `EngineConfig.bot_shallow_k: float = 0.0` (0 = off), `bot_shallow_floor: float = 0.4` —
  added next to `bot_deepen_cap` (engine.py ~:915-925).
- Depth application (engine.py ~:1670-1677, right AFTER the bot_deepen block):
  `_depth *= max(bot_shallow_floor, 1.0 - bot_shallow_k * max(z6[i], 0.0))`.
  Ordered after bot_deepen: if both are configured the deepened depth is what gets shrunk
  (sweeps run deepen off).
- `use_bot` gate (engine.py ~:2475) extended with `or cfg.bot_shallow_k > 0` so score6 is
  mapped into `_run_symbol` when only the shallow knob is on.

## Deviation from the literal spec (raw score6 -> causal z), and why
The task said "reuse the SAME z variable bot_deepen uses". bot_deepen in fact uses the RAW
score6 (`bot[i]`) — there was never a z in the engine. That convention was built for the
prob-like zigzag head (raw mean +0.074). The wave-start head
(`bottom_structure_entry_regression` h8 penalty1.5) predicts a penalized return with raw
mean -0.252 / std 0.083 (measured, probes P1/P2 logs): `max(raw, 0)` would be 0 on ~99.9%
of bars — the knob would be a structural no-op and the whole sweep meaningless.
WAVESTART_DESIGN.md specifies `z6` explicitly (`_depth *= clip(1 - k*max(z6,0), floor, 1)`,
"dùng chung z của score6"). So `_run_symbol` precomputes `bot_z` = causal per-symbol
252/60 rolling z of score6 (engine.py ~:1107-1119, exact same pattern/window as `xscore_z`
:1084-1089 and the recombine/union z, so z 0.9 means the same thing as the ensemble5
`z_threshold: 0.9`). Computed only when `bot_shallow_k > 0`; bot_deepen untouched (raw).

## Parity guard (template 2646, seed 42) — PASS
- BEFORE edit: composite=729.6 pnl=127.25301810134556 pf=5.963310002880663
  mdd=0.17454927579356125 trades=1384 (matches recorded baseline).
- AFTER field+branch edit: identical. AFTER z-precompute edit: identical.
- trades CSV and signals CSV byte-identical across before/after/after2
  (`parity/{before,after,after2}/`).
- `pytest stock_ml/tests/regression/test_champions.py -x`: 1 passed, 12 skipped
  (skips = missing local data for those champions, pre-existing).

## Sweep
- Script: `bs_sweep.py` (pattern = probes_bchannel.py driver/worker; clones 2646 by name,
  `eng.update` with ONLY entry_ensemble5 + bot_shallow_*; seed 42; 10-min kill guard;
  per-run CSV export under `runs/<name>/` for trade diffing).
- Trade diff: `bs_diff.py <name>` vs `parity/before/trades_*.csv`
  (new/lost/repriced cohorts, per-year, exit reasons).
- Results (seed 42, baseline 729.6 / 127.25 / 5.963 / 0.17455 / 1384):

  | run | comp | D | pnl | pf | mdd | trades |
  |---|---|---|---|---|---|---|
  | bs_so_k05_f44 | 688.0 | -41.6 | 124.81 | 5.044 | 0.21747 | 1415 |
  | bs_so_k10_f44 | 683.4 | -46.2 | 125.13 | 4.861 | 0.22510 | 1431 |
  | bs_so_k10_f55 | 693.2 | -36.4 | 125.70 | 5.027 | 0.21406 | 1421 |
  | bs_un_k05_f44 | 684.7 | -44.9 | 124.61 | 4.990 | 0.22205 | 1424 |
  | bs_un_k10_f44 | 679.7 | -49.9 | 124.83 | 4.812 | 0.22971 | 1440 |
  | bs_un_k10_f55 | 689.5 | -40.1 | 125.40 | 4.973 | 0.21829 | 1428 |
  | bs_so_k05_f66 (follow-up) | 702.2 | -27.4 | 125.68 | 5.282 | 0.20184 | 1404 |

  Fully MONOTONE: every dose hurts; weaker dose = less damage; union < score-only at every
  (k,floor). Extrapolates to 0 only at knob-off -> second follow-up skipped (no positive
  configuration exists for this knob).

## Cohort diff (bs_diff.py, best variant bs_so_k05_f66 vs champion)
- NOT additive fills but occupancy RESHUFFLE: 447 new vs 427 lost entries (common only
  957/1384). Shallower limits fill EARLIER -> the single lot is occupied sooner -> the
  whole downstream trade sequence shifts (the F1 displacement lesson, now via timing).
- New cohort: WR 0.532, mean +0.0801, sum +35.8. Lost cohort: WR 0.541, mean +0.0780,
  sum +33.3 -> near like-for-like swap, slightly WORSE quality in.
- 468 common trades repriced +0.82% higher entry -> pnl delta -4.09 (the direct cost of
  shallowing: same trade, worse price). Net pnl -1.57; composite -27.4 mostly via mdd
  0.175 -> 0.202 (reshuffled sequence clusters drawdown).
- Blindspot years get NOTHING: new-fill mean pnl 2022 +0.040, 2024 +0.004, 2026 -0.005;
  the positive new fills land in 2020/21/25 which were already strong. Strong-dose variant
  (bs_so_k10_f44) same story, worse (lost WR 0.569 > new 0.524).

## Verdict
Quality-gated shallow fill does NOT convert wave-start signals into net-positive trades.
The z-gate engages on any z>0 (~half of buys mildly shallowed), the repriced common trades
pay for it, the displaced cohort is as good as the new one, and the blindspot years see
zero edge. Consistent with the P-series: the champion is occupancy-bound; changing fill
economics only reshuffles which signals hold the lot. No multi-seed warranted. The knob
stays in the engine gated default-off (parity proven); next viable wave-start route per
W3 is the confirmation-breakout entry (buy thrust-high break: capture 89%, +10.6%/21bar),
which ADDS a mechanism instead of re-pricing the existing one.
