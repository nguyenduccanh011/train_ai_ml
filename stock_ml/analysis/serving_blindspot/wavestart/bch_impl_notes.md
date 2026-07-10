# entry_bchannel_* — confirmation-breakout B-entry: implementation notes (2026-07-09)

## Knob (engine.py only, gated default-off)
- `EngineConfig.entry_bchannel_z: float | None = None` (master switch),
  `entry_bchannel_break_lookback: int = 4`, `entry_bchannel_below_ma: int | None = 20`,
  `entry_bchannel_stop_lookback: int | None = None` — added after `bot_shallow_floor`
  (engine.py ~:927-946) with the W3-A4 anatomy numbers in the comment (capture 89%,
  +10.6%/21bar, false-fill 41% vs 86% for raw -2% limits).
- z6 = the SAME causal per-symbol 252/60 rolling z of score6 built for bot_shallow
  (engine.py ~:1130-1143); its gate is extended with `or cfg.entry_bchannel_z is not None`.
  `use_bot` (~:2554-2555) extended the same way so score6 is mapped into `_run_symbol`.
- Below-MA context SMA precomputed once per symbol (`bch_ma`, ~:1160-1166).
- Trigger block (~:1680-1708): in the NOT-in-position branch, checked BEFORE the core
  `if sig > 0:` branch — **design choice: the B-entry preempts a same-bar core signal**
  (the core path is a patient limit that would fill later & lower; B is the at-market
  confirmation buy — if both fire on the same bar the confirmation wins the slot).
  Conditions: z6[i] >= entry_bchannel_z AND close[i] > max(high[i-lb..i-1]) AND
  close[i] < SMA(below_ma) (None disables) AND reentry_cooldown_bars respected AND i<n-1.
  Fill mirrors resume_reentry: entry_idx=i+1, fill_buy(closes[i+1]).
  Market weak/chop gates are NOT applied to B (mirror of resume_reentry).
- B-only structural stop (~:1900-1909, right after the core structural_stop): when
  `entry_bchannel_stop_lookback` set and the position is a B-position, exit when
  low[i] <= min(low[signal_i-N+1..signal_i]) (no buffer), next-bar close fill,
  reason `bch_stop`. All other exits inherit the champion machinery unchanged.

## B-trade tagging: side log, NOT a Trade field
The Trade dataclass/trades_to_dataframe schema is hash-frozen by the regression goldens
(trades CSV byte-parity + tests/regression checksums) — adding an `entry_channel` field
would add a CSV column and break byte-identity even knob-off. So B-entries are recorded to
a module-level side log `engine.BCH_ENTRY_LOG` (symbol, signal_date, entry_date), cleared
at the top of run_backtest ONLY when the B-channel is armed (knob-off never touches module
state). run_backtest is called once per experiment (experiment.py:3265, in-process), so the
sweep worker dumps the log to `runs/<name>/b_entries.csv` after run_template_experiment
returns; diagnostics join it to the trades CSV on (symbol, entry_date).

## Loop-topology caveat (noted, accepted)
The core pullback path resolves each signal in an inner forward scan and jumps `i` to the
fill bar — bars inside a fill-scan window are not re-visited by the outer loop, so a
B-trigger occurring strictly inside another signal's successful fill-scan is not seen.
This is the same visibility the resume_reentry block has; acceptable for a first sweep.

## Parity guard (template 2646, seed 42) — PASS
- BEFORE edit (parity/bch_before): composite=729.6 pnl=127.25301810134556
  pf=5.963310002880663 mdd=0.17454927579356125 trades=1384 (== recorded baseline).
- AFTER edit knob-off (parity/bch_after): identical scalars; ALL export CSVs byte-identical
  (trades/signals/daily/symbol/yearly via cmp).
- `pytest stock_ml/tests/regression/test_champions.py -x`: 1 passed, 12 skipped
  (skips = missing local data for those champions, pre-existing).
- Synthetic smoke: B-entry fires on a thrust bar below SMA20, close_next fill, side log
  populated, `bch_stop` exits on a crash through the pre-trigger low; knob-off run leaves
  the log untouched.

## Strict-audit collision (found on the first sweep attempt)
`check_entry_integrity` (src/backtest/integrity.py:36) FAILs any trade whose
entry_signal_date is not a buy-signal bar — B-entries violate this BY DESIGN (the whole
point is entering where the core channel emits nothing). `strict_audit` is hardcoded True
for template runs (experiment.py:359). Fix kept OUT of repo source: bch_sweep.py's worker
monkeypatches integrity.check_entry_integrity to exempt exactly the engine-logged B-entries
(key symbol+signal_date from BCH_ENTRY_LOG) and audit the rest as usual. fill_offset still
covers B-trades (close_next). A production integration would need integrity.py to learn
the B-channel — a known integration cost of this mechanism.

## Sweep (seed 42, clone 2646 + ENS5 score-only head; baseline 729.6/127.25/5.963/0.17455/1384)
Runs: bch_sweep.py (templates 2722-2726 + follow-ups 2727/2728 by clone name); diagnostics
bch_diff.py (B-cohort via b_entries.csv join, displacement, idle-slot). Runtime ~30-40s/run.

| run | comp | D | pnl | pf | mdd | trades | B-fires |
|---|---|---|---|---|---|---|---|
| bch_z09        | 682.8 | -46.8 | 122.92 | 4.663 | 0.22347 | 1650 | 371 |
| bch_z12        | 687.3 | -42.3 | 123.18 | 4.819 | 0.21652 | 1591 | 286 |
| bch_z09_stop10 | 686.3 | -43.3 | 123.17 | 4.697 | 0.22098 | 1651 | 371 |
| bch_z12_stop10 | 690.6 | -39.0 | 123.40 | 4.851 | 0.21391 | 1592 | 286 |
| bch_z09_lb8    | 721.1 |  -8.5 | 126.33 | 5.669 | 0.18284 | 1426 |  73 |
| bch_z12_lb8 (f1: stack the improving confirmation axis with the z gate) | 724.2 | -5.4 | 126.79 | 5.745 | 0.18152 | 1418 | 58 |
| bch_z15_lb8 (f2: extreme-selectivity point of the same axis) | 726.2 | -3.4 | 126.87 | 5.803 | 0.17838 | 1411 | 46 |

f3 skipped: dose-response is strictly monotone in B-count across BOTH axes (z and
break_lookback) and extrapolates to 0 only at knob-off — same shape as the bs_ sweep; no
positive configuration exists.

## Cohort diagnostics (bch_diff.py vs parity/bch_before champion export)
- The mechanism IS structurally additive as designed: 100% of B-entries (z*_lb8; 99.2% at
  z09) fall OUTSIDE every champion holding interval, ~66% have no champion entry on the
  symbol within +/-10 bdays. The idle-slot premise verified — and it still loses.
- B-cohort: WR 0.43-0.48, median pnl ~-0.3%, median hold 3d, exits ~100% 'signal' — the
  champion's sell head fires immediately on below-MA20 names, so the exit machinery
  structurally refuses to ride a wave start. The positive sum is a 2020-bull artifact:
  ex-2020 the B-cohort is ~0 or negative in EVERY config (z09_lb8 -0.51, z12_lb8 -0.17,
  z15_lb8 -0.04). Blindspot years get nothing again (2022 -0.11..+0.09, 2024 +0.05, 2026
  +0.20..-0.14).
- Displacement net negative in EVERY config (B + knock-on-new - lost): z09 -3.23, z12
  -2.97, z09_lb8 -0.92, z12_lb8 -0.46, z15_lb8 -0.38. The ~34% of B-fires that land near
  a core entry occupy the slot days earlier and displace premium trades (lost cohort WR
  0.60-0.70, mean +0.10-0.12, med hold 24-26d) with WR-0.45/3-day scratches.
- Composite damage is mdd/pf-led (0.175 -> 0.178-0.223), the same drawdown-clustering
  signature as bs_. B-only structural stop is a non-factor: fires 18/371 (5%) — signal-exit
  kills the trade first.

## Verdict — KILL (criterion met in ALL configs)
Pre-registered kill-criterion: B-cohort net-negative OR displacement cost > B-gains in all
configs. Both hold: ex-2020 the B-cohort is net-negative everywhere, and displacement net
is negative everywhere (-0.38 .. -3.23). The idle-slot hypothesis was TRUE (100% outside
champion holds) and still insufficient — the failure moved from entry occupancy (bs_) to
the EXIT side: the champion sell head instantly closes below-MA20 positions, so wave-start
trades can never ride to the +10.6%/21bar the anatomy promised, while their slot-tax on
core trades is immediate. The wave-start line is CLOSED at the current engine architecture
(single-slot, champion exit stack). No multi-seed warranted. Remaining route (previously
deferred): multi-position architecture + a wave-start-aware exit head. Knobs stay in the
engine gated default-off (parity proven byte-identical).
