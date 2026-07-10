# Strategy Analysis Toolkit

How to X-ray a trained strategy — diagnose what its entry/exit heads predict, where the blind spots
are, which features help vs hurt, and how every trade was decided. Tools live in
[`results/_swing_capture/`](../results/_swing_capture/) (index: that dir's `README.md`).

Built 2026-06-18 for the champion (template 1930). They work on **any** template that has persisted
fold artifacts under `results/tmpl_<id>_*/folds/`.

---

## 1. The reproduction model (read once)

Every tool rebuilds the strategy the same way (all in `_harness.py`):

```
template config (DB)                     # _cfg(tmpl) -> ExperimentConfig
  -> per-fold raw head-score parquets    # results/tmpl_<id>_*/folds/*.parquet  (one dir per seed)
  -> recombine_signals(s0, cfg)          # -> `signal` stream + per-head scores (score, score2..5, exit_score)
  -> run_backtest(sig, oh, engcfg)       # -> trades (entry/exit dates, prices, pnl_pct, exit_reason,
                                         #    entry_signal_date = the bar whose limit filled)
```

Pooling over the fold dirs (= seeds) gives seed-robust numbers; most tools print `/fold` (÷ number of
fold-sets) so magnitudes read as a single backtest. Features (`feature_xray`) are deterministic from
OHLCV, so it builds them **once**, no fold loop.

Key data objects:
- **`sig`** — per (symbol, date): `signal` (>0 = the model wants to buy here, before the engine's
  pullback/market gates), plus raw head scores.
- **`trades`** — one row per round-trip. `entry_signal_date` is the engine's last signal bar (i*) whose
  4.5% limit filled; `entry_date` is the fill bar; `exit_reason` is exact.

---

## 2. Quick start

```bash
# from repo root. default template = 1930.
python results/_swing_capture/model_xray.py            # full report — start here
python results/_swing_capture/feature_xray.py          # per-feature diagnostics (retrains; slower)
python results/_swing_capture/trade_reasons.py         # per-trade entry/exit reason + by-year
python results/_swing_capture/missed_runup.py 3 0.05   # missed run-up (gap bars, runup thr)
python results/_swing_capture/rootcause_buffer.py 0.05 # why the pullback crutch (runup thr)

python results/_swing_capture/fragility.py             # edge concentration + seed dispersion
python results/_swing_capture/regime_breadth.py        # pnl/win/risk conditioned on market regime
python results/_swing_capture/gate_drain.py            # per-gate signal-drop counterfactual

python results/_swing_capture/model_xray.py 1844       # any other template id
```

---

## 3. The tools

### `model_xray.py` — consolidated report (run first)
Single OHLCV pass; sections:
- **A. Prediction quality / masking** — per head: `IC_fwdpeak` (score vs forward 20-bar peak) vs
  `IC_realized` (score vs realized pnl). `gap = fwdpeak − realized` is how much the **exit hides** the
  head's signal. Negative `IC_realized` = the head is anti-predictive on what you actually capture.
- **B. PnL / risk** — pnl percentiles, MFE/MAE, capture (realized/MFE), loss tail.
- **C. Entry timing** — position-in-wave (0 bottom → 1 peak), % late (>0.7).
- **D. Exit timing** — per exit_reason: n, pnl, win%, **capture**, **giveback** (MFE − realized),
  **exit-lag** (bars sold after the in-hold peak). This is where the profit attribution lives.
- **E. Coverage** — % of >10% up-waves missed while flat (caveat: single-lot per symbol overstates).
- **F. Regime** — pnl/win/MAE by entry-vol bucket.
- **X1** intra-hold swings, **X2** post-exit continuation (swing left behind), **X3** per-leg IC,
  **X4** exit position-in-wave, **X5** sideways/idle bars (capital efficiency).

### `feature_xray.py` — per-feature diagnostics
Regenerates the exact feature matrix via `build_feature_frame` (no leak), then per feature:
- `rawIC / zIC / rankIC` vs the head's target — **raw≫z** ⇒ per-symbol z-norm is killing it (use raw /
  cross-sectional); **rank≫raw** ⇒ needs a non-linear transform.
- `topIC / botIC` — IC vs forward return measured **separately in top vs bottom zones**. Opposite signs
  = the feature can disambiguate a top from a dip; same sign = blind (the core wall).
- `gain%` (LightGBM split-gain) + `permIC` (IC drop when the feature is shuffled on a holdout). **permIC
  ≤ 0 while gain is spent ⇒ HARMFUL** (the model overfits noise).
- tags: `HARMFUL`, `NOISY(z-kills|nonlinear)`, `REDUNDANT` (|corr|>0.92). Retrains deterministically,
  so it is reproducible but slower (~1–2 min/head).

### `trade_reasons.py` — per-trade reasons
- **Exit reason** = exact engine tag. **Entry reason** = reconstructed: which head's causal z (252/60)
  cleared its threshold at `entry_signal_date` (heads co-fire → both a non-exclusive participation table
  and a single "primary" label). Plus by-year and the entry×exit cross-matrix.

### `missed_runup.py` — the "ran away before we filled" gap
Walks signal-episodes while flat. For each: first-signal → last-signal price run-up (the potential you
sat through waiting for the 4.5% dip). Reports how many episodes ran ≥ thr, how many fully missed
(no trade) vs entered late, and **what fraction of late entries finished below the missed run-up**.

### `rootcause_buffer.py` — why the pullback crutch
Counterfactual: enter at-market at the first signal, **hold to the same exit bar** (isolates the entry).
Decomposes the extra drawdown from dropping the buffer into **pre-peak** (no exit can avoid it →
structural) vs **post-peak** (exit-fixable), and tests whether the entry knife is predictable at the
signal bar (vol / extension). Discriminates "weak exit" vs "entry can't pick safe zones".

### `fragility.py` — edge concentration / seed robustness
Two robustness lenses the giant-winner wall demands (metrics are pnl/PF/MAE-pressure, **not** a composite
re-score):
- **concentration** — drop the top-N trades, the single top symbol, the single top year, and read how much
  pnl/PF survives. A high `cum%ofPnl` from a few trades (or one dominant year) ⇒ the edge is concentrated,
  treat lever gains as fragile. (On 1930: trade-robust — top-10 drop keeps 97% — but **year-concentrated**,
  2021 ≈ 42%.)
- **seed dispersion** — per fold-set pnl/win/PF/MAE-pressure mean ± std + CV. A lever's gain inside ±1 std is
  seed-noise. (Post determinism-fix, 1930 CV ≈ 0.)

### `regime_breadth.py` — regime conditioning
model_xray F only buckets by per-symbol entry-vol; this conditions outcomes on the **market** regime
(universe breadth = %>SMA50 at entry, plus by-year). Reads `IC(breadth, pnl)` and, per breadth quartile /
year, n / avgPnL / win% / worst-loss / **%-of-gross-loss**. The avgPnL/win gap lo→hi = regime edge-strength;
the loss-share gap = regime risk concentration. Exposes whether the strategy is regime-fragile (e.g. 1930:
hi-breadth avgPnL ≈ 2× lo, 2022 the weak cluster) — points to a breadth-aware exit/sizing lever, not entry-skip.

### `gate_drain.py` — per-gate signal-drop counterfactual
Makes the gate accounting reproducible. (1) **signal→fill**: raw buy-intent episodes (rising edges of
`signal>0`) vs trades filled = the fill rate (1930: ~48%, half the intent drained, mostly by the pullback
limit). (2) **per-gate counterfactual**: toggle ONE engine gate off the champion and re-run the same signal
stream; `dPnL = pnl(variant) − pnl(baseline)`. `dPnL<0` ⇒ the gate is net-additive (earns its keep); `dPnL>0`
⇒ a drag worth revisiting. On 1930 every gate is net-additive (pullback −2877u/PF→2.29, signal-exit OFF triples
MAE-pressure) — the net-positive-cohort wall, made into a table.

### Prior-research probes (kept for reference; predate `_harness.py`, carry own scaffolding)
Reach for these to probe specific walls; they are not part of the maintained single-pass suite:
- `probe_leadtime.py` — **detectability/lead-time**: AUC separating "near a perfect peak/bottom" from
  mid-leg, and how much of the perfect-vs-confirm capture an early-exit / early-entry threshold recovers.
  The cleanest read of the top-detection wall and the entry-knife trade-off.
- `diag_rawz_zones.py` — raw-vs-z **separability at oracle zones** (the structural-target finding: z-norm
  masks raw separability; the exit fires late).
- `oracle_ceiling.py` — perfect buy-bottom/sell-peak **ceiling** under finite-capital compounding (an upper
  bound; capital-model framing, kept out of the active per-symbol suite).
- `entry_sandbox.py` — entry-timing rule proxies on oracle legs. `missed_waves.py` — >10% up-waves missed
  while flat. `cohort_waste.py` — bad-trade cohorts + score-floor counterfactual under compounding.

---

## 4. Recommended workflow

1. `model_xray.py` → headline + where the money/risk is (exit-reason table) + blind-spot overview.
2. If a head looks anti-predictive (A) or you suspect bad inputs → `feature_xray.py` to find
   HARMFUL/NOISY/REDUNDANT features and the top/bottom-separability gap.
3. To attribute outcomes → `trade_reasons.py` (which reason made/lost money, in which years).
4. To probe a specific complaint (e.g. "misses strong runners") → `missed_runup.py` then
   `rootcause_buffer.py` for the causal decomposition.
5. Before trusting any gain → `fragility.py` (does it survive drop-top-N + sit outside seed ±std?),
   `regime_breadth.py` (is the edge/risk just one regime?), `gate_drain.py` (is the gate you touched
   actually net-additive?).
6. Record durable findings in the project memory; re-run any tool on a candidate template to compare.

---

## 5. Project-specific interpretation cautions (anti-metrics)

These have repeatedly fooled analysis here:
- **Forward-peak IC looks great but is not realizable.** Always read `IC_realized` next to it; the gap
  is the exit, and the peak round-trips (direction-of-peak is predictable, realizability is not).
- **Total pnl by score-quintile is ~flat** — a few giant winners spread across all quintiles. Win-rate
  is predictable; total pnl is not. Don't gate on a head that only lifts win-rate.
- **book-MDD ≠ the objective.** The composite penalizes **per-symbol** MDD; a config that lowers
  aggregate book-MDD via diversification but raises per-symbol MDD is correctly scored worse.
- **Per-symbol "missed while flat" overstates opportunity** — single-lot means flat on X often = capital
  deployed on Y. It sizes per-signal opportunity, not a capital-model pnl.

---

## 6. Writing a new probe

Import the harness instead of copy-pasting scaffolding:

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import _cfg, _build, iter_folds, zigzag, _ic   # + _oh, fold_dirs, signals_and_trades

cfg = _cfg(1930); engcfg = _build(dict(cfg.engine))
for fold_name, sig, oh, trades in iter_folds(cfg, engcfg, 1930):
    ...  # your analysis; pool across folds
```

`_harness.py` exposes: `_cfg(tmpl)`, `_build(eng)`, `_oh(syms)`, `fold_dirs(tmpl)`,
`signals_and_trades(cfg, engcfg, fold_dir)`, `iter_folds(cfg, engcfg, tmpl)`, `zigzag(close, pct,
min_bars)`, `_ic(a,b)`, `_rankic(a,b)`, and re-exports `_causal_zscore_by_symbol`.

---

## 7. Reference findings (champion 1930, as of 2026-06-18)

- All realized profit comes from two mechanical exits (`overext_trail` capture 0.81, `trailing_stop`
  0.70); the ML `signal` exit is ~75% of trades and nets ≈0 (capture −0.09). Exit, not entry, decides
  the outcome.
- Heads lead the forward **peak** (IC 0.07–0.18) but the exit nulls it to ≈0 realized; `mfe`/`fwd`
  heads are anti-predictive on realized.
- Dropping the 4.5% pullback: 85% of the extra drawdown is **pre-peak** (no exit can avoid it). The
  buffer is a prediction-free per-symbol-MDD price discount, not a fix for a weak exit.
- Only ~4/45 entry features separate tops from bottoms; features predict at **bottoms**, go blind at
  **tops** → the top-detection wall, quantified. Candidate fix = clean **structural** top features.

See [`RESEARCH_STRATEGY_MAP.md`](RESEARCH_STRATEGY_MAP.md) for the full champion + wall map.
