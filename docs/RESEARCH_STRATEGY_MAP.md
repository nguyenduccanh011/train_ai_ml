# Research Strategy & Wall Map — Stock ML

Long-term direction document. Captures the current champion, the **foundational understanding**
of why the strategy behaves as it does, the complete map of **tested-and-refuted** levers (so we
stop re-walking them), the **validated winning patterns**, and the **genuine breakthrough
directions** that lie outside the current frame.

---

## 0. CẬP NHẬT 2026-07-29 (đọc trước — thân doc bên dưới là snapshot 06-18, đã lỗi thời một phần)

**Champion hiện tại**: dòng **dl63size (tmpl 3443)**, retrain trên data Fireant 901-mã (composite ~662,
WR ~62%, CAGR-NAV Stage-1 ~66%/DD −13.7%). Số deploy Stage-2 chính thức (Sieu Tin Hieu, 3-seed T+2,
`stat_mode=causal`): **CAGR 138.5% / DD −13.0%** — golden-guarded byte-exact
(xem [refactor/PORTFOLIO_LAYER_UNIFICATION.md](refactor/PORTFOLIO_LAYER_UNIFICATION.md)).

**Khám phá lớn kể từ snapshot 06-18** (chi tiết trong doc unification + memory phiên):
- **Tầng danh mục (Stage-2) là nguồn alpha lớn nhất tìm được**: conviction-sizing (tổng vốn≤1) +
  slot-preemption + meta-priority LGBM + gate SKIP-causal/ret7/overshoot + green-trail/early-cut,
  K=10. Từ 07-2026 là MỘT module duy nhất `stock_ml/portfolio/` dùng chung backtest ↔ serving.
- **Panel/universe là siêu-tham-số**: CAGR lồi, đỉnh ~200-300 mã; full-901 nén conviction làm
  winner bị chiếm slot; universe hindsight thổi phồng CAGR gấp đôi (survivorship); dyn900 cao là do
  penny; **61-mã champion vẫn thắng Calmar** so mọi universe rộng đã thử.
- **Data**: Sieu Tin Hieu > DuckDB train (AAS/BSI train chưa back-adjust); snapshot Desktop trễ CA
  mới (vụ PVD −8.8pp); lớp rights-issue NON_ADJUSTABLE không provider nào adjust.
- **Nguyên lý audit causal** (chống gắn cờ nhầm): so **mốc-QUYẾT-ĐỊNH** vs **mốc-DATA**. Overshoot
  filter = causal (+6.1pp, quyết-tại-fill, data quá khứ); short_tilt5 = leak thật (quyết-sớm,
  data-muộn). Pipeline giờ deterministic tuyệt đối (LGBM deterministic+force_col_wise).
- **Đã bão hòa** (đừng đi lại): entry-selection + execution per-trade (preemption, meta-prio,
  entry-bỏ-lỡ, regime-entry-gate, time-stop — bị preempt mask). **Dư địa thật còn lại = quản trị
  exposure CẤP DANH MỤC**: regime-sizing, breadth-exit, VN30F-hedge — giờ rẻ để test vì mỗi thí
  nghiệm chỉ là một variant chạy qua `run_portfolio` với golden guard.

> Thân doc dưới đây giữ nguyên làm sử liệu snapshot 2026-06-18 (champion khi đó là t1930,
> composite 489; các phần "wall" về entry/exit vẫn đúng và đã được các vòng 07-2026 xác nhận thêm).

> To **analyse** a strategy (prediction quality, blind spots, per-trade reasons, feature health),
> see [`STRATEGY_ANALYSIS_TOOLKIT.md`](STRATEGY_ANALYSIS_TOOLKIT.md) — start with
> `results/_swing_capture/model_xray.py`.

---

## 1. Current champion

**`t1930` (n2_velo_volnorm_h20) — comp 489.0 · #1.** Promoted 2026-06-16 over the t1830→t1844 chain;
the change vs t1844 is a **vol-normalized velocity exit** (`velocity_exit_regression` h20,
`vol_normalize`, vol_window 40, upside_horizon 8) at signal_threshold 2.0 — seed-robust, leakage-clean.

Config (engine_config on top of the 5-head dual-ML recombine), largely inherited from the t1830 line:
- **5 UNION entry heads**: momentum (`score`) + reversal (`score2`, reversal_entry, z0.9)
  + continuation (`score3`, continuation_entry p0.5, z0.7) + mfe (`score4`, mfe_regression h20, z0.7)
  + fwd_pen (`score5`, forward_return_penalized h20, z0.7). Entry `upleg_abovema20` gate.
- **Patient entry**: 4.5% pullback limit, window 50 (drops runaways; no strength-skip, no fill-if-missed).
- **Profit engine**: `overext_bull` + `trailing_activate` + trend-intact trail-skip (the two mechanical
  trails carry ~all realized pnl; see the toolkit's exit-reason attribution).
- **Risk**: `exit_force_gate` downleg + low-breadth cut (488-universe breadth-collapse = 2022 mdd
  control) + entry/exit market gates.

Reproduce/X-ray via the analysis toolkit (`results/_swing_capture/`, see
[`STRATEGY_ANALYSIS_TOOLKIT.md`](STRATEGY_ANALYSIS_TOOLKIT.md)); multi-seed via the per-template
`results/tmpl_1930_*/folds` dirs.

**Diagnostic findings (2026-06-18, toolkit run):** profit is 100% from the two mechanical trails
(`overext_trail` capture 0.81 / `trailing_stop` 0.70); the ML `signal` exit is ~75% of trades and nets
≈0 (capture −0.09). Entry heads lead the forward peak (IC 0.07–0.18) but the exit nulls it to ≈0
realized (`mfe`/`fwd` heads anti-predictive on realized). Dropping the 4.5% pullback: 85% of the extra
drawdown is **pre-peak** (exit-unfixable) — the buffer is a prediction-free per-symbol-MDD discount, not
a fix for a weak exit. Only ~4/45 entry features separate tops from bottoms (features predict at bottoms,
blind at tops) → the top-detection wall, now quantified; candidate fix = clean **structural** top features.

---

## 2. The foundational truth (read this first)

The strategy is **per-symbol, single-lot, price-daily** (verified: composite uses `mdd_per_symbol`,
backtest loops `groupby(symbol)`, book-MDD is display-only, no portfolio capital). The objective is
**per-symbol signal quality**, not portfolio diversification.

**ML genuinely predicts — the "pure-rule" verdict is a measurement artifact.** The entry heads
predict the forward **PEAK magnitude** very well: `IC(score, MFE_fwd20) = +0.24` (mfe head +0.25).
But the **mechanical exit** collapses that onto realized pnl: `IC(score, realized_pnl) = +0.026` —
the exit **hides ~9–25× of the entry ML's predictive signal**. The long-standing "entry head is
anti-predictive within the losers" finding is an **artifact the exit manufactures**, not a property
of the entry.

**Why unmasking still doesn't pay.** The predicted peak is **transient — it round-trips**. For
top-momentum trades that exit as signal-losers: mean `fwd_ret20 = −2.57%`, only **42.6%** are still
profitable by bar 20, despite a mean intrabar peak of +9.56%. And `corr(entry_z, exit_z) = 0.76`
(the heads agree → no cheap "incoherence pocket" to gate on).

> **Irreducible truth:** *direction-of-the-PEAK is predictable; REALIZABILITY-of-the-peak is not.*
> The ML predicts the exact high you cannot capture. The mechanical reactive exits are
> composite-optimal **given** that unpredictability. This is one level deeper than "direction is
> unpredictable" — the *peak* direction IS predictable, its *realizability* is the wall.

Corollary wall (the one we hit ~8×): a few **unpredictable giant winners are spread across every
cohort**, so **WIN-RATE and per-bar are predictable but TOTAL-PNL is flat across quintiles** → every
low-quality cohort is still net-positive → **exclusion/selection always loses pnl**. Only two things
add value: **ADD orthogonal winners** (union heads) and **tighten risk on regime-clusters** (breadth).

---

## 3. Validated WINNING patterns (what actually moved comp)

1. **Orthogonal WINNER heads via UNION** — each a *different entry style* with its own forward target,
   unioned (can only ADD, never cut): reversal +3.7, breakout +2.5, mfe +2.3, fwd_pen +0.6 (now
   **saturating**; continuation-h20 / reward_risk / RS-feature heads were FLAT = redundant).
2. **Profit-engine tuning** — `overext_bull` release lets healthy runners ride into trailing instead
   of premature top-sells (+4 comp, pnl↑ AND mdd↓). Now **exhausted** at thr 0.35.
3. **Regime-cluster EXIT tightening** — `breadth-collapse` cut (488-universe `pct_above_ma50<0.25`
   → tighter downleg6) cleanly removes the 2022 catastrophe-cluster losers without clipping other-year
   winners (+8.8 comp, mdd 1.69→1.37). The **single biggest mdd lever**.

Meta-rule: **a real win ADDS orthogonal winners or tightens a separable regime-risk cluster. Anything
that EXCLUDES/SELECTS or RE-TIMES a pivot hits the wall.**

---

## 4. WALL MAP — tested & REFUTED (with the WHY). Do not re-walk.

| Lever | Result | Why it failed |
|---|---|---|
| Entry-head reorient (continuation/fwd_ret target) + selectivity | NEG, monotone | rules already capture the near-high edge; z is anti, tightening cuts winners |
| Tighten entry on range_pos / breadth / RS (all |IC| 0.10–0.18) | NEG | strong but REDUNDANT with momentum (already entered) |
| Breadth as ENTRY gate | NEG | low-breadth cohort still net-positive |
| Breadth / RS / xsec as HEAD feature | FLAT | per-symbol target can't use a market-timing signal; redundant |
| Cross-sectional (csrank) union normalization | NEG | union bypasses gates → adds risky gate-bypass names |
| `entry_breakout_floor` (gate momentum buy by breakout csrank) | NEG | removes net-positive momentum trades |
| Peak-timing exit (top_reversal_exit / pre-peak head) | NEG | peak vs pullback-continue indistinguishable real-time; clips runners |
| Exhaustion (distribution/momentum-decay/vol-churn) exit | WEAK | |IC| ≤ 0.047, below actionable; mostly continuation signals |
| Downleg variants (dl6–20, dlvol, multicycle OR, Donchian, wave-struct) | NEG | 12% is optimal; cut-direction unpredictable (|r|≤0.075) |
| `downleg_skip_bull` (regime release of tail-cut) | NEG | coincident bull flag is ON into the top → disarms before the reversal |
| Exit ML head replacing/exposed (lower signal_threshold) | NEG | reward_risk sign-inverted (sells winners); layered = clips |
| Entry-exit coherence veto (hold on high entry-z) | NEG | signal-exits are coherent cuts; entry-z flipped vs extension |
| WR-expression (csrank-conditioned tighter exit on losers) | NEG | mechanical exits already harvest those losers; no pnl slack |
| `min_age`/incubate/`reentry_cooldown` exit timing | NEG | deferring the cut lands at a worse price; mdd↑ |
| entry_skip_nonbull / real-index (VN30) regime | NEG | removes net-positive trades / worse timing |
| 6th UNION head: volume-region-balance (`updown_vol_20`/`ad_balance_20`, mfe target, z0.7→−1.1) | FLAT/NEG | residual-IC +0.19 vs head-scores did NOT convert: firing region overlaps the 5 heads; lower z adds net-neutral trades + mdd↑ (saturation, same as continuation/RS heads) |
| `dist_arm` distribution-armed protective trail (ad_balance_20 ≤ thr on in-profit peak → tight band) | NEG | round-trippers vs riders separate at the PEAK (ad_bal 1.2 vs 2.3) but the arm fires on EVERY low-ad_balance bar mid-advance → clips riders; pnl↔mdd slider lands −2 to −32 comp at every (thr,trail,min_gain). Does lower mdd (1.24→1.15 at min_gain 0.40) but pnl loss > mdd gain |
| `traded_value` money-flow exit (tv_z 60d / tv_trend 5d-20d at peak) | NEG (magnitude, not selection) | tv_z IC(giveback) +0.171 (highest found) BUT round-trippers vs riders have IDENTICAL turnover at peak (tv_z 1.58 == 1.58) → predicts giveback MAGNITUDE like ATR (redundant with vol_protect), can NOT select which peak round-trips. traded_value was the only unused duckdb column; it is another magnitude axis, not the realizability axis |
| Real-index (VN30F1M futures) regime-cluster EXIT gate (`exit_force_gate_vn30`, downleg6/8 during VN30 confirmed-downtrend/drawdown) | NEG (worse than the EW proxy) | the EW 488-univ breadth-collapse gate already in the champion is worth +10.1 comp (mdd 1.716→1.236); VN30-as-replacement scores 471.4 — BELOW even no-gate (476.5) — and stacked-on-champ is −7 to −32. The VN30 large-cap index is a NOISIER regime proxy for an all-cap 61-name book (VN30 can be down while the traded names are fine, and vice versa); 2022 cluster coincides with VN30 downtrend but EW-488-breadth (matches the traded universe's composition) separates it more cleanly. Real index ≠ better breadth here |

Composite is NOT a fake-optimum crutch: per-bar already re-tuned 0.27→0.13, total 0.20→0.34 (Sortino);
`mdd_per_symbol` is correct for the per-symbol goal (book-MDD improvements via diversification are
irrelevant here and correctly scored lower).

---

## 5. Genuine breakthrough directions (outside the current frame)

The price-daily + per-symbol signal space is systematically exhausted, **including the masked layer**.
Real further gains require something the current frame doesn't have:

1. **New orthogonal DATA** — the "realizability" signal (which peaks DON'T round-trip) is not in
   price-daily. **Confirmed empirically (2026-06-16):** every price/volume/turnover signal that
   correlates with the giveback does so as a *magnitude* predictor (turnover-z IC 0.171, ATR 0.165,
   dist_count 0.094, bull_trap 0.163) but **round-trippers and riders are INDISTINGUISHABLE at the
   peak** (e.g. turnover-z 1.58 == 1.58; ad_balance 1.2 vs 2.3 separates only until you make it causal,
   then it clips riders). The only unused duckdb column, `traded_value`, was tested and is a magnitude
   axis too. Candidates to INGEST (none in the duckdb today; intraday exists but is too sparse, ~3.8k
   rows): **foreign/institutional flow per day, sector/industry membership + intra-sector RS,
   fundamentals, real index/breadth of actual indices.** This is the one class that could break the
   peak-realizability wall.
2. **A realizability TARGET** — instead of training entry heads on forward-return/MFE (which the exit
   masks), train/evaluate against *realized-under-this-strategy's-exits* (a closed-loop meta-label) AND
   add a feature that distinguishes durable advances from round-trips (likely needs #1).

   > **Sequence-model-on-path is REFUTED (2026-06-16, `_tmp_analysis/seq_path_proof2.py`).** The
   > "different model class" escape hatch — a GRU/1D-CNN on the raw 40-bar PATH instead of point
   > features — was tested head-to-head vs LightGBM on a 207k-sample temporal split (purge-gapped).
   > Forward-DIRECTION OOS rank-IC: GRU −0.009 / CNN +0.030 / LGB-point +0.001 / LGB-flat +0.027 —
   > **~0.03 for every representation; the path shape adds nothing.** Forward-DOWNSIDE (round-trip
   > magnitude): LGB-point **+0.281** crushes GRU +0.104 / CNN +0.146 — the sequence encoding DILUTES
   > the vol-magnitude signal point features already capture. So direction is absent at *any* model
   > class and magnitude is fully in point-vol features. The wall is in the DATA, not the model
   > representation. This closes #2 within the current data → realizability needs #1 (new data).
3. **Portfolio structure** (conviction sizing / capital allocation) — currently **excluded** by the
   single-buy/sell per-symbol constraint. Would let the predicted-peak-magnitude express as *size*
   rather than *timing*. Off the table unless the constraint is revisited.

---

## 6. Operational notes

- **Replay harness** (zero-retrain, recombine/engine A-B): `_tmp_analysis/replay_t1804.py`
  (`TEMPLATE_ID` override) + `ab1828.py` (champion = `+{exit_force_gate_lowbreadth downleg6/0.25}`).
- **Multi-seed**: 3 fold-dirs `results/tmpl_1830_*`; paired ON/OFF per dir removes seed noise (~±3 comp).
  Always require a win to hold on ALL 3 seeds; require seed42 LAST on DB retrains (config_hash clobbers).
- **Off-by-default research knobs added this session** (golden-green, champion unchanged): `entry_ensemble2/3/4`
  (score3/4/5 heads), per-head `norm:csrank`, `entry_breadth_gate`, `exit_force_gate_lowbreadth`,
  `downleg_skip_bull`, `breadth_features`/`xsec_features` head hooks, `entry_head_csrank_gate`.
- **Pipeline** now supports up to 5 entry heads + 488-universe breadth/xsec feature injection.
