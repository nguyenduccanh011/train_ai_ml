# SIGNAL-LAYER ARCHAEOLOGY — explored vs virgin axes
Date: 2026-07-09. Source: Postgres stockml (strategy_templates n=2,508; leaderboard_runs n=2,890 incl. superseded; component_slots n=4,865) + code (stock_ml/src/pipeline/experiment.py, stock_ml/src/targets/registry.py, stock_ml/src/features/catalog.py).
Champion: template 2646 `n2_2643_wavestruct_la05_lamp02`, composite 729.6 (seed-42). All scores below are `composite_score`, score_mode=live, market vn_stock only.

> NOTE (in-flight, today 2026-07-09): non-superseded runs already ABOVE champion exist —
> `xq_snr_t08_g27` 733.3 (+3.7, exit_snr_extend_threshold=0.8/min_gain 0.27 family, t2730),
> `rq_et17` 730.8 (+1.2, entry_threshold −1.7, t2737), `w0_snr_10` 730.2 (t2677).
> The pyramid stack `w0_pyr_u10_r02_snr10` 983.3 (t2681) is superseded=True (execution layer, rejected).

## 1. CHAMPION LINEAGE (name/description chain, engine_config diffs)

| gen | id | name | best | Δ | lever ADDED (engine_config diff vs parent) | lever class |
|---|---|---|---|---|---|---|
| 0 | 488 | cont_r2_en015_pen05_xp070_xh20__fs_lvup126 | 368.7 | — | fs_lvup126 entry feature family born | features |
| … | ~1828-1830 | n2_5h_lobr6 | 662.3 | — | 4-ensemble OR-union + breadth-collapse exit gate (`exit_force_gate_lowbreadth` downleg6 @ pct_above_ma50<0.25) | ensemble + regime-exit |
| 1 | 1831 | n2_brmfe | — | — | `entry_ensemble3.breadth_features=[pct_above_ma50, adv_pct]` (488-univ breadth into mfe head) | features (breadth) |
| 2 | 1834 | n2_brmfe_protect | 664.5 | +2.2 | `signal_exit_protect_lo/hi=[0.08,0.20)` + require_trend (hold winners mid-giveback) | exit-hold |
| 3 | 1837 | n2_brmfe_protect_ox12 | 666.1 | +1.6 | protect_hi 0.20→0.99, overext_pct .115→.12 | exit-hold tune |
| 4 | 1842 | n2_brmfe_velo_x20 | — | +2.7 | VELOCITY exit head integrated, signal_threshold 5.5→2.0 | **exit LABEL** |
| 5 | 1844 | n2_velo_h15 | 676.0 | ~+10 | velocity horizon 15 | exit label param |
| 6 | 1849→1930 | n2_velo_volnorm_h20 | 681.0 | +5 | `vol_normalize=true, vol_window=40, h=20, upside_horizon=8` | **exit LABEL** |
| 7 | 2373 | n2_consgate2 | 688.8 | +7.8 | `exit_gate="cons2"` (ML sell only in distribution zone; SOLD_THEN_RAN fix) | exit gate |
| 8 | 2387 | n2_consgate_w20 | 689.3 | +0.5 | exit_gate → cons2_w20 | exit gate tune |
| 9 | 2409 | n2_consw20_conv04 | 689.7 | +0.4 | `entry_pullback_conv_scale/k=0.4/floor=0.5` (conviction-graded pullback) | entry-fill × signal strength |
| 10 | 2415 | n2_..._volgate | 691.6 | +1.9 | `entry_pullback_conv_vol_z=1.0, vol_lb=40` (regime-adaptive conv gate) | regime × entry |
| 11 | 2416 | n2_..._vg_combo | 694.7 | +3.1 | `entry_pullback_conv_use_combo=true` (4-head weighted combo modulates fill) | mini-stacking |
| 12 | 2417 | n2_..._vg_combo_hb | 709.0 | **+14.3** | `conv_combo_w=[.3,.25,.2,.25]` + `conv_head_w=0.5` (ML head-z blended 50/50 with price signal) | mini-stacking |
| 13 | 2429 | n2_..._hb_nbpbw | 719.1 | **+10.1** | `nonbull_ma_win 40→35, persist 3→2, pullback_window 50→40, combo_w rpos-heavy` | regime-exit + tune |
| 14 | 2515 | n2_2429_riskstruct80_downpress | 721.5 | +2.4 | `trailing_struct_donch_win=80 + apply_overext`; exit features → exit_vol_downpress | exit structure + **exit features** |
| 15 | 2643 | n2_2515_volhold_rs8 | 729.3 | **+7.8** | `signal_exit_hold_*` (7 keys: ATR-stretch vs MA, score3-z conditional hold, rs_scale 8) | exit-hold × ML-score |
| 16 | 2646 | n2_2643_wavestruct_la05_lamp02 | 729.6 | +0.3 | `signal_exit_hold_legage_scale=.5/legage_pct=.06/legamp_scale=.2` (zigzag leg-age wave-hold) | exit-hold × structure |

Lesson from lineage: every big jump (+5 or more) came from (a) a NEW exit LABEL (velocity, vol-normalized), (b) an exit GATE conditioned on market/consolidation regime, (c) blending ML head z-scores into a decision that was previously price-only (conv combo, +14.3 — the single biggest signal-layer promotion ever), or (d) score-conditional exit-hold (+7.8). Pure threshold tuning never gave more than ~+1.5.

## 2. EXPLORED MAP

### (a) Entry target types (component_slots slot_type='entry', n=2,519 slots)
| type | n templates | best | param ranges swept |
|---|---|---|---|
| triple_barrier | 1,286 | **729.6** (excl. superseded pyr 983.3) | h {12,15,20,25,30,40,45}, pt {.08–.25}, sl {.03–.12} — DENSE |
| continuation_entry_regression | 656 | 625.8 | h {3–20}, penalty {.3–2.5}, trend_window {20–120} — DENSE, dead end |
| (global target / legacy '?') | 181 | 428.3 | — |
| action_oracle | 141 | 259.6 | 13 params swept — dead end (SMAC classifier family) |
| zigzag_pivot (bottom) | 128 | 370.7 | pct/tau/min_leg dense — dead end as MAIN head |
| forward_return_regression | 45 | 674.0 | h {10,15,20} |
| reward_risk_regression | 31 | 517.6 | h {8–30} |
| reversal_entry_regression | 21 | 444.4 | h/dip_window/min_fwd_rally (main-head); lives on as ensemble slot 1 |
| forward_return_penalized | 8 | 464.4 | ensemble slot 4 resident |
| continuation_recov_entry | 6 | 463.6 | recov_dd {.08,.12} — barely, OLD bases only |
| amplitude_oracle | 5 | −97.8 | dead |
| swing_value_regression | 4 | 349.6 | barely |
| multi_horizon_return | 3 | 451.3 | barely |
| bottom_structure_entry | 2 | 467.6 | main-head; heavily used in ensemble5 THIS WEEK (best 727.4, below champ) |
| mfe_regression | 2 | 296.8 | ensemble slot 3 resident |

### (b) Exit target types (slot_type='exit', n=2,346)
| type | n | best | note |
|---|---|---|---|
| velocity_exit_regression | 558 | **729.6** | h {5–25}, upside_h {1–20}, vol_window {20–80} — DENSE around champion (h20/u8/vw40) |
| reward_risk_regression | 835 | 694.3 | old-era zX |
| zigzag_pivot (peak) | 406 | 599.7 | dense, dead |
| risk_exit_regression | 270 | 716.6 | h {3–60} dense |
| forward_drawdown_regression | 158 | 700.2 | incl. vol_normalize variants |
| downleg_depth_regression | 19 | 696.3 | pct/max_span/peak_decay — moderate |
| triple_barrier (short) | 15 | 520.9 | dead |
| mfe / multi_horizon / swing_value | 10 | 540.6 | barely |
| **trend_scanning_exit** | **2** | **705.7** | BARELY touched as main exit head (windows [5,10,20]); as 2nd exit-ensemble head once (t2408, 666) |

### (c) Feature sets (slot feature_set_name; catalog.py defines 129 named sets)
- ENTRY: entry_lvup126_recov n=564 best **729.6** (champion, since gen ~2515). entry_lvup126_lean n=1,084 best 625.8 (old era). ~75 entry sets tried; ALL non-champion sets peak 668–716 (entry_recov_newsig 716.0, entry_lvup126_volstruct 715.4, entry_recov_mp 713.5, entry_lvup126_vwap 712.2, entry_recov_flowcross 710.9 — all tested ON the modern base and LOST 13–19 pts). Sector/RS/dyn/seq families all <300 on their bases.
- EXIT: exit_vol_downpress n=90 best **729.6**. exit_vol_market n=1,263 best 722.1 (previous resident). ~48 exit sets tried; nearest challengers exit_vol_dist2 724.6, exit_vol_dist 720.0, exit_vol_phase 716.2 — all below.
- Cross-sectional features: entry_xsec_features (t2639, 719.0, −10), exit_xsec_features (n=6, t2640 best 725.7, −4) — tried 2026-07, LOST.
- Catalog sets NEVER used in any slot: entry_dyn_pure, entry_rs_pure, entry_volbalance, entry_reversal_lean, leading_ew, leading_deriv, leading_v4 (all minor variants of losing families — low value).
- Model algos: lightgbm/entry_plain n=1,990 best 729.6. RF best 60.9, MLP 211.3, torch_GRU 185.0, xgboost registered in code (models/registry.py:425) but **0 entry/exit components ever used it**. Capacity variants tried once each on old bases (entry_hicap_l31 303.2, entry_modcap_l15 483.1).

### (d) Ensemble architecture (engine_config entry_ensemble..5)
- Count: 0 heads n=1,878 (old eras) / 4 heads n=522 (modern standard) / 5 heads n=19 (this week's bottom_structure & zigzag probes, best 729.6 = no gain) / 1–3 heads n=101.
- Slot residency is FROZEN since ~t1830: slot1 reversal_entry (640/642), slot2 continuation (549/549), slot3 mfe+breadth (604/612), slot4 fwd_return_penalized (540/544). Alternatives in slots: ~15 templates total, ever.
- z_thresholds effectively UNSWEPT on the modern base: slot1 0.9 (628/642), slot2 0.7 (543/549), slot3 0.7 (604/612), slot4 0.7 (542/544). The handful of off-values were single probes.
- norm: zscore everywhere except **7 templates csrank on slot3** (dyncsr88–97, 2026-07: best 714.1, −5) and 1 today (rq_e4csr75 719.7, −10). csrank as ensemble THRESHOLD = tried and lost.
- exit_ensemble (2nd exit head, OR-union sell): n=22, best 666.0 (trend_scanning t2408); risk/downleg/zigzag-peak 2nd heads all lost — axis tried, closed at the OLD base though (never re-tried above 700).

### (e) Signal-side knob sweeps (distinct values × where)
| knob | swept values (n templates) | on modern (≥715) base? |
|---|---|---|
| entry_threshold (zE buy bar) | 26 values −5.0..1.0; −1.9 n=555, −1.2 n=897 | {−3.0,−2.6,−2.2,−2.1,−1.9,−1.7} — today's −1.7 gives **+1.2 (730.8)**; gaps −1.8, −1.6, −1.5 untested on champ |
| signal_threshold (zX sell bar in decoupled) | 30 values; 2.0 n=995, 5.5/5.0 old-era | **only {1.8, 2.0, 2.2}** among ≥715. 1.6/1.7/1.9/2.1/2.3–2.6 never on champ base |
| exit_threshold | 0.07 n=2,073 (mostly inert in decoupled); 1.75–3.0 probed on 2482 base (720.7, −1) | thin |
| z-norm lookback | **NEVER swept — hardcoded window=252, min_periods=60** (experiment.py:2624-2625) | 0 values |
| entry/exit z EMA smoothing | strategy-variant only, span hardcoded 5 (experiment.py:2587,2605,2613); decoupled_ema: 8 templates, **1 run ever, 725.2 (−4)** | 1 datapoint |
| entry ensemble slot z_thr | frozen .9/.7/.7/.7 (see d) | no |
| regime_model slot | **0 templates EVER** (component_slots has only entry/exit types). Wiring exists only in portfolio layer (portfolio/gating.py:22 apply_regime_gate ← portfolio/policies.py:69); engine-side regime is via gates instead | — |
| regime gates in engine | nonbull (n=843), lowbreadth exit (n=608), cons gate (n=163), entry_market z-gate (n=1,137) — all resident in champion. regime_index_symbol=VN30F1M n=3 (old base, 492.7) | resident |
| seasonality / calendar | **nothing anywhere** — no month/weekday/Tet features in catalog.py, no calendar keys in any engine_config | 0 |
| meta-labeling / stacking | closest thing = entry_pullback_conv_use_combo (weighted 4-head z combo, modulates FILL DEPTH only, +14.3 historically). No 2nd-stage model on 1st-stage trades exists in code | partial |

## 3. VIRGIN LIST (exists in code, never/barely templated)

1. **entry_head_csrank_gate** — experiment.py:2485-2512 (consumed 2149, plumbed 2718). Routes a head's CROSS-SECTIONAL rank as an entry gate (csrank>=pct); forensic in-code comment: per-symbol z FLIPS the mfe/momentum head's outcome signal (raw IC +0.05 → z −0.03) but csrank PRESERVES it (lo→hi quintile WR 22%→37%). **0 templates ever.** Distinct from the csrank-as-threshold probes (those replaced the union trigger; this gates the existing buy).
2. **exit_force_gate_vn30** — experiment.py:1681-1700 (_vn30_regime_mask), consumed at 2387-2394. Real-index (VN30F1M) risk-off exit tightening, "real-index analog of the EW breadth-collapse gate", forensic vn30_regime_1875: the 2022 MDD cluster coincides with VN30 risk-off. **0 templates.** Config-only key.
3. **downleg_skip_bull** — experiment.py:2350-2360. Releases the downleg tail-cut in a SUSTAINED bull (index above MA for persist days) so trend-intact winners aren't clipped; loose backstop still applies. **0 templates.** Direct shot at the −100u exit-giveback pool.
4. **entry_z_low_threshold (U-shaped entry)** — experiment.py:2234-2240. Also-buy the extreme-LOW zE tail; in-code diagnostic: lowest decile (~−12% off 20d high) bounces +2.4%/20d. 4 templates, ALL on the pre-700 era (best 333.3, t758). Never on the champion base.
5. **z-norm lookback window / min_periods** — experiment.py:2624-2625, hardcoded (252, 60) since inception; `_causal_zscore_by_symbol` already parametrized. Governs zE, zX AND all 4 ensemble z's. Zero values ever tried. 2-line change to expose.
6. **EMA smoothing spans on decoupled** — spans hardcoded 5, selected by strategy suffix (experiment.py:2587/2605/2613). One leaderboard run ever (decoupled_ema 725.2). Span grid (3/5/8/12) × side (entry-only / exit-only / both) unexplored; exit-side smoothing changes WHEN the zX>2.0 sell fires = exit-timing seam.
7. **trend_scanning_exit as main exit head on the modern base** — targets/trend_scanning.py, registry.py:113. n=2 ever, best 705.7 on an old base; López de Prado let-profits-run label, mechanistically matched to the giveback pool. Config-only (slot target_config swap).
8. **early_wave_v2 / early_wave_exit** — targets/early_wave.py:46/111, registered (registry.py:123-124). 3-class early-wave entry + companion exit. **0 slots ever.** Config-only.
9. **Ensemble slot z_threshold micro-grid + slot-target swaps** — slots frozen at .9/.7/.7/.7; continuation_recov (6 old templates), zigzag bottom in slot (2), swing_value (4) barely tried; per-slot sweep on champion base virgin. Config-only.
10. **entry_breadth_gate on the recombine family** — experiment.py:1502-1522/2508-2512. n=44 but ALL on the SMAC/GRU side quest (best 201.4); never on the decoupled champion. Entry-side breadth timing is proven as a FEATURE (ensemble3) and as an EXIT gate, never as an entry gate here.
11. **regime_model slot end-to-end** — schema + portfolio wiring exist (gating.py:22, policies.py:69) but the template backtest path never consumes it; `portfolio` engine key n=3, best −146. Needs pipeline wiring = CODE-heavy.
12. **Seasonality/calendar axis** — nothing in code. Needs feature + catalog work = code-heavy.
13. **xgboost algo for entry/exit components** — models/registry.py:425/434/492; 0 components. DB-insert + slot pointer = config-only, but LGBM-vs-XGB rarely moves composite; low priority.
14. **True meta-labeling (2nd-stage model on 1st-stage entries)** — absent from code. The conv-combo (+14.3, biggest signal promotion ever) is the degenerate 1-layer version and stopped at modulating fill depth; a stage-2 head could gate/size the trade itself. CODE-heavy but historically the richest lever class.

## 4. TOP-8 VIRGIN AXES (plausibility × cheapness)

| # | axis | why (mechanism vs champion's edge) | cost | exact test |
|---|---|---|---|---|
| 1 | entry_head_csrank_gate | Forensic-proven preserved ordering (WR 22→37%) never routed into the buy; big_win factory = keep only cross-sectionally strong candidates | config | `engine.entry_head_csrank_gate={"score_col":"score3","pct":0.25,...}` sweep pct {.15,.25,.4} |
| 2 | downleg_skip_bull | Directly releases the tail-cut that clips trend-intact winners = the −100u giveback pool, regime-conditioned like every past MDD win | config | `engine.downleg_skip_bull={"ma_win":50,"persist":3}` (+ma_win 35) |
| 3 | exit_force_gate_vn30 | New information (real index ≠ 488-EW proxy); regime-cluster exit tightening is the historically validated pattern (2429's +10 was exactly this class) | config | `engine.exit_force_gate_vn30={"index_symbol":"VN30F1M","gate":"downleg6",...}` |
| 4 | signal_threshold × exit_z_ema_span grid on champ | The zX>2.0 sell bar is THE exit-timing seam; only 3 values ever tried at ≥715, smoothing has 1 datapoint (span locked at 5) | config (strategy suffix) | signal_threshold {1.6,1.8,1.9,2.1,2.2,2.4} × strategy `..._decoupled_ema`; expose span if grid hits |
| 5 | z-norm lookback (252/60) | Every z in the system flows through one untested constant; shorter window = adaptive-regime z, longer = stable; interacts with the patient-fill edge | 2-line code | expose `engine.z_window/z_min_periods`, sweep {126, 252, 504} × {40, 60, 90} |
| 6 | trend_scanning_exit head @ modern base | Let-profits-run label aimed exactly at giveback; 705.7 on a 681-era base (≈ par) never re-tested after +50 pts of base improvements | config (slot target_config) | exit slot `{"type":"trend_scanning_exit","windows":[10,20,40]}` keep exit_vol_downpress features |
| 7 | entry_z_low_threshold (U-shape) | +2.4%/20d bounce cohort documented in-code; champion's reversal ensemble covers dips but NOT the deep-washout tail; feeds the big_win factory with deep fills | config | `engine.entry_z_low_threshold` {−2.5, −3.0} on champ (watch knife overlap w/ downleg veto) |
| 8 | ensemble slot z_thr micro-grid + slot swaps | 4-head union is the entry factory; thresholds frozen at .9/.7/.7/.7 since t1830 while the base moved +65 pts; continuation_recov/early_wave_v2 never given a slot | config | per-slot ±0.2 grid; slot2 target → continuation_recov_entry; slot5 → early_wave_v2 |

Deliberately NOT ranked: regime_model slot (needs pipeline wiring), seasonality (needs new features), meta-labeling stage-2 (code-heavy — but flag it as the highest-ceiling code project given conv-combo's +14.3).

## 5. "ALREADY DUG — DON'T RE-DIG" WARNINGS
- velocity_exit hyperparams: h×upside_h×vol_window dense-swept (558 templates); champion h20/u8/vw40 is the optimum. Best alternatives: risk_exit 716.6, fwd_drawdown 700.2, downleg_depth 696.3.
- Entry feature sets: ~75 variants; everything beats-tested ON the modern base lost 13–19 pts (recov_newsig 716, volstruct 715.4, vwap 712.2, flowcross 710.9). Same for exit sets (dist2 724.6, dist 720). Marginal-feature axis is heavily mined.
- csrank as ensemble trigger norm: 8 templates 2026-07, all −5..−20. (The GATE form, #1 above, is different and untested.)
- Cross-sectional feature injection: entry_xsec 719.0 / exit_xsec 725.7 — lost.
- exit_ensemble (2nd exit head OR-union): 22 templates, best 666 — lost on the old base; only worth re-touching via #6's main-head swap, not the union.
- entry_ensemble5: 19 templates this week (bottom_structure/zigzag), best 727.4–729.6 = no gain.
- Non-LGBM algos: RF/MLP/GRU catastrophically below (60–211). SMAC/action_oracle classifier family: 145 templates, ceiling 259.6.
- entry_threshold: −1.7 already run today (730.8); the −1.8/−1.6 completion is a 2-run errand, not a discovery axis.
- triple_barrier entry grid (pt/sl/h): 1,286 templates, dense. Champion pt.15/sl.08/h30 sits in a mapped basin.
- Pyramiding / sizing / execution: closed this week (w0_pyr 983.3 superseded=rejected).
