"""Canonical DSL feature catalog — the single source of truth for definitions + sets.

Both ``scripts/seed_features.py`` (DB mirror for API/UI) and
``src/features/resolver.py`` (backtest evaluation) read from here, so a feature's
formula lives in exactly one place. Adding a feature = one line in ``FEATURES``.

The DB ``feature_def``/``feature_set`` tables are a *read-only projection* of this
module (one-way: catalog → ``seed_features`` → DB). The API/UI never write feature
definitions, so the DB can never diverge from what the resolver actually runs.

Per-symbol features are faithful ports of the legacy leading_v2 builder (parity
gate in Phase 5). Cross-sectional features leave warmup NaN (no bfill) — the
leakage fix from the design doc §5.2.
"""

from __future__ import annotations

from stock_ml.src.features.dsl.engine import extract_deps
from stock_ml.src.features.dsl.hashing import expr_hash
from stock_ml.src.features.dsl.parser import parse

# Base features first so dependency edges (#ref) resolve to already-defined names.
FEATURES: dict[str, str] = {
    # --- ohlcv_basic ---
    "ret_1d": "$close / Ref($close, 1) - 1",
    "ret_5d": "$close / Ref($close, 5) - 1",
    "ret_10d": "$close / Ref($close, 10) - 1",
    "ret_20d": "$close / Ref($close, 20) - 1",
    "close_to_open": "$close / $open - 1",
    # --- moving_averages ---
    "sma_5_ratio": "$close / Mean($close, 5) - 1",
    "sma_20_ratio": "$close / Mean($close, 20) - 1",
    "sma_50_ratio": "$close / Mean($close, 50) - 1",
    "sma_200_ratio": "$close / Mean($close, 200) - 1",
    "ema_10_ratio": "$close / EMA($close, 10) - 1",
    "sma5_cross_sma20": "Mean($close, 5) / Mean($close, 20) - 1",
    # --- ma slope (3-bar normalized slope of the moving average + alignment) ---
    "ma5_slope": "Delta(Mean($close, 5), 3) / Mean($close, 5)",
    "ma10_slope": "Delta(Mean($close, 10), 3) / Mean($close, 10)",
    "ma20_slope": "Delta(Mean($close, 20), 3) / Mean($close, 20)",
    "ma5_accel": "Delta(#ma5_slope, 3)",
    "ma_align": "(Mean($close, 5) > Mean($close, 10)) * 1.0 + (Mean($close, 10) > Mean($close, 20)) * 1.0",
    # --- momentum ---
    "rsi_14": "RSI($close, 14)",
    "rsi_7": "RSI($close, 7)",
    # Cutler's RSI (simple-MA smoothing) — causal variant, more reactive than Wilder.
    "rsi_14_sma": "RSISMA($close, 14)",
    "macd_line": "MACD($close, 12, 26, 9).line",
    "macd_hist": "MACD($close, 12, 26, 9).hist",
    "roc_10": "ROC($close, 10)",
    # --- trend ---
    "adx_14": "ADX($high, $low, $close, 14).adx",
    "plus_di_14": "ADX($high, $low, $close, 14).plus_di",
    "minus_di_14": "ADX($high, $low, $close, 14).minus_di",
    # --- volatility ---
    "atr_14_ratio": "ATR($high, $low, $close, 14) / $close",
    "bb_width_20": "Bollinger($close, 20, 2).width",
    "bb_pct_20": "Bollinger($close, 20, 2).pct",
    "realized_vol_10": "Std(Pct($close, 1), 10)",
    # --- volume_advanced ---
    # Denominator guard: a non-trading stretch (volume == 0 across the whole window)
    # makes Mean == 0 and the raw ratio 0/0 = NaN. Adding (Mean <= 0) keeps the
    # denominator 1 there -> ratio = 0 (numerator is also 0), while bars with real
    # volume keep the exact denominator (bit-identical), same idiom as the wick ratios.
    "volume_ratio_5": "$volume / (Mean($volume, 5) + (Mean($volume, 5) <= 0) * 1.0)",
    "volume_ratio_20": "$volume / (Mean($volume, 20) + (Mean($volume, 20) <= 0) * 1.0)",
    "obv_slope_10": "Delta(OBV($close, $volume), 10) / Abs(OBV($close, $volume))",
    "mfi_14": "MFI($high, $low, $close, $volume, 14)",
    # --- market_structure ---
    "dist_52w_high": "$close / Max($close, 252) - 1",
    "dist_52w_low": "$close / Min($close, 252) - 1",
    "high_low_pct_5d": "(Max($high, 5) - Min($low, 5)) / $close",
    # --- short-horizon recovery-from-washout (this-session diagnostic: at the SAME entry
    #     score zE these carry ORTHOGONAL fwd-return signal the continuation score misses —
    #     within-zE rank-IC dist_10d_low +0.053, recov_setup +0.036; in the BUY zone the top
    #     recov_setup quintile doubles fwd10 +2.74% vs +1.33%. "Vừa rũ đáy rồi hồi" = mean-
    #     reversion-after-washout regime, distinct from the coincident continuation level.) ---
    "dist_10d_low": "$close / Min($close, 10) - 1",
    # position in the 20d range [0,1]; denom guard for a flat (frozen) window where
    # Max == Min -> range 0 -> 0/0 NaN, same idiom as bb_pct / the wick ratios.
    "range_pos_20": (
        "($close - Min($close, 20)) / ((Max($close, 20) - Min($close, 20)) "
        "+ ((Max($close, 20) - Min($close, 20)) <= 0) * 1.0)"
    ),
    # deep recent pullback (10-bar min of the 20d-high drawdown, negated -> depth>=0)
    # times how far it has recovered up the 20d range -> high only when a DEEP dip is now
    # RESOLVED (washed out then recovered), the confirmed-reversal setup zE under-weights.
    "recov_setup": "(0 - Min($close / Max($close, 20) - 1, 10)) * #range_pos_20",
    # --- exhaustion ---
    # Denominator guard: on a limit-lock bar high == low (whole day at one price),
    # so range = 0 and the raw ratio is 0/0 = NaN. Adding ($high <= $low) makes the
    # denominator 1 there -> ratio = 0 (numerator is also 0), while normal bars
    # (high > low) keep the exact denominator (bit-identical). Direction of a lock
    # is carried separately by is_limit_lock.
    "upper_wick_ratio": "($high - Max($open, $close)) / (($high - $low) + ($high <= $low) * 1.0)",
    "lower_wick_ratio": "(Min($open, $close) - $low) / (($high - $low) + ($high <= $low) * 1.0)",
    "body_ratio": "Abs($open - $close) / (($high - $low) + ($high <= $low) * 1.0)",
    # TOP-STRUCTURE features (C-arc 2026-06-18, top_struct_probe): the structural TOP signals the
    # momentum/extension entry set is blind to (opposite-signed top-vs-bottom IC). Fed to the entry-head
    # retrain so it can natively down-score du-dinh (right-shoulder) entries. Causal (O/H/L only).
    "upper_wick_5": "Mean(($high - Max($open, $close)) / (($high - $low) + ($high <= $low) * 1.0), 5)",
    "lower_high_20": "(Max($high, 20) < Ref(Max($high, 20), 20)) * 1.0",
    # Structure break: close pierces the 20-bar support that existed as of the prior bar
    # (causal, Ref shift). Forensic (SMAC v5): trades that NEVER break their entry-support
    # win 96% (+9.9%) vs 51% if they break it — the single strongest winner/loser separator
    # a TA reads (higher-low held vs lower-low broken), which the model was blind to.
    "broke_support_20": "($close < Ref(Min($low, 20), 1)) * 1.0",
    # MACD-HIST SHAPE family (user temporal-pattern insight 2026-06-18): a single point loses the curve
    # SHAPE (slope=0 at BOTH a forming top and bottom); give the model the raw value + relative percentile
    # + slope/curvature/lags so trees read the shape (bottom curving up vs top curving down). For the EXIT
    # head (rollover detection = the dead-zone giveback). All causal (EMA/Ref/TsRank past-only).
    "macd_line_sh": "EMA($close, 12) - EMA($close, 26)",
    "macd_hist_sh": "#macd_line_sh - EMA(#macd_line_sh, 9)",
    "macd_hist_slope_sh": "#macd_hist_sh - Ref(#macd_hist_sh, 1)",
    "macd_hist_accel_sh": "#macd_hist_slope_sh - Ref(#macd_hist_slope_sh, 1)",
    "macd_hist_l1_sh": "Ref(#macd_hist_sh, 1)",
    "macd_hist_l2_sh": "Ref(#macd_hist_sh, 2)",
    "macd_hist_slope_l1_sh": "Ref(#macd_hist_slope_sh, 1)",
    "macd_hist_slope_l2_sh": "Ref(#macd_hist_slope_sh, 2)",
    "macd_hist_pctile_252_sh": "TsRank(#macd_hist_sh, 252)",
    "macd_hist_slope_pctile_252_sh": "TsRank(#macd_hist_slope_sh, 252)",
    # RSI/extension SHAPE (research 2026-06-18: rsi_slope_5 topIC -0.155 = STRONGEST top-signal; the POINT
    # rsi_14 only -0.040 = shape>>point; ext_ma20_slope5 -0.146). Slope/percentile of momentum, for the
    # ENTRY head to tell a dip-that-bottoms from a dip-that-knifes (declining RSI at a dip = knife). Causal.
    "rsi_14_slope5_sh": "#rsi_14 - Ref(#rsi_14, 5)",
    "rsi_14_pctile_252_sh": "TsRank(#rsi_14, 252)",
    "ext_ma20_sh": "$close / Mean($close, 20) - 1",
    "ext_ma20_slope5_sh": "#ext_ma20_sh - Ref(#ext_ma20_sh, 5)",
    "high_low_pct": "($high - $low) / $close",
    # Signed limit-lock flag: +1 = locked at ceiling (close jumped up), -1 = locked
    # at floor (close dropped), 0 = normal bar. Detected as a zero-range bar that
    # still traded (high == low and volume > 0); direction from the day's close move.
    "is_limit_lock": "Sign(Delta($close, 1)) * (($high <= $low) * 1.0) * (($volume > 0) * 1.0)",
    # --- volatility_regime ---
    "atr_regime": "ATR($high, $low, $close, 14) / Mean(ATR($high, $low, $close, 14), 50) - 1",
    # eps in the denominator so a fully-flat window (Mean of band width == 0 over 50
    # bars, i.e. >=~70 identical closes on thin/price-locked names) yields a finite
    # ~ -1 (no squeeze anomaly) instead of NaN, which would trip the fail-loud no-NaN
    # guard. Blue-chips never flatten this long so the eps is inert for them.
    "bb_squeeze": (
        "(Bollinger($close, 20, 2).upper - Bollinger($close, 20, 2).lower) "
        "/ (Mean(Bollinger($close, 20, 2).upper - Bollinger($close, 20, 2).lower, 50) + 1e-12) - 1"
    ),
    "vol_percentile_60": "TsRank(Std(Pct($close, 1), 10), 60) / 60",
    # --- divergence (rolling corr of price vs oscillator; < 0 = bearish divergence,
    # price pushing higher while momentum fades — a reversal tell the model lacked) ---
    "rsi_div_20": "Corr($close, RSI($close, 14), 20)",
    "macd_div_20": "Corr($close, MACD($close, 12, 26, 9).hist, 20)",
    # Clean DIRECTIONAL divergence (fixes rsi_div_20's flaws: Corr is ±inf on flat-RSI
    # windows, stuck ~+1 92% of the time since RSI∝price, and non-directional). Rank-gap
    # in [-1,1]: POSITIVE = price at a 20d-high while the oscillator is NOT (bearish
    # divergence = top); NEGATIVE = bullish divergence (bottom). Bounded, no inf, signed.
    "div_rank_rsi": "(TsRank($close, 20) - TsRank(RSI($close, 14), 20)) / 20.0",
    "div_rank_macd": "(TsRank($close, 20) - TsRank(MACD($close, 12, 26, 9).hist, 20)) / 20.0",
    # MACD-histogram MOMENTUM (acceleration): change of hist over 2/5/10 bars, /close for
    # scale-free pooling. NEGATIVE = hist falling = momentum decelerating (weakening top
    # tell); POSITIVE = strengthening. The level (macd_hist) is gain-148-low; the CHANGE
    # is the user's "hist tăng/giảm so 2-5-10 phiên" idea — a directional rollover signal.
    "macd_hist_chg_1": "Delta(MACD($close, 12, 26, 9).hist, 1) / $close",
    "macd_hist_chg_2": "Delta(MACD($close, 12, 26, 9).hist, 2) / $close",
    "macd_hist_chg_3": "Delta(MACD($close, 12, 26, 9).hist, 3) / $close",
    "macd_hist_chg_5": "Delta(MACD($close, 12, 26, 9).hist, 5) / $close",
    "macd_hist_chg_10": "Delta(MACD($close, 12, 26, 9).hist, 10) / $close",
    # INDICATOR DYNAMICS (user 2026-06-20): Williams %R + its velocity; MA-spread (ribbon) WIDTH and its
    # EXPANSION/CONTRACTION rate ('suy giãn ra/thu lại'); longer-horizon RSI velocity. The point level
    # the head already sees; these give the multi-day CHANGE/expansion the user asked about.
    "williams_r_14": "(Max($high, 14) - $close) / (Max($high, 14) - Min($low, 14) + (Max($high, 14) - Min($low, 14) <= 0) * 1.0)",
    "williams_r_slope5": "#williams_r_14 - Ref(#williams_r_14, 5)",
    "ma_spread_10_50": "(Mean($close, 10) - Mean($close, 50)) / $close",
    "ma_spread_expand5": "Delta(#ma_spread_10_50, 5)",
    "ma_spread_20_100": "(Mean($close, 20) - Mean($close, 100)) / $close",
    "rsi_14_slope10": "#rsi_14 - Ref(#rsi_14, 10)",
    # RSI-MA (smoothed RSI = signal line) + RSI vs its MA (momentum-of-RSI crossing) — user
    # 2026-06-20 "RSI, MA RSI". EMA ribbon (faster than the SMA ma_spread) + its expansion.
    "rsi_ma_5": "Mean(#rsi_14, 5)",
    "rsi_vs_ma5": "#rsi_14 - Mean(#rsi_14, 5)",
    "ema_spread_8_21": "(EMA($close, 8) - EMA($close, 21)) / $close",
    "ema_spread_expand5": "Delta(#ema_spread_8_21, 5)",
    # --- TEMPORAL / SHAPE features (user 2026-06-18): a point-in-time slope can't tell a forming
    # TOP (slope seq 5,3,2,1,0) from a forming BOTTOM (-5,-3,-2,-1,0) — same endpoint, opposite SHAPE.
    # The eye reads the lagged SEQUENCE + curvature (acceleration) + relative state, not one value.
    # Give the tree the lagged slope sequence (Ref of the 1-bar hist slope), the 2nd difference
    # (ACCELERATION = curvature, the key top-vs-bottom discriminator when slope endpoints match), and
    # percentile-normalized level/slope (relative state beats absolute across stocks). All causal.
    "macd_hist_slope_l1": "Ref(#macd_hist_chg_1, 1)",
    "macd_hist_slope_l2": "Ref(#macd_hist_chg_1, 2)",
    "macd_hist_slope_l3": "Ref(#macd_hist_chg_1, 3)",
    "macd_hist_slope_l4": "Ref(#macd_hist_chg_1, 4)",
    "macd_hist_slope_l5": "Ref(#macd_hist_chg_1, 5)",
    "macd_hist_accel": "Delta(#macd_hist_chg_1, 1)",
    "macd_hist_accel_3": "Delta(#macd_hist_chg_1, 3)",
    "macd_hist_pctile_252": "TsRank(MACD($close, 12, 26, 9).hist, 252) / 252",
    "macd_hist_slope_pctile_252": "TsRank(#macd_hist_chg_1, 252) / 252",
    # --- multi-horizon range position (3-month / 6-month high & low) ---
    "dist_63d_high": "$close / Max($close, 63) - 1",
    "dist_63d_low": "$close / Min($close, 63) - 1",
    "dist_126d_high": "$close / Max($close, 126) - 1",
    "dist_126d_low": "$close / Min($close, 126) - 1",
    # --- candle streak (count of up-days in window; green-streak / exhaustion proxy) ---
    "up_days_5": "Sum((Delta($close, 1) > 0) * 1.0, 5)",
    "up_days_10": "Sum((Delta($close, 1) > 0) * 1.0, 10)",
    # --- DOWN-leg persistence (mirror of up_days): count of down-closes in a window = how
    #     sustained the current decline is (the "how long has it been falling" structural axis
    #     for the exit head, distinct from the symmetric vol-magnitude features). ---
    "down_days_5": "Sum((Delta($close, 1) < 0) * 1.0, 5)",
    "down_days_10": "Sum((Delta($close, 1) < 0) * 1.0, 10)",
    # --- distribution / accumulation days (volume-confirmed pressure over a window — the
    #     user's "số ngày phân phối": a down day on HIGHER volume = institutional distribution;
    #     accum_day = up day on rising volume). Count over 25 sessions (IBD convention).
    #     dist_day_vol20 uses volume-vs-20d-avg instead of vs-prior for a sustained-heavy-
    #     selling read. Causal (Ref/Mean of past bars). Down >=0.2% threshold avoids flat-close noise. ---
    "dist_day_25": "Sum( (($close <= Ref($close, 1) * 0.998) * 1.0) * (($volume > Ref($volume, 1)) * 1.0), 25)",
    "dist_day_vol20_25": "Sum( (($close < Ref($close, 1)) * 1.0) * (($volume > Mean($volume, 20)) * 1.0), 25)",
    "accum_day_25": "Sum( (($close >= Ref($close, 1) * 1.002) * 1.0) * (($volume > Ref($volume, 1)) * 1.0), 25)",
    # --- VOLUME-REGION BALANCE (the SEQUENCE/region read the per-bar features lack): 20-bar
    #     up-volume / down-volume ratio + accum-minus-dist candle balance. Validated orthogonal to
    #     the 5 entry heads' SCORES (resIC vs winner +0.19, 7/7 yrs positive) — the head TARGETS
    #     don't reward up/down volume-region quality. Feeds a 6th winner-meta-label entry head. ---
    "updown_vol_20": "Sum( ($close > Ref($close, 1)) * $volume, 20) / (Sum( ($close <= Ref($close, 1)) * $volume, 20) + (Sum( ($close <= Ref($close, 1)) * $volume, 20) <= 0) * 1.0)",
    "ad_balance_20": "Sum( ($close >= Ref($close, 1) * 1.002) * 1.0, 20) - Sum( ($close <= Ref($close, 1) * 0.998) * 1.0, 20)",
    # --- VOLUME-AT-PRICE / VWAP-distance (shape_probe gate: genuinely NEW vs the catalog's accum-day
    #     COUNTS — these read WHERE the money traded vs current price, the accumulation-base read).
    #     dist_vwap_20 was the strongest single entry feature found: IC_pnl +0.195 (> score3 +0.138),
    #     IC_mfe +0.296; close ABOVE the 20d volume-weighted avg = volume accumulated BELOW = supported.
    #     net_acc_vol_30 = net signed-volume fraction (smart-accumulation slope, IC_pnl +0.139, IC_mae
    #     +0.094 — predicts shallower drawdown, the dimension the pullback crutch owns). Causal. ---
    "vwap_20": "Sum($close * $volume, 20) / (Sum($volume, 20) + (Sum($volume, 20) <= 0) * 1.0)",
    "vwap_40": "Sum($close * $volume, 40) / (Sum($volume, 40) + (Sum($volume, 40) <= 0) * 1.0)",
    "dist_vwap_20": "($close - #vwap_20) / (#vwap_20 + (#vwap_20 <= 0) * 1.0)",
    "dist_vwap_40": "($close - #vwap_40) / (#vwap_40 + (#vwap_40 <= 0) * 1.0)",
    "net_acc_vol_30": "Sum( (($close > $open) * 1.0 - ($close < $open) * 1.0) * $volume, 30) / (Sum($volume, 30) + (Sum($volume, 30) <= 0) * 1.0)",
    # CROSSOVERS (user proposal 2026-06-19, "các giao cắt model có biết được không"): give the head
    # the MA/price/volume/MACD cross STATE explicitly (Sign of the gap = above/below = which side of the
    # cross). The model had ratios (sma_20_ratio) but not the discrete cross state across MA pairs.
    "ma_cross_20_50": "Sign(Mean($close, 20) - Mean($close, 50))",
    "px_cross_ma20": "Sign($close - Mean($close, 20))",
    "vol_breakout_20": "$volume / (Mean($volume, 20) + (Mean($volume, 20) <= 0) * 1.0)",
    "macd_cross": "Sign(MACD($close, 12, 26, 9).line - MACD($close, 12, 26, 9).signal)",
    # --- the user's "nến giảm mạnh có khối lượng lớn" (heavy-volume selling bar): a big red
    #     close (>=3% down) on volume >=1.5x its 20d avg = a distribution/climax bar; the
    #     count over 10 sessions = recent heavy-selling pressure; ratio_5 = summed down-day
    #     volume intensity over 5 bars. Causal (Ref/Mean of past bars). DIRECTIONAL (top-side). ---
    "down_vol_spike": "(($close < Ref($close, 1) * 0.97) * 1.0) * (($volume > Mean($volume, 20) * 1.5) * 1.0)",
    "down_vol_count_10": "Sum( (($close < Ref($close, 1) * 0.98) * 1.0) * (($volume > Mean($volume, 20) * 1.3) * 1.0), 10)",
    "down_vol_intensity_5": "Sum( ((Delta($close, 1) < 0) * 1.0) * #volume_ratio_20, 5)",
    # --- 2-candle reversal patterns (the user's "cặp nến đảo chiều"): bearish engulfing = a
    #     prior up-body fully engulfed by a current down-body (top reversal); bullish mirror
    #     (bottom reversal). 1.0 on the pattern bar, else 0. Causal (Ref of the prior bar). ---
    "bear_engulf": "((Ref($close, 1) > Ref($open, 1)) * 1.0) * (($close < $open) * 1.0) * (($open >= Ref($close, 1)) * 1.0) * (($close <= Ref($open, 1)) * 1.0)",
    "bull_engulf": "((Ref($close, 1) < Ref($open, 1)) * 1.0) * (($close > $open) * 1.0) * (($open <= Ref($close, 1)) * 1.0) * (($close >= Ref($open, 1)) * 1.0)",
    # --- indecision cluster (the user's "giá dao động lưỡng lự"): count of small-body (doji)
    #     bars over 10 sessions — a choppy/indecisive stretch. Reuses body_ratio (<=0.1). ---
    "doji_count_10": "Sum((#body_ratio <= 0.1) * 1.0, 10)",
    # --- breakout / accumulation-break (close making a new N-bar high = breaking the
    # prior range; pairs with bb_squeeze for "breakout from accumulation") + a longer
    # volume-explosion ratio (today's volume vs its 60-bar base) ---
    "breakout_20": "($close >= Max($close, 20)) * 1.0",
    "breakout_60": "($close >= Max($close, 60)) * 1.0",
    "volume_ratio_60": "$volume / (Mean($volume, 60) + (Mean($volume, 60) <= 0) * 1.0)",
    # --- Heiken-Ashi state (smoothed reversal candles; recursive HA_open, causal) ---
    "ha_color": "HeikenAshi($open, $high, $low, $close).color",
    "ha_body": "HeikenAshi($open, $high, $low, $close).body",
    "ha_trend": "HeikenAshi($open, $high, $low, $close).trend",
    # --- liquidity (leading_v3 Group D base) ---
    "volume_20d_avg": "Mean($volume, 20)",
    "price_level": "($close > 50000) * 1.0",
    "volume_stability": "Std($volume, 20) / Mean($volume, 20)",
    # --- cross-sectional rank (leading_v3 Group A) ---
    "momentum_rank": "CSRank(#ret_20d)",
    # DERIVATIVE of the cross-sectional rank = leadership rotation (10-bar change of this stock's
    # 20d-momentum rank vs the universe). The LEVEL (momentum_rank) is in the exit head; the TREND
    # (the project's strongest measured per-symbol signal, IC_winner +0.17 = accumulation/distribution
    # rotation) was never a model feature. Causal: CSRank per-date (no future) then per-symbol Delta.
    "cs_rank_trend": "Delta(#momentum_rank, 10)",
    # RS DYNAMICS (user 2026-06-20): the rise/fall, EMA, and multi-period EXPANSION of relative
    # strength. RS at 5/20/60d horizons; rs_spread_5_60 = short-vs-long RS ('co giãn RS nhiều chu
    # kỳ' = leadership accelerating when short RS > long RS); RS slope at 5/20 + acceleration;
    # EMA of RS (smoothed) + RS vs its EMA (RS crossing its own signal).
    "rs_rank_5": "CSRank(#ret_5d)",
    "rs_rank_60": "CSRank($close / Ref($close, 60) - 1)",
    "rs_spread_5_60": "#rs_rank_5 - #rs_rank_60",
    "rs_spread_5_20": "#rs_rank_5 - #momentum_rank",
    "rs_trend_5": "Delta(#momentum_rank, 5)",
    "rs_trend_20": "Delta(#momentum_rank, 20)",
    "rs_accel": "#cs_rank_trend - Ref(#cs_rank_trend, 5)",
    "rs_ema5": "EMA(#momentum_rank, 5)",
    "rs_vs_ema5": "#momentum_rank - EMA(#momentum_rank, 5)",
    "volatility_rank": "CSRank(#realized_vol_10)",
    "volume_rank": "CSRank($volume)",
    "rsi_rank": "CSRank(#rsi_14)",
    "price_strength_rank": "CSRank(#sma_20_ratio)",
    # CROSS-SECTIONAL oversold ranks (2026-07-11, mean-rev-as-feature): monetize the decorrelated
    # mean-rev source as a CROSS-SECTIONAL rank the entry ML can use in ONE book (a separate mean-rev
    # sleeve failed on slot-displacement). Most-below-MA20 / biggest-recent-drop ranks highest.
    # Cross-sectional -> survives the recombine z-scoring.
    "osold_rank_20": "CSRank(0 - #sma_20_ratio)",
    "osold_rank_ret5": "CSRank(0 - #ret_5d)",
    # --- low-vol x near-52w-high composites (feature_ic_research winners: cross-sectional
    #     rank-IC20 0.076-0.077 full / 0.098-0.110 oos22 — the strongest VN fwd-return
    #     signal found, edge GREW in 2022+; NO market-index dependency. Feeds the
    #     continuation entry so it buys low-volatility names near their 52w high.) ---
    "lowvol_rank_60": "CSRank(0 - Std(Pct($close, 1), 60))",
    "lowvol_park_rank": "CSRank(0 - Mean(Log($high / $low) * Log($high / $low), 20))",
    "nearhigh_rank": "CSRank(#dist_52w_high)",
    "comp_nh_lv": "CSRank(#dist_52w_high) * CSRank(0 - Std(Pct($close, 1), 60))",
    "comp_park_nh": "CSRank(0 - Mean(Log($high / $low) * Log($high / $low), 20)) * CSRank(#dist_52w_high)",
    # R42 additions: the higher-OOS-IC composites from feature_ic_research (oos22 IC20
    # 0.094-0.110, GREATER than comp_nh_lv) not yet wired into any entry set. The uptrend-
    # gated low-vol legs (× market_trend) had the strongest 2022+ edge; comp_lv_nh_rev has
    # the highest short-horizon ic5 (0.066) → earlier signal → more trades.
    "nearhigh126_rank": "CSRank(#dist_126d_high)",
    "comp_lowvol_uptrend": "CSRank(0 - Std(Pct($close, 1), 60)) * #market_trend",
    "comp_park_uptrend": "CSRank(0 - Mean(Log($high / $low) * Log($high / $low), 20)) * #market_trend",
    "comp_nh_x_lv126": "CSRank(#dist_126d_high) * CSRank(0 - Std(Pct($close, 1), 60))",
    "comp_lv_nh_rev": (
        "CSRank(0 - Std(Pct($close, 1), 60)) + CSRank(#dist_52w_high) + CSRank(0 - #ret_5d)"
    ),
    # R47: the remaining IC-research composites that add a 12-1 MOMENTUM dimension to the
    # low-vol × near-high core (oos22 IC20 ~0.098). comp_tri_and = the multiplicative triple
    # (all three high), comp_v2h1 = vol-weighted additive (vol is the stronger leg).
    "comp_tri_and": (
        "CSRank(#dist_52w_high) * CSRank(0 - Std(Pct($close, 1), 60)) "
        "* (0.5 + 0.5 * CSRank(Ref($close, 21) / Ref($close, 252) - 1))"
    ),
    "comp_v2h1": "2.0 * CSRank(0 - Std(Pct($close, 1), 60)) + CSRank(#dist_52w_high)",
    # --- sector-relative (leading_v3 Group B) ---
    "return_vs_sector": "#ret_20d - CSGroupMedian(#ret_20d, by=$sector)",
    "momentum_vs_sector": "#macd_hist - CSGroupMedian(#macd_hist, by=$sector)",
    "volume_vs_sector": "$volume / (CSGroupMedian($volume, by=$sector) + 1e-10)",
    "volatility_vs_sector": "#realized_vol_10 - CSGroupMedian(#realized_vol_10, by=$sector)",
    "strength_vs_sector": "CSGroupRank(#sma_20_ratio, by=$sector)",
    # SECTOR-RS DYNAMICS (2026-06-20): is the stock GAINING/LOSING strength within its sector
    # (sector rotation INTO this name) — the change of the sector-relative signals over time.
    "return_vs_sector_slope5": "Delta(#return_vs_sector, 5)",
    "strength_vs_sector_slope5": "Delta(#strength_vs_sector, 5)",
    "beta_to_sector": "CSGroupZScore(#ret_1d, by=$sector)",
    # --- market regime (leading_v3 Group C) ---
    "market_trend": "($market_close > Mean($market_close, 200)) * 1.0",
    # Faster regime gate than market_trend (MA200). shakeout_vs_top forensic (2026-07-11):
    # mkt>MA50 is the single strongest shakeout-vs-top discriminator (AUC 0.78) — a 'signal'
    # exit in a bull tape (mkt>MA50) is usually a shakeout that recovers; below MA50 it is a
    # real top. MA200 is too slow to catch the regime flip that decides which.
    "market_trend_50": "($market_close > Mean($market_close, 50)) * 1.0",
    "market_volatility_regime": (
        "(Std(Pct($market_close, 1), 20) "
        "> Quantile(Std(Pct($market_close, 1), 20), 200, 0.9)) * 1.0"
    ),
    "regime_interaction_momentum": "#macd_hist * #market_trend",
    "regime_interaction_strength": "#sma_20_ratio * (1 - #market_volatility_regime)",
    # --- liquidity rank (leading_v3 Group D, derived) ---
    "volume_rank_20d": "CSRank(#volume_20d_avg)",
    # --- zigzag swing structure (causal, leakage-safe; 6% reversal / min 3-bar leg) ---
    # Confirmed group (stable past structure) + developing group (current progress).
    # zz_n_pivots is a maturity flag (0..3): treat other zz_* as placeholders when low.
    "zz_last_dir": "ZigZag($close, 0.06, 3).last_dir",
    "zz_last_leg_return": "ZigZag($close, 0.06, 3).last_leg_return",
    "zz_last_leg_dur": "ZigZag($close, 0.06, 3).last_leg_dur",
    "zz_prev_leg_return": "ZigZag($close, 0.06, 3).prev_leg_return",
    "zz_prev_leg_dur": "ZigZag($close, 0.06, 3).prev_leg_dur",
    "zz_bars_since_pivot": "ZigZag($close, 0.06, 3).bars_since_pivot",
    "zz_return_since_pivot": "ZigZag($close, 0.06, 3).return_since_pivot",
    "zz_progress_to_deviation": "ZigZag($close, 0.06, 3).progress_to_deviation",
    "zz_dist_to_confirm": "ZigZag($close, 0.06, 3).dist_to_confirm",
    "zz_price_pos_in_swing": "ZigZag($close, 0.06, 3).price_pos_in_swing",
    "zz_max_adverse_since_pivot": "ZigZag($close, 0.06, 3).max_adverse_since_pivot",
    "zz_n_pivots": "ZigZag($close, 0.06, 3).n_pivots",
    # --- reversal-confirmation (V-bottom vs falling-knife; validated 2026-06-08) ---
    # Causal price/volume confirmations that a deep dip has actually turned. Real
    # V-bottoms show a reversal AFTER the fall; falling knives keep dropping with none.
    # >=2 of {up2, hi_low, reclaim5, vol_thrust} fired separates the winners (fwd10
    # +1.7~2.5%, win 59-60%) from knives (+0.47%). Used by the regime-conditional
    # reversal entry (buy cheap only when market itself is washed out, market_mom_60<-0.05).
    "rev_up2": "($close > Ref($close, 1)) * (Ref($close, 1) > Ref($close, 2))",
    "rev_hi_low": "($low > Ref($low, 1)) * 1.0",
    "rev_reclaim5": "($close >= Mean($close, 5)) * (Ref($close, 1) < Ref(Mean($close, 5), 1))",
    "rev_vol_thrust": "($close > Ref($close, 1)) * ($volume > 1.3 * Mean($volume, 20))",
    "reversal_nconf": "#rev_up2 + #rev_hi_low + #rev_reclaim5 + #rev_vol_thrust",
    "dist_20d_high": "$close / Quantile($close, 20, 1.0) - 1",
    "market_mom_60": "$market_close / Ref($market_close, 60) - 1",
    # === leading_ew enrichment (early_wave recipe parity) ============================
    # Faithful ports of the "leading" feature set missing from leading_v2. All causal
    # (rolling/Ref of past+current bars). Denominator guards follow the catalog idiom:
    # a zero-range/zero-volume window keeps the denominator 1 (numerator is also 0) so
    # the result is 0 instead of a fail-loud NaN. Helpers (typical_price/clv/vwap_20)
    # are building blocks referenced via #ref and are NOT exposed in any set.
    "typical_price": "($high + $low + $close) / 3",
    "clv": "(($close - $low) - ($high - $close)) / (($high - $low) + ($high <= $low) * 1.0)",
    "vwap_20": "Sum(#typical_price * $volume, 20) / (Sum($volume, 20) + (Sum($volume, 20) <= 0) * 1.0)",
    # momentum (leading §2.2): Stochastic %K/%D, Williams %R
    "stoch_k": (
        "100 * ($close - Min($low, 14)) "
        "/ ((Max($high, 14) - Min($low, 14)) + ((Max($high, 14) - Min($low, 14)) <= 0) * 1.0)"
    ),
    "stoch_d": "Mean(#stoch_k, 3)",
    # MULTI-PERIOD / RIBBON / SMOOTHED features (user 2026-06-18): the knife cohort is causally near-
    # identical to winners on SINGLE features, but multi-period SPREADS separate better (knife-vs-winner
    # sep: rsi_fastslow +0.213, ribbon +0.189, rsi_smooth_slope +0.181, vs single rsi/ext ~0). The eye
    # reads fast-vs-slow momentum + EMA fanning, not one value. All causal.
    "rsi_21": "RSI($close, 21)",
    "rsi_50": "RSI($close, 50)",
    "rsi_fastslow": "RSI($close, 7) - RSI($close, 21)",  # multi-period RSI spread (accel/decel)
    "rsi_smooth_slope": "Delta(EMA(RSI($close, 14), 3), 3)",  # smoothed RSI slope (less whipsaw)
    "ema_5_ratio": "$close / EMA($close, 5) - 1",
    "ema_20_ratio": "$close / EMA($close, 20) - 1",
    "ema_50_ratio": "$close / EMA($close, 50) - 1",
    "ema_stretch_5_50": "EMA($close, 5) / EMA($close, 50) - 1",  # short-vs-long EMA distance
    "ema_ribbon_width": "(EMA($close, 5) - EMA($close, 50)) / $close",  # EMA-ribbon fan width
    "ema_ribbon_expand": "Delta(#ema_ribbon_width, 5)",  # ribbon expanding(+)/contracting(-)
    "macd_hist_smooth_slope": "Delta(EMA(MACD($close, 12, 26, 9).hist, 3), 3)",  # smoothed hist slope
    "williams_r": (
        "0 - 100 * (Max($high, 14) - $close) "
        "/ ((Max($high, 14) - Min($low, 14)) + ((Max($high, 14) - Min($low, 14)) <= 0) * 1.0)"
    ),
    # trend (leading §2.3): CCI, Aroon
    "cci_20": (
        "(#typical_price - Mean(#typical_price, 20)) "
        "/ (0.015 * Mean(Abs(#typical_price - Mean(#typical_price, 20)), 20) "
        "+ (Mean(Abs(#typical_price - Mean(#typical_price, 20)), 20) <= 0) * 1.0)"
    ),
    "aroon_up": "Aroon($high, $low, 25).up",
    "aroon_down": "Aroon($high, $low, 25).down",
    # volume advanced (leading §2.5): CMF, VWAP ratio
    "cmf_20": "Sum(#clv * $volume, 20) / (Sum($volume, 20) + (Sum($volume, 20) <= 0) * 1.0)",
    "vwap_ratio": "$close / (#vwap_20 + (#vwap_20 <= 0) * $close) - 1",
    # leading signals (leading §2.6): the entry-cascade core
    "vol_surge_ratio": "Mean($volume, 5) / (Mean($volume, 20) + (Mean($volume, 20) <= 0) * 1.0)",
    "pv_divergence": "#volume_ratio_20 - Abs(#ret_5d)",
    "atr_contraction": (
        "ATR($high, $low, $close, 5) "
        "/ (ATR($high, $low, $close, 20) + (ATR($high, $low, $close, 20) <= 0) * 1.0)"
    ),
    "bb_width_percentile": "TsRank(#bb_width_20, 60) / 60",
    "close_position_in_range": "($close - $low) / (($high - $low) + ($high <= $low) * 1.0)",
    "close_pos_ma5": "Mean(#close_position_in_range, 5)",
    "obv_price_divergence": (
        "#obv_slope_10 - Delta($close, 10) / (Mean($close, 10) + (Mean($close, 10) <= 0) * 1.0)"
    ),
    "dist_to_resistance": "(Max($high, 20) - $close) / $close",
    "dist_to_support": "($close - Min($low, 20)) / $close",
    "range_position_20d": (
        "($close - Min($low, 20)) "
        "/ ((Max($high, 20) - Min($low, 20)) + ((Max($high, 20) - Min($low, 20)) <= 0) * 1.0)"
    ),
    "higher_lows_count": "Sum(($low > Ref($low, 1)) * 1.0, 4)",
    "consolidation_score": "Sum((#high_low_pct < 0.02) * 1.0, 10)",
    "rsi_slope_5d": "Delta(#rsi_14, 5)",
    "breakout_setup_score": (
        "(#vol_surge_ratio > 1.2) * 1.0 + (#bb_width_percentile < 0.3) * 1.0 "
        "+ (#dist_to_resistance < 0.02) * 1.0 + (#close_pos_ma5 > 0.6) * 1.0 "
        "+ (#higher_lows_count >= 3) * 1.0"
    ),
    # === DERIVED-SIGNAL HUNT (2026-06-17, no-new-data lead): genuinely-NEW daily-OHLCV families
    #     NOT yet in the catalog — trend-QUALITY (Kaufman efficiency ratio), return-regime
    #     (lag-1 autocorrelation), compression/expansion microstructure (NR7 / inside-bar / gap),
    #     and an ATR-channel (Keltner) orthogonal to the std-based Bollinger. Theory: the clean-trend
    #     runners the pullback drops should score HIGH on efficiency ratio (net move per unit of
    #     path noise) — a trend-QUALITY axis the level/momentum features don't carry. Causal. ===
    # Kaufman Efficiency Ratio: |net move over N| / sum of |1-bar moves over N|. ->1 = clean trend,
    # ->0 = chop. Denom guard for a frozen (no-movement) window.
    "efficiency_ratio_10": "Abs(Delta($close, 10)) / (Sum(Abs(Delta($close, 1)), 10) + (Sum(Abs(Delta($close, 1)), 10) <= 0) * 1.0)",
    "efficiency_ratio_20": "Abs(Delta($close, 20)) / (Sum(Abs(Delta($close, 1)), 20) + (Sum(Abs(Delta($close, 1)), 20) <= 0) * 1.0)",
    # NR7: current bar is the narrowest range of the last 7 (compression -> impending expansion).
    "nr7": "(($high - $low) <= Min($high - $low, 7)) * 1.0",
    # Inside bar: range fully inside the prior bar (contraction / coil before a move).
    "inside_bar": "(($high < Ref($high, 1)) * 1.0) * (($low > Ref($low, 1)) * 1.0)",
    # Overnight gap (open vs prior close), signed magnitude — distinct from close_to_open (intraday).
    "gap_oc": "$open / Ref($close, 1) - 1",
    # Keltner channel position: (close - EMA20) / (2*ATR20). ATR-based, orthogonal to std-based bb_pct.
    "keltner_pct_20": "($close - EMA($close, 20)) / (2.0 * ATR($high, $low, $close, 20) + (ATR($high, $low, $close, 20) <= 0) * 1.0)",
    # === MULTI-PERIOD SPREAD / SMOOTHED-OSCILLATOR family (user 2026-06-18): "co giãn các đường EMA/RSI
    #     nhiều chu kỳ" + smoothed variants of noisy oscillators + money-flow change. EMA-ribbon WIDTH
    #     (EMA5 vs EMA50) = trend expansion; its CHANGE = contraction(<0, trend stalling -> a top tell
    #     distinct from sideways consolidation). RSI fast-slow SPREAD (7 vs 14) + change = momentum
    #     widening/narrowing across periods. Smoothed MACD-hist (3-bar mean) de-noises the whippy hist.
    #     Money-flow CHANGE (mfi/cmf delta) = accumulation->distribution shift. All causal. ===
    "ema_ribbon_w": "(EMA($close, 5) - EMA($close, 50)) / $close",
    "ema_ribbon_chg5": "Delta(#ema_ribbon_w, 5)",
    "ema_ribbon_chg10": "Delta(#ema_ribbon_w, 10)",
    "rsi_spread_7_14": "RSI($close, 7) - RSI($close, 14)",
    "rsi_spread_chg3": "Delta(#rsi_spread_7_14, 3)",
    "macd_hist_sma3": "Mean(MACD($close, 12, 26, 9).hist, 3)",
    "mfi_chg5": "Delta(#mfi_14, 5)",
    "cmf_chg5": "Delta(#cmf_20, 5)",
    # === PV-CHANNEL probe (2026-07-09, analysis/serving_blindspot/ohlcv OHLCV_VIRGIN_MAP.md):
    #     the ONE orthogonal survivor of the 25-candidate OHLCV screen — continuous price-volume
    #     co-movement corr(ret1, dlogV, 10): resid-IC +0.041 on the champion exit label (5/5 folds,
    #     ~5 sigma over null). Zero-volume bars are NaN-ed via $volume/($volume>0) (0/0 = NaN) so
    #     dlogV stays NaN there — same semantics as the screening harness (screen_ic.py). Causal:
    #     rolling windows end at t. ===
    #     Fillna 0 = neutral "no co-movement information": Corr is undefined (NaN) on dead
    #     stretches (constant price or NaN-guarded zero-volume windows) and the pipeline's
    #     fail-loud policy forbids interior NaN — the screening harness simply skipped those
    #     rows per-date, so 0 is the faithful model-side equivalent.
    "pv_corr_10": "Fillna(Corr(Pct($close, 1), Delta(Log($volume / ($volume > 0)), 1), 10), 0.0)",
    # True range helper (elementwise max of the 3 TR legs; NaN legs skipped like pandas max axis=1).
    "tr_1": "Max(Max($high - $low, Abs($high - Ref($close, 1))), Abs($low - Ref($close, 1)))",
    # NR7-style compression POSITION: TR / max(TR, 7). 1 = widest bar of the week, ->0 = narrow-range
    # coil. Denominator guard follows the catalog idiom (degenerate all-zero week -> 0, not NaN).
    "nr_pos_7": "#tr_1 / (Max(#tr_1, 7) + (Max(#tr_1, 7) <= 0) * 1.0)",
}


_LEADING_V2 = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "close_to_open",
    "sma_5_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "ema_10_ratio",
    "sma5_cross_sma20",
    "rsi_14",
    "rsi_7",
    "macd_line",
    "macd_hist",
    "roc_10",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "atr_14_ratio",
    "bb_width_20",
    "bb_pct_20",
    "realized_vol_10",
    "volume_ratio_5",
    "volume_ratio_20",
    "obv_slope_10",
    "mfi_14",
    "dist_52w_high",
    "dist_52w_low",
    "high_low_pct_5d",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "body_ratio",
    "high_low_pct",
    "atr_regime",
    "bb_squeeze",
    "vol_percentile_60",
    "is_limit_lock",
]

# The robust exit_vol_dist base (2369, nopullback line best): vol/dist magnitude + market context +
# distribution-day volume. Supply-axis exit sets extend this (the one axis not realizability-walled).
_EXIT_VOL_DIST = [
    "atr_14_ratio",
    "realized_vol_10",
    "vol_percentile_60",
    "bb_width_20",
    "volatility_rank",
    "high_low_pct_5d",
    "ma5_accel",
    "dist_63d_high",
    "dist_52w_high",
    "sma_20_ratio",
    "bb_pct_20",
    "market_volatility_regime",
    "market_trend",
    "momentum_rank",
    "dist_day_25",
    "dist_day_vol20_25",
]

_LEADING_V3_EXTRA = [
    "momentum_rank",
    "volatility_rank",
    "volume_rank",
    "rsi_rank",
    "price_strength_rank",
    "return_vs_sector",
    "momentum_vs_sector",
    "volume_vs_sector",
    "volatility_vs_sector",
    "strength_vs_sector",
    "beta_to_sector",
    "market_trend",
    "market_volatility_regime",
    "regime_interaction_momentum",
    "regime_interaction_strength",
    "volume_20d_avg",
    "volume_rank_20d",
    "price_level",
    "volume_stability",
]

_BASIC_V1 = [
    "ret_1d",
    "ret_5d",
    "sma_5_ratio",
    "sma_20_ratio",
    "rsi_14",
    "volume_ratio_20",
    "high_low_pct",
    "atr_14_ratio",
]

# The "leading" indicators present in the early_wave guide but absent from leading_v2:
# Stochastic/Williams momentum, CCI/Aroon trend, CMF/VWAP volume, and the §2.6 leading
# entry-cascade signals (vol surge, breakout setup, range position, support/resistance
# distance, higher-lows, OBV divergence, bb-width percentile).
_LEADING_EW_EXTRA = [
    "stoch_k",
    "stoch_d",
    "williams_r",
    "cci_20",
    "aroon_up",
    "aroon_down",
    "cmf_20",
    "vwap_ratio",
    "vol_surge_ratio",
    "pv_divergence",
    "atr_contraction",
    "bb_width_percentile",
    "close_position_in_range",
    "close_pos_ma5",
    "obv_price_divergence",
    "dist_to_resistance",
    "dist_to_support",
    "range_position_20d",
    "higher_lows_count",
    "consolidation_score",
    "rsi_slope_5d",
    "breakout_setup_score",
]

_MA_SLOPE = ["ma5_slope", "ma10_slope", "ma20_slope", "ma5_accel", "ma_align"]

_ZZ_SWING = [
    "zz_last_dir",
    "zz_last_leg_return",
    "zz_last_leg_dur",
    "zz_prev_leg_return",
    "zz_prev_leg_dur",
    "zz_bars_since_pivot",
    "zz_return_since_pivot",
    "zz_progress_to_deviation",
    "zz_dist_to_confirm",
    "zz_price_pos_in_swing",
    "zz_max_adverse_since_pivot",
    "zz_n_pivots",
]

# Curated "reversal/context" extras tested on the zigzag champion (round-9): the
# genuinely-NEW price info not already in leading_zz — divergence, 3m/6m range
# position, up-day streaks. (Confirmed-pivot distance & swing position are already
# covered by zz_return_since_pivot / zz_price_pos_in_swing.)
_REVERSAL_EXTRA = [
    "rsi_div_20",
    "macd_div_20",
    "dist_63d_high",
    "dist_63d_low",
    "dist_126d_high",
    "dist_126d_low",
    "up_days_5",
    "up_days_10",
]

_HA = ["ha_color", "ha_body", "ha_trend"]

# Curated ENTRY set for "score the base-of-wave setups": volume explosion, breakout from
# accumulation, rising-MA trend, Heiken-Ashi green reversal, and zigzag swing context —
# the bullish-reversal signals the user wants the entry head to light up on.
_ENTRY_BASE = ["breakout_20", "breakout_60", "volume_ratio_60"]

# Exit peak/exhaustion core (the exit_peak_v1 content) — factored out so v2/v3 extend it.
_EXIT_PEAK_CORE = [
    "dist_52w_high",
    "sma_20_ratio",
    "sma_50_ratio",
    "rsi_14",
    "rsi_7",
    "bb_pct_20",
    "macd_hist",
    "roc_10",
    "ret_5d",
    "ret_10d",
    "atr_regime",
    "atr_14_ratio",
    "mfi_14",
    "volume_ratio_20",
    "obv_slope_10",
    *_MA_SLOPE,
    *_ZZ_SWING,
]
_EXIT_PEAK_V2_EXTRA = ["rsi_div_20", "macd_div_20", "dist_63d_high", "dist_126d_high", "up_days_10"]

# Zero/near-zero-gain features on the zigzag-peak-post exit target (LightGBM gain
# importance, measured 2026-06-04). Dropped by the exit_peak_lean ablation sets.
_EXIT_DEAD = {
    "ret_10d",
    "zz_n_pivots",
    "ha_color",
    "mfi_14",
    "macd_div_20",
    "obv_slope_10",
    "rsi_div_20",
    "zz_prev_leg_dur",
    "lower_wick_ratio",
    "sma_50_ratio",
}

# set name -> (description, ordered member feature names)
SETS: dict[str, tuple[str, list[str]]] = {
    "basic_v1": ("8 leakage-safe per-symbol features", _BASIC_V1),
    "entry_rev_regime": (
        "Regime-conditional reversal entry: confirm count, 20d-high distance, market 60d momentum",
        ["reversal_nconf", "dist_20d_high", "market_mom_60"],
    ),
    "entry_rev_quality": (
        "Reversal-confirm EARLY entry gated by champion quality (long-term uptrend): "
        "confirm count, 20d-high distance, SMA200 trend (dip-in-trend, not structural decliner)",
        ["reversal_nconf", "dist_20d_high", "sma_200_ratio"],
    ),
    "leading": ("V22 baseline per-symbol features (alias of leading_v2)", _LEADING_V2),
    "leading_v2": ("37 per-symbol features (price/momentum/trend/volatility)", _LEADING_V2),
    # leading_v2 + the full "leading" enrichment from the early_wave guide (59):
    # Stochastic/Williams/CCI/Aroon/CMF/VWAP + the §2.6 leading entry-cascade signals.
    "leading_ew": (
        "leading_v2 + early_wave 'leading' enrichment (stoch/williams/cci/aroon/cmf/vwap "
        "+ leading entry-cascade signals) (59)",
        _LEADING_V2 + _LEADING_EW_EXTRA,
    ),
    # leading_v2 + a long-term trend gate (close/SMA200). Lets the entry regressor
    # distinguish an oversold dip from a structural collapse (the falling-knife fix:
    # AAS 2022 fell -49% while leading_v2's oversold features kept signalling buy).
    "leading_v2_trend": (
        "leading_v2 + sma_200_ratio long-term trend gate (38)",
        _LEADING_V2 + ["sma_200_ratio"],
    ),
    "leading_v3": (
        "leading_v2 + cross-sectional rank + sector-relative + market regime (55)",
        _LEADING_V2 + _LEADING_V3_EXTRA,
    ),
    "leading_deriv": (
        "Derivative-optimized per-symbol features (no cross-sectional RS)",
        _LEADING_V2,
    ),
    "leading_v4": (
        "leading_v3 implementable subset (heikin-ashi/MTF blocks not yet ported)",
        _LEADING_V2 + _LEADING_V3_EXTRA,
    ),
    # Curated relative-strength + market-regime set: leading_v2 plus only the
    # train-clean cross-sectional/regime features. Excludes beta_to_sector (per-date
    # std collapses in thin sectors -> >5% NaN, trips the integrity gate) AND the
    # sector-relative features (return_vs_sector / strength_vs_sector), whose thin-
    # sector + short-history rows leave ~0.6% NaN that the fail-loud train gate
    # rejects. The 3 kept extras are NaN-free after warmup-trim.
    "rs_regime_v1": (
        "leading_v2 + train-clean cross-sectional momentum & market-regime features (39)",
        _LEADING_V2
        + [
            "momentum_rank",
            "market_trend",
            "market_volatility_regime",
        ],
    ),
    # leading_v2 + MA slopes + causal zigzag swing structure (54). For the dual-ML
    # zigzag experiment: gives the entry/exit regressors explicit swing-progress
    # context (esp. helps the exit model read deterioration in downtrends).
    "leading_zz": (
        "leading_v2 + MA slopes + causal zigzag swing features (54)",
        _LEADING_V2 + _MA_SLOPE + _ZZ_SWING,
    ),
    # leading_zz + reversal/context extras (divergence, 3m/6m range, up-day streaks).
    # For the zigzag dual-ML champion entry: tests whether the new price signals lift
    # per-trade selectivity (NOT expected to break the forward-IC ceiling — same info
    # class as price, per the R5/R7 findings).
    "leading_zz_plus": (
        "leading_zz + RSI/MACD divergence + 3m/6m range + up-day streaks (62)",
        _LEADING_V2 + _MA_SLOPE + _ZZ_SWING + _REVERSAL_EXTRA,
    ),
    # leading_zz + Heiken-Ashi reversal state (entry HA test).
    "leading_zz_ha": (
        "leading_zz + Heiken-Ashi color/body/trend (57)",
        _LEADING_V2 + _MA_SLOPE + _ZZ_SWING + _HA,
    ),
    # Base-of-wave entry set: leading_zz + HA + breakout/accumulation-break + vol-explosion.
    # Gives the entry head every bullish-reversal signal (volume bình nổ, vượt biên tích luỹ,
    # MA tăng, HA xanh đảo chiều, swing context) so triple-barrier can score them high.
    "entry_base_v1": (
        "leading_zz + HA + breakout(20/60) + volume_ratio_60 (60)",
        _LEADING_V2 + _MA_SLOPE + _ZZ_SWING + _HA + _ENTRY_BASE,
    ),
    # Minimal price-rule set for trend/dip rule-entry templates (close vs SMA200 +
    # RSI). Kept separate from leading_v2 so adding the SMA200 ratio does not change
    # any existing model's feature matrix.
    "rule_trend_v1": (
        "close-vs-SMA200 + RSI for trend/dip rule entries (2)",
        ["sma_200_ratio", "rsi_14"],
    ),
    # dip_in_trend entry: trend filter + Cutler-RSI oversold dip (both causal).
    "rule_dip_v1": (
        "close-vs-SMA200 + Cutler RSI for the dip-in-trend entry (2)",
        ["sma_200_ratio", "rsi_14_sma"],
    ),
    # Curated EXIT peak/exhaustion set for the dual-ML downside head (sumEX xrise arc).
    # leading_zz (full leading_v2 + slopes + swing) shifted the exit frontier outward, but
    # carries the bulk mean-reversion features (oversold/dip) that are noise for a SELL head.
    # This drops them and keeps only top-proximity, overbought, momentum/trend rollover,
    # volume-climax and zigzag swing-structure signals — the information a peak detector
    # actually needs. NO sma_200_ratio (it hurt the exit head in leading_v2_trend).
    "exit_peak_v1": (
        "Curated exit peak/exhaustion: top-proximity + overbought + rollover + swing (32)",
        _EXIT_PEAK_CORE,
    ),
    # exit_peak_v1 + top-relevant reversal extras: bearish divergence (price up while
    # RSI/MACD fade = classic top) + 3m/6m high-distance + up-day streak (exhaustion).
    "exit_peak_v2": (
        "exit_peak_v1 + RSI/MACD divergence + 3m/6m high-dist + up-day streak (37)",
        _EXIT_PEAK_CORE + _EXIT_PEAK_V2_EXTRA,
    ),
    # exit_peak_v2 + Heiken-Ashi reversal state (exit HA test, on top of the v2 winner).
    "exit_peak_v3": (
        "exit_peak_v2 + Heiken-Ashi color/body/trend (40)",
        _EXIT_PEAK_CORE + _EXIT_PEAK_V2_EXTRA + _HA,
    ),
    # exit_peak_v3 + distance-above-base (dist to 52w/126d LOW) — tests the user's
    # "đỉnh xa nền giá" (extended distribution top) idea with an absolute base-distance
    # measure, distinct from sma_200_ratio (which hurt the exit head). Lets the peak
    # detector tell an extended distribution top from a peak still near its base.
    "exit_peak_v4": (
        "exit_peak_v3 + distance-above-52w/126d low (extension from base) (42)",
        _EXIT_PEAK_CORE + _EXIT_PEAK_V2_EXTRA + _HA + ["dist_52w_low", "dist_126d_low"],
    ),
    # exit_peak_v3 + candle-exhaustion shape (upper/lower wick + body ratio) so the exit
    # head can see indecision / long-upper-wick rejection bars at a top (the 7-9/6 DPM
    # blind spot: small-body, long-wick warning candles the v3 set was blind to).
    "exit_peak_v5": (
        "exit_peak_v3 + candle exhaustion: upper/lower wick + body ratio (43)",
        _EXIT_PEAK_CORE
        + _EXIT_PEAK_V2_EXTRA
        + _HA
        + ["upper_wick_ratio", "lower_wick_ratio", "body_ratio"],
    ),
    # Entry set with the FAST same-bar direction signals (close_to_open, ret_1d) the
    # exit_peak set lacks — so an entry monotone +1 on them can suppress the dump-spike
    # (buy green/turn-up bars, not the red down-candle the dip-buyer fires on).
    "entry_turn_v1": (
        "exit_peak_v3 + close_to_open + ret_1d (fast same-bar direction) (42)",
        _EXIT_PEAK_CORE + _EXIT_PEAK_V2_EXTRA + _HA + ["close_to_open", "ret_1d"],
    ),
    # Ablation: exit_peak_v5 minus the 10 zero/near-zero-GAIN features the exit tree
    # never splits on (measured on the zigzag-peak-post target): macd_div_20/rsi_div_20
    # (divergence is DEAD here), mfi_14, ha_color, ret_10d, zz_n_pivots, obv_slope_10,
    # lower_wick_ratio, sma_50_ratio, zz_prev_leg_dur. With feature_fraction=0.6 each
    # split-subsample then samples useful features more often (de-noise).
    "exit_peak_lean": (
        "exit_peak_v5 minus 10 dead-gain features (33)",
        [
            c
            for c in (
                _EXIT_PEAK_CORE
                + _EXIT_PEAK_V2_EXTRA
                + _HA
                + ["upper_wick_ratio", "lower_wick_ratio", "body_ratio"]
            )
            if c not in _EXIT_DEAD
        ],
    ),
    # exit_peak_lean + ADX trend-strength & -DI directional (untested on the exit; a
    # weakening trend = falling ADX / rising -DI — info the dead divergence feats lacked).
    "exit_peak_lean_adx": (
        "exit_peak_lean + adx_14 + minus_di_14 (35)",
        [
            c
            for c in (
                _EXIT_PEAK_CORE
                + _EXIT_PEAK_V2_EXTRA
                + _HA
                + ["upper_wick_ratio", "lower_wick_ratio", "body_ratio"]
            )
            if c not in _EXIT_DEAD
        ]
        + ["adx_14", "minus_di_14"],
    ),
    # exit_peak_v5 + ADX/-DI, WITHOUT dropping anything (ablation showed dropping the
    # dead-gain feats HURT -3 via bagging, but adding adx/-DI helped). Add-only variant.
    "exit_peak_v6": (
        "exit_peak_v5 + adx_14 + minus_di_14 (45)",
        _EXIT_PEAK_CORE
        + _EXIT_PEAK_V2_EXTRA
        + _HA
        + ["upper_wick_ratio", "lower_wick_ratio", "body_ratio", "adx_14", "minus_di_14"],
    ),
    # Best ablation set (lean_adx) + CLEAN directional divergence (replaces the broken
    # Corr-based rsi_div/macd_div). Tests whether divergence, properly normalized &
    # condensed, finally carries signal the exit head will use.
    "exit_peak_div": (
        "exit_peak_lean_adx + div_rank_rsi + div_rank_macd (37)",
        [
            c
            for c in (
                _EXIT_PEAK_CORE
                + _EXIT_PEAK_V2_EXTRA
                + _HA
                + ["upper_wick_ratio", "lower_wick_ratio", "body_ratio"]
            )
            if c not in _EXIT_DEAD
        ]
        + ["adx_14", "minus_di_14", "div_rank_rsi", "div_rank_macd"],
    ),
    # exit_peak_div + the VOLATILITY features that exit_ic_research found are the strongest
    # forward-downside predictors (atr_14_ratio fwd_dd-IC20 0.30/risk 0.07 oos22, realized_vol
    # 0.29/0.056, bb_width 0.04) yet are MISSING from exit_peak_div (it had atr_14_ratio +
    # atr_regime + bb_pct_20 but NOT realized_vol_10 / bb_width_20 / vol_percentile_60). Tests
    # whether adding the absent high-IC vol legs sharpens the sell (downside-aware exit).
    "exit_peak_div_vol": (
        "exit_peak_div + realized_vol_10 + bb_width_20 + vol_percentile_60 (40)",
        [
            c
            for c in (
                _EXIT_PEAK_CORE
                + _EXIT_PEAK_V2_EXTRA
                + _HA
                + ["upper_wick_ratio", "lower_wick_ratio", "body_ratio"]
            )
            if c not in _EXIT_DEAD
        ]
        + [
            "adx_14",
            "minus_di_14",
            "div_rank_rsi",
            "div_rank_macd",
            "realized_vol_10",
            "bb_width_20",
            "vol_percentile_60",
        ],
    ),
    "exit_peak_div_rvol": (
        "exit_peak_div + realized_vol_10 only (the #2 exit-IC feature, absent) (38)",
        [
            c
            for c in (
                _EXIT_PEAK_CORE
                + _EXIT_PEAK_V2_EXTRA
                + _HA
                + ["upper_wick_ratio", "lower_wick_ratio", "body_ratio"]
            )
            if c not in _EXIT_DEAD
        ]
        + ["adx_14", "minus_di_14", "div_rank_rsi", "div_rank_macd", "realized_vol_10"],
    ),
    # LEAN downside-vol exit set (conditional_exit_ic.py 2026-06-08): the champion's
    # exit_peak_div_rvol BURIES the high-IC vol legs (atr fwd_dd-IC 0.30, realized_vol 0.29)
    # among ~33 dead overextension/divergence/HA features (discrimination: those are ~0 or
    # NEGATIVE). The diagnostic shows downside is LEADING-predictable by vol MAGNITUDE
    # (atr->fwd_dd skip-3 = -0.20, -0.247 NEAR-HIGH). This keeps ONLY the proven leading vol
    # predictors + near-high proximity (where the edge concentrates), nothing diluting.
    "exit_vol_lean": (
        "lean downside-vol exit: vol magnitude + near-high proximity, no dead overextension (11)",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "volatility_rank",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
        ],
    ),
    # exit_vol_lean + train-clean market-context legs (the UNTRIED lever: tops are market-
    # driven; single-stock price can't see the index/breadth rollover). Same 3 clean
    # cross-sectional/regime features used by rs_regime_v1 (NaN-safe after warmup-trim).
    "exit_vol_market": (
        "exit_vol_lean + market regime/breadth context (14)",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "volatility_rank",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
            "market_volatility_regime",
            "market_trend",
            "momentum_rank",
        ],
    ),
    # EXIT + MACD-HIST SHAPE (user temporal-pattern insight 2026-06-18): the rollover SHAPE the
    # hand-coded MACD shield read, given to the EXIT head as raw + percentile + slope/accel/lags so it
    # can learn a better-timed protective exit than the single-point head (which the B-retrain under-tested
    # by using a 5-bar MEAN = shape-destroying).
    "exit_vol_market_shape": (
        "exit_vol_market + MACD-hist SHAPE family (raw+pctile+slope/accel/lags) for rollover timing (24)",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "volatility_rank",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
            "market_volatility_regime",
            "market_trend",
            "momentum_rank",
            "macd_hist_sh",
            "macd_hist_pctile_252_sh",
            "macd_hist_slope_sh",
            "macd_hist_accel_sh",
            "macd_hist_l1_sh",
            "macd_hist_l2_sh",
            "macd_hist_slope_l1_sh",
            "macd_hist_slope_l2_sh",
            "macd_hist_slope_pctile_252_sh",
            "macd_line_sh",
        ],
    ),
    # BRANCH A: exit_vol_market + same-wave DOWNLEG STRUCTURE (the untested cut-timing axis,
    # project_t1804). The reward_risk head on vol-magnitude features is direction-symmetric and
    # lags the wave; give it the STRUCTURE of the current decline — DEPTH (dist_20d_high =
    # drawdown from the 20-bar high), AGE (aroon_up/down = bars since the recent high/low),
    # PERSISTENCE (down_days_5/10 = how many of the last N closes were down), and LEG structure
    # (higher_lows_count, rsi/macd divergence). Hypothesis: depth/age/persistence are
    # magnitude-correlated (the one predictable axis), so the head can scale the cut to how far
    # the leg has already run rather than firing symmetrically at tops AND bottoms.
    "exit_vol_downleg": (
        "exit_vol_market + downleg structure (depth/age/persistence/leg, 22)",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "volatility_rank",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
            "market_volatility_regime",
            "market_trend",
            "momentum_rank",
            "dist_20d_high",
            "aroon_up",
            "aroon_down",
            "down_days_5",
            "down_days_10",
            "higher_lows_count",
            "rsi_div_20",
            "macd_div_20",
        ],
    ),
    # exit_vol_market + DISTRIBUTION-day pressure (volume-confirmed selling). Tests whether
    # "số ngày phân phối" lets the reward_risk exit head see downside the volatility features miss.
    "exit_vol_dist": (
        "exit_vol_market + distribution-day counts (volume-confirmed selling)",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "volatility_rank",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
            "market_volatility_regime",
            "market_trend",
            "momentum_rank",
            "dist_day_25",
            "dist_day_vol20_25",
        ],
    ),
    # exit_vol_dist (the robust volume head, 2369) + MACD-hist TEMPORAL SHAPE (user insight): give the
    # head the lagged slope SEQUENCE + acceleration (curvature) + percentile so it reads top-vs-bottom
    # SHAPE, not a point-in-time slope. Tests whether sequence info un-sticks the phase-noise result.
    "exit_vol_dist_shape": (
        "exit_vol_dist + MACD-hist temporal shape (lagged slope seq + accel + percentile, 26)",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "volatility_rank",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
            "market_volatility_regime",
            "market_trend",
            "momentum_rank",
            "dist_day_25",
            "dist_day_vol20_25",
            "macd_hist_chg_1",
            "macd_hist_slope_l1",
            "macd_hist_slope_l2",
            "macd_hist_slope_l3",
            "macd_hist_slope_l4",
            "macd_hist_slope_l5",
            "macd_hist_accel",
            "macd_hist_accel_3",
            "macd_hist_pctile_252",
            "macd_hist_slope_pctile_252",
        ],
    ),
    # Lean shape: the minimal sequence that carries curvature — level slope + acceleration + 2 lags +
    # percentile (guards against the 26-feat full set overfitting the small exit-target signal).
    "exit_vol_dist_shape_lean": (
        "exit_vol_dist + MACD-hist core shape (slope + accel + 2 lags + percentile, 21)",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "volatility_rank",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
            "market_volatility_regime",
            "market_trend",
            "momentum_rank",
            "dist_day_25",
            "dist_day_vol20_25",
            "macd_hist_chg_1",
            "macd_hist_accel",
            "macd_hist_slope_l1",
            "macd_hist_slope_l2",
            "macd_hist_pctile_252",
        ],
    ),
    # exit_vol_market + 2-candle bearish engulfing + doji-cluster (the candle-pattern axis the
    # exit head currently has zero features for).
    "exit_vol_candle": (
        "exit_vol_market + bearish engulfing + doji-cluster",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "volatility_rank",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
            "market_volatility_regime",
            "market_trend",
            "momentum_rank",
            "bear_engulf",
            "doji_count_10",
        ],
    ),
    # exit_vol_market (MAGNITUDE: vol/dist + market context) UNION momentum-DECELERATION /
    # divergence (PHASE). Phase diagnosis (project_exit_leadlag_frontier): the reward_risk head
    # on exit_vol_market is FLAT at tops + lags ~10 bars because vol features lag the wave; it
    # carries "how big" but not "when". Add macd_hist_chg (decel), ma slopes (rollover), rsi/macd
    # divergence + extension context (dist_126d_high, up_days_10) so ONE head sees both axes —
    # the principled in-model fix (not a bolted-on rollover OR-rule). A/B vs 958 (comp 455.5).
    "exit_vol_phase": (
        "exit_vol_market + momentum deceleration / divergence / extension (PHASE, 23)",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "volatility_rank",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
            "market_volatility_regime",
            "market_trend",
            "momentum_rank",
            "macd_hist_chg_1",
            "macd_hist_chg_3",
            "macd_hist_chg_5",
            "ma10_slope",
            "ma20_slope",
            "rsi_div_20",
            "macd_div_20",
            "dist_126d_high",
            "up_days_10",
        ],
    ),
    # FULL velocity-DYNAMICS exit (user 2026-06-20, "tốc độ thay đổi hist" = sell when momentum
    # DECELERATES at a top): exit_vol_market base + the complete rate-of-change stack — macd-hist
    # 1st+2nd derivative + smooth-slope, RSI velocity, EMA-ribbon CONTRACTION (momentum fading),
    # %R velocity, MA slopes. Adds the deceleration sequence the vol-magnitude base is blind to.
    "exit_vol_dyn": (
        "exit_vol_market + full momentum-deceleration velocity stack (accel + %R + ribbon-contract, 24)",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "volatility_rank",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
            "market_volatility_regime",
            "market_trend",
            "momentum_rank",
            "macd_hist_chg_3",
            "macd_hist_chg_5",
            "macd_hist_accel",
            "macd_hist_smooth_slope",
            "rsi_14_slope10",
            "rsi_smooth_slope",
            "ema_ribbon_expand",
            "williams_r_slope5",
            "ma10_slope",
            "ma20_slope",
        ],
    ),
    # exit_directional (top-timing, NO symmetric vol) + the user's velocity ADDITIONS the directional
    # base lacks: macd-hist 2nd-derivative (accel), %R level+velocity, EMA-ribbon contraction. The
    # purest directional-top-DECELERATION exit — every feature separates tops from bottoms AND adds
    # the rate-of-change the original directional set only had as 1st-derivative.
    "exit_dir_dyn": (
        "exit_directional + macd-hist accel + %R + ribbon-contract (20, directional deceleration)",
        [
            "sma_20_ratio",
            "bb_pct_20",
            "dist_63d_high",
            "dist_52w_high",
            "dist_126d_high",
            "macd_hist_chg_1",
            "macd_hist_chg_3",
            "macd_hist_chg_5",
            "ma10_slope",
            "ma20_slope",
            "ma5_accel",
            "rsi_div_20",
            "macd_div_20",
            "up_days_10",
            "macd_hist_accel",
            "macd_hist_smooth_slope",
            "rsi_14_slope10",
            "williams_r_14",
            "williams_r_slope5",
            "ema_ribbon_expand",
        ],
    ),
    # No-cross ablation: exit_vol_market stripped of ALL cross-sectional / market
    # context (volatility_rank, momentum_rank, market_trend, market_volatility_regime)
    # — 10 purely per-symbol volatility/distance features. Pairs with leading_v2 on
    # the entry slot to make a fully per-symbol (no-universe-dependency) champion.
    "exit_vol_nocross": (
        "exit_vol_market minus cross-sectional/market context (10, per-symbol only)",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
        ],
    ),
    # DIRECTIONAL-ONLY exit (top-vs-bottom timing): proof (exit_head_shape_cmp.py +
    # feature top/bottom separation) that the head fires at BOTTOMS not tops because the
    # vol-MAGNITUDE features (atr/realized_vol/vol_percentile/bb_width/volatility_rank/
    # high_low_pct) are direction-SYMMETRIC (~0.01σ top-vs-bottom) — high at BOTH tops
    # (about to drop) and bottoms (just dropped) — so the head learns a spurious "high
    # vol => sell" rule and the DIRECTIONAL features get out-voted. This set DROPS every
    # symmetric vol feature and keeps ONLY features that separate tops from bottoms
    # (extension/dist-from-high 0.9-1.1σ, macd_hist/slope 0.7σ, divergence). Vol predicts
    # how BIG a drop is, not WHEN — wrong tool for exit TIMING. Subtractive fix.
    "exit_directional": (
        "directional top-timing features only — NO symmetric vol magnitude (14)",
        [
            "sma_20_ratio",
            "bb_pct_20",
            "dist_63d_high",
            "dist_52w_high",
            "dist_126d_high",
            "macd_hist_chg_1",
            "macd_hist_chg_3",
            "macd_hist_chg_5",
            "ma10_slope",
            "ma20_slope",
            "ma5_accel",
            "rsi_div_20",
            "macd_div_20",
            "up_days_10",
        ],
    ),
    # exit_directional + directional MARKET context (trend + xsec momentum rank), dropping
    # the symmetric market_volatility_regime.
    "exit_directional_mkt": (
        "exit_directional + directional market context (trend + momentum rank, 16)",
        [
            "sma_20_ratio",
            "bb_pct_20",
            "dist_63d_high",
            "dist_52w_high",
            "dist_126d_high",
            "macd_hist_chg_1",
            "macd_hist_chg_3",
            "macd_hist_chg_5",
            "ma10_slope",
            "ma20_slope",
            "ma5_accel",
            "rsi_div_20",
            "macd_div_20",
            "up_days_10",
            "market_trend",
            "momentum_rank",
        ],
    ),
    # TOP-TELLS exit: the user's explicit sell signals — heavy-volume down bars
    # (down_vol_*), distribution-day counts (dist_day_*), CLEAN directional divergence
    # (div_rank_*: + = price-high-but-momentum-not = bearish), bear engulfing, macd-hist
    # rollover, and extension/dist-from-high. All DIRECTIONAL/leading top signals, NO
    # symmetric vol. Tested on an EXPOSED baseline (overext off, low sell-z) so the head's
    # quality actually drives exits — the fair test the masked champion hides.
    "exit_toptells": (
        "user top-tells: down-volume bars + distribution days + clean divergence + extension (16)",
        [
            "down_vol_spike",
            "down_vol_count_10",
            "down_vol_intensity_5",
            "dist_day_25",
            "dist_day_vol20_25",
            "div_rank_rsi",
            "div_rank_macd",
            "macd_hist_chg_3",
            "macd_hist_chg_5",
            "bear_engulf",
            "sma_20_ratio",
            "bb_pct_20",
            "dist_63d_high",
            "dist_126d_high",
            "up_days_10",
            "market_trend",
        ],
    ),
    # exit_peak_div + distance-above-base (dist_*_low) — RE-tested after the loader
    # zero-close fix made these finite (they were inf-poisoned in R14, hence useless).
    # The user's "đỉnh xa nền" (extended distribution top) signal, now computable.
    # exit_peak_div + MACD-hist momentum (chg 2/5/10) — the user's "hist tăng/giảm so
    # N phiên" idea. Probe peak-vs-bottom separation 0.95 (chg_2) — the strongest
    # directional signal found (2x the rank-divergence). Exit monotone -1 (hist falling
    # = sell).
    # ENTRY set: exit_peak_v3 + the fast directional momentum/divergence signals, to
    # let an entry monotone (+macd_hist_chg, -div_rank_macd) suppress the dump-spike
    # (hist falling hard on a red bar -> entry forced low) where slow ma_slope couldn't.
    "entry_mom_v1": (
        "exit_peak_v3 + macd_hist_chg 2/5 + div_rank_macd (43)",
        _EXIT_PEAK_CORE
        + _EXIT_PEAK_V2_EXTRA
        + _HA
        + ["macd_hist_chg_2", "macd_hist_chg_5", "div_rank_macd"],
    ),
    # Entry WITHOUT the discrete zigzag-state features (which flip violently on the
    # swing-completing dump bar and overpower everything → the dump-spike). Replaces
    # the rigid step-function zz state with continuous momentum/divergence signals, so
    # the entry can't key on the zz-flip. Tests "drop the inflexible zz" hypothesis.
    "entry_nozz": (
        "exit_peak_v3 minus zz-state + macd_hist_chg 2/5 + div_rank_macd (continuous) (33)",
        [c for c in (_EXIT_PEAK_CORE + _EXIT_PEAK_V2_EXTRA + _HA) if c not in _ZZ_SWING]
        + ["macd_hist_chg_2", "macd_hist_chg_5", "div_rank_macd"],
    ),
    # entry_nozz minus atr_14_ratio (the new 45%-gain dominator that spikes on the
    # high-range dump bar — the post-zz dump-spike driver). Whack-a-mole test: does the
    # spike soften or does another magnitude feature just take over?
    # BREAKOUT entry: the user's ideal entry zone — volume breakout from accumulation +
    # rising-MA trend + HA reversal. NO zz-state, NO atr_14_ratio (the dump-spike
    # dominators). Paired with a forward-return target + DECOUPLED buy (zE alone) so the
    # entry alpha is actually USED — and breakout feats are LOW on a crash, so it should
    # fire at the 4/8-type breakout, not the 29/8 dump.
    "entry_breakout_v1": (
        "volume-breakout + accumulation + MA-trend + HA, no zz/atr (22)",
        [
            "breakout_20",
            "breakout_60",
            "volume_ratio_20",
            "volume_ratio_60",
            "bb_squeeze",
            "bb_width_20",
            "bb_pct_20",
            "sma_5_ratio",
            "sma_20_ratio",
            "sma_50_ratio",
            "ma5_slope",
            "ma10_slope",
            "ma_align",
            "rsi_14",
            "macd_hist",
            "macd_hist_chg_5",
            "ha_color",
            "ha_body",
            "ha_trend",
            "close_to_open",
            "ret_5d",
            "up_days_5",
        ],
    ),
    "entry_nozz_noatr": (
        "entry_nozz minus atr_14_ratio + atr_regime kept (32)",
        [
            c
            for c in (_EXIT_PEAK_CORE + _EXIT_PEAK_V2_EXTRA + _HA)
            if c not in _ZZ_SWING and c != "atr_14_ratio"
        ]
        + ["macd_hist_chg_2", "macd_hist_chg_5", "div_rank_macd"],
    ),
    "exit_peak_mom": (
        "exit_peak_div + macd_hist_chg 2/5/10 (40)",
        [
            c
            for c in (
                _EXIT_PEAK_CORE
                + _EXIT_PEAK_V2_EXTRA
                + _HA
                + ["upper_wick_ratio", "lower_wick_ratio", "body_ratio"]
            )
            if c not in _EXIT_DEAD
        ]
        + [
            "adx_14",
            "minus_di_14",
            "div_rank_rsi",
            "div_rank_macd",
            "macd_hist_chg_2",
            "macd_hist_chg_5",
            "macd_hist_chg_10",
        ],
    ),
    # exit_peak_div + the user's macd-hist acceleration over 1/2/3/5 bars (full short-
    # horizon momentum-of-momentum). Test whether shorter horizons add over the existing.
    "exit_peak_mom2": (
        "exit_peak_div + macd_hist_chg 1/2/3/5 (40)",
        [
            c
            for c in (
                _EXIT_PEAK_CORE
                + _EXIT_PEAK_V2_EXTRA
                + _HA
                + ["upper_wick_ratio", "lower_wick_ratio", "body_ratio"]
            )
            if c not in _EXIT_DEAD
        ]
        + [
            "adx_14",
            "minus_di_14",
            "div_rank_rsi",
            "div_rank_macd",
            "macd_hist_chg_1",
            "macd_hist_chg_2",
            "macd_hist_chg_3",
            "macd_hist_chg_5",
        ],
    ),
    # Entry with CROSS-SECTIONAL rank info (this stock vs all others each day) on top of
    # price — the genuinely-new information that RAISES the entry model's forward-IC
    # (price-only 0.059 -> +xsec 0.073, +24%). Not a mask: the model predicts better.
    "entry_xsec_v1": (
        "leading_v2 + cross-sectional ranks (momentum/rsi/strength/vol/volume) (42)",
        _LEADING_V2
        + ["momentum_rank", "rsi_rank", "price_strength_rank", "volatility_rank", "volume_rank"],
    ),
    # entry_xsec_v1 + the low-vol x near-52w-high IC winners (rank legs + their product).
    # Tests whether the strongest measured cross-sectional fwd-return signal lifts the
    # honest swing entry (IC 0.073 -> the lvnh composite measured 0.076-0.110 oos).
    "entry_xsec_lvnh": (
        "entry_xsec_v1 + low-vol60/near-high ranks + product (IC winner)",
        _LEADING_V2
        + [
            "momentum_rank",
            "rsi_rank",
            "price_strength_rank",
            "volatility_rank",
            "volume_rank",
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
        ],
    ),
    "entry_xsec_park": (
        "entry_xsec_v1 + parkinson-lowvol/near-high ranks + product",
        _LEADING_V2
        + [
            "momentum_rank",
            "rsi_rank",
            "price_strength_rank",
            "volatility_rank",
            "volume_rank",
            "lowvol_park_rank",
            "nearhigh_rank",
            "comp_park_nh",
        ],
    ),
    "entry_lvnh_lean": (
        "leading_v2 + ONLY the low-vol x near-high IC winners (no other xsec ranks)",
        _LEADING_V2 + ["lowvol_rank_60", "nearhigh_rank", "comp_nh_lv"],
    ),
    # R40 found lean (IC-winner only) BEATS full xsec — the other ranks were noise. Refine:
    "entry_park_lean": (
        "leading_v2 + parkinson-lowvol/near-high IC winners only",
        _LEADING_V2 + ["lowvol_park_rank", "nearhigh_rank", "comp_park_nh"],
    ),
    "entry_comp_lean": (
        "leading_v2 + ONLY the low-vol x near-high product (most minimal)",
        _LEADING_V2 + ["comp_nh_lv"],
    ),
    # R42: swap in the higher-OOS-IC composites on top of the champion lean entry to lift
    # per-trade quality (avg_pnl) at the same ~950-trade volume — the "more trades AND more
    # pnl" lever. _lvup adds the single best-OOS comp_lowvol_uptrend; _lvup126 stacks the
    # 126d-near-high variant; _parkup uses the parkinson family; _lvnhrev adds the high-ic5
    # 3-factor reversal composite to try to lift trade COUNT without losing IC.
    "entry_lvup_lean": (
        "entry_lvnh_lean + comp_lowvol_uptrend (best OOS22 IC 0.110)",
        _LEADING_V2 + ["lowvol_rank_60", "nearhigh_rank", "comp_nh_lv", "comp_lowvol_uptrend"],
    ),
    "entry_lvup126_lean": (
        "entry_lvnh_lean + uptrend-gated + 126d-near-high composites",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
        ],
    ),
    # Champion entry + ACCUMULATION/DISTRIBUTION-day counts (volume-confirmed buying vs selling
    # pressure over 25 sessions) — feed the entry head the "đang được gom hay đang bị phân phối" read.
    "entry_lvup126_accdist": (
        "entry_lvup126_lean + accumulation/distribution-day counts",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "accum_day_25",
            "dist_day_25",
        ],
    ),
    # Champion entry + bullish engulfing (2-candle bottom-reversal confirm).
    "entry_lvup126_engulf": (
        "entry_lvup126_lean + bullish engulfing (bottom-reversal confirm)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "bull_engulf",
        ],
    ),
    # Champion entry (lvup126) + the CROSS-SECTIONAL ranks (this stock vs all others each day) —
    # genuinely-new info that RAISES the entry model's forward-IC (price-only 0.059 -> +xsec 0.073,
    # +24%; entry_xsec_v1 note). Isolates the xsec contribution on top of the exact t518 features.
    "entry_lvup126_xsec": (
        "entry_lvup126_lean + cross-sectional ranks (momentum/rsi/strength/vol/volume)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "momentum_rank",
            "rsi_rank",
            "price_strength_rank",
            "volatility_rank",
            "volume_rank",
        ],
    ),
    # Champion entry + the missing CHEAPNESS legs (sma_200_ratio, dist_126d_low, dist_63d_low).
    # tail_signature.py showed tail winners are bought below 200MA / deep below longer-term levels;
    # lvup126_lean has dist_52w_* but NOT sma_200_ratio / 126d-63d lows. Feed them so the entry model
    # can LEARN the cheapness edge (rank-IC 0.238 on realized trades) instead of gating (which cuts trades).
    "entry_lvup126_cheap": (
        "entry_lvup126_lean + sma_200_ratio + dist_126d_low + dist_63d_low (cheapness legs)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "sma_200_ratio",
            "dist_126d_low",
            "dist_63d_low",
        ],
    ),
    # Champion entry + MARKET-CONTEXT legs (the per-symbol head is blind to the tape — the
    # entry_market_gate binary win +6 came from exactly this gap; young_loser_hunt: market 5d
    # return WR 0.51 weak → 0.69 strong, monotonic). Feed market_trend + vol regime so the head
    # can LEARN the market-regime conditioning (graded) instead of only the binary date-gate.
    "entry_lvup126_mkt": (
        "entry_lvup126_lean + market_trend + market_volatility_regime (tape context)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "market_trend",
            "market_volatility_regime",
        ],
    ),
    "entry_parkup_lean": (
        "leading_v2 + parkinson low-vol/near-high + parkinson-uptrend (OOS IC 0.101)",
        _LEADING_V2 + ["lowvol_park_rank", "nearhigh_rank", "comp_park_nh", "comp_park_uptrend"],
    ),
    # Feature hygiene: entry_lvup126_lean minus 8 near-zero-IC noise features (single-feature
    # forward-return |t20| < 2 -> no signal either direction, only dilution). Informative
    # negative-IC vol/low-vol features are KEPT (they encode the low-vol edge, not noise).
    "entry_lvup126_clean": (
        "entry_lvup126_lean minus near-zero-IC noise (minus_di/obv_slope/ret_10d/roc_10/rsi_14/rsi_7/bb_pct_20/vol_pct_60)",
        [
            c
            for c in (
                _LEADING_V2
                + [
                    "lowvol_rank_60",
                    "nearhigh_rank",
                    "comp_nh_lv",
                    "comp_lowvol_uptrend",
                    "comp_nh_x_lv126",
                ]
            )
            if c
            not in {
                "minus_di_14",
                "obv_slope_10",
                "ret_10d",
                "roc_10",
                "rsi_14",
                "rsi_7",
                "bb_pct_20",
                "vol_percentile_60",
            }
        ],
    ),
    # Champion entry (lvup126) + short-horizon recovery-from-washout proxies. Tests the
    # this-session finding that "recently washed out, now recovering" carries fwd-return
    # signal ORTHOGONAL to the continuation score (within-zE rank-IC +0.05) — feed the model
    # the ingredients so it can upweight confirmed reversals instead of scoring them coincidently.
    "entry_lvup126_recov": (
        "entry_lvup126_lean + short-horizon recovery (dist_10d_low, range_pos_20, recov_setup)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
        ],
    ),
    # TOP-AWARE entry retrain (C-arc 2026-06-18): recov + the structural TOP signals the head is blind to,
    # so the entry head learns to down-score du-dinh/right-shoulder entries instead of a post-hoc gate.
    "entry_lvup126_topaware": (
        "entry_lvup126_recov + top-structure (upper_wick_5, lower_high_20)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "upper_wick_5",
            "lower_high_20",
        ],
    ),
    # SHAPE entry (research 2026-06-18, user shape>point insight): recov + the STRONGEST researched shape
    # signals (rsi_slope_5 IC -0.155, ext_ma20_slope5 -0.146, rsi/macd pctile + macd-hist) so the entry head
    # can read momentum-rollover SHAPE (knife-dip vs bottoming-dip), not the inert single point.
    "entry_lvup126_recov_shape": (
        "entry_lvup126_recov + RSI/ext/MACD SHAPE (slope+pctile, the strongest researched top-signals)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "rsi_14_slope5_sh",
            "rsi_14_pctile_252_sh",
            "ext_ma20_slope5_sh",
            "macd_hist_sh",
            "macd_hist_slope_sh",
            "macd_hist_pctile_252_sh",
        ],
    ),
    # KNIFE-AXIS entry (2026-06-18, decouple_oracle + trade_dump_cohorts): the HEAVY_LOSS cohort is a
    # falling-knife profile the recov set is blind to — bought ~30% below the 63d high (dist_63d_high)
    # and below the 200MA (structural decliner), vs BIG_WIN bought near highs / above MAs. recov already
    # carries sma_50_ratio + range_pos_20 + dist_52w_high; what it lacks is the SHORT-horizon high-distance
    # (dist_63d_high) and the long-term structural gate (sma_200_ratio). Fed to the entry head and trained
    # in the PULLBACK-OFF sandbox so the model can learn knife-vs-strength natively (replace the crutch),
    # judged by sandbox delta vs entry_lvup126_recov (NOT the crutch-loaded composite, which masks it).
    "entry_recov_knife": (
        "entry_lvup126_recov + dist_63d_high + sma_200_ratio (falling-knife / structural-decliner axis)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "dist_63d_high",
            "sma_200_ratio",
        ],
    ),
    # THRUST entry (2026-07-11, user hypothesis #3 "signal quality on default universe"): champion
    # recov + the classic breakout-timing signals never fed as a SET — bb_squeeze (volatility
    # compression before expansion), volume-thrust (volume_ratio_5/20), consecutive-up (up_days_5/10),
    # and shakeout lower-wick (rejection candle). Tests whether momentum-continuation entry timing
    # improves when the head can read squeeze→expansion + volume confirmation natively. Judged by
    # composite/pnl delta vs entry_lvup126_recov on the SAME 61-sym universe (fair).
    "entry_lvup126_thrust": (
        "entry_lvup126_recov + squeeze/volume-thrust/consecutive-up/shakeout-wick",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "bb_squeeze",
            "volume_ratio_5",
            "volume_ratio_20",
            "up_days_5",
            "up_days_10",
            "lower_wick_ratio",
        ],
    ),
    # SMAC-entry fix (2026-06-19): the single-model action classifier's worst losers are 2022
    # bear-market knife-catches (NVL -81%, all entered into a falling tape). entry_recov_knife
    # gives the per-stock knife/structural-decliner axis; this adds the MARKET-regime/tape
    # context (market_trend + market_volatility_regime) on top so the ENTER head can also down-
    # score buys in a falling/high-vol market regime — addressing the loss at its source (don't
    # enter) rather than via an exit rule. Judged by sandbox delta vs entry_lvup126_recov.
    "entry_recov_knife_mkt": (
        "entry_recov_knife + market_trend + market_volatility_regime (knife profile + tape regime)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "dist_63d_high",
            "sma_200_ratio",
            "market_trend",
            "market_volatility_regime",
        ],
    ),
    # SMAC stage-1 STRUCTURE/POSITION-STATE proxies (2026-06-19, TA-forensic): give the single
    # action model the price-structure a TA reads but it was blind to — DEPTH (dist_20d_high =
    # drawdown from recent peak ≈ position drawdown-from-peak), AGE (aroon_up/down = bars since
    # the recent high/low ≈ bars-held), PERSISTENCE (down_days_10), SWING STRUCTURE (higher_lows_
    # count, lower_high_20, broke_support_20 = the 96%-WR held-support discriminator) and
    # DIVERGENCE (rsi_div_20, macd_div_20). Batch market-feature PROXIES for position-state
    # (Stage 1: validate the signal before the interactive-engine rewrite).
    "entry_struct": (
        "entry_lvup126_recov + structure/position-state proxies + divergence (TA blind spots)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "dist_20d_high",
            "aroon_up",
            "aroon_down",
            "down_days_10",
            "higher_lows_count",
            "lower_high_20",
            "broke_support_20",
            "rsi_div_20",
            "macd_div_20",
        ],
    ),
    # SMAC new-angle (2026-06-19): cross-sectional RELATIVE STRENGTH / leadership — buy the
    # dips of LEADERS (stocks outperforming the universe), not laggards (a core TA edge the
    # single-symbol oracle is blind to). momentum_rank = 20d-return rank vs universe (RS level);
    # cs_rank_trend = 10-bar change of that rank (leadership rotation / accumulation, the
    # strongest per-symbol signal in prior research); price_strength_rank = trend-strength rank.
    "entry_recov_rs": (
        "entry_lvup126_recov + cross-sectional relative-strength/leadership (momentum_rank, cs_rank_trend, price_strength_rank)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "momentum_rank",
            "cs_rank_trend",
            "price_strength_rank",
        ],
    ),
    # co-adaptation test (2026-07-11): pruned 11-feature core, paired with ALTERNATIVE targets to test
    # whether the 48-feature optimum is target-CONDITIONAL (a different target may prefer a different
    # feature set/count). Under triple_barrier: 48 >> 11. If a new target flips that, co-adaptation is real.
    "entry_rs_core11": (
        "entry_recov_rs pruned to 11 leadership/recovery core features",
        [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "momentum_rank",
            "cs_rank_trend",
            "price_strength_rank",
        ],
    ),
    # SMAC volume-niche (2026-06-19): confirm the bottom with VOLUME — a TA reads capitulation
    # (down_vol_count_10 = selling-climax bars), accumulation (accum_day_25, updown_vol_20,
    # net_acc_vol_30, cmf_20), money-flow vs price divergence (obv_price_divergence), and
    # volume-at-price support (dist_vwap_20 = the strongest single entry feature found,
    # IC_pnl +0.195: close above the 20d VWAP = volume accumulated BELOW = supported).
    # SMAC indicator-DYNAMICS niche (user 2026-06-20): give the model the multi-bar CHANGE /
    # expansion-contraction the snapshot features miss — MA & EMA ribbon spread + their expansion
    # rate ('suy giãn ra/thu lại'), Williams %R + velocity, RSI velocity + RSI-vs-its-MA, MACD
    # histogram rate-of-change + acceleration. The dynamics/sequence axis, on the v15 RS base.
    "entry_recov_rs_dyn": (
        "entry_recov_rs + indicator dynamics (MA/EMA ribbon spread+expansion, %R+vel, RSI vel/vs-MA, MACD hist chg+accel)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "momentum_rank",
            "cs_rank_trend",
            "price_strength_rank",
            "ma_spread_10_50",
            "ma_spread_expand5",
            "ema_spread_8_21",
            "ema_spread_expand5",
            "williams_r_14",
            "williams_r_slope5",
            "rsi_14_slope10",
            "rsi_vs_ma5",
            "macd_hist_chg_5",
            "macd_hist_accel",
        ],
    ),
    # SMAC new niches (2026-06-20) on the v18 dynamics base. SQUEEZE = volatility contraction
    # (coiling) precedes expansion — directly relevant to the model's big-swing selection.
    "entry_dyn_sqz": (
        "entry_recov_rs_dyn + volatility squeeze/contraction (bb_squeeze, bb_width_percentile, atr_contraction, consolidation_score)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "momentum_rank",
            "cs_rank_trend",
            "price_strength_rank",
            "ma_spread_10_50",
            "ma_spread_expand5",
            "ema_spread_8_21",
            "ema_spread_expand5",
            "williams_r_14",
            "williams_r_slope5",
            "rsi_14_slope10",
            "rsi_vs_ma5",
            "macd_hist_chg_5",
            "macd_hist_accel",
            "bb_squeeze",
            "bb_width_percentile",
            "atr_contraction",
            "consolidation_score",
        ],
    ),
    # TREND EFFICIENCY = clean trend vs chop (Kaufman ER) + ADX trend strength.
    "entry_dyn_eff": (
        "entry_recov_rs_dyn + trend efficiency (efficiency_ratio_10/20, adx_14)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "momentum_rank",
            "cs_rank_trend",
            "price_strength_rank",
            "ma_spread_10_50",
            "ma_spread_expand5",
            "ema_spread_8_21",
            "ema_spread_expand5",
            "williams_r_14",
            "williams_r_slope5",
            "rsi_14_slope10",
            "rsi_vs_ma5",
            "macd_hist_chg_5",
            "macd_hist_accel",
            "efficiency_ratio_10",
            "efficiency_ratio_20",
            "adx_14",
        ],
    ),
    # SECTOR-RELATIVE strength (strong WITHIN sector) — robust subset (beta_to_sector excluded: NaN-prone).
    "entry_dyn_sec": (
        "entry_recov_rs_dyn + sector-relative (return/momentum/volume/strength vs sector)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "momentum_rank",
            "cs_rank_trend",
            "price_strength_rank",
            "ma_spread_10_50",
            "ma_spread_expand5",
            "ema_spread_8_21",
            "ema_spread_expand5",
            "williams_r_14",
            "williams_r_slope5",
            "rsi_14_slope10",
            "rsi_vs_ma5",
            "macd_hist_chg_5",
            "macd_hist_accel",
            "return_vs_sector",
            "momentum_vs_sector",
            "volume_vs_sector",
            "strength_vs_sector",
        ],
    ),
    # De-diluted: just the SINGLE strongest volume feature (dist_vwap_20) on the RS base.
    "entry_recov_rs_vwap": (
        "entry_recov_rs + dist_vwap_20 (volume-at-price support, strongest single volume feature)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "momentum_rank",
            "cs_rank_trend",
            "price_strength_rank",
            "dist_vwap_20",
        ],
    ),
    "entry_recov_rs_vol": (
        "entry_recov_rs + volume confirmation (dist_vwap_20, net_acc_vol_30, obv_price_divergence, accum_day_25, updown_vol_20, down_vol_count_10, cmf_20)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "momentum_rank",
            "cs_rank_trend",
            "price_strength_rank",
            "dist_vwap_20",
            "net_acc_vol_30",
            "obv_price_divergence",
            "accum_day_25",
            "updown_vol_20",
            "down_vol_count_10",
            "cmf_20",
        ],
    ),
    # DERIVED-SIGNAL HUNT (2026-06-17): champion entry (recov) + genuinely-NEW daily-OHLCV families,
    # isolated per family so the harness can attribute any lift. Base = entry_lvup126_recov.
    "entry_recov_eff": (
        "recov + Kaufman efficiency ratio 10/20 (trend QUALITY: clean-trend vs chop)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "efficiency_ratio_10",
            "efficiency_ratio_20",
        ],
    ),
    "entry_recov_micro": (
        "recov + compression/expansion microstructure (nr7, inside_bar, gap_oc)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "nr7",
            "inside_bar",
            "gap_oc",
        ],
    ),
    "entry_recov_kelt": (
        "recov + Keltner channel position (ATR-channel, orthogonal to std-based bb_pct)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "keltner_pct_20",
        ],
    ),
    # MULTI-PERIOD / RIBBON entry (user 2026-06-18): give the entry heads the fast-vs-slow momentum +
    # EMA-fanning the knife cohort (causally near-identical to winners on single features) DOES separate
    # on (rsi_fastslow +0.213, ribbon +0.189, smooth-slope +0.181). Goal: pick fewer late-stage knives.
    "entry_recov_mp": (
        "recov + multi-period RSI/EMA-ribbon/smoothed momentum (fast-vs-slow + fanning + co-giãn)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "rsi_fastslow",
            "rsi_21",
            "rsi_50",
            "rsi_smooth_slope",
            "ema_ribbon_width",
            "ema_ribbon_expand",
            "ema_stretch_5_50",
            "macd_hist_smooth_slope",
        ],
    ),
    "entry_recov_mp_lean": (
        "recov + the 3 best knife-vs-winner separators (rsi_fastslow + smooth-slope + ribbon)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "rsi_fastslow",
            "rsi_smooth_slope",
            "ema_ribbon_width",
        ],
    ),
    # FULL INDICATOR-DYNAMICS set (user 2026-06-20): give the entry head the EXPANSION/CONTRACTION +
    # VELOCITY + multi-day SEQUENCE of every momentum tool — MA slopes/accel/align, MA-spread width +
    # expansion rate, RSI multi-period + velocity + smooth, MACD-hist velocity + acceleration + lagged
    # sequence, Williams %R + velocity, EMA-ribbon width/expand/stretch. The point levels the head sees;
    # this adds the rate-of-change / co-giãn it is blind to.
    "entry_recov_dyn": (
        "recov + full indicator DYNAMICS (MA/RSI/%R/MACD-hist velocity + expansion + lagged sequence)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "ma5_slope",
            "ma10_slope",
            "ma20_slope",
            "ma5_accel",
            "ma_align",
            "ma_spread_10_50",
            "ma_spread_expand5",
            "ma_spread_20_100",
            "rsi_fastslow",
            "rsi_21",
            "rsi_50",
            "rsi_smooth_slope",
            "rsi_14_slope10",
            "ema_ribbon_width",
            "ema_ribbon_expand",
            "ema_stretch_5_50",
            "macd_hist_chg_3",
            "macd_hist_chg_5",
            "macd_hist_accel",
            "macd_hist_slope_l1",
            "macd_hist_slope_l2",
            "williams_r_14",
            "williams_r_slope5",
        ],
    ),
    # PURE indicator-DYNAMICS (user 2026-06-20, "raw un-mask" path): ONLY the expansion/velocity/accel
    # signals, NO recov/price-level base. Used as a SEPARATE csrank ensemble head — its cross-sectional
    # rank = "which symbol is accelerating/expanding HARDEST right now" (raw-preserving; per-symbol z
    # FLATTENS the expansion magnitude, csrank keeps it). The pure set keeps the head's csrank a clean
    # dynamics-strength ranking, not diluted by the point-level features the main head already carries.
    "entry_dyn_pure": (
        "PURE dynamics: MA/RSI/%R/MACD-hist velocity + expansion + accel (no price-level base) for a csrank head",
        [
            "ema_ribbon_width",
            "ema_ribbon_expand",
            "ema_stretch_5_50",
            "ma_spread_10_50",
            "ma_spread_expand5",
            "ma_spread_20_100",
            "ma5_slope",
            "ma10_slope",
            "ma20_slope",
            "ma5_accel",
            "rsi_fastslow",
            "rsi_14_slope10",
            "rsi_smooth_slope",
            "rsi_21",
            "rsi_50",
            "macd_hist_chg_3",
            "macd_hist_chg_5",
            "macd_hist_accel",
            "macd_hist_smooth_slope",
            "williams_r_14",
            "williams_r_slope5",
        ],
    ),
    # PURE cross-sectional RELATIVE-STRENGTH / leadership (user 2026-06-20, "RS vs index" axis): the
    # strongest per-symbol signal found (cs_rank_trend IC_winner +0.17 vs dynamics ~0.05-0.11). A
    # SEPARATE csrank ensemble head fires on the cross-sectionally STRONGEST/most-accumulating names —
    # the "buy strength" TA principle, tested in the realizable raw/csrank form (per-symbol z would
    # flatten the rank's cross-sectional meaning; the rank IS already cross-sectional, csrank keeps it).
    "entry_rs_pure": (
        "PURE cross-sectional leadership / relative-strength ranks for a csrank head",
        [
            "momentum_rank",
            "cs_rank_trend",
            "price_strength_rank",
            "nearhigh_rank",
            "nearhigh126_rank",
            "volume_rank_20d",
            "strength_vs_sector",
        ],
    ),
    "entry_recov_deriv": (
        "recov + ALL new derived signals (eff ratio + micro nr7/inside/gap + keltner)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "efficiency_ratio_10",
            "efficiency_ratio_20",
            "nr7",
            "inside_bar",
            "gap_oc",
            "keltner_pct_20",
        ],
    ),
    # Champion entry (recov) + VOLUME-STRUCTURE (volume-at-price/VWAP-distance + net-accumulation).
    # shape_probe gate: dist_vwap_20 IC_pnl +0.195 (>score3), net_acc_vol_30 IC_mae +0.094 (drawdown).
    # GENUINELY new vs the catalog point-features (reads WHERE money traded, not accum-day counts).
    "entry_lvup126_volstruct": (
        "entry_lvup126_recov + volume-at-price (dist_vwap_20/40) + net-accumulation-vol_30",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "dist_vwap_20",
            "dist_vwap_40",
            "net_acc_vol_30",
        ],
    ),
    # Isolate the single strongest new feature (dist_vwap_20) on top of recov.
    "entry_lvup126_vwap": (
        "entry_lvup126_recov + dist_vwap_20 only (isolate the strongest new feature)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "dist_vwap_20",
        ],
    ),
    # NEW-SIGNAL entry (2026-06-17 feature audit): recov + the 3 genuinely-untried features the audit
    # surfaced — dist_vwap_20 (volume-at-price, IC_pnl +0.195>score3), ma5_accel (acceleration, exit had
    # it, entry didn't), cs_rank_trend (DERIVATIVE of the xsec rank, the +0.17 leadership-rotation signal).
    "entry_recov_newsig": (
        "entry_lvup126_recov + dist_vwap_20 + ma5_accel + cs_rank_trend (audit's new signals)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "dist_vwap_20",
            "ma5_accel",
            "cs_rank_trend",
        ],
    ),
    # USER FEATURE PROPOSAL (2026-06-19): money-flow at events + price vs volume-breakout zone +
    # crossovers — all 3 ideas on the champion entry. dist_vwap (price vs where money traded),
    # net_acc_vol (signed money-flow accumulation), ma/px/vol/macd cross state.
    "entry_recov_flowcross": (
        "entry_lvup126_recov + money-flow (net_acc) + volume-zone (dist_vwap) + crossovers (ma/px/vol/macd)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "range_pos_20",
            "recov_setup",
            "dist_vwap_20",
            "dist_vwap_40",
            "net_acc_vol_30",
            "ma_cross_20_50",
            "px_cross_ma20",
            "vol_breakout_20",
            "macd_cross",
        ],
    ),
    # NEW-SIGNAL exit: vol-norm magnitude head + momentum-DERIVATIVE rollover (rsi_slope_5d,
    # macd_hist_chg_3 = how fast momentum is rolling over) + cs_rank_trend (relative-strength rollover).
    # The pure derivative-rollover set the audit found missing from exit_vol_market.
    "exit_volnorm_deriv": (
        "exit_vol_market + rsi_slope_5d + macd_hist_chg_3 + cs_rank_trend (momentum-derivative rollover)",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "volatility_rank",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
            "market_volatility_regime",
            "market_trend",
            "momentum_rank",
            "rsi_slope_5d",
            "macd_hist_chg_3",
            "cs_rank_trend",
        ],
    ),
    # DISTRIBUTION-AWARE exit (2026-06-18, exit_struct_discriminate.py): exit_vol_market + the 3 signals
    # that POSITIVELY separate a real top from a premature (SOLD_THEN_RAN) exit — consolidation_score
    # (sep +0.82sd, the distribution/sideways regime = the user's "dao động đi ngang khi động lượng giảm",
    # the standout), macd_hist_chg_5 (+0.38, hist giảm dần), dist_day_25 (+0.28, số ngày phân phối). The
    # cons-gate already helped (+1.7); this feeds the SAME signal to the exit HEAD so it can score
    # distribution tops natively. Pair with exit_gate=cons2 for the feature+gate synergy test.
    "exit_vol_dist2": (
        "exit_vol_market + consolidation_score + macd_hist_chg_5 + dist_day_25 (distribution-aware)",
        [
            "atr_14_ratio",
            "realized_vol_10",
            "vol_percentile_60",
            "bb_width_20",
            "volatility_rank",
            "high_low_pct_5d",
            "ma5_accel",
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "bb_pct_20",
            "market_volatility_regime",
            "market_trend",
            "momentum_rank",
            "consolidation_score",
            "macd_hist_chg_5",
            "dist_day_25",
        ],
    ),
    # SUPPLY/VOLUME-AXIS exit sets (2026-06-18, user: the volume axis is the one NOT realizability-walled
    # — exit_vol_dist (volume-distribution-days) won robust +2.78; PRICE-shape lost. Lean further into
    # REAL-SUPPLY signals on the dist base. Each adds ONE focused supply theme (keep lean to avoid the
    # dilution that sank the 26-feat shape set). _D = the robust exit_vol_dist 16-feature base.
    "exit_vol_obvdiv": (
        "exit_vol_dist + OBV / effort-vs-result divergence (volume up, price flat = distribution)",
        _EXIT_VOL_DIST + ["obv_price_divergence", "obv_slope_10", "pv_divergence"],
    ),
    "exit_vol_flow": (
        "exit_vol_dist + money-flow (CMF + MFI = net buying/selling pressure)",
        _EXIT_VOL_DIST + ["cmf_20", "mfi_14"],
    ),
    "exit_vol_downpress": (
        "exit_vol_dist + down-volume selling pressure (intensity/count + up:down volume)",
        _EXIT_VOL_DIST + ["down_vol_intensity_5", "down_vol_count_10", "updown_vol_20"],
    ),
    # REGIME-AWARE exit (2026-07-11, shakeout_vs_top forensic hb_60/61): exit_vol_downpress is
    # LOCAL price/vol — forensic showed 49% of 'signal' exits are shakeouts (dip recovers +13.7%)
    # and local features barely separate shakeout from top (AUC~0.5). The strong discriminator is
    # MARKET REGIME (mkt>MA50 AUC 0.78) + stock trend/RS. The base already carries the WEAK regime
    # feats (market_trend=MA200, momentum_rank=20d); this adds the STRONG missing ones so the exit
    # head HOLDS bull-tape shakeouts and SELLS bear-tape tops. Monotone -1 (bullish regime lowers
    # the sell score) can be added on the exit ML later; start unconstrained.
    "exit_vol_regime": (
        "exit_vol_downpress + market regime (mkt>MA50, mkt mom 60) + stock trend MA50 + RS 60d",
        _EXIT_VOL_DIST
        + [
            "down_vol_intensity_5",
            "down_vol_count_10",
            "updown_vol_20",
            "market_trend_50",
            "market_mom_60",
            "sma_50_ratio",
            "rs_rank_60",
        ],
    ),
    # CROSS-SECTIONAL RS exit (2026-07-11): exit_vol_regime (market-regime common-factor) FAILED
    # because the recombine z-scores the exit signal per-symbol/time and washes out slow common
    # factors. Cross-sectional RS RANKS are PER-SYMBOL relative (survive z-scoring) — sell weak-RS
    # names sooner, hold strong-RS names. Pairs with the ft_rs entry RS win (entry_recov_rs). Base
    # already carries momentum_rank; add the leadership/trend/60d-RS ranks not in it.
    "exit_vol_rs": (
        "exit_vol_downpress + cross-sectional RS ranks (cs_rank_trend, price_strength_rank, rs_rank_60)",
        _EXIT_VOL_DIST
        + [
            "down_vol_intensity_5",
            "down_vol_count_10",
            "updown_vol_20",
            "cs_rank_trend",
            "price_strength_rank",
            "rs_rank_60",
        ],
    ),
    # PATH-AWARE exit (2026-07-11, root reasoning: the exit is PATH-DEPENDENT — depends on how far
    # into the move / how old / how much given back — but the exit ML head is STATELESS per-bar, so
    # the RULES (max_hold/trailing) that DO carry path-state beat it (~2x mask). exit_vol_rs already
    # has DEPTH/giveback (dist_63d_high, dist_52w_high) + overextension (sma_20_ratio) but LACKS the
    # AGE + REALIZED-LEG structure: aroon (bars since recent high/low), up_days_10 (up-streak),
    # dist_63d_low (how far ABOVE the recent trough = realized up-leg amplitude), dist_20d_high
    # (short-term giveback). These pivot-anchored proxies let the stateless head APPROXIMATE the
    # path-state the rules hardcode -> raw exit signal may strengthen enough to unmask the rules.
    "exit_vol_path": (
        "exit_vol_rs + path-proxy (aroon age, up_days streak, dist_63d_low leg, dist_20d_high giveback)",
        _EXIT_VOL_DIST
        + [
            "down_vol_intensity_5",
            "down_vol_count_10",
            "updown_vol_20",
            "cs_rank_trend",
            "price_strength_rank",
            "rs_rank_60",
            "aroon_up",
            "aroon_down",
            "up_days_10",
            "dist_63d_low",
            "dist_20d_high",
        ],
    ),
    # PV-CHANNEL probe sets (2026-07-09, OHLCV_VIRGIN_MAP): champion exit set + the surviving
    # price-volume microstructure channel. NEW names (not edits) so the champion's
    # exit_vol_downpress stays byte-identical and its per-feature caches uncollided.
    "exit_vol_downpress_pv": (
        "exit_vol_downpress + pv_corr_10 + clv + nr_pos_7 (pv-channel probe, full)",
        _EXIT_VOL_DIST
        + [
            "down_vol_intensity_5",
            "down_vol_count_10",
            "updown_vol_20",
            "pv_corr_10",
            "clv",
            "nr_pos_7",
        ],
    ),
    "exit_vol_downpress_pvonly": (
        "exit_vol_downpress + pv_corr_10 only (isolate the main pv channel)",
        _EXIT_VOL_DIST
        + ["down_vol_intensity_5", "down_vol_count_10", "updown_vol_20", "pv_corr_10"],
    ),
    "exit_vol_vwap": (
        "exit_vol_dist + anchored-VWAP value (price vs where volume traded)",
        _EXIT_VOL_DIST + ["dist_vwap_20", "dist_vwap_40"],
    ),
    "exit_vol_adbal": (
        "exit_vol_dist + accumulation/distribution balance (ad_balance + net-accum-vol)",
        _EXIT_VOL_DIST + ["ad_balance_20", "net_acc_vol_30"],
    ),
    "exit_vol_supplymix": (
        "exit_vol_dist + curated best-of supply (obv-div + cmf + down-vol + ad-balance)",
        _EXIT_VOL_DIST
        + ["obv_price_divergence", "cmf_20", "down_vol_intensity_5", "ad_balance_20"],
    ),
    # FORGOTTEN-SEQUENCE exit sets (2026-06-18 feature audit): undeployed SEQUENCE features encoding
    # WHERE-IN-THE-SWING (pivot-anchored) + leadership ROTATION + proper Corr-divergence + Dow structure
    # — a DIFFERENT kind of sequence than the (walled) MACD-momentum-shape.
    "exit_vol_swing": (
        "exit_vol_dist + zigzag swing-position (return/bars/max-adverse since pivot)",
        _EXIT_VOL_DIST
        + ["zz_return_since_pivot", "zz_bars_since_pivot", "zz_max_adverse_since_pivot"],
    ),
    "exit_vol_csrt": (
        "exit_vol_dist + cs_rank_trend (cross-sectional leadership rotation, untested on exit)",
        _EXIT_VOL_DIST + ["cs_rank_trend"],
    ),
    "exit_vol_swing_csrt": (
        "exit_vol_dist + swing-position + leadership rotation (forgotten-sequence combo)",
        _EXIT_VOL_DIST
        + [
            "zz_return_since_pivot",
            "zz_bars_since_pivot",
            "zz_max_adverse_since_pivot",
            "cs_rank_trend",
        ],
    ),
    "exit_vol_dowstruct": (
        "exit_vol_dist + Dow structure (lower_high/higher_lows) + trend age (Aroon)",
        _EXIT_VOL_DIST + ["lower_high_20", "higher_lows_count", "aroon_up", "aroon_down"],
    ),
    "exit_vol_divproper": (
        "exit_vol_dist + proper rolling-Corr divergence (macd/rsi div + rank-div + obv-div)",
        _EXIT_VOL_DIST + ["macd_div_20", "rsi_div_20", "div_rank_macd", "obv_price_divergence"],
    ),
    # TOP-CLASSIFIER input set (2026-06-18): the SHAPE/SEQUENCE features the audit surfaced, paired with
    # a zigzag-PEAK target (where price-shape IS predictive, unlike the walled exit-downside-regression).
    # Context + macd-shape (lags/accel/percentile) + MA-rollover + Corr-divergence + Dow structure + trend
    # age (Aroon) + pivot-anchored swing-position + volume-distribution confirmation.
    "exit_topclf": (
        "top-classifier inputs: shape+divergence+Dow+swing-position+volume (for a zigzag-peak target)",
        [
            "dist_63d_high",
            "dist_52w_high",
            "sma_20_ratio",
            "rsi_14",
            "atr_14_ratio",
            "macd_hist_chg_1",
            "macd_hist_accel",
            "macd_hist_slope_l1",
            "macd_hist_slope_l2",
            "macd_hist_pctile_252",
            "ma10_slope",
            "ma20_slope",
            "macd_div_20",
            "rsi_div_20",
            "div_rank_macd",
            "lower_high_20",
            "higher_lows_count",
            "aroon_up",
            "aroon_down",
            "zz_return_since_pivot",
            "zz_bars_since_pivot",
            "zz_max_adverse_since_pivot",
            "dist_day_25",
            "down_vol_intensity_5",
        ],
    ),
    "entry_lvnhrev_lean": (
        "entry_lvnh_lean + comp_lv_nh_rev 3-factor (highest ic5 -> earlier/more trades)",
        _LEADING_V2 + ["lowvol_rank_60", "nearhigh_rank", "comp_nh_lv", "comp_lv_nh_rev"],
    ),
    # DEDICATED BREAKOUT/CONTINUATION set for the 3rd ensemble head (score3): the near-high,
    # range-break, resistance-break, rising-MA, volume-thrust structure that distinguishes a
    # genuine breakout-continuation from a dip — the signal the dip-leaning primary/reversal
    # heads can't see. Feeds the continuation_entry target so the head fires on real breakouts.
    # VOLUME-REGION-BALANCE set for the 6th union entry head (winner meta-label): up/down-volume
    # ratio + accum/dist candle balance + CMF/OBV/MFI — the SEQUENCE/region volume read orthogonal
    # to the 5 heads' targets (resIC vs winner +0.19, 7/7 yrs). The "ADD a head" wall-clearing lever.
    "entry_volbalance": (
        "leading_v2 + volume-region balance (updown_vol, ad_balance, cmf, accum/dist days, obv-div)",
        _LEADING_V2
        + [
            "updown_vol_20",
            "ad_balance_20",
            "cmf_20",
            "accum_day_25",
            "dist_day_25",
            "obv_slope_10",
            "obv_price_divergence",
            "mfi_14",
            "up_days_10",
            "down_days_10",
        ],
    ),
    "entry_breakout": (
        "leading_v2 + breakout/continuation structure (range-break, resistance, MA-slope, vol-thrust)",
        _LEADING_V2
        + [
            "breakout_20",
            "breakout_60",
            "dist_20d_high",
            "dist_to_resistance",
            "range_pos_20",
            "ma5_slope",
            "ma20_slope",
            "ma_align",
            "vol_surge_ratio",
            "close_pos_ma5",
            "breakout_setup_score",
        ],
    ),
    # Reversal-CONFIRMATION entry: lvup126 quality composites + the signals that tell a
    # reversal_entry-trained model WHEN a dip has actually turned (so it stops buying knives and
    # times confirmed bottoms). Recovery-from-washout (dist_*_low, range_pos_20, recov_setup),
    # reversal momentum/turn (macd_hist_chg 2/5, rsi/macd divergence), and causal zigzag-bottom
    # structure (last_dir, return/bars_since_pivot, price_pos_in_swing, dist_to_confirm,
    # max_adverse). Paired with the reversal_entry target (dip-gate + min_fwd_rally confirmation).
    "entry_reversal_confirm": (
        "lvup126 quality + recovery/divergence/zigzag-bottom reversal-confirmation features",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "dist_63d_low",
            "range_pos_20",
            "recov_setup",
            "macd_hist_chg_2",
            "macd_hist_chg_5",
            "div_rank_macd",
            "rsi_div_20",
            "macd_div_20",
            "up_days_10",
            "zz_last_dir",
            "zz_return_since_pivot",
            "zz_bars_since_pivot",
            "zz_price_pos_in_swing",
            "zz_dist_to_confirm",
            "zz_max_adverse_since_pivot",
        ],
    ),
    # LEAN reversal-confirm for the ensemble 2nd head: entry_reversal_confirm MINUS the 6 zigzag
    # zz_* features (which added NOISE — t1805 with the full set scored below the shared-feature
    # ensemble). Keeps the strong never-worked separators (range_pos/recov/63d-low) + divergence
    # + up-days/higher-lows, dropping the zigzag-structure noise.
    "entry_reversal_lean": (
        "entry_reversal_confirm minus zigzag-structure (lean reversal separators only)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "dist_10d_low",
            "dist_63d_low",
            "range_pos_20",
            "recov_setup",
            "macd_hist_chg_5",
            "rsi_div_20",
            "macd_div_20",
            "up_days_10",
            "higher_lows_count",
        ],
    ),
    # R44: stack the three R43 winners (lvup126 comp 138.2 + lvnhrev's reversal leg) — combine
    # the best-OOS quality composites with the high-ic5 volume leg in one entry head.
    "entry_lvup126rev_lean": (
        "entry_lvup126_lean + comp_lv_nh_rev (stack all R43 winners)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "comp_lv_nh_rev",
        ],
    ),
    # R47: add a 12-1 momentum dimension to the champion entry (entry_lvup126_lean).
    "entry_lvup126tri_lean": (
        "entry_lvup126_lean + comp_tri_and (momentum-gated triple)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "comp_tri_and",
        ],
    ),
    # R47: swap the 126d leg for the momentum-gated triple (replacement, not stack).
    "entry_lvup_tri_lean": (
        "entry_lvnh_lean + comp_lowvol_uptrend + comp_tri_and (momentum instead of 126)",
        _LEADING_V2
        + ["lowvol_rank_60", "nearhigh_rank", "comp_nh_lv", "comp_lowvol_uptrend", "comp_tri_and"],
    ),
    # R47: vol-weighted additive composite added to the champion.
    "entry_lvup126_v2h1_lean": (
        "entry_lvup126_lean + comp_v2h1 (vol-weighted additive)",
        _LEADING_V2
        + [
            "lowvol_rank_60",
            "nearhigh_rank",
            "comp_nh_lv",
            "comp_lowvol_uptrend",
            "comp_nh_x_lv126",
            "comp_v2h1",
        ],
    ),
    "entry_lvnh_mom": (
        "lvnh lean + momentum_rank (research 3-factor comp_lv_nh_mom)",
        _LEADING_V2 + ["lowvol_rank_60", "nearhigh_rank", "comp_nh_lv", "momentum_rank"],
    ),
    # Minimal rule set for a breakout entry (price breaks 20d-high + volume confirm).
    "rule_breakout_v1": (
        "breakout_20 + volume_ratio_20 for the breakout rule entry (2)",
        ["breakout_20", "volume_ratio_20"],
    ),
    "exit_peak_div_ext": (
        "exit_peak_div + dist_52w_low + dist_126d_low (clean extension-from-base) (39)",
        [
            c
            for c in (
                _EXIT_PEAK_CORE
                + _EXIT_PEAK_V2_EXTRA
                + _HA
                + ["upper_wick_ratio", "lower_wick_ratio", "body_ratio"]
            )
            if c not in _EXIT_DEAD
        ]
        + [
            "adx_14",
            "minus_di_14",
            "div_rank_rsi",
            "div_rank_macd",
            "dist_52w_low",
            "dist_126d_low",
        ],
    ),
}

# MEAN-REV-AS-FEATURE (2026-07-11): ft_rs entry (cross-sectional RS) + cross-sectional oversold
# ranks — let the entry ML pick oversold-bounce names IN ONE BOOK (a separate mean-rev sleeve
# failed on slot-displacement; cross-sectional ranks survive z-scoring). Fills momentum-dead years.
SETS["entry_recov_rs_mr"] = (
    "entry_recov_rs + cross-sectional oversold ranks (osold_rank_20, osold_rank_ret5)",
    list(SETS["entry_recov_rs"][1]) + ["osold_rank_20", "osold_rank_ret5"],
)

# SMAC round-2 niches (2026-06-20): extend the v20_sec winning stack (RS+dynamics+sector)
# with one more orthogonal niche each (DRY — reuse its column list, no retyping/typos).
_V20SEC = list(SETS["entry_dyn_sec"][1])
SETS["entry_sec_pa"] = (
    "v20_sec + price-action reversal (bull_engulf, lower_wick_ratio, upper_wick_5, body_ratio, doji_count_10)",
    _V20SEC + ["bull_engulf", "lower_wick_ratio", "upper_wick_5", "body_ratio", "doji_count_10"],
)
SETS["entry_sec_mhr"] = (
    "v20_sec + multi-horizon range position (dist_63d/126d high & low)",
    _V20SEC + ["dist_63d_high", "dist_63d_low", "dist_126d_high", "dist_126d_low"],
)
SETS["entry_sec_liq"] = (
    "v20_sec + liquidity/turnover (volume_rank_20d, volume_stability, price_level)",
    _V20SEC + ["volume_rank_20d", "volume_stability", "price_level"],
)
SETS["entry_sec_rsdyn"] = (
    "v20_sec + RS DYNAMICS (multi-horizon RS, RS spread/expansion, RS slope/accel, EMA-RS)",
    _V20SEC
    + [
        "rs_rank_5",
        "rs_rank_60",
        "rs_spread_5_60",
        "rs_spread_5_20",
        "rs_trend_5",
        "rs_trend_20",
        "rs_accel",
        "rs_ema5",
        "rs_vs_ema5",
    ],
)
# De-diluted: only the NOVEL RS-dynamics (multi-period RS expansion + RS-vs-its-EMA), the user's
# 'co giãn RS nhiều chu kỳ' — distinct from cs_rank_trend (already the Δ10 RS slope).
SETS["entry_sec_rsexp"] = (
    "v20_sec + RS multi-period expansion + RS-vs-EMA (de-diluted)",
    _V20SEC + ["rs_spread_5_60", "rs_spread_5_20", "rs_vs_ema5"],
)
# High-trade immediate-bleeder fix: structure-break + downtrend-persistence features so the
# model can avoid the 88%-of-losers no-bounce bottoms (they break support / make lower highs).
SETS["entry_dyn_sec_struct"] = (
    "v20_sec + structure (broke_support_20, lower_high_20, down_days_10, higher_lows_count)",
    _V20SEC + ["broke_support_20", "lower_high_20", "down_days_10", "higher_lows_count"],
)
# Round-3 fresh niches on v20_sec.
SETS["entry_sec_ha"] = (
    "v20_sec + Heiken-Ashi smoothed reversal state (ha_color, ha_trend, ha_body)",
    _V20SEC + ["ha_color", "ha_trend", "ha_body"],
)
SETS["entry_sec_mkt"] = (
    "v20_sec + market regime at entry (market_trend, market_volatility_regime)",
    _V20SEC + ["market_trend", "market_volatility_regime"],
)
SETS["entry_sec_secdyn"] = (
    "v20_sec + sector-RS DYNAMICS (gaining/losing strength within sector = rotation)",
    _V20SEC + ["return_vs_sector_slope5", "strength_vs_sector_slope5"],
)

# RAW per-bar primitives for the SEQUENCE model (v49): deliberately NOT pre-engineered
# slopes/spreads/ranks — a recurrent net should learn its own temporal dynamics from the
# raw bar shape (returns, intrabar wick/body, volume, vol, level). Tests whether the
# GRU-on-engineered-features failure (v48) was a wrong-input artifact vs a real no-signal.
SETS["seq_raw"] = (
    "raw per-bar primitives for sequence models (returns/shape/volume/vol/level)",
    [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "close_to_open",
        "high_low_pct",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "body_ratio",
        "volume_ratio_5",
        "volume_ratio_20",
        "atr_14_ratio",
        "realized_vol_10",
        "rsi_14",
        "bb_pct_20",
        "dist_52w_high",
        "dist_52w_low",
    ],
)
# seq_raw + cross-sectional/sector CONTEXT (ranks + sector-relative LEVELS, the info the
# lightgbm snapshot has) but deliberately NO pre-engineered slopes/spreads (those HURT the
# GRU in v48). Tests whether matching lightgbm's information closes the GRU-vs-snapshot gap
# (i.e. is the v49 gap architecture, or just missing cross-sectional inputs?).
SETS["seq_rich"] = (
    "seq_raw + cross-sectional/sector context (ranks + relative levels, no slopes)",
    SETS["seq_raw"][1]
    + [
        "rs_rank_5",
        "rs_rank_60",
        "momentum_rank",
        "cs_rank_trend",
        "price_strength_rank",
        "nearhigh_rank",
        "lowvol_rank_60",
        "range_pos_20",
        "dist_10d_low",
        "return_vs_sector",
        "strength_vs_sector",
        "momentum_vs_sector",
    ],
)


def set_members(name: str) -> list[str]:
    """Ordered member feature names for a set name (raises on unknown set)."""
    if name not in SETS:
        raise KeyError(f"Unknown feature set '{name}'. Available: {sorted(SETS)}")
    return list(SETS[name][1])


def compute_expr_hashes(features: dict[str, str] | None = None) -> dict[str, str]:
    """expr_hash for each feature, folding in recursive dependency hashes.

    Changing an upstream feature's formula flips every downstream hash, so the
    content-addressed store invalidates correctly.
    """
    features = features if features is not None else FEATURES
    cache: dict[str, str] = {}
    in_progress: set[str] = set()

    def _h(name: str) -> str:
        if name in cache:
            return cache[name]
        if name in in_progress:
            raise ValueError(f"Cyclic feature dependency at '{name}'")
        in_progress.add(name)
        node = parse(features[name])
        refs, _ = extract_deps(node)
        dep_hashes = [_h(r) for r in sorted(refs) if r in features]
        in_progress.discard(name)
        cache[name] = expr_hash(node, dep_hashes=dep_hashes)
        return cache[name]

    for name in features:
        _h(name)
    return cache
