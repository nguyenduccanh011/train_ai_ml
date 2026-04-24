import numpy as np
from collections import defaultdict

from .defaults import (
    DEFAULT_PARAMS, FEATURE_DEFAULTS, RULE_PRIORITY_SYMBOLS, SCORE5_RISKY_SYMBOLS,
    SYMBOL_PROFILES,
)
from .indicators import compute_indicators, detect_trend_strength, get_regime_adapter


def backtest_unified(y_pred, returns, df_test, feature_cols, y_pred_exit=None, **config):
    cfg = {**DEFAULT_PARAMS, **config}

    initial_capital = cfg["initial_capital"]
    commission = cfg["commission"]
    tax = cfg["tax"]
    record_trades = cfg["record_trades"]
    mod_a = cfg["mod_a"]; mod_b = cfg["mod_b"]; mod_c = cfg["mod_c"]; mod_d = cfg["mod_d"]
    mod_e = cfg["mod_e"]; mod_f = cfg["mod_f"]; mod_g = cfg["mod_g"]; mod_h = cfg["mod_h"]
    mod_i = cfg["mod_i"]; mod_j = cfg["mod_j"]

    fast_exit_strong = cfg["fast_exit_strong"]
    fast_exit_moderate = cfg["fast_exit_moderate"]
    fast_exit_weak = cfg["fast_exit_weak"]
    fast_exit_hb_buffer = cfg["fast_exit_hb_buffer"]
    peak_protect_strong_threshold = cfg["peak_protect_strong_threshold"]
    peak_protect_normal_threshold = cfg["peak_protect_normal_threshold"]
    hard_cap_weak = cfg["hard_cap_weak"]
    hard_cap_moderate_mult = cfg["hard_cap_moderate_mult"]
    hard_cap_strong_mult = cfg["hard_cap_strong_mult"]
    hard_cap_strong_floor = cfg["hard_cap_strong_floor"]
    hard_cap_moderate_floor = cfg["hard_cap_moderate_floor"]
    time_decay_bars = cfg["time_decay_bars"]
    time_decay_mult = cfg["time_decay_mult"]

    v22_mode = cfg["v22_mode"]

    patch_smart_hardcap = cfg["patch_smart_hardcap"]
    patch_pp_restore = cfg["patch_pp_restore"]
    patch_long_horizon = cfg["patch_long_horizon"]
    patch_symbol_tuning = cfg["patch_symbol_tuning"]
    patch_rule_ensemble = cfg["patch_rule_ensemble"]
    patch_noise_filter = cfg["patch_noise_filter"]
    patch_adaptive_hardcap = cfg["patch_adaptive_hardcap"]
    patch_pp_2of3 = cfg["patch_pp_2of3"]

    v26_wider_hardcap = cfg["v26_wider_hardcap"]
    v26_relaxed_entry = cfg["v26_relaxed_entry"]
    v26_skip_choppy = cfg["v26_skip_choppy"]
    v26_extended_hold = cfg["v26_extended_hold"]
    v26_strong_rule_ensemble = cfg["v26_strong_rule_ensemble"]
    v26_min_position = cfg["v26_min_position"]
    v26_score5_penalty = cfg["v26_score5_penalty"]
    v26_hardcap_confirm_strong = cfg["v26_hardcap_confirm_strong"]

    v27_selective_choppy = cfg["v27_selective_choppy"]
    v27_hardcap_two_step = cfg["v27_hardcap_two_step"]
    v27_rule_priority = cfg["v27_rule_priority"]
    v27_dynamic_score5_penalty = cfg["v27_dynamic_score5_penalty"]
    v27_trend_persistence_hold = cfg["v27_trend_persistence_hold"]

    v28_early_wave_filter = cfg["v28_early_wave_filter"]
    v28_crash_guard = cfg["v28_crash_guard"]
    v28_wave_acceleration_entry = cfg["v28_wave_acceleration_entry"]
    v28_early_loss_cut = cfg["v28_early_loss_cut"]
    v28_cycle_peak_exit = cfg["v28_cycle_peak_exit"]
    v28_early_loss_cut_threshold = cfg["v28_early_loss_cut_threshold"]
    v28_early_loss_cut_days = cfg["v28_early_loss_cut_days"]

    v29_adaptive_peak_lock = cfg["v29_adaptive_peak_lock"]
    v29_apl_trigger = cfg["v29_adaptive_peak_lock_trigger"]
    v29_apl_keep = cfg["v29_adaptive_peak_lock_keep"]
    v29_atr_velocity_exit = cfg["v29_atr_velocity_exit"]
    v29_atr_velocity_k = cfg["v29_atr_velocity_k"]
    v29_atr_velocity_min_profit = cfg["v29_atr_velocity_min_profit"]
    v29_tighter_trail_high_profit = cfg["v29_tighter_trail_high_profit"]
    v29_high_profit_trigger = cfg["v29_high_profit_trigger"]
    v29_high_profit_trail = cfg["v29_high_profit_trail"]
    v29_reversal_after_peak = cfg["v29_reversal_after_peak"]
    v29_reversal_peak_trigger = cfg["v29_reversal_peak_trigger"]
    v29_reversal_ret2_threshold = cfg["v29_reversal_ret2_threshold"]
    v29_breakout_strength_entry = cfg["v29_breakout_strength_entry"]
    v29_relstrength_filter = cfg["v29_relstrength_filter"]
    v29_rs_ret20_threshold = cfg["v29_rs_ret20_threshold"]
    v29_peak_lock_high_beta_only = cfg["v29_peak_lock_high_beta_only"]
    v29_profit_safety_net = cfg["v29_profit_safety_net"]
    v29_profit_safety_trigger = cfg["v29_profit_safety_trigger"]
    v29_hardcap_after_peak = cfg["v29_hardcap_after_peak"]
    v29_hardcap_after_peak_trigger = cfg["v29_hardcap_after_peak_trigger"]
    v29_hardcap_after_peak_floor = cfg["v29_hardcap_after_peak_floor"]

    # V30 flags
    v30_peak_proximity_filter  = cfg["v30_peak_proximity_filter"]
    v30_peak_prox_dist_threshold = cfg["v30_peak_prox_dist_threshold"]
    v30_peak_prox_rally10_min  = cfg["v30_peak_prox_rally10_min"]
    v30_rally_extension_filter = cfg["v30_rally_extension_filter"]
    v30_rally10_hard_block     = cfg["v30_rally10_hard_block"]
    v30_rally20_hard_block     = cfg["v30_rally20_hard_block"]
    v30_pullback_only_entry    = cfg["v30_pullback_only_entry"]
    v30_pullback_min_pct       = cfg["v30_pullback_min_pct"]
    v30_rally_position_scaling = cfg["v30_rally_position_scaling"]
    v30_rps_tier1_rally        = cfg["v30_rps_tier1_rally"]
    v30_rps_tier2_rally        = cfg["v30_rps_tier2_rally"]
    v30_signal_exit_defer      = cfg["v30_signal_exit_defer"]
    v30_sed_defer_bars         = cfg["v30_sed_defer_bars"]
    v30_sed_min_cum_ret        = cfg["v30_sed_min_cum_ret"]
    v30_momentum_hold_override = cfg["v30_momentum_hold_override"]
    v30_mho_min_profit         = cfg["v30_mho_min_profit"]
    v30_mho_rsi_max            = cfg["v30_mho_rsi_max"]
    v30_chandelier_trail       = cfg["v30_chandelier_trail"]
    v30_chand_atr_mult         = cfg["v30_chand_atr_mult"]
    v30_chand_profit_trigger   = cfg["v30_chand_profit_trigger"]
    v30_atr_aware_hardcap      = cfg["v30_atr_aware_hardcap"]
    v30_atr_hc_mult            = cfg["v30_atr_hc_mult"]
    v30_atr_hc_floor           = cfg["v30_atr_hc_floor"]
    v30_atr_hc_ceiling         = cfg["v30_atr_hc_ceiling"]
    v30_hardcap_two_step_v2    = cfg["v30_hardcap_two_step_v2"]
    v30_hc2_step1_loss         = cfg["v30_hc2_step1_loss"]
    v30_hc2_step2_loss         = cfg["v30_hc2_step2_loss"]
    v30_regime_aware_hardcap   = cfg["v30_regime_aware_hardcap"]
    v30_rah_choppy_cap         = cfg["v30_rah_choppy_cap"]
    v30_rah_trending_cap       = cfg["v30_rah_trending_cap"]

    # V31 flags
    v31_peak_chasing_guard     = cfg.get("v31_peak_chasing_guard", False)
    v31_pcg_ret5d_thresh       = cfg.get("v31_pcg_ret5d_thresh", 0.08)
    v31_pcg_dist_thresh        = cfg.get("v31_pcg_dist_thresh", 10.0)
    v31_pcg_action             = cfg.get("v31_pcg_action", "half_size")  # "half_size" | "skip"
    v31_adaptive_defer         = cfg.get("v31_adaptive_defer", False)
    v31_ad_min_cum_ret         = cfg.get("v31_ad_min_cum_ret", 0.02)
    v31_ad_max_bars            = cfg.get("v31_ad_max_bars", 7)
    v31_ad_use_ema_confirm     = cfg.get("v31_ad_use_ema_confirm", True)
    v31_hardcap_after_profit   = cfg.get("v31_hardcap_after_profit", False)
    v31_hap_profit_trigger     = cfg.get("v31_hap_profit_trigger", 0.05)
    v31_hap_floor              = cfg.get("v31_hap_floor", -0.03)
    v31_profile_sizing         = cfg.get("v31_profile_sizing", False)
    v31_ps_momentum_mult       = cfg.get("v31_ps_momentum_mult", 1.4)
    v31_ps_highbeta_mult       = cfg.get("v31_ps_highbeta_mult", 1.2)
    v31_ps_defensive_mult      = cfg.get("v31_ps_defensive_mult", 0.75)
    v31_ps_bank_mult           = cfg.get("v31_ps_bank_mult", 0.85)
    v31_short_hold_exit_filter = cfg.get("v31_short_hold_exit_filter", False)
    v31_shef_min_hold          = cfg.get("v31_shef_min_hold", 10)
    v31_shef_min_pnl           = cfg.get("v31_shef_min_pnl", -0.03)
    v31_enriched_log           = cfg.get("v31_enriched_log", False)

    # V32 flags
    # A: HAP preempt — thoát TRƯỚC hard_cap nếu đã từng có lãi >= trigger rồi rớt về floor
    v32_hap_preempt            = cfg.get("v32_hap_preempt", False)
    v32_hap_pre_trigger        = cfg.get("v32_hap_pre_trigger", 0.05)   # max_profit >= 5%
    v32_hap_pre_floor          = cfg.get("v32_hap_pre_floor", -0.03)    # cur_ret <= -3%
    # B: Trend-weak oversold exit — khi trend=weak + dist_sma20 < threshold + còn lãi → exit sớm
    v32_weak_oversold_exit     = cfg.get("v32_weak_oversold_exit", False)
    v32_woe_dist_thresh        = cfg.get("v32_woe_dist_thresh", -0.07)  # dist_sma20 < -7%
    v32_woe_min_profit         = cfg.get("v32_woe_min_profit", 0.0)     # only when pnl > 0
    v32_woe_hold_min           = cfg.get("v32_woe_hold_min", 5)         # hold >= 5 days
    # C: Dynamic hard_cap tighten — khi dist_sma20 < -8% thì cap floor lỏng hơn không còn giá trị
    v32_dynamic_hc_dist        = cfg.get("v32_dynamic_hc_dist", False)
    v32_dhc_dist_thresh        = cfg.get("v32_dhc_dist_thresh", -0.08)  # dist_sma20 < -8%
    v32_dhc_tight_cap          = cfg.get("v32_dhc_tight_cap", -0.07)    # exit at -7% nếu thỏa mãn
    # D: Profit ratchet exit — khi đạt max_profit >= trigger rồi rớt về keep * max_profit
    v32_profit_ratchet         = cfg.get("v32_profit_ratchet", False)
    v32_pr_trigger             = cfg.get("v32_pr_trigger", 0.08)        # max_profit >= 8%
    v32_pr_keep                = cfg.get("v32_pr_keep", 0.30)           # sàn = 30% của max_profit
    # E: Signal exit weak-trend early — khi signal + trend=weak + dist_sma20 < -5% → không block
    v32_signal_weak_exit       = cfg.get("v32_signal_weak_exit", False)
    v32_swe_dist_thresh        = cfg.get("v32_swe_dist_thresh", -0.05)  # dist_sma20 < -5%

    # V33 flags
    # A: Multi-tier trailing ratchet — tighten floor progressively as max_profit grows
    v33_trailing_ratchet       = cfg.get("v33_trailing_ratchet", False)
    v33_tr_tier1_trigger       = cfg.get("v33_tr_tier1_trigger", 0.12)  # >= 12%: floor = tier1_keep * max_profit
    v33_tr_tier1_keep          = cfg.get("v33_tr_tier1_keep", 0.40)
    v33_tr_tier2_trigger       = cfg.get("v33_tr_tier2_trigger", 0.20)  # >= 20%: floor = tier2_keep * max_profit
    v33_tr_tier2_keep          = cfg.get("v33_tr_tier2_keep", 0.55)
    v33_tr_tier3_trigger       = cfg.get("v33_tr_tier3_trigger", 0.35)  # >= 35%: floor = tier3_keep * max_profit
    v33_tr_tier3_keep          = cfg.get("v33_tr_tier3_keep", 0.65)

    # B: Trend-reversal exit — khi max_profit>threshold VÀ close<ema8 VÀ rsi thấp → thoát ngay
    v33_trend_rev_exit         = cfg.get("v33_trend_rev_exit", False)
    v33_tre_min_profit         = cfg.get("v33_tre_min_profit", 0.08)    # chỉ sau khi đạt >= 8% profit
    v33_tre_rsi_thresh         = cfg.get("v33_tre_rsi_thresh", 50.0)    # rsi14 < 50
    v33_tre_hold_min           = cfg.get("v33_tre_hold_min", 5)         # hold >= 5 ngày

    # C: Recovery-peak entry filter — block khi giá đã tăng nhanh + dist_sma20 vẫn dương mà trend không mạnh
    v33_recovery_peak_filter   = cfg.get("v33_recovery_peak_filter", False)
    v33_rpf_ret10_thresh       = cfg.get("v33_rpf_ret10_thresh", 0.12)  # ret 10 ngày > 12%
    v33_rpf_dist_sma20_thresh  = cfg.get("v33_rpf_dist_sma20_thresh", 0.03)  # dist_sma20 > 3%
    v33_rpf_require_weak       = cfg.get("v33_rpf_require_weak", True)  # chỉ block nếu trend != strong

    # D: HAP consecutive drop — HAP preempt chỉ trigger sau N ngày giảm liên tiếp (tránh bán đáy spike)
    v33_hap_consec_drop        = cfg.get("v33_hap_consec_drop", False)
    v33_hcd_min_days           = cfg.get("v33_hcd_min_days", 2)         # phải giảm ít nhất 2 ngày liên tiếp

    # E: RSI oversold exit block — không thoát bởi signal/hap khi rsi đang oversold (đáy điều chỉnh)
    v33_rsi_oversold_block     = cfg.get("v33_rsi_oversold_block", False)
    v33_rob_rsi_thresh         = cfg.get("v33_rob_rsi_thresh", 32.0)    # block nếu rsi14 < 32 (oversold)
    v33_rob_max_hold           = cfg.get("v33_rob_max_hold", 5)         # chỉ block trong N ngày đầu oversold

    # F: Signal exit confirm — require 2 consecutive signal bars to exit (giảm false exits)
    v33_signal_confirm_exit    = cfg.get("v33_signal_confirm_exit", False)
    v33_sce_min_pnl            = cfg.get("v33_sce_min_pnl", -0.02)     # chỉ block nếu pnl > -2%
    v33_sce_min_profit_seen    = cfg.get("v33_sce_min_profit_seen", 0.03)  # và đã từng có profit >= 3%
    v33_sce_max_hold           = cfg.get("v33_sce_max_hold", 90)        # không block khi hold >= 90d (tránh zombie)

    # V35 flags
    v35_relax_cooldown         = cfg.get("v35_relax_cooldown", False)
    v35_cooldown_after_big_loss = cfg.get("v35_cooldown_after_big_loss", 1)
    v35_cooldown_after_loss    = cfg.get("v35_cooldown_after_loss", 0)
    v35_skip_price_proximity   = cfg.get("v35_skip_price_proximity", False)
    v35_single_bar_signal      = cfg.get("v35_single_bar_signal", False)
    v35_rule_override          = cfg.get("v35_rule_override", False)
    v35_rule_override_min_score = cfg.get("v35_rule_override_min_score", 1)
    v35_hybrid_entry           = cfg.get("v35_hybrid_entry", False)
    v35_hybrid_size            = cfg.get("v35_hybrid_size", 0.5)

    # === V38 flags ===
    # V38b: stall-exit — exit khi giu N ngay ma chua dat dc profit threshold
    v38b_stall_exit            = cfg.get("v38b_stall_exit", False)
    v38b_stall_min_hold        = cfg.get("v38b_stall_min_hold", 6)
    v38b_stall_max_profit      = cfg.get("v38b_stall_max_profit", 0.02)  # max_price_profit < 2%
    v38b_stall_pnl_thresh      = cfg.get("v38b_stall_pnl_thresh", -0.02) # cur_ret < -2%
    # V38c: HA-driven exit — bearish_reversal_signal hoac late_wave + body_shrinking + cur_ret<0
    v38c_ha_exit               = cfg.get("v38c_ha_exit", False)
    v38c_ha_min_hold           = cfg.get("v38c_ha_min_hold", 3)
    v38c_ha_pnl_thresh         = cfg.get("v38c_ha_pnl_thresh", 0.0)     # exit khi cur_ret < 0%
    # V38d: anti-fomo entry filter
    v38d_fomo_filter           = cfg.get("v38d_fomo_filter", False)
    v38d_fomo_ret5d_thresh     = cfg.get("v38d_fomo_ret5d_thresh", 0.06)   # entry_ret_5d > 6% block
    v38d_fomo_dist_thresh      = cfg.get("v38d_fomo_dist_thresh", 0.06)    # dist_sma20 > 6% block
    # V38d: rule co-pilot exit — neu rule_signal=False sau N ngay giu va co loi -> chot
    v38d_copilot_exit          = cfg.get("v38d_copilot_exit", False)
    v38d_copilot_min_hold      = cfg.get("v38d_copilot_min_hold", 4)
    v38d_copilot_min_profit    = cfg.get("v38d_copilot_min_profit", 0.03) # da co profit >=3%

    # === V39 flags ===
    # V39a: signal exit defer mở rộng — signal exit chỉ kích hoạt sau min_hold ngày
    # Giải quyết bucket 21-30d WR=12% (signal exit quá sớm với early_wave fw=8d target)
    v39a_signal_exit_min_hold  = cfg.get("v39a_signal_exit_min_hold", 0)  # 0 = off

    # V39a2: kết hợp thêm rule confirm — signal exit cần MACD<0 AND Close<MA20
    v39a_rule_confirm_exit     = cfg.get("v39a_rule_confirm_exit", False)
    # V39g: chỉ defer khi price_max_profit đã đạt threshold (tránh defer trades stall)
    v39g_rule_confirm_min_maxprofit = cfg.get("v39g_rule_confirm_min_maxprofit", 0.0)

    # V39b: HAP reform — raise trigger + require min_hold trước khi HAP active
    # Giải quyết 87 trades HAP preempt 100% lỗ do bắt đầu sóng quá sớm
    v39b_hap_min_hold          = cfg.get("v39b_hap_min_hold", 0)          # 0 = off (tương thích ngược)
    v39b_hap_trigger           = cfg.get("v39b_hap_trigger", None)        # None = dùng v32_hap_pre_trigger

    # V39d: per-symbol rule-exit hybrid — các mã stable-trend dùng rule exit thay signal exit
    v39d_rule_exit_symbols     = cfg.get("v39d_rule_exit_symbols", set())  # set of symbols

    # --- Indicators ---
    ind = compute_indicators(df_test, mod_e=mod_e)
    n = ind["n"]
    close = ind["close"]; opn = ind["opn"]; high = ind["high"]; low = ind["low"]; volume = ind["volume"]
    sma10 = ind["sma10"]; sma20 = ind["sma20"]; sma50 = ind["sma50"]; sma100 = ind["sma100"]
    ema8 = ind["ema8"]; macd_line = ind["macd_line"]; macd_hist = ind["macd_hist"]
    atr14 = ind["atr14"]; avg_vol20 = ind["avg_vol20"]
    local_low_20 = ind["local_low_20"]
    ret_2d = ind["ret_2d"]; ret_3d = ind["ret_3d"]
    ret_5d = ind["ret_5d"]; ret_20d = ind["ret_20d"]; ret_60d = ind["ret_60d"]
    ret_acceleration = ind["ret_acceleration"]; days_since_low_10 = ind["days_since_low_10"]
    dist_sma20 = ind["dist_sma20"]
    drop_from_peak_20 = ind["drop_from_peak_20"]; rsi14 = ind["rsi14"]
    stabilized_sideways = ind["stabilized_sideways"]
    consolidation_breakout = ind["consolidation_breakout"]
    secondary_breakout = ind["secondary_breakout"]
    vshape_bypass = ind["vshape_bypass"]
    days_above_ma20 = ind["days_above_ma20"]
    days_above_sma50 = ind["days_above_sma50"]
    rule_signal = ind["rule_signal"]; rule_consecutive = ind["rule_consecutive"]
    dist_from_52w_high = ind["dist_from_52w_high"]
    dates = ind["dates"]; symbols = ind["symbols"]
    feat_arrays = ind["feat_arrays"]

    # V38c: extract HA columns from df_test if available (leading_v4)
    _HA_COLS = ["ha_bearish_reversal_signal", "ha_late_wave",
                "ha_upper_shadow_growing", "ha_body_shrinking",
                "ha_green_streak"]
    ha_arrays = {}
    for c in _HA_COLS:
        if c in df_test.columns:
            arr = df_test[c].values.astype(float)
            arr = np.where(np.isnan(arr), 0.0, arr)
            ha_arrays[c] = arr
        else:
            ha_arrays[c] = np.zeros(n)

    MODEL_B_MIN_HOLD = cfg.get("model_b_min_hold", 3)

    # --- State ---
    equity = np.zeros(n)
    equity[0] = initial_capital
    position = 0
    trades = []
    current_entry_day = 0
    entry_equity = 0
    max_equity_in_trade = 0
    max_price_in_trade = 0
    hold_days = 0
    position_size = 1.0
    consecutive_exit_signals = 0
    consecutive_below_ema8 = 0

    hard_cap_pending_bars = 0
    HARD_CAP_CONFIRM_STRONG = 2 if v26_hardcap_confirm_strong else 1
    HARD_CAP_CONFIRM_MODERATE = 1
    HARD_CAP_CONFIRM_WEAK = 0

    pp_pending_bars = 0

    MIN_HOLD = 6
    ZOMBIE_BARS = 14
    PROFIT_LOCK_THRESHOLD = 0.12
    PROFIT_LOCK_MIN = 0.06
    HARD_STOP = 0.08
    ATR_MULT = 1.8
    COOLDOWN_AFTER_BIG_LOSS = 5
    QUICK_REENTRY_WINDOW = 3
    STRONG_TREND_TRAIL_MULT = 0.45
    V22_SIGNAL_HARD_CAP = 0.12

    cooldown_remaining = 0
    last_exit_price = 0
    last_exit_reason = ""
    last_exit_bar = -999
    entry_close = 0
    v30_hc2_halved = False       # track if we already halved position (C2 two-step)
    v30_defer_bars_remaining = 0  # bars left to defer signal exit (B1)

    # V31 state
    v31_defer_bars_remaining = 0  # adaptive defer counter

    # V33 state
    v33_consec_below_ema8 = 0     # consecutive days close < ema8 (for trend_rev_exit)
    v33_prev_signal_exit = False  # last bar also had signal exit (for signal_confirm_exit)

    def gf(name, idx):
        return feat_arrays[name][idx] if idx < n else FEATURE_DEFAULTS.get(name, 0)

    entry_features = {}
    counters = defaultdict(int)

    for i in range(1, n):
        pred = int(y_pred[i - 1])
        ret = returns[i] if not np.isnan(returns[i]) else 0
        raw_signal = 1 if pred == 1 else 0
        new_position = raw_signal
        exit_reason = "signal"

        wp = gf("range_position_20d", i)
        dp = gf("dist_to_resistance", i)
        rs = gf("rsi_slope_5d", i)
        vs = gf("vol_surge_ratio", i)
        bs = gf("breakout_setup_score", i)
        hl = gf("higher_lows_count", i)
        od = gf("obv_price_divergence", i)
        bb = gf("bb_width_percentile", i)

        trend = detect_trend_strength(i, ind)
        regime_cfg = get_regime_adapter(i, trend, ind, patch_symbol_tuning=patch_symbol_tuning)
        dp_floor = regime_cfg["dp_floor"]
        ret5_hot = regime_cfg["ret5_hot"]
        sym = str(symbols[i]) if i < n else "?"
        profile = SYMBOL_PROFILES.get(sym, "balanced")

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # V35: rule trigger (used by hybrid entry / rule_override below)
        rule_trigger_now = (
            i >= 26
            and not np.isnan(macd_hist[i]) and macd_hist[i] > 0
            and not np.isnan(sma20[i]) and close[i] > sma20[i]
            and close[i] > opn[i]
        )

        # === ENTRY LOGIC ===
        quick_reentry = False
        breakout_entry = False
        vshape_entry = False
        v35_hybrid_now = False  # set True when entry triggered by V35 hybrid path

        if new_position == 0 and position == 0 and last_exit_reason == "trailing_stop":
            bars_since_exit = i - last_exit_bar
            if (bars_since_exit <= QUICK_REENTRY_WINDOW and trend in ("strong", "moderate") and
                macd_line[i] > 0 and not np.isnan(sma20[i]) and close[i] > sma20[i]):
                new_position = 1; quick_reentry = True

        bo_quality_ok = True
        if mod_f:
            bo_quality_ok = (macd_hist[i] > 0 and close[i] > opn[i] and
                            not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])

        if new_position == 0 and position == 0 and consolidation_breakout[i] and bo_quality_ok:
            new_position = 1; breakout_entry = True

        if mod_e and new_position == 0 and position == 0 and secondary_breakout[i] and bo_quality_ok:
            new_position = 1; breakout_entry = True; counters["secondary_bo"] += 1

        if mod_a and new_position == 0 and position == 0 and vshape_bypass[i]:
            if not np.isnan(ema8[i]) and close[i] >= ema8[i] * 0.99:
                new_position = 1; vshape_entry = True; counters["vshape"] += 1

        # V35c: hybrid entry — fire when ML didn't but rule trigger fires
        if (v35_hybrid_entry and new_position == 0 and position == 0
                and rule_trigger_now and cooldown_remaining == 0):
            new_position = 1
            v35_hybrid_now = True

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            if cooldown_remaining > 0:
                new_position = 0
        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            if last_exit_price > 0 and last_exit_reason != "trailing_stop":
                if abs(close[i] / last_exit_price - 1) < 0.03:
                    if not (v35_skip_price_proximity and rule_trigger_now) and not v35_hybrid_now:
                        new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry and not breakout_entry and not vshape_entry and not v35_hybrid_now:
            prev_pred = int(y_pred[i - 2]) if i >= 2 else 0
            if bs >= 4 and vs > 1.2: pass
            elif trend == "strong" and rs > 0: pass
            elif v26_relaxed_entry and trend == "strong" and rule_consecutive[i] >= 3:
                pass
            elif v35_single_bar_signal and rule_trigger_now:
                pass
            elif prev_pred != 1: new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3 and not breakout_entry: new_position = 0

        strong_breakout_context = (trend == "strong" and (bs >= 3 or vs > 1.5 or breakout_entry))
        entry_alpha_ok = True
        entry_score = 0
        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            near_sma_support = (not np.isnan(sma20[i]) and close[i] <= sma20[i] * 1.02 and close[i] >= sma20[i] * 0.97)
            near_local_low = (not np.isnan(local_low_20[i]) and close[i] <= local_low_20[i] * 1.05)
            in_uptrend_macro = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] > sma50[i])
            if trend == "strong": min_score = 1
            elif (near_sma_support or near_local_low) and in_uptrend_macro: min_score = 2
            elif in_uptrend_macro and rs > 0: min_score = 2
            else: min_score = 3
            if entry_score < min_score and not breakout_entry: entry_alpha_ok = False
            if wp > 0.9 and rs <= 0 and bs < 2 and trend != "strong" and not breakout_entry: entry_alpha_ok = False
            if bb > 0.85 and bs < 2 and entry_score < 4 and trend != "strong" and not breakout_entry: entry_alpha_ok = False
            if entry_alpha_ok and wp > 0.78 and bb < 0.35 and trend == "weak" and not breakout_entry: entry_alpha_ok = False
            if entry_alpha_ok and dp < dp_floor:
                if entry_score < 4 and not strong_breakout_context: entry_alpha_ok = False

        if new_position == 1 and position == 0 and not vshape_entry:
            if ret_5d[i] > ret5_hot and not strong_breakout_context:
                if not (v35_rule_override and rule_trigger_now and entry_score >= v35_rule_override_min_score) and not v35_hybrid_now:
                    entry_alpha_ok = False
        if new_position == 1 and position == 0 and not vshape_entry and entry_alpha_ok:
            if drop_from_peak_20[i] <= -0.15 and not stabilized_sideways[i]: entry_alpha_ok = False
        if new_position == 1 and position == 0 and entry_alpha_ok:
            vol_floor = 0.7 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else 0
            if vol_floor > 0 and volume[i] < vol_floor: entry_alpha_ok = False
        if mod_g and new_position == 1 and position == 0 and not vshape_entry and entry_alpha_ok:
            if (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] < sma50[i] and
                close[i] < sma50[i] and ret_60d[i] < -0.10):
                entry_alpha_ok = False; counters["bear_blocked"] += 1
        if mod_j and new_position == 1 and position == 0 and not vshape_entry and not breakout_entry and entry_alpha_ok:
            if (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and
                abs(sma20[i] / sma50[i] - 1) < 0.02 and abs(ret_20d[i]) < 0.06 and
                bb < 0.45 and trend == "weak"):
                entry_alpha_ok = False; counters["chop_blocked"] += 1

        # V26-C: Skip choppy regime entirely
        if v26_skip_choppy and new_position == 1 and position == 0 and entry_alpha_ok:
            if regime_cfg["choppy_regime"]:
                if not (v35_rule_override and rule_trigger_now) and not v35_hybrid_now:
                    entry_alpha_ok = False; counters["v26_choppy_skipped"] += 1

        # V27: selective choppy filter
        if (not v26_skip_choppy) and v27_selective_choppy and new_position == 1 and position == 0 and entry_alpha_ok:
            if regime_cfg["choppy_regime"]:
                vol_ok = (not np.isnan(avg_vol20[i]) and volume[i] >= 0.95 * avg_vol20[i])
                quality_ok = (
                    breakout_entry or
                    (trend in ("strong", "moderate") and bs >= 3 and vs > 1.15 and vol_ok) or
                    (rule_consecutive[i] >= 3 and rs > 0 and vol_ok)
                )
                if not quality_ok:
                    entry_alpha_ok = False
                    counters["v27_choppy_low_quality_skipped"] += 1
                else:
                    counters["v27_choppy_quality_kept"] += 1

        if patch_noise_filter and new_position == 1 and position == 0 and entry_alpha_ok:
            entry_score_nf = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            if trend == "weak" and entry_score_nf < 3 and ret_5d[i] > 0.03:
                entry_alpha_ok = False; counters["noise_filtered"] += 1

        # V28-K1: Early wave filter — block when wave is already mature
        # (price has been rising for >7 of last 10 days AND ret_5d > 8%)
        if v28_early_wave_filter and new_position == 1 and position == 0 and entry_alpha_ok and not vshape_entry:
            wave_mature = (days_since_low_10[i] > 7 and ret_5d[i] > 0.08)
            # Also block if ret_2d is much smaller than ret_5d/2 (momentum slowing)
            wave_exhausted = (ret_5d[i] > 0.06 and ret_2d[i] < 0.005 and ret_2d[i] < ret_5d[i] * 0.15)
            if (wave_mature or wave_exhausted) and not breakout_entry:
                if not (v35_rule_override and rule_trigger_now) and not v35_hybrid_now:
                    entry_alpha_ok = False
                    counters["v28_late_wave_blocked"] += 1

        # V28-K2: Crash guard — block new entries when broad market is crashing
        # Proxy: ret_20d of current symbol < -12% (systemic crash period)
        if v28_crash_guard and new_position == 1 and position == 0 and entry_alpha_ok:
            market_crash = (ret_20d[i] < -0.12)
            if market_crash and not vshape_entry:
                entry_alpha_ok = False
                counters["v28_crash_guard_blocked"] += 1

        # V38d: anti-fomo entry filter — block khi gia da chay nong (entry_ret_5d/dist_sma20 cao)
        # Tru breakout_entry / vshape_entry (giu special-case)
        if v38d_fomo_filter and new_position == 1 and position == 0 and entry_alpha_ok \
                and not breakout_entry and not vshape_entry:
            ret5_now = ret_5d[i] if not np.isnan(ret_5d[i]) else 0.0
            dist_now = dist_sma20[i] if not np.isnan(dist_sma20[i]) else 0.0
            if ret5_now > v38d_fomo_ret5d_thresh or dist_now > v38d_fomo_dist_thresh:
                entry_alpha_ok = False
                counters["v38d_fomo_blocked"] += 1

        # V28-K3: Wave acceleration bonus entry
        # Allow entry when wave is just starting: ret_2d > 0 but ret_5d still low
        if v28_wave_acceleration_entry and new_position == 0 and position == 0:
            early_wave = (ret_2d[i] > 0.015 and ret_3d[i] > 0.02 and
                          ret_5d[i] < 0.05 and ret_acceleration[i] > 0.005)
            accel_quality = (trend in ("strong", "moderate") and rs > 0 and
                             not np.isnan(sma20[i]) and close[i] > sma20[i] and
                             not np.isnan(avg_vol20[i]) and volume[i] > 0.9 * avg_vol20[i])
            if early_wave and accel_quality and not regime_cfg["choppy_regime"]:
                new_position = 1
                position_size = 0.40
                counters["v28_wave_accel_entry"] += 1

        # V30-A1: Peak-proximity filter — block entry when near 20d high AND already rallied
        if v30_peak_proximity_filter and new_position == 1 and position == 0 and not vshape_entry:
            peak_prox = drop_from_peak_20[i] >= v30_peak_prox_dist_threshold  # close to peak
            already_rallied = ret_5d[i] > v30_peak_prox_rally10_min * 0.65 or (
                i >= 10 and close[i] > 0 and close[max(0,i-10)] > 0 and
                close[i] / close[max(0,i-10)] - 1 > v30_peak_prox_rally10_min
            )
            if peak_prox and already_rallied and not breakout_entry:
                entry_alpha_ok = False
                counters["v30_peak_prox_blocked"] += 1

        # V30-A2: Rally-extension filter — block when price has run too far too fast
        if v30_rally_extension_filter and new_position == 1 and position == 0 and entry_alpha_ok and not vshape_entry:
            ret10 = (close[i] / close[max(0,i-10)] - 1) if i >= 10 and close[max(0,i-10)] > 0 else 0
            ret20_val = ret_20d[i]
            if ret10 > v30_rally10_hard_block or ret20_val > v30_rally20_hard_block:
                entry_alpha_ok = False
                counters["v30_rally_ext_blocked"] += 1

        # V30-A3: Pullback-only entry — only enter after at least v30_pullback_min_pct pullback from 5d high
        if v30_pullback_only_entry and new_position == 1 and position == 0 and entry_alpha_ok and not vshape_entry and not breakout_entry:
            high_5d = np.max(high[max(0,i-5):i+1]) if i >= 5 else high[i]
            pullback_pct = (high_5d - close[i]) / high_5d if high_5d > 0 else 0
            if pullback_pct < v30_pullback_min_pct:
                entry_alpha_ok = False
                counters["v30_pullback_blocked"] += 1

        # V29-P5: Breakout strength early entry — clear recent high with dry-then-spike volume
        if v29_breakout_strength_entry and new_position == 0 and position == 0:
            if i >= 20 and not np.isnan(avg_vol20[i]):
                recent_high_10 = np.max(high[max(0, i - 10):i])
                breakout_clean = close[i] > recent_high_10 * 1.005
                dry_vol = np.nanmean(volume[max(0, i - 10):max(0, i - 3)])
                vol_spike = dry_vol > 0 and volume[i] > 1.6 * dry_vol
                acceptable = (ret_5d[i] < 0.08 and trend in ("strong", "moderate") and
                              not np.isnan(sma20[i]) and close[i] > sma20[i])
                if breakout_clean and vol_spike and acceptable and not regime_cfg["choppy_regime"]:
                    new_position = 1
                    position_size = 0.45
                    breakout_entry = True
                    counters["v29_bo_strength_entry"] += 1

        # V29-P6: Relative-strength filter — block new entries when symbol has been weak
        if v29_relstrength_filter and new_position == 1 and position == 0 and not vshape_entry:
            if trend != "strong" and ret_20d[i] < v29_rs_ret20_threshold:
                new_position = 0
                counters["v29_rs_filter_blocked"] += 1

        if new_position == 1 and position == 0 and not entry_alpha_ok:
            new_position = 0; counters["alpha_blocked"] += 1

        # V31-A: Peak-chasing guard — reduce/skip when ret5d very hot + dist_sma20 large
        if v31_peak_chasing_guard and new_position == 1 and position == 0 and not vshape_entry:
            pcg_trigger = (
                ret_5d[i] > v31_pcg_ret5d_thresh and
                dist_sma20[i] * 100 > v31_pcg_dist_thresh
            )
            if pcg_trigger:
                if v31_pcg_action == "skip":
                    new_position = 0; counters["v31_pcg_skipped"] += 1
                else:  # half_size
                    position_size = position_size * 0.50 if position_size > 0 else 0.25
                    counters["v31_pcg_halved"] += 1

        # V33-C: Recovery-peak filter — block entry khi giá đã hồi phục nhanh từ đáy
        # Mục đích: tránh mua đỉnh sóng hồi (73-74% lệnh thua có giá thấp hơn 5d/10d trước)
        # Chỉ block khi: ret10d > threshold VÀ dist_sma20 > threshold VÀ trend != strong
        if (v33_recovery_peak_filter and new_position == 1 and position == 0
                and not vshape_entry and not breakout_entry):
            ret10_val = (close[i] / close[max(0, i - 10)] - 1) if i >= 10 and close[max(0, i - 10)] > 0 else 0
            dist_ok = not np.isnan(dist_sma20[i]) and dist_sma20[i] > v33_rpf_dist_sma20_thresh
            ret_ok = ret10_val > v33_rpf_ret10_thresh
            trend_ok = (not v33_rpf_require_weak) or (trend != "strong")
            if ret_ok and dist_ok and trend_ok:
                new_position = 0; counters["v33_rpf_blocked"] += 1

        # Rule ensemble
        if patch_rule_ensemble:
            ml_buy = (new_position == 1 and position == 0)
            rule_buy = (rule_signal[i] == 1)
            if rule_buy and not ml_buy and position == 0 and trend == "strong":
                new_position = 1
                position_size = 0.30
                counters["rule_only_entry"] += 1

        # V26-E: Stronger rule ensemble
        if v26_strong_rule_ensemble and not patch_rule_ensemble:
            ml_buy = (new_position == 1 and position == 0)
            if not ml_buy and position == 0 and rule_consecutive[i] >= 3:
                if trend in ("strong", "moderate"):
                    new_position = 1
                    position_size = 0.35
                    counters["v26_strong_rule_entry"] += 1
        elif v26_strong_rule_ensemble and patch_rule_ensemble:
            if position == 0 and new_position == 0 and rule_consecutive[i] >= 3:
                if trend == "moderate":
                    new_position = 1
                    position_size = 0.30
                    counters["v26_strong_rule_moderate"] += 1

        # V27: rule-priority entry
        if v27_rule_priority and position == 0 and new_position == 0:
            if sym in RULE_PRIORITY_SYMBOLS and rule_consecutive[i] >= 2 and trend in ("strong", "moderate"):
                new_position = 1
                position_size = 0.35 if trend == "moderate" else 0.40
                counters["v27_rule_priority_entry"] += 1

        # Position sizing
        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            atr_ratio = (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03
            if vshape_entry: position_size = 0.50
            elif v35_hybrid_now: position_size = v35_hybrid_size
            elif trend == "strong" and entry_score >= 4: position_size = 0.95
            elif trend == "strong" and entry_score >= 3: position_size = 0.90
            elif trend == "moderate" and entry_score >= 3: position_size = 0.50
            elif trend == "weak": position_size = 0.30
            else: position_size = 0.50
            if atr_ratio > 0.055: position_size = min(position_size, 0.35)
            elif atr_ratio > 0.040: position_size = min(position_size, 0.50)
            if trend == "weak": position_size = min(position_size, 0.40)
            elif trend == "moderate": position_size = min(position_size, 0.70)
            if close[i] <= opn[i] and not vshape_entry: position_size *= 0.75
            if ret_5d[i] > ret5_hot: position_size = min(position_size, 0.40)
            position_size *= regime_cfg["size_mult"]

            # V26-G: Score-5 penalty
            if v26_score5_penalty and entry_score == 5:
                position_size *= 0.75
                counters["v26_score5_penalized"] += 1

            # V27: dynamic score-5 penalty
            if v27_dynamic_score5_penalty and entry_score == 5:
                score5_penalty = 1.0
                if sym in SCORE5_RISKY_SYMBOLS:
                    score5_penalty *= 0.70
                if trend == "weak":
                    score5_penalty *= 0.85
                if bb < 0.40 and trend != "strong":
                    score5_penalty *= 0.90
                if rule_consecutive[i] >= 3 and trend in ("strong", "moderate"):
                    score5_penalty = max(score5_penalty, 0.80)
                if score5_penalty < 0.999:
                    position_size *= score5_penalty
                    counters["v27_score5_dynamic_penalized"] += 1

            # V31-D: Profile-aware position sizing
            if v31_profile_sizing:
                if profile == "momentum":
                    position_size *= v31_ps_momentum_mult
                elif profile == "high_beta":
                    position_size *= v31_ps_highbeta_mult
                elif profile == "defensive":
                    position_size *= v31_ps_defensive_mult
                elif profile == "bank":
                    position_size *= v31_ps_bank_mult
                counters["v31_profile_sized"] += 1

            # V30-A4: Rally-aware position scaling
            if v30_rally_position_scaling and not vshape_entry and not breakout_entry:
                ret10_ps = (close[i] / close[max(0,i-10)] - 1) if i >= 10 and close[max(0,i-10)] > 0 else 0
                if ret10_ps > v30_rps_tier2_rally:
                    new_position = 0; counters["v30_rps_blocked"] += 1
                elif ret10_ps > v30_rps_tier1_rally:
                    position_size *= 0.70; counters["v30_rps_scaled"] += 1

            position_size = max(0.25, min(position_size, 1.0))

            # V26-F: Minimum position threshold
            if v26_min_position and position_size < 0.28:
                new_position = 0; counters["v26_min_pos_blocked"] += 1

        # === EXIT LOGIC ===
        if position == 1:
            projected = equity[i - 1] * (1 + ret * position_size)
            max_equity_in_trade = max(max_equity_in_trade, projected)
            if close[i] > max_price_in_trade: max_price_in_trade = close[i]
            cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
            max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0
            price_max_profit = (max_price_in_trade / entry_close - 1) if entry_close > 0 else 0
            price_cur_ret = (close[i] / entry_close - 1) if entry_close > 0 else 0
            strong_uptrend = trend == "strong"
            atr_ratio_now = (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03

            if not np.isnan(atr14[i]) and close[i] > 0:
                atr_stop = ATR_MULT * atr14[i] / close[i]
                atr_stop = max(0.025, min(atr_stop, 0.06))
            else:
                atr_stop = 0.04

            # 0) HARD STOP
            if cum_ret <= -HARD_STOP:
                new_position = 0; exit_reason = "hard_stop"

            # Model B exit — separate exit model override (runs after hard_stop only)
            elif (y_pred_exit is not None and hold_days >= MODEL_B_MIN_HOLD
                    and y_pred_exit[i - 1] == 1):
                new_position = 0; exit_reason = "model_b_exit"
                counters["model_b_exit"] += 1

            # V38b: Stall-exit — giu N ngay nhung max profit qua thap, dang lo nho -> chot
            elif v38b_stall_exit and hold_days >= v38b_stall_min_hold and \
                    price_max_profit < v38b_stall_max_profit and \
                    price_cur_ret <= v38b_stall_pnl_thresh:
                new_position = 0; exit_reason = "v38b_stall_exit"
                counters["v38b_stall_exit"] += 1

            # V38c: HA-driven exit — bearish reversal HOAC late_wave + body_shrinking, dang lo
            elif v38c_ha_exit and hold_days >= v38c_ha_min_hold and \
                    price_cur_ret < v38c_ha_pnl_thresh:
                ha_bear = ha_arrays["ha_bearish_reversal_signal"][i] >= 0.5
                ha_late = (ha_arrays["ha_late_wave"][i] >= 0.5 and
                           ha_arrays["ha_body_shrinking"][i] >= 0.5)
                if ha_bear or ha_late:
                    new_position = 0; exit_reason = "v38c_ha_exit"
                    counters["v38c_ha_exit"] += 1

            # V38d: Rule co-pilot exit — co profit nhung rule_signal het hoat dong (close<MA20 hoac MACD<=0)
            elif v38d_copilot_exit and hold_days >= v38d_copilot_min_hold and \
                    price_max_profit >= v38d_copilot_min_profit:
                rule_off = (
                    (np.isnan(macd_hist[i]) or macd_hist[i] <= 0) or
                    (not np.isnan(sma20[i]) and close[i] < sma20[i])
                )
                if rule_off and price_cur_ret < price_max_profit * 0.6:
                    # giu it nhat 60% peak; neu duoi thi co-pilot exit
                    new_position = 0; exit_reason = "v38d_copilot_exit"
                    counters["v38d_copilot_exit"] += 1

            # V28-K4: Early loss cut — must run BEFORE hard_cap to intercept crash entries
            # Cut loss at a tighter threshold within first N hold days
            elif v28_early_loss_cut and hold_days <= v28_early_loss_cut_days and not vshape_entry:
                if price_cur_ret <= v28_early_loss_cut_threshold:
                    new_position = 0; exit_reason = "v28_early_loss_cut"
                    counters["v28_early_loss_cut"] += 1

            # V32-A: HAP preempt — chạy TRƯỚC hard_cap: nếu đã từng có >= trigger% lãi rồi rớt về floor
            # Mục đích: bắt các trade "lên +5-15% rồi rớt -12%" mà V31-C miss do ordering bug
            # V33-D variant: chỉ trigger sau N ngày giảm liên tiếp (tránh bán đáy spike 1 ngày)
            # V39b variant: require hold_days >= v39b_hap_min_hold + optional tighter trigger
            elif v32_hap_preempt:
                hap_trigger_eff = v39b_hap_trigger if v39b_hap_trigger is not None else v32_hap_pre_trigger
                hap_price_ok = (price_max_profit >= hap_trigger_eff and
                                price_cur_ret <= v32_hap_pre_floor)
                if v39b_hap_min_hold > 0 and hold_days < v39b_hap_min_hold:
                    hap_price_ok = False  # HAP chưa active do hold quá ngắn
                if hap_price_ok:
                    if v33_hap_consec_drop:
                        # Chỉ trigger nếu đã giảm >= v33_hcd_min_days ngày liên tiếp
                        consec_drop_ok = (v33_consec_below_ema8 >= v33_hcd_min_days)
                        if consec_drop_ok:
                            new_position = 0; exit_reason = "v32_hap_preempt"
                            counters["v32_hap_preempt"] += 1
                        # else: giữ vị thế thêm, chờ xác nhận
                    else:
                        new_position = 0; exit_reason = "v32_hap_preempt"
                        counters["v32_hap_preempt"] += 1

            # V30-C1: ATR-aware hard cap — replaces fixed floor when enabled
            elif v30_atr_aware_hardcap:
                atr_cap = max(v30_atr_hc_floor, min(v30_atr_hc_ceiling, v30_atr_hc_mult * atr_ratio_now))
                if price_cur_ret <= -atr_cap:
                    new_position = 0; exit_reason = "signal_hard_cap"
                    counters["v30_atr_hc"] += 1

            # V30-C2: Two-step hard cap — partial halve then full exit
            elif v30_hardcap_two_step_v2:
                if price_cur_ret <= v30_hc2_step2_loss:
                    new_position = 0; exit_reason = "signal_hard_cap"
                    v30_hc2_halved = False
                    counters["v30_hc2_full"] += 1
                elif price_cur_ret <= v30_hc2_step1_loss and not v30_hc2_halved:
                    position_size = position_size * 0.50
                    v30_hc2_halved = True
                    counters["v30_hc2_half"] += 1

            # === HARD CAP / SIGNAL HARD CAP ===
            elif v22_mode:
                # V22: simple adaptive or flat hard cap
                if cfg["v22_adaptive_hard_cap"]:
                    if profile == "high_beta":
                        cap = max(cfg["v22_hard_cap_floor_hb"], cfg["v22_hard_cap_mult_hb"] * atr_ratio_now)
                    else:
                        cap = max(cfg["v22_hard_cap_floor"], cfg["v22_hard_cap_mult_std"] * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        counters["signal_hard_cap"] += 1
                elif price_cur_ret <= -V22_SIGNAL_HARD_CAP:
                    new_position = 0; exit_reason = "signal_hard_cap"
                    counters["signal_hard_cap"] += 1
            elif patch_smart_hardcap:
                if trend == "weak":
                    if price_cur_ret <= hard_cap_weak:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        hard_cap_pending_bars = 0
                        counters["signal_hard_cap"] += 1
                elif trend == "moderate":
                    if v26_wider_hardcap:
                        cap = max(hard_cap_moderate_floor * 1.3, hard_cap_moderate_mult * 1.3 * atr_ratio_now)
                    elif patch_adaptive_hardcap:
                        cap = max(0.10 if atr_ratio_now < 0.025 else hard_cap_moderate_floor,
                                  hard_cap_moderate_mult * atr_ratio_now)
                    else:
                        cap = max(hard_cap_moderate_floor, hard_cap_moderate_mult * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        hard_cap_pending_bars += 1
                        hc_confirm_bars = 1 + HARD_CAP_CONFIRM_MODERATE
                        if v27_hardcap_two_step and profile in ("high_beta", "momentum"):
                            hc_confirm_bars += 1
                        if hard_cap_pending_bars >= hc_confirm_bars:
                            new_position = 0; exit_reason = "signal_hard_cap"
                            hard_cap_pending_bars = 0
                            counters["signal_hard_cap"] += 1
                    else:
                        hard_cap_pending_bars = 0
                else:  # strong
                    if v26_wider_hardcap:
                        cap = max(hard_cap_strong_floor * 1.4, hard_cap_strong_mult * 1.4 * atr_ratio_now)
                        if atr_ratio_now > 0.04:
                            cap = max(0.25, 4.0 * atr_ratio_now)
                        elif profile == "high_beta":
                            cap = max(0.22, 4.0 * atr_ratio_now)
                    elif patch_adaptive_hardcap:
                        if atr_ratio_now > 0.04:
                            cap = max(0.20, 3.5 * atr_ratio_now)
                        elif profile == "high_beta":
                            cap = max(0.18, 3.5 * atr_ratio_now)
                        else:
                            cap = max(hard_cap_moderate_floor, hard_cap_strong_mult * atr_ratio_now)
                    else:
                        if profile == "high_beta":
                            cap = max(0.18, 3.5 * atr_ratio_now)
                        else:
                            cap = max(hard_cap_moderate_floor, hard_cap_strong_mult * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        hard_cap_pending_bars += 1
                        hc_confirm_bars = 1 + HARD_CAP_CONFIRM_STRONG
                        if v27_hardcap_two_step and profile in ("high_beta", "momentum"):
                            hc_confirm_bars += 1
                        if hard_cap_pending_bars >= hc_confirm_bars:
                            new_position = 0; exit_reason = "signal_hard_cap"
                            hard_cap_pending_bars = 0
                            counters["signal_hard_cap"] += 1
                    else:
                        hard_cap_pending_bars = 0
            else:
                # V23 original hard_cap (no confirm bars)
                if trend == "weak":
                    if price_cur_ret <= hard_cap_weak:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        counters["signal_hard_cap"] += 1
                elif trend == "moderate":
                    cap = max(hard_cap_moderate_floor, hard_cap_moderate_mult * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        counters["signal_hard_cap"] += 1
                else:
                    if profile == "high_beta":
                        cap = max(hard_cap_strong_floor, hard_cap_strong_mult * atr_ratio_now)
                    else:
                        cap = max(hard_cap_moderate_floor, hard_cap_strong_mult * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        counters["signal_hard_cap"] += 1

            # Fast exit loss
            if new_position == 1:
                # V30-C3: regime-aware hard cap post-check (overrides standard cap if tighter needed)
                if v30_regime_aware_hardcap:
                    choppy_now = regime_cfg["choppy_regime"]
                    rah_cap = v30_rah_choppy_cap if choppy_now else (
                        v30_rah_trending_cap if trend == "strong" else -0.09)
                    if price_cur_ret <= rah_cap:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        counters["v30_rah_cap"] += 1
                if v22_mode:
                    # V22: smart fast exit with trend_healthy check
                    do_fast_exit = False
                    ft = cfg["v22_fast_exit_threshold_hb"] if profile == "high_beta" else cfg["v22_fast_exit_threshold_std"]
                    mt = ft + 0.02

                    trend_healthy = (cfg["v22_fast_exit_skip_strong"] and strong_uptrend and
                                    macd_line[i] > 0 and
                                    not np.isnan(sma20[i]) and close[i] > sma20[i] * 0.97)

                    vol_selling = (cfg["v22_fast_exit_vol_confirm"] and
                                  not np.isnan(avg_vol20[i]) and volume[i] > 1.3 * avg_vol20[i] and
                                  close[i] < opn[i])

                    if price_cur_ret < ft and hold_days > 3:
                        if trend_healthy and not vol_selling:
                            counters["fast_exit_saved"] += 1
                        else:
                            do_fast_exit = True
                    elif (price_cur_ret < mt and hold_days > 2 and
                          macd_hist[i] < 0 and not np.isnan(ema8[i]) and close[i] < ema8[i]):
                        if trend_healthy and not vol_selling:
                            counters["fast_exit_saved"] += 1
                        else:
                            do_fast_exit = True

                    if do_fast_exit:
                        new_position = 0; exit_reason = "fast_exit_loss"
                        counters["fast_exit_loss"] += 1
                else:
                    # V23+: graduated fast_exit_loss by trend
                    hb_buf = fast_exit_hb_buffer if profile == "high_beta" else 0.0
                    if trend == "strong":
                        ft = fast_exit_strong - hb_buf
                    elif trend == "moderate":
                        ft = fast_exit_moderate - hb_buf
                    else:
                        ft = fast_exit_weak - hb_buf
                    mt = ft + 0.02

                    do_fast_exit = False
                    if price_cur_ret < ft and hold_days > 3:
                        do_fast_exit = True
                    elif (price_cur_ret < mt and hold_days > 2 and
                          macd_hist[i] < 0 and not np.isnan(ema8[i]) and close[i] < ema8[i]):
                        do_fast_exit = True

                    if do_fast_exit:
                        new_position = 0; exit_reason = "fast_exit_loss"
                        counters["fast_exit_loss"] += 1

            # ATR stop
            if new_position == 1 and cum_ret <= -atr_stop:
                new_position = 0; exit_reason = "stop_loss"

            # V29-P1: Adaptive peak lock — once max_profit reaches trigger,
            #     ratchet a stop floor = keep * max_profit. Prevents "ran +20% then closed -10%".
            if new_position == 1 and v29_adaptive_peak_lock and price_max_profit >= v29_apl_trigger:
                apply_lock = True
                if v29_peak_lock_high_beta_only and profile not in ("high_beta", "momentum"):
                    apply_lock = False
                if apply_lock:
                    floor_ret = price_max_profit * v29_apl_keep
                    if price_cur_ret <= floor_ret:
                        new_position = 0; exit_reason = "v29_peak_lock"
                        counters["v29_peak_lock"] += 1

            # V31-C: Hardcap-after-profit — once we've had >=5% profit, tighten exit floor to -3%
            # Prevents "ran +5-15% then closed -13%" which is the main hard_cap loss pattern
            if (new_position == 1 and v31_hardcap_after_profit and
                price_max_profit >= v31_hap_profit_trigger and
                price_cur_ret <= v31_hap_floor):
                new_position = 0; exit_reason = "v31_hap_exit"
                counters["v31_hap_exit"] += 1

            # V32-B: Trend-weak oversold exit — khi trend=weak + giá quá xa SMA20 + còn có lãi → exit
            # Logic: nếu trend đã yếu + giá dưới SMA20 quá 7% → thị trường đang sập, không nên giữ
            if (new_position == 1 and v32_weak_oversold_exit and
                trend == "weak" and hold_days >= v32_woe_hold_min and
                not np.isnan(dist_sma20[i]) and dist_sma20[i] < v32_woe_dist_thresh and
                price_cur_ret >= v32_woe_min_profit):
                new_position = 0; exit_reason = "v32_weak_oversold"
                counters["v32_weak_oversold"] += 1

            # V32-C: Dynamic hard_cap tighten — khi dist_sma20 < -8% → exit ở -7% thay vì -12%
            # Ghi chú: chạy khi new_position vẫn = 1 (hard_cap chưa bắt vì trade chưa đến ngưỡng cũ)
            if (new_position == 1 and v32_dynamic_hc_dist and
                not np.isnan(dist_sma20[i]) and dist_sma20[i] < v32_dhc_dist_thresh and
                price_cur_ret <= v32_dhc_tight_cap):
                new_position = 0; exit_reason = "v32_dynamic_hc"
                counters["v32_dynamic_hc"] += 1

            # V32-D: Profit ratchet exit — khi đã đạt >= trigger% profit rồi rớt về keep * max_profit
            # Nhẹ hơn V29-P1 vì dùng price-based thay vì equity-based, và keep thấp hơn
            if (new_position == 1 and v32_profit_ratchet and
                price_max_profit >= v32_pr_trigger):
                pr_floor = price_max_profit * v32_pr_keep
                if price_cur_ret <= pr_floor:
                    new_position = 0; exit_reason = "v32_profit_ratchet"
                    counters["v32_profit_ratchet"] += 1

            # V33-A: Multi-tier trailing ratchet — tighten floor progressively theo mức profit đạt được
            # Mục đích: fix -2177% drag từ signal_hard_cap (401 lệnh max_profit>5% nhưng kết thúc lỗ>5%)
            # Khác V32-D: multi-tier, sàn tăng dần theo profit lớn; tập trung vào big winners
            if new_position == 1 and v33_trailing_ratchet:
                if price_max_profit >= v33_tr_tier3_trigger:
                    tr_floor = price_max_profit * v33_tr_tier3_keep
                elif price_max_profit >= v33_tr_tier2_trigger:
                    tr_floor = price_max_profit * v33_tr_tier2_keep
                elif price_max_profit >= v33_tr_tier1_trigger:
                    tr_floor = price_max_profit * v33_tr_tier1_keep
                else:
                    tr_floor = None
                if tr_floor is not None and price_cur_ret <= tr_floor:
                    new_position = 0; exit_reason = "v33_trailing_ratchet"
                    counters["v33_trailing_ratchet"] += 1

            # V33-B: Trend-reversal exit — bán ngay khi max_profit đạt ngưỡng + trend đảo chiều
            # Mục đích: bắt "bán ở đỉnh phân phối" sớm hơn signal/hard_cap
            # Logic: price_max_profit >= min_profit VÀ close < ema8 N ngày VÀ rsi < thresh
            if new_position == 1 and v33_trend_rev_exit:
                if not np.isnan(ema8[i]) and close[i] < ema8[i]:
                    v33_consec_below_ema8 += 1
                else:
                    v33_consec_below_ema8 = 0
                if (price_max_profit >= v33_tre_min_profit and
                    hold_days >= v33_tre_hold_min and
                    v33_consec_below_ema8 >= 2 and
                    not np.isnan(rsi14[i]) and rsi14[i] < v33_tre_rsi_thresh):
                    new_position = 0; exit_reason = "v33_trend_rev_exit"
                    counters["v33_trend_rev_exit"] += 1
            else:
                # track ema8 below streak dù v33_trend_rev_exit off (cần cho v33_hap_consec_drop)
                if not np.isnan(ema8[i]) and close[i] < ema8[i]:
                    v33_consec_below_ema8 += 1
                else:
                    v33_consec_below_ema8 = 0

            # V29-P8: Profit safety net — once max_profit >= trigger, never let it close negative
            if (new_position == 1 and v29_profit_safety_net and
                price_max_profit >= v29_profit_safety_trigger and price_cur_ret < 0):
                new_position = 0; exit_reason = "v29_profit_safety"
                counters["v29_profit_safety"] += 1

            # V29-P9: Hardcap-after-peak — tighten hard cap once we've had a meaningful peak
            if (new_position == 1 and v29_hardcap_after_peak and
                price_max_profit >= v29_hardcap_after_peak_trigger and
                price_cur_ret <= v29_hardcap_after_peak_floor):
                new_position = 0; exit_reason = "v29_hardcap_after_peak"
                counters["v29_hardcap_after_peak"] += 1

            # V29-P4: Reversal after peak — exit fast when ret_2d collapses after big run-up
            if (new_position == 1 and v29_reversal_after_peak and
                price_max_profit >= v29_reversal_peak_trigger and
                ret_2d[i] <= v29_reversal_ret2_threshold):
                new_position = 0; exit_reason = "v29_reversal_after_peak"
                counters["v29_reversal_after_peak"] += 1

            # V29-P2: ATR velocity exit — exit on a fast 2-day decline > k*ATR after meaningful profit
            if (new_position == 1 and v29_atr_velocity_exit and
                price_max_profit >= v29_atr_velocity_min_profit and
                not np.isnan(atr14[i]) and close[i] > 0 and i >= 2):
                two_d_drop = close[i - 2] - close[i]
                atr_threshold = v29_atr_velocity_k * atr14[i]
                if two_d_drop >= atr_threshold:
                    new_position = 0; exit_reason = "v29_atr_velocity"
                    counters["v29_atr_velocity"] += 1

            # V30-B3: Chandelier trailing — ATR×k trailing from max_high when in profit
            if (new_position == 1 and v30_chandelier_trail and
                price_max_profit >= v30_chand_profit_trigger and
                not np.isnan(atr14[i]) and close[i] > 0):
                chand_stop = max_price_in_trade - v30_chand_atr_mult * atr14[i]
                if close[i] < chand_stop:
                    new_position = 0; exit_reason = "v30_chandelier"
                    counters["v30_chandelier"] += 1

            # Peak protection
            if new_position == 1 and mod_b:
                if v22_mode:
                    # V22: simple threshold + 3-in-1
                    if price_max_profit >= 0.20:
                        price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                        heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
                        bearish_candle = close[i] < opn[i]
                        if price_below_sma10 and heavy_vol and bearish_candle:
                            new_position = 0; exit_reason = "peak_protect_dist"
                            counters["peak_protect"] += 1
                elif patch_pp_restore:
                    if price_max_profit >= 0.25:
                        pp_threshold_active = 0.10
                    elif strong_uptrend:
                        pp_threshold_active = peak_protect_strong_threshold
                    else:
                        pp_threshold_active = peak_protect_normal_threshold

                    pp_bonus = regime_cfg.get("pp_sensitivity_bonus", 0)
                    pp_threshold_active = max(0.08, pp_threshold_active - pp_bonus)

                    if price_max_profit >= pp_threshold_active:
                        price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                        if price_below_sma10:
                            pp_pending_bars += 1
                            heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.3 * avg_vol20[i])
                            if pp_pending_bars >= 2 or heavy_vol:
                                new_position = 0; exit_reason = "peak_protect_dist"
                                pp_pending_bars = 0
                                counters["peak_protect"] += 1
                        else:
                            pp_pending_bars = 0
                elif patch_pp_2of3:
                    pp_threshold = peak_protect_strong_threshold if strong_uptrend else peak_protect_normal_threshold
                    if price_max_profit >= pp_threshold:
                        price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                        heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
                        bearish_candle = close[i] < opn[i]
                        checks = sum([price_below_sma10, heavy_vol, bearish_candle])
                        if checks >= 2:
                            new_position = 0; exit_reason = "peak_protect_dist"
                            counters["peak_protect"] += 1
                else:
                    pp_threshold = peak_protect_strong_threshold if strong_uptrend else peak_protect_normal_threshold
                    if price_max_profit >= pp_threshold:
                        price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                        heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
                        bearish_candle = close[i] < opn[i]
                        if price_below_sma10 and heavy_vol and bearish_candle:
                            new_position = 0; exit_reason = "peak_protect_dist"
                            counters["peak_protect"] += 1

            # EMA8 peak protect
            if mod_b and new_position == 1 and position == 1:
                if price_max_profit >= 0.15:
                    if not np.isnan(ema8[i]) and close[i] < ema8[i]:
                        consecutive_below_ema8 += 1
                    else:
                        consecutive_below_ema8 = 0
                    if consecutive_below_ema8 >= 2 and price_cur_ret < price_max_profit * 0.75:
                        new_position = 0; exit_reason = "peak_protect_ema"
                        counters["peak_protect"] += 1

            # V28-K5: Cycle peak exit — if ret_3d turns negative after being up >8%, exit quickly
            if v28_cycle_peak_exit and new_position == 1 and position == 1:
                if price_max_profit >= 0.08 and ret_3d[i] < -0.02 and price_cur_ret < price_max_profit * 0.70:
                    heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.2 * avg_vol20[i])
                    if heavy_vol or ret_3d[i] < -0.04:
                        new_position = 0; exit_reason = "v28_cycle_peak"
                        counters["v28_cycle_peak"] += 1

            # Hybrid exit
            if new_position == 1 and strong_uptrend and cum_ret > 0.05 and max_profit > 0.08:
                macd_bearish = macd_hist[i] < 0 and macd_hist[i - 1] >= 0 if i > 0 else False
                price_below_ma20 = close[i] < sma20[i] if not np.isnan(sma20[i]) else False
                if macd_bearish and price_below_ma20:
                    new_position = 0; exit_reason = "hybrid_exit"
                elif price_below_ma20 and cum_ret < max_profit * 0.5:
                    new_position = 0; exit_reason = "hybrid_exit"

            # Adaptive trailing
            elif new_position == 1 and max_profit > 0.03:
                if max_profit > 0.25: trail_pct = 0.18
                elif max_profit > 0.15: trail_pct = 0.25
                elif max_profit > 0.08: trail_pct = 0.40
                else: trail_pct = 0.65
                # V29-P3: tighten when max_profit is very high (lock more profit)
                if v29_tighter_trail_high_profit and max_profit >= v29_high_profit_trigger:
                    trail_pct = min(trail_pct, v29_high_profit_trail)
                if strong_uptrend: trail_pct *= STRONG_TREND_TRAIL_MULT
                elif trend == "moderate": trail_pct *= 0.7
                giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                if giveback >= trail_pct:
                    new_position = 0; exit_reason = "trailing_stop"

            # Profit lock
            if new_position == 1 and max_profit >= PROFIT_LOCK_THRESHOLD:
                if cum_ret < PROFIT_LOCK_MIN and not strong_uptrend:
                    if patch_symbol_tuning and regime_cfg.get("disable_profit_lock_in_strong") and trend == "strong":
                        pass
                    else:
                        new_position = 0; exit_reason = "profit_lock"

            # Zombie
            if new_position == 1 and hold_days >= ZOMBIE_BARS and cum_ret < 0.01:
                if not strong_uptrend:
                    new_position = 0; exit_reason = "zombie_exit"

            # Extended hold / trend persistence
            extended_min_hold = MIN_HOLD
            if v26_extended_hold and strong_uptrend and cum_ret > 0.05:
                extended_min_hold = 12
            if v27_trend_persistence_hold and strong_uptrend and cum_ret > 0.03:
                trend_supported = (
                    (not np.isnan(sma20[i]) and close[i] >= sma20[i] * 0.995) and
                    (macd_hist[i] > -0.01)
                )
                if trend_supported:
                    extended_min_hold = max(extended_min_hold, 10)
                    counters["v27_persistence_hold"] += 1

            # Min hold
            if new_position == 0 and exit_reason not in (
                    "stop_loss", "hard_stop", "hybrid_exit", "peak_protect_dist",
                    "peak_protect_ema", "fast_loss_cut", "signal_hard_cap",
                    "fast_exit_loss", "v31_hap_exit", "model_b_exit") and hold_days < extended_min_hold:
                if cum_ret > -atr_stop: new_position = 1

            # V39a: signal exit min hold — block signal exit trước khi đủ hold_days
            # Root cause: early_wave fw=8d target tạo bias thoát sớm ở 21-30d (WR=12%)
            # Nếu rule_confirm bật: thêm điều kiện MACD<0 AND Close<MA20 mới cho exit
            if (new_position == 0 and exit_reason == "signal" and
                    v39a_signal_exit_min_hold > 0 and hold_days < v39a_signal_exit_min_hold):
                new_position = 1; counters["v39a_min_hold_blocked"] += 1
            if (new_position == 0 and exit_reason == "signal" and v39a_rule_confirm_exit):
                macd_bearish_now = (not np.isnan(macd_hist[i]) and macd_hist[i] < 0)
                below_ma20_now = (not np.isnan(sma20[i]) and close[i] < sma20[i])
                # V39g: chỉ defer khi max_profit đã đạt threshold (trades stall thì không defer)
                min_mp_ok = (price_max_profit >= v39g_rule_confirm_min_maxprofit)
                if not (macd_bearish_now and below_ma20_now) and min_mp_ok:
                    new_position = 1; counters["v39a_rule_confirm_blocked"] += 1

            # V31-E: Short-hold exit filter — block signal exits if hold too short and pnl not bad enough
            # Prevents the flood of hold<10d signal exits at -4% that drag WR down
            if (new_position == 0 and exit_reason == "signal" and v31_short_hold_exit_filter and
                hold_days < v31_shef_min_hold and cum_ret > v31_shef_min_pnl):
                new_position = 1; counters["v31_shef_blocked"] += 1

            # V32-E: Signal-weak-oversold passthrough — KHÔNG block signal exit khi trend=weak + dist<thresh
            # Bổ trợ V31-E: dù hold chưa đủ, nếu market oversold nặng thì vẫn cho exit
            if (new_position == 1 and exit_reason == "signal" and v32_signal_weak_exit and
                trend == "weak" and
                not np.isnan(dist_sma20[i]) and dist_sma20[i] < v32_swe_dist_thresh):
                new_position = 0; counters["v32_swe_pass"] += 1

            # V33-E: RSI oversold block — không thoát bởi signal/hap_preempt khi rsi oversold sâu
            # Mục đích: tránh bán đúng đáy điều chỉnh (85 lệnh giá tăng >8% trong 5 ngày sau bán)
            if (new_position == 0 and v33_rsi_oversold_block and
                exit_reason in ("signal", "v32_hap_preempt", "v32_weak_oversold") and
                not np.isnan(rsi14[i]) and rsi14[i] < v33_rob_rsi_thresh and
                price_cur_ret > -0.08):  # chỉ block khi chưa lỗ quá nặng
                new_position = 1; counters["v33_rob_blocked"] += 1

            # V33-F: Signal confirm exit — require 2 consecutive signal bars to exit
            # Mục đích: lọc false signal exit (776 exits WR 57% nhưng avg +7.9% — khá nhiều false exits)
            # Safety: chỉ block khi hold_days < 90 để tránh zombie trades quá dài
            if (new_position == 0 and exit_reason == "signal" and v33_signal_confirm_exit and
                cum_ret > v33_sce_min_pnl and price_max_profit >= v33_sce_min_profit_seen and
                hold_days < v33_sce_max_hold):
                if not v33_prev_signal_exit:
                    new_position = 1  # chờ bar tiếp theo confirm
                    counters["v33_sce_deferred"] += 1
                # else: bar trước cũng signal → cho exit

            # V30-B2: Momentum-hold override — run before confirmed exit scoring
            if (new_position == 0 and exit_reason == "signal" and
                v30_momentum_hold_override and cum_ret >= v30_mho_min_profit and
                not np.isnan(rsi14[i]) and rsi14[i] < v30_mho_rsi_max and
                not np.isnan(sma20[i]) and close[i] > sma20[i] and
                not np.isnan(ema8[i]) and close[i] > ema8[i] and
                macd_hist[i] > 0):
                new_position = 1; counters["v30_mho_saved"] += 1

            # V30-B1: Signal-exit defer — hold extra bars when still in uptrend and profitable
            if new_position == 0 and exit_reason == "signal" and v30_signal_exit_defer:
                if cum_ret >= v30_sed_min_cum_ret and trend in ("strong", "moderate"):
                    if v30_defer_bars_remaining > 0:
                        v30_defer_bars_remaining -= 1
                        new_position = 1; counters["v30_sed_defer"] += 1
                    else:
                        v30_defer_bars_remaining = v30_sed_defer_bars  # arm for next time
                else:
                    v30_defer_bars_remaining = 0  # reset when not qualifying

            # V31-B: Adaptive defer — defer signal exit while momentum confirmed (up to max_bars)
            # Smarter than V30's flat-N-bars: resets counter when EMA confirm fails
            if new_position == 0 and exit_reason == "signal" and v31_adaptive_defer:
                ema_ok = (not np.isnan(ema8[i]) and close[i] >= ema8[i]) if v31_ad_use_ema_confirm else True
                if cum_ret >= v31_ad_min_cum_ret and trend in ("strong", "moderate") and ema_ok:
                    if v31_defer_bars_remaining < v31_ad_max_bars:
                        v31_defer_bars_remaining += 1
                        new_position = 1; counters["v31_ad_defer"] += 1
                    else:
                        v31_defer_bars_remaining = 0  # max bars reached — let exit through
                else:
                    v31_defer_bars_remaining = 0

            # V39d: per-symbol rule-exit hybrid — với các mã stable-trend đã thua rule nhiều
            # Thay thế signal exit bằng rule-based exit: thoát khi Close<MA20 OR MACD_hist<=0
            # Mục đích: 21 mã V37a thua Rule do signal exit sớm, rule bắt trend 30-60d tốt hơn
            if (new_position == 0 and exit_reason == "signal" and v39d_rule_exit_symbols and
                    sym in v39d_rule_exit_symbols):
                rule_exit_now = (
                    (not np.isnan(macd_hist[i]) and macd_hist[i] <= 0) or
                    (not np.isnan(sma20[i]) and close[i] < sma20[i])
                )
                if not rule_exit_now:
                    new_position = 1; counters["v39d_rule_exit_held"] += 1
                # else: rule đồng thuận → cho signal exit thông qua

            # Confirmed signal exit
            if new_position == 0 and exit_reason == "signal" and mod_h:
                below_ma20 = (not np.isnan(sma20[i]) and close[i] < sma20[i])
                below_ma50 = (not np.isnan(sma50[i]) and close[i] < sma50[i])
                heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.4 * avg_vol20[i])
                bearish_candle = close[i] < opn[i]
                macd_falling = macd_hist[i] < macd_hist[i - 1] if i > 0 else False
                below_ema8 = (not np.isnan(ema8[i]) and close[i] < ema8[i] * 0.997)
                weak_rebound = (ret_5d[i] < 0.01 and rs <= 0)
                bearish_score = 0.0
                bearish_score += 2.0 if below_ma50 else 0.0
                bearish_score += 1.0 if below_ma20 else 0.0
                bearish_score += 1.0 if macd_hist[i] < -0.03 else 0.0
                bearish_score += 0.8 if (macd_hist[i] < 0 and macd_falling) else 0.0
                bearish_score += 0.8 if (bearish_candle and heavy_vol) else 0.0
                bearish_score += 0.7 if below_ema8 else 0.0
                bearish_score += 0.5 if weak_rebound else 0.0
                score_threshold = regime_cfg["exit_score_threshold"]
                if cum_ret > 0.06 and max_profit > 0.10 and trend == "strong": score_threshold += 0.7
                if hold_days < 7 and cum_ret > -0.02: score_threshold += 0.4
                if cum_ret < -0.03: score_threshold -= 0.4
                if not v22_mode:
                    if hold_days > time_decay_bars and cum_ret < 0.02:
                        score_threshold *= time_decay_mult
                        counters["time_decay_exit"] += 1
                    elif hold_days > 15 and cum_ret < 0.03:
                        score_threshold *= 0.60
                    elif hold_days > 10 and cum_ret < 0.01:
                        score_threshold *= 0.75
                else:
                    if hold_days > 15 and cum_ret < 0.03:
                        score_threshold *= 0.60
                    elif hold_days > 10 and cum_ret < 0.01:
                        score_threshold *= 0.75
                if bearish_score < score_threshold:
                    new_position = 1; counters["confirmed_exit_blocked"] += 1

            if new_position == 0 and exit_reason == "signal":
                if cum_ret < 0: confirm_bars = 0
                else:
                    confirm_bars = regime_cfg["base_confirm_bars"]
                    if cum_ret < -0.03: confirm_bars = max(1, confirm_bars - 1)
                if raw_signal == 0: consecutive_exit_signals += 1
                else: consecutive_exit_signals = 0
                if consecutive_exit_signals < confirm_bars: new_position = 1
                else: consecutive_exit_signals = 0

            if new_position == 0 and exit_reason == "signal":
                if cum_ret > 0.03 and trend == "strong": new_position = 1

            if mod_i and new_position == 0 and exit_reason == "signal":
                still_supported = (not np.isnan(sma20[i]) and close[i] >= sma20[i] * 0.99)
                trend_ok = trend in ("strong", "moderate")
                if (cum_ret > 0.03 and max_profit > 0.06 and trend_ok and still_supported and
                    macd_hist[i] > -0.02):
                    new_position = 1; counters["trend_carry_saved"] += 1

            # Long-horizon carry
            if patch_long_horizon:
                if (new_position == 0 and exit_reason in ("signal", "peak_protect_dist", "peak_protect_ema",
                                                           "profit_lock", "trailing_stop")):
                    long_horizon_regime = (
                        ret_60d[i] > 0.30 and
                        not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and not np.isnan(sma100[i]) and
                        sma20[i] > sma50[i] > sma100[i] and
                        days_above_sma50[i] >= 20 and
                        cum_ret > 0.15
                    )
                    if long_horizon_regime:
                        hard_breakdown = (
                            (close[i] < sma50[i] * 0.97) or
                            (i >= 3 and macd_hist[i] < 0 and macd_hist[i-1] < 0 and macd_hist[i-2] < 0)
                        )
                        if not hard_breakdown:
                            new_position = 1
                            counters["long_horizon_carry"] += 1

        else:
            consecutive_exit_signals = 0
            consecutive_below_ema8 = 0
            v33_consec_below_ema8 = 0  # reset when not in position

        # Track prev signal exit for V33-F confirm
        v33_prev_signal_exit = (new_position == 0 and exit_reason == "signal") if position == 1 else False

        # EXECUTE
        cost = 0
        if new_position != position:
            if new_position == 1:
                deploy = equity[i - 1] * position_size
                cost = deploy * commission
                entry_equity = deploy - cost
                max_equity_in_trade = entry_equity
                current_entry_day = i; hold_days = 0
                consecutive_exit_signals = 0; consecutive_below_ema8 = 0
                hard_cap_pending_bars = 0; pp_pending_bars = 0
                entry_close = close[i]; max_price_in_trade = close[i]
                v30_hc2_halved = False
                v30_defer_bars_remaining = 0
                v31_defer_bars_remaining = 0
                v33_consec_below_ema8 = 0
                v33_prev_signal_exit = False
                entry_features = {
                    "entry_wp": wp, "entry_dp": dp, "entry_rs": rs,
                    "entry_vs": vs, "entry_bs": bs, "entry_hl": hl,
                    "entry_od": od, "entry_bb": bb,
                    "entry_score": sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2]),
                    "entry_date": str(dates[i])[:10], "entry_symbol": str(symbols[i]),
                    "position_size": position_size, "entry_trend": trend,
                    "quick_reentry": quick_reentry, "breakout_entry": breakout_entry,
                    "vshape_entry": vshape_entry,
                    "entry_ret_5d": round(ret_5d[i] * 100, 2),
                    "entry_drop20d": round(drop_from_peak_20[i] * 100, 2),
                    "entry_dist_sma20": round(dist_sma20[i] * 100, 2),
                    "entry_profile": regime_cfg["profile"],
                    "entry_choppy_regime": regime_cfg["choppy_regime"],
                }
            else:
                cost = equity[i - 1] * position_size * (commission + tax)
                pnl_pct_now = (close[i] / entry_close - 1) * 100 if entry_close > 0 else 0
                if v35_relax_cooldown:
                    cooldown_remaining = v35_cooldown_after_big_loss if pnl_pct_now < -5 else v35_cooldown_after_loss
                else:
                    cooldown_remaining = COOLDOWN_AFTER_BIG_LOSS if pnl_pct_now < -5 else 3
                last_exit_price = close[i]; last_exit_reason = exit_reason; last_exit_bar = i
                if record_trades and entry_equity > 0:
                    pnl_pct = (close[i] / entry_close - 1) * 100 if entry_close > 0 else 0
                    max_pnl_pct = (max_equity_in_trade - entry_equity) / entry_equity * 100
                    trade_rec = {
                        "entry_day": current_entry_day, "exit_day": i,
                        "holding_days": i - current_entry_day,
                        "pnl_pct": round(pnl_pct, 2), "max_profit_pct": round(max_pnl_pct, 2),
                        "exit_reason": exit_reason, "exit_date": str(dates[i])[:10],
                        **entry_features,
                    }
                    if v31_enriched_log:
                        trade_rec["exit_trend"] = trend
                        trade_rec["exit_dist_sma20"] = round(dist_sma20[i] * 100, 2)
                        trade_rec["exit_ret_5d"] = round(ret_5d[i] * 100, 2)
                        trade_rec["exit_rsi14"] = round(float(rsi14[i]), 2) if not np.isnan(rsi14[i]) else 50.0
                        trade_rec["exit_macd_hist"] = round(float(macd_hist[i]), 5) if not np.isnan(macd_hist[i]) else 0.0
                        trade_rec["price_max_profit_pct"] = round(price_max_profit * 100, 2)
                        trade_rec["exit_above_sma20"] = int(not np.isnan(sma20[i]) and close[i] >= sma20[i])
                        trade_rec["exit_above_ema8"] = int(not np.isnan(ema8[i]) and close[i] >= ema8[i])
                        trade_rec["exit_vol_ratio"] = round(float(volume[i] / avg_vol20[i]), 2) if (not np.isnan(avg_vol20[i]) and avg_vol20[i] > 0) else 1.0
                    trades.append(trade_rec)
                entry_equity = 0; max_equity_in_trade = 0; max_price_in_trade = 0; position_size = 1.0
                hard_cap_pending_bars = 0; pp_pending_bars = 0

        if position == 1:
            equity[i] = equity[i - 1] * (1 + ret * position_size) - cost
            hold_days += 1
        else:
            equity[i] = equity[i - 1] - cost
        position = new_position

    if position == 1 and entry_equity > 0 and record_trades:
        pnl_pct = (close[-1] / entry_close - 1) * 100 if entry_close > 0 else 0
        trades.append({
            "entry_day": current_entry_day, "exit_day": n - 1,
            "holding_days": n - 1 - current_entry_day,
            "pnl_pct": round(pnl_pct, 2), "exit_reason": "end",
            "exit_date": str(dates[-1])[:10], **entry_features,
        })

    return {
        "equity_curve": equity, "trades": trades,
        "total_return_pct": round((equity[-1] / initial_capital - 1) * 100, 2),
        "final_equity": round(equity[-1]),
        **{f"n_{k}": v for k, v in counters.items()},
    }
