import os as _os
import yaml as _yaml


def _load_symbol_configs():
    """Load symbol classification configs from models.yaml. Fallback to hardcoded values."""
    try:
        config_path = _os.path.join(
            _os.path.dirname(__file__), '..', '..', 'config', 'models.yaml'
        )
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = _yaml.safe_load(f)

        profiles = {}
        for profile_name, syms in cfg.get('symbol_profiles', {}).items():
            for sym in (syms or []):
                profiles[str(sym)] = profile_name

        rule_priority = set(str(s) for s in cfg.get('rule_priority_symbols', []))
        score5_risky = set(str(s) for s in cfg.get('score5_risky_symbols', []))
        return profiles, rule_priority, score5_risky
    except Exception:
        return _FALLBACK_SYMBOL_PROFILES, _FALLBACK_RULE_PRIORITY, _FALLBACK_SCORE5_RISKY


_FALLBACK_SYMBOL_PROFILES = {
    "ACB": "bank", "BID": "bank", "MBB": "bank", "TCB": "bank",
    "AAV": "high_beta", "AAS": "high_beta", "SSI": "high_beta", "VND": "high_beta",
    "DGC": "momentum", "HPG": "momentum", "VIC": "momentum",
    "FPT": "defensive", "REE": "defensive", "VNM": "defensive",
}
_FALLBACK_RULE_PRIORITY = {"AAA", "SSN", "TEG", "GAS", "PLX", "IJC", "DQC"}
_FALLBACK_SCORE5_RISKY = {"AAA", "IJC", "ITC", "VHM", "TEG", "QBS", "KMR", "SSN", "PLX"}

SYMBOL_PROFILES, RULE_PRIORITY_SYMBOLS, SCORE5_RISKY_SYMBOLS = _load_symbol_configs()


FEATURE_NAMES = [
    "rsi_slope_5d", "vol_surge_ratio", "range_position_20d",
    "dist_to_resistance", "breakout_setup_score", "bb_width_percentile",
    "higher_lows_count", "obv_price_divergence",
]

FEATURE_DEFAULTS = {
    "rsi_slope_5d": 0, "vol_surge_ratio": 1.0, "range_position_20d": 0.5,
    "dist_to_resistance": 0.05, "breakout_setup_score": 0, "bb_width_percentile": 0.5,
    "higher_lows_count": 0, "obv_price_divergence": 0,
}

DEFAULT_PARAMS = {
    "initial_capital": 100_000_000,
    "commission": 0.0015,
    "tax": 0.001,
    "record_trades": True,
    "model_b_min_hold": 3,
    # Mods
    "mod_a": True, "mod_b": True, "mod_c": False, "mod_d": False,
    "mod_e": True, "mod_f": True, "mod_g": True, "mod_h": True,
    "mod_i": True, "mod_j": True,
    # V23 tunable params
    "fast_exit_strong": -0.08,
    "fast_exit_moderate": -0.06,
    "fast_exit_weak": -0.04,
    "fast_exit_hb_buffer": 0.02,
    "peak_protect_strong_threshold": 0.15,
    "peak_protect_normal_threshold": 0.20,
    "hard_cap_weak": -0.10,
    "hard_cap_moderate_mult": 2.0,
    "hard_cap_strong_mult": 3.0,
    "hard_cap_strong_floor": 0.15,
    "hard_cap_moderate_floor": 0.12,
    "time_decay_bars": 20,
    "time_decay_mult": 0.50,
    # V22-specific flags
    "v22_mode": False,
    "v22_fast_exit_skip_strong": True,
    "v22_fast_exit_vol_confirm": True,
    "v22_fast_exit_threshold_hb": -0.07,
    "v22_fast_exit_threshold_std": -0.05,
    "v22_adaptive_hard_cap": True,
    "v22_hard_cap_mult_hb": 3.0,
    "v22_hard_cap_mult_std": 2.5,
    "v22_hard_cap_floor": 0.12,
    "v22_hard_cap_floor_hb": 0.15,
    # V24 patch flags
    "patch_smart_hardcap": False,
    "patch_pp_restore": False,
    "patch_long_horizon": False,
    "patch_symbol_tuning": False,
    "patch_rule_ensemble": False,
    "patch_noise_filter": False,
    "patch_adaptive_hardcap": False,
    "patch_pp_2of3": False,
    # V26 patch flags
    "v26_wider_hardcap": False,
    "v26_relaxed_entry": False,
    "v26_skip_choppy": False,
    "v26_extended_hold": False,
    "v26_strong_rule_ensemble": False,
    "v26_min_position": False,
    "v26_score5_penalty": False,
    "v26_hardcap_confirm_strong": False,
    # V27 patch flags
    "v27_selective_choppy": False,
    "v27_hardcap_two_step": False,
    "v27_rule_priority": False,
    "v27_dynamic_score5_penalty": False,
    "v27_trend_persistence_hold": False,
    # V28 patch flags
    # K1: Early wave filter — block entry when wave too mature (days_since_low_10 > 7 and ret_5d > 8%)
    "v28_early_wave_filter": False,
    # K2: Market crash guard — block entry when multi-stock simultaneous dump detected via index proxy
    #     (ret_20d < -12% = broad market falling hard)
    "v28_crash_guard": False,
    # K3: Wave acceleration entry — bonus entry when ret_2d > 0 but ret_5d still < 5% (early wave)
    "v28_wave_acceleration_entry": False,
    # K4: Intermediate loss cut — exit faster (within 5d) if pnl < -5% instead of waiting for hard_cap
    "v28_early_loss_cut": False,
    # K5: Cycle peak exit — if ret_3d < -2% after being up >8%, tighten trailing to exit
    "v28_cycle_peak_exit": False,
    # early_loss_cut threshold (default -5%)
    "v28_early_loss_cut_threshold": -0.05,
    "v28_early_loss_cut_days": 5,
    # V29 patch flags — ưu tiên fix "ran up big then closed negative"
    # P1: Adaptive peak-lock — ratchet a profit floor based on max profit so far
    "v29_adaptive_peak_lock": False,
    "v29_adaptive_peak_lock_trigger": 0.10,
    "v29_adaptive_peak_lock_keep": 0.40,
    # P2: ATR-based velocity exit — exit on fast 2-day decline after meaningful profit
    "v29_atr_velocity_exit": False,
    "v29_atr_velocity_k": 1.6,
    "v29_atr_velocity_min_profit": 0.03,
    # P3: Tighter trail when max_profit is very high
    "v29_tighter_trail_high_profit": False,
    "v29_high_profit_trigger": 0.20,
    "v29_high_profit_trail": 0.12,
    # P4: Reversal-after-peak exit (max_profit > t AND ret_2d < th)
    "v29_reversal_after_peak": False,
    "v29_reversal_peak_trigger": 0.10,
    "v29_reversal_ret2_threshold": -0.04,
    # P5: Breakout-strength early entry
    "v29_breakout_strength_entry": False,
    # P6: Relative-strength filter (skip weak-vs-self symbols in non-strong trend)
    "v29_relstrength_filter": False,
    "v29_rs_ret20_threshold": -0.05,
    # P7: Selective peak lock — apply P1 only to high_beta/momentum profiles
    "v29_peak_lock_high_beta_only": False,
    # P8: Profit safety net — once max_profit >= 25%, never let it close negative
    "v29_profit_safety_net": False,
    "v29_profit_safety_trigger": 0.25,
    # P9: Hardcap reduce after peak — if max_profit >= 15%, tighten hard_cap to break-even -3%
    "v29_hardcap_after_peak": False,
    "v29_hardcap_after_peak_trigger": 0.15,
    "v29_hardcap_after_peak_floor": -0.03,

    # ── V30 patch flags ──────────────────────────────────────────────────────
    # A1: Peak-proximity entry filter — block entry when price ≤2% from 20d high AND rally10d > thresh
    "v30_peak_proximity_filter": False,
    "v30_peak_prox_dist_threshold": -0.02,   # dist_from_peak_20 >= this means near peak
    "v30_peak_prox_rally10_min": 0.08,        # rally_10d threshold to trigger block

    # A2: Rally-extension entry filter — block entry when ret_10d or ret_20d too hot
    "v30_rally_extension_filter": False,
    "v30_rally10_hard_block": 0.12,          # ret_10d > this => block unconditionally
    "v30_rally20_hard_block": 0.18,          # ret_20d > this => block unconditionally

    # A3: Pullback-only entry — only allow entry when price has pulled back ≥ thresh from 5d high
    "v30_pullback_only_entry": False,
    "v30_pullback_min_pct": 0.03,            # require close to be ≥ 3% below recent 5d high

    # A4: Rally-aware position scaling — reduce size when entry is extended
    "v30_rally_position_scaling": False,
    "v30_rps_tier1_rally": 0.05,             # rally_10d above this → scale to 0.75
    "v30_rps_tier2_rally": 0.10,             # rally_10d above this → skip (block)

    # B1: Signal-exit defer — when signal exit fires, hold extra N days if still in uptrend
    "v30_signal_exit_defer": False,
    "v30_sed_defer_bars": 3,                  # bars to defer
    "v30_sed_min_cum_ret": 0.03,             # only defer when in profit

    # B2: Momentum-hold override — suppress signal exit when momentum still healthy
    "v30_momentum_hold_override": False,
    "v30_mho_min_profit": 0.05,             # only apply when trade already up 5%
    "v30_mho_rsi_max": 72,                  # only apply when RSI below this (not overbought)

    # B3: Chandelier trailing — ATR×k from max_high when profit >= trigger
    "v30_chandelier_trail": False,
    "v30_chand_atr_mult": 3.0,
    "v30_chand_profit_trigger": 0.05,

    # C1: ATR-aware hard cap — replace fixed floor with ATR-based cap
    "v30_atr_aware_hardcap": False,
    "v30_atr_hc_mult": 1.5,                 # cap = atr_mult × atr_ratio (min 0.08, max 0.20)
    "v30_atr_hc_floor": 0.08,
    "v30_atr_hc_ceiling": 0.20,

    # C2: Two-step hard cap — partial exit at step1, full at step2
    "v30_hardcap_two_step_v2": False,
    "v30_hc2_step1_loss": -0.04,            # at this loss, halve position
    "v30_hc2_step2_loss": -0.08,            # at this loss, full exit

    # C3: Regime-aware hard cap — tighter in choppy, wider in trending
    "v30_regime_aware_hardcap": False,
    "v30_rah_choppy_cap": -0.05,
    "v30_rah_trending_cap": -0.12,

    # === V35 flags (loosen entry filters for V-shape recoveries) ===
    # V35b: Relaxed cooldown — only N bar after big loss instead of 3-5
    "v35_relax_cooldown": False,
    "v35_cooldown_after_big_loss": 1,
    "v35_cooldown_after_loss": 0,
    # V35b: Skip price-proximity block (still keep for trailing_stop exits)
    "v35_skip_price_proximity": False,
    # V35b: Single-bar consecutive gate (drop 2-bar requirement)
    "v35_single_bar_signal": False,
    # V35b: Rule override — ignore ret5_hot, early_wave_filter, choppy when basic rule fires
    "v35_rule_override": False,
    "v35_rule_override_min_score": 1,
    # V35c: Hybrid entry — also enter on rule_trigger when ML pred==0
    "v35_hybrid_entry": False,
    "v35_hybrid_size": 0.5,
}
