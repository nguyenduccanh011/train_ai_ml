"""Trading cost defaults and legacy parameter constants.

VN market realistic baseline:
  - commission: 0.15% per side (broker fee)
  - tax: 0.10% on sell only (Vietnamese stock transfer tax)
  - slippage: 0.10% per side (typical bid-ask + execution drift on liquid VN30 names)
"""

from __future__ import annotations

import os as _os
import yaml as _yaml


def _load_symbol_configs():
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

DEFAULT_TRADING_COST: dict[str, float] = {
    "commission": 0.0015,
    "tax": 0.0010,
    "slippage": 0.0010,
}

DEFAULT_INITIAL_CAPITAL: float = 100_000_000.0

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
    "use_model_b_exit": False,
    "model_b_min_hold": 3,
    "model_b_require_trend_break": False,
    "model_b_trend_break_use_sma20": True,
    "model_b_trend_break_use_ema8": True,
    "model_b_trend_break_use_macd": False,
    "model_b_trend_break_use_red_candle": False,
    "model_b_min_negative_signals": 2,
    "model_b_profit_only_if_losing_structure": False,
    "model_b_profit_threshold": 0.0,
    "exit_mode": "rule",
    "pnl_mode": "log",
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
    "v28_early_wave_filter": False,
    "v28_crash_guard": False,
    "v28_wave_acceleration_entry": False,
    "v28_early_loss_cut": False,
    "v28_cycle_peak_exit": False,
    "v28_early_loss_cut_threshold": -0.05,
    "v28_early_loss_cut_days": 5,
    # V29 patch flags
    "v29_adaptive_peak_lock": False,
    "v29_adaptive_peak_lock_trigger": 0.10,
    "v29_adaptive_peak_lock_keep": 0.40,
    "v29_atr_velocity_exit": False,
    "v29_atr_velocity_k": 1.6,
    "v29_atr_velocity_min_profit": 0.03,
    "v29_tighter_trail_high_profit": False,
    "v29_high_profit_trigger": 0.20,
    "v29_high_profit_trail": 0.12,
    "v29_reversal_after_peak": False,
    "v29_reversal_peak_trigger": 0.10,
    "v29_reversal_ret2_threshold": -0.04,
    "v29_breakout_strength_entry": False,
    "v29_relstrength_filter": False,
    "v29_rs_ret20_threshold": -0.05,
    "v49_near_breakout_entry": False,
    "v49_nb_lookback": 10,
    "v49_nb_close_ratio": 0.98,
    "v49_nb_vol_mult": 1.6,
    "v49_nb_ret5d_max": 0.12,
    "v49_nb_position_size": 0.40,
    "v29_peak_lock_high_beta_only": False,
    "v29_profit_safety_net": False,
    "v29_profit_safety_trigger": 0.25,
    "v29_hardcap_after_peak": False,
    "v29_hardcap_after_peak_trigger": 0.15,
    "v29_hardcap_after_peak_floor": -0.03,
    # V30 patch flags
    "v30_peak_proximity_filter": False,
    "v30_peak_prox_dist_threshold": -0.02,
    "v30_peak_prox_rally10_min": 0.08,
    "v30_rally_extension_filter": False,
    "v30_rally10_hard_block": 0.12,
    "v30_rally20_hard_block": 0.18,
    "v30_pullback_only_entry": False,
    "v30_pullback_min_pct": 0.03,
    "v30_rally_position_scaling": False,
    "v30_rps_tier1_rally": 0.05,
    "v30_rps_tier2_rally": 0.10,
    "v30_signal_exit_defer": False,
    "v30_sed_defer_bars": 3,
    "v30_sed_min_cum_ret": 0.03,
    "v30_momentum_hold_override": False,
    "v30_mho_min_profit": 0.05,
    "v30_mho_rsi_max": 72,
    "v30_chandelier_trail": False,
    "v30_chand_atr_mult": 3.0,
    "v30_chand_profit_trigger": 0.05,
    "v30_atr_aware_hardcap": False,
    "v30_atr_hc_mult": 1.5,
    "v30_atr_hc_floor": 0.08,
    "v30_atr_hc_ceiling": 0.20,
    "v30_hardcap_two_step_v2": False,
    "v30_hc2_step1_loss": -0.04,
    "v30_hc2_step2_loss": -0.08,
    "v30_regime_aware_hardcap": False,
    "v30_rah_choppy_cap": -0.05,
    "v30_rah_trending_cap": -0.12,
    # V35 flags
    "v35_relax_cooldown": False,
    "v35_cooldown_after_big_loss": 1,
    "v35_cooldown_after_loss": 0,
    "v35_skip_price_proximity": False,
    "v35_single_bar_signal": False,
    "v35_rule_override": False,
    "v35_rule_override_min_score": 1,
    "v35_hybrid_entry": False,
    "v35_hybrid_size": 0.5,
}
