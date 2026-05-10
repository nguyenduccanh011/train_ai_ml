from __future__ import annotations

from typing import Any

from src.backtest.defaults import SYMBOL_PROFILES
from src.backtest.engine import backtest_unified

V28_CONFIG: dict[str, Any] = {
    "patch_smart_hardcap": True,
    "patch_pp_restore": True,
    "patch_long_horizon": True,
    "patch_symbol_tuning": True,
    "patch_rule_ensemble": True,
    "patch_noise_filter": False,
    "patch_adaptive_hardcap": False,
    "patch_pp_2of3": False,
    "v26_wider_hardcap": False,
    "v26_relaxed_entry": True,
    "v26_skip_choppy": True,
    "v26_extended_hold": False,
    "v26_strong_rule_ensemble": True,
    "v26_min_position": False,
    "v26_score5_penalty": False,
    "v26_hardcap_confirm_strong": False,
    "v27_selective_choppy": False,
    "v27_hardcap_two_step": True,
    "v27_rule_priority": True,
    "v27_dynamic_score5_penalty": True,
    "v27_trend_persistence_hold": True,
    "v28_early_wave_filter": True,
    "v28_crash_guard": False,
    "v28_wave_acceleration_entry": False,
    "v28_early_loss_cut": True,
    "v28_cycle_peak_exit": False,
    "v28_early_loss_cut_threshold": -0.04,
    "v28_early_loss_cut_days": 5,
}

V30_DELTA: dict[str, Any] = {
    "v30_signal_exit_defer": True,
    "v30_sed_defer_bars": 3,
    "v30_sed_min_cum_ret": 0.02,
    "v30_peak_proximity_filter": False,
    "v30_rally_extension_filter": False,
    "v30_pullback_only_entry": False,
    "v30_rally_position_scaling": False,
    "v30_momentum_hold_override": False,
    "v30_chandelier_trail": False,
    "v30_atr_aware_hardcap": False,
    "v30_hardcap_two_step_v2": False,
    "v30_regime_aware_hardcap": False,
}

V31_DELTA: dict[str, Any] = {
    "v31_short_hold_exit_filter": True,
    "v31_shef_min_hold": 18,
    "v31_shef_min_pnl": -0.03,
    "v31_hardcap_after_profit": True,
    "v31_hap_profit_trigger": 0.05,
    "v31_hap_floor": -0.03,
    "v31_peak_chasing_guard": False,
    "v31_adaptive_defer": False,
    "v31_profile_sizing": False,
    "v31_enriched_log": True,
}

V32_DELTA: dict[str, Any] = {
    "v32_hap_preempt": True,
    "v32_hap_pre_trigger": 0.05,
    "v32_hap_pre_floor": -0.05,
    "v32_weak_oversold_exit": True,
    "v32_woe_dist_thresh": -0.09,
    "v32_woe_min_profit": 0.0,
    "v32_woe_hold_min": 5,
    "v32_dynamic_hc_dist": False,
    "v32_profit_ratchet": False,
    "v32_signal_weak_exit": False,
    "v31_enriched_log": True,
}

V37A_RELAX_PROFILES = {"bank", "defensive", "balanced"}

V37A_RELAX_FLAGS: dict[str, Any] = {
    "v35_rule_override": True,
    "v35_rule_override_min_score": 1,
    "v35_skip_price_proximity": True,
    "v35_relax_cooldown": True,
    "v35_cooldown_after_big_loss": 2,
    "v35_cooldown_after_loss": 1,
}

V39D_RULE_EXIT_SYMBOLS = {
    "PVS",
    "AAS",
    "BSR",
    "PVD",
    "KBC",
    "AAV",
    "GAS",
    "FRT",
    "BCM",
    "PLX",
    "SBT",
    "BID",
}

V39D_FLAGS: dict[str, Any] = {
    "v39a_signal_exit_min_hold": 35,
    "v39b_hap_trigger": 0.08,
    "v39b_hap_min_hold": 15,
    "v39d_rule_exit_symbols": V39D_RULE_EXIT_SYMBOLS,
}


def backtest_v28(
    y_pred: Any, returns: Any, df_test: Any, feature_cols: Any, **kwargs: Any
) -> dict[str, Any]:
    merged = {**V28_CONFIG, **kwargs}
    return backtest_unified(y_pred, returns, df_test, feature_cols, **merged)


def backtest_v29(
    y_pred: Any, returns: Any, df_test: Any, feature_cols: Any, **kwargs: Any
) -> dict[str, Any]:
    return backtest_v28(y_pred, returns, df_test, feature_cols, **kwargs)


def backtest_v30(
    y_pred: Any, returns: Any, df_test: Any, feature_cols: Any, **kwargs: Any
) -> dict[str, Any]:
    merged = {**V30_DELTA, **kwargs}
    return backtest_v29(y_pred, returns, df_test, feature_cols, **merged)


def backtest_v31(
    y_pred: Any, returns: Any, df_test: Any, feature_cols: Any, **kwargs: Any
) -> dict[str, Any]:
    merged = {**V31_DELTA, **kwargs}
    return backtest_v30(y_pred, returns, df_test, feature_cols, **merged)


def backtest_v32(
    y_pred: Any, returns: Any, df_test: Any, feature_cols: Any, **kwargs: Any
) -> dict[str, Any]:
    merged = {**V32_DELTA, **kwargs}
    return backtest_v31(y_pred, returns, df_test, feature_cols, **merged)


def backtest_v34(
    y_pred: Any, returns: Any, df_test: Any, feature_cols: Any, **kwargs: Any
) -> dict[str, Any]:
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)


def backtest_v35b(
    y_pred: Any, returns: Any, df_test: Any, feature_cols: Any, **kwargs: Any
) -> dict[str, Any]:
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)


def backtest_v37a(
    y_pred: Any, returns: Any, df_test: Any, feature_cols: Any, **kwargs: Any
) -> dict[str, Any]:
    sym = "?"
    if "symbol" in df_test.columns and len(df_test) > 0:
        sym = str(df_test["symbol"].iloc[0])
    profile = SYMBOL_PROFILES.get(sym, "balanced")

    merged = dict(kwargs)
    if profile in V37A_RELAX_PROFILES:
        merged.update(V37A_RELAX_FLAGS)
    else:
        for key, value in V37A_RELAX_FLAGS.items():
            merged[key] = False if isinstance(value, bool) else merged.get(key)

    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)


def backtest_v37d(
    y_pred: Any, returns: Any, df_test: Any, feature_cols: Any, **kwargs: Any
) -> dict[str, Any]:
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)


def backtest_v39d(
    y_pred: Any, returns: Any, df_test: Any, feature_cols: Any, **kwargs: Any
) -> dict[str, Any]:
    merged = {**kwargs, **V39D_FLAGS}
    return backtest_v37a(y_pred, returns, df_test, feature_cols, **merged)


def backtest_v42(
    y_pred: Any, returns: Any, df_test: Any, feature_cols: Any, **kwargs: Any
) -> dict[str, Any]:
    return backtest_v37a(y_pred, returns, df_test, feature_cols, **kwargs)
