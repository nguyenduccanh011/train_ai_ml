"""V37a — Per-profile V35 flag activation.

Insight: V35b flags (rule_override + relax_cooldown + skip_price_proximity)
giúp bank/defensive (slow movers) bắt sóng sớm hơn, nhưng bị fomo trên
momentum/high_beta. V37a bật V35 flags chỉ cho profile phù hợp.

V37a = V34 engine + per-profile dispatch:
  - bank/defensive/balanced → V35b flags ON (rule_override + skip_proximity + relax_cooldown(2,1))
  - momentum/high_beta     → V34 engine giữ nguyên
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32
from src.backtest.defaults import SYMBOL_PROFILES


V37A_RELAX_PROFILES = {"bank", "defensive", "balanced"}

V37A_RELAX_FLAGS = dict(
    v35_rule_override=True,
    v35_rule_override_min_score=1,
    v35_skip_price_proximity=True,
    v35_relax_cooldown=True,
    v35_cooldown_after_big_loss=2,
    v35_cooldown_after_loss=1,
)


def backtest_v37a(y_pred, returns, df_test, feature_cols, **kwargs):
    """Per-profile dispatch: relax flags only for slow-mover profiles."""
    sym = "?"
    if "symbol" in df_test.columns and len(df_test) > 0:
        sym = str(df_test["symbol"].iloc[0])
    profile = SYMBOL_PROFILES.get(sym, "balanced")

    merged = dict(kwargs)
    if profile in V37A_RELAX_PROFILES:
        for k, v in V37A_RELAX_FLAGS.items():
            merged.setdefault(k, v)
            merged[k] = v
    else:
        # Force V35 flags OFF for momentum/high_beta (keep V34 strict behavior)
        for k in V37A_RELAX_FLAGS:
            merged[k] = False if isinstance(V37A_RELAX_FLAGS[k], bool) else merged.get(k)

    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)
