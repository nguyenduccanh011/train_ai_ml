"""Analyze VND V11 trades in detail — find losing/inefficient trades."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v11_compare import backtest_v11

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "portable_data", "vn_stock_ai_dataset_cleaned")

config = {
    "data": {"data_dir": data_dir},
    "split": {"method": "walk_forward", "train_years": 4, "test_years": 1,
              "gap_days": 0, "first_test_year": 2020, "last_test_year": 2025},
    "target": {"type": "trend_regime", "trend_method": "dual_ma",
               "short_window": 5, "long_window": 20, "classes": 3},
}

loader = DataLoader(data_dir)
splitter = WalkForwardSplitter.from_config(config)
target_gen = TargetGenerator.from_config(config)

raw_df = loader.load_all(symbols=["VND"])
engine = FeatureEngine(feature_set="leading")
df = engine.compute_for_all_symbols(raw_df)
df = target_gen.generate_for_all_symbols(df)
feature_cols = engine.get_feature_columns(df)
df = df.dropna(subset=feature_cols + ["target"])

all_trades = []
for window, train_df, test_df in splitter.split(df):
    model = build_model("lightgbm")
    X_train = np.nan_to_num(train_df[feature_cols].values)
    y_train = train_df["target"].values.astype(int)
    model.fit(X_train, y_train)

    sym_test = test_df[test_df["symbol"] == "VND"].reset_index(drop=True)
    if len(sym_test) < 10:
        continue
    X_sym = np.nan_to_num(sym_test[feature_cols].values)
    y_pred = model.predict(X_sym)
    rets = sym_test["return_1d"].values

    r = backtest_v11(y_pred, rets, sym_test, feature_cols)
    for t in r["trades"]:
        t["window"] = window.label
    all_trades.extend(r["trades"])

# Sort by PnL
all_trades.sort(key=lambda x: x["pnl_pct"])

print("=" * 120)
print(f"VND V11: {len(all_trades)} trades total")
print("=" * 120)

# Losing trades
losers = [t for t in all_trades if t["pnl_pct"] < 0]
marginals = [t for t in all_trades if -2 <= t["pnl_pct"] <= 2]
print(f"\nLOSING TRADES ({len(losers)}):")
print(f"{'Entry Date':<12} {'Exit Date':<12} {'Days':>4} {'PnL%':>7} {'MaxP%':>6} {'Exit':>14} "
      f"{'Trend':>8} {'WP':>5} {'RS':>5} {'VS':>5} {'BS':>3} {'HL':>3} {'Score':>5} "
      f"{'Ret5d':>6} {'Drop20':>7} {'DistMA20':>8} {'PosSize':>7} {'Breakout':>8}")
print("-" * 160)
for t in losers:
    print(f"{t.get('entry_date','?'):<12} {t.get('exit_date','?'):<12} {t.get('holding_days',0):>4} "
          f"{t['pnl_pct']:>+6.1f}% {t.get('max_profit_pct',0):>+5.1f}% {t.get('exit_reason','?'):>14} "
          f"{t.get('entry_trend','?'):>8} {t.get('entry_wp',0):>5.2f} {t.get('entry_rs',0):>5.2f} "
          f"{t.get('entry_vs',0):>5.2f} {t.get('entry_bs',0):>3.0f} {t.get('entry_hl',0):>3.0f} "
          f"{t.get('entry_score',0):>5} {t.get('entry_ret_5d',0):>+5.1f}% "
          f"{t.get('entry_drop20d',0):>+6.1f}% {t.get('entry_dist_sma20',0):>+7.1f}% "
          f"{t.get('position_size',1):>6.2f} {str(t.get('breakout_entry',False)):>8}")

print(f"\nMARGINAL TRADES (-2% to +2%): {len(marginals)}")
print(f"{'Entry Date':<12} {'Exit Date':<12} {'Days':>4} {'PnL%':>7} {'MaxP%':>6} {'Exit':>14} "
      f"{'Trend':>8} {'WP':>5} {'RS':>5} {'VS':>5} {'BS':>3} {'HL':>3} {'Score':>5} "
      f"{'Ret5d':>6} {'Drop20':>7} {'DistMA20':>8}")
print("-" * 160)
for t in marginals:
    print(f"{t.get('entry_date','?'):<12} {t.get('exit_date','?'):<12} {t.get('holding_days',0):>4} "
          f"{t['pnl_pct']:>+6.1f}% {t.get('max_profit_pct',0):>+5.1f}% {t.get('exit_reason','?'):>14} "
          f"{t.get('entry_trend','?'):>8} {t.get('entry_wp',0):>5.2f} {t.get('entry_rs',0):>5.2f} "
          f"{t.get('entry_vs',0):>5.2f} {t.get('entry_bs',0):>3.0f} {t.get('entry_hl',0):>3.0f} "
          f"{t.get('entry_score',0):>5} {t.get('entry_ret_5d',0):>+5.1f}% "
          f"{t.get('entry_drop20d',0):>+6.1f}% {t.get('entry_dist_sma20',0):>+7.1f}%")

# Summary statistics for losers
if losers:
    print(f"\n{'='*80}")
    print("PATTERN ANALYSIS - LOSING TRADES:")
    print(f"  Avg entry_wp (range_position): {np.mean([t.get('entry_wp',0) for t in losers]):.3f}")
    print(f"  Avg entry_rs (rsi_slope):      {np.mean([t.get('entry_rs',0) for t in losers]):.3f}")
    print(f"  Avg entry_vs (vol_surge):      {np.mean([t.get('entry_vs',0) for t in losers]):.3f}")
    print(f"  Avg entry_score:               {np.mean([t.get('entry_score',0) for t in losers]):.2f}")
    print(f"  Avg entry_ret_5d:              {np.mean([t.get('entry_ret_5d',0) for t in losers]):.2f}%")
    print(f"  Avg entry_drop20d:             {np.mean([t.get('entry_drop20d',0) for t in losers]):.2f}%")
    print(f"  Avg entry_dist_sma20:          {np.mean([t.get('entry_dist_sma20',0) for t in losers]):.2f}%")
    print(f"  Avg holding_days:              {np.mean([t.get('holding_days',0) for t in losers]):.1f}")
    print(f"  Avg max_profit_pct:            {np.mean([t.get('max_profit_pct',0) for t in losers]):.2f}%")
    trends = [t.get('entry_trend','?') for t in losers]
    from collections import Counter
    print(f"  Trend distribution:            {dict(Counter(trends))}")
    exits = [t.get('exit_reason','?') for t in losers]
    print(f"  Exit reasons:                  {dict(Counter(exits))}")

# Winners for comparison
winners = [t for t in all_trades if t["pnl_pct"] > 5]
if winners:
    print(f"\nPATTERN ANALYSIS - WINNING TRADES (>{5}%):")
    print(f"  Avg entry_wp (range_position): {np.mean([t.get('entry_wp',0) for t in winners]):.3f}")
    print(f"  Avg entry_rs (rsi_slope):      {np.mean([t.get('entry_rs',0) for t in winners]):.3f}")
    print(f"  Avg entry_vs (vol_surge):      {np.mean([t.get('entry_vs',0) for t in winners]):.3f}")
    print(f"  Avg entry_score:               {np.mean([t.get('entry_score',0) for t in winners]):.2f}")
    print(f"  Avg entry_ret_5d:              {np.mean([t.get('entry_ret_5d',0) for t in winners]):.2f}%")
    print(f"  Avg entry_drop20d:             {np.mean([t.get('entry_drop20d',0) for t in winners]):.2f}%")
    print(f"  Avg entry_dist_sma20:          {np.mean([t.get('entry_dist_sma20',0) for t in winners]):.2f}%")
    print(f"  Avg holding_days:              {np.mean([t.get('holding_days',0) for t in winners]):.1f}")
    trends_w = [t.get('entry_trend','?') for t in winners]
    print(f"  Trend distribution:            {dict(Counter(trends_w))}")
