"""
Xem các trades cuối cùng của backtest (tháng 12/2025)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADES_FILE = (
    ROOT
    / "results/experiments/v22_exit_ablation_round42/v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only/trades.csv"
)

df = pd.read_csv(TRADES_FILE)
df["entry_date"] = pd.to_datetime(df["entry_date"])
df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")

# Trades tháng 12/2025
dec_2025 = df[df["entry_date"] >= "2025-12-01"].sort_values("entry_date", ascending=False)

print("=" * 80)
print(f"TRADES THANG 12/2025: {len(dec_2025)} trades")
print("=" * 80)
print(f"\n{'Symbol':<8} {'Entry':<12} {'Exit':<12} {'PnL %':<10} {'Days':<6} {'Status':<8}")
print("-" * 80)

for _, row in dec_2025.iterrows():
    entry = row["entry_date"].strftime("%Y-%m-%d")
    exit_str = "OPEN" if pd.isna(row["exit_date"]) else row["exit_date"].strftime("%Y-%m-%d")
    status = "MO" if pd.isna(row["exit_date"]) else "DONG"
    pnl = row.get("pnl_pct", 0)
    days = row.get("holding_days", 0)
    print(f"{row['symbol']:<8} {entry:<12} {exit_str:<12} {pnl:>9.2f} {days:>6} {status:<8}")

open_trades = dec_2025[dec_2025["exit_date"].isna()]
print(f"\n=> Dang mo: {len(open_trades)} trades")
if len(open_trades) > 0:
    for _, row in open_trades.iterrows():
        print(
            f"   {row['symbol']}: Entry {row['entry_date'].strftime('%Y-%m-%d')} @ {row['entry_price']:.2f}"
        )
print("=" * 80)
