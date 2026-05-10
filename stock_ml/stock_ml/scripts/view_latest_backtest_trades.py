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


def main():
    print("=" * 80)
    print("TRADES CUOI CUNG CUA BACKTEST (THANG 12/2025)")
    print("=" * 80)

    df = pd.read_csv(TRADES_FILE)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")

    # Lọc trades entry trong tháng 12/2025
    dec_2025 = df[df["entry_date"] >= "2025-12-01"].copy()
    dec_2025 = dec_2025.sort_values("entry_date", ascending=False)

    print(f"\nTong trades entry trong thang 12/2025: {len(dec_2025)}")
    print("\n" + "=" * 80)
    print(
        f"{'Symbol':<8} {'Entry':<12} {'Exit':<12} {'Entry $':<10} {'Exit $':<10} {'PnL %':<10} {'Days':<6} {'Status':<8}"
    )
    print("-" * 90)

    for _, row in dec_2025.iterrows():
        symbol = row["symbol"]
        entry = row["entry_date"].strftime("%Y-%m-%d")
        entry_price = row.get("entry_price", 0)
        exit_price = row.get("exit_price", 0)
        pnl = row.get("pnl_pct", 0)
        days = row.get("holding_days", 0)

        if pd.isna(row["exit_date"]):
            exit_str = "OPEN"
            status = "MO"
        else:
            exit_str = row["exit_date"].strftime("%Y-%m-%d")
            status = "DONG"

        print(
            f"{symbol:<8} {entry:<12} {exit_str:<12} {entry_price:<10.2f} {exit_price:<10.2f} {pnl:>9.2f} {days:>6} {status:<8}"
        )

    # Stats
    open_trades = dec_2025[dec_2025["exit_date"].isna()]
    closed_trades = dec_2025[~dec_2025["exit_date"].isna()]

    print("\n" + "=" * 80)
    print("THONG KE:")
    print(f"  - Tong: {len(dec_2025)}")
    print(f"  - Dang mo (chua dong trong backtest): {len(open_trades)}")
    print(f"  - Da dong: {len(closed_trades)}")

    if len(open_trades) > 0:
        print("\nCAC MA DANG MO (entry thang 12, chua co exit trong backtest):")
        for _, row in open_trades.iterrows():
            print(
                f"  - {row['symbol']}: Entry {row['entry_date'].strftime('%Y-%m-%d')} @ {row['entry_price']:.2f}"
            )

    if len(closed_trades) > 0:
        avg_pnl = closed_trades["pnl_pct"].mean()
        win_rate = (closed_trades["pnl_pct"] > 0).sum() / len(closed_trades) * 100
        print("\nTHONG KE TRADES DA DONG:")
        print(f"  - Avg PnL: {avg_pnl:.2f}%")
        print(f"  - Win rate: {win_rate:.1f}%")

    print("=" * 80)
    print("\nLUU Y:")
    print("- Backtest ket thuc 2025-12-31")
    print("- Cac ma 'DANG MO' la cac entry chua co exit trong backtest window")
    print("- De xem tin hieu realtime 2026, dung server: http://127.0.0.1:5002")
    print("=" * 80)


if __name__ == "__main__":
    main()
