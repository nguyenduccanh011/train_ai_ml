"""
Script đơn giản: Tìm tín hiệu mua gần đây từ kết quả backtest
Không cần chạy lại model, chỉ đọc file trades.csv
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Đường dẫn tuyệt đối
ROOT = Path(__file__).resolve().parents[1]
TRADES_FILE = (
    ROOT
    / "results/experiments/v22_exit_ablation_round42/v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only/trades.csv"
)


def main():
    print("=" * 80)
    print("TIM TIN HIEU MUA GAN DAY - TU KET QUA BACKTEST")
    print("=" * 80)

    if not TRADES_FILE.exists():
        print(f"Khong tim thay file: {TRADES_FILE}")
        return

    # Đọc trades
    df = pd.read_csv(TRADES_FILE)
    print(f"Tong so trades: {len(df)}")

    # Chuyển đổi date columns
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")

    # Lọc trades trong 90 ngày gần nhất
    cutoff_date = datetime.now() - timedelta(days=90)
    recent_df = df[df["entry_date"] >= cutoff_date].copy()

    print(f"Trades trong 90 ngay gan day: {len(recent_df)}")

    if recent_df.empty:
        print("\nKhong co trade nao trong 90 ngay gan day")
        print("Luu y: Backtest ket thuc 2025-12-31")
        return

    # Sắp xếp theo entry_date
    recent_df = recent_df.sort_values("entry_date", ascending=False)

    # Hiển thị
    print("\n" + "=" * 80)
    print("TRADES GAN DAY:")
    print("=" * 80)
    print(
        f"\n{'Symbol':<8} {'Entry Date':<12} {'Exit Date':<12} {'PnL %':<10} {'Hold Days':<10} {'Status':<10}"
    )
    print("-" * 80)

    for _, row in recent_df.head(50).iterrows():
        symbol = row["symbol"]
        entry = row["entry_date"].strftime("%Y-%m-%d")
        exit_date = row["exit_date"]

        if pd.isna(exit_date):
            exit_str = "OPEN"
            status = "MO"
        else:
            exit_str = exit_date.strftime("%Y-%m-%d")
            status = "DONG"

        pnl = row.get("pnl_pct", 0)
        hold = row.get("holding_days", 0)

        print(f"{symbol:<8} {entry:<12} {exit_str:<12} {pnl:>9.2f} {hold:>10} {status:<10}")

    # Thống kê
    open_trades = recent_df[recent_df["exit_date"].isna()]
    closed_trades = recent_df[~recent_df["exit_date"].isna()]

    print("\n" + "=" * 80)
    print("THONG KE:")
    print(f"  - Tong trades: {len(recent_df)}")
    print(f"  - Dang mo: {len(open_trades)}")
    print(f"  - Da dong: {len(closed_trades)}")

    if len(closed_trades) > 0:
        avg_pnl = closed_trades["pnl_pct"].mean()
        win_rate = (closed_trades["pnl_pct"] > 0).sum() / len(closed_trades) * 100
        print(f"  - Avg PnL (closed): {avg_pnl:.2f}%")
        print(f"  - Win rate: {win_rate:.1f}%")

    # Lưu file
    output = ROOT / "recent_trades_90days.csv"
    recent_df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\nDa luu vao: {output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
