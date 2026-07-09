"""
Script đơn giản: Tìm tín hiệu mua chưa bán từ các file prediction đã có
Nhanh hơn vì không cần chạy lại model
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_open_signals_from_results():
    """Tìm open signals từ kết quả backtest đã có"""
    results_dir = ROOT / "results" / "experiments" / "v22_exit_ablation_round42"

    # Tìm thư mục kết quả của model top 1
    model_dirs = list(
        results_dir.glob(
            "v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only"
        )
    )

    if not model_dirs:
        print("Khong tim thay ket qua model top 1")
        return []

    model_dir = model_dirs[0]
    trades_file = model_dir / "trades.csv"

    if not trades_file.exists():
        print(f"Khong tim thay file trades: {trades_file}")
        return []

    print(f"Doc file trades: {trades_file}")
    df = pd.read_csv(trades_file)

    print(f"Tong so trades: {len(df)}")

    # Lọc các trade chưa có exit_date (đang mở)
    open_trades = df[df["exit_date"].isna() | (df["exit_date"] == "")]

    print(f"Trades dang mo: {len(open_trades)}")

    if open_trades.empty:
        return []

    # Sắp xếp theo entry_date
    open_trades = open_trades.sort_values("entry_date", ascending=False)

    results = []
    for _, trade in open_trades.iterrows():
        results.append(
            {
                "symbol": trade["symbol"],
                "entry_date": trade["entry_date"],
                "entry_price": trade.get("entry_price", 0),
                "position_size": trade.get("position_size", 0),
            }
        )

    return results


def main():
    print("\n" + "=" * 80)
    print("TIM TIN HIEU MUA CHUA BAN - TU KET QUA BACKTEST")
    print("=" * 80)
    print("Model: v22_exit_ablation_round42 (Top 1)")
    print("=" * 80)

    open_signals = find_open_signals_from_results()

    if not open_signals:
        print("\nKhong tim thay tin hieu nao dang mo trong ket qua backtest")
        print("\nLuu y: Backtest ket thuc vao 2025, nen khong co open signals")
        print("De tim open signals realtime, can chay model tren du lieu moi nhat")
        return

    print(f"\nTim thay {len(open_signals)} tin hieu dang mo:")
    print("-" * 80)
    print(f"{'Symbol':<10} {'Entry Date':<15} {'Entry Price':<15} {'Position Size':<15}")
    print("-" * 80)

    for signal in open_signals:
        print(
            f"{signal['symbol']:<10} {signal['entry_date']:<15} "
            f"{signal['entry_price']:<15.2f} {signal['position_size']:<15.3f}"
        )

    # Save to CSV
    output_file = ROOT / "open_signals_from_backtest.csv"
    pd.DataFrame(open_signals).to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nDa luu vao: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
