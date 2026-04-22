"""
Visualize V12 trades on price chart + audit PnL calculations
"""
import sys, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v12_compare import backtest_v12


def run_and_visualize(symbol="VND"):
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

    raw_df = loader.load_all(symbols=[symbol])
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    all_trades = []
    all_test_dfs = []

    for window, train_df, test_df in splitter.split(df):
        model = build_model("lightgbm")
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        sym_test = test_df[test_df["symbol"] == symbol].reset_index(drop=True)
        if len(sym_test) < 10:
            continue
        X_sym = np.nan_to_num(sym_test[feature_cols].values)
        y_pred = model.predict(X_sym)
        rets = sym_test["return_1d"].values

        # Use Fix C only (best config)
        r = backtest_v12(y_pred, rets, sym_test, feature_cols,
                         fix_a=False, fix_b=False, fix_c=True, fix_d=False)
        
        # Add actual prices for verification
        for t in r["trades"]:
            entry_idx = t["entry_day"]
            exit_idx = t["exit_day"]
            t["entry_price"] = float(sym_test["close"].iloc[entry_idx])
            t["exit_price"] = float(sym_test["close"].iloc[exit_idx])
            t["calculated_pnl"] = round((t["exit_price"] / t["entry_price"] - 1) * 100, 2)
            t["window"] = window
            t["symbol"] = symbol

        all_trades.extend(r["trades"])
        all_test_dfs.append(sym_test)

    # ═══════════════════════════════════════
    # AUDIT: Check PnL accuracy
    # ═══════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"PnL AUDIT — {symbol} ({len(all_trades)} trades)")
    print(f"{'='*80}")
    
    discrepancies = []
    high_pnl = []
    
    for t in all_trades:
        diff = abs(t["pnl_pct"] - t["calculated_pnl"])
        if diff > 1.0:  # More than 1% difference
            discrepancies.append(t)
        if abs(t["pnl_pct"]) > 50:
            high_pnl.append(t)
    
    if discrepancies:
        print(f"\n⚠️  {len(discrepancies)} trades with PnL discrepancy > 1%:")
        print(f"{'Entry Date':<12} {'Exit Date':<12} {'Hold':<5} {'Recorded%':<10} {'Calculated%':<12} {'Entry$':<10} {'Exit$':<10} {'Reason'}")
        print("-" * 90)
        for t in discrepancies[:20]:
            print(f"{t.get('entry_date','?'):<12} {t.get('exit_date','?'):<12} "
                  f"{t['holding_days']:<5} {t['pnl_pct']:>+8.2f}%  {t['calculated_pnl']:>+8.2f}%   "
                  f"{t['entry_price']:>8.0f}  {t['exit_price']:>8.0f}  {t['exit_reason']}")
    else:
        print("\n✅ All trades PnL match price-based calculation (within 1%)")
    
    if high_pnl:
        print(f"\n🔍 {len(high_pnl)} trades with |PnL| > 50%:")
        print(f"{'Entry Date':<12} {'Exit Date':<12} {'Hold':<5} {'PnL%':<10} {'Entry$':<10} {'Exit$':<10} {'Reason'}")
        print("-" * 80)
        for t in high_pnl:
            print(f"{t.get('entry_date','?'):<12} {t.get('exit_date','?'):<12} "
                  f"{t['holding_days']:<5} {t['pnl_pct']:>+8.2f}%  "
                  f"{t['entry_price']:>8.0f}  {t['exit_price']:>8.0f}  {t['exit_reason']}")
    else:
        print("\n✅ No trades with |PnL| > 50%")

    # ═══════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════
    print(f"\n{'='*80}")
    print("ALL TRADES:")
    print(f"{'#':<3} {'Entry':<12} {'Exit':<12} {'Hold':<5} {'PnL%':<9} {'Entry$':<9} {'Exit$':<9} {'Reason':<15} {'Trend'}")
    print("-" * 100)
    for idx, t in enumerate(all_trades):
        flag = "⚠️" if abs(t["pnl_pct"]) > 50 else ("❌" if t["pnl_pct"] < -5 else ("✅" if t["pnl_pct"] > 10 else "  "))
        print(f"{idx+1:<3} {t.get('entry_date','?'):<12} {t.get('exit_date','?'):<12} "
              f"{t['holding_days']:<5} {t['pnl_pct']:>+7.2f}% {t['entry_price']:>8.0f} {t['exit_price']:>8.0f} "
              f"{t['exit_reason']:<15} {t.get('entry_trend','?')} {flag}")

    # ═══════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════
    print(f"\n📊 Generating chart...")
    
    # Combine all test periods
    full_df = pd.concat(all_test_dfs, ignore_index=True)
    date_col = "date" if "date" in full_df.columns else "timestamp"
    full_df[date_col] = pd.to_datetime(full_df[date_col])
    full_df = full_df.sort_values(date_col).reset_index(drop=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [3, 1]})
    
    # Price chart
    ax1 = axes[0]
    ax1.plot(full_df[date_col], full_df["close"], color="gray", linewidth=0.8, alpha=0.8, label="Close")
    
    # SMA
    sma20 = full_df["close"].rolling(20).mean()
    ax1.plot(full_df[date_col], sma20, color="blue", linewidth=0.5, alpha=0.5, label="SMA20")
    
    # Map trades to full_df dates
    for t in all_trades:
        entry_date = pd.to_datetime(t.get("entry_date"))
        exit_date = pd.to_datetime(t.get("exit_date"))
        entry_price = t["entry_price"]
        exit_price = t["exit_price"]
        pnl = t["pnl_pct"]
        
        # Entry marker (green triangle up)
        ax1.scatter(entry_date, entry_price, marker="^", color="green", s=80, zorder=5)
        
        # Exit marker
        if pnl >= 0:
            ax1.scatter(exit_date, exit_price, marker="v", color="blue", s=80, zorder=5)
        else:
            ax1.scatter(exit_date, exit_price, marker="v", color="red", s=100, zorder=5)
        
        # Connect entry-exit with line
        color = "green" if pnl >= 0 else "red"
        alpha = min(0.8, 0.3 + abs(pnl) / 50)
        ax1.plot([entry_date, exit_date], [entry_price, exit_price], 
                 color=color, linewidth=1.5, alpha=alpha)
        
        # Label big trades
        if abs(pnl) > 15:
            ax1.annotate(f"{pnl:+.0f}%", xy=(exit_date, exit_price),
                        fontsize=7, color=color, fontweight="bold",
                        xytext=(5, 10 if pnl > 0 else -15), textcoords="offset points")
    
    ax1.set_title(f"{symbol} — V12 Trades (Fix C)", fontsize=14)
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    
    # PnL per trade
    ax2 = axes[1]
    trade_dates = [pd.to_datetime(t.get("exit_date")) for t in all_trades]
    pnls = [t["pnl_pct"] for t in all_trades]
    colors = ["green" if p >= 0 else "red" for p in pnls]
    ax2.bar(trade_dates, pnls, color=colors, width=5, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("PnL %")
    ax2.set_title("Per-Trade PnL")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"v12_trades_{symbol}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="VND")
    args = parser.parse_args()
    run_and_visualize(args.symbol)
