"""
Compare Rule-based strategy vs V9 ML Model
Rule: BUY when MACD_hist > 0 AND Close > MA20 AND Close > Open
      SELL when MACD_hist < 0 AND Close < MA20 AND Close < Open
"""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v9_compare import backtest_v9


def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = pd.Series(close).ewm(span=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def backtest_rule(df_sym, commission=0.0015, tax=0.001):
    """Rule-based: BUY when MACD_hist>0 + C>MA20 + C>O, SELL opposite."""
    close = df_sym["close"].values
    opn = df_sym["open"].values
    n = len(close)

    ma20 = pd.Series(close).rolling(20, min_periods=10).mean().values
    _, _, macd_hist = compute_macd(close)

    date_col = "timestamp" if "timestamp" in df_sym.columns else "date"
    dates = df_sym[date_col].values if date_col in df_sym.columns else np.arange(n)

    position = 0
    trades = []
    entry_price = 0
    entry_idx = 0

    for i in range(26, n):  # start after MACD warm-up
        buy_signal = macd_hist[i] > 0 and close[i] > ma20[i] and close[i] > opn[i]
        sell_signal = macd_hist[i] < 0 and close[i] < ma20[i] and close[i] < opn[i]

        if position == 0 and buy_signal:
            position = 1
            entry_price = close[i]
            entry_idx = i
        elif position == 1 and sell_signal:
            exit_price = close[i]
            pnl_pct = (exit_price / entry_price - 1) * 100
            # subtract commission+tax
            pnl_pct -= (commission + tax) * 2 * 100
            trades.append({
                "entry_date": str(dates[entry_idx])[:10],
                "exit_date": str(dates[i])[:10],
                "holding_days": i - entry_idx,
                "pnl_pct": round(pnl_pct, 2),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "exit_reason": "rule_sell",
            })
            position = 0

    return trades


def summarize_trades(trades, label=""):
    if not trades:
        return None
    n = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    pnls = [t["pnl_pct"] for t in trades]
    total = sum(pnls)
    avg = np.mean(pnls)
    wr = len(wins) / n * 100
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    avg_hold = np.mean([t["holding_days"] for t in trades])
    max_loss = min(pnls)
    pf = abs(sum(t["pnl_pct"] for t in wins)) / abs(sum(t["pnl_pct"] for t in losses)) if losses and sum(t["pnl_pct"] for t in losses) != 0 else 99

    return {
        "label": label, "trades": n, "wr": wr, "avg_pnl": avg,
        "total_pnl": total, "avg_win": avg_win, "avg_loss": avg_loss,
        "avg_hold": avg_hold, "max_loss": max_loss, "pf": pf,
    }


def print_summary(s):
    if not s:
        print("  No trades")
        return
    print(f"  {s['label']}:")
    print(f"    Trades: {s['trades']}  WR: {s['wr']:.1f}%  PF: {s['pf']:.2f}")
    print(f"    Avg PnL: {s['avg_pnl']:+.2f}%  Total: {s['total_pnl']:+.1f}%")
    print(f"    Avg Win: {s['avg_win']:+.2f}%  Avg Loss: {s['avg_loss']:+.2f}%  Max Loss: {s['max_loss']:+.2f}%")
    print(f"    Avg Hold: {s['avg_hold']:.1f}d")


def main():
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

    # Test on multiple symbols
    test_symbols = ["VND", "AAS", "BID", "SSI", "FPT", "ACB", "MBB", "DGC"]
    symbols = [s for s in test_symbols if s in loader.symbols]

    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set="leading_v2")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    # Collect V9 trades
    v9_by_sym = defaultdict(list)
    for window, train_df, test_df in splitter.split(df):
        model = build_model("lightgbm")
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        for sym in test_df["symbol"].unique():
            if sym not in symbols:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            rets = sym_test["return_1d"].values if "return_1d" in sym_test.columns else np.zeros(len(sym_test))
            r9 = backtest_v9(y_pred, rets, sym_test, feature_cols)
            for t in r9["trades"]:
                t["symbol"] = sym
            v9_by_sym[sym].extend(r9["trades"])

    # Collect Rule-based trades (on same test period 2020-2025)
    rule_by_sym = defaultdict(list)
    for sym in symbols:
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        # Filter to test period (2020+)
        sym_data[date_col] = pd.to_datetime(sym_data[date_col])
        sym_test = sym_data[sym_data[date_col] >= "2020-01-01"].reset_index(drop=True)
        if len(sym_test) < 50:
            continue
        trades = backtest_rule(sym_test)
        rule_by_sym[sym] = trades

    # Print comparison
    print("=" * 130)
    print("📊 RULE-BASED (MACD+MA20+Candle) vs V9 ML MODEL — 2020-2025")
    print("=" * 130)
    print("Rule: BUY when MACD_hist > 0 AND Close > MA20 AND Close > Open")
    print("      SELL when MACD_hist < 0 AND Close < MA20 AND Close < Open")
    print()

    all_rule, all_v9 = [], []

    print(f"{'Symbol':<8} │ {'Rule #':>6} {'Rule WR':>8} {'Rule Avg':>9} {'Rule Tot':>9} {'Rule MaxDD':>10} │ "
          f"{'V9 #':>5} {'V9 WR':>7} {'V9 Avg':>8} {'V9 Tot':>8} {'V9 MaxDD':>9} │ {'Winner':>8}")
    print("─" * 130)

    for sym in symbols:
        rt = rule_by_sym.get(sym, [])
        vt = v9_by_sym.get(sym, [])
        all_rule.extend(rt)
        all_v9.extend(vt)

        rs = summarize_trades(rt, "Rule")
        vs = summarize_trades(vt, "V9")

        rn = rs["trades"] if rs else 0
        rwr = rs["wr"] if rs else 0
        ravg = rs["avg_pnl"] if rs else 0
        rtot = rs["total_pnl"] if rs else 0
        rdd = rs["max_loss"] if rs else 0

        vn = vs["trades"] if vs else 0
        vwr = vs["wr"] if vs else 0
        vavg = vs["avg_pnl"] if vs else 0
        vtot = vs["total_pnl"] if vs else 0
        vdd = vs["max_loss"] if vs else 0

        winner = "V9 ✅" if vavg > ravg else "Rule ✅"
        print(f"{sym:<8} │ {rn:>6} {rwr:>7.1f}% {ravg:>+8.2f}% {rtot:>+8.1f}% {rdd:>+9.1f}% │ "
              f"{vn:>5} {vwr:>6.1f}% {vavg:>+7.2f}% {vtot:>+7.1f}% {vdd:>+8.1f}% │ {winner:>8}")

    # Aggregate
    print("\n" + "=" * 130)
    print("📊 AGGREGATE COMPARISON")
    print("=" * 130)

    rs = summarize_trades(all_rule, "Rule-Based (MACD+MA20+Candle)")
    vs = summarize_trades(all_v9, "V9 ML Model (LightGBM + EMA 5/20)")
    print_summary(rs)
    print()
    print_summary(vs)

    if rs and vs:
        print(f"\n{'═' * 80}")
        print(f"📈 VERDICT:")
        print(f"{'═' * 80}")
        print(f"  Trades:    Rule={rs['trades']:4d}  vs  V9={vs['trades']:4d}")
        print(f"  WR:        Rule={rs['wr']:5.1f}%  vs  V9={vs['wr']:5.1f}%  {'V9 ✅' if vs['wr'] > rs['wr'] else 'Rule ✅'}")
        print(f"  Avg PnL:   Rule={rs['avg_pnl']:+.2f}%  vs  V9={vs['avg_pnl']:+.2f}%  {'V9 ✅' if vs['avg_pnl'] > rs['avg_pnl'] else 'Rule ✅'}")
        print(f"  Total PnL: Rule={rs['total_pnl']:+.1f}%  vs  V9={vs['total_pnl']:+.1f}%  {'V9 ✅' if vs['total_pnl'] > rs['total_pnl'] else 'Rule ✅'}")
        print(f"  PF:        Rule={rs['pf']:.2f}  vs  V9={vs['pf']:.2f}  {'V9 ✅' if vs['pf'] > rs['pf'] else 'Rule ✅'}")
        print(f"  Avg Loss:  Rule={rs['avg_loss']:+.2f}%  vs  V9={vs['avg_loss']:+.2f}%  {'V9 ✅' if vs['avg_loss'] > rs['avg_loss'] else 'Rule ✅'}")
        print(f"  Max Loss:  Rule={rs['max_loss']:+.1f}%  vs  V9={vs['max_loss']:+.1f}%  {'V9 ✅' if vs['max_loss'] > rs['max_loss'] else 'Rule ✅'}")
        print(f"  Avg Hold:  Rule={rs['avg_hold']:.1f}d  vs  V9={vs['avg_hold']:.1f}d")

    # Detailed trade-by-trade for VND (top performer)
    print(f"\n{'═' * 130}")
    print(f"📋 DETAILED TRADE COMPARISON — VND (Top V9 performer)")
    print(f"{'═' * 130}")
    print("\n--- Rule-Based Trades ---")
    for i, t in enumerate(rule_by_sym.get("VND", [])[:15]):
        cls = "WIN" if t["pnl_pct"] > 0 else "LOSS"
        print(f"  {i+1:2d}. [{cls:4s}] {t['entry_date']} → {t['exit_date']}  "
              f"Hold:{t['holding_days']:3d}d  PnL:{t['pnl_pct']:+7.2f}%  "
              f"Entry:{t['entry_price']:.2f} Exit:{t['exit_price']:.2f}")

    print("\n--- V9 ML Trades ---")
    for i, t in enumerate(v9_by_sym.get("VND", [])[:15]):
        cls = "WIN" if t["pnl_pct"] > 0 else "LOSS"
        print(f"  {i+1:2d}. [{cls:4s}] {t.get('entry_date','')} → {t.get('exit_date','')}  "
              f"Hold:{t['holding_days']:3d}d  PnL:{t['pnl_pct']:+7.2f}%  "
              f"Reason:{t.get('exit_reason','')}")


if __name__ == "__main__":
    main()
