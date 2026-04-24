"""Phase A1: per-symbol comparison V34 vs Rule.

Group trades by symbol, compute metrics + composite score for each,
output CSV + log top symbols where V34 underperforms Rule.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.evaluation.scoring import calc_metrics, composite_score


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "entry_symbol" in df.columns and "symbol" in df.columns:
        df = df.drop(columns=["entry_symbol"])
    elif "entry_symbol" in df.columns:
        df = df.rename(columns={"entry_symbol": "symbol"})
    return df


def trades_to_dicts(df: pd.DataFrame) -> list:
    cols = ["pnl_pct", "entry_date", "symbol", "holding_days"]
    sub = df[[c for c in cols if c in df.columns]].copy()
    return sub.to_dict("records")


def per_symbol_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for sym, g in df.groupby("symbol"):
        trades = trades_to_dicts(g)
        m = calc_metrics(trades)
        cs = composite_score(m, trades)
        rows.append({
            "symbol": sym,
            f"n_{label}": m["trades"],
            f"wr_{label}": m["wr"],
            f"avg_pnl_{label}": m["avg_pnl"],
            f"total_pnl_{label}": m["total_pnl"],
            f"pf_{label}": m["pf"],
            f"max_loss_{label}": m["max_loss"],
            f"comp_{label}": cs,
        })
    return pd.DataFrame(rows)


def main():
    res_dir = ROOT / "results"
    out_dir = ROOT / "analysis" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    v34 = load_trades(res_dir / "trades_v34.csv")
    rule = load_trades(res_dir / "trades_rule.csv")

    print(f"V34: {len(v34)} trades, {v34['symbol'].nunique()} symbols")
    print(f"Rule: {len(rule)} trades, {rule['symbol'].nunique()} symbols")

    m_v34 = per_symbol_metrics(v34, "v34")
    m_rule = per_symbol_metrics(rule, "rule")

    merged = m_v34.merge(m_rule, on="symbol", how="outer").fillna(0)
    merged["comp_diff"] = merged["comp_v34"] - merged["comp_rule"]
    merged["pnl_diff"] = merged["total_pnl_v34"] - merged["total_pnl_rule"]
    merged["v34_loses"] = (merged["comp_diff"] < 0) & (merged["n_rule"] > 0) & (merged["n_v34"] > 0)

    merged = merged.sort_values("comp_diff")
    out_path = out_dir / "v34_vs_rule_symbol.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[OK] Wrote {out_path} ({len(merged)} symbols)")

    # Filter symbols where both have trades, V34 loses on comp
    losers = merged[merged["v34_loses"]].copy()
    print(f"\nSymbols where V34 loses Rule on composite (both have trades): {len(losers)} / {(merged['n_v34']>0).sum()}")

    # Top 30 worst by comp diff
    print("\n=== TOP 30 worst (V34 - Rule comp) ===")
    cols_show = ["symbol", "n_v34", "n_rule", "comp_v34", "comp_rule", "comp_diff",
                 "total_pnl_v34", "total_pnl_rule", "pnl_diff", "wr_v34", "wr_rule"]
    print(losers.head(30)[cols_show].to_string(index=False))

    # Top 30 worst by total PnL diff
    print("\n=== TOP 30 worst by total_pnl diff ===")
    losers_pnl = merged[(merged["n_v34"] > 0) & (merged["n_rule"] > 0)].sort_values("pnl_diff").head(30)
    print(losers_pnl[cols_show].to_string(index=False))

    # Quick aggregates
    both = merged[(merged["n_v34"] > 0) & (merged["n_rule"] > 0)]
    print(f"\n=== Aggregate (symbols with both v34 + rule trades, n={len(both)}) ===")
    print(f"  V34 wins: {(both['comp_diff']>0).sum()}  ({100*(both['comp_diff']>0).mean():.1f}%)")
    print(f"  Rule wins: {(both['comp_diff']<0).sum()}  ({100*(both['comp_diff']<0).mean():.1f}%)")
    print(f"  Avg comp diff: {both['comp_diff'].mean():+.1f}")
    print(f"  Total pnl V34: {both['total_pnl_v34'].sum():+.0f}%")
    print(f"  Total pnl Rule: {both['total_pnl_rule'].sum():+.0f}%")


if __name__ == "__main__":
    main()
