"""Compare V35b vs V34 vs Rule per symbol on focus 21."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.evaluation.scoring import calc_metrics, composite_score


def load(name):
    df = pd.read_csv(ROOT / "results" / f"trades_{name}.csv")
    if "entry_symbol" in df.columns and "symbol" in df.columns:
        df = df.drop(columns=["entry_symbol"])
    return df


def per_sym(df, label):
    rows = []
    for sym, g in df.groupby("symbol"):
        trades = g[["pnl_pct", "entry_date", "symbol", "holding_days"]].to_dict("records")
        m = calc_metrics(trades); cs = composite_score(m, trades)
        rows.append({"symbol": sym, f"n_{label}": m["trades"], f"pnl_{label}": m["total_pnl"], f"comp_{label}": cs})
    return pd.DataFrame(rows)


v34 = load("v34"); v35b = load("v35b"); v35c = load("v35c"); rule = load("rule")
m = per_sym(v34, "v34").merge(per_sym(v35b, "v35b"), on="symbol", how="outer")\
    .merge(per_sym(v35c, "v35c"), on="symbol", how="outer").merge(per_sym(rule, "rule"), on="symbol", how="outer").fillna(0)

m["v35b-v34"] = m["comp_v35b"] - m["comp_v34"]
m["v35b-rule"] = m["comp_v35b"] - m["comp_rule"]

print("\n=== Per-symbol composite (v34 / v35b / v35c / rule) ===")
m_sorted = m.sort_values("v35b-v34", ascending=False)
print(m_sorted[["symbol", "comp_v34", "comp_v35b", "comp_v35c", "comp_rule", "v35b-v34", "v35b-rule"]].to_string(index=False))

print(f"\n=== Aggregate ===")
print(f"Symbols where V35b > V34: {(m['v35b-v34'] > 0).sum()} / {len(m)}")
print(f"Symbols where V35b > Rule: {(m['v35b-rule'] > 0).sum()} / {len(m)}")
print(f"Symbols V34 thua Rule -> V35b sua duoc: {((m['comp_v34']<m['comp_rule']) & (m['comp_v35b']>=m['comp_rule'])).sum()}")
print(f"Avg comp diff V35b - V34: {m['v35b-v34'].mean():+.1f}")
print(f"Avg comp diff V35b - Rule: {m['v35b-rule'].mean():+.1f}")
