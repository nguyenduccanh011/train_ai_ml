"""
Analysis-only script: dump per-trade CSV for V19.1, V19.3, V22-Final, V23, V16, Rule.
Does NOT modify model code. Used to investigate entry/exit quality per symbol/per trade.
"""
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_v19_1_compare import run_test as run_test_base, run_rule_test, backtest_v19_1
from run_v19_3_compare import backtest_v19_3
from run_v22_final import backtest_v22
from run_v23_optimal import backtest_v23
from run_v16_compare import backtest_v16

SYMBOLS = "ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def collect(label, fn, **extra):
    def bt(y_pred, returns, df_test, feature_cols, **kwargs):
        return fn(y_pred, returns, df_test, feature_cols, **{**kwargs, **extra})
    trades = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=bt)
    df = pd.DataFrame(trades)
    df["model"] = label
    df.to_csv(os.path.join(OUT, f"trades_{label}.csv"), index=False)
    return df

print("Collecting V19.1..."); d191 = collect("v19_1", backtest_v19_1)
print("Collecting V19.3..."); d193 = collect("v19_3", backtest_v19_3)
print("Collecting V22-Final..."); d22 = collect("v22", backtest_v22)
print("Collecting V23 (best=pp_s=0.12)..."); d23 = collect("v23", backtest_v23, peak_protect_strong_threshold=0.12)

# V16 has different signature (no mod_h/i/j)
def bt_v16(y_pred, returns, df_test, feature_cols, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k not in ("mod_h", "mod_i", "mod_j")}
    return backtest_v16(y_pred, returns, df_test, feature_cols, **kwargs)
def run_v16():
    # Use run_test_base but inject a fn that strips mod_h/i/j
    return run_test_base(SYMBOLS, True, True, False, False, True, True, True, backtest_fn=bt_v16)
print("Collecting V16...")
try:
    trades = run_v16()
    d16 = pd.DataFrame(trades); d16["model"] = "v16"
    d16.to_csv(os.path.join(OUT, "trades_v16.csv"), index=False)
except Exception as e:
    print(f"V16 collect failed: {e}")
    d16 = pd.DataFrame()

print("Collecting Rule...")
rule = run_rule_test(SYMBOLS)
dr = pd.DataFrame(rule); dr["model"] = "rule"
dr.to_csv(os.path.join(OUT, "trades_rule.csv"), index=False)

# =============== ANALYSIS ===============
all_dfs = {"v19_1": d191, "v19_3": d193, "v22": d22, "v23": d23, "rule": dr}
if len(d16): all_dfs["v16"] = d16

print("\n" + "=" * 100)
print("TOP 10 BIGGEST LOSSES PER MODEL")
print("=" * 100)
for lbl, df in all_dfs.items():
    if "pnl_pct" not in df.columns or len(df) == 0:
        continue
    top_loss = df.nsmallest(10, "pnl_pct")[["entry_symbol", "entry_date", "exit_date", "pnl_pct", "exit_reason", "entry_trend" if "entry_trend" in df.columns else "pnl_pct", "holding_days"]]
    print(f"\n-- {lbl} --")
    print(top_loss.to_string(index=False))

print("\n" + "=" * 100)
print("TOP 10 BIGGEST WINS PER MODEL")
print("=" * 100)
for lbl, df in all_dfs.items():
    if "pnl_pct" not in df.columns or len(df) == 0:
        continue
    top_win = df.nlargest(10, "pnl_pct")[["entry_symbol", "entry_date", "exit_date", "pnl_pct", "exit_reason", "holding_days"]]
    print(f"\n-- {lbl} --")
    print(top_win.to_string(index=False))

print("\n" + "=" * 100)
print("SAME-SYMBOL COMPARISON: overlap trades")
print("=" * 100)
# For each symbol, compute correlation of pnl between models and find divergent trades
focus = ["HPG", "VND", "FPT", "AAV", "REE", "DGC", "MBB"]
for sym in focus:
    print(f"\n--- {sym} ---")
    for lbl, df in all_dfs.items():
        if len(df) == 0 or "entry_symbol" not in df.columns: continue
        sub = df[df["entry_symbol"] == sym]
        if len(sub) == 0: continue
        wr = (sub["pnl_pct"] > 0).mean() * 100
        print(f"  {lbl:<7} n={len(sub):>3}  total={sub['pnl_pct'].sum():>+7.1f}%  "
              f"avg={sub['pnl_pct'].mean():>+6.2f}%  WR={wr:>4.1f}%  "
              f"maxL={sub['pnl_pct'].min():>+6.1f}%  maxW={sub['pnl_pct'].max():>+6.1f}%  "
              f"avgHold={sub['holding_days'].mean():>5.1f}d")

# Exit reason distribution
print("\n" + "=" * 100)
print("EXIT REASON DISTRIBUTION")
print("=" * 100)
for lbl, df in all_dfs.items():
    if "exit_reason" not in df.columns or len(df) == 0: continue
    print(f"\n-- {lbl} --")
    g = df.groupby("exit_reason").agg(n=("pnl_pct", "size"), total=("pnl_pct", "sum"),
                                       avg=("pnl_pct", "mean"), wr=("pnl_pct", lambda x: (x > 0).mean() * 100))
    print(g.round(2).to_string())

# Year analysis focus on 2022 (worst year V23)
print("\n" + "=" * 100)
print("2022 DEEP DIVE (V23 worst year)")
print("=" * 100)
for lbl, df in all_dfs.items():
    if len(df) == 0: continue
    df2 = df.copy()
    df2["year"] = df2["entry_date"].astype(str).str[:4]
    s = df2[df2["year"] == "2022"]
    if len(s) == 0: continue
    print(f"\n-- {lbl} 2022 --  n={len(s)}  total={s['pnl_pct'].sum():+.1f}%  avg={s['pnl_pct'].mean():+.2f}%  WR={(s['pnl_pct']>0).mean()*100:.1f}%")
    print(s.groupby(["entry_symbol"]).agg(n=("pnl_pct", "size"), tot=("pnl_pct", "sum")).round(1).to_string())

print("\nDONE")
