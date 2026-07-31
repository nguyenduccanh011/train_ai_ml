# -*- coding: utf-8 -*-
# Audit finding #3: overlay-rewritten exits priced at raw close (0 sell slippage)
# vs slippage embedded in base-engine exits. Empirical check on registered runs.
import duckdb
import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
RUNS = {
    "gtos_3379": "template/x2_struct_to_k10_cs5ma50_r7ec_gtos-69338138",
    "osdef_3376": "template/x2_struct_to_k10_cs5ma50_r7ec_osdef-69338138",
    "gtrail_3377": "template/x2_struct_to_k10_cs5ma50_r7ec_gtrail-69338138",
    "earlycut_3375": "template/x2_struct_to_k10_cs5ma50_r7earlycut-69338138",
}

con = psycopg2.connect(**PG)
frames = {}
for tag, rid in RUNS.items():
    df = pd.read_sql(
        "select symbol, entry_date, exit_date, entry_price, exit_price, exit_reason, pnl_pct "
        "from run_trades where run_id=%s", con, params=(rid,))
    print(f"{tag}: {len(df)} trades in run_trades")
    frames[tag] = df
con.close()

cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol, CAST(date AS DATE) AS date, close FROM ohlcv WHERE timeframe='1D'").fetchdf()
cx.close()
px["date"] = pd.to_datetime(px["date"])

for tag, df in frames.items():
    if len(df) == 0:
        continue
    df = df[df.exit_date.notna()].copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    m = df.merge(px.rename(columns={"date": "exit_date", "close": "close_exit"}),
                 on=["symbol", "exit_date"], how="left")
    m = m.merge(px.rename(columns={"date": "entry_date", "close": "close_entry"}),
                on=["symbol", "entry_date"], how="left")
    m["r_exit"] = m.exit_price / m.close_exit - 1.0
    m["r_entry"] = m.entry_price / m.close_entry - 1.0
    print(f"\n=== {tag} ===  (n={len(m)}, close-match miss={int(m.close_exit.isna().sum())})")
    g = m.dropna(subset=["r_exit"]).groupby("exit_reason")["r_exit"].agg(["count", "min", "max", "mean"])
    print("exit_price/duckdb_close - 1 by exit_reason:")
    print(g.to_string(float_format=lambda x: f"{x:+.6f}"))
    ge = m.dropna(subset=["r_entry"])["r_entry"]
    print(f"entry_price/duckdb_close - 1 : min {ge.min():+.6f} max {ge.max():+.6f} n={len(ge)}")

# ---- quantify NAV impact on gtrail & gtos: reprice overlay exits at close*(1-0.0015) ----
print("\n---- impact estimate (weight=1/K=0.10 per trade, sequential compounding approx) ----")
S0 = 0.001
FEE = 0.004
s_new = 0.001
for tag in ["gtos_3379", "gtrail_3377", "earlycut_3375", "osdef_3376"]:
    df = frames[tag]
    if len(df) == 0:
        continue
    df = df[df.exit_date.notna()].copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    m = df.merge(px.rename(columns={"date": "exit_date", "close": "close_exit"}),
                 on=["symbol", "exit_date"], how="left").dropna(subset=["close_exit"])
    aff = m[m.exit_reason.isin(["early_cut", "green_trail"])].copy()
    # NavSim2 effective net: x_raw=exit/(1-S0); net = x_raw*(1-s)/(e_raw*(1+s)) -1 -FEE ; e_raw=entry/(1+S0)
    def net_of(ep, xp):
        return (xp / (1 - S0) * (1 - s_new)) / (ep / (1 + S0) * (1 + s_new)) - 1 - FEE
    aff["net_reg"] = net_of(aff.entry_price, aff.exit_price)                       # as registered (raw close)
    aff["net_fix"] = net_of(aff.entry_price, aff.close_exit * (1 - 0.0015))        # engine-consistent slippage
    aff["dnet"] = aff.net_reg - aff.net_fix
    w = 0.10
    import numpy as np
    nav_ratio = float(np.prod(1 + w * aff.net_reg) / np.prod(1 + w * aff.net_fix))
    yrs = (m.exit_date.max() - pd.to_datetime(df.entry_date).min()).days / 365.25
    print(f"{tag}: affected={len(aff)} ({len(aff)/len(m)*100:.1f}% of {len(m)}), "
          f"mean dnet={aff.dnet.mean()*100:.4f}% per trade, "
          f"NAV subsidy multiplier ~x{nav_ratio:.4f} over {yrs:.1f}y "
          f"-> CAGR inflation ~{(nav_ratio**(1/yrs)-1)*100:.2f}% relative on (1+CAGR)")
