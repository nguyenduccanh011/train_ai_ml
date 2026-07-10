import sqlite3
import numpy as np
import pandas as pd

AN = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
DB = r"C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
pd.set_option("display.width", 250)

P = pd.read_csv(AN + r"/tmp_tail_sim_per_trade.csv", parse_dates=["entry_date", "exit_date"])
con = sqlite3.connect(DB)

def path(sym, d0, d1):
    q = pd.read_sql("select date,open,high,low,close from ohlcv where symbol=? and date>=? and date<=? order by date",
                    con, params=(sym, d0, d1))
    return q

# --- verify MSR 2025-04-03 under stop_8 ---
r = P[(P.symbol == "MSR") & (P.entry_date == "2025-04-03")].iloc[0]
print("MSR trade:", r.entry_date.date(), "->", r.exit_date.date(), "base", round(r.pnl_pct, 4),
      "sim_stop_8", round(r.sim_stop_8, 4), "simdate", r.simdate_stop_8)
q = path("MSR", "2025-04-03", "2025-04-15")
q["ret_vs_entry_low"] = None
print(q.to_string(index=False))
# entry price from trades_metrics
T = pd.read_csv(AN + r"/trades_metrics.csv", parse_dates=["entry_date"])
tr = T[(T.symbol == "MSR") & (T.entry_date == "2025-04-03")].iloc[0]
print("entry_price:", tr.entry_price, "stop_8 level:", tr.entry_price * 0.92)
print("low/entry-1:", (q.low / tr.entry_price - 1).round(4).tolist())
print("close/entry-1:", (q.close / tr.entry_price - 1).round(4).tolist())

# --- verify TIG 2020-03-09 under stop_8 ---
r = P[(P.symbol == "TIG") & (P.entry_date == "2020-03-09")].iloc[0]
tr = T[(T.symbol == "TIG") & (T.entry_date == "2020-03-09")].iloc[0]
print("\nTIG trade base", round(r.pnl_pct, 4), "sim_stop_8", round(r.sim_stop_8, 4), "simdate", r.simdate_stop_8,
      "entry_price", tr.entry_price)
q = path("TIG", "2020-03-09", "2020-03-20")
print((q.assign(lowr=(q.low / tr.entry_price - 1).round(4), closer=(q.close / tr.entry_price - 1).round(4))).to_string(index=False))

# --- decomposition of delta by base-pnl bucket for each rule ---
base = P.pnl_pct.to_numpy()
buckets = [("big_win>=15", base >= 0.15), ("win5..15", (base >= 0.05) & (base < 0.15)),
           ("small-5..5", (base > -0.05) & (base < 0.05)), ("loss-15..-5", (base > -0.15) & (base <= -0.05)),
           ("bigloss<=-15", base <= -0.15)]
labs = [c[4:] for c in P.columns if c.startswith("sim_")]
print("\ndelta_u by base bucket (sum of sim-base over triggered trades):")
hdr = "rule".ljust(14) + "".join(b[0].rjust(14) for b in buckets)
print(hdr)
for lab in labs:
    sim = P[f"sim_{lab}"].to_numpy()
    delta = np.where(np.isnan(sim), 0.0, sim - base)
    row = lab.ljust(14)
    for name, m in buckets:
        row += f"{delta[m].sum():+.2f}u".rjust(14)
    print(row)

# stop damage by exit-year (gap risk concentration)
print("\nstop_8 delta by entry year:")
P["yr"] = P.entry_date.dt.year
d8 = np.where(np.isnan(P.sim_stop_8), 0, P.sim_stop_8 - base)
print(pd.Series(d8).groupby(P.yr).sum().round(2).to_dict())

# how many stop_8 triggered trades ended worse than engine (stop sold lower than eventual engine exit)
for lab in labs:
    sim = P[f"sim_{lab}"].to_numpy()
    trig = ~np.isnan(sim)
    worse = trig & (sim < base)
    better = trig & (sim > base)
    print(f"{lab}: triggered {trig.sum()}, made-worse {worse.sum()} ({(sim-base)[worse].sum():.2f}u), "
          f"made-better {better.sum()} (+{(sim-base)[better].sum():.2f}u)")
con.close()
