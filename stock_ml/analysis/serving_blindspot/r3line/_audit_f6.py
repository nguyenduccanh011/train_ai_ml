# -*- coding: utf-8 -*-
"""Audit finding #6: in-sample selection premium. (1) count overlay-modified trades in
registered runs; (2) split-window stability of swept GT constant on real prices."""
import psycopg2, duckdb, pandas as pd, numpy as np
from collections import defaultdict

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)

runs = {
 "gtos":  "template/x2_struct_to_k10_cs5ma50_r7ec_gtos-69338138",
 "gtrail":"template/x2_struct_to_k10_cs5ma50_r7ec_gtrail-69338138",
 "ec":    "template/x2_struct_to_k10_cs5ma50_r7earlycut-69338138",
}
print("=== (1) overlay mechanism occurrence in registered run_trades ===")
for lab, rid in runs.items():
    df = pd.read_sql("select exit_reason, count(*) n from run_trades where run_id=%s group by 1 order by 2 desc", con, params=(rid,))
    tot = df.n.sum()
    print(lab, "total", tot, dict(zip(df.exit_reason, df.n)))

# (2) GT split-window stability using r7earlycut registered trades + duckdb closes
tr = pd.read_sql("""select symbol, entry_date, exit_date, entry_price, exit_price, exit_reason
                    from run_trades where run_id=%s and exit_date is not null""", con, params=(runs["ec"],))
d = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
syms = tuple(tr.symbol.unique().tolist())
px = d.execute(f"select symbol, date, close from ohlcv where timeframe='1D' and symbol in {syms} order by symbol, date").df()
CLO, DIDX, INV = {}, {}, {}
for s, g in px.groupby("symbol"):
    g = g.reset_index(drop=True)
    CLO[s] = g["close"].values
    DIDX[s] = {str(dd)[:10]: i for i, dd in enumerate(g["date"])}
    INV[s] = {i: str(dd)[:10] for i, dd in enumerate(g["date"])}

GTS = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]
res = defaultdict(lambda: defaultdict(list))  # period -> gt -> list of delta log-returns
nmod = defaultdict(lambda: defaultdict(int))
green = tr[tr.exit_reason != "early_cut"]
n_green_elig = 0
for r in green.itertuples():
    di = DIDX.get(r.symbol, {}); ei = di.get(str(r.entry_date)[:10]); xi = di.get(str(r.exit_date)[:10])
    if ei is None or xi is None or xi <= ei + 2:
        continue
    c = CLO[r.symbol]
    if c[ei + 2] / r.entry_price - 1.0 < 0.0:
        continue  # red@s2 would have been early-cut; keep green group only
    n_green_elig += 1
    period = "2020-2022" if str(r.entry_date)[:4] in ("2020", "2021", "2022") else "2023-2026"
    r_orig = np.log(r.exit_price / r.entry_price)
    for gt in GTS:
        peak = c[ei]; ek, ep = xi, r.exit_price; hit = False
        for b in range(ei + 2, xi + 1):
            peak = max(peak, c[b])
            if c[b] <= peak * (1 - gt):
                if b + 1 <= xi: ek, ep = b + 1, float(c[b + 1])
                else:           ek, ep = b, float(c[b])
                hit = True; break
        res[period][gt].append(np.log(ep / r.entry_price) - r_orig)
        if hit: nmod[period][gt] += 1

print("\n=== (2) GT split-window stability (green@s2 trades of registered r7earlycut run, n=%d) ===" % n_green_elig)
print("mean delta log-return vs original exit (bps) | trades modified")
for period in sorted(res):
    n = len(res[period][GTS[0]])
    line = f"{period} (n={n}): "
    for gt in GTS:
        line += f" GT{int(gt*100)}%={10000*np.mean(res[period][gt]):+6.1f}({nmod[period][gt]})"
    print(line)
for period in sorted(res):
    best = max(GTS, key=lambda g: np.mean(res[period][g]))
    print(f"  argmax mean-delta {period}: GT={best}")
pooled = {g: np.mean(res["2020-2022"][g] + res["2023-2026"][g]) for g in GTS}
print("  pooled argmax:", max(pooled, key=pooled.get), {g: round(10000*v,1) for g, v in pooled.items()})
