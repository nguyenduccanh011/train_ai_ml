# -*- coding: utf-8 -*-
"""P1-E1 buoc 3 — sensitivity grid cho verdict kill/go:
K x REB_EVERY x scheme, model pred_rank_mean (IC tot nhat) + pred_reg_mean doi chieu.
Do: sprd_tu_net (top vs universe EW, NET cost — tieu chi kill), sprd_tu_gross,
turnover, top_net. Pooled >=2022. Muc dich: chac chan kill khong phai do 1 lua chon
tham so thuc thi (K=20/61 qua loang, tuan qua day cost).
Out: p1_sensitivity.csv
"""
import os

import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\pureml"
COST_RT = 0.007
LAG = 1
MIN_SYM = 40

pred = pd.read_parquet(os.path.join(OUT, "p1_preds.parquet"))
ds = pd.read_parquet(os.path.join(OUT, "p1_dataset.parquet"),
                     columns=["symbol", "date", "close"])

def add_rank_mean(df, cols, name):
    r = [df.groupby("date")[c].rank(pct=True) for c in cols]
    df[name] = pd.concat(r, axis=1).mean(axis=1)

add_rank_mean(pred, ["pred_reg_s42", "pred_reg_s7", "pred_reg_s99"], "pred_reg_mean")
add_rank_mean(pred, ["pred_rank_s42", "pred_rank_s7", "pred_rank_s99"], "pred_rank_mean")

cal = pd.DatetimeIndex(sorted(pred.date.unique()))
ds = ds[ds.date >= cal[0]]
piv_c = ds.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
grid_dates = piv_c.index
gpos = {d: i for i, d in enumerate(grid_dates)}
pred_piv = {m: pred.pivot_table(index="date", columns="symbol", values=m, aggfunc="last")
            for m in ["pred_rank_mean", "pred_reg_mean"]}

def period_ret(members, t, t_next):
    i0, i1 = gpos[t] + LAG, gpos[t_next] + LAG
    if i1 >= len(grid_dates):
        return np.nan
    r = (piv_c.iloc[i1][members] / piv_c.iloc[i0][members] - 1.0).dropna()
    return float(r.mean()) if len(r) else np.nan

def run(mcol, K, reb_every, scheme):
    k_out = int(K * 1.5)
    pp = pred_piv[mcol]
    reb_dates = list(cal[::reb_every])
    rows, port = [], set()
    for j, t in enumerate(reb_dates[:-1]):
        t_next = reb_dates[j + 1]
        if t not in pp.index:
            continue
        s = pp.loc[t].dropna()
        if len(s) < MIN_SYM:
            continue
        rk = s.rank(ascending=False)
        top = set(rk[rk <= K].index)
        if scheme == "plain":
            new_port = top
        else:
            stay = {x for x in port if x in rk.index and rk[x] <= k_out}
            new_port = stay | top
        entered, exited = new_port - port, port - new_port
        n = max(len(new_port), 1)
        cost = (len(entered) + len(exited)) / 2.0 * COST_RT / n
        turnover = (len(entered) + len(exited)) / 2.0 / n
        rows.append(dict(date=t, year=pd.Timestamp(t).year, turnover=turnover, cost=cost,
                         r_top=period_ret(sorted(new_port), t, t_next),
                         r_uni=period_ret(sorted(s.index), t, t_next),
                         n_port=len(new_port)))
        port = new_port
    pf = pd.DataFrame(rows).dropna(subset=["r_top", "r_uni"])
    ppy = 252.0 / reb_every
    res = []
    for label, g in [("ge2022", pf[pf.year >= 2022])] + \
                    [(str(y), pf[pf.year == y]) for y in sorted(pf.year.unique())]:
        if not len(g):
            continue
        res.append(dict(model=mcol, scheme=scheme, K=K, reb=reb_every, period=label,
                        n_reb=len(g), n_port=round(float(g.n_port.mean()), 1),
                        turnover=round(float(g.turnover.mean()), 3),
                        top_net_ann=round(float((g.r_top - g.cost).mean() * ppy), 4),
                        uni_ann=round(float(g.r_uni.mean() * ppy), 4),
                        sprd_tu_gross=round(float((g.r_top - g.r_uni).mean() * ppy), 4),
                        sprd_tu_net=round(float((g.r_top - g.cost - g.r_uni).mean() * ppy), 4)))
    return res

allrows = []
for mcol in ["pred_rank_mean", "pred_reg_mean"]:
    for K in [5, 10, 15, 20]:
        for reb in [5, 10, 20]:
            for scheme in ["plain", "hyst"]:
                allrows += run(mcol, K, reb, scheme)

sens = pd.DataFrame(allrows)
sens.to_csv(os.path.join(OUT, "p1_sensitivity.csv"), index=False)
pd.set_option("display.width", 250)
hd = sens[(sens.period == "ge2022")].sort_values(["model", "sprd_tu_net"], ascending=[True, False])
print(hd.to_string(index=False))
best = hd.iloc[hd.groupby("model", sort=False).cumcount().eq(0).to_numpy().nonzero()[0]]
print("\nP1_03_DONE")
