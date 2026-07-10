# -*- coding: utf-8 -*-
"""P1-E1 buoc 4 — bai kiem tra pha (falsification) cho cell song sot:
lambdarank + rebalance THANG (khop horizon label 20 bar) + hysteresis.

 (1) PHASE: quet 20 offset ngay bat dau rebalance -> phan bo sprd_tu_net >=2022
     (neu chi 1-2 phase duong -> may man lich; can dai da so duong)
 (2) PER-SEED: s42/s7/s99 rieng (khong chi seed-mean)
 (3) FILL-RATE thang: cohort ten moi vao top-K moi thang, limit 4.5%/40 bar champion
     + chat luong phan fill vs khong-limit (adverse selection check)
 (4) OVERLAP champion tren membership thang
Out: p1_robust.csv + in bang
"""
import os

import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\pureml"
TRADES = (r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
          r"\signalq\nicheloss\gb_x08_s42_trades.csv")
COST_RT = 0.007
LAG = 1
MIN_SYM = 40
REB = 20
PB_DEPTH, PB_WIN = 0.045, 40

pred = pd.read_parquet(os.path.join(OUT, "p1_preds.parquet"))
ds = pd.read_parquet(os.path.join(OUT, "p1_dataset.parquet"),
                     columns=["symbol", "date", "open", "low", "close"])

def add_rank_mean(df, cols, name):
    r = [df.groupby("date")[c].rank(pct=True) for c in cols]
    df[name] = pd.concat(r, axis=1).mean(axis=1)

add_rank_mean(pred, ["pred_rank_s42", "pred_rank_s7", "pred_rank_s99"], "pred_rank_mean")

cal = pd.DatetimeIndex(sorted(pred.date.unique()))
ds = ds[ds.date >= cal[0]]
piv_c = ds.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
piv_o = ds.pivot_table(index="date", columns="symbol", values="open", aggfunc="last").sort_index()
piv_l = ds.pivot_table(index="date", columns="symbol", values="low", aggfunc="last").sort_index()
grid_dates = piv_c.index
gpos = {d: i for i, d in enumerate(grid_dates)}
C, O, L = piv_c.to_numpy(), piv_o.to_numpy(), piv_l.to_numpy()
sym_grid = {s: i for i, s in enumerate(piv_c.columns)}

MODELS = ["pred_rank_mean", "pred_rank_s42", "pred_rank_s7", "pred_rank_s99"]
pred_piv = {m: pred.pivot_table(index="date", columns="symbol", values=m, aggfunc="last")
            for m in MODELS}

def period_ret(members, t, t_next):
    i0, i1 = gpos[t] + LAG, gpos[t_next] + LAG
    if i1 >= len(grid_dates):
        return np.nan
    r = (piv_c.iloc[i1][members] / piv_c.iloc[i0][members] - 1.0).dropna()
    return float(r.mean()) if len(r) else np.nan

def run(mcol, K, offset, collect_entries=False):
    k_out = int(K * 1.5)
    pp = pred_piv[mcol]
    reb_dates = list(cal[offset::REB])
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
        stay = {x for x in port if x in rk.index and rk[x] <= k_out}
        new_port = stay | top
        entered, exited = new_port - port, port - new_port
        n = max(len(new_port), 1)
        rows.append(dict(date=t, year=pd.Timestamp(t).year,
                         cost=(len(entered) + len(exited)) / 2.0 * COST_RT / n,
                         r_top=period_ret(sorted(new_port), t, t_next),
                         r_uni=period_ret(sorted(s.index), t, t_next),
                         entered=",".join(sorted(entered)) if collect_entries else "",
                         members=",".join(sorted(new_port))))
        port = new_port
    return pd.DataFrame(rows)

def sprd_net(pf, y_ge=2022):
    g = pf.dropna(subset=["r_top", "r_uni"])
    g = g[g.year >= y_ge]
    return float((g.r_top - g.cost - g.r_uni).mean() * 252 / REB)

# ---------------- (1)+(2) phase x seed ----------------
print("=== (1)+(2) phase sweep (20 offsets) x model, K=5/10/20, reb=20, hyst ===", flush=True)
rows = []
for mcol in MODELS:
    for K in [5, 10, 20]:
        vals = [sprd_net(run(mcol, K, off)) for off in range(20)]
        v = np.array(vals)
        rows.append(dict(model=mcol, K=K,
                         sprd_min=round(v.min(), 4), sprd_p25=round(np.percentile(v, 25), 4),
                         sprd_med=round(np.median(v), 4), sprd_p75=round(np.percentile(v, 75), 4),
                         sprd_max=round(v.max(), 4), pct_pos=round(float((v > 0).mean()), 2)))
        print(rows[-1], flush=True)
rob = pd.DataFrame(rows)
rob.to_csv(os.path.join(OUT, "p1_robust.csv"), index=False)

# ---------------- (3) fill-rate thang ----------------
print("\n=== (3) fill-rate pullback 4.5%/40 tren cohort thang (rank_mean) ===", flush=True)
fill_out = []
for K in [5, 10, 20]:
    pf = run("pred_rank_mean", K, 0, collect_entries=True)
    ev = []
    for r in pf.itertuples():
        if not r.entered:
            continue
        it = gpos[r.date]
        for s in r.entered.split(","):
            js = sym_grid.get(s)
            if js is None or np.isnan(C[it, js]):
                continue
            limit = C[it, js] * (1 - PB_DEPTH)
            fill_i, fill_px = None, None
            for k in range(1, PB_WIN + 1):
                i2 = it + k
                if i2 >= len(grid_dates):
                    break
                if not np.isnan(L[i2, js]) and L[i2, js] <= limit:
                    fill_i = i2
                    fill_px = min(O[i2, js], limit) if not np.isnan(O[i2, js]) else limit
                    break
            def fwd(i0, px, h):
                i1 = i0 + h
                if px is None or i1 >= len(grid_dates) or np.isnan(C[i1, js]):
                    return np.nan
                return C[i1, js] / px - 1.0 - COST_RT
            ia = it + LAG
            apx = C[ia, js] if ia < len(grid_dates) else np.nan
            ev.append(dict(year=pd.Timestamp(r.date).year, filled=fill_i is not None,
                           dtf=(fill_i - it) if fill_i else np.nan,
                           f20=fwd(fill_i, fill_px, 20) if fill_i else np.nan,
                           f40=fwd(fill_i, fill_px, 40) if fill_i else np.nan,
                           a20=fwd(ia, apx, 20), a40=fwd(ia, apx, 40)))
    e = pd.DataFrame(ev)
    g = e[e.year >= 2022]
    f = g[g.filled]
    row = dict(K=K, n_orders=len(g), fill_rate=round(float(g.filled.mean()), 3),
               med_days=float(f.dtf.median()),
               fill_f20=round(float(f.f20.mean()), 4), fill_f40=round(float(f.f40.mean()), 4),
               all_a20=round(float(g.a20.mean()), 4), all_a40=round(float(g.a40.mean()), 4),
               unfil_a20=round(float(g.loc[~g.filled, 'a20'].mean()), 4),
               unfil_a40=round(float(g.loc[~g.filled, 'a40'].mean()), 4))
    fill_out.append(row)
    print(row, flush=True)
pd.DataFrame(fill_out).to_csv(os.path.join(OUT, "p1_fill_monthly.csv"), index=False)

# ---------------- (4) overlap champion (thang, K=5/10) ----------------
print("\n=== (4) overlap champion gb_x08 s42 (membership thang) ===", flush=True)
tr = pd.read_csv(TRADES, parse_dates=["entry_date", "exit_date"])
tr = tr[tr.entry_date >= "2021-01-01"]
open_iv = list(zip(tr.symbol, tr.entry_date, tr.exit_date))
for K in [5, 10]:
    pf = run("pred_rank_mean", K, 0)
    memb = {r.date: set(r.members.split(",")) for r in pf.itertuples()}
    reb_arr = pd.DatetimeIndex(sorted(memb.keys()))
    hits = tot = 0
    for r in tr.itertuples():
        idx = reb_arr.searchsorted(r.entry_date, side="right") - 1
        if idx < 0:
            continue
        tot += 1
        hits += r.symbol in memb[reb_arr[idx]]
    slot_hit = slot_tot = 0
    for t, mm in memb.items():
        td = pd.Timestamp(t)
        holding = {s for s, e0, e1 in open_iv if e0 <= td <= e1}
        slot_hit += len(mm & holding)
        slot_tot += len(mm)
    print(f"K={K}: champion-entry in top-K = {hits}/{tot} = {hits/tot:.2f} | "
          f"book slot overlap = {slot_hit}/{slot_tot} = {slot_hit/slot_tot:.2f}", flush=True)

print("P1_04_DONE")
