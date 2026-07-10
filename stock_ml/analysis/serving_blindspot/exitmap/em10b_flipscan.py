# -*- coding: utf-8 -*-
"""em10b: quet ranh gioi CHIEU NGUOC (flag khi trang thai co phieu con KHOE / lenh gia)
tren feature da tinh o em10 (stockstate_mkt.csv / stockstate_struct.csv).
Muc dich: falsification day du — neu ca 2 chieu deu khong tach duoc thi mat phan tach khong ton tai.
"""
import itertools

import numpy as np
import pandas as pd

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
pd.set_option("display.width", 250)

GRIDS = {
    "ma20_rel": [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02],
    "ma60_rel": [-0.08, -0.04, -0.02, 0.0, 0.02, 0.05, 0.08],
    "blw20_run": [1, 2, 3, 5],
    "snr21": [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2],
    "sell_run": [1, 2, 3, 5],
    "rs5": [-0.06, -0.04, -0.02, 0.0, 0.02],
    "idio5": [-0.06, -0.04, -0.02, 0.0, 0.02],
    "age": [10, 20, 30, 40, 60, 80],
    "gain": [0.0, 0.05, 0.1, 0.2, 0.3],
    "giveback_e": [0.05, 0.08, 0.1, 0.15, 0.2],
}


def scan(name, fn):
    g = pd.read_csv(EM + f"/stockstate_{fn}.csv")
    g = g[g.year_entry >= 2022]
    u_pha = -g.delta[g.delta < 0].sum()
    u_cuu = g.delta[g.delta > 0].sum()
    res = []
    for var, thrs in GRIDS.items():
        for thr, op in itertools.product(thrs, ["<", ">="]):
            fl = ((g[var] < thr) if op == "<" else (g[var] >= thr)) & g[var].notna()
            if fl.sum() == 0 or fl.sum() == len(g):
                continue
            cut = -g.delta[fl & (g.delta < 0)].sum()
            lost = g.delta[fl & (g.delta > 0)].sum()
            megas = g[fl & (g.delta > 0.2)]
            res.append(dict(rule=f"{var}{op}{thr}", n=int(fl.sum()), cut=round(cut, 3),
                            lost=round(lost, 3), net=round(cut - lost, 3),
                            pha_cov=round(cut / max(u_pha, 1e-9), 3),
                            cuu_loss=round(lost / max(u_cuu, 1e-9), 3),
                            n_mega=len(megas)))
    R = pd.DataFrame(res).sort_values("net", ascending=False)
    print(f"\n===== [{name}] >=2022 n={len(g)} u_pha={u_pha:.2f} u_cuu={u_cuu:.2f} — "
          f"TOP 15 CA HAI CHIEU =====")
    print(R.head(15).to_string(index=False))
    # cap doi tu top 10 don
    top = R.head(10)
    pres = []
    parsed = []
    for _, rr in top.iterrows():
        var = rr.rule.split("<")[0].split(">=")[0]
        op = "<" if "<" in rr.rule else ">="
        thr = float(rr.rule.replace(var + op, ""))
        parsed.append((var, op, thr))
    for (va, oa, ta), (vb, ob, tb) in itertools.combinations(parsed, 2):
        if va == vb:
            continue
        fa = ((g[va] < ta) if oa == "<" else (g[va] >= ta)) & g[va].notna()
        fb = ((g[vb] < tb) if ob == "<" else (g[vb] >= tb)) & g[vb].notna()
        fl = fa & fb
        if fl.sum() == 0:
            continue
        cut = -g.delta[fl & (g.delta < 0)].sum()
        lost = g.delta[fl & (g.delta > 0)].sum()
        megas = g[fl & (g.delta > 0.2)]
        pres.append(dict(rule=f"{va}{oa}{ta} & {vb}{ob}{tb}", n=int(fl.sum()),
                         cut=round(cut, 3), lost=round(lost, 3), net=round(cut - lost, 3),
                         pha_cov=round(cut / max(u_pha, 1e-9), 3),
                         cuu_loss=round(lost / max(u_cuu, 1e-9), 3), n_mega=len(megas)))
    P = pd.DataFrame(pres).sort_values("net", ascending=False)
    print(f"\n[{name}] TOP 10 KEP (ca hai chieu):")
    print(P.head(10).to_string(index=False))
    return R, P


scan("MKT_DROP", "mkt")
scan("STRUCT", "struct")
