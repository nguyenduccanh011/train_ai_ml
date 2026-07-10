# -*- coding: utf-8 -*-
"""pr5_60: CONG TO t2936 — T6 so dyn_mh25 (t2929, recent-tilt):
(a) f22/f23/f25 + full, ca 2 che do, ab_noT vs dyn25 head-to-head;
(b) overlap trades exact + fuzzy ±5d;
(c) blend thu tren giay: 50/50 pnl-stream co dang chay run hybrid khong."""
import math
import sys

import pandas as pd

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

R3 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r3line"
AB = f"{R3}/ab_noT_s42_trades.csv"
DYN = f"{R3}/xg_dyn_mh25_s42_trades.csv"


def delta(sa, sg):
    d = (sa["mean"] / sg["mean"] - 1) * 100
    sd = math.sqrt((sa["sd"] / sg["mean"]) ** 2
                   + (sg["sd"] * sa["mean"] / sg["mean"] ** 2) ** 2) * 100
    return f"{d:+.1f}%±{sd:.1f} ({d/sd:+.1f}sd)"


print("=== (a) ab_noT vs dyn_mh25 head-to-head ===")
for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22"),
                ("2023-01-01", "f23"), ("2025-01-01", "f25")):
    sa_ = NavSim2(AB, lo)
    sd_ = NavSim2(DYN, lo)
    for mode, fee in (("adv", 0.0008), ("noadv", None)):
        a = shuffle_stats(sa_, K=25, advance_fee=fee, n=20)
        d = shuffle_stats(sd_, K=25, advance_fee=fee, n=20)
        print(f"  {tag} {mode}: ab x{a['mean']:.2f}±{a['sd']:.2f} (DDw {a['dd_worst']*100:.1f}) "
              f"dyn25 x{d['mean']:.2f}±{d['sd']:.2f} (DDw {d['dd_worst']*100:.1f}) "
              f"-> ab vs dyn {delta(a, d)}", flush=True)

print("\n=== (b) OVERLAP trades ===")
ta = pd.read_csv(AB)
td = pd.read_csv(DYN)
ka = set(zip(ta.symbol, ta.entry_date.astype(str).str[:10]))
kd = set(zip(td.symbol, td.entry_date.astype(str).str[:10]))
exact = len(ka & kd)
print(f"  ab n={len(ka)} dyn n={len(kd)} exact={exact} "
      f"({exact/len(ka)*100:.1f}% cua ab, {exact/len(kd)*100:.1f}% cua dyn)")
da = {(s, d) for s, d in ka}
dd_ = pd.to_datetime(td.entry_date)
td2 = td.assign(d=dd_)
ta2 = ta.assign(d=pd.to_datetime(ta.entry_date))
fz = 0
dyn_by_sym = {s: sorted(g.d.tolist()) for s, g in td2.groupby("symbol")}
for r in ta2.itertuples():
    ds = dyn_by_sym.get(r.symbol, [])
    if any(abs((x - r.d).days) <= 5 for x in ds):
        fz += 1
print(f"  fuzzy±5d: {fz} ({fz/len(ta2)*100:.1f}% cua ab)")

print("\n=== (c) blend 50/50 pnl-stream (moi ben K=50 slot, gop trades) ===")
# gop 2 CSV, cham NAV K=50 de xem co ke som diversify — proxy re cho hybrid
comb = pd.concat([ta[["symbol", "entry_date", "exit_date", "entry_price", "exit_price",
                      "holding_days", "pnl_pct", "exit_reason"]],
                  td[["symbol", "entry_date", "exit_date", "entry_price", "exit_price",
                      "holding_days", "pnl_pct", "exit_reason"]]])
tmp = f"{R3}/pr5_tmp_blend.csv"
comb.to_csv(tmp, index=False)
for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22"), ("2025-01-01", "f25")):
    sb = NavSim2(tmp, lo)
    for mode, fee in (("adv", 0.0008), ("noadv", None)):
        b = shuffle_stats(sb, K=50, advance_fee=fee, n=20)
        print(f"  blend K50 {tag} {mode}: x{b['mean']:.2f}±{b['sd']:.2f} "
              f"DDw {b['dd_worst']*100:.1f}%", flush=True)
print("PR5_60_DONE")
