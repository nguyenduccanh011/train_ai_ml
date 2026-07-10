# -*- coding: utf-8 -*-
"""pr4_40: T4 DD/tail —
(a) bootstrap subsample 80% trades x200 reps -> phan phoi MaxDD/final (mh16 vs gb, full adv);
(b) hanh vi crash 2025-04 (mean series 20 perm): DD sau nhat trong 2025-03..06;
(c) knife check: tail pnl cua cohort exit=max_hold (khong phai stop);
(d) concentration symbol: top-5 symbol share pnl."""
import random
import statistics
import sys

import pandas as pd

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/pr3")
sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from pr3_lib import NavSim2S  # noqa: E402
from nh_nav2 import NavSim2  # noqa: E402

R3 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r3line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
MH16 = f"{R3}/r3_mh16_s42_trades.csv"

print("=== (a) bootstrap subsample 80% x200 (full adv, K25) ===")
for name, csv in (("mh16", MH16), ("gb", GB)):
    sim = NavSim2(csv, "2020-01-01")
    all_trades = list(sim.trades)
    finals, dds = [], []
    for rep in range(200):
        rng = random.Random(rep)
        sim.trades = rng.sample(all_trades, int(len(all_trades) * 0.8))
        m = sim.run(K=25, advance_fee=0.0008, order_seed=rep)
        finals.append(m["final"])
        dds.append(m["maxdd"] * 100)
    sim.trades = all_trades
    dds_s = sorted(dds)  # am nhat dau tien
    fin_s = sorted(finals)
    print(f"  {name}: NAV p5={fin_s[10]:.2f} med={fin_s[100]:.2f} p95={fin_s[189]:.2f} | "
          f"DD med={statistics.median(dds):.1f}% p90={dds_s[20]:.1f}% p95={dds_s[10]:.1f}% "
          f"worst={dds_s[0]:.1f}%", flush=True)

print("\n=== (b) crash 2025-04 (mean 20-perm series) ===")
for name, csv in (("mh16", MH16), ("gb", GB)):
    sim = NavSim2S(csv, "2020-01-01")
    series = []
    for seed in range(20):
        ns = sim.run_series(K=25, advance_fee=0.0008, order_seed=seed)
        series.append(ns.set_index("date")["nav"])
    mean_nav = pd.concat(series, axis=1).mean(axis=1)
    w = mean_nav["2025-02-01":"2025-07-31"]
    dd = (w / w.cummax() - 1) * 100
    trough = dd.idxmin()
    # recovery: ngay dau tien sau trough NAV vuot lai dinh truoc do
    peak_val = w[:trough].max()
    rec = w[trough:][w[trough:] >= peak_val]
    rec_d = str(rec.index[0].date()) if len(rec) else ">2025-07-31"
    print(f"  {name}: DD sau nhat {dd.min():.1f}% tai {trough.date()}, phuc hoi {rec_d}")
    yoy = mean_nav["2025-12-31":"2026-12-31"]
    print(f"       NAV 2025-01-02 {mean_nav['2025-01-02':].iloc[0]:.2f} -> cuoi {mean_nav.iloc[-1]:.2f}")

print("\n=== (c) knife check cohort max_hold (mh16) ===")
t = pd.read_csv(MH16)
p = t.pnl_pct if t.pnl_pct.abs().median() < 1 else t.pnl_pct / 100
t = t.assign(p=p)
for rs, g in t.groupby("exit_reason"):
    print(f"  {rs}: n={len(g)} mean={g.p.mean()*100:+.2f}% med={g.p.median()*100:+.2f}% "
          f"min={g.p.min()*100:.1f}% p5={g.p.quantile(0.05)*100:.1f}% "
          f"sh<-10%={100*(g.p < -0.10).mean():.1f}% sh<-15%={100*(g.p < -0.15).mean():.1f}% "
          f"sum={g.p.sum()*100:+.0f}u")
mh = t[t.exit_reason == "max_hold"]
worst = mh.nsmallest(8, "p")[["symbol", "entry_date", "exit_date", "holding_days", "p"]]
print("  8 lenh max_hold te nhat:")
for r in worst.itertuples():
    print(f"    {r.symbol} {r.entry_date}->{r.exit_date} h{r.holding_days} {r.p*100:+.1f}%")

print("\n=== (d) concentration symbol (share tong pnl duong) ===")
for name, csv in (("mh16", MH16), ("gb", GB)):
    t = pd.read_csv(csv)
    p = t.pnl_pct if t.pnl_pct.abs().median() < 1 else t.pnl_pct / 100
    bysym = p.groupby(t.symbol).sum().sort_values(ascending=False)
    tot = p.sum()
    print(f"  {name}: n_sym={t.symbol.nunique()} top5={bysym.head(5).sum()/tot*100:.0f}% tong pnl "
          f"({', '.join(f'{s} {v*100:.0f}u' for s, v in bysym.head(5).items())})")
print("PR4_40_DONE")
