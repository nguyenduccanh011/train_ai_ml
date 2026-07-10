# -*- coding: utf-8 -*-
"""pr5_40: CONG TO t2936 (ab_noT) — T2 co che bo-T co tao rui ro moi:
(a) cohort overext_trail 203 lenh: SOLD-THEN-RALLIED — sau khi ban-vao-suc-manh,
    gia chay tiep bao nhieu (max close +5/+10 bar; counterfactual giu toi cap bar 17);
(b) tail winner: ab co mat runner khong (p95/p99/max, share top-10, n>=+30%/+50%)
    so gbmh16 (giu T) / gb (khong cap) / t2907;
(c) bootstrap subsample 80% x200: phan phoi NAV/MaxDD ab vs gb vs t2907;
(d) knife check theo exit_reason cua ab."""
import random
import statistics
import sys

import pandas as pd

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from nh_nav2 import NavSim2  # noqa: E402

R3 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r3line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
AB = f"{R3}/ab_noT_s42_trades.csv"
T2907 = f"{R3}/r3_mh16_s42_trades.csv"
GBMH16 = f"{R3}/r3_gb_mh16_s42_trades.csv"

print("=== (a) SOLD-THEN-RALLIED cohort overext_trail + trailing_stop cua ab ===")
sim = NavSim2(AB, "2020-01-01")  # da load gia raw close theo symbol
t = pd.read_csv(AB)
p = t.pnl_pct if t.pnl_pct.abs().median() < 1 else t.pnl_pct / 100
t = t.assign(p=p)
for reason in ("overext_trail", "trailing_stop"):
    g = t[t.exit_reason == reason]
    rows = []
    for r in g.itertuples():
        s = r.symbol
        idx = sim.sym_idx.get(s, {})
        ed, xd = str(r.entry_date)[:10], str(r.exit_date)[:10]
        if ed not in idx or xd not in idx:
            continue
        i0, i1 = idx[ed], idx[xd]
        closes = sim.sym_close[s]
        x = closes[i1]
        # tiep dien sau exit
        nxt5 = closes[i1 + 1: i1 + 6]
        nxt10 = closes[i1 + 1: i1 + 11]
        # counterfactual: giu toi cap (bar 17 tinh tu entry, nhu gb+mh16 lam)
        j_cap = min(i0 + 17, len(closes) - 1)
        rows.append(dict(
            p=r.p, hold=r.holding_days,
            m5=(max(nxt5) / x - 1) if nxt5 else 0.0,
            m10=(max(nxt10) / x - 1) if nxt10 else 0.0,
            c10=(closes[min(i1 + 10, len(closes) - 1)] / x - 1),
            cap=(closes[j_cap] / x - 1) if j_cap > i1 else 0.0))
    d = pd.DataFrame(rows)
    print(f"\n  {reason}: n={len(d)} pnl mean={d.p.mean()*100:+.1f}% hold_mean={d.hold.mean():.1f}")
    print(f"    max-rally +5 bar sau exit : mean {d.m5.mean()*100:+.1f}% med {d.m5.median()*100:+.1f}% "
          f"sh>=+5% {100*(d.m5 >= 0.05).mean():.0f}% sh>=+10% {100*(d.m5 >= 0.10).mean():.0f}%")
    print(f"    max-rally +10 bar sau exit: mean {d.m10.mean()*100:+.1f}% med {d.m10.median()*100:+.1f}% "
          f"sh>=+10% {100*(d.m10 >= 0.10).mean():.0f}% sh>=+20% {100*(d.m10 >= 0.20).mean():.0f}%")
    print(f"    close +10 bar (khong max) : mean {d.c10.mean()*100:+.1f}% med {d.c10.median()*100:+.1f}% "
          f"sh>0 {100*(d.c10 > 0).mean():.0f}%")
    print(f"    counterfactual GIU TOI CAP bar17: gia them mean {d.cap.mean()*100:+.1f}% "
          f"med {d.cap.median()*100:+.1f}% | sh am {100*(d.cap < 0).mean():.0f}% "
          f"| tong bo lo {d.cap.sum()*100:+.0f}u tren {len(d)} lenh "
          f"(doi lai giai phong {(17 - d.hold).sum():.0f} symbol-ngay slot)", flush=True)

print("\n=== (b) TAIL WINNER — ab co mat runner? ===")
for name, csv in (("ab_noT", AB), ("gbmh16", GBMH16), ("t2907", T2907), ("gb", GB)):
    tt = pd.read_csv(csv)
    pp = tt.pnl_pct if tt.pnl_pct.abs().median() < 1 else tt.pnl_pct / 100
    tot = pp.sum()
    top10 = pp.nlargest(10).sum()
    print(f"  {name}: p90={pp.quantile(0.90)*100:+.1f}% p95={pp.quantile(0.95)*100:+.1f}% "
          f"p99={pp.quantile(0.99)*100:+.1f}% max={pp.max()*100:+.1f}% | "
          f"n>=+30% {int((pp >= 0.30).sum())} n>=+50% {int((pp >= 0.50).sum())} | "
          f"top10_share={top10/tot*100:.0f}% (tong {tot*100:.0f}u)", flush=True)

print("\n=== (c) bootstrap subsample 80% x200 (full adv, K25) ===")
for name, csv in (("ab_noT", AB), ("t2907", T2907), ("gb", GB)):
    sm = NavSim2(csv, "2020-01-01")
    all_trades = list(sm.trades)
    finals, dds = [], []
    for rep in range(200):
        rng = random.Random(rep)
        sm.trades = rng.sample(all_trades, int(len(all_trades) * 0.8))
        m = sm.run(K=25, advance_fee=0.0008, order_seed=rep)
        finals.append(m["final"])
        dds.append(m["maxdd"] * 100)
    sm.trades = all_trades
    dds_s = sorted(dds)
    fin_s = sorted(finals)
    print(f"  {name}: NAV p5={fin_s[10]:.2f} med={fin_s[100]:.2f} p95={fin_s[189]:.2f} | "
          f"DD med={statistics.median(dds):.1f}% p90={dds_s[20]:.1f}% p95={dds_s[10]:.1f}% "
          f"worst={dds_s[0]:.1f}%", flush=True)

print("\n=== (d) knife check theo exit_reason (ab) ===")
for rs, g in t.groupby("exit_reason"):
    print(f"  {rs}: n={len(g)} mean={g.p.mean()*100:+.2f}% med={g.p.median()*100:+.2f}% "
          f"min={g.p.min()*100:.1f}% p5={g.p.quantile(0.05)*100:.1f}% "
          f"sh<-10%={100*(g.p < -0.10).mean():.1f}% sh<-15%={100*(g.p < -0.15).mean():.1f}% "
          f"sum={g.p.sum()*100:+.0f}u")
mh = t[t.exit_reason == "max_hold"]
print("  8 lenh max_hold te nhat:")
for r in mh.nsmallest(8, "p")[["symbol", "entry_date", "exit_date", "holding_days", "p"]].itertuples():
    print(f"    {r.symbol} {r.entry_date}->{r.exit_date} h{r.holding_days} {r.p*100:+.1f}%")
print("PR5_40_DONE")
