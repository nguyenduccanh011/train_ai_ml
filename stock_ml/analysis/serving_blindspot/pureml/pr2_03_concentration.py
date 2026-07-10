# -*- coding: utf-8 -*-
"""pr2 (cong to) tội 3+4b: concentration/lottery + gap-xuyen-stop.

- Top-10 lenh chiem bao nhieu % pnl (per seed, pm2 vs gb/champ s42).
- Drop top-5 lenh: composite con lai (pm2 per-seed; gb/champ s42 doi chieu cong bang).
- Seed-fragility mega runner: LPB/VTP/BSR + top-10 winner overlap qua 5 seed.
- Gap-through-stop: lenh hard_stop fill te hon -10% bao nhieu, phan phoi.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from stock_ml.src.evaluation.scoring import calc_metrics, calc_mdd_per_symbol, composite_score  # noqa: E402

D = Path(__file__).parent
SB = D.parent

FILES = {
    ("pm2", 42): D / "pm2_pm2_hs10_zx25_s42_trades.csv",
    ("pm2", 7): D / "pm2_pm2_hs10_zx25_s7_trades.csv",
    ("pm2", 99): D / "pm2_pm2_hs10_zx25_s99_trades.csv",
    ("pm2", 555): D / "pm2_pm2_hs10_zx25_s555_trades.csv",
    ("pm2", 123): D / "pm2_pm2_hs10_zx25_s123_trades.csv",
    ("gb_x08", 42): SB / "signalq/nicheloss/gb_x08_s42_trades.csv",
    ("champ", 42): SB / "signalq/st_champ2646_s42_trades.csv",
}
GBX08_COMP = {42: 735.0, 7: 736.7, 99: 728.1, 555: 735.7, 123: 733.3}


def load(p):
    return pd.read_csv(p, parse_dates=["entry_date", "exit_date"])


def comp_of(df, n_sym):
    tr = [dict(symbol=r.symbol, entry_date=str(r.entry_date.date()), pnl_pct=r.pnl_pct,
               holding_days=r.holding_days) for r in df.itertuples()]
    m = calc_metrics(tr)
    m["n_symbols"] = n_sym
    return composite_score(m, tr), m["total_pnl"]


print("=== TOI 3: CONCENTRATION ===")
for (name, seed), p in FILES.items():
    df = load(p)
    n_sym = df.symbol.nunique()
    tot = df.pnl_pct.sum()
    top = df.nlargest(10, "pnl_pct")
    top5 = df.nlargest(5, "pnl_pct")
    c_full, _ = comp_of(df, n_sym)
    c_no5, pnl_no5 = comp_of(df.drop(top5.index), n_sym)
    c_no10, pnl_no10 = comp_of(df.drop(top.index), n_sym)
    print(f"{name} s{seed}: pnl={tot:.1f} top5={top5.pnl_pct.sum():.1f} ({top5.pnl_pct.sum()/tot*100:.0f}%) "
          f"top10={top.pnl_pct.sum():.1f} ({top.pnl_pct.sum()/tot*100:.0f}%) "
          f"comp full={c_full:.1f} -top5={c_no5:.1f} -top10={c_no10:.1f}")
    if name == "pm2":
        print(f"   vs gb_x08 full comp s{seed}={GBX08_COMP[seed]}: d(-top5)={c_no5-GBX08_COMP[seed]:+.1f}")
        print("   top5:", [(r.symbol, str(r.entry_date.date()), round(r.pnl_pct, 2), int(r.holding_days))
                           for r in top5.itertuples()])

print("\n=== TOI 3b: MEGA-RUNNER SEED FRAGILITY (pm2, lenh pnl>1.0 = >100%) ===")
mega_by_seed = {}
for seed in (42, 7, 99, 555, 123):
    df = load(FILES[("pm2", seed)])
    mega = df[df.pnl_pct > 1.0]
    mega_by_seed[seed] = {(r.symbol, r.entry_date.year) for r in mega.itertuples()}
    post21 = mega[mega.entry_date >= "2022-01-01"]
    print(f"s{seed}: n_mega={len(mega)} pnl_mega={mega.pnl_pct.sum():.1f} "
          f"(={mega.pnl_pct.sum()/df.pnl_pct.sum()*100:.0f}% tong) | mega entry>=2022: n={len(post21)} "
          f"pnl={post21.pnl_pct.sum():.1f} {sorted((r.symbol, str(r.entry_date.date()), round(r.pnl_pct,2)) for r in post21.itertuples())}")
common = set.intersection(*mega_by_seed.values())
union = set.union(*mega_by_seed.values())
print(f"mega chung ca 5 seed: {len(common)}/{len(union)}: {sorted(common)}")

print("\n=== TOI 4b: GAP-THROUGH-STOP (pm2 per seed) ===")
for seed in (42, 7, 99, 555, 123):
    df = load(FILES[("pm2", seed)])
    hs = df[df.exit_reason == "hard_stop"]
    worse = hs[hs.pnl_pct < -0.115]  # te hon stop -10% + cost ~1.5%
    q = hs.pnl_pct.quantile([0.05, 0.25, 0.5])
    print(f"s{seed}: n_hardstop={len(hs)} mean={hs.pnl_pct.mean():.3f} median={q[0.5]:.3f} "
          f"p25={q[0.25]:.3f} p05={q[0.05]:.3f} | n<-11.5%={len(worse)} "
          f"n<-15%={len(hs[hs.pnl_pct < -0.15])} n<-20%={len(hs[hs.pnl_pct < -0.20])} "
          f"worst={hs.pnl_pct.min():.3f} pnl_extra_loss_vs_-10%={((hs.pnl_pct[hs.pnl_pct < -0.10] + 0.10).sum()):.2f}u")
s42 = load(FILES[("pm2", 42)])
hs42 = s42[s42.exit_reason == "hard_stop"]
bad = hs42[hs42.pnl_pct < -0.15].sort_values("pnl_pct")
print("\ns42 lenh hard_stop < -15%:")
for r in bad.itertuples():
    print(f"  {r.symbol} {r.entry_date.date()} -> {r.exit_date.date()} pnl={r.pnl_pct:.3f} hold={int(r.holding_days)}")
