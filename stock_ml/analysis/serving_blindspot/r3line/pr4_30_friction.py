# -*- coding: utf-8 -*-
"""pr4_30: T3 friction/thuc thi —
(a) fee sweep R 0.6/0.7/0.8/0.9/1.0 (full + f22, adv; delta vs gb cung R);
(b) ENTRY LAG +1 phien AP CA HAI BEN (don da lat r2c) — full/f22, adv + noadv;
(c) EXIT LAG +1 phien: (c1) chi tren lenh max_hold cua mh16 (gb giu nguyen);
    (c2) tat ca lenh CA HAI ben — full adv.
(d) per-trade edge mh16 vs gb vs r2c."""
import math
import sys

import pandas as pd

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/pr3")
sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from pr3_lib import NavSim2S  # noqa: E402
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

R3 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r3line"
R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
MH16 = f"{R3}/r3_mh16_s42_trades.csv"
R2C = f"{R2}/r2c_oxt04_p42_s42_trades.csv"


def prop_delta(sa, sg):
    d = (sa["mean"] / sg["mean"] - 1) * 100
    sd = math.sqrt((sa["sd"] / sg["mean"]) ** 2
                   + (sg["sd"] * sa["mean"] / sg["mean"] ** 2) ** 2) * 100
    return f"{d:+.1f}%±{sd:.1f} ({d/sd:+.1f}sd)"


print("=== (a) FEE SWEEP roundtrip R (adv 0.08%) ===")
for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22")):
    sm, sg_ = NavSim2(MH16, lo), NavSim2(GB, lo)
    for R in (0.006, 0.007, 0.008, 0.009, 0.010):
        sa = shuffle_stats(sm, K=25, roundtrip=R, advance_fee=0.0008, n=20)
        sg = shuffle_stats(sg_, K=25, roundtrip=R, advance_fee=0.0008, n=20)
        print(f"  {tag} R{R*100:.1f}: mh16 x{sa['mean']:.2f}±{sa['sd']:.2f} "
              f"gb x{sg['mean']:.2f}±{sg['sd']:.2f} -> {prop_delta(sa, sg)}", flush=True)


class LagSim(NavSim2S):
    def apply_exit_lag(self, lag=1, only_reason=None, reasons=None):
        """Doi exit +lag phien theo lich symbol; x_raw moi = close raw ngay moi."""
        n_app = 0
        for t, rs in zip(self.trades, reasons):
            if only_reason is not None and rs != only_reason:
                continue
            s = t["symbol"]
            closes = self.sym_close[s]
            n_i1 = t["i1"] + lag
            if n_i1 >= len(closes):
                continue
            inv = {v: k for k, v in self.sym_idx[s].items()}
            t["i1"] = n_i1
            t["x_raw"] = closes[n_i1]
            t["exit_date"] = inv[n_i1]
            n_app += 1
        return n_app


def reasons_for(sim, csv):
    """Map exit_reason theo dung thu tu trades da giu lai trong sim."""
    ta = pd.read_csv(csv)
    ta = ta[ta.entry_date.astype(str).str[:10] >= sim.date_lo]
    rs = []
    it = ta.itertuples()
    for r in it:
        s = r.symbol
        ed, xd = str(r.entry_date)[:10], str(r.exit_date)[:10]
        if ed not in sim.sym_idx.get(s, {}) or xd not in sim.sym_idx.get(s, {}):
            continue
        rs.append(str(r.exit_reason) if hasattr(r, "exit_reason") else "?")
    assert len(rs) == len(sim.trades)
    return rs


print("\n=== (b) ENTRY LAG +1 AP CA HAI BEN ===")
for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22")):
    for mode, fee in (("adv", 0.0008), ("noadv", None)):
        sm = LagSim(MH16, lo)
        dm = sm.apply_entry_lag(1)
        sg_ = LagSim(GB, lo)
        dg = sg_.apply_entry_lag(1)
        sa = shuffle_stats(sm, K=25, advance_fee=fee, n=20)
        sg = shuffle_stats(sg_, K=25, advance_fee=fee, n=20)
        print(f"  lag+1 {tag} {mode}: mh16 x{sa['mean']:.2f}±{sa['sd']:.2f} (drop {dm}) "
              f"gb x{sg['mean']:.2f}±{sg['sd']:.2f} (drop {dg}) -> {prop_delta(sa, sg)}",
              flush=True)

print("\n=== (c1) EXIT LAG +1 chi lenh max_hold cua mh16 (gb nguyen) full/f22 adv+noadv ===")
for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22")):
    sm = LagSim(MH16, lo)
    rs = reasons_for(sm, MH16)
    from collections import Counter
    if tag == "full":
        print("  exit_reason mh16:", Counter(rs).most_common())
    n_app = sm.apply_exit_lag(1, only_reason="max_hold", reasons=rs)
    for mode, fee in (("adv", 0.0008), ("noadv", None)):
        sa = shuffle_stats(sm, K=25, advance_fee=fee, n=20)
        sg = shuffle_stats(NavSim2(GB, lo), K=25, advance_fee=fee, n=20)
        print(f"  xlag_mh {tag} {mode} (ap {n_app} lenh): mh16 x{sa['mean']:.2f}±{sa['sd']:.2f} "
              f"vs gb x{sg['mean']:.2f}±{sg['sd']:.2f} -> {prop_delta(sa, sg)}", flush=True)

print("\n=== (c2) EXIT LAG +1 tat ca lenh CA HAI ben (full adv) ===")
sm = LagSim(MH16, "2020-01-01")
na = sm.apply_exit_lag(1, reasons=reasons_for(sm, MH16))
sg_ = LagSim(GB, "2020-01-01")
ng = sg_.apply_exit_lag(1, reasons=reasons_for(sg_, GB))
sa = shuffle_stats(sm, K=25, advance_fee=0.0008, n=20)
sg = shuffle_stats(sg_, K=25, advance_fee=0.0008, n=20)
print(f"  xlag_all full adv: mh16 x{sa['mean']:.2f}±{sa['sd']:.2f} ({na}) "
      f"gb x{sg['mean']:.2f}±{sg['sd']:.2f} ({ng}) -> {prop_delta(sa, sg)}")

print("\n=== (d) PER-TRADE EDGE ===")
for name, csv in (("mh16", MH16), ("gb", GB), ("r2c", R2C)):
    t = pd.read_csv(csv)
    p = t.pnl_pct if t.pnl_pct.abs().median() < 1 else t.pnl_pct / 100
    win, loss = p[p > 0], p[p <= 0]
    pf = win.sum() / abs(loss.sum())
    hold = t.holding_days if "holding_days" in t else None
    print(f"  {name}: n={len(t)} mean={p.mean()*100:+.2f}% med={p.median()*100:+.2f}% "
          f"wr={len(win)/len(t)*100:.1f}% pf={pf:.2f} hold_med={hold.median():.0f} "
          f"hold_mean={hold.mean():.1f} maxloss={p.min()*100:.1f}% p1={p.quantile(0.01)*100:.1f}% "
          f"p5={p.quantile(0.05)*100:.1f}% sh<-10%={100*(p < -0.10).mean():.1f}% "
          f"sh<-15%={100*(p < -0.15).mean():.1f}%")
print("PR4_30_DONE")
