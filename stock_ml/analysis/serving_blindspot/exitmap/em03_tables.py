"""EXIT MAP gb_x08 s42 — aggregation tables from gbx08_enriched.csv."""
import numpy as np
import pandas as pd

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
E = pd.read_csv(EM + "/gbx08_enriched.csv", parse_dates=["entry_date", "exit_date"])
pd.set_option("display.width", 220)

def agg(g):
    return pd.Series(dict(
        n=len(g), pnl=g.pnl_pct.sum(), pnl_per=g.pnl_pct.mean(),
        wr=(g.pnl_pct > 0).mean(), hold_med=g.holding_days.median(),
        mfe_med=g.mfe.median(), giveback_u=g.giveback_u.sum(),
        eff_med=g.eff.median(),
        post5=(g.post_max_c >= 0.05).mean(), post10=(g.post_max_c >= 0.10).mean(),
    ))

print("=" * 100)
print("1) ATTRIBUTION BY RULE (all years)")
T = E.groupby("label").apply(agg).sort_values("pnl", ascending=False)
print(T.round(3).to_string())

print("\n1b) RULE x EXIT-YEAR (n / pnl / wr)")
specs = [("n", "pnl_pct", "count"), ("pnl", "pnl_pct", "sum"),
         ("wr", "pnl_pct", lambda x: (x > 0).mean()), ("hold_med", "holding_days", "median")]
for m, val, fn in specs:
    P = E.pivot_table(index="label", columns="year_exit", values=val, aggfunc=fn)
    print(f"\n  metric={m}")
    print(P.round(2).to_string())

print("\n" + "=" * 100)
print("2) MFE MAP")
tot_pnl, tot_mfe = E.pnl_pct.sum(), E.mfe.sum()
print(f"total realized {tot_pnl:.1f}u | total MFE(high) {tot_mfe:.1f}u | total giveback {E.giveback_u.sum():.1f}u")
print(f"portfolio-level efficiency (sum pnl / sum mfe) = {tot_pnl/tot_mfe:.3f}")
Ec = E.copy(); Ec["mfe_pos"] = Ec.mfe.clip(lower=0)
win = E[E.pnl_pct > 0]
print(f"winners only: eff = {win.pnl_pct.sum()/win.mfe.sum():.3f} (pnl {win.pnl_pct.sum():.1f} / mfe {win.mfe.sum():.1f})")
los = E[E.pnl_pct <= 0]
print(f"losers: n={len(los)} pnl {los.pnl_pct.sum():.1f} mfe {los.mfe.sum():.1f} (missed profit on losers)")

print("\n2a) giveback_u by rule x year_exit")
P = E.pivot_table(index="label", columns="year_exit", values="giveback_u", aggfunc="sum")
P["ALL"] = P.sum(axis=1)
print(P.round(1).to_string())

hb = pd.cut(E.holding_days, [0, 5, 10, 20, 40, 1000], labels=["1-5", "6-10", "11-20", "21-40", ">40"])
print("\n2b) giveback_u / eff by hold bucket")
P = E.groupby(hb, observed=True).apply(
    lambda g: pd.Series(dict(n=len(g), pnl=g.pnl_pct.sum(), giveback_u=g.giveback_u.sum(),
                             eff_port=g.pnl_pct.sum() / max(g.mfe.sum(), 1e-9),
                             eff_med=g.eff.median())))
print(P.round(3).to_string())

print("\n2c) giveback_u by MFE bucket (where does the giveback live?)")
mb = pd.cut(E.mfe, [-1, 0.03, 0.08, 0.15, 0.27, 0.5, 10],
            labels=["<3%", "3-8%", "8-15%", "15-27%", "27-50%", ">50%"])
P = E.groupby(mb, observed=True).apply(
    lambda g: pd.Series(dict(n=len(g), pnl=g.pnl_pct.sum(), mfe=g.mfe.sum(),
                             giveback_u=g.giveback_u.sum(),
                             eff_port=g.pnl_pct.sum() / max(g.mfe.sum(), 1e-9))))
print(P.round(2).to_string())
print("\n2d) same but rule x MFE bucket (giveback_u)")
P = E.pivot_table(index="label", columns=mb, values="giveback_u", aggfunc="sum", observed=True)
print(P.round(1).to_string())

print("\n2e) snr_extend harvest check: deferred_before trades")
D = E[E.deferred_before == True]  # noqa: E712
print(f"deferred-at-some-point: n={len(D)} pnl {D.pnl_pct.sum():.2f} giveback_u {D.giveback_u.sum():.2f}")
print(D.groupby("label").apply(agg).round(3).to_string())

print("\n" + "=" * 100)
print("3) COHORT SOLD-THEN-RALLIED (post-exit max close within 20 bars)")
for X in [0.03, 0.05, 0.10, 0.15]:
    C = E[E.post_max_c >= X]
    # counterfactuals: hold 20 more bars -> exit at post_end_c (neutral) / post_max_c (oracle)
    u_neutral = (C.post_end_c * (1 + C.pnl_pct)).sum()  # approx compounding in entry units
    u_oracle = (C.post_max_c * (1 + C.pnl_pct)).sum()
    print(f"X={X:.0%}: n={len(C)} ({len(C)/len(E):.1%}) pnl_realized={C.pnl_pct.sum():+.1f} "
          f"| cf hold+20 end={u_neutral:+.1f}u | cf oracle-peak={u_oracle:+.1f}u")
C = E[E.post_max_c >= 0.05].copy()
print("\n3a) sold-then-rallied (X=5%) by rule")
print(C.groupby("label").apply(
    lambda g: pd.Series(dict(n=len(g), pnl=g.pnl_pct.sum(),
                             cf_end20_u=(g.post_end_c * (1 + g.pnl_pct)).sum(),
                             cf_peak_u=(g.post_max_c * (1 + g.pnl_pct)).sum(),
                             snr_med=g.snr_at.median(), gain_d_med=g.gain_d.median(),
                             giveback_d_med=g.giveback_d.median()))).round(3).to_string())
print("\n3b) sold-then-rallied (X=5%) by year_exit: n and cf_end20_u")
print(C.groupby("year_exit").apply(
    lambda g: pd.Series(dict(n=len(g), cf_end20_u=(g.post_end_c * (1 + g.pnl_pct)).sum()))).round(2).to_string())
print("\n3c) decision-bar state: rallied (X>=5%) vs not")
N = E[(E.post_max_c < 0.05) & E.post_max_c.notna()]
for col in ["snr_at", "gain_d", "peak_d", "giveback_d", "holding_days", "mfe"]:
    print(f"  {col:14s} rallied_med={C[col].median():+.3f}  flat_med={N[col].median():+.3f}")

print("\n" + "=" * 100)
print("4) COHORT SOLD-LATE (exit fill >= k bars after peak-close AND gave back >= y)")
E["late_gb_u"] = (E.mfe_c - E.pnl_pct)  # close-peak giveback in entry units
for k, y in [(3, 0.05), (5, 0.10), (10, 0.10)]:
    L = E[(E.bars_after_peak >= k) & (E.late_gb_u >= y)]
    print(f"k={k} bars, giveback>={y:.0%}: n={len(L)} realized={L.pnl_pct.sum():+.1f} "
          f"late_giveback_u={L.late_gb_u.sum():+.1f}")
L = E[(E.bars_after_peak >= 5) & (E.late_gb_u >= 0.10)]
print("\n4a) sold-late (k=5,y=10%) by rule")
print(L.groupby("label").apply(
    lambda g: pd.Series(dict(n=len(g), pnl=g.pnl_pct.sum(), late_gb_u=g.late_gb_u.sum(),
                             hold_med=g.holding_days.median(), mfe_med=g.mfe_c.median()))).round(3).to_string())
print("\n4b) sold-late by year_exit (late_gb_u)")
print(L.groupby("year_exit").late_gb_u.agg(["count", "sum"]).round(1).to_string())

print("\n4c) NET trade-off: sold-early cf_end20 vs sold-late giveback, by year_exit")
C2 = E[E.post_max_c >= 0.05]
a = C2.groupby("year_exit").apply(lambda g: (g.post_end_c * (1 + g.pnl_pct)).sum())
b = L.groupby("year_exit").late_gb_u.sum()
J = pd.DataFrame({"early_left_u(end20)": a, "late_giveback_u": b}).fillna(0)
J["net_room"] = J["early_left_u(end20)"] - 0 * J["late_giveback_u"]
print(J.round(1).to_string())
