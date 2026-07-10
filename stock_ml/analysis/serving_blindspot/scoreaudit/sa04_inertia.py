# -*- coding: utf-8 -*-
"""SCORE AUDIT buoc 4 — do i cua ong dan quyet dinh + co che nuot tin hieu moi.

A. DO I: voi moi kenh & nam, can dich z bao nhieu sigma (delta) de LAT them 1%/5%
   so quyet dinh hien tai (them bar vuot nguong khi score duoc cong delta*sigma —
   dich tuc thoi, rolling stats qua khu chua kip hap thu). delta = khoang cach
   quantile cua phan phoi z den nguong.
B. NUOT TIN HIEU THAT: head exit thay the reward_risk_regression h10 (bundle
   n2_3h_brk05, cung universe/split/seed) tron vao exit_score voi trong so w:
   blend = (1-w)*s_std + w*alt_std (chuan hoa per-symbol causal truoc khi tron,
   nhu mot head moi duoc cong vao raw), sau do z-norm 252/60 + threshold 2.0 +
   gate cons2_w20 — do % quyet dinh sell thay doi theo nam, va so sell MOI o
   2024-26 (vung doi). Doi chieu: alt head DUNG MOT MINH vuot z>2.0 bao nhieu %.
C. NUOT TIN HIEU TONG QUAT: tin hieu doc lap g~N(0,1) (IC nho) tron w=0.1/0.2/0.3
   -> % quyet dinh doi (trung binh 20 seed).
Out: sa04_inertia.csv, sa04_blend.csv + bang in.
"""
import os

import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\scoreaudit"
ALT = r"f:\PROJECTS\train_ai_ml\bundles\bundle_n2_3h_brk05_z07_2025-01-01_wf"
CH = {"z_score": -1.9, "z_score2": 0.9, "z_score3": 0.7, "z_score4": 0.7,
      "z_score5": 0.7, "z_exit_score": 2.0}

ph = pd.read_parquet(os.path.join(OUT, "sa_scores.parquet"))
ph["year"] = ph.date.str[:4]
ph.loc[ph.date >= "2026-01-01", "year"] = "2026H1"


def causal_z(s, sym, w=252, mp=60):
    def _cz(x):
        m = x.rolling(w, min_periods=mp).mean()
        sd = x.rolling(w, min_periods=mp).std()
        return (x - m) / (sd + 1e-12)
    return s.groupby(sym, sort=False).transform(_cz)


# ---------- A. inertia ----------
rows = []
for zc, thr in CH.items():
    for y, g in ph.groupby("year"):
        z = g[zc].dropna()
        n = len(z)
        cur = (z > thr).mean()
        d = {}
        for pct in (0.01, 0.05):
            # delta sao cho P(thr - delta < z <= thr) = pct (lat quyet dinh 0->1)
            lo, hi = 0.0, 12.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if ((z > thr - mid) & (z <= thr)).mean() < pct:
                    lo = mid
                else:
                    hi = mid
            d[pct] = hi if hi < 11.9 else np.nan
        rows.append(dict(score=zc, year=y, n=n, rate_cur=cur * 100,
                         sig_flip1=d[0.01], sig_flip5=d[0.05]))
ine = pd.DataFrame(rows)
ine.to_csv(os.path.join(OUT, "sa04_inertia.csv"), index=False)
pd.set_option("display.width", 250)
print("==== DO I: sigma can cong vao score de LAT them 1%/5% bar-quyet-dinh ====")
piv = ine.pivot_table(index="year", columns="score", values="sig_flip5")
print("sigma de lat 5% (theo nam):")
print(piv.round(2).to_string())
piv1 = ine.pivot_table(index="year", columns="score", values="sig_flip1")
print("sigma de lat 1% (theo nam):")
print(piv1.round(2).to_string())

# ---------- B. blend voi head that ----------
alt = pd.read_parquet(os.path.join(ALT, "prediction_history.parquet"),
                      columns=["symbol", "date", "exit_score"])
alt["date"] = pd.to_datetime(alt["date"]).dt.strftime("%Y-%m-%d")
alt = alt.rename(columns={"exit_score": "alt_exit"})
ph = ph.merge(alt, on=["symbol", "date"], how="left")
print("\nalt head (reward_risk h10) phu:", ph.alt_exit.notna().mean() * 100, "% bar")

# chuan hoa per-symbol causal (expanding std tu qua khu) de tron 2 head khac don vi
def _std_causal(s, sym):
    def _f(x):
        m = x.expanding(60).mean()
        sd = x.expanding(60).std()
        return (x - m) / (sd + 1e-12)
    return s.groupby(sym, sort=False).transform(_f)

ph = ph.sort_values(["symbol", "date"]).reset_index(drop=True)
s_std = _std_causal(ph["exit_score"], ph["symbol"])
a_std = _std_causal(ph["alt_exit"], ph["symbol"])
base_sell = (ph.z_exit_score > 2.0) & ph.gate_exit
print("corr(s_std, alt_std) =", round(float(s_std.corr(a_std)), 3))

rows = []
for w in (0.1, 0.2, 0.3, 0.5, 1.0):
    blend = (1 - w) * s_std + w * a_std
    zb = causal_z(blend, ph["symbol"])
    sell_b = (zb > 2.0) & ph.gate_exit & ph.alt_exit.notna()
    for y, g in ph.assign(sb=sell_b, s0=base_sell & ph.alt_exit.notna()).groupby("year"):
        changed = (g.sb != g.s0).mean() * 100
        rows.append(dict(w=w, year=y, rate_base=g.s0.mean() * 100,
                         rate_blend=g.sb.mean() * 100, pct_changed=changed,
                         new_sells=int((g.sb & ~g.s0).sum()),
                         lost_sells=int((~g.sb & g.s0).sum())))
bl = pd.DataFrame(rows)
bl.to_csv(os.path.join(OUT, "sa04_blend.csv"), index=False)
print("\n==== BLEND head that (w=trong so head moi) — sell rate %/bar & thay doi ====")
print(bl.round(3).to_string(index=False))

# alt head dung mot minh: z 252/60 co vuot 2.0 khong?
z_alt = causal_z(ph["alt_exit"], ph["symbol"])
print("\nalt head alone: % bar z>2.0 theo nam:")
for y, g in ph.assign(za=z_alt).groupby("year"):
    print("  %s %.3f%% (q99 z = %.2f)" % (y, (g.za > 2.0).mean() * 100,
                                          g.za.quantile(0.99)))

# ---------- C. noise blend tong quat ----------
rng = np.random.default_rng(0)
res = {w: [] for w in (0.1, 0.2, 0.3)}
zx = ph["z_exit_score"].to_numpy()
gate = ph["gate_exit"].to_numpy()
ok = ~np.isnan(zx)
for seed in range(20):
    gnoise = rng.standard_normal(len(ph))
    for w in res:
        zb = ((1 - w) * zx + w * gnoise) / np.sqrt((1 - w) ** 2 + w ** 2)
        sb = (zb > 2.0) & gate & ok
        s0 = (zx > 2.0) & gate & ok
        res[w].append((sb != s0).mean() * 100)
print("\n==== NOISE blend (tin hieu doc lap sigma=1): % quyet dinh sell doi ====")
for w, v in res.items():
    print("  w=%.1f -> %.3f%% bar doi (~%.1f%% so voi sell-rate pooled %.3f%%)" % (
        w, np.mean(v), 100 * np.mean(v) / (100 * ((zx > 2.0) & gate & ok).mean()),
        100 * ((zx > 2.0) & gate & ok).mean()))
