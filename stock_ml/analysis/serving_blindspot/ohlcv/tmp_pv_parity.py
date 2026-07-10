# -*- coding: utf-8 -*-
"""Parity check: new catalog features (pv_corr_10, clv, nr_pos_7) vs screen_ic.py formulas.

Also asserts the champion set exit_vol_downpress is untouched (19 members, exact list).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3

REPO = Path(r"f:\PROJECTS\train_ai_ml")
sys.path.insert(0, str(REPO))

from stock_ml.src.features.catalog import SETS, FEATURES  # noqa: E402
from stock_ml.src.features.resolver import FeatureResolver  # noqa: E402

# 1. champion set byte-identity ------------------------------------------------
OLD_19 = [
    "atr_14_ratio", "realized_vol_10", "vol_percentile_60", "bb_width_20",
    "volatility_rank", "high_low_pct_5d", "ma5_accel",
    "dist_63d_high", "dist_52w_high", "sma_20_ratio", "bb_pct_20",
    "market_volatility_regime", "market_trend", "momentum_rank",
    "dist_day_25", "dist_day_vol20_25",
    "down_vol_intensity_5", "down_vol_count_10", "updown_vol_20",
]
assert SETS["exit_vol_downpress"][1] == OLD_19, "champion set CHANGED!"
new_full = SETS["exit_vol_downpress_pv"][1]
assert new_full == OLD_19 + ["pv_corr_10", "clv", "nr_pos_7"], new_full
new_only = SETS["exit_vol_downpress_pvonly"][1]
assert new_only == OLD_19 + ["pv_corr_10"], new_only
print("set membership OK: old 19 intact, pv=22, pvonly=20")

# 2. numeric parity on real data ----------------------------------------------
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"
SYMS = ["AAA", "SSI", "HPG", "VND", "DIG"]
con = sqlite3.connect(DB)
q = ("select symbol, date, open, high, low, close, volume from ohlcv "
     f"where symbol in ({','.join('?' * len(SYMS))}) order by symbol, date")
df = pd.read_sql(q, con, params=SYMS)
con.close()
df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)].reset_index(drop=True)
print(f"panel {df.symbol.nunique()} syms {len(df)} bars")

resolver = FeatureResolver.from_catalog()
# resolve just the 3 features via a throwaway member list
resolver.set_members["_pv_parity"] = ["pv_corr_10", "clv", "nr_pos_7"]
feat, _, _ = resolver.resolve(df.copy(), ["_pv_parity"], cache=False)

# screen_ic.py reference formulas (per symbol)
ref_parts = []
for s, g in df.groupby("symbol"):
    g = g.reset_index(drop=True)
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"].astype(float)
    pc = c.shift(1)
    ret1 = c.pct_change()
    rng_hl = (h - l).where(h > l)
    r = pd.DataFrame({"symbol": s, "date": g["date"]})
    dlv = np.log(v.replace(0, np.nan)).diff()
    r["ref_pv_corr_10"] = ret1.rolling(10).corr(dlv)
    r["ref_clv"] = ((c - l) - (h - c)) / rng_hl
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    r["ref_nr_pos_7"] = tr / tr.rolling(7).max().replace(0, np.nan)
    ref_parts.append(r)
ref = pd.concat(ref_parts, ignore_index=True)

m = feat.merge(ref, on=["symbol", "date"], how="left")


def cmp(new, old, note=""):
    a, b = m[new].to_numpy(float), m[old].to_numpy(float)
    both = ~np.isnan(a) & ~np.isnan(b)
    md = np.nanmax(np.abs(a[both] - b[both])) if both.any() else np.nan
    only_a = int((~np.isnan(a) & np.isnan(b)).sum())
    only_b = int((np.isnan(a) & ~np.isnan(b)).sum())
    print(f"{new:12s} n_both={both.sum():6d} maxdiff={md:.3e} "
          f"catalog-only-vals={only_a} screen-only-vals={only_b}  {note}")
    return md


d1 = cmp("pv_corr_10", "ref_pv_corr_10")
d2 = cmp("clv", "ref_clv", "(catalog clv: h==l -> 0 not NaN, known idiom)")
d3 = cmp("nr_pos_7", "ref_nr_pos_7", "(catalog: all-zero week -> 0 not NaN)")

# clv difference should ONLY be at h==l bars
hl = m.merge(df[["symbol", "date", "high", "low"]], on=["symbol", "date"],
             suffixes=("", "_raw"))
lock = (hl["high_raw"] <= hl["low_raw"]).to_numpy()
clv_c, clv_r = hl["clv"].to_numpy(float), hl["ref_clv"].to_numpy(float)
mismatch_pos = np.isnan(clv_r) & ~np.isnan(clv_c) & ~lock
print(f"clv extra-value rows outside limit-lock bars: {int(mismatch_pos.sum())}")

ok = (d1 < 1e-9) and (d2 < 1e-9) and (d3 < 1e-9) and mismatch_pos.sum() == 0
print("PARITY", "PASS" if ok else "FAIL")
