# -*- coding: utf-8 -*-
"""pr3_21: tội 2 — bỏ top-20 trade (theo pnl_pct) CẢ HAI bên rồi chấm lại NAV
(K25 adv, n=20). Nếu delta sập -> lợi thế compound tựa vài lệnh."""
import sys
import math
import os
import tempfile

import pandas as pd

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
CSV_P42 = f"{R2}/r2c_oxt04_p42_s42_trades.csv"
CSV_GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"


def drop_top(csv, k):
    t = pd.read_csv(csv)
    t = t.sort_values("pnl_pct", ascending=False).iloc[k:]
    f = os.path.join(tempfile.gettempdir(), f"pr3_drop{k}_" + os.path.basename(csv))
    t.to_csv(f, index=False)
    return f


def delta(sa, sg):
    d = (sa["mean"] / sg["mean"] - 1) * 100
    sd = math.sqrt((sa["sd"] / sg["mean"]) ** 2
                   + (sg["sd"] * sa["mean"] / sg["mean"] ** 2) ** 2) * 100
    return f"{d:+.1f}%±{sd:.1f} ({d/sd:+.1f}sd)"


for k in (10, 20):
    for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22")):
        sa = shuffle_stats(NavSim2(drop_top(CSV_P42, k), lo), K=25, advance_fee=0.0008, n=20)
        sg = shuffle_stats(NavSim2(drop_top(CSV_GB, k), lo), K=25, advance_fee=0.0008, n=20)
        print(f"drop-top-{k} {tag}: p42 x{sa['mean']:.2f}±{sa['sd']:.2f} "
              f"gb x{sg['mean']:.2f}±{sg['sd']:.2f} -> {delta(sa, sg)}", flush=True)
print("DONE")
