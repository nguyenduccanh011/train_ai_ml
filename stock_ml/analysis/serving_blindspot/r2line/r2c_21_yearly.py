# -*- coding: utf-8 -*-
"""r2c_21: yearly NAV (K25 adv, mean 20 perm) cho cac diem r2c vs ox04/gb — xem lat 2024."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "na_audit"))
from r2c_01_anatomy import yearly_mean  # noqa: E402
from nh_nav2 import CSV_GB, R2DIR  # noqa: E402

kw = dict(K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008)
names = sys.argv[1:] or ["r2c_oxt03"]
rows = {"gb": yearly_mean(CSV_GB, "2020-01-01", 20, **kw),
        "ox04": yearly_mean(f"{R2DIR}/r2b_oxtrail04_s42_trades.csv", "2020-01-01", 20, **kw)}
for n in names:
    rows[n] = yearly_mean(f"{R2DIR}/{n}_s42_trades.csv", "2020-01-01", 20, **kw)

years = sorted(rows["gb"])
hdr = "year | gb | ox04 | " + " | ".join(names) + " || " + \
      " | ".join(f"{n}-gb" for n in names)
print(hdr)
for y in years:
    g = rows["gb"][y]
    line = f"{y} | {g*100:+.1f} | {rows['ox04'][y]*100:+.1f}"
    for n in names:
        line += f" | {rows[n][y]*100:+.1f}"
    line += " ||"
    for n in names:
        line += f" {(rows[n][y]-g)*100:+.1f} |"
    print(line)
