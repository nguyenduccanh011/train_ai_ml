# -*- coding: utf-8 -*-
"""HY manh C: dinh luong swing-bo-phi trong lenh om dai cua pm2_hs10_zx25 (s42)
vs champion/gb_x08 chop-and-reenter tren CUNG ma + CUNG khoang thoi gian.

1) Lenh hold>100d cua pm2: dem song-nguoc >=15% / >=20% (peak->trough tren close,
   zigzag hindsight 15% trong doi lenh). Oracle ban-dinh-mua-lai-day tung song hoan chinh
   (gia recover >=15% tu trough truoc khi exit): extra factor = (P/T)*(1-0.007).
   missed_u = (1+pnl) * (PROD factors - 1). Downleg cuoi (peak->exit, chua recover)
   bao rieng lam "giveback cuoi" (khong tinh vao oracle chop).
2) Champ 2646 s42 + gb_x08 s42: cac trade cung ma co entry_date trong [entry, exit]
   cua lenh pm2 -> tong pnl_pct (u thuc, unit-weight nhu composite).
Out: hy_01_swing.csv + bang tong hop stdout.
"""
import os

import duckdb
import numpy as np
import pandas as pd

BASE = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
HERE = os.path.join(BASE, "pureml")
COST_RT = 0.007  # 1 vong ban-mua lai
THR = 0.15

pm2 = pd.read_csv(os.path.join(HERE, "pm2_pm2_hs10_zx25_s42_trades.csv"),
                  parse_dates=["entry_date", "exit_date"])
hold = pm2[pm2.holding_days > 100].reset_index(drop=True)
print(f"pm2 hold>100d: {len(hold)} lenh, pnl {hold.pnl_pct.sum():+.1f}u "
      f"(toan bo run: {len(pm2)} lenh {pm2.pnl_pct.sum():+.1f}u)")

champ = pd.read_csv(os.path.join(BASE, "signalq", "st_champ2646_s42_trades.csv"),
                    parse_dates=["entry_date", "exit_date"])
gb = pd.read_csv(os.path.join(BASE, "exitmap", "gbx08_s42_trades.csv"),
                 parse_dates=["entry_date", "exit_date"])

syms = sorted(hold.symbol.unique())
duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = duck.execute(
    "SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) "
    "ORDER BY symbol, date".format(",".join(f"'{s}'" for s in syms))).df()
duck.close()
px["date"] = pd.to_datetime(px["date"])
PX = {s: g.set_index("date")["close"] for s, g in px.groupby("symbol")}


def zigzag_waves(c: np.ndarray, thr: float):
    """Hindsight zigzag: tra ve list song hoan chinh (peak, trough) voi drawdown>=thr
    va recover>=thr tu trough; + downleg cuoi chua recover (peak, last) neu >=thr."""
    waves = []
    tail = None
    direction = 1  # bat dau coi nhu dang up tu entry
    ext_hi = c[0]
    ext_lo = c[0]
    peak = None
    for p in c[1:]:
        if direction == 1:
            if p > ext_hi:
                ext_hi = p
            elif p <= ext_hi * (1.0 - thr):
                direction = -1
                peak = ext_hi
                ext_lo = p
        else:
            if p < ext_lo:
                ext_lo = p
            elif p >= ext_lo * (1.0 + thr):
                waves.append((peak, ext_lo))
                direction = 1
                ext_hi = p
                peak = None
    if direction == -1 and peak is not None and c[-1] <= peak * (1.0 - thr):
        tail = (peak, c[-1])
    return waves, tail


rows = []
for _, t in hold.iterrows():
    s = PX[t.symbol]
    c = s.loc[t.entry_date:t.exit_date].to_numpy(float)
    if len(c) < 3:
        continue
    waves, tail = zigzag_waves(c, THR)
    n15 = len(waves)
    n20 = sum(1 for p, tr in waves if tr / p <= 0.80)
    depths = [1.0 - tr / p for p, tr in waves]
    fac = 1.0
    for p, tr in waves:
        f = (p / tr) * (1.0 - COST_RT)
        if f > 1.0:
            fac *= f
    missed = (1.0 + t.pnl_pct) * (fac - 1.0)
    tail_gb = (1.0 - tail[1] / tail[0]) if tail else 0.0
    # champ / gb tren cung ma + cung khoang
    cm = champ[(champ.symbol == t.symbol) & (champ.entry_date >= t.entry_date) &
               (champ.entry_date <= t.exit_date)]
    gm = gb[(gb.symbol == t.symbol) & (gb.entry_date >= t.entry_date) &
            (gb.entry_date <= t.exit_date)]
    rows.append(dict(
        symbol=t.symbol, entry=t.entry_date.date(), exit=t.exit_date.date(),
        hold=int(t.holding_days), pnl=t.pnl_pct, year=t.entry_date.year,
        n_wave15=n15, n_wave20=n20, max_depth=max(depths, default=0.0),
        oracle_factor=fac, missed_u=missed, tail_giveback=tail_gb,
        champ_n=len(cm), champ_u=cm.pnl_pct.sum(),
        gb_n=len(gm), gb_u=gm.pnl_pct.sum()))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "hy_01_swing.csv"), index=False)

print("\n=== TONG HOP (lenh pm2 hold>100d, n=%d) ===" % len(df))
print(f"pm2 u thuc:              {df.pnl.sum():+8.1f}")
print(f"song-nguoc >=15%%: {int(df.n_wave15.sum())} song trong {int((df.n_wave15 > 0).sum())} lenh; "
      f">=20%%: {int(df.n_wave20.sum())} song trong {int((df.n_wave20 > 0).sum())} lenh")
print(f"oracle chop missed_u:    {df.missed_u.sum():+8.1f}  (tru cost 0.7%%/vong)")
print(f"tail giveback cuoi>=15%%: n={int((df.tail_giveback >= THR).sum())}, "
      f"tong depth-u xap xi {((1+df.pnl)*df.tail_giveback).sum():+.1f}")
print(f"champ 2646 cung song:    {df.champ_u.sum():+8.1f}u / {int(df.champ_n.sum())} trades")
print(f"gb_x08 cung song:        {df.gb_u.sum():+8.1f}u / {int(df.gb_n.sum())} trades")

print("\n=== THEO NAM (entry-year cua lenh pm2) ===")
agg = df.groupby("year").agg(n=("pnl", "size"), pm2_u=("pnl", "sum"),
                             waves15=("n_wave15", "sum"), missed=("missed_u", "sum"),
                             champ_u=("champ_u", "sum"), champ_n=("champ_n", "sum"),
                             gb_u=("gb_u", "sum"), gb_n=("gb_n", "sum")).round(1)
print(agg.to_string())

print("\n=== TOP-15 lenh missed_u lon nhat ===")
top = df.sort_values("missed_u", ascending=False).head(15)
print(top[["symbol", "entry", "exit", "hold", "pnl", "n_wave15", "max_depth",
           "missed_u", "champ_n", "champ_u", "gb_n", "gb_u"]].round(2).to_string(index=False))

# phan xu: tren cac lenh CO song >=15%
m = df[df.n_wave15 > 0]
print("\n=== PHAN XU: chi cac lenh co song>=15%% (n=%d) ===" % len(m))
print(f"pm2 om: u thuc {m.pnl.sum():+.1f} + bo phi {m.missed_u.sum():+.1f} "
      f"= tran oracle {(m.pnl + m.missed_u).sum():+.1f}")
print(f"champ chop thuc te cung song: {m.champ_u.sum():+.1f}u ({int(m.champ_n.sum())} tr) | "
      f"gb: {m.gb_u.sum():+.1f}u ({int(m.gb_n.sum())} tr)")
