# -*- coding: utf-8 -*-
"""hc_01_oracle: ORACLE CEILING cho y tuong hold-class (phan lop tran 16 vs 40/60).

Counterfactual GIU TIEP cho 1.027 lenh cham tran mh16:
  - Exit stack toi gian = TRAN MOI (close tai bar i1+24 ~ tran 40 phien,
    i1+44 ~ tran 60 phien). KHONG tai lap signal/trailing trong extension
    (bao thu ve phia don gian; oracle bu bang chon lenh perfect-foresight).
  - new_exit_price = exit_price * close[new_i1]/close[i1]  (db back-adjusted,
    da verify ratio1~0.9995 o hc_00).
  - Giu tiep KHONG ton them roundtrip; recycle ton 0.6%/vong (nhu nh_nav2).

Doi chung chi phi co hoi (control chain) cho tung lenh capped:
  - Von giai phong tai exit d, T+2 -> lenh thuc te ke tiep entry >= d+2 phien
    (tren global calendar), tie-break alphabet nhu sim; chain den het cua so
    extension (D_end = ngay tran moi); leg cuoi pro-rate tuyen tinh theo bar;
    gap = cash 0%. Return net (roundtrip 0.006) nhu nh_nav2.

Output:
  (a) % lenh capped ma giu-tiep THANG control chain (40 va 60)
  (b) hc_capped_ext.csv: per-trade extension/control de step 2 dung
  (c) 3 CSV oracle trades -> hc_02_nav.py cham NAV
"""
import sqlite3
import bisect
import pandas as pd
import numpy as np

BASE = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r3line"
DB = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
CSV = f"{BASE}/r3_mh16_s42_trades.csv"
OUT = f"{BASE}/holdclass"
S0, FEE, ROUNDTRIP = 0.001, 0.004, 0.006
SETTLE = 2
EXT = {40: 24, 60: 44}   # tran moi -> so bar them sau i1 (i1 = i0+17)

df = pd.read_csv(CSV)
df["entry_date"] = df["entry_date"].astype(str).str[:10]
df["exit_date"] = df["exit_date"].astype(str).str[:10]
df = df.reset_index().rename(columns={"index": "tid"})

syms = sorted(set(df.symbol))
con = sqlite3.connect(DB)
px = pd.read_sql_query(
    "SELECT symbol,date,close FROM ohlcv WHERE symbol IN (%s) AND date>='2019-06-01'"
    % ",".join("?" * len(syms)), con, params=syms)
con.close()
closes, dates, idx = {}, {}, {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date")
    closes[s] = g["close"].to_numpy()
    dates[s] = g["date"].tolist()
    idx[s] = {d: i for i, d in enumerate(dates[s])}
gcal = sorted(set(px["date"]))
gidx = {d: i for i, d in enumerate(gcal)}


def gpos(d):  # vi tri d tren global calendar (ceil)
    return gidx.get(d, bisect.bisect_left(gcal, d))


# net return cua 1 lenh theo quy uoc nh_nav2 (roundtrip 0.006)
s_new = (ROUNDTRIP - FEE) / 2.0
def net_of(e_price, x_price):
    e_raw, x_raw = e_price / (1 + S0), x_price / (1 - S0)
    return (x_raw * (1 - s_new)) / (e_raw * (1 + s_new)) - 1.0 - FEE

df["net"] = [net_of(r.entry_price, r.exit_price) for r in df.itertuples()]

# --- extension cho capped ---
cap = df[df.exit_reason == "max_hold"].copy()
ext_rows = []
for r in cap.itertuples():
    s = r.symbol
    i1 = idx[s][r.exit_date]
    row = dict(tid=r.tid, symbol=s, entry_date=r.entry_date, exit_date=r.exit_date,
               net16=r.net, pnl16=r.pnl_pct)
    for mh, k in EXT.items():
        j = min(i1 + k, len(closes[s]) - 1)
        mult = closes[s][j] / closes[s][i1]
        row[f"ext_mult{mh}"] = mult          # he so gia them (khong ton phi)
        row[f"ext_date{mh}"] = dates[s][j]
        row[f"ext_bars{mh}"] = j - i1
        row[f"clip{mh}"] = int(j < i1 + k)
    ext_rows.append(row)
ext = pd.DataFrame(ext_rows)

# --- control chain (recycle) ---
pool = df.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
pool_dates = pool["entry_date"].tolist()

def control_chain(exit_d, end_d):
    """Compound net cua chuoi lenh thuc te ke tiep tu exit_d den end_d."""
    mult = 1.0
    cur_free = exit_d           # ngay von duoc giai phong (truoc settle)
    p_end = gpos(end_d)
    while True:
        p_free = gpos(cur_free)
        p_avail = p_free + SETTLE
        if p_avail >= len(gcal):
            break
        d_avail = gcal[p_avail]
        k = bisect.bisect_left(pool_dates, d_avail)
        if k >= len(pool):
            break
        t = pool.iloc[k]        # alphabet-first cung ngay (pool da sort)
        p_in = gpos(t.entry_date)
        if p_in >= p_end:
            break
        p_out = gpos(t.exit_date)
        if p_out <= p_end:
            mult *= (1.0 + t.net)
            cur_free = t.exit_date
        else:  # leg cuoi: pro-rate tuyen tinh theo bar
            frac = (p_end - p_in) / max(p_out - p_in, 1)
            mult *= (1.0 + t.net * frac)
            break
    return mult

for mh in EXT:
    ext[f"ctrl_mult{mh}"] = [control_chain(r.exit_date, r[f"ext_date{mh}"])
                             for _, r in ext.iterrows()]
    ext[f"hold_win{mh}"] = ext[f"ext_mult{mh}"] > ext[f"ctrl_mult{mh}"]
    ext[f"edge{mh}"] = ext[f"ext_mult{mh}"] - ext[f"ctrl_mult{mh}"]

ext["year"] = ext.entry_date.str[:4]
print("=== (a) GIU-TIEP vs RECYCLE (per-trade, control chain thuc te) ===")
for mh in EXT:
    w = ext[f"hold_win{mh}"]
    print(f"\n-- tran {mh} (them {EXT[mh]} bar) --")
    print(f"hold thang: {w.sum()}/{len(ext)} = {w.mean()*100:.1f}%")
    print(f"ext_mult mean {ext[f'ext_mult{mh}'].mean():.4f} | "
          f"ctrl_mult mean {ext[f'ctrl_mult{mh}'].mean():.4f} | "
          f"edge mean {ext[f'edge{mh}'].mean():+.4f} median {ext[f'edge{mh}'].median():+.4f}")
    print("theo nam (n, %hold thang, edge mean):")
    g = ext.groupby("year").agg(n=(f"hold_win{mh}", "size"),
                                win=(f"hold_win{mh}", "mean"),
                                edge=(f"edge{mh}", "mean"))
    print((g.assign(win=lambda x: (x.win * 100).round(1),
                    edge=lambda x: x.edge.round(4))).to_string())

ext.to_csv(f"{OUT}/hc_capped_ext.csv", index=False)

# --- build oracle trades CSV ---
def build_oracle(name, choose):
    """choose(r) -> None (giu 16) hoac mh (40/60)."""
    out = df.copy()
    n_ext = 0
    emap = ext.set_index("tid")
    for tid, r in emap.iterrows():
        mh = choose(r)
        if mh is None:
            continue
        n_ext += 1
        out.loc[out.tid == tid, "exit_date"] = r[f"ext_date{mh}"]
        out.loc[out.tid == tid, "exit_price"] = \
            df.loc[df.tid == tid, "exit_price"].iloc[0] * r[f"ext_mult{mh}"]
        out.loc[out.tid == tid, "exit_reason"] = f"oracle_mh{mh}"
    path = f"{OUT}/hc_oracle_{name}_trades.csv"
    out.drop(columns=["tid", "net"]).to_csv(path, index=False)
    print(f"[oracle {name}] extended {n_ext}/{len(emap)} -> {path}")
    return path

# O1: 2 lop 16/40, oracle = extend khi hold thang control
build_oracle("h40", lambda r: 40 if r.hold_win40 else None)
# O2: 2 lop 16/60
build_oracle("h60", lambda r: 60 if r.hold_win60 else None)
# O3: 3 lop 16/40/60 argmax edge (tran tuyet doi)
def best(r):
    cands = [(r.edge40, 40), (r.edge60, 60)]
    e, mh = max(cands)
    return mh if e > 0 else None
build_oracle("best", best)
print("\nDone.")
