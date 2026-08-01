"""PRODUCTION Stage-2 replay: apply the champion portfolio layer (hb_deploy_gtos) to
Stage-1 base trades DERIVED FROM THE PRODUCTION BUNDLE (retrained on Sieu Tin Hieu data),
sourcing conviction/meta panels from serving's market.duckdb + NAV marked on serving's
ohlcv.db (via NavSim2). Gives the HONEST production CAGR (single-seed = bundle seed).

Faithful port of hb_deploy_gtos.py: rewrite(early-cut/green-trail) + gates(conv/ret7/overshoot)
+ meta-priority(LGBM walk-forward) + K=10 sim (conviction sizing + preemption + T+2 + advance_fee)."""

from __future__ import annotations

import os
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path("F:/PROJECTS/train_ai_ml")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from nh_nav2 import FEE, NavSim2

SRC = REPO / "_champ_src"
SERVING_DUCK = os.environ.get(
    "REPLAY_DUCK", "C:/Users/DUC CANH PC/Desktop/stock-serving/market_data/market.duckdb"
)
_BASE_PARQUET = os.environ.get("REPLAY_BASE", str(SRC / "prod_base_trades.parquet"))
_SIG_PARQUET = os.environ.get("REPLAY_SIG", str(SRC / "prod_signals.parquet"))
_LABEL = os.environ.get("REPLAY_LABEL", "PRODUCTION (Sieu Tin Hieu)")
K, MARGIN, KCONV, TPLUS, R5THR, SKIP, GT = 10, 0.005, 2.0, 2, 0.02, 0.40, 0.08
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
FCOLS = [
    "score",
    "exit_score",
    "ret5",
    "ret20",
    "ret60",
    "vol20",
    "dist_h20",
    "dist_h63",
    "dist_l20",
    "dist_l63",
    "ma20r",
    "ma50r",
    "atr_pct",
    "updays10",
    "volr",
    "rs_mom20",
    "rs_mom60",
]

# ---------- panels from serving market.duckdb (full market, Sieu Tin Hieu) ----------
cx = duckdb.connect(SERVING_DUCK, read_only=True)
px = cx.execute(
    "SELECT symbol,date,low,close,high,volume FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date"
).fetchdf()
cx.close()
px["date"] = pd.to_datetime(px["date"])
CLO = {}
HI = {}
LO = {}
DIDX = {}
INV = {}
parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy()
    c, l, h = g["close"], g["low"], g["high"]
    CLO[s] = c.values
    HI[s] = h.values
    LO[s] = l.values
    DIDX[s] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])}
    INV[s] = {i: d.strftime("%Y-%m-%d") for i, d in enumerate(g["date"])}
    dd = c.diff()
    up = dd.clip(lower=0).rolling(14).mean()
    dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1
    g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9))
    g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c
    g["dist_ma50"] = c / c.rolling(50).mean() - 1
    g["ret5"] = c / c.shift(7) - 1  # ret7 (7-day window)
    parts.append(g[["symbol", "date", "volume"] + CS4 + ["atrpct", "dist_ma50", "ret5"]])
P = pd.concat(parts, ignore_index=True)
# ghost-bar mask (parity with stock_ml.portfolio.build_market_panel): a non-traded bar
# (volume<=0) must not vote in the cross-section — NaN its factors before ranking.
_ghost = P["volume"].fillna(0) <= 0
P.loc[_ghost, CS4 + ["atrpct", "dist_ma50"]] = np.nan
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
b4 = [c + "_r" for c in CS4]
P["cs5_ma50"] = P[b4 + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CSm = {
    (r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5)
    for r in P.itertuples()
}
R5 = {
    (r.symbol, str(r.date.date())): (r.ret5 if pd.notna(r.ret5) else np.nan) for r in P.itertuples()
}


# ---------- meta features (hb_112) from serving market.duckdb, 61 trade syms ----------
def meta_features(syms):
    d = duckdb.connect(SERVING_DUCK, read_only=True)
    ph = ",".join("?" * len(syms))
    q = d.execute(
        f"select symbol,date,open,high,low,close,volume from ohlcv where timeframe='1D' and symbol in ({ph}) order by symbol,date",
        syms,
    ).fetchdf()
    d.close()
    q["date"] = pd.to_datetime(q["date"])
    out = []
    for s, g in q.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        c = g["close"]
        g["ret5"] = c.pct_change(5)
        g["ret20"] = c.pct_change(20)
        g["ret60"] = c.pct_change(60)
        g["vol20"] = c.pct_change().rolling(20).std()
        g["dist_h20"] = c / c.rolling(20).max() - 1
        g["dist_h63"] = c / c.rolling(63).max() - 1
        g["dist_l20"] = c / c.rolling(20).min() - 1
        g["dist_l63"] = c / c.rolling(63).min() - 1
        g["ma20r"] = c / c.rolling(20).mean() - 1
        g["ma50r"] = c / c.rolling(50).mean() - 1
        tr = pd.concat(
            [g["high"] - g["low"], (g["high"] - c.shift()).abs(), (g["low"] - c.shift()).abs()],
            axis=1,
        ).max(axis=1)
        g["atr_pct"] = tr.rolling(14).mean() / c
        g["updays10"] = (c.diff() > 0).rolling(10).sum()
        g["volr"] = g["volume"] / g["volume"].rolling(20).mean()
        out.append(
            g[
                [
                    "symbol",
                    "date",
                    "ret5",
                    "ret20",
                    "ret60",
                    "vol20",
                    "dist_h20",
                    "dist_h63",
                    "dist_l20",
                    "dist_l63",
                    "ma20r",
                    "ma50r",
                    "atr_pct",
                    "updays10",
                    "volr",
                ]
            ]
        )
    df = pd.concat(out)
    df["rs_mom20"] = df.groupby("date")["ret20"].rank(pct=True)
    df["rs_mom60"] = df.groupby("date")["ret60"].rank(pct=True)
    return df


def build_tr_df(trades, signals, feat):
    """meta training frame from prod base trades (ORIGINAL) + prod recombined signals."""
    tr = trades[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].copy()
    tr = tr[tr.exit_date.notna()]
    sig = signals[(signals.signal == 1) & signals.score.notna()][
        ["symbol", "date", "score", "exit_score"]
    ].copy()
    tr["entry_date"] = pd.to_datetime(tr["entry_date"])
    tr["exit_date"] = pd.to_datetime(tr["exit_date"])
    tr["pnl"] = tr.exit_price / tr.entry_price - 1.0
    tr["yr"] = tr.entry_date.dt.year
    sig["date"] = pd.to_datetime(sig["date"])
    parts = []
    for s, tg in tr.groupby("symbol"):
        sg = sig[sig.symbol == s].sort_values("date")
        m = pd.merge_asof(
            tg.sort_values("entry_date"),
            sg[["date", "score", "exit_score"]].rename(columns={"date": "entry_date"}),
            on="entry_date",
            direction="backward",
        )
        parts.append(m)
    tr = pd.concat(parts).merge(
        feat.rename(columns={"date": "entry_date"}), on=["symbol", "entry_date"], how="left"
    )
    tr["edkey"] = tr.entry_date.dt.strftime("%Y-%m-%d")
    tr["t_pnl"] = tr["pnl"]
    return tr


def meta_preds(tr, tgt="t_pnl"):
    pm = {}
    for ty in range(2021, 2027):
        train = tr[tr.exit_date < f"{ty}-01-01"].dropna(subset=FCOLS + [tgt])
        test = tr[tr.yr == ty].dropna(subset=FCOLS)
        if len(train) < 100 or not len(test):
            continue
        mdl = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=15,
            min_data_in_leaf=30,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=5,
            lambda_l2=1.0,
            verbose=-1,
            deterministic=True,
            force_col_wise=True,
            random_state=1,
        )
        mdl.fit(train[FCOLS], train[tgt])
        for (_, row), p in zip(test.iterrows(), mdl.predict(test[FCOLS])):
            pm[(row.symbol, row.edkey)] = float(p)
    return pm


# ---------- exit rewrite (early-cut / green-trail) ----------
def rewrite(cvtr):
    ned, nep, nreason = [], [], []
    for r in cvtr.itertuples():
        di = DIDX.get(r.symbol, {})
        ei = di.get(str(r.entry_date)[:10])
        xi = di.get(str(r.exit_date)[:10])
        if ei is None or xi is None or xi <= ei:
            ned.append(r.exit_date)
            nep.append(r.exit_price)
            nreason.append(r.exit_reason)
            continue
        c = CLO[r.symbol]
        if xi > ei + 3 and c[ei + 2] / r.entry_price - 1.0 < 0.0:
            ned.append(INV[r.symbol][ei + 3])
            nep.append(float(c[ei + 3]))
            nreason.append("early_cut")
        else:
            ek = xi
            ep_ = r.exit_price
            rs = r.exit_reason
            peak = c[ei]
            for b in range(ei + 2, xi + 1):
                peak = max(peak, c[b])
                if c[b] <= peak * (1 - GT) and b >= ei + 2:
                    ek, ep_ = (b + 1, float(c[b + 1])) if b + 1 <= xi else (b, float(c[b]))
                    rs = "green_trail"
                    break
            ned.append(INV[r.symbol][ek])
            nep.append(ep_)
            nreason.append(rs)
    o = cvtr.copy()
    o["exit_date"] = pd.to_datetime(ned)
    o["exit_price"] = nep
    o["exit_reason"] = nreason
    return o


# ---------- sim (prun_metric port: conviction sizing + gates + preemption + T+2) ----------
OSMAP = None
OSTHR = 9.9
SKIP_BY_YEAR = None  # {year: skip_threshold} for causal per-year gate; None = use global SKIP
SKIP_BY_YEAR_DEFAULT = 0.40  # fallback for the earliest year (no prior history yet)


def _prep(sim, pm, cm, mu, sd):
    s_new = (0.006 - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
        t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd
        t["_w"] = min(max(1.0 + KCONV * z, 0.4), 1.8)
        raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)


def _skip_for(entry_date):
    """SKIP threshold for a trade entered on `entry_date`.
    - SKIP_BY_YEAR (causal mode): per-year threshold computed from PAST convictions only,
      so adding a future fold NEVER changes past years' results and there is no look-ahead.
    - else: the global SKIP constant (fixed/off mode)."""
    if SKIP_BY_YEAR is None:
        return SKIP
    y = int(str(entry_date)[:4])
    return SKIP_BY_YEAR.get(y, SKIP_BY_YEAR_DEFAULT)


def _entries(sim, r5map):
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] < _skip_for(t["entry_date"]):
            continue
        v = r5map.get((t["symbol"], t["entry_date"]), np.nan)
        if not np.isnan(v) and v < R5THR:
            continue
        if OSMAP is not None:
            ov = OSMAP.get((t["symbol"], t["entry_date"]))
            if ov is not None and ov > OSTHR:
                continue
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: t["prio"], reverse=True)
    return entries


def prun_metric(sim, pm, cm, r5map, mu, sd, tplus, advance_fee=0.0008):
    globals()["_TPLUS"] = tplus
    for t in sim.trades:  # T+2 min-hold extend
        be, bx = t["i0"], t["i1"]
        sc0 = sim.sym_close[t["symbol"]]
        if tplus and (bx - be) < tplus:
            nb = min(be + tplus, len(sc0) - 1)
            t["i1"] = nb
            t["x_raw"] = sc0[nb]
    _prep(sim, pm, cm, mu, sd)
    entries = _entries(sim, r5map)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar

    def lv(leg, dt):
        s = leg["symbol"]
        j = si[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]
        j = min(max(j, i0), i1)
        r = (
            leg["ratio1"]
            if i1 == i0
            else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        )
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]
        leg["last_val"] = v
        return v

    def mk(t, size, di):
        s = t["symbol"]
        c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]
        xe = t["p0"] * (1.0 + t["net"])
        return dict(
            symbol=s,
            i0=t["i0"],
            i1=t["i1"],
            invested=size,
            net=t["net"],
            p0=t["p0"],
            ratio0=t["p0"] / c0,
            ratio1=xe / c1,
            last_val=size,
            exit_date=t["exit_date"],
            prio=t["prio"],
            be_di=di,
        )

    cash = 1.0
    pend = defaultdict(float)
    legs = []
    exits = defaultdict(list)
    ns = []
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0)
        pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee)
                legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs)
        nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size
                leg = mk(t, size, di)
                legs.append(leg)
                exits[t["exit_date"]].append(leg)
            elif legs:
                cand = [l for l in legs if (di - l["be_di"]) >= tplus]
                if not cand:
                    continue
                c = min(cand, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > MARGIN:
                    vnow = lv(c, dt)
                    cash += vnow * (1.0 - advance_fee)
                    legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size
                        leg = mk(t, size, di)
                        legs.append(leg)
                        exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs)
        ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"])
    d["date"] = pd.to_datetime(d["date"])
    nav = d["nav"]
    fin = float(nav.iloc[-1])
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return fin, fin ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min()), yrs, d


# ================= MAIN =================
base = pd.read_parquet(_BASE_PARQUET)
sig = pd.read_parquet(_SIG_PARQUET)
base["entry_date"] = pd.to_datetime(base["entry_date"])
base["exit_date"] = pd.to_datetime(base["exit_date"])
base["sigd"] = pd.to_datetime(base["entry_signal_date"])
base["ed"] = base["entry_date"].dt.strftime("%Y-%m-%d")
closed = base[base.exit_date.notna()].copy()
syms = sorted(closed.symbol.unique().tolist())
print(f"[replay] base closed trades={len(closed)} syms={len(syms)}", flush=True)

# meta priority
feat = meta_features(syms)
tr_meta = build_tr_df(closed, sig, feat)
pm = meta_preds(tr_meta, tgt="t_pnl")
print(f"[replay] meta preds={len(pm)} (walk-forward 2021-2026)", flush=True)

# rewrite exits
rw = rewrite(closed)

# conv / ret7 / overshoot maps keyed by (symbol, ed=entry_date str)
cm = {(r.symbol, r.ed): CSm.get((r.symbol, str(r.sigd.date())), 0.5) for r in rw.itertuples()}
r5 = {(r.symbol, r.ed): R5.get((r.symbol, str(r.sigd.date())), np.nan) for r in rw.itertuples()}
osm = {}
for r in rw.itertuples():
    di = DIDX.get(r.symbol, {})
    si_ = di.get(str(r.sigd.date()))
    fi = di.get(r.ed)
    if si_ is not None and fi is not None and fi >= si_:
        osm[(r.symbol, r.ed)] = (r.entry_price - LO[r.symbol][si_ : fi + 1].min()) / CLO[r.symbol][
            si_
        ]
globals()["OSTHR"] = float(np.nanpercentile([v for v in osm.values()], 90))
globals()["OSMAP"] = osm
# conviction stats (mu is still used by _prep for the sizing z-score; that z is per-trade
# relative and panel-robust, so it stays on the global mu — only the GATE goes causal below)
cv = [cm.get((r.symbol, r.ed), 0.5) for r in rw.itertuples()]
mu = statistics.mean(cv)
sd_ = statistics.pstdev(cv) or 1.0

# Panel-adaptive conviction GATE (SKIP). cs5_ma50 is a CROSS-SECTIONAL rank, so a wider panel
# COMPRESSES the distribution (conv_mu drifts up) and inflates every conv past a fixed SKIP,
# letting mediocre names crowd winners out of the book. Raise SKIP by the amount the panel
# inflated the mean: SKIP = 0.40 + max(0, conv_mu - MU_REF) * GAIN.
#
# SKIP_MODE:
#   causal (default) — per-year SKIP from PAST convictions only (expanding mean over trades
#                      that entered BEFORE that year). Adding a future fold (e.g. 2027) does
#                      NOT change any earlier year's SKIP, so past backtest results stay
#                      byte-stable and there is NO look-ahead. First year (no history) -> 0.40.
#   fixed          — one SKIP from the WHOLE-period mean (simpler, but adding a fold shifts the
#                      global mean and thus retroactively perturbs past years). env SKIP_MODE=fixed
#   off            — disable, fixed 0.40. env SKIP_MODE=off (or SKIP_GAIN=0)
_skip_gain = float(os.environ.get("SKIP_GAIN", "2.0"))
_skip_mode = os.environ.get("SKIP_MODE", "causal").lower()
_MU_REF = 0.644
if _skip_gain <= 0 or _skip_mode == "off":
    print(f"[replay] OSTHR={OSTHR:.4f} conv_mu={mu:.4f} SKIP={SKIP:.4f} (mode=off)", flush=True)
elif _skip_mode == "fixed":
    SKIP = 0.40 + max(0.0, mu - _MU_REF) * _skip_gain
    print(
        f"[replay] OSTHR={OSTHR:.4f} conv_mu={mu:.4f} SKIP={SKIP:.4f} (mode=fixed gain={_skip_gain})",
        flush=True,
    )
else:  # causal
    _tr_year = [(int(str(r.ed)[:4]), cm.get((r.symbol, r.ed), 0.5)) for r in rw.itertuples()]
    _years = sorted({y for y, _ in _tr_year})
    SKIP_BY_YEAR = {}
    for y in _years:
        past = [c for (yy, c) in _tr_year if yy < y]  # PAST-only: entered before year y
        if len(past) < 30:  # too little history -> neutral gate
            SKIP_BY_YEAR[y] = 0.40
        else:
            mu_past = statistics.mean(past)
            SKIP_BY_YEAR[y] = 0.40 + max(0.0, mu_past - _MU_REF) * _skip_gain
    globals()["SKIP_BY_YEAR"] = SKIP_BY_YEAR
    _rng = f"{min(SKIP_BY_YEAR.values()):.3f}-{max(SKIP_BY_YEAR.values()):.3f}"
    print(
        f"[replay] OSTHR={OSTHR:.4f} conv_mu={mu:.4f} SKIP=causal-per-year {_rng} "
        f"({ {y: round(s, 3) for y, s in SKIP_BY_YEAR.items()} }) gain={_skip_gain}",
        flush=True,
    )

# write rewritten trades CSV for NavSim2 (marks NAV on serving ohlcv.db = Sieu Tin Hieu).
# dates as 'YYYY-MM-DD' so NavSim2 sym_idx lookup + (symbol,ed) keys align with pm/cm 10-char keys.
cvf = SRC / "_prod_rewritten.csv"
_out = rw[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].copy()
_out["entry_date"] = pd.to_datetime(_out["entry_date"]).dt.strftime("%Y-%m-%d")
_out["exit_date"] = pd.to_datetime(_out["exit_date"]).dt.strftime("%Y-%m-%d")
_out.to_csv(cvf, index=False)


def run(tplus):
    return prun_metric(NavSim2(str(cvf), date_lo="2020-01-01"), pm, cm, r5, mu, sd_, tplus)


fin0, cg0, dd0, yrs, nav0 = run(0)
fin2, cg2, dd2, _, nav2 = run(2)
print(f"\n===== {_LABEL} CAGR (single-seed) =====", flush=True)
print(
    f"  T+0: NAV x{fin0:.2f}  CAGR {100 * cg0:.1f}%  DD {100 * dd0:.1f}%  ({yrs:.2f} yrs)",
    flush=True,
)
print(f"  T+2: NAV x{fin2:.2f}  CAGR {100 * cg2:.1f}%  DD {100 * dd2:.1f}%", flush=True)
print("  (registered champion DuckDB reference: T+2 CAGR 132.1% / DD -13.8%)", flush=True)
# per-year NAV growth (T+2)
nav2["y"] = nav2["date"].dt.year
yr = nav2.groupby("y")["nav"].agg(["first", "last"])
yr["ret%"] = 100 * (yr["last"] / yr["first"] - 1)
print("  per-year NAV return (T+2):\n", yr["ret%"].round(1).to_string(), flush=True)
print("PROD_REPLAY_DONE", flush=True)
