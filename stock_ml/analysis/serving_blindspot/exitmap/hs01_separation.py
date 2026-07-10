"""DEEP STOP separation analysis (Buoc 1) — gb_x08 seed-42, config-only hard_stop probe design.

Engine facts (engine.py, verified):
  - hard_stop rule: fires khi "hard_stop" in exit_priority AND hard_stop_pct is not None
    AND hold_bars >= min_hold_bars(=2) AND lows[i]/entry_fill - 1 <= hard_stop_pct (DAU AM).
  - Fill: exit_idx = i+1, close_next -> fill_sell(closes[i+1]); net = gross - round_trip_cost.
  - costs 2783: commission .0015 x2 + tax .001 = round_trip .004; slippage .0015 in fill.
First-order sim per trade: first eligible touch bar i in [ei+min_hold, xi-1] with
low[i]/ep-1 <= X -> new exit fill close[i+1]*(1-slip); dpnl = new_net - actual pnl_pct.
Slot-free effect = 0 (conservative).
"""
import duckdb
import numpy as np
import pandas as pd

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
SLIP = 0.0015
RT = 0.0015 * 2 + 0.001
MIN_HOLD = 2

tr = pd.read_csv(EM + "/gbx08_enriched2.csv", parse_dates=["entry_date", "exit_date"])
print("trades:", len(tr), "sum pnl:", round(tr.pnl_pct.sum(), 2))
print("exit_reason counts:", tr.exit_reason.value_counts().to_dict())

syms = sorted(tr.symbol.unique())
duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, open, high, low, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) "
    "ORDER BY symbol, date".format(",".join(f"'{s}'" for s in syms))).df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])
SYM = {s: dict(dates=pd.DatetimeIndex(g["date"]), low=g["low"].to_numpy(float),
               close=g["close"].to_numpy(float))
       for s, g in bars.groupby("symbol")}

LEVELS = [-0.10, -0.12, -0.14, -0.15, -0.16, -0.18, -0.20]

rows = []
parity_err = []
for _, t in tr.iterrows():
    s = SYM[t.symbol]
    di = s["dates"]
    ei = di.get_indexer([t.entry_date])[0]
    xi = di.get_indexer([t.exit_date])[0]
    ep = t.entry_price
    if ei < 0 or xi < 0:
        rows.append(None); continue
    # parity: recompute actual pnl for close-fill exits
    if t.exit_reason in ("signal", "hard_stop", "max_hold", "trailing_stop"):
        recon = s["close"][xi] * (1 - SLIP) / ep - 1 - RT
        parity_err.append(abs(recon - t.pnl_pct))
    lo = s["low"][ei:xi + 1]
    dd = lo / ep - 1.0                      # low-based drawdown path
    ages = np.arange(len(dd))
    # trigger-eligible bars: age >= MIN_HOLD and i <= xi-1 (decision bar of actual exit at latest)
    elig = (ages >= MIN_HOLD) & (ages <= (xi - ei - 1)) if xi > ei else np.zeros(len(dd), bool)
    rec = dict(idx=t.name, pnl=t.pnl_pct, mae=dd.min(), sym=t.symbol,
               entry=t.entry_date, exit=t.exit_date, hold=xi - ei,
               year=t.entry_date.year, reason=t.exit_reason)
    for X in LEVELS:
        hit = np.where(elig & (dd <= X))[0]
        key = f"{int(round(-X*100))}"
        if len(hit):
            i_rel = int(hit[0])
            fill_i = min(ei + i_rel + 1, len(s["close"]) - 1)
            new_pnl = s["close"][fill_i] * (1 - SLIP) / ep - 1 - RT
            rec["t" + key] = i_rel                     # age at first touch
            rec["p" + key] = new_pnl                   # simulated stop pnl
        else:
            rec["t" + key] = np.nan
            rec["p" + key] = np.nan
    rows.append(rec)

df = pd.DataFrame([r for r in rows if r is not None])
print("parity close-fill exits: n=%d max_err=%.6f" % (len(parity_err), max(parity_err)))

tail = df[df.pnl <= -0.15]
win = df[df.pnl > 0]
mid = df[(df.pnl <= 0) & (df.pnl > -0.15)]
print("\n=== cohorts: tail(<=-15%%)=%d sum=%.2f | winners=%d sum=%.2f | mid-losers=%d sum=%.2f"
      % (len(tail), tail.pnl.sum(), len(win), win.pnl.sum(), len(mid), mid.pnl.sum()))
print("\nTAIL trades:")
print(tail[["sym", "entry", "exit", "hold", "pnl", "mae", "reason",
            "t15", "t18", "t20"]].to_string(index=False))

print("\n=== MAE distribution (low-based, full trade) ===")
for name, g in [("tail", tail), ("winners", win), ("mid-losers", mid)]:
    q = g.mae.quantile([0.01, 0.05, 0.25, 0.5]).round(3).to_dict()
    print(f"{name:10s} n={len(g):4d} min={g.mae.min():.3f} q={q}")

print("\n=== winners/mid that TOUCH level (trigger-eligible bars only) ===")
print(f"{'X':>5} | {'winN':>4} {'winPnL':>7} {'megaN(>=.3)':>11} {'megaPnL':>8} {'maxWpnl':>8} | "
      f"{'midN':>4} {'midPnL':>7} | {'tailN':>5} {'tailPnL':>8}")
for X in LEVELS:
    key = f"{int(round(-X*100))}"
    wt = win[win["t" + key].notna()]
    mt = mid[mid["t" + key].notna()]
    tt = tail[tail["t" + key].notna()]
    mega = wt[wt.pnl >= 0.30]
    print(f"{X:5.2f} | {len(wt):4d} {wt.pnl.sum():7.2f} {len(mega):11d} {mega.pnl.sum():8.2f} "
          f"{(wt.pnl.max() if len(wt) else float('nan')):8.2f} | {len(mt):4d} {mt.pnl.sum():7.2f} | "
          f"{len(tt):5d} {tt.pnl.sum():8.2f}")

print("\n=== age at first touch (bars from entry) ===")
for X in [-0.12, -0.15, -0.18]:
    key = f"{int(round(-X*100))}"
    for name, g in [("tail", tail), ("winners", win)]:
        t_ = g["t" + key].dropna()
        if len(t_):
            print(f"X={X} {name:8s} n={len(t_):3d} ages: min={t_.min():.0f} med={t_.median():.0f} "
                  f"max={t_.max():.0f} list={sorted(t_.astype(int).tolist())[:25]}")

print("\n=== first-order dpnl per level X (stop fill next-bar close, slot effect = 0) ===")
print(f"{'X':>5} | {'nTrig':>5} | {'dTail':>7} {'dWin':>7} {'dMid':>7} | {'dTotal':>7} | "
      f"{'newSum':>8} | worst single dpnl")
for X in LEVELS:
    key = f"{int(round(-X*100))}"
    trig = df[df["p" + key].notna()].copy()
    trig["dpnl"] = trig["p" + key] - trig.pnl
    d_tail = trig.loc[trig.pnl <= -0.15, "dpnl"].sum()
    d_win = trig.loc[trig.pnl > 0, "dpnl"].sum()
    d_mid = trig.loc[(trig.pnl <= 0) & (trig.pnl > -0.15), "dpnl"].sum()
    tot = trig.dpnl.sum()
    worst = trig.nsmallest(3, "dpnl")[["sym", "pnl", "dpnl"]].round(3).values.tolist()
    print(f"{X:5.2f} | {len(trig):5d} | {d_tail:7.3f} {d_win:7.3f} {d_mid:7.3f} | {tot:7.3f} | "
          f"{df.pnl.sum() + tot:8.2f} | {worst}")

# per-year breakdown of dpnl for the most promising levels
print("\n=== dpnl by entry-year (levels -15/-18/-20) ===")
for X in [-0.15, -0.18, -0.20]:
    key = f"{int(round(-X*100))}"
    trig = df[df["p" + key].notna()].copy()
    trig["dpnl"] = trig["p" + key] - trig.pnl
    print(f"X={X}:", trig.groupby("year").dpnl.sum().round(3).to_dict())

df.to_csv(EM + "/hs01_touch_sim.csv", index=False)
print("\nsaved hs01_touch_sim.csv")
