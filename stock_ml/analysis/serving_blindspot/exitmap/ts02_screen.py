# -*- coding: utf-8 -*-
"""TWOSIDED HEAD SCREEN — step 2+3: OOS screen by yearly folds + decision proxy + pain cohorts.

Folds y in 2022..2026: train = in_pos bars whose LABEL WINDOW fully ends before
y-01-01 (purged), test = in_pos bars with date in year y. Three feature sets:
  GATE     = exact force-gate inputs (12)          -> baseline "what gates see"
  GATEPOS  = GATE + position state (5)             -> what gates + snr-defer style rules see
  FULL     = GATEPOS + 19 exit_vol_downpress feats -> the proposed two-sided head
Two targets: rem_mfe (remaining upside), rem_mae (remaining downside), LightGBM.

Outputs: ts_preds.parquet (per-bar OOS preds, incl. post-exit extension bars),
ts_screen_results.json, printed tables.

Metrics:
  (a) pooled Spearman IC per fold + within-trade mean IC (timing view)
  (b) incremental = IC_full - IC_gate(pos); residual-IC of rank(pred_full) after
      rank-OLS on pred_gatepos; AUC on event labels (rem_mae<=-5%, rem_mfe>=+5%)
  (c) null band: 200 within-trade circular shifts of labels (keeps autocorr)
  (d) decision proxy: exit first bar with pred_up < k * max(-pred_dn,0)
      variants: EARLY (only can exit earlier than real) / REPLACE (may also hold
      up to +20 bars past real exit; slot knock-on NOT counted -- caveat).
      Same-trade comparison, pnl convention = close-fill minus implied per-trade
      round-trip cost (matches recorded pnl_pct at the real exit bar exactly).
  (e) cohorts: sold-then-rallied (663) at real decision bar; PDR suppress
      victims (pdr_forensic_scan.csv, delta<0) in the suppress window.
"""
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
RNG = np.random.default_rng(42)
N_NULL = 200

D = pd.read_parquet(EM + "/ts_bars.parquet")
META = pd.read_parquet(EM + "/ts_trades_meta.parquet")
D["date"] = pd.to_datetime(D["date"])
D["label_end_date"] = pd.to_datetime(D["label_end_date"])
D["year"] = D.date.dt.year

FEAT19 = ["atr_14_ratio", "realized_vol_10", "vol_percentile_60", "bb_width_20",
          "volatility_rank", "high_low_pct_5d", "ma5_accel", "dist_63d_high",
          "dist_52w_high", "sma_20_ratio", "bb_pct_20", "market_volatility_regime",
          "market_trend", "momentum_rank", "dist_day_25", "dist_day_vol20_25",
          "down_vol_intensity_5", "down_vol_count_10", "updown_vol_20"]
GATE = ["leg12", "leg6", "bma20p2", "nonbull_mkt", "breadth_ma50", "lowbreadth_mkt",
        "drop5_z", "mkt_drop", "snr20", "f_dl12", "f_nb", "f_lb"]
POS = ["gain", "peak", "giveback", "age", "bars_since_peak"]
SETS = {"gate": GATE, "gatepos": GATE + POS, "full": GATE + POS + FEAT19}
TGTS = ["rem_mfe", "rem_mae"]
FOLDS = [2022, 2023, 2024, 2025, 2026]

PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
              min_child_samples=60, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=42, verbose=-1)


def ic(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 30:
        return np.nan
    return float(spearmanr(a[m], b[m]).statistic)


def within_trade_ic(df, pcol, tcol, min_bars=8):
    vals = []
    for _, g in df.groupby("trade_id"):
        if len(g) < min_bars or g[tcol].isna().any():
            continue
        if g[tcol].std() == 0 or g[pcol].std() == 0:
            continue
        vals.append(spearmanr(g[pcol], g[tcol]).statistic)
    return (float(np.nanmean(vals)), len(vals)) if vals else (np.nan, 0)


# ------------------------------------------------------------------ train folds
ip = D[D.in_pos & D.rem_mfe.notna()].copy()
print(f"in_pos labeled bars: {len(ip)}, trades: {ip.trade_id.nunique()}")

pred_frames = []
res = {}
for y in FOLDS:
    tr_mask = ip.label_end_date < f"{y}-01-01"
    te_mask = ip.year == y
    trn, tst = ip[tr_mask], ip[te_mask]
    if len(tst) < 100:
        print(f"fold {y}: test too small ({len(tst)}), skip"); continue
    # score ALL bars of year y (incl. post-exit extension) for the decision sim
    allb = D[(D.year == y)].copy()
    fold = {"n_train": len(trn), "n_test": len(tst),
            "n_train_trades": int(trn.trade_id.nunique()),
            "n_test_trades": int(tst.trade_id.nunique())}
    preds_fold = allb[["trade_id", "symbol", "date", "in_pos", "age",
                       "bars_vs_exit", "rem_mfe", "rem_mae"]].copy()
    for sname, cols in SETS.items():
        for tgt in TGTS:
            m = LGBMRegressor(**PARAMS)
            m.fit(trn[cols], trn[tgt])
            preds_fold[f"p_{tgt}_{sname}"] = m.predict(allb[cols])
    preds_fold["fold"] = y
    pred_frames.append(preds_fold)

    tp = preds_fold[preds_fold.in_pos & preds_fold.rem_mfe.notna()]
    for tgt in TGTS:
        yv = tp[tgt].to_numpy()
        row = {}
        for sname in SETS:
            p = tp[f"p_{tgt}_{sname}"].to_numpy()
            row[f"ic_{sname}"] = round(ic(p, yv), 4)
        # residual IC: rank(pred_full) residualized on rank(pred_gatepos)
        rf = pd.Series(tp[f"p_{tgt}_full"]).rank(pct=True).to_numpy()
        rg = pd.Series(tp[f"p_{tgt}_gatepos"]).rank(pct=True).to_numpy()
        B = np.column_stack([np.ones(len(rg)), rg])
        beta, *_ = np.linalg.lstsq(B, rf, rcond=None)
        row["ic_resid_full_on_gatepos"] = round(ic(rf - B @ beta, yv), 4)
        # event AUC
        ev = (yv <= -0.05) if tgt == "rem_mae" else (yv >= 0.05)
        if 0 < ev.sum() < len(ev):
            for sname in SETS:
                p = tp[f"p_{tgt}_{sname}"].to_numpy()
                sgn = -1.0 if tgt == "rem_mae" else 1.0
                row[f"auc_{sname}"] = round(float(roc_auc_score(ev, sgn * p)), 4)
        # within-trade timing IC
        wt, nw = within_trade_ic(tp, f"p_{tgt}_full", tgt)
        row["wt_ic_full"], row["wt_n"] = round(wt, 4), nw
        wtg, _ = within_trade_ic(tp, f"p_{tgt}_gatepos", tgt)
        row["wt_ic_gatepos"] = round(wtg, 4)
        # null band: within-trade circular shift of labels vs full pred
        pfull = tp[f"p_{tgt}_full"].to_numpy()
        gidx = tp.groupby("trade_id").indices
        nulls = []
        for _ in range(N_NULL):
            ys = yv.copy()
            for _, idx in gidx.items():
                if len(idx) > 2:
                    ys[idx] = np.roll(yv[idx], RNG.integers(1, len(idx)))
            nulls.append(ic(pfull, ys))
        row["null_lo"], row["null_hi"] = (round(float(np.percentile(nulls, 2.5)), 4),
                                          round(float(np.percentile(nulls, 97.5)), 4))
        fold[tgt] = row
    res[y] = fold
    print(f"fold {y}: train {len(trn)} test {len(tst)} | "
          f"mfe ic g/gp/f {fold['rem_mfe']['ic_gate']}/{fold['rem_mfe']['ic_gatepos']}"
          f"/{fold['rem_mfe']['ic_full']} resid {fold['rem_mfe']['ic_resid_full_on_gatepos']} "
          f"null ({fold['rem_mfe']['null_lo']},{fold['rem_mfe']['null_hi']}) | "
          f"mae ic g/gp/f {fold['rem_mae']['ic_gate']}/{fold['rem_mae']['ic_gatepos']}"
          f"/{fold['rem_mae']['ic_full']} resid {fold['rem_mae']['ic_resid_full_on_gatepos']} "
          f"null ({fold['rem_mae']['null_lo']},{fold['rem_mae']['null_hi']})", flush=True)

P = pd.concat(pred_frames, ignore_index=True)
P.to_parquet(EM + "/ts_preds.parquet", index=False)
print("saved ts_preds.parquet:", len(P))

# ------------------------------------------------------------------ decision proxy
# per-trade implied cost so close-fill pnl at the REAL exit == recorded pnl_pct
META2 = META.copy()
META2["entry_date"] = pd.to_datetime(META2["entry_date"])
META2["exit_date"] = pd.to_datetime(META2["exit_date"])

bars_all = D[["trade_id", "date", "bars_vs_exit", "in_pos"]].copy()
# close price series per (trade, bar) - recover from gain: close = ep*(1+gain)
bars_all["close"] = None  # rebuilt below from D.gain
D2 = D.merge(META2[["trade_id", "entry_price", "exit_price", "pnl_pct",
                    "exit_reason", "year_exit"]], on="trade_id", how="left")
D2["close_px"] = D2.entry_price * (1.0 + D2.gain)
D2 = D2.merge(P[["trade_id", "date"] + [c for c in P.columns if c.startswith("p_")]],
              on=["trade_id", "date"], how="left")

KGRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
sim_rows = []
elig = META2[(META2.exit_reason != "open") & (META2.entry_date >= "2022-01-01")]
print(f"decision-proxy eligible trades (entry>=2022, closed): {len(elig)}")

trade_bars = {tid: g.sort_values("date").reset_index(drop=True)
              for tid, g in D2[D2.trade_id.isin(elig.trade_id)].groupby("trade_id")}

for sname in ["gatepos", "full"]:
    for k in KGRID:
        for variant in ["early", "replace"]:
            for tid, t in elig.set_index("trade_id").iterrows():
                g = trade_bars.get(tid)
                if g is None or len(g) < 2:
                    continue
                up = g[f"p_rem_mfe_{sname}"].to_numpy()
                dn = np.maximum(-g[f"p_rem_mae_{sname}"].to_numpy(), 0.0)
                fire = up < k * dn
                fire &= ~np.isnan(up)
                scan = g.in_pos.to_numpy() if variant == "early" else np.ones(len(g), bool)
                cand = np.where(fire & scan)[0]
                # real decision bar index within g = bars_vs_exit == -1
                dec_real = np.where(g.bars_vs_exit.to_numpy() == -1)[0]
                dec_real = dec_real[0] if len(dec_real) else len(g) - 1
                if len(cand):
                    jdec = int(cand[0]) if variant == "replace" else min(int(cand[0]), dec_real)
                else:
                    jdec = dec_real if variant == "early" else len(g) - 2  # hold to ext end
                jfill = min(jdec + 1, len(g) - 1)
                fill_px = g.close_px.iloc[jfill]
                cost = (g.close_px.iloc[dec_real + 1] if dec_real + 1 < len(g)
                        else t.exit_price) / t.entry_price - 1.0 - t.pnl_pct
                pnl_sim = fill_px / t.entry_price - 1.0 - cost
                hold_sim = int(g.age.iloc[jfill])
                sim_rows.append(dict(set=sname, k=k, variant=variant, trade_id=tid,
                                     year_exit=int(t.year_exit), pnl_act=t.pnl_pct,
                                     hold_act=int(g.age.iloc[min(dec_real + 1, len(g) - 1)]),
                                     pnl_sim=pnl_sim, hold_sim=hold_sim,
                                     fired=bool(len(cand)),
                                     moved=jdec != dec_real))
SIM = pd.DataFrame(sim_rows)
SIM.to_parquet(EM + "/ts_sim.parquet", index=False)

def sim_table(df, yr_min):
    s = df[df.year_exit >= yr_min]
    out = []
    for (sname, k, variant), g in s.groupby(["set", "k", "variant"]):
        out.append(dict(set=sname, k=k, variant=variant, n=len(g),
                        pnl_act=round(g.pnl_act.sum(), 2), pnl_sim=round(g.pnl_sim.sum(), 2),
                        d_pnl=round(g.pnl_sim.sum() - g.pnl_act.sum(), 2),
                        psd_act=round(g.pnl_act.sum() / g.hold_act.sum(), 5),
                        psd_sim=round(g.pnl_sim.sum() / g.hold_sim.sum(), 5),
                        moved=int(g.moved.sum())))
    return pd.DataFrame(out)

T22 = sim_table(SIM, 2022)
T24 = sim_table(SIM, 2024)
print("\n=== decision proxy, exit-year >= 2022 ===")
print(T22.to_string(index=False))
print("\n=== decision proxy, exit-year >= 2024 ===")
print(T24.to_string(index=False))

# ------------------------------------------------------------------ cohorts
dec = D2[D2.bars_vs_exit == -1].merge(META2[["trade_id", "rallied", "label"]],
                                      on="trade_id", how="left")
dec = dec[dec.date >= "2022-01-01"].dropna(subset=["p_rem_mfe_full"])
coh = {}
for sname in ["gatepos", "full"]:
    margin = dec[f"p_rem_mfe_{sname}"] + dec[f"p_rem_mae_{sname}"]  # up - dnmag
    hold_sig = dec[f"p_rem_mfe_{sname}"] > np.maximum(-dec[f"p_rem_mae_{sname}"], 0)
    r = dec.rallied.astype(bool)
    coh[sname] = {
        "n_dec": len(dec), "n_rallied": int(r.sum()),
        "p_hold_given_rallied": round(float(hold_sig[r].mean()), 4),
        "p_hold_given_not": round(float(hold_sig[~r].mean()), 4),
        "auc_margin_vs_rallied": round(float(roc_auc_score(r, margin)), 4),
    }
print("\n=== cohort sold-then-rallied (decision bars >=2022) ===")
print(json.dumps(coh, indent=1))

# PDR suppress victims
pdr = pd.read_csv(EM + "/pdr_forensic_scan.csv",
                  parse_dates=["entry_date", "exit_date", "sw_date"])
key = META2[["trade_id", "symbol", "entry_date", "exit_date"]]
pdr = pdr.merge(key, on=["symbol", "entry_date", "exit_date"], how="inner")
pdr = pdr[pdr.sw_date >= "2022-01-01"]
pdr["victim"] = pdr.delta < 0  # actual worse than selling at suppressed bar
pdr_res = {}
for sname in ["gatepos", "full"]:
    rates = []
    for _, r in pdr.iterrows():
        g = trade_bars.get(r.trade_id)
        if g is None:
            g = D2[D2.trade_id == r.trade_id].sort_values("date").reset_index(drop=True)
        w = g[(g.date >= r.sw_date) & g.in_pos].head(5)
        if not len(w) or w[f"p_rem_mfe_{sname}"].isna().all():
            rates.append(np.nan); continue
        fire = (w[f"p_rem_mfe_{sname}"]
                < np.maximum(-w[f"p_rem_mae_{sname}"], 0)).any()
        rates.append(float(fire))
    pdr["fire_" + sname] = rates
    sub = pdr.dropna(subset=["fire_" + sname])
    v, nv = sub[sub.victim], sub[~sub.victim]
    pdr_res[sname] = {
        "n": len(sub), "n_victim": len(v),
        "p_fire_victim": round(float(v["fire_" + sname].mean()), 4) if len(v) else None,
        "p_fire_nonvictim": round(float(nv["fire_" + sname].mean()), 4) if len(nv) else None,
        "n_victim_5pct": int((sub.delta <= -0.05).sum()),
        "p_fire_victim_5pct": round(float(sub[sub.delta <= -0.05]["fire_" + sname].mean()), 4)
        if (sub.delta <= -0.05).any() else None,
    }
print("\n=== cohort PDR suppress-victims (sw_date >=2022, fire within 5 bars) ===")
print(json.dumps(pdr_res, indent=1))

out = {"folds": res,
       "decision_proxy_ge2022": T22.to_dict("records"),
       "decision_proxy_ge2024": T24.to_dict("records"),
       "cohort_rallied": coh, "cohort_pdr": pdr_res}
with open(EM + "/ts_screen_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print("\nsaved ts_screen_results.json")
