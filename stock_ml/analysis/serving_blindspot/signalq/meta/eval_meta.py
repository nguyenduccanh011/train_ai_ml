"""Walk-forward evaluation of learned conv-combo scorers vs the hand-crafted champion combo.

Folds: train = entry_year < Y, test = entry_year == Y, for Y in 2021..2026.
Scorers:
  hand   = champion strength 0.5*price_combo + 0.5*sigmoid(zE)   (reconstructed, no fit)
  ridge  = RidgeCV on the 5 head z's (zE, z2..z5)                 (linear, learned)
  ridge+ = RidgeCV on z's + engine price components               (linear, learned)
  cfg    = non-negative weights over the CONFIG-EXPRESSIBLE basis
           [rsi_s, ext_s, eff_s, rpos, sig(zE), sig(z4), sig(z3)] (sum-to-1; maps 1:1 to
           entry_pullback_conv_combo_w + head_w/mfe_w/score3_w)
  lgbm   = shallow LightGBM on z's + price components + context
Metrics per fold: Spearman IC vs pnl_pct; precision@top-quintile for big_win;
bottom-quintile separation (mean pnl of bottom 20% by score minus mean of the rest —
the combo's real job: which trades deserve the DEEPER fill).
Nulls: shuffled-y refits (ridge & lgbm), 20 permutations per fold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

HERE = Path(__file__).resolve().parent
META = HERE / "meta_dataset.csv"

Z_COLS = ["zE", "z2", "z3", "z4", "z5"]
PRICE_COLS = ["rsi_s", "ext_s", "eff_s", "rpos"]
CTX_COLS = ["ret5", "ret20", "dist_ma20", "atr_ratio", "snr20", "breadth", "uni_snr20"]
LGB_COLS = Z_COLS + ["zX"] + PRICE_COLS + CTX_COLS
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
RNG = np.random.default_rng(42)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.nan_to_num(x, nan=0.0)))


def cfg_basis(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df.rsi_s, df.ext_s, df.eff_s, df.rpos,
        _sig(df.zE), _sig(df.z4), _sig(df.z3),
    ])


def fit_ridge(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0])
    m.fit((Xtr - mu) / sd, ytr)
    return m.predict((Xte - mu) / sd), m


def fit_cfg(Xtr, ytr, Xte):
    # NNLS on centered y over the sum-to-1 basis; add intercept via centering the basis
    w, _ = nnls(Xtr - Xtr.mean(0), ytr - ytr.mean())
    if w.sum() <= 1e-12:
        w = np.ones(Xtr.shape[1])
    w = w / w.sum()
    return Xte @ w, w


def fit_lgbm(Xtr, ytr, Xte, seed=42):
    m = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=3, num_leaves=7,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, verbose=-1)
    m.fit(Xtr, ytr)
    return m.predict(Xte), m


def metrics(score, te: pd.DataFrame) -> dict:
    ic = spearmanr(score, te.pnl_pct).statistic
    q = np.quantile(score, [0.2, 0.8])
    top = te[score >= q[1]]
    bot = te[score <= q[0]]
    rest = te[score > q[0]]
    return {
        "ic": ic,
        "p_top_bigwin": top.big_win.mean(),
        "bot_pnl": bot.pnl_pct.mean(),
        "rest_pnl": rest.pnl_pct.mean(),
        "bot_sep": bot.pnl_pct.mean() - rest.pnl_pct.mean(),
    }


def main() -> None:
    df = pd.read_csv(META, parse_dates=["entry_date", "signal_date"])
    rows, null_rows, cfg_ws = [], [], []
    for yr in YEARS:
        tr, te = df[df.entry_year < yr], df[df.entry_year == yr]
        if len(te) < 30:
            continue
        ytr = tr.pnl_pct.to_numpy()
        scores = {"hand": te.champ_strength.to_numpy()}

        s, _ = fit_ridge(tr[Z_COLS].to_numpy(), ytr, te[Z_COLS].to_numpy())
        scores["ridge_z"] = s
        s, _ = fit_ridge(tr[Z_COLS + PRICE_COLS].to_numpy(), ytr,
                         te[Z_COLS + PRICE_COLS].to_numpy())
        scores["ridge_zp"] = s
        s, w = fit_cfg(cfg_basis(tr), ytr, cfg_basis(te))
        scores["cfg_w"] = s
        cfg_ws.append({"year": yr, **dict(zip(
            ["w_rsi", "w_ext", "w_eff", "w_rpos", "w_zE", "w_z4", "w_z3"], w.round(4)))})
        if HAS_LGB:
            s, _ = fit_lgbm(tr[LGB_COLS].to_numpy(), ytr, te[LGB_COLS].to_numpy())
            scores["lgbm"] = s

        for name, s in scores.items():
            rows.append({"year": yr, "model": name, "n_tr": len(tr), "n_te": len(te),
                         "big_base": te.big_win.mean(), **metrics(s, te)})

        # shuffled-y nulls (learned models only)
        for k in range(20):
            yp = RNG.permutation(ytr)
            s, _ = fit_ridge(tr[Z_COLS].to_numpy(), yp, te[Z_COLS].to_numpy())
            null_rows.append({"year": yr, "model": "ridge_z", "perm": k,
                              "ic": spearmanr(s, te.pnl_pct).statistic})
            if HAS_LGB:
                s, _ = fit_lgbm(tr[LGB_COLS].to_numpy(), yp, te[LGB_COLS].to_numpy(),
                                seed=100 + k)
                null_rows.append({"year": yr, "model": "lgbm", "perm": k,
                                  "ic": spearmanr(s, te.pnl_pct).statistic})

    res = pd.DataFrame(rows)
    res.to_csv(HERE / "wf_results.csv", index=False)
    pd.DataFrame(cfg_ws).to_csv(HERE / "cfg_weights_by_fold.csv", index=False)

    print("=== OOS Spearman IC by fold ===")
    print(res.pivot(index="year", columns="model", values="ic").round(4).to_string())
    print("\n=== mean IC across folds (2021-2026, n-weighted) ===")
    for mname, g in res.groupby("model"):
        print(f"  {mname:9s} mean_ic={np.average(g.ic, weights=g.n_te):+.4f} "
              f"(unweighted {g.ic.mean():+.4f}, sign+ {int((g.ic > 0).sum())}/{len(g)})")
    print("\n=== precision@top-quintile (big_win) ===")
    piv = res.pivot(index="year", columns="model", values="p_top_bigwin").round(3)
    piv["base_rate"] = res.groupby("year").big_base.first().round(3)
    print(piv.to_string())
    print("\n=== bottom-quintile separation (bot mean pnl - rest mean pnl) ===")
    print(res.pivot(index="year", columns="model", values="bot_sep").round(4).to_string())
    print("\n=== bottom-quintile mean pnl (lower = better deep-fill targeting) ===")
    print(res.pivot(index="year", columns="model", values="bot_pnl").round(4).to_string())

    if null_rows:
        nl = pd.DataFrame(null_rows)
        print("\n=== shuffled-y null OOS IC (20 perms x fold) ===")
        for mname, g in nl.groupby("model"):
            print(f"  {mname:8s} mean={g.ic.mean():+.4f} sd={g.ic.std():.4f} "
                  f"p95=|{g.ic.abs().quantile(0.95):.4f}|")

    print("\n=== learned cfg weights per fold (config-expressible) ===")
    print(pd.DataFrame(cfg_ws).to_string(index=False))

    # final all-data cfg weights (for a potential mt_ clone)
    _, w = fit_cfg(cfg_basis(df), df.pnl_pct.to_numpy(), cfg_basis(df))
    print("\nfinal cfg weights (all data):",
          dict(zip(["w_rsi", "w_ext", "w_eff", "w_rpos", "w_zE", "w_z4", "w_z3"],
                   w.round(4))))


if __name__ == "__main__":
    main()
