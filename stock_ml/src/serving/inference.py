"""Serving inference: bundle + OHLCV -> signals, via the exact champion chain.

This is the single entry point a production service calls. It reproduces the
research signal chain end-to-end without the DB or training code:
    build_feature_frame(with_targets=False) -> predict_slot_signals -> recombine_signals
"""

from __future__ import annotations

import re

import pandas as pd

from stock_ml.src.pipeline.experiment import (
    ExperimentConfig,
    build_feature_frame,
    predict_slot_signals,
    recombine_signals,
)
from stock_ml.src.serving.bundle import LoadedBundle

_OHLCV_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]
_ENTRY_HEAD_RE = re.compile(r"entry(\d+)$")  # entry2, entry3, ... ensemble heads
_EXIT_HEAD_RE = re.compile(r"exit(\d+)$")    # exit2, ... ensemble exit heads


def generate_signals_from_bundle(bundle: LoadedBundle, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Produce final buy/sell/hold signals for ``ohlcv`` using a loaded bundle.

    Args:
        bundle: result of stock_ml.src.serving.bundle.load_bundle.
        ohlcv: [symbol, date, open, high, low, close, volume] with ENOUGH warmup
               history before the dates of interest (>= manifest['min_warmup_bars'])
               so trailing features and the 252-bar recombine z-window are warm.

    Returns:
        DataFrame [symbol, date, signal, score, ...] where signal ∈ {-1, 0, 1}.

    Raises:
        ValueError: if the resolved feature columns drift from the bundle's spec
                (would silently change signals) or required columns are missing.
    """
    missing = [c for c in _OHLCV_COLS if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"generate_signals_from_bundle: ohlcv missing columns {missing}")

    cfg = ExperimentConfig(**bundle.config)
    (feat, entry_feat_cols, exit_feat_cols, _et, _xt,
     entry2_feat_cols, entry3_feat_cols,
     entry4_feat_cols, entry5_feat_cols, entry6_feat_cols) = build_feature_frame(
        ohlcv[_OHLCV_COLS].copy(),
        cfg,
        requested_symbols=sorted(ohlcv["symbol"].unique()),
        data_root="",
        with_targets=False,
    )
    # Per-head entry feature columns (None = shares the primary entry features). Indexed by
    # the head number k so entryK -> entryK_feat_cols, for any head count the bundle carries.
    _entry_head_feat = {
        2: entry2_feat_cols, 3: entry3_feat_cols,
        4: entry4_feat_cols, 5: entry5_feat_cols, 6: entry6_feat_cols,
    }

    # Guard against silent feature-pipeline drift between export host and serving.
    spec_entry = bundle.feature_spec.get("entry_feat_cols")
    if spec_entry is not None and list(entry_feat_cols) != list(spec_entry):
        raise ValueError(
            "generate_signals_from_bundle: resolved entry feature columns differ from "
            f"the bundle's feature_spec ({len(entry_feat_cols)} vs {len(spec_entry)}) — "
            "feature pipeline drift would silently change signals."
        )

    # Per-bar NaN isolation (bug #2): a handful of old bars have NaN features (e.g.
    # bb_pct_20 in a std==0 zone). Drop ONLY those rows, with a logged count, instead of
    # failing the whole universe — so the full history can be loaded for a warm z-window /
    # downleg leg-state. The model layer (predict_slot_signals) stays fail-loud; we hand it
    # clean rows. Dropped bars carry no signal (the backtest skips them too), so they don't
    # affect agreement. Recent live data is normally clean, making this a no-op there.
    feat_cols_all = sorted(set(entry_feat_cols) | set(exit_feat_cols))
    nan_rows = feat[feat_cols_all].isna().any(axis=1)
    if nan_rows.any():
        n = int(nan_rows.sum())
        syms = feat.loc[nan_rows, "symbol"].nunique()
        print(f"[serving] dropping {n} NaN-feature bar(s) across {syms} symbol(s) "
              f"(std==0 / data gaps) — they carry no signal")
        feat = feat[~nan_rows].copy()
        if feat.empty:
            raise ValueError("generate_signals_from_bundle: all bars NaN after isolation")

    # ENSEMBLE heads (N-head champions n2_3h_/4h_/5h_*): a model set carries entry2..entryN and optional
    # exit2. Build the head lists generically (entryK -> scoreK, exitK -> exit_scoreK) so ANY head count
    # serves identically to the backtest. Then score `frame` with that model set.
    def _score(model_set: dict, frame: pd.DataFrame) -> pd.DataFrame:
        entry_ensemble = [
            (f"score{int(m.group(1))}", model_set[name], _entry_head_feat.get(int(m.group(1))))
            for name in sorted(model_set) if (m := _ENTRY_HEAD_RE.fullmatch(name))
        ]
        exit_ensemble = [
            (f"exit_score{int(m.group(1))}", model_set[name], exit_feat_cols)
            for name in sorted(model_set) if (m := _EXIT_HEAD_RE.fullmatch(name))
        ]
        return predict_slot_signals(
            model_set["entry"], model_set.get("exit"), frame, entry_feat_cols, cfg,
            exit_feat_cols=exit_feat_cols, entry_ensemble=entry_ensemble, exit_ensemble=exit_ensemble,
        )

    if bundle.fold_models:
        # §11.9: replay the walk-forward — each bar scored by the model that OWNED its year (mapped by
        # calendar year, matching how the backtest keys universe_by_year; years beyond the last fold use
        # the last fold-model, before the first use the first). History reproduces the backtest by
        # construction, so NO z-window seed is needed. This is the exact per-fold predict the backtest ran.
        fm = bundle.fold_models
        # Per-fold universe mask: the backtest masks each fold's test to universe_by_year[Y] (a symbol
        # only carries signals for the years it was IN the universe). Serving must mask identically, or a
        # symbol's trailing z-window would include bars from years it wasn't traded — flipping band-edge
        # signals. Features stay computed over the full union (CSRank parity); only the kept symbols differ.
        uby = bundle.manifest.get("universe_by_year") or {}
        years = sorted(fm)
        fold_year = feat["date"].dt.year.clip(years[0], years[-1])
        parts = []
        for y in years:
            sub = feat[fold_year == y]
            allowed = uby.get(str(y), uby.get(y)) if uby else None
            if allowed is not None:
                sub = sub[sub["symbol"].isin(set(allowed))]
            if not sub.empty:
                parts.append(_score(fm[y], sub))
        raw = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
        return recombine_signals(raw, cfg)

    # Legacy single-model bundle: score with the one model, then (optionally) seed the z-window from the
    # exported walk-forward predictions so past bars' trailing z-score matches the backtest (§11.9 deprecates
    # this — a single model cannot reproduce history without the seed; fold-model bundles do it by construction).
    raw = _score(bundle.models, feat)
    ph = bundle.prediction_history
    if ph is not None and not ph.empty:
        ph = ph.copy()
        ph["date"] = pd.to_datetime(ph["date"]).dt.normalize()
        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
        seed_cols = [c for c in ph.columns if c not in ("symbol", "date") and c in raw.columns]
        raw = raw.merge(
            ph[["symbol", "date", *seed_cols]],
            on=["symbol", "date"], how="left", suffixes=("", "_hist"),
        )
        seed = raw["score_hist"].notna()
        for c in seed_cols:
            raw.loc[seed, c] = raw.loc[seed, f"{c}_hist"].astype(raw[c].dtype)
        raw = raw.drop(columns=[f"{c}_hist" for c in seed_cols])

    return recombine_signals(raw, cfg)
