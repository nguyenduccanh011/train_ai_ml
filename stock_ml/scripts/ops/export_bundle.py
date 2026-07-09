"""Export a leaderboard template into a self-contained serving bundle.

Re-fits the template's models on ALL data up to a cutoff (single-fit, NOT
walk-forward — see memory ``project_production_bundle_deploy``: production is the
last walk_forward_year fold extended forward, re-exported on a schedule), then
serializes models + resolved config + feature spec into a bundle that a separate
production service loads via stock_ml.src.serving.bundle.load_bundle.

Config source (one required):
  --template-id N         load the resolved ExperimentConfig from the DB
  --config-json PATH      load a pre-dumped config dict (dataclasses.asdict of
                          ExperimentConfig) — use when the DB is unreachable from
                          the host (e.g. asyncpg/Windows); dump it via docker exec.

Usage:
  python stock_ml/scripts/export_bundle.py \
      --config-json /tmp/cfg958.json --symbols-file /tmp/univ958.txt \
      --duckdb market_data/market.duckdb --cutoff 2026-01-01 \
      --retrain-schedule yearly --out bundles/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

import pandas as pd  # noqa: E402

from src.pipeline.experiment import (  # noqa: E402
    ExperimentConfig,
    build_feature_frame,
    train_fold,
)
from src.serving.bundle import write_bundle  # noqa: E402

# Min history bars a serving host must prefetch before a signal date so trailing
# features (sma_200 etc.) and the 252-bar recombine z-window are warm.
_MIN_WARMUP_BARS = 260


def _load_config(args) -> ExperimentConfig:
    if args.config_json:
        raw = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        return ExperimentConfig(**raw)
    if args.template_id is not None:
        import asyncio

        from scripts.run_template import load_template_config

        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        return asyncio.run(load_template_config(args.template_id))
    raise SystemExit("export_bundle: provide --config-json or --template-id")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _resolve_symbols(args) -> list[str]:
    if args.symbols_file:
        text = Path(args.symbols_file).read_text(encoding="utf-8")
        return [s.strip() for s in text.replace(",", "\n").splitlines() if s.strip()]
    if args.symbols:
        return [s.strip() for s in args.symbols.split(",") if s.strip()]
    raise SystemExit("export_bundle: provide --symbols-file or --symbols")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--template-id", type=int, default=None)
    p.add_argument("--config-json", default=None)
    p.add_argument("--symbols", default=None, help="comma-separated symbols")
    p.add_argument("--symbols-file", default=None, help="file with symbols (newline/comma)")
    p.add_argument("--duckdb", default="market_data/market.duckdb")
    p.add_argument("--cutoff", default=None,
                   help="single-fit on data <= cutoff (YYYY-MM-DD). Ignored with "
                        "--replicate-last-fold.")
    p.add_argument("--replicate-last-fold", action="store_true",
                   help="train the EXACT last walk-forward fold (same train window via the "
                        "split config) so the bundle == the backtest's last-fold model.")
    p.add_argument("--pred-history-csv", default=None,
                   help="CSV[symbol,date,score,exit_score] of backtest predictions to embed "
                        "as the z-window seed.")
    p.add_argument("--history-mode", choices=["single", "walk_forward"], default="single",
                   help="single: embed only warmup (< test window) — current model owns the "
                        "test window onward. walk_forward: embed the FULL backtest predictions "
                        "so historical signals replay each fold exactly (== backtest); future "
                        "bars still use this (latest) model.")
    p.add_argument("--retrain-schedule", default="yearly")
    p.add_argument("--out", default="bundles")
    args = p.parse_args()

    if not args.replicate_last_fold and not args.cutoff:
        raise SystemExit("export_bundle: provide --cutoff or --replicate-last-fold")

    cfg = _load_config(args)
    symbols = _resolve_symbols(args)

    print(f"[export] template={cfg.name} strategy={cfg.strategy} "
          f"mode={'replicate-last-fold' if args.replicate_last_fold else 'single-fit'}")

    from src.data.loader import get_loader

    loader = get_loader(args.duckdb)
    available = set(loader.list_symbols())
    requested = [s for s in symbols if s in available]
    missing = sorted(set(symbols) - available)
    if missing:
        print(f"[export] WARNING {len(missing)} symbols not in dataset: {missing[:10]}")
    if not requested:
        raise SystemExit("export_bundle: none of the requested symbols are in the dataset")

    raw = loader.load_many(requested)
    ohlcv = raw[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()
    # Single-fit clips to the cutoff; replicate-fold keeps the full series (the splitter
    # carves the train window itself and the test window extends to the data tail).
    if not args.replicate_last_fold:
        ohlcv = ohlcv[pd.to_datetime(ohlcv["date"]) <= pd.Timestamp(args.cutoff)].copy()
    if ohlcv.empty:
        raise SystemExit("export_bundle: no OHLCV after clipping")
    print(f"[export] {len(ohlcv)} bars, {ohlcv['symbol'].nunique()} symbols, "
          f"through {pd.to_datetime(ohlcv['date']).max().date()}")

    (feat, entry_feat_cols, exit_feat_cols, entry_target_cfg, exit_target_cfg,
     entry2_feat_cols, entry3_feat_cols,
     entry4_feat_cols, entry5_feat_cols, entry6_feat_cols) = build_feature_frame(
        ohlcv, cfg, requested_symbols=requested, data_root=args.duckdb, with_targets=True
    )

    if args.replicate_last_fold:
        # Recreate the backtest's last walk-forward fold and train on ITS train portion,
        # so the bundle models are bit-identical to the model that owned the last test
        # window. cutoff = that fold's test_start.
        from src.data.splitter import YearSplitter

        sc = cfg.split
        splitter = YearSplitter(
            train_years=sc.get("train_years", 4),
            test_years=sc.get("test_years", 1),
            gap_days=sc.get("gap_days", 25),
            first_test_year=sc["first_test_year"],
            last_test_year=sc["last_test_year"],
        )
        data_max = pd.to_datetime(feat["date"]).max()
        splitter.last_test_end = data_max + pd.Timedelta(days=1)
        folds = list(splitter.split(feat))
        if not folds:
            raise SystemExit("export_bundle: splitter produced no folds")
        w_last, train_src, _test_df = folds[-1]
        cutoff_used = pd.Timestamp(w_last.test_start)
        # Drop rows the model cannot train on (un-ripe forward-target tail of a symbol
        # whose data ends mid-window, e.g. a suspension/delisting, + NaN-feature zones) —
        # same fail-loud guard the single-fit path honors. The backtest's effective train
        # set excludes these too (see experiment.py train count via dropna).
        _need = ["target_entry", "target_exit"] + sorted(
            set(entry_feat_cols) | set(exit_feat_cols)
        )
        _n0 = len(train_src)
        train_src = train_src.dropna(subset=[c for c in _need if c in train_src.columns])
        if len(train_src) < _n0:
            print(f"[export] replicate-fold: dropped {_n0 - len(train_src)} un-ripe/NaN train row(s)")
        print(f"[export] last fold: train {pd.to_datetime(train_src['date']).min().date()}.."
              f"{pd.to_datetime(train_src['date']).max().date()} -> test_start {cutoff_used.date()}")
    else:
        # Single-fit on all data up to cutoff; drop rows the model cannot train on
        # (un-ripe label tail + NaN-feature zones) explicitly, since train_fold is fail-loud.
        need_cols = ["target_entry", "target_exit"] + sorted(
            set(entry_feat_cols) | set(exit_feat_cols)
        )
        n0 = len(feat)
        train_src = feat.dropna(subset=[c for c in need_cols if c in feat.columns])
        print(f"[export] fit rows: {len(train_src)}/{n0} (dropped un-ripe/NaN)")
        cutoff_used = pd.Timestamp(args.cutoff)
    if train_src.empty:
        raise SystemExit("export_bundle: no training rows")

    test_stub = train_src.groupby("symbol", group_keys=False).tail(3)
    # ENSEMBLE heads (N-head champions n2_3h_/4h_/5h_*): train_fold fits entry2..entry5 + exit2
    # on their own targets when the corresponding target columns exist (built by
    # build_feature_frame from engine_config.entry_ensemble..entry_ensemble4 / exit_ensemble).
    # Collect the fitted ensemble models via out_models so the bundle carries EVERY head — else
    # serving silently drops the ensemble union buys and live signals diverge from the backtest.
    def _tcol(name: str) -> str | None:
        return name if name in train_src.columns else None

    ensemble_models: dict = {}
    entry_model, exit_model, _ = train_fold(
        train_src,
        test_stub,
        entry_feat_cols,
        cfg,
        exit_feat_cols=exit_feat_cols if exit_feat_cols != entry_feat_cols else None,
        entry_target_col="target_entry",
        exit_target_col="target_exit",
        entry2_target_col=_tcol("target_entry2"),
        entry2_feat_cols=entry2_feat_cols,
        entry3_target_col=_tcol("target_entry3"),
        entry3_feat_cols=entry3_feat_cols,
        entry4_target_col=_tcol("target_entry4"),
        entry4_feat_cols=entry4_feat_cols,
        entry5_target_col=_tcol("target_entry5"),
        entry5_feat_cols=entry5_feat_cols,
        exit2_target_col=_tcol("target_exit2"),
        out_models=ensemble_models,
    )

    models = {"entry": entry_model}
    if exit_model is not None:
        models["exit"] = exit_model
    models.update(ensemble_models)  # entry2 / entry3 / exit2 when present
    print(f"[export] fitted models: {sorted(models)}")

    # --- prediction history (z-window seed / full walk-forward replay) ---
    prediction_history = None
    if args.pred_history_csv:
        ph = pd.read_csv(args.pred_history_csv)
        ph["date"] = pd.to_datetime(ph["date"]).dt.normalize()
        # Carry ensemble prediction columns too when the backtest CSV has them, so the
        # serving z-window for score2/score3 (the union-buy bands) is seeded exactly as in
        # the backtest, not re-predicted by the current model on warmup bars.
        _ph_cols = ["symbol", "date", "score", "exit_score"] + [
            c for c in ph.columns
            if c not in ("symbol", "date", "score", "exit_score")
            and (c.startswith("score") or c.startswith("exit_score"))
        ]
        ph = ph[_ph_cols].copy()
        if args.history_mode == "single":
            # Only the warmup before the test window — the latest model owns the rest.
            ph = ph[ph["date"] < cutoff_used].copy()
            print(f"[export] mode=single: embedding {len(ph)} warmup rows (< {cutoff_used.date()})")
        else:
            # Full backtest predictions — historical signals replay each fold exactly.
            print(f"[export] mode=walk_forward: embedding {len(ph)} full backtest rows "
                  f"(history replays backtest; future uses this model)")
        prediction_history = ph

    # --- serialize bundle ---
    cutoff = cutoff_used
    suffix = "_wf" if args.history_mode == "walk_forward" else ""
    bundle_name = f"bundle_{cfg.name}_{cutoff.date()}{suffix}"
    out_dir = Path(args.out) / bundle_name
    config_dict = dataclasses.asdict(cfg)
    feature_spec = {
        "entry_features": cfg.entry_features or cfg.feature_set,
        "exit_features": cfg.exit_features or cfg.feature_set,
        "entry_feat_cols": list(entry_feat_cols),
        "exit_feat_cols": list(exit_feat_cols),
        "entry_target": entry_target_cfg,
        "exit_target": exit_target_cfg,
    }
    # ENSEMBLE feature columns / targets, generic over head count (N-head champions). None feat
    # cols = head shares the primary entry features. Serving re-resolves these from the same DSL
    # catalog (build_feature_frame); recorded here for provenance + the drift guard. The engine
    # key for entryK is entry_ensemble (k=2) / entry_ensemble{k-1} (k>=3).
    _entry_head_feat = {2: entry2_feat_cols, 3: entry3_feat_cols,
                        4: entry4_feat_cols, 5: entry5_feat_cols}
    for k, fc in _entry_head_feat.items():
        if f"entry{k}" in ensemble_models:
            ens_key = "entry_ensemble" if k == 2 else f"entry_ensemble{k - 1}"
            feature_spec[f"entry{k}_feat_cols"] = list(fc) if fc else None
            feature_spec[f"entry{k}_target"] = (cfg.engine.get(ens_key) or {}).get("target")
    if "exit2" in ensemble_models:
        feature_spec["exit2_target"] = (cfg.engine.get("exit_ensemble") or {}).get("target")
    manifest_extra = {
        "template_name": cfg.name,
        "template_id": args.template_id,
        "strategy": cfg.strategy,
        "cutoff_date": str(cutoff.date()),
        "universe": requested,
        "seed": cfg.seed,
        "min_warmup_bars": _MIN_WARMUP_BARS,
        "retrain_schedule": args.retrain_schedule,
        "history_mode": args.history_mode,
        "git_sha": _git_sha(),
        # The backtest computed CSRank / equal-weight market-index / nonbull gates over the
        # symbols it was FED (= this bundle's `requested` universe). Serving must feed the SAME
        # universe to reproduce those scope-dependent features — i.e. 'traded' scope. (Breadth /
        # xsec / regime are read from the full market.duckdb regardless, so they need no parity.)
        # The serving SignalEngine honours manifest['feature_scope'] (stock-serving CLAUDE.md §11).
        "feature_scope": "traded",
    }
    path = write_bundle(
        out_dir, models=models, config=config_dict,
        feature_spec=feature_spec, manifest_extra=manifest_extra,
        prediction_history=prediction_history,
    )
    print(f"[export] wrote bundle -> {path}")


if __name__ == "__main__":
    main()
