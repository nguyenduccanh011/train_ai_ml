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

import pandas as pd  # noqa: E402

from stock_ml.src.pipeline.experiment import (  # noqa: E402
    ExperimentConfig,
    build_feature_frame,
    train_fold,
)
from stock_ml.src.serving.bundle import write_bundle  # noqa: E402

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


def _write_parity(bundle_path: Path, *, as_of: str) -> None:
    """Emit ``parity.json`` — the attestation the serving-side ``deploy.py`` gate reads (§11.5.2/R7).

    Measured 2026-07-31: 0/6 live bundles carried this file, so ``gate_parity`` always fell back to
    ``--force`` — a formality. Export now always ships it. ``status`` is PENDING at export: only the
    two-way signal+trade equivalence run (§11.5.3, in-container) may flip it to PASS. It records the
    generating wheel so deploy can refuse a wheel-mismatch (the measuring-stick pin, §11.5.2).
    """
    import hashlib

    from stock_ml.src.serving.resolved import _catalog_fingerprint, _wheel_version

    def _sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    fp = hashlib.sha256(
        "".join(
            _sha(bundle_path / f)
            for f in ("manifest.json", "config.json", "feature_spec.json")
            if (bundle_path / f).is_file()
        ).encode()
    ).hexdigest()[:12]
    parity = {
        "schema": "parity.json/1",
        "status": "PENDING",
        "wheel": _wheel_version(),
        "catalog_fingerprint": _catalog_fingerprint(),
        "bundle_fingerprint": fp,
        "as_of": as_of,
        "generated_by": "export_bundle",
        "note": "status flips to PASS only after the two-way signal+trade equivalence attestation "
        "(ENGINE_UPGRADE §11.5.3); until then deploy refuses it without --force.",
    }
    (bundle_path / "parity.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[export] wrote parity.json (status=PENDING, wheel={parity['wheel']}, fp={fp})")


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
    p.add_argument(
        "--cutoff",
        default=None,
        help="single-fit on data <= cutoff (YYYY-MM-DD). Ignored with --replicate-last-fold.",
    )
    p.add_argument(
        "--replicate-last-fold",
        action="store_true",
        help="train the EXACT last walk-forward fold (same train window via the "
        "split config) so the bundle == the backtest's last-fold model.",
    )
    p.add_argument(
        "--fold-models",
        action="store_true",
        help="§11.9: train EVERY walk-forward fold and ship each model, so serving re-runs "
        "the walk-forward (bar year Y scored by model-fold-Y) — history reproduces the "
        "backtest by construction, no z-window seed. Implies replicate-last-fold semantics.",
    )
    p.add_argument(
        "--fold-models-from-run",
        default=None,
        help="§11.9 robust: load the EXACT per-fold models the BACKTEST persisted (run with "
        "STOCKML_PERSIST_FOLD_MODELS=1) from that run's folds dir "
        "(<out_dir>/<run_id>/folds, holding *.models.joblib) instead of re-training. Ships "
        "the verbatim model that produced run_signals → serving == backtest exactly "
        "(attest 100%). Implies replicate-last-fold semantics.",
    )
    p.add_argument(
        "--pred-history-csv",
        default=None,
        help="CSV[symbol,date,score,exit_score] of backtest predictions to embed "
        "as the z-window seed.",
    )
    p.add_argument(
        "--history-mode",
        choices=["single", "walk_forward"],
        default="single",
        help="single: embed only warmup (< test window) — current model owns the "
        "test window onward. walk_forward: embed the FULL backtest predictions "
        "so historical signals replay each fold exactly (== backtest); future "
        "bars still use this (latest) model.",
    )
    p.add_argument("--retrain-schedule", default="yearly")
    p.add_argument("--out", default="bundles")
    args = p.parse_args()

    if (
        not args.replicate_last_fold
        and not args.fold_models
        and not args.fold_models_from_run
        and not args.cutoff
    ):
        raise SystemExit(
            "export_bundle: provide --cutoff, --replicate-last-fold, "
            "--fold-models, or --fold-models-from-run"
        )

    cfg = _load_config(args)

    # Dynamic point-in-time universe (dyn_topn universe_slug -> cfg.universe_policy):
    # resolve per-fold universes EXACTLY like the experiment runner (feature build on the
    # union of fold lists, the splitter masks each fold to its year), plus the CURRENT
    # year's list which becomes the bundle's serving universe in the manifest.
    universe_by_year: dict[int, list[str]] | None = None
    serve_universe: list[str] | None = None
    serve_year: int | None = None
    if getattr(cfg, "universe_policy", None):
        from stock_ml.src.data.universe_resolver import resolve_universes

        _sp = cfg.split
        fold_years = list(
            range(_sp["first_test_year"], _sp["last_test_year"] + 1, _sp.get("test_years", 1))
        )
        serve_year = pd.Timestamp.today().year
        all_years = fold_years + ([serve_year] if serve_year not in fold_years else [])
        _resolved = resolve_universes(cfg.universe_policy, all_years, args.duckdb)
        universe_by_year = {y: _resolved[y] for y in fold_years}
        serve_universe = sorted(_resolved[serve_year])
        symbols = sorted(set().union(*universe_by_year.values()))
        print(
            "[export] universe_policy: "
            + ", ".join(f"{y}={len(_resolved[y])}" for y in all_years)
            + f"; union(folds)={len(symbols)}; serving {serve_year}: {len(serve_universe)}"
        )
    else:
        symbols = _resolve_symbols(args)

    print(
        f"[export] template={cfg.name} strategy={cfg.strategy} "
        f"mode={'replicate-last-fold' if args.replicate_last_fold else 'single-fit'}"
    )

    from stock_ml.src.data.loader import get_loader

    # §13.3.1 fetch-on-miss: pull survivorship-correct names the corrected universe (§13.9) surfaces
    # but the local cache lacks, so the export universe matches the trained one (no silent shrink).
    if universe_by_year is not None and str(args.duckdb).endswith(".duckdb"):
        from stock_ml.src.data.duckdb_loader import ensure_symbols_cached

        ensure_symbols_cached(args.duckdb, symbols)

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
    # Single-fit clips to the cutoff; replicate-fold / fold-models keep the full series (the splitter
    # carves each train window itself and the test windows extend to the data tail).
    if not args.replicate_last_fold and not args.fold_models and not args.fold_models_from_run:
        ohlcv = ohlcv[pd.to_datetime(ohlcv["date"]) <= pd.Timestamp(args.cutoff)].copy()
    if ohlcv.empty:
        raise SystemExit("export_bundle: no OHLCV after clipping")
    print(
        f"[export] {len(ohlcv)} bars, {ohlcv['symbol'].nunique()} symbols, "
        f"through {pd.to_datetime(ohlcv['date']).max().date()}"
    )

    (
        feat,
        entry_feat_cols,
        exit_feat_cols,
        entry_target_cfg,
        exit_target_cfg,
        entry2_feat_cols,
        entry3_feat_cols,
        entry4_feat_cols,
        entry5_feat_cols,
        entry6_feat_cols,
    ) = build_feature_frame(
        ohlcv, cfg, requested_symbols=requested, data_root=args.duckdb, with_targets=True
    )

    if args.replicate_last_fold or args.fold_models or args.fold_models_from_run:
        # Recreate the backtest's walk-forward folds via the split config. replicate-last-fold trains the
        # LAST fold (bundle == that model); --fold-models trains ALL folds (§11.9); --fold-models-from-run
        # loads them from the backtest instead of training. Either way the last fold's test_start is the
        # serve cutoff.
        from stock_ml.src.data.splitter import YearSplitter

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
        # universe_by_year: dynamic universe masks each fold to its year's list (None = legacy).
        folds = list(splitter.split(feat, universe_by_year=universe_by_year))
        if not folds:
            raise SystemExit("export_bundle: splitter produced no folds")
        w_last, train_src, _test_df = folds[-1]
        cutoff_used = pd.Timestamp(w_last.test_start)
        # Drop rows the model cannot train on (un-ripe forward-target tail of a symbol
        # whose data ends mid-window, e.g. a suspension/delisting, + NaN-feature zones) —
        # same fail-loud guard the single-fit path honors. The backtest's effective train
        # set excludes these too (see experiment.py train count via dropna).
        _need = ["target_entry", "target_exit"] + sorted(set(entry_feat_cols) | set(exit_feat_cols))
        _n0 = len(train_src)
        train_src = train_src.dropna(subset=[c for c in _need if c in train_src.columns])
        if len(train_src) < _n0:
            print(
                f"[export] replicate-fold: dropped {_n0 - len(train_src)} un-ripe/NaN train row(s)"
            )
        print(
            f"[export] last fold: train {pd.to_datetime(train_src['date']).min().date()}.."
            f"{pd.to_datetime(train_src['date']).max().date()} -> test_start {cutoff_used.date()}"
        )
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

    # ENSEMBLE heads (N-head champions n2_3h_/4h_/5h_*): train_fold fits entry2..entry5 + exit2 on their
    # own targets when the target columns exist. Collect them via out_models so the bundle carries EVERY
    # head — else serving silently drops the ensemble union buys and live signals diverge from the backtest.
    def _train_models(tsrc: pd.DataFrame) -> dict:
        """Fit the full head set on one train frame and return {entry, exit, entry2.., exit2}."""

        def _tcol(name: str) -> str | None:
            return name if name in tsrc.columns else None

        om: dict = {}
        e_m, x_m, _ = train_fold(
            tsrc,
            tsrc.groupby("symbol", group_keys=False).tail(3),
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
            out_models=om,
        )
        m = {"entry": e_m}
        if x_m is not None:
            m["exit"] = x_m
        m.update(om)  # entry2 / entry3 / exit2 when present (om also carries 'entry' == e_m)
        return m

    fold_models_collected: dict[int, dict] | None = None
    if args.fold_models_from_run:
        # §11.9 robust: ship the EXACT models the backtest persisted (no re-training). The re-trained
        # fold-model differs from the one that produced run_signals by ~0.01 in score, flipping band-edge
        # signals (~5% mismatch); loading verbatim makes serving == backtest by construction.
        import joblib

        run_folds = Path(args.fold_models_from_run)
        mfiles = sorted(run_folds.glob("*.models.joblib"))
        if not mfiles:
            raise SystemExit(
                f"export_bundle: no *.models.joblib in {run_folds} — re-run the backtest with "
                "STOCKML_PERSIST_FOLD_MODELS=1 so it persists per-fold models."
            )
        fold_models_collected = {}
        for mf in mfiles:
            rec = joblib.load(mf)
            fold_models_collected[int(rec["test_year"])] = rec["models"]
        print(
            f"[export] loaded persisted fold-models for {sorted(fold_models_collected)} "
            f"from {run_folds}",
            flush=True,
        )
        models = fold_models_collected[max(fold_models_collected)]
    elif args.fold_models:
        # §11.9: train EVERY fold; the serving path replays the walk-forward from these so history
        # reproduces the backtest by construction. Serve models = the last fold (owns the live window).
        _need = ["target_entry", "target_exit"] + sorted(set(entry_feat_cols) | set(exit_feat_cols))
        fold_models_collected = {}
        for w, ftrain, _ft in folds:
            ftrain = ftrain.dropna(subset=[c for c in _need if c in ftrain.columns])
            if ftrain.empty:
                continue
            fold_models_collected[int(w.test_year)] = _train_models(ftrain)
            print(
                f"[export] fold {w.test_year}: {sorted(fold_models_collected[int(w.test_year)])} "
                f"(train {len(ftrain)} rows)",
                flush=True,
            )
        if not fold_models_collected:
            raise SystemExit("export_bundle: --fold-models produced no non-empty folds")
        models = fold_models_collected[max(fold_models_collected)]
    else:
        models = _train_models(train_src)
    print(
        f"[export] fitted models: {sorted(models)}"
        + (f" + fold-models for {sorted(fold_models_collected)}" if fold_models_collected else "")
    )

    # --- prediction history (z-window seed / full walk-forward replay) ---
    prediction_history = None
    if args.pred_history_csv:
        ph = pd.read_csv(args.pred_history_csv)
        ph["date"] = pd.to_datetime(ph["date"]).dt.normalize()
        # Carry ensemble prediction columns too when the backtest CSV has them, so the
        # serving z-window for score2/score3 (the union-buy bands) is seeded exactly as in
        # the backtest, not re-predicted by the current model on warmup bars.
        _ph_cols = ["symbol", "date", "score", "exit_score"] + [
            c
            for c in ph.columns
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
            print(
                f"[export] mode=walk_forward: embedding {len(ph)} full backtest rows "
                f"(history replays backtest; future uses this model)"
            )
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
    _entry_head_feat = {
        2: entry2_feat_cols,
        3: entry3_feat_cols,
        4: entry4_feat_cols,
        5: entry5_feat_cols,
    }
    for k, fc in _entry_head_feat.items():
        if f"entry{k}" in models:
            ens_key = "entry_ensemble" if k == 2 else f"entry_ensemble{k - 1}"
            feature_spec[f"entry{k}_feat_cols"] = list(fc) if fc else None
            feature_spec[f"entry{k}_target"] = (cfg.engine.get(ens_key) or {}).get("target")
    if "exit2" in models:
        feature_spec["exit2_target"] = (cfg.engine.get("exit_ensemble") or {}).get("target")
    manifest_extra = {
        "template_name": cfg.name,
        "template_id": args.template_id,
        "strategy": cfg.strategy,
        "cutoff_date": str(cutoff.date()),
        # Dynamic universe: serving trades the CURRENT year's point-in-time list (resolver
        # runs once per year); static universe: the requested list as before.
        "universe": serve_universe if serve_universe is not None else requested,
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
    if serve_universe is not None:
        # Dynamic-universe provenance: the policy + every fold's resolved list (resolver
        # vintage) so a redeploy / next-year re-resolve can be reproduced and diffed.
        manifest_extra["universe_slug"] = (getattr(cfg, "universe", None) or {}).get("slug")
        manifest_extra["universe_policy"] = cfg.universe_policy
        manifest_extra["universe_serve_year"] = serve_year
        manifest_extra["universe_by_year"] = {str(y): v for y, v in universe_by_year.items()}
    # Self-sufficient export record (§11.5.1): the 227 resolved engine nodes + z-window + catalog
    # fingerprint + wheel version + data declaration that config.json alone cannot regenerate the
    # signal from. Built from the SAME engine deserializer the backtest uses (R3, 0-number-change).
    from stock_ml.src.backtest.engine import engine_config_from_dict
    from stock_ml.src.serving.resolved import build_resolved_config

    _engine, _ = engine_config_from_dict(cfg.engine)
    _serve_universe = serve_universe if serve_universe is not None else requested
    resolved_config = build_resolved_config(
        cfg,
        _engine,
        universe=_serve_universe,
        universe_slug=manifest_extra.get("universe_slug")
        or (getattr(cfg, "universe", None) or {}).get("slug"),
        as_of=str(cutoff.date()),
    )
    path = write_bundle(
        out_dir,
        models=models,
        config=config_dict,
        feature_spec=feature_spec,
        manifest_extra=manifest_extra,
        prediction_history=prediction_history,
        resolved_config=resolved_config,
        fold_models=fold_models_collected,
    )
    print(f"[export] wrote bundle -> {path}")
    _write_parity(path, as_of=str(cutoff.date()))


if __name__ == "__main__":
    main()
