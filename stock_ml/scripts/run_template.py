"""Template-based experiment runner (DB-First Phase 0-3).

Loads strategy template from DB, converts to ExperimentConfig, and runs experiment.

Usage:
    python stock_ml/scripts/run_template.py \
        --template-id 5 \
        --seed 42 \
        [--symbols AAA,SSI] \
        [--out results/]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Both roots are needed: src.* imports resolve via stock_ml/, while the source
# modules use absolute stock_ml.src.* imports that need the repo root (stock_ml
# is a namespace package with no __init__.py).
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.market_profile import resolve_run_context
from src.pipeline.experiment import ExperimentConfig, run_experiment
from src.utils.config_loader import get_pipeline_symbols
from src.utils.env import resolve_data_dir


def _run_async(coro):
    """Run a coroutine in a fresh event loop, disposing the shared async engine
    afterwards. The module-level asyncpg engine pools connections bound to the
    loop that created them; without disposal, a connection from a previous
    asyncio.run() is reused on a closed loop and crashes on Windows
    ('NoneType' object has no attribute 'send'). Disposing inside each loop keeps
    every run self-contained.
    """

    async def _wrapper():
        try:
            return await coro
        finally:
            from db.engine import async_engine

            await async_engine.dispose()

    return asyncio.run(_wrapper())


def _collect_provenance(data_root: str):
    """Capture (git_sha, data_snapshot_date, lib_versions) so a run is reproducible."""
    import subprocess
    from importlib.metadata import PackageNotFoundError, version

    git_sha = None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        git_sha = out.stdout.strip() or None
    except Exception:
        git_sha = None

    libs = {}
    for pkg in ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "pyarrow", "duckdb"]:
        with contextlib.suppress(PackageNotFoundError):
            libs[pkg] = version(pkg)

    snapshot = None
    try:
        if str(data_root).endswith(".duckdb"):
            import duckdb

            con = duckdb.connect(str(data_root), read_only=True)
            snapshot = con.execute("SELECT max(date) FROM ohlcv").fetchone()[0]
            con.close()
    except Exception:
        snapshot = None

    return git_sha, snapshot, libs


async def _persist_run_detail(run_id: str, frames: dict) -> None:
    """Persist trades, signals, and stats from a backtest to the DB.

    P3: consumes the in-memory frames returned by ``run_experiment`` (no CSV
    round-trip on disk). Each frame is serialized to an in-memory CSV buffer and
    parsed with the same coercions the on-disk path used, so the persisted rows are
    byte-for-byte identical to the legacy file-ingest path (golden parity).
    """
    import csv
    import io

    from db.engine import async_engine
    from db.repositories.signal_repo import RunSignalRepository
    from db.repositories.symbol_stat_repo import RunSymbolStatRepository
    from db.repositories.trade_repo import RunTradeRepository
    from db.repositories.yearly_stat_repo import RunYearlyStatRepository
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    def _rows(key: str) -> list[dict]:
        """DictReader over an in-memory CSV of the named frame (None/empty -> [])."""
        df = frames.get(key)
        if df is None or df.empty:
            return []
        return list(csv.DictReader(io.StringIO(df.to_csv(index=False))))

    async_session_maker = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session_maker() as session:
            # Trades
            trades = [
                {
                    "symbol": row.get("symbol"),
                    "entry_date": row.get("entry_date"),
                    "entry_price": float(row.get("entry_price", 0)) or None,
                    "exit_date": row.get("exit_date"),
                    "exit_price": float(row.get("exit_price", 0)) or None,
                    "holding_days": float(row.get("holding_days", 0)) or None,
                    "pnl_pct": float(row.get("pnl_pct", 0)),
                    "exit_reason": row.get("exit_reason"),
                    "entry_signal_date": row.get("entry_signal_date"),
                }
                for row in _rows("trades")
            ]
            trade_repo = RunTradeRepository(session)
            # Replace semantics: clear this run's prior rows so re-running a
            # template doesn't accumulate duplicates (run_trades has no unique
            # index, so on_conflict_do_nothing can't dedupe across runs).
            await trade_repo.delete_by_run_id(run_id)
            inserted = await trade_repo.bulk_insert(run_id, trades)
            print(f"  Inserted {inserted} trades")

            # Yearly stats
            yearly_stats = [
                {
                    "year": int(row.get("year", 0)),
                    "trades": int(row.get("n_trades", 0)) if row.get("n_trades") else None,
                    "win_rate": float(row.get("win_rate", 0)) or None,
                    "total_pnl": float(row.get("total_pnl", 0)) or None,
                    "max_drawdown": float(row.get("max_drawdown", 0)) or None,
                    "avg_pnl": float(row.get("avg_pnl", 0)) or None,
                    "med_pnl": float(row.get("med_pnl", 0)) or None,
                    "std_pnl": float(row.get("std_pnl", 0)) or None,
                    "max_win": float(row.get("max_win", 0)) or None,
                    "max_loss": float(row.get("max_loss", 0)) or None,
                    "profit_factor": float(row.get("profit_factor", 0)) or None,
                    "avg_hold": float(row.get("avg_hold_days", 0)) or None,
                }
                for row in _rows("yearly")
            ]
            yearly_repo = RunYearlyStatRepository(session)
            await yearly_repo.delete_by_run_id(run_id)  # replace semantics
            inserted = await yearly_repo.bulk_insert(run_id, yearly_stats)
            print(f"  Inserted {inserted} yearly stat records")

            # Symbol stats
            symbol_stats = [
                {
                    "symbol": row.get("symbol"),
                    "trades": int(row.get("n_trades", 0)) if row.get("n_trades") else None,
                    "win_rate": float(row.get("win_rate", 0)) or None,
                    "total_pnl": float(row.get("total_pnl", 0)) or None,
                    "avg_pnl": float(row.get("avg_pnl", 0)) or None,
                    "med_pnl": float(row.get("med_pnl", 0)) or None,
                    "std_pnl": float(row.get("std_pnl", 0)) or None,
                    "max_win": float(row.get("max_win", 0)) or None,
                    "max_loss": float(row.get("max_loss", 0)) or None,
                    "profit_factor": float(row.get("profit_factor", 0)) or None,
                    "avg_hold": float(row.get("avg_hold_days", 0)) or None,
                }
                for row in _rows("symbol")
            ]
            symbol_repo = RunSymbolStatRepository(session)
            await symbol_repo.delete_by_run_id(run_id)  # replace semantics
            inserted = await symbol_repo.bulk_insert(run_id, symbol_stats)
            print(f"  Inserted {inserted} symbol stat records")

            # Per (symbol, date) signals + continuous alpha score.
            signals = []
            for row in _rows("signals"):
                date_val = (row.get("date") or "")[:10]  # YYYY-MM-DD
                score_raw = row.get("score")
                exit_raw = row.get("exit_score")  # dual-ML only; None otherwise
                signals.append(
                    {
                        "symbol": row.get("symbol"),
                        "date": date_val,
                        "signal": int(float(row.get("signal", 0))),
                        "score": float(score_raw) if score_raw not in (None, "") else None,
                        "exit_score": float(exit_raw) if exit_raw not in (None, "") else None,
                    }
                )
            signal_repo = RunSignalRepository(session)
            await signal_repo.delete_by_run_id(run_id)  # replace semantics
            inserted = await signal_repo.bulk_insert(run_id, signals)
            print(f"  Inserted {inserted} signal records")

            await session.commit()
    except Exception as e:
        # Fail loud: persisting trades/stats is what the dashboard reads. Hiding a write
        # failure behind a warning leaves the UI showing stale/partial data.
        print(f"[error] Failed to persist run detail: {e}")
        traceback.print_exc()
        raise


async def load_template_config(template_id: int) -> ExperimentConfig:
    """Load template from DB async."""
    from db.engine import async_engine
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    async_session_maker = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session_maker() as session:
        return await ExperimentConfig.from_template_id_async(template_id, session)


def run_template_experiment(
    template_id: int,
    seed: int | None = None,
    symbols: str | None = None,
    out_dir: Path | None = None,
    export_csv: bool = False,
) -> dict:
    """Load template from DB and run experiment synchronously.

    Args:
        template_id: ID of StrategyTemplateModel in DB
        seed: Override seed
        symbols: Override symbols (comma-separated)
        out_dir: Output directory for results

    Returns:
        Result summary dict
    """
    try:
        # Load template config (async)
        config = _run_async(load_template_config(template_id))
        print(f"✓ Loaded template {template_id}: {config.name}")

        # Apply CLI overrides
        if seed is not None:
            config.seed = seed
            print(f"  Override: seed={seed}")

        # Resolve data directory from market
        context = resolve_run_context({"market": config.market})
        if not context.resolved_data_dir:
            raise ValueError(f"No data directory configured for market {config.market}")
        # Honor STOCK_DATA_DIR override and consistent path resolution (same as
        # get_pipeline_symbols), so local runs and container runs agree.
        data_root = resolve_data_dir(context.resolved_data_dir)

        # Resolve symbols
        if symbols:
            sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            print(f"  Override: symbols={sym_list}")
        else:
            sym_list = get_pipeline_symbols(
                market=config.market,
                universe_config=config.universe,
            )
            print(f"  Resolved symbols: {len(sym_list)} symbols")

        # Run experiment
        output_dir = out_dir or Path("results")
        # Fold-cache key MUST depend on the config that affects predictions (split/gap,
        # targets, features, model, seed) — NOT just template_id. Keying by template_id alone
        # let a re-run after a config change (e.g. gap_days 25->85 for the leakage fix) restore
        # stale fold signals and silently reproduce the OLD result. Fingerprint the
        # prediction-affecting config so any such change invalidates the cache (fresh retrain).
        _fp_src = {
            "strategy": config.strategy,
            "split": config.split,
            "seed": config.seed,
            "feature_set": config.feature_set,
            "entry_features": config.entry_features,
            "exit_features": config.exit_features,
            "entry_target": config.entry_target,
            "exit_target": config.exit_target,
            "target": config.target,
            "entry_model": config.entry_model,
            "exit_model": config.exit_model,
        }
        _fp = hashlib.sha256(json.dumps(_fp_src, sort_keys=True, default=str).encode()).hexdigest()[
            :10
        ]
        result_dict = run_experiment(
            cfg=config,
            data_root=str(data_root),
            symbols=sym_list,
            out_dir=str(output_dir),
            run_id=f"tmpl_{template_id}_{_fp}",
            export_csv=export_csv,
        )
        print("✓ Experiment completed")

        # Upsert to DB leaderboard
        from datetime import UTC, datetime

        from db.repositories.run_repo import LeaderboardRunRepository
        from src.evaluation.scoring import composite_score
        from src.leaderboard.schema import (
            Artifacts,
            CostProfile,
            LeaderboardRow,
            LifecycleState,
            TargetConfig,
        )

        # Convert result_dict (summary) to LeaderboardRow
        # Always upsert even if no signals (0 trades is valid result)
        run_id: str | None = None
        if result_dict:
            summary = result_dict
            # P3: detach the in-memory detail frames (trades/signals/yearly/symbol)
            # before building the leaderboard row; they feed _persist_run_detail
            # directly, replacing the old CSV-on-disk round-trip.
            frames = summary.pop("_run_detail_frames", {})
            agg = summary.get("aggregate", {})
            cfg = summary.get("config", {})
            engine_cfg = cfg.get("engine", {})

            # Faithful per-slot metadata: the leaderboard's single `feature_set` /
            # `target_type` columns must reflect what the entry/exit heads actually
            # used, not the legacy global template field. For non-decoupled models
            # entry == exit, so the combined form collapses back to a single value.
            _global_fs = summary.get("feature_set", "unknown")
            _entry_fs = cfg.get("entry_features") or _global_fs
            _exit_fs = cfg.get("exit_features") or _global_fs
            feature_set_label = _entry_fs if _entry_fs == _exit_fs else f"{_entry_fs}+{_exit_fs}"
            _global_tt = cfg.get("target", {}).get("type", "unknown")
            _entry_tt = (cfg.get("entry_target") or {}).get("type") or _global_tt
            _exit_tt = (cfg.get("exit_target") or {}).get("type") or _entry_tt
            target_type_label = _entry_tt if _entry_tt == _exit_tt else f"{_entry_tt}+{_exit_tt}"

            run_name = summary.get("name", "unknown")
            bundle = "template"
            # Compute deterministic hash from actual config dict (URL-safe hex chars only)
            config_hash = hashlib.sha256(
                json.dumps(cfg, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            run_id = f"{bundle}/{run_name}-{config_hash[:8]}"
            now = datetime.now(UTC)

            # Pass the trades list so composite_score uses the SAME per-symbol MDD +
            # yearly-consistency path as the offline rescore. WITHOUT trades it falls back
            # to mdd≈abs(max_loss) and yr_cv=0 (no yearly penalty) — inflating the score
            # ~30 pts vs the rescore and making run-stored scores incomparable to rescored
            # ones (root-caused 2026-06-11).
            _tr_df = frames.get("trades")
            _trades_list = None
            if _tr_df is not None and not _tr_df.empty:
                _trades_list = [
                    {
                        "symbol": rec.get("symbol"),
                        "entry_date": str(rec.get("entry_date")),
                        "pnl_pct": float(rec.get("pnl_pct", 0) or 0),
                        "holding_days": float(rec.get("holding_days", 0) or 0),
                    }
                    for rec in _tr_df.to_dict("records")
                ]
            score = composite_score(
                metrics={
                    "trades": agg.get("n_trades", 0),
                    "avg_pnl": agg.get("avg_pnl", 0.0),
                    "total_pnl": agg.get("total_pnl", 0.0),
                    "pf": agg.get("profit_factor", 0.0),
                    "max_loss": agg.get("max_loss", 0.0),
                    "avg_hold": agg.get("avg_hold_days", 0.0),
                    "sharpe": summary.get("sharpe", 0.0),
                    "n_symbols": summary.get("n_symbols", 0),
                },
                trades=_trades_list,
            )

            first_test_year = summary.get("first_test_year") or 2020
            last_test_year = summary.get("last_test_year") or 2024
            backtest_window_key = summary.get(
                "backtest_window_key", f"{first_test_year}-{last_test_year}"
            )

            row = LeaderboardRow(
                run_id=run_id,
                bundle=bundle,
                run_name=run_name,
                config_hash=config_hash,
                generated_at=str(now.isoformat()),
                strategy=summary.get("strategy", "unknown"),
                market=summary.get("market", "vn_stock"),
                feature_set=feature_set_label,
                entry_model=summary.get("entry_model", "unknown"),
                timeframe=summary.get("timeframe", "unknown"),
                model_mode=summary.get("model_mode", "ml_only"),
                signal_mode=summary.get("signal_mode", "entry_first"),
                direction=summary.get("direction", "long"),
                universe_slug=summary.get("universe_slug"),
                universe_version=summary.get("universe_version"),
                target=TargetConfig(
                    type=target_type_label,
                    # Honest max forward span across both per-slot targets (exposed by
                    # experiment.py); fall back to the global horizon for older summaries.
                    forward_window=cfg.get(
                        "target_forward_window",
                        cfg.get("target", {}).get("horizon", 5),
                    ),
                ),
                trades=agg.get("n_trades", 0),
                wr=agg.get("win_rate", 0.0),
                pf=agg.get("profit_factor", 0.0),
                avg_pnl=agg.get("avg_pnl", 0.0),
                total_pnl=agg.get("total_pnl", 0.0),
                pnl_pct=agg.get("total_pnl", 0.0) * 100,
                avg_hold=agg.get("avg_hold_days", 0.0),
                max_win=agg.get("max_win", 0.0),
                max_loss=agg.get("max_loss", 0.0),
                sharpe=summary.get("sharpe", 0.0),
                max_drawdown=summary.get("max_drawdown", 0.0),
                mdd_per_symbol=summary.get("mdd_per_symbol", 0.0),
                yearly_consistency=summary.get("yearly_consistency", 0.0),
                n_symbols=summary.get("n_symbols", 0),
                first_test_year=first_test_year,
                last_test_year=last_test_year,
                backtest_window_key=backtest_window_key,
                fairness_group_key=f"{bundle}_{summary.get('market', 'unknown')}",
                composite_score=score,
                cost_profile=CostProfile(
                    commission=float(engine_cfg.get("commission", 0.0025)),
                    tax=float(engine_cfg.get("tax", 0.001)),
                    slippage=float(engine_cfg.get("slippage", 0.0015)),
                ),
                artifacts=Artifacts(trades_csv="", meta_json="", model_pkl=""),
                state=LifecycleState.trained,
            )

            git_sha, data_snapshot_date, lib_versions = _collect_provenance(str(data_root))

            async def upsert_to_leaderboard():
                from db.engine import async_engine
                from sqlalchemy.ext.asyncio import AsyncSession
                from sqlalchemy.orm import sessionmaker

                async_session_maker = sessionmaker(
                    async_engine, class_=AsyncSession, expire_on_commit=False
                )
                async with async_session_maker() as async_session:
                    repo = LeaderboardRunRepository(async_session)
                    await repo.upsert(
                        row,
                        run_seed=config.seed,
                        template_config_hash=config_hash,
                        template_id=template_id,
                        git_sha=git_sha,
                        data_snapshot_date=data_snapshot_date,
                        lib_versions=lib_versions,
                    )
                    await async_session.commit()

            _run_async(upsert_to_leaderboard())
            print(f"✓ Upserted result to leaderboard: {run_id}")

            # Persist run details (trades, signals, stats) to DB from the in-memory
            # frames returned by run_experiment (P3: no CSV round-trip).
            print("• Persisting run details to database...")
            _run_async(_persist_run_detail(run_id, frames))
            print("✓ Run details persisted to database")

        return {
            "success": True,
            "template_id": template_id,
            "template_name": config.name,
            "run_id": run_id,
        }

    except Exception as e:
        print(f"✗ Experiment failed: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run template-based experiment")
    parser.add_argument("--template-id", type=int, required=True, help="StrategyTemplate ID")
    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument("--symbols", help="Override symbols (comma-separated)")
    parser.add_argument("--out", type=Path, help="Output directory")
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also write detail CSVs + summary JSON to disk (debug). DB persistence "
        "happens regardless; default off (P3: DB is the source of truth).",
    )
    args = parser.parse_args()

    result = run_template_experiment(
        template_id=args.template_id,
        seed=args.seed,
        symbols=args.symbols,
        out_dir=args.out,
        export_csv=args.export_csv,
    )

    if result["success"]:
        print("\n✓ Template experiment completed successfully")
        sys.exit(0)
    else:
        print(f"\n✗ Template experiment failed: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
