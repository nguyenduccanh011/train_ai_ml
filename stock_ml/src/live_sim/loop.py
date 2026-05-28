"""Live simulator main loop — day-by-day walk-forward execution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.data.splitter import YearSplitter
from src.features.basic import FEATURE_COLS, add_features
from src.live_sim.config import LiveSimConfig
from src.live_sim.executor import EntryExecutor, ExitEvaluator
from src.live_sim.reporter import Reporter
from src.live_sim.signals import SignalGenerator
from src.live_sim.state import ClosedTrade, SimState
from src.models.baseline import BaselineModel


class LiveSimEngine:
    """Day-by-day live simulator."""

    def __init__(self, cfg: LiveSimConfig):
        self.cfg = cfg
        self.reporter = Reporter(cfg.out_dir)

    def run(self) -> dict:
        """Run the live simulator.

        Returns:
            summary dict with metrics and outputs
        """
        print(f"[LiveSim] loading {len(self.cfg.symbols)} symbols...")
        loader = DataLoader(self.cfg.data_root)
        available = set(loader.list_symbols())
        requested = [s for s in self.cfg.symbols if s in available]
        missing = sorted(set(self.cfg.symbols) - available)
        if missing:
            print(f"  [warn] missing symbols: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if not requested:
            raise ValueError("no symbols available in the dataset")

        print(f"[LiveSim] loading OHLCV for {len(requested)} symbols...")
        full_ohlcv = loader.load_many(requested)
        print(f"[LiveSim] loaded {len(full_ohlcv)} bars")

        print("[LiveSim] computing features...")
        full_feat = add_features(full_ohlcv)
        full_feat = self.cfg.target.apply(full_feat)

        sim_start = pd.Timestamp(self.cfg.sim_start)
        sim_end = pd.Timestamp(self.cfg.sim_end)

        print(
            f"[LiveSim] training model on data <= {sim_start.date() - pd.Timedelta(days=self.cfg.gap_days)}..."
        )
        splitter = YearSplitter(
            train_years=self.cfg.train_years,
            gap_days=self.cfg.gap_days,
            first_test_year=sim_start.year,
            last_test_year=sim_start.year,
        )
        windows = splitter.windows()
        if not windows:
            raise ValueError(f"No train window for {sim_start.year}")

        w = windows[0]
        train_mask = (pd.to_datetime(full_feat["date"]) >= w.train_start) & (
            pd.to_datetime(full_feat["date"]) < w.train_end
        )
        train_df = full_feat[train_mask].copy()
        train_clean = train_df.dropna(subset=["target", *FEATURE_COLS])

        if train_clean.empty:
            raise ValueError("no usable training rows")

        X_tr = train_clean[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_tr = train_clean["target"].to_numpy(dtype=np.float64)

        model = BaselineModel(seed=self.cfg.seed).fit(X_tr, y_tr)
        print(
            f"[LiveSim] model trained on {len(X_tr)} rows, "
            f"{train_clean['symbol'].nunique()} symbols"
        )

        print(f"[LiveSim] running simulation {sim_start.date()} -> {sim_end.date()}...")
        sim_dates = pd.bdate_range(sim_start, sim_end)

        signal_gen = SignalGenerator(model, self.cfg)
        entry_exec = EntryExecutor(self.cfg.engine.cost, self.cfg.engine)
        exit_eval = ExitEvaluator(self.cfg.engine.cost, self.cfg.engine)

        state = SimState()

        for i, today in enumerate(sim_dates):
            if i > 0:
                yesterday = sim_dates[i - 1]
            else:
                # First trading day: find the last date in history before today
                feat_dates = pd.to_datetime(full_feat["date"]).dt.normalize()
                mask = feat_dates < today.normalize()
                if not mask.any():
                    raise ValueError(f"no history available before {today}")
                yesterday = pd.Timestamp(feat_dates[mask].max())

            # Check if bars exist for today first
            bars_mask = pd.to_datetime(full_ohlcv["date"]).dt.normalize() == today.normalize()
            bars_today = full_ohlcv[bars_mask].copy()

            if bars_today.empty:
                print(f"  [warn] no bars for {today.date()}, skipping")
                continue

            history_mask = pd.to_datetime(full_feat["date"]).dt.normalize() <= yesterday.normalize()
            history_feat = full_feat[history_mask].copy()

            if history_feat.empty:
                raise ValueError(f"no history at {yesterday}")

            frozen = signal_gen.generate(yesterday, today, history_feat)
            self.reporter.write_signal_log(frozen)

            entries = entry_exec.execute(frozen, bars_today, state)
            for entry in entries:
                state.open_position(entry)

            exits = exit_eval.evaluate(state, frozen, bars_today)
            for exit_trade in exits:
                state.closed_trades.append(exit_trade)

            state.current_date = today
            state.advance_holds()

            self.reporter.write_day(today, entries, exits, frozen, state)

            if (i + 1) % 20 == 0:
                print(
                    f"  [{i + 1}/{len(sim_dates)}] {today.date()}: "
                    f"entries={len(entries)}, exits={len(exits)}, "
                    f"open={state.n_open()}"
                )

        print("[LiveSim] closing remaining positions at end of sim...")
        for sym in list(state.open_positions.keys()):
            pos = state.open_positions[sym]
            last_bar = full_ohlcv.iloc[-1]
            close_price = float(last_bar["close"])
            exit_price = self.cfg.engine.cost.fill_sell(close_price)
            gross = exit_price / pos.entry_price - 1.0
            net = gross - self.cfg.engine.cost.round_trip_cost()

            trade = ClosedTrade(
                symbol=sym,
                entry_date=pos.entry_date,
                entry_price=pos.entry_price,
                exit_date=pd.Timestamp(last_bar["date"]),
                exit_price=exit_price,
                holding_days=int((pd.Timestamp(last_bar["date"]) - pos.entry_date).days),
                pnl_pct=float(net),
                exit_reason="end_of_sim",
                entry_signal_date=pos.entry_signal_date,
                signal_hash="",
            )

            state.closed_trades.append(trade)
            state.close_position(sym)

        print("[LiveSim] writing summary...")
        summary = self.reporter.write_summary(state, self.cfg, self.cfg.sim_start, self.cfg.sim_end)

        print(f"[LiveSim] done. outputs in {self.cfg.out_dir}")
        print(f"  trades: {summary['n_trades']}")
        print(f"  total_pnl: {summary['metrics']['total_pnl']:.4f}")
        print(f"  win_rate: {summary['metrics']['win_rate']:.2%}")
        print(f"  composite_score: {summary['composite_score']:.4f}")

        return summary
