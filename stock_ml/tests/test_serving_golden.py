"""Serving golden — the production signal chain on a committed bundle (ENGINE_UPGRADE §11.5.3 / Phase 2b).

This is the "cái cân" (§11.5.3, replaces R6): the engine drives the EXACT production serving path
``load_bundle -> generate_signals_from_bundle`` (build_feature_frame -> predict_slot_signals ->
recombine_signals) on a committed OHLCV slice, and the discrete ``signal`` column (buy/sell/hold ∈
{1,0,-1}) must match a committed golden byte-for-byte. Any silent engine-default drift (a changed
recombine z-window, a moved band, a feature-pipeline change) flips a signal and fails here — which is
the whole point of a golden that runs the real chain rather than 2/227 fields on random-walk data.

Scope (vòng-3 decision 2026-07-31): **signal-level, runs on every plane.** Signals are discrete, so the
comparison is stable across Windows↔Linux; the float-sensitive trade/NAV layer (``trades_to_dataframe``
→ CAGR) is the Linux-only exact-match layer added on top later (§11.5.3 full, §9.3.4 — VND 40 vs 37 drift).

Self-contained by design: the fixture strategy uses a purely per-symbol feature set (``leading_v2``), so
the chain reads ONLY the fed OHLCV slice — no host ``market.duckdb``, no VNINDEX CSV, no network. (The
catalogue champions additionally read market-breadth via a CWD-relative ``market_data/market.duckdb`` that
ignores ``STOCK_DATA_DIR`` — the §1.2 hardcoded-path defect — so they are deliberately NOT used here; a
market-context golden waits on the Phase 1 data contract that makes that source injectable.)

NO SWITCH (§9 refute / §11.5.3): no env-gated regen, no auto-create-on-missing, no skip-on-missing. A
missing fixture is a HARD failure, never a silent self-void (the three legacy goldens all self-voided —
``test_recombine_snapshot.py:116-120`` / ``test_baseline_snapshot.py:129-134`` write-then-return,
``test_portfolio_golden.py`` skips on an absolute path). To refresh after an intentional re-baseline
(Phase 4), regenerate with ``fixtures/serving_golden/regen.py`` and commit — never edit the golden by hand.

Fixture (``fixtures/serving_golden/``, committed & self-contained):
  - ``bundle/``            mini single-fit bundle of the ``regression_dual_ml_recombine`` strategy
                          (template 112 ``sumEX_thr15``; entry forward-return + exit forward-drawdown
                          heads, per-symbol ``leading_v2`` features) on 6 liquid symbols ≤ 2024-01-01.
  - ``ohlcv_slice.parquet`` those 6 symbols 2022-06..2024-06 (warm for the 252-bar z-window).
  - ``golden_signals.csv``  the discrete signal the production chain emits on the slice.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_FX = Path(__file__).parent / "fixtures" / "serving_golden"
_BUNDLE = _FX / "bundle"
_SLICE = _FX / "ohlcv_slice.parquet"
_GOLDEN = _FX / "golden_signals.csv"


def _require(path: Path) -> Path:
    # No skip-on-missing: a missing fixture means the "cái cân" is not actually running — fail loud.
    if not path.exists():
        raise AssertionError(
            f"serving-golden fixture missing: {path}. This test must NOT self-void — regenerate via "
            f"fixtures/serving_golden/regen.py and commit (§11.5.3)."
        )
    return path


def test_serving_golden_signal_chain() -> None:
    """Production chain on the committed bundle reproduces the golden discrete signals exactly."""
    from stock_ml.src.serving.bundle import load_bundle
    from stock_ml.src.serving.inference import generate_signals_from_bundle

    bundle = load_bundle(_require(_BUNDLE))
    ohlcv = pd.read_parquet(_require(_SLICE))
    golden = pd.read_csv(_require(_GOLDEN))
    golden["date"] = golden["date"].astype(str)

    out = generate_signals_from_bundle(bundle, ohlcv)[["symbol", "date", "signal"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Row set (symbol,date) must be identical — a dropped/added bar is a chain change.
    assert out[["symbol", "date"]].values.tolist() == golden[["symbol", "date"]].values.tolist(), (
        "serving-golden: (symbol,date) row set drifted from the golden — the production chain emits a "
        "different set of bars than when the golden was pinned."
    )
    # Discrete signal must match exactly (platform-stable, no tolerance = no hidden switch).
    mism = out.index[out["signal"].values != golden["signal"].values]
    assert len(mism) == 0, (
        f"serving-golden: {len(mism)} signal(s) differ from the golden (first: "
        f"{out.loc[mism[0], ['symbol', 'date']].to_dict() if len(mism) else None}). "
        "The engine changed a production signal — if intentional, re-baseline (Phase 4) and regen the golden."
    )


def test_serving_golden_is_nontrivial() -> None:
    """Guard the guard: the golden must carry real buys AND sells, else a match proves nothing."""
    golden = pd.read_csv(_require(_GOLDEN))
    counts = golden["signal"].value_counts().to_dict()
    assert counts.get(1, 0) > 0 and counts.get(-1, 0) > 0, (
        f"serving-golden is degenerate (needs buys and sells): {counts}"
    )
