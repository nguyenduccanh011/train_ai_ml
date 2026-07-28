"""Golden guard for the champion Stage-2 portfolio overlay.

Step 0 of docs/refactor/PORTFOLIO_LAYER_UNIFICATION.md: every refactor step must
reproduce the pinned production numbers byte-exact. Two layers:

- test_fixture_integrity: fast, always on — the committed per-seed base/signal
  parquets must match their pinned MD5s (catches accidental fixture edits).
- test_champion_prod_overlay_golden: slow (~2-3 min/seed), needs the local serving
  data stores + nh_nav2 — run explicitly during refactor steps:

      RUN_PORTFOLIO_GOLDEN=1 pytest stock_ml/tests/test_portfolio_golden.py -q

If a data-store MD5 mismatches, the snapshot drifted (e.g. a new corporate-action
sync): that invalidates the golden — re-pin DELIBERATELY (re-run the 3 seeds,
update the JSON in its own commit), never loosen the assert.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GOLDEN_PATH = Path(__file__).parent / "goldens" / "champion_prod_overlay.json"
FIXTURES = Path(__file__).parent / "goldens" / "fixtures"
GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def test_fixture_integrity():
    for name, md5 in GOLDEN["fixtures_md5"].items():
        p = FIXTURES / name
        assert p.exists(), f"missing golden fixture: {name}"
        assert _md5(p) == md5, f"golden fixture drifted: {name}"


@pytest.mark.skipif(
    not os.environ.get("RUN_PORTFOLIO_GOLDEN"),
    reason="slow golden replay; set RUN_PORTFOLIO_GOLDEN=1 to run",
)
@pytest.mark.parametrize("seed", ["42", "21", "123"])
def test_champion_prod_overlay_golden(seed):
    g = GOLDEN["seeds"][seed]
    for path_str, md5 in GOLDEN["data_md5"].items():
        p = Path(path_str)
        if not p.exists():
            pytest.skip(f"pinned data store not on this machine: {path_str}")
        assert _md5(p) == md5, (
            f"pinned data snapshot drifted: {path_str} — golden invalid; "
            "re-pin deliberately (see module docstring), do not loosen"
        )

    env = dict(
        os.environ,
        REPLAY_BASE=str(FIXTURES / g["base"]),
        REPLAY_SIG=str(FIXTURES / g["signals"]),
    )
    out = subprocess.run(
        [sys.executable, str(REPO / "_champ_prod_replay.py")],
        capture_output=True, text=True, env=env, cwd=REPO, timeout=1800,
    )
    assert "PROD_REPLAY_DONE" in out.stdout, (
        f"replay did not complete:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
    )

    m0 = re.search(r"T\+0: NAV x([\d.]+)\s+CAGR ([\d.]+)%\s+DD (-[\d.]+)%", out.stdout)
    m2 = re.search(r"T\+2: NAV x([\d.]+)\s+CAGR ([\d.]+)%\s+DD (-[\d.]+)%", out.stdout)
    assert m0 and m2, f"could not parse replay output:\n{out.stdout[-2000:]}"
    got = {
        "t0_nav_x": m0.group(1), "t0_cagr": m0.group(2), "t0_dd": m0.group(3),
        "t2_nav_x": m2.group(1), "t2_cagr": m2.group(2), "t2_dd": m2.group(3),
    }
    want = {k: g[k] for k in got}
    assert got == want, f"seed {seed} drifted from golden: got={got} want={want}"

    rewritten = REPO / "_champ_src" / "_prod_rewritten.csv"
    assert _md5(rewritten) == g["rewritten_trades_md5"], (
        f"seed {seed}: rewritten trades CSV differs from golden (byte-parity broken)"
    )


# ---------------------------------------------------------------------------
# Module parity (steps 1-3): stock_ml.portfolio.run_portfolio must reproduce the
# SAME golden numbers as the _champ_prod_replay.py reference, byte-exact.
# ---------------------------------------------------------------------------
SERVING_DUCK = "C:/Users/DUC CANH PC/Desktop/stock-serving/market_data/market.duckdb"
SERVING_OHLCV = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
NAVSIM_DATE_HI = "2026-07-08"  # nh_nav2.DB cutoff the golden was pinned on


@pytest.mark.skipif(
    not os.environ.get("RUN_PORTFOLIO_GOLDEN"),
    reason="slow golden replay; set RUN_PORTFOLIO_GOLDEN=1 to run",
)
@pytest.mark.parametrize("seed", ["42", "21", "123"])
def test_module_parity_golden(seed):
    import pandas as pd

    from stock_ml.portfolio import DuckDBContext, PortfolioConstants, run_portfolio

    g = GOLDEN["seeds"][seed]
    for path_str in (SERVING_DUCK, SERVING_OHLCV):
        if not Path(path_str).exists():
            pytest.skip(f"serving data store not on this machine: {path_str}")

    base = pd.read_parquet(FIXTURES / g["base"])
    sig = pd.read_parquet(FIXTURES / g["signals"])
    ctx = DuckDBContext(SERVING_DUCK, SERVING_OHLCV, date_hi=NAVSIM_DATE_HI)

    r2 = run_portfolio(base, sig, ctx=ctx, C=PortfolioConstants(tplus=2))
    got2 = {"t2_nav_x": f"{r2['nav']:.2f}", "t2_cagr": f"{100 * r2['cagr']:.1f}",
            "t2_dd": f"{100 * r2['maxdd']:.1f}"}
    want2 = {k: g[k] for k in got2}
    assert got2 == want2, f"seed {seed} T+2 module drifted: got={got2} want={want2}"

    r0 = run_portfolio(base, sig, ctx=ctx, C=PortfolioConstants(tplus=0))
    got0 = {"t0_nav_x": f"{r0['nav']:.2f}", "t0_cagr": f"{100 * r0['cagr']:.1f}",
            "t0_dd": f"{100 * r0['maxdd']:.1f}"}
    want0 = {k: g[k] for k in got0}
    assert got0 == want0, f"seed {seed} T+0 module drifted: got={got0} want={want0}"

    # rewritten-trades byte parity: reproduce the reference CSV exactly
    rw = r2["rewritten"]
    out = rw[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"]).dt.strftime("%Y-%m-%d")
    out["exit_date"] = pd.to_datetime(out["exit_date"]).dt.strftime("%Y-%m-%d")
    tmp = REPO / "_champ_src" / "_module_rewritten_parity.csv"
    out.to_csv(tmp, index=False)
    try:
        assert _md5(tmp) == g["rewritten_trades_md5"], (
            f"seed {seed}: module rewritten trades differ from golden CSV"
        )
    finally:
        tmp.unlink(missing_ok=True)


def test_base_output_guard():
    """OUTPUT trades (already-overlaid) must be rejected loudly — the 151%->73% trap."""
    import pandas as pd

    from stock_ml.portfolio import run_portfolio

    bad = pd.DataFrame({
        "symbol": ["AAA"], "entry_date": ["2024-01-05"], "exit_date": ["2024-02-05"],
        "entry_signal_date": ["2024-01-02"], "entry_price": [10.0], "exit_price": [11.0],
        "exit_reason": ["preempt"],
    })
    with pytest.raises(ValueError, match="overlay-level exit_reason"):
        run_portfolio(bad, pd.DataFrame(), ctx=None)
