"""Parity guard for the bot_shallow_* engine knob (quality-gated shallow fill).

Runs champion template 2646 (n2_2643_wavestruct_la05_lamp02) seed=42 with CSV export
and prints the leaderboard row. Run once BEFORE the engine edit (tag=before) and once
AFTER (tag=after); the knob is gated default-off so the numbers must be identical.

Usage: python stock_ml/analysis/serving_blindspot/wavestart/bs_parity.py <before|after>
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402

from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
TMPL = 2646
SEED = 42
OUT = Path(__file__).resolve().parent / "parity"


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "before"
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    r = run_template_experiment(template_id=TMPL, seed=SEED, out_dir=out_dir, export_csv=True)
    run_id = r.get("run_id")
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
        "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    row = cur.fetchone()
    con.close()
    print(f"PARITY[{tag}] run_id={run_id}")
    print(f"PARITY[{tag}] composite={row[0]!r} pnl={row[1]!r} pf={row[2]!r} "
          f"mdd={row[3]!r} trades={row[4]!r}")


if __name__ == "__main__":
    main()
