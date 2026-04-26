"""Build V47 tri-route trades artifact from source model CSVs."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

VERSION = "v47_tri_route"
SOURCES = ["v37a_exit", "v42_base", "v44_bottom_base"]
ROUTE_V42 = {"DGC", "KBC", "VIC", "VJC", "VND"}
ROUTE_V44 = {"FRT", "HSG", "MSN", "NVL", "SAB"}


def _load(version: str) -> pd.DataFrame:
    path = RESULTS / f"trades_{version}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build() -> pd.DataFrame:
    trades = {version: _load(version) for version in SOURCES}
    symbols = sorted(set().union(*(set(df["symbol"]) for df in trades.values())))

    parts = []
    for symbol in symbols:
        if symbol in ROUTE_V42:
            source = "v42_base"
        elif symbol in ROUTE_V44:
            source = "v44_bottom_base"
        else:
            source = "v37a_exit"

        chunk = trades[source][trades[source]["symbol"] == symbol].copy()
        chunk["route_source"] = source
        parts.append(chunk)

    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "entry_date"]).reset_index(drop=True)


def write_meta(df: pd.DataFrame) -> None:
    base_meta_path = RESULTS / "trades_v37a_exit.meta.json"
    symbols = sorted(df["symbol"].unique().tolist())
    min_rows = 2000
    if base_meta_path.exists():
        base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
        symbols = base_meta.get("symbols", symbols)
        min_rows = base_meta.get("min_rows", min_rows)

    meta = {
        "version": VERSION,
        "run_key": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "tools/build_v47_tri_route.py",
        "symbols": symbols,
        "n_symbols": len(symbols),
        "min_rows": min_rows,
        "strategy": "tri_route_simulated",
        "feature_set": "mixed",
        "source_versions": SOURCES,
        "route": {
            "v42_base": sorted(ROUTE_V42),
            "v44_bottom_base": sorted(ROUTE_V44),
            "v37a_exit": "all_other_symbols",
        },
        "note": "Per-symbol routing artifact built from source trade CSVs; not a trainable pipeline strategy.",
    }
    (RESULTS / f"trades_{VERSION}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    df = build()
    out = RESULTS / f"trades_{VERSION}.csv"
    df.to_csv(out, index=False)
    write_meta(df)
    print(f"{VERSION}: {len(df)} trades, total_pnl={df['pnl_pct'].sum():+.1f}%")
    print(df["route_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
