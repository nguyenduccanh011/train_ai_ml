"""Phase A2: same-wave trade matching V34 vs Rule.

For each symbol, pair v34 trades with rule trades that overlap in time
([entry_date, exit_date] intersect). Compute entry/exit lag + price diff.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.env import resolve_data_dir
from src.data.loader import DataLoader


def load_v34() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "trades_v34.csv")
    # Drop duplicate symbol col if both exist
    if "entry_symbol" in df.columns and "symbol" in df.columns:
        df = df.drop(columns=["entry_symbol"])
    elif "entry_symbol" in df.columns:
        df = df.rename(columns={"entry_symbol": "symbol"})
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    return df


def load_rule() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "trades_rule.csv")
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    return df


def attach_prices(df: pd.DataFrame, loader: DataLoader, label: str) -> pd.DataFrame:
    """Add entry_price/exit_price columns by lookup OHLCV close."""
    if "entry_price" in df.columns and "exit_price" in df.columns:
        return df
    out = df.copy()
    out["entry_price"] = np.nan
    out["exit_price"] = np.nan
    for sym, g in df.groupby("symbol"):
        try:
            ohlcv = loader.load_symbol(sym)
        except Exception as e:
            print(f"  [skip {sym}] {e}")
            continue
        ohlcv["date"] = pd.to_datetime(ohlcv["timestamp"]).dt.tz_localize(None)
        ohlcv = ohlcv.set_index("date")
        for idx in g.index:
            edate = df.at[idx, "entry_date"]
            xdate = df.at[idx, "exit_date"]
            try:
                # Find nearest available bar (forward-fill if exact missing)
                ep = ohlcv.loc[ohlcv.index <= edate, "close"].iloc[-1] if (ohlcv.index <= edate).any() else np.nan
                xp = ohlcv.loc[ohlcv.index <= xdate, "close"].iloc[-1] if (ohlcv.index <= xdate).any() else np.nan
                out.at[idx, "entry_price"] = ep
                out.at[idx, "exit_price"] = xp
            except Exception:
                pass
    return out


def match_same_wave(v34: pd.DataFrame, rule: pd.DataFrame) -> pd.DataFrame:
    """For each symbol: cartesian-pair trades whose [entry,exit] intervals overlap."""
    pairs = []
    symbols = sorted(set(v34["symbol"].unique()) | set(rule["symbol"].unique()))
    for sym in symbols:
        v_g = v34[v34["symbol"] == sym].reset_index(drop=True)
        r_g = rule[rule["symbol"] == sym].reset_index(drop=True)
        # Detect overlaps
        matched_v = set()
        matched_r = set()
        for vi, vt in v_g.iterrows():
            for ri, rt in r_g.iterrows():
                # Overlap check
                if vt["entry_date"] <= rt["exit_date"] and rt["entry_date"] <= vt["exit_date"]:
                    pairs.append({
                        "symbol": sym,
                        "v34_entry": vt["entry_date"],
                        "v34_exit": vt["exit_date"],
                        "v34_entry_price": vt.get("entry_price", np.nan),
                        "v34_exit_price": vt.get("exit_price", np.nan),
                        "v34_pnl": vt["pnl_pct"],
                        "v34_exit_reason": vt["exit_reason"],
                        "rule_entry": rt["entry_date"],
                        "rule_exit": rt["exit_date"],
                        "rule_entry_price": rt["entry_price"],
                        "rule_exit_price": rt["exit_price"],
                        "rule_pnl": rt["pnl_pct"],
                        "entry_lag_days": (vt["entry_date"] - rt["entry_date"]).days,
                        "exit_lag_days": (vt["exit_date"] - rt["exit_date"]).days,
                        "entry_price_diff_pct": (vt.get("entry_price", np.nan) - rt["entry_price"]) / rt["entry_price"] * 100 if rt["entry_price"] > 0 else np.nan,
                        "pnl_diff": vt["pnl_pct"] - rt["pnl_pct"],
                    })
                    matched_v.add(vi)
                    matched_r.add(ri)
        # Track missed (rule trades with no v34 overlap)
        for ri, rt in r_g.iterrows():
            if ri not in matched_r:
                pairs.append({
                    "symbol": sym,
                    "v34_entry": pd.NaT, "v34_exit": pd.NaT,
                    "v34_entry_price": np.nan, "v34_exit_price": np.nan,
                    "v34_pnl": np.nan, "v34_exit_reason": "MISSED",
                    "rule_entry": rt["entry_date"], "rule_exit": rt["exit_date"],
                    "rule_entry_price": rt["entry_price"], "rule_exit_price": rt["exit_price"],
                    "rule_pnl": rt["pnl_pct"],
                    "entry_lag_days": np.nan, "exit_lag_days": np.nan,
                    "entry_price_diff_pct": np.nan,
                    "pnl_diff": -rt["pnl_pct"],
                })
    return pd.DataFrame(pairs)


def classify(row) -> str:
    if row["v34_exit_reason"] == "MISSED":
        return "MISSED_WAVE"
    flags = []
    if pd.notna(row["entry_lag_days"]) and row["entry_lag_days"] > 3:
        if pd.notna(row["entry_price_diff_pct"]) and row["entry_price_diff_pct"] > 2:
            flags.append("LATE_ENTRY_WORSE_PRICE")
        else:
            flags.append("LATE_ENTRY")
    elif pd.notna(row["entry_lag_days"]) and row["entry_lag_days"] < -3:
        flags.append("EARLY_ENTRY")
    if pd.notna(row["exit_lag_days"]):
        if row["exit_lag_days"] < -3 and row["pnl_diff"] < -2:
            flags.append("EARLY_EXIT_WORSE")
        elif row["exit_lag_days"] > 3 and row["pnl_diff"] < -2:
            flags.append("LATE_EXIT_WORSE")
    if not flags:
        flags.append("OK")
    return "+".join(flags)


def main():
    out_dir = ROOT / "analysis" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trades + OHLCV ...")
    v34 = load_v34()
    rule = load_rule()
    data_dir = resolve_data_dir("../portable_data/vn_stock_ai_dataset_cleaned")
    loader = DataLoader(data_dir)

    # Restrict to the 12 worst symbols + DCM (per A1) to keep it fast
    worst = pd.read_csv(out_dir / "v34_vs_rule_symbol.csv").sort_values("comp_diff").head(20)["symbol"].tolist()
    focus_syms = sorted(set(worst + ["DCM", "PVS", "GAS", "PLX", "BCM", "PVD", "OCB", "EIB"]))
    print(f"Focus symbols ({len(focus_syms)}): {focus_syms}")

    v34 = v34[v34["symbol"].isin(focus_syms)].copy()
    rule = rule[rule["symbol"].isin(focus_syms)].copy()
    print(f"V34 trades: {len(v34)}, Rule trades: {len(rule)}")

    print("Attaching V34 prices ...")
    v34 = attach_prices(v34, loader, "v34")

    print("Matching same-wave pairs ...")
    pairs = match_same_wave(v34, rule)
    pairs["pattern"] = pairs.apply(classify, axis=1)

    out_path = out_dir / "wave_match.csv"
    pairs.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[OK] Wrote {out_path} ({len(pairs)} pairs)")

    # Pattern breakdown
    print("\n=== Pattern breakdown (across focus symbols) ===")
    pat = pairs["pattern"].value_counts()
    print(pat.to_string())

    # Examples per pattern
    for pattern in ["MISSED_WAVE", "LATE_ENTRY_WORSE_PRICE", "EARLY_EXIT_WORSE", "LATE_EXIT_WORSE"]:
        sub = pairs[pairs["pattern"].str.contains(pattern, na=False)].sort_values("pnl_diff").head(15)
        if len(sub):
            print(f"\n=== Top 15 {pattern} (by worst pnl_diff) ===")
            cols = ["symbol", "rule_entry", "rule_pnl", "v34_entry", "v34_pnl",
                    "entry_lag_days", "entry_price_diff_pct", "exit_lag_days", "pnl_diff"]
            print(sub[cols].to_string(index=False))

    # DCM specifically
    print("\n=== DCM all pairs ===")
    dcm = pairs[pairs["symbol"] == "DCM"].sort_values("rule_entry")
    cols = ["rule_entry", "rule_entry_price", "rule_pnl", "v34_entry", "v34_entry_price", "v34_pnl",
            "entry_lag_days", "entry_price_diff_pct", "pattern"]
    print(dcm[cols].to_string(index=False))


if __name__ == "__main__":
    main()
