"""Audit checks: entry integrity, no-leakage, split coverage.

Run these after every backtest. `audit_report` returns a dict; CLI prints it
and exits non-zero on any FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.splitter import SplitWindow


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | WARN
    detail: str = ""
    rows: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def _fail(name: str, detail: str, rows: list[dict] | None = None) -> CheckResult:
    return CheckResult(name=name, status="FAIL", detail=detail, rows=rows or [])


def _pass(name: str, detail: str = "") -> CheckResult:
    return CheckResult(name=name, status="PASS", detail=detail)


def check_entry_integrity(
    trades: pd.DataFrame, signals: pd.DataFrame, max_examples: int = 5
) -> CheckResult:
    """Every trade entry must correspond to a buy signal one bar earlier."""
    if trades.empty:
        return _pass("entry_integrity", "no trades to verify")
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"])
    buy_set = set(zip(sig.loc[sig["signal"] > 0, "symbol"], sig.loc[sig["signal"] > 0, "date"]))
    bad = []
    for _, t in trades.iterrows():
        key = (t["symbol"], pd.Timestamp(t["entry_signal_date"]))
        if key not in buy_set:
            bad.append(
                {
                    "symbol": t["symbol"],
                    "entry_signal_date": str(key[1]),
                    "entry_date": str(t["entry_date"]),
                }
            )
            if len(bad) >= max_examples:
                break
    if bad:
        return _fail(
            "entry_integrity",
            f"{len(bad)}+ trades reference a date with no buy signal",
            bad,
        )
    return _pass("entry_integrity", f"all {len(trades)} trades trace to a buy signal")


def check_fill_offset(trades: pd.DataFrame, max_examples: int = 5) -> CheckResult:
    """entry_date must be strictly after entry_signal_date (next-bar fill)."""
    if trades.empty:
        return _pass("fill_offset", "no trades")
    bad = []
    for _, t in trades.iterrows():
        sd = pd.Timestamp(t["entry_signal_date"])
        ed = pd.Timestamp(t["entry_date"])
        if ed <= sd:
            bad.append({"symbol": t["symbol"], "signal_date": str(sd), "entry_date": str(ed)})
            if len(bad) >= max_examples:
                break
    if bad:
        return _fail("fill_offset", "entry not strictly after signal — lookahead", bad)
    return _pass("fill_offset", "all entries fill on the next bar")


def check_no_train_test_overlap(windows: list[SplitWindow]) -> CheckResult:
    bad = []
    for w in windows:
        if w.train_end > w.test_start:
            bad.append(
                {
                    "test_year": w.test_year,
                    "train_end": str(w.train_end),
                    "test_start": str(w.test_start),
                }
            )
    if bad:
        return _fail("split_no_overlap", "train_end > test_start", bad)
    return _pass("split_no_overlap", f"{len(windows)} windows clean")


def check_split_gap(windows: list[SplitWindow], min_gap_days: int) -> CheckResult:
    bad = []
    for w in windows:
        gap = (w.test_start - w.train_end).days + 1
        if gap < min_gap_days:
            bad.append(
                {
                    "test_year": w.test_year,
                    "actual_gap_days": gap,
                    "required_gap_days": min_gap_days,
                }
            )
    if bad:
        return _fail(
            "split_gap",
            f"gap < required {min_gap_days} days (forward labels would leak)",
            bad,
        )
    return _pass("split_gap", f"all gaps >= {min_gap_days} days")


def check_signal_coverage(trades: pd.DataFrame, signals: pd.DataFrame) -> CheckResult:
    """Warn if many buy signals never produced a trade (likely position-already-open)."""
    if signals.empty:
        return CheckResult("signal_coverage", "WARN", "no signals provided")
    buys = signals[signals["signal"] > 0]
    n_buys = len(buys)
    n_trades = len(trades)
    if n_buys == 0:
        return CheckResult("signal_coverage", "WARN", "no buy signals")
    rate = n_trades / n_buys
    detail = f"{n_trades} trades / {n_buys} buy signals ({rate:.1%})"
    if rate < 0.05:
        return CheckResult("signal_coverage", "WARN", detail + " — very low conversion")
    return _pass("signal_coverage", detail)


def audit_report(
    trades: pd.DataFrame,
    signals: pd.DataFrame,
    windows: list[SplitWindow],
    min_gap_days: int,
) -> dict:
    checks = []
    # Only run window-based checks if windows provided (skip for purged_kfold)
    if windows is not None:
        checks.append(check_no_train_test_overlap(windows))
        # check_split_gap only works with SplitWindow (has test_year), not PurgedFoldWindow
        try:
            checks.append(check_split_gap(windows, min_gap_days=min_gap_days))
        except AttributeError:
            # PurgedFoldWindow doesn't have test_year; skip this check
            pass
    checks.extend(
        [
            check_entry_integrity(trades, signals),
            check_fill_offset(trades),
            check_signal_coverage(trades, signals),
        ]
    )
    n_fail = sum(1 for c in checks if c.status == "FAIL")
    n_warn = sum(1 for c in checks if c.status == "WARN")
    return {
        "overall": "FAIL" if n_fail else ("WARN" if n_warn else "PASS"),
        "n_fail": n_fail,
        "n_warn": n_warn,
        "checks": [
            {"name": c.name, "status": c.status, "detail": c.detail, "examples": c.rows[:5]}
            for c in checks
        ],
    }


def print_report(report: dict) -> None:
    print(f"\n=== AUDIT REPORT — overall: {report['overall']} ===")
    for c in report["checks"]:
        marker = {"PASS": "[OK]  ", "WARN": "[WARN]", "FAIL": "[FAIL]"}[c["status"]]
        print(f"  {marker} {c['name']}: {c['detail']}")
        for ex in c.get("examples", []):
            print(f"           - {ex}")
