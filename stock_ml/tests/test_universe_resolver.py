"""Unit tests for the point-in-time universe resolver (docs/UPGRADE_DYNAMIC_UNIVERSE.md §4 row 1).

The resolver now ranks by MATCHED ADTV from the source (§13.9); these tests mock that source
(``fetch_universe``) with synthetic per-``as_of`` liquidity so the Python logic — top-N, lookback-min,
hysteresis, sticky-drop, the equal-value tiebreak, index/derivative exclusion, and the ADTV floor —
is exercised deterministically without a network call. Each ``as_of`` "{Y}-01-05" is the window that
ranks the ≈calendar year (Y-1): the prior year of test-year Y.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.data.splitter import YearSplitter  # noqa: E402
from src.data.universe_resolver import (  # noqa: E402
    is_nonstock,
    parse_universe_policy_slug,
    resolve_universes,
)


@pytest.fixture
def fake_universe(monkeypatch):
    """Install a synthetic source. Call with ``{as_of_str: {symbol: adtv_value_vnd}}``; a symbol
    present at an ``as_of`` == it passed that window's min_sessions_traded gate."""

    def _install(by_asof: dict[str, dict[str, float]]) -> None:
        def _fetch(as_of, **_kw):
            dates = [as_of] if isinstance(as_of, str) else list(as_of)
            return {
                d: [{"symbol": s, "adtv_value": v, "sessions_traded": 250}
                    for s, v in by_asof.get(d, {}).items()]
                for d in dates
            }

        import src.data.sieutinhieu as sth

        monkeypatch.setattr(sth, "fetch_universe", _fetch)

    return _install


def test_topn_prior_year_and_filters(fake_universe):
    # Window for 2024 = as_of 2024-01-05. Ranking BIG>MID>SMALL; VNINDEX/VN30F1M present but are
    # non-stock and must be dropped; THIN/FUTURE simply absent (failed the gate / not point-in-time).
    fake_universe({"2024-01-05": {"BIG": 10e9, "MID": 5e9, "SMALL": 1e9,
                                  "VNINDEX": 99e9, "VN30F1M": 99e9}})
    out = resolve_universes({"mode": "dynamic_topn", "n": 2, "metric": "adv",
                             "lookback": "prior_year"}, [2024])
    assert out == {2024: ["BIG", "MID"]}


def test_deterministic_including_adv_ties(fake_universe):
    fake_universe({"2024-01-05": {"Z": 2e9, "X": 2e9, "Y": 2e9}})
    p = {"mode": "dynamic_topn", "n": 2}
    assert resolve_universes(p, [2024]) == resolve_universes(p, [2024]) == {2024: ["X", "Y"]}


def test_multi_year_resolves_each_fold_from_its_own_prior_year(fake_universe):
    fake_universe({"2023-01-05": {"OLD": 9e9}, "2024-01-05": {"NEW": 9e9}})
    out = resolve_universes({"mode": "dynamic_topn", "n": 5}, [2023, 2024])
    assert out == {2023: ["OLD"], 2024: ["NEW"]}


def test_lookback_2y_smooths_one_year_bubble(fake_universe):
    # BUBBLE huge only in the recent window, quiet the year before; STEADY moderate in both.
    fake_universe({
        "2024-01-05": {"BUBBLE": 10e9, "STEADY": 5e9},   # ≈2023
        "2023-01-05": {"BUBBLE": 0.1e9, "STEADY": 5e9},  # ≈2022
    })
    p1 = resolve_universes({"mode": "dynamic_topn", "n": 1}, [2024])
    p2 = resolve_universes({"mode": "dynamic_topn", "n": 1, "lookback": "prior_2y"}, [2024])
    assert p1 == {2024: ["BUBBLE"]}          # prior_year sees only the bubble
    assert p2 == {2024: ["STEADY"]}          # prior_2y takes the MIN -> the quiet year wins


def test_hysteresis_keeps_incumbent(fake_universe):
    fake_universe({
        "2023-01-05": {"AAA": 10e9, "BBB": 1e9},   # prior of 2023: A leads
        "2024-01-05": {"AAA": 2e9, "BBB": 10e9},   # prior of 2024: B overtakes, A slips to rank 2
    })
    off = resolve_universes({"mode": "dynamic_topn", "n": 1}, [2023, 2024])
    on = resolve_universes({"mode": "dynamic_topn", "n": 1, "hysteresis": 2.0}, [2023, 2024])
    assert off == {2023: ["AAA"], 2024: ["BBB"]}    # churn: A swapped out
    assert on == {2023: ["AAA"], 2024: ["AAA"]}     # incumbent held (rank 2 < 1*2.0)


def test_sticky_additive_universe(fake_universe):
    fake_universe({
        "2023-01-05": {"AAA": 10e9, "BBB": 5e9, "CCC": 1e9},
        "2024-01-05": {"AAA": 5e9, "BBB": 10e9, "CCC": 1e9},
    })
    out = resolve_universes({"mode": "dynamic_topn", "n": 1, "sticky_drop": 3.0}, [2023, 2024])
    assert out == {2023: ["AAA"], 2024: ["AAA", "BBB"]}  # A kept (rank 2 < 3) AND B added, uncapped


def test_sticky_drops_collapsed_incumbent(fake_universe):
    # A fails the gate the second year (absent from that window) -> dropped even under sticky.
    fake_universe({
        "2023-01-05": {"AAA": 10e9, "BBB": 5e9},
        "2024-01-05": {"BBB": 5e9},
    })
    out = resolve_universes({"mode": "dynamic_topn", "n": 1, "sticky_drop": 5.0}, [2023, 2024])
    assert out == {2023: ["AAA"], 2024: ["BBB"]}


def test_min_adv_floor(fake_universe):
    # floor = 5 tỷ = 5e9 VND: BIG (10e9) passes, SMALL (1e9) filtered.
    fake_universe({"2024-01-05": {"BIG": 10e9, "SMALL": 1e9}})
    out = resolve_universes({"mode": "dynamic_topn", "n": 100, "min_adv_ty": 5.0}, [2024])
    assert out == {2024: ["BIG"]}


# ---- Splitter masking (unchanged: does not touch the source) -------------------------------------
def _panel(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, end=end)
    return pd.concat(
        [pd.DataFrame({"symbol": s, "date": dates, "close": 1.0}) for s in symbols],
        ignore_index=True,
    )


def test_splitter_masks_each_fold_to_its_own_universe():
    df = _panel(["AAA", "BBB"], "2021-01-01", "2024-12-31")
    sp = YearSplitter(train_years=2, test_years=1, gap_days=0,
                      first_test_year=2023, last_test_year=2024)
    uni = {2023: ["AAA"], 2024: ["AAA", "BBB"]}

    folds = {w.test_year: (tr, te) for w, tr, te in sp.split(df, universe_by_year=uni)}

    tr23, te23 = folds[2023]
    assert set(tr23["symbol"]) == {"AAA"} and set(te23["symbol"]) == {"AAA"}
    tr24, te24 = folds[2024]
    assert set(tr24["symbol"]) == {"AAA", "BBB"} and set(te24["symbol"]) == {"AAA", "BBB"}


def test_splitter_without_universe_is_unchanged():
    df = _panel(["AAA", "BBB"], "2021-01-01", "2024-12-31")
    sp = YearSplitter(train_years=2, test_years=1, gap_days=0,
                      first_test_year=2023, last_test_year=2024)

    legacy = [(w.test_year, len(tr), len(te)) for w, tr, te in sp.split(df)]
    none_arg = [(w.test_year, len(tr), len(te)) for w, tr, te in sp.split(df, universe_by_year=None)]

    assert legacy == none_arg
    assert {y for y, _a, _b in legacy} == {2023, 2024}


def test_parse_universe_policy_slug():
    assert parse_universe_policy_slug(None) is None
    assert parse_universe_policy_slug("vn61_standard") is None
    p = parse_universe_policy_slug("dyn_topn:n=400,metric=adv,lookback=prior_year,min_sessions=100")
    assert p == {"mode": "dynamic_topn", "n": 400, "metric": "adv",
                 "lookback": "prior_year", "min_sessions": 100}
    p2 = parse_universe_policy_slug("dyn_topn:n=61,lookback=prior_2y,hysteresis=1.5,min_adv_ty=10")
    assert p2 == {"mode": "dynamic_topn", "n": 61, "lookback": "prior_2y",
                  "hysteresis": 1.5, "min_adv_ty": 10.0}
    assert parse_universe_policy_slug("dyn_topn:n=150") == {"mode": "dynamic_topn", "n": 150}
    with pytest.raises(ValueError):
        parse_universe_policy_slug("dyn_topn:n=100,bogus=1")


def test_is_nonstock():
    for s in ["VNINDEX", "HNX30", "VN30", "HNXINDEX", "UPINDEX", "VNXALL",
              "VN30F1M", "VN30F2508", "VN100F1M", "ABCF1M"]:
        assert is_nonstock(s), s
    for s in ["FPT", "HPG", "VNM", "FIR"]:
        assert not is_nonstock(s), s
