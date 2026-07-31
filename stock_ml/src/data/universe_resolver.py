"""Point-in-time universe resolver (docs/UPGRADE_DYNAMIC_UNIVERSE.md, Bước 2).

Resolves the tradable universe of each walk-forward test-year CAUSALLY: the top-N equities by
prior-year liquidity, using ONLY data strictly before the test year. Indices and derivatives are
excluded; no symbol is force-included (fully dynamic).

Liquidity is measured at the SOURCE (Siêu Tín Hiệu ``/symbols/universe``, ``basis=matched``) — NOT
from the local back-adjusted OHLCV cache. This is the §13.9 fix: the cache-SQL version ranked by
``avg(volume*close)`` (back-adjust distorts price but not volume, block-trades inflate it up to 20×)
and gated by ``count(*)`` (22% of bars are volume=0 ghost sessions), so 11-31%/yr of the universe
were already-dead symbols. The endpoint ranks by matched average daily traded value and gates by
``min_sessions_traded`` (real matched sessions), point-in-time & append-only (stable per ``as_of``).

The lookback-min ("sustained liquidity"), hysteresis and sticky-drop incumbent logic are unchanged;
only the per-window (metric, session-count) numbers now come from the correct source.
"""

from __future__ import annotations

import re

# Indices + derivatives are not tradable equities -> excluded from the pool.
NONSTOCK_EXACT = {"VNINDEX", "HNX30", "VN30", "HNXINDEX", "UPINDEX", "VNXALL"}
_DERIVATIVE_RE = re.compile(r"F\d+M$")


def is_nonstock(symbol: str) -> bool:
    return (
        symbol in NONSTOCK_EXACT
        or bool(_DERIVATIVE_RE.search(symbol))
        or symbol.startswith("VN30F")
        or symbol.startswith("VN100F")
    )


def parse_universe_policy_slug(slug: str | None) -> dict | None:
    """Interim §7.1 persistence: the template's `universe_slug` string IS the policy —
    DB-persisted, so a leaderboard row reproduces from the DB alone.
    Format: 'dyn_topn:n=400,metric=adv,lookback=prior_year,min_sessions=100'.
    Any other slug (registered static universes) -> None. Unknown keys fail loud."""
    if not slug or not slug.startswith("dyn_topn:"):
        return None
    policy: dict = {"mode": "dynamic_topn"}
    for part in slug.split(":", 1)[1].split(","):
        k, _, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if k in ("n", "min_sessions"):
            policy[k] = int(v)
        elif k in ("hysteresis", "min_adv_ty", "sticky_drop"):
            policy[k] = float(v)
        elif k in ("metric", "lookback"):
            policy[k] = v
        else:
            raise ValueError(f"unknown universe_slug policy key: {k!r} in {slug!r}")
    if "n" not in policy:
        raise ValueError(f"universe_slug policy missing n: {slug!r}")
    return policy


_LOOKBACK_YEARS = {"prior_year": 1, "prior_2y": 2, "prior_3y": 3}


# Trailing-session window per lookback year. The server caps `sessions` at 250 ≈ one trading year,
# so each "prior year" is one endpoint call with a 250-session window ending just after that year.
_WINDOW_SESSIONS = 250


def _asof_for(test_year: int, j: int) -> str:
    """`as_of` whose trailing 250-session window ≈ calendar year (test_year - j).

    j=1 -> the year immediately before the test year; j=2 -> two years before, etc. A window ending
    early-January of year Z covers ≈ all of Z-1 (probe: as_of 2021-01-04, sessions 250 -> 2020-01-07
    .. 2021-01-04). The server snaps `as_of` down to the last real session on/before the date.
    """
    return f"{test_year - j + 1}-01-05"


def resolve_universes(
    policy: dict, test_years: list[int], duck: str | None = None
) -> dict[int, list[str]]:
    """Resolve the per-fold universe ONCE for every test-year of a run, from the source (§13.9).

    The returned dict is shared by the union-load (requested symbols = union of all fold lists) and
    the splitter (each fold masked down to its own year's list). ``duck`` is accepted for call-site
    compatibility but UNUSED — liquidity now comes from ``/symbols/universe`` (the local cache is the
    OHLCV store used elsewhere, not the universe measuring stick).

    policy keys (all causal — only data < test_year):
      n            top-N cap (required; set large for floor-only policies)
      metric       'adv' — matched average daily traded value (basis=matched)
      lookback     'prior_year' (default) | 'prior_2y' | 'prior_3y' — the metric is the MIN of the
                   per-year matched-ADTVs over the k prior years ("sustained liquidity": rank by the
                   WEAKEST year — a one-year volume bubble cannot buy a slot, and a symbol must have
                   traded min_sessions in EVERY lookback year). k=1 reduces to plain prior-year ADTV.
      min_sessions min real matched sessions required in each lookback year (default 100)
      min_adv_ty   absolute ADTV floor in tỷ VND (billions); None=off
      hysteresis   float k (e.g. 1.5): an incumbent keeps its slot while its current rank < N*k; freed
                   slots fill by rank. Universe stays capped at N. First test-year = plain top-N. 0=off.
      sticky_drop  float k (e.g. 3.0): ADDITIVE universe — every year the current top-N joins,
                   incumbents STAY until "really bad" (current rank >= N*k, or no longer passing
                   min_sessions/lookback at all). UNCAPPED. Mutually exclusive with hysteresis. 0=off.
    """
    mode = policy.get("mode", "dynamic_topn")
    if mode != "dynamic_topn":
        raise ValueError(f"unsupported universe_policy mode: {mode!r}")
    metric = policy.get("metric", "adv")
    if metric != "adv":
        raise ValueError(f"unsupported universe_policy metric: {metric!r}")
    lookback = policy.get("lookback", "prior_year")
    lb = _LOOKBACK_YEARS.get(lookback)
    if lb is None:
        raise ValueError(f"unsupported universe_policy lookback: {lookback!r}")
    n = int(policy["n"])
    min_sessions = int(policy.get("min_sessions", 100))
    min_adv_ty = policy.get("min_adv_ty")
    hyst = float(policy.get("hysteresis") or 0.0)
    sticky = float(policy.get("sticky_drop") or 0.0)
    if hyst and sticky:
        raise ValueError("universe_policy: hysteresis and sticky_drop are mutually exclusive")

    from src.data.sieutinhieu import fetch_universe

    test_years = sorted(test_years)
    # One batched call for every prior-year window across all folds (append-only source -> stable).
    all_asof = sorted({_asof_for(y, j) for y in test_years for j in range(1, lb + 1)})
    ranked = fetch_universe(
        all_asof, sessions=_WINDOW_SESSIONS, min_sessions_traded=min_sessions, basis="matched"
    )
    # as_of -> {symbol: matched ADTV (VND)}. Presence == passed the min_sessions_traded gate.
    adtv: dict[str, dict[str, float]] = {
        a: {r["symbol"]: float(r["adtv_value"]) for r in rows} for a, rows in ranked.items()
    }

    out: dict[int, list[str]] = {}
    prev: set[str] | None = None
    for year in test_years:
        windows = [_asof_for(year, j) for j in range(1, lb + 1)]
        common = set.intersection(*(set(adtv[w]) for w in windows))  # traded every lookback year
        metric_by_sym = {s: min(adtv[w][s] for w in windows) for s in common}
        if min_adv_ty is not None:
            floor = float(min_adv_ty) * 1e9  # tỷ VND -> VND (adtv_value is raw VND)
            metric_by_sym = {s: a for s, a in metric_by_sym.items() if a >= floor}
        ranked_syms = [s for s, _a in sorted(metric_by_sym.items(), key=lambda kv: (-kv[1], kv[0]))
                       if not is_nonstock(s)]
        if hyst and prev is not None:
            keep_rank = int(n * hyst)
            incumbents = [s for i, s in enumerate(ranked_syms) if s in prev and i < keep_rank]
            inc_set = set(incumbents)
            newcomers = [s for s in ranked_syms if s not in inc_set]
            out[year] = (incumbents + newcomers)[:n]
        elif sticky and prev is not None:
            drop_rank = int(n * sticky)
            keep = [s for i, s in enumerate(ranked_syms) if s in prev and i < drop_rank]
            keep_set = set(keep)
            out[year] = keep + [s for s in ranked_syms[:n] if s not in keep_set]
        else:
            out[year] = ranked_syms[:n]
        prev = set(out[year])
    return out
