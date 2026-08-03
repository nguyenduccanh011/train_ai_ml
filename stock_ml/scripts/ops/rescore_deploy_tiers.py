"""rescore_deploy_tiers: re-score the 6 PRODUCTION dyn deploy tiers onto the DECLARED 1477 panel
under their OWN per-tier configs (NOT the board reference config), writing the official
`overlay/_dyn*` leaderboard_nav rows.

These tiers (serving/tiers.yaml) each = a base bundle + a PortfolioConstants override; they hold
no BASE trades of their own — base + signals come from the parent engine run. register_overlay's
CLI only expresses (k, liqcol, date_lo), so the w_invvol / liqcol-OFF tiers need this dedicated
scorer. Numbers land on the SAME 1477 panel + fingerprint as the rest of the board (overlay_panel_fp
uniform), so the production rows stop mixing panels. Engine parity backtest≡serving is proven
separately (DEPLOY_DYN_TIERS.md §7.7 champion parity, 6 decimals); this only rebases the DISPLAYED
train numbers. The §5 store-901 numbers are printed alongside for the panel-shift delta.

Run: python stock_ml/scripts/ops/rescore_deploy_tiers.py            # all 6
     python stock_ml/scripts/ops/rescore_deploy_tiers.py --dry-run  # score + print, no DB write
"""

from __future__ import annotations

import argparse
import time

import psycopg2

from stock_ml.db.overlay_persist import persist_overlay
from stock_ml.db.overlay_scoring import (
    default_context,
    load_run_frames,
    overlay_config_hash,
    panel_fingerprint,
    pg_dsn,
)
from stock_ml.portfolio import PortfolioConstants, build_panel_bundle, run_portfolio

_DYN300 = "template/_dyn300_onerun-69338138"
_DYN900 = "template/_dyn900_onerun-69338138"

# Authoritative per-tier config — mirror of serving/tiers.yaml `constants` (override on
# PortfolioConstants defaults: k=10, stat_mode=causal, date_lo=2020-01-01, liqcol OFF). `ref901`
# = the 29/07 store-901 number (DEPLOY_DYN_TIERS.md §5) for the panel-shift delta.
TIERS = [
    dict(row="overlay/_dyn300_k10_floor5", base=_DYN300, ref901="180.4/-21.4",
         label="PROD vốn nhỏ — sàn ADV≥5 tỷ",
         constants=dict(tplus=2, liqcol_adv10_ty=5.0, liqcol_adv252_ty=0.0)),
    dict(row="overlay/_dyn300_k10_floor10", base=_DYN300, ref901="151.1/-24.6",
         label="PROD vốn vừa — sàn ADV≥10 tỷ",
         constants=dict(tplus=2, liqcol_adv10_ty=10.0, liqcol_adv252_ty=0.0)),
    dict(row="overlay/_dyn300_k10_liqcol", base=_DYN300, ref901="189.9/-22.8",
         label="Cân bằng — veto big-quiet 5 tỷ",
         constants=dict(tplus=2, liqcol_adv10_ty=5.0)),
    dict(row="overlay/_dyn300_k6_liqcol", base=_DYN300, ref901="203.8/-23.5",
         label="Tăng trưởng — K=6 + liqcol 5 tỷ",
         constants=dict(tplus=2, k=6, liqcol_adv10_ty=5.0)),
    dict(row="overlay/_dyn300_invvol15", base=_DYN300, ref901="186.7/-26.6",
         label="Diện rộng — inverse-vol 1.5 (no veto)",
         constants=dict(tplus=2, w_invvol=1.5)),
    dict(row="overlay/_dyn900_k16", base=_DYN900, ref901="248.4/-28.8",
         label="Nền tín hiệu — K=16 (penny, chỉ tham khảo)",
         constants=dict(tplus=2, k=16)),
]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="score + print, do not write the DB")
    args = ap.parse_args()

    ctx = default_context()  # asserts the declared-panel artifact identity (fail-loud)
    panel_fp = panel_fingerprint()
    con = psycopg2.connect(pg_dsn())

    # Cache panel bundles by _bundle_key (differs only in need_tv: the liqcol/floor tiers build the
    # traded-value history, invvol15/k16 do not) — build each at most once (~56s), reuse across tiers.
    bundles: dict = {}

    def bundle_for(C: PortfolioConstants) -> dict:
        need_tv = C.liqcol_adv10_ty is not None or C.w_liq_full_ty is not None
        if need_tv not in bundles:
            t0 = time.time()
            bundles[need_tv] = build_panel_bundle(ctx, C)
            print(f"  built panel bundle (need_tv={need_tv}) in {time.time() - t0:.0f}s", flush=True)
        return bundles[need_tv]

    print(f"panel_fp={panel_fp} | tiers={len(TIERS)}{' (DRY-RUN)' if args.dry_run else ''}\n", flush=True)
    print(f"{'tier':38} {'1477 CAGR/DD':>16}   {'§5 store-901':>14}   n_gated  off%", flush=True)
    for t in TIERS:
        try:
            C = PortfolioConstants(**t["constants"])
            # base + signals both from the parent; load_run_frames drops non-stock legs (ONE
            # chokepoint shared with the board — PORTFOLIO_WRITE_UNIFICATION_IMPL §5.1).
            base, sig = load_run_frames(con, t["base"])
            # deploy tiers always get the full detail tab -> emit_pending=True + persist detail=True
            r = run_portfolio(base, sig, ctx=ctx, C=C, bundle=bundle_for(C), emit_pending=True)
            note = (
                f"OVERLAY TIER {t['label']} | stock_ml.portfolio 0.4.4 causal T+{C.tplus} "
                f"k{C.k} {t['constants']} | panel 1477 fp {panel_fp} date_hi {ctx.date_hi} | base={t['base']}"
            )
            ch = overlay_config_hash(C, panel_fp)
            if not args.dry_run:
                persist_overlay(con, t["row"], r, C, ch, panel_fp, note, detail=True)
                con.commit()
        except Exception as exc:  # noqa: BLE001 — one bad tier must not kill the rest
            con.rollback()
            print(f"ERR {t['row']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(
            f"{t['row'].split('/')[-1]:38} {r['cagr'] * 100:7.1f}% / {r['maxdd'] * 100:5.1f}   "
            f"{t['ref901']:>14}   {r['n_gated']:4}/{r['n_base']:<4} {r['offpanel_frac'] * 100:3.0f}",
            flush=True,
        )
    con.close()
    print(f"\nXONG{' (dry-run, không ghi DB)' if args.dry_run else ''}.", flush=True)


if __name__ == "__main__":
    main()
