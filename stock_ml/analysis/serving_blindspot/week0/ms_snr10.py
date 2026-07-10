"""Week-0 multi-seed validation of w0_snr_10 (template 2677, existing — NO clone).

2677 = champion 2646 + exit_snr_extend_threshold=1.0 window=20 min_gain=0.27.
Seed-42 already done (731.8 vs 729.6). Here: seeds 7, 99, 555, 123.
Each seed runs in a child subprocess with a 600s kill-guard.
DOES NOT touch 2646, DOES NOT clone, DOES NOT supersede anything.

Usage: python stock_ml/analysis/serving_blindspot/week0/ms_snr10.py [seed ...]
       (default: 7 99 555 123)
Child mode (internal): ms_snr10.py --child <seed>
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

TMPL = 2677  # w0_snr_10 — exists, never cloned/modified here
NAME = "w0_snr_10"
KILL_GUARD_S = 600
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")

# Champion 2646 per-seed baselines (fresh this session, from ms_CONTROL / audit logs)
CHAMP = {
    42:  {"composite": 729.6, "total_pnl": 127.25301810134556, "pf": 5.963310002880663, "mdd": 0.17454927579356125, "trades": 1384},
    7:   {"composite": 731.5, "total_pnl": 127.97968159718411, "pf": 5.998602169225148, "mdd": 0.17866099093461837, "trades": 1381},
    99:  {"composite": 722.5, "total_pnl": 126.57375397180982, "pf": 5.8241312481879115, "mdd": 0.1823136003933448, "trades": 1392},
    555: {"composite": 730.4, "total_pnl": 127.93968971584329, "pf": 5.941658453335072, "mdd": 0.18018071750724832, "trades": 1386},
    123: {"composite": 728.3, "total_pnl": 127.93042870877923, "pf": 5.88642233345499, "mdd": 0.17958977520094893, "trades": 1382},
}
# 2677 seed-42, already run earlier this session (leaderboard row verified)
SNR10_S42 = {"composite": 731.8, "total_pnl": 127.88436825628057, "pf": 6.003758950623839, "mdd": 0.17454927579356125, "trades": 1377}


def read_row(run_id: str):
    import psycopg2
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone()
    con.close()
    return r


def child(seed: int) -> None:
    from stock_ml.scripts.run_template import run_template_experiment
    t0 = time.time()
    r = run_template_experiment(template_id=TMPL, seed=seed)
    dt = time.time() - t0
    row = read_row(r.get("run_id"))
    rec = {"name": NAME, "template_id": TMPL, "seed": seed, "run_id": r.get("run_id"),
           "runtime_s": round(dt, 1),
           "composite": float(row[0]) if row[0] is not None else None,
           "total_pnl": float(row[1]) if row[1] is not None else None,
           "pf": float(row[2]) if row[2] is not None else None,
           "mdd": float(row[3]) if row[3] is not None else None,
           "trades": int(row[4]) if row[4] is not None else None}
    print("MS_SNR10_ROW " + json.dumps(rec), flush=True)


def main() -> None:
    seeds = [int(a) for a in sys.argv[1:]] or [7, 99, 555, 123]
    results = {42: dict(SNR10_S42)}
    for seed in seeds:
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(seed)],
                           cwd=str(REPO), capture_output=True, text=True, timeout=KILL_GUARD_S)
        line = next((l for l in p.stdout.splitlines() if l.startswith("MS_SNR10_ROW ")), None)
        if p.returncode != 0 or line is None:
            print(f"SEED {seed} FAILED rc={p.returncode}\n--- tail stdout ---\n"
                  + "\n".join(p.stdout.splitlines()[-15:])
                  + "\n--- tail stderr ---\n" + "\n".join(p.stderr.splitlines()[-15:]), flush=True)
            raise SystemExit(1)
        print(line, flush=True)
        results[seed] = json.loads(line[len("MS_SNR10_ROW "):])

    # ---- comparison table ----
    hdr = f"{'seed':>5} {'2677 comp':>10} {'2646 comp':>10} {'d_comp':>7} {'d_pnl':>7} {'2677 mdd':>9} {'2646 mdd':>9} {'d_tr':>5} {'tr%':>6}"
    print("\n" + hdr)
    print("-" * len(hdr))
    order = [42, 7, 99, 555, 123]
    for s in order:
        if s not in results:
            continue
        v, c = results[s], CHAMP[s]
        dtr = v["trades"] - c["trades"]
        print(f"{s:>5} {v['composite']:>10.1f} {c['composite']:>10.1f} {v['composite']-c['composite']:>+7.1f} "
              f"{v['total_pnl']-c['total_pnl']:>+7.2f} {v['mdd']:>9.4f} {c['mdd']:>9.4f} {dtr:>+5d} {100*dtr/c['trades']:>+6.2f}")
    inb = [s for s in (42, 7, 99, 555) if s in results]
    m_v = sum(results[s]["composite"] for s in inb) / len(inb)
    m_c = sum(CHAMP[s]["composite"] for s in inb) / len(inb)
    print(f"\nmean in-batch (42/7/99/555): 2677={m_v:.2f}  2646={m_c:.2f}  delta={m_v-m_c:+.2f}")
    if 123 in results:
        print(f"out-of-batch seed 123:       2677={results[123]['composite']:.1f}  2646={CHAMP[123]['composite']:.1f}  "
              f"delta={results[123]['composite']-CHAMP[123]['composite']:+.1f}")
    out = Path(__file__).with_suffix(".results.json")
    out.write_text(json.dumps({str(k): v for k, v in results.items()}, indent=2))
    print(f"results -> {out}")
    print("MS_SNR10_DONE")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        child(int(sys.argv[2]))
    else:
        main()
