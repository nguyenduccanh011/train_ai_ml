import sqlalchemy as sa, json
e = sa.create_engine("postgresql+psycopg2://stockml:stockml_dev@localhost:5433/stockml")
with e.connect() as c:
    rows = list(c.execute(sa.text(
        "select r.run_name, n.cagr_adv, n.nav_adv, r.trades, t.engine_config "
        "from leaderboard_runs r join leaderboard_nav n on n.run_id=r.run_id "
        "left join strategy_templates t on t.name=r.run_name "
        "where r.run_name like 'xh_%' and r.superseded=false order by n.cagr_adv desc")))
    for r in rows:
        m = r._mapping
        eng = m["engine_config"]
        eng = json.loads(eng) if isinstance(eng, str) else (eng or {})
        g = lambda k: eng.get(k)
        print(f"{m['run_name']:22} CAGR={m['cagr_adv']:.3f} NAV={m['nav_adv']:.2f} tr={m['trades']:>4} | "
              f"skipMA={g('signal_exit_skip_if_mkt_above_ma')} marg={g('signal_exit_skip_if_mkt_margin')} "
              f"wo={g('signal_exit_skip_if_mkt_winner_only')} mh={g('max_hold_bars')} "
              f"oxt={g('overext_trail_pct')} snrT={g('exit_snr_extend_threshold')} "
              f"snrG={g('exit_snr_min_gain')} tstop={g('trailing_stop_pct')}")
    print(f"\ntotal xh_ variants: {len(rows)}")
