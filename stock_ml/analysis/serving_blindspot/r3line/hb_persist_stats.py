# -*- coding: utf-8 -*-
"""Complete the champion detail page: compute yearly + symbol stats from the persisted combo trades and
persist them (RunYearlyStatRepository / RunSymbolStatRepository) so all detail-page tabs populate."""
from __future__ import annotations
import os, sys, asyncio
from pathlib import Path
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2, pandas as pd, numpy as np
from db.engine import async_engine
from db.repositories.yearly_stat_repo import RunYearlyStatRepository
from db.repositories.symbol_stat_repo import RunSymbolStatRepository
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
CHAMP_RID = "template/x2_struct_to_k16preempt_cssize-69338138"

con = psycopg2.connect(**PG)
tr = pd.read_sql("select symbol,entry_date,pnl_pct,holding_days from run_trades where run_id=%s", con, params=(CHAMP_RID,))
con.close()
tr["yr"] = pd.to_datetime(tr["entry_date"]).dt.year


def agg(g):
    p = g["pnl_pct"]; wins = p[p > 0]; loss = p[p <= 0]
    pf = float(wins.sum() / abs(loss.sum())) if loss.sum() != 0 else float(wins.sum())
    return dict(trades=int(len(g)), win_rate=float((p > 0).mean()), total_pnl=float(p.sum()),
                avg_pnl=float(p.mean()), med_pnl=float(p.median()), std_pnl=float(p.std() or 0),
                max_win=float(p.max()), max_loss=float(p.min()), profit_factor=pf,
                avg_hold=float(g["holding_days"].mean()))


yearly = []
for y, g in tr.groupby("yr"):
    d = agg(g); d["year"] = int(y); d["max_drawdown"] = 0.0; yearly.append(d)
symbol = []
for s, g in tr.groupby("symbol"):
    d = agg(g); d["symbol"] = s; symbol.append(d)
print(f"yearly rows={len(yearly)} symbol rows={len(symbol)}", flush=True)


async def go():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        yr = RunYearlyStatRepository(s); await yr.delete_by_run_id(CHAMP_RID)
        n1 = await yr.bulk_insert(CHAMP_RID, yearly)
        sr = RunSymbolStatRepository(s); await sr.delete_by_run_id(CHAMP_RID)
        n2 = await sr.bulk_insert(CHAMP_RID, symbol)
        await s.commit(); print(f"persisted yearly={n1} symbol={n2}", flush=True)
    await async_engine.dispose()

asyncio.run(go())
print("STATS_DONE")
