import pandas as pd
import numpy as np

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

T = pd.read_csv(r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/trades_metrics.csv",
                parse_dates=["entry_date", "exit_date", "signal_date"])
print("closed trades:", len(T))
print("net sum pnl:", T.pnl_pct.sum())
gross_profit = T.loc[T.pnl_pct > 0, "pnl_pct"].sum()
gross_loss = T.loc[T.pnl_pct < 0, "pnl_pct"].sum()
print(f"gross_profit={gross_profit:.2f}u  gross_loss={gross_loss:.2f}u  PF={gross_profit/-gross_loss:.3f}")

for thr in (-0.10, -0.15, -0.20):
    B = T[T.pnl_pct <= thr]
    print(f"\n===== pnl <= {thr:.0%}: n={len(B)}  sum={B.pnl_pct.sum():.2f}u "
          f"({B.pnl_pct.sum()/gross_profit:.2%} of gross profit; {B.pnl_pct.sum()/T.pnl_pct.sum():.2%} of net)")
    print("exit_reason:", B.exit_reason.value_counts().to_dict())
    print("year(entry):", B.entry_date.dt.year.value_counts().sort_index().to_dict())
    print("year(exit):", B.exit_date.dt.year.value_counts().sort_index().to_dict())
    print("month(exit) top6:", B.exit_date.dt.strftime("%Y-%m").value_counts().head(6).to_dict())
    print(f"holding_days: mean={B.holding_days.mean():.1f} median={B.holding_days.median():.0f} "
          f"p90={B.holding_days.quantile(.9):.0f} max={B.holding_days.max()}")
    print(f"wait_bars: mean={B.wait_bars.mean():.1f} median={B.wait_bars.median():.0f} "
          f">=20 bars: {(B.wait_bars>=20).mean():.1%}  ==40: {(B.wait_bars>=40).mean():.1%}")
    print(f"hot_run share: {B.hot_run.mean():.1%} (all trades {T.hot_run.mean():.1%})")
    print(f"touch_bar_red share: {B.touch_bar_red.mean():.1%} (all trades {T.touch_bar_red.mean():.1%})")
    print(f"mae_pct: mean={B.mae_pct.mean():.3f} median={B.mae_pct.median():.3f} min={B.mae_pct.min():.3f}")
    print(f"mfe_pct: mean={B.mfe_pct.mean():.3f}  share mfe<3%: {(B.mfe_pct<0.03).mean():.1%} (never worked)")
    lateness = B.pnl_pct - B.mae_pct   # realized above trough; ~0 => sold AT the bottom
    print(f"lateness (pnl - mae): mean={lateness.mean():.3f} median={lateness.median():.3f}; "
          f"share exited within 3pp of trough: {(lateness<=0.03).mean():.1%}")
    print(f"post_ret21 after exit: mean={B.post_ret21.mean():.3f} median={B.post_ret21.median():.3f}; "
          f"share bounced >+5% in 21b: {(B.post_ret21>0.05).mean():.1%}")
    print(f"pre_ret20 at signal: mean={B.pre_ret20.mean():.3f}; signal_to_fill_drop mean={B.signal_to_fill_drop.mean():.3f}")
    print(f"spans_bad_adjustment: {int(B.spans_bad_adjustment.sum())}")

# worst singles
W = T.nsmallest(20, "pnl_pct")[["symbol","entry_date","exit_date","signal_date","pnl_pct","mae_pct","mfe_pct",
                                "holding_days","wait_bars","exit_reason","hot_run","touch_bar_red","pre_ret20","post_ret21"]]
print("\n===== worst 20 trades =====")
print(W.to_string(index=False))
w10 = T.nsmallest(10, "pnl_pct").pnl_pct.sum(); w20 = T.nsmallest(20, "pnl_pct").pnl_pct.sum()
print(f"\nworst10 sum={w10:.2f}u = {w10/gross_profit:.2%} of gross profit = {w10/T.pnl_pct.sum():.2%} of net (+{T.pnl_pct.sum():.1f}u)")
print(f"worst20 sum={w20:.2f}u = {w20/gross_profit:.2%} of gross profit = {w20/T.pnl_pct.sum():.2%} of net")
print(f"largest single loss: {T.pnl_pct.min():.4f} ({T.loc[T.pnl_pct.idxmin(),'symbol']} "
      f"{T.loc[T.pnl_pct.idxmin(),'entry_date'].date()} -> {T.loc[T.pnl_pct.idxmin(),'exit_date'].date()})")
print("count <= -20%:", (T.pnl_pct <= -0.20).sum(), " | count <= -25%:", (T.pnl_pct <= -0.25).sum(),
      " | count <= -30%:", (T.pnl_pct <= -0.30).sum())

# how many big losses had MAE beyond -20/-30 intra-trade
for thr in (-0.15,):
    B = T[T.pnl_pct <= thr]
    print(f"\nbig_loss (<= -15%) MAE tail: mae<=-20%: {(B.mae_pct<=-0.20).sum()}, mae<=-25%: {(B.mae_pct<=-0.25).sum()}, mae<=-30%: {(B.mae_pct<=-0.30).sum()}")
# whole book MAE tail (risk exposure even on trades that recovered)
print(f"ALL closed trades MAE tail: mae<=-15%: {(T.mae_pct<=-0.15).sum()}, mae<=-20%: {(T.mae_pct<=-0.20).sum()}, mae<=-30%: {(T.mae_pct<=-0.30).sum()}")
rec = T[(T.mae_pct <= -0.15) & (T.pnl_pct > -0.10)]
print(f"trades that dipped <=-15% MAE but recovered to > -10% final: n={len(rec)} sum_pnl={rec.pnl_pct.sum():.2f}u winners among them: {(rec.pnl_pct>0).sum()} (+{rec.loc[rec.pnl_pct>0,'pnl_pct'].sum():.2f}u)")
