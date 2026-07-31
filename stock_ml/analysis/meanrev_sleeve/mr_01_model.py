"""ML mean-reversion sleeve — nguồn lợi trực giao với momentum champion.
Label: forward 8d return sau dip setup. Walk-forward LightGBM (purge overlap).
Daily OHLCV only, 1-slot/symbol, equal-weight, no-external. Prosecute >=2022.
"""
from __future__ import annotations
import duckdb, numpy as np, pandas as pd
import lightgbm as lgb

HOLD = 8
COST = 0.004
TOPQ = 0.80          # vào lệnh khi pred >= quantile 0.80 của train
DIP = dict(ret5=-0.05, rsi=45)   # setup mean-reversion

def load(n_univ=100):
    con = duckdb.connect('market_data/market.duckdb', read_only=True)
    liq = con.execute("""SELECT symbol FROM ohlcv WHERE timeframe='1D' AND date>='2021-01-01'
        GROUP BY symbol HAVING COUNT(*)>400 ORDER BY AVG(traded_value) DESC LIMIT ?""",[n_univ]).df()
    syms = tuple(liq['symbol'])
    oh = con.execute(f"""SELECT symbol,date,open,high,low,close,volume,traded_value
        FROM ohlcv WHERE timeframe='1D' AND symbol IN {syms} AND date>='2017-06-01'
        ORDER BY symbol,date""").df()
    con.close(); oh['date'] = pd.to_datetime(oh['date'])
    return oh

def feats(g):
    c,h,l,v = g['close'],g['high'],g['low'],g['volume']
    o = g['open']
    ma20,ma50,ma100 = c.rolling(20).mean(),c.rolling(50).mean(),c.rolling(100).mean()
    std20 = c.rolling(20).std()
    d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100-100/(1+up/(dn+1e-9))
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    ret1 = c.pct_change(); ret5 = c/c.shift(5)-1; ret10 = c/c.shift(10)-1
    downdays = (ret1<0).astype(int).groupby((ret1>=0).cumsum()).cumsum()
    dd20 = c/c.rolling(20).max()-1
    dd60 = c/c.rolling(60).max()-1
    lowwick = (np.minimum(o,c)-l)/(h-l+1e-9)
    rng = (h-l)/c
    volz = (v-v.rolling(20).mean())/(v.rolling(20).std()+1e-9)
    df = pd.DataFrame({
        'rsi':rsi, 'rsi_chg':rsi.diff(3),
        'dist_ma20':c/ma20-1, 'dist_ma50':c/ma50-1, 'dist_ma100':c/ma100-1,
        'bb_pct':(c-ma20)/(2*std20+1e-9), 'bb_width':(4*std20)/ma20,
        'ret1':ret1, 'ret5':ret5, 'ret10':ret10,
        'downdays':downdays, 'dd20':dd20, 'dd60':dd60,
        'atr_ratio':atr/c, 'lowwick':lowwick, 'rng':rng, 'volz':volz,
        'ma100_slope':ma100/ma100.shift(20)-1,
    })
    # label pnl thực (để chấm backtest) + label bounce (để train classifier)
    entry = c.shift(-1)
    fwd = c.shift(-1-HOLD)/entry - 1
    df['pnl'] = fwd - COST
    fmax = pd.concat([h.shift(-k)/entry-1 for k in range(1,HOLD+1)],axis=1).max(axis=1)
    fmin = pd.concat([l.shift(-k)/entry-1 for k in range(1,HOLD+1)],axis=1).min(axis=1)
    df['label'] = ((fmax>=0.04) & (fmin>-0.06)).astype(int)   # bật ≥4% mà không thủng −6% trước
    # setup: dip trong UPTREND dài hạn (close>MA100) — tách pullback lành khỏi bear
    df['setup'] = ((ret5<DIP['ret5']) | (rsi<DIP['rsi'])) & (c>ma100) & (ma100>ma100.shift(20))
    df['date'] = g['date'].values; df['close']=c.values; df['symbol']=g['symbol'].values
    return df

FEATCOLS = ['rsi','rsi_chg','dist_ma20','dist_ma50','dist_ma100','bb_pct','bb_width',
            'ret1','ret5','ret10','downdays','dd20','dd60','atr_ratio','lowwick','rng','volz','ma100_slope']

def build(oh):
    parts=[feats(g.sort_values('date')) for _,g in oh.groupby('symbol')]
    D = pd.concat(parts,ignore_index=True)
    D['yr'] = D['date'].dt.year
    return D

def walk_forward(D):
    D = D[D['setup']].dropna(subset=FEATCOLS+['label','pnl']).copy()
    trades=[]
    imp=None
    for ty in range(2022,2027):
        cut = pd.Timestamp(f'{ty}-01-01') - pd.Timedelta(days=HOLD+3)
        tr = D[D['date']<cut]; te = D[D['date'].dt.year==ty]
        if len(tr)<500 or len(te)<20: continue
        m = lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=31,
                               min_child_samples=40,subsample=0.8,colsample_bytree=0.8,
                               random_state=42,verbose=-1)
        m.fit(tr[FEATCOLS],tr['label'])
        thr = np.quantile(m.predict_proba(tr[FEATCOLS])[:,1],TOPQ)
        te = te.copy(); te['pred']=m.predict_proba(te[FEATCOLS])[:,1]
        sel = te[te['pred']>=thr]
        for _,r in sel.iterrows():
            trades.append((r['symbol'],r['date'],r['yr'],r['pnl']))  # pnl = net return thực
        fi = pd.Series(m.feature_importances_,index=FEATCOLS)
        imp = fi if imp is None else imp+fi
    T = pd.DataFrame(trades,columns=['symbol','date','yr','pnl'])
    # 1-slot/symbol: bỏ lệnh chồng cùng mã trong HOLD phiên
    T = T.sort_values(['symbol','date'])
    keep=[]; last={}
    for _,r in T.iterrows():
        ld=last.get(r['symbol'])
        if ld is None or (r['date']-ld).days>HOLD*1.6:
            keep.append(True); last[r['symbol']]=r['date']
        else: keep.append(False)
    T=T[keep]
    return T, imp

if __name__=='__main__':
    oh=load(100); D=build(oh)
    print(f"bars={len(D)} setup-bars={int(D['setup'].sum())} univ={D['symbol'].nunique()}")
    T,imp=walk_forward(D)
    Y=list(range(2022,2027))
    yr={y:g['pnl'].sum() for y,g in T.groupby('yr')}
    print(f"\n=== ML MEAN-REV SLEEVE (walk-forward OOS 2022-26) ===")
    print(f"  n={len(T)} Σ={T['pnl'].sum():+.1f}u mean={T['pnl'].mean()*100:+.2f}% WR={100*(T.pnl>0).mean():.0f}%")
    print("  per-year: "+" ".join(f"{y}:{yr.get(y,0):+.1f}(n{len(T[T.yr==y])})" for y in Y))
    print(f"\n  top features:\n{imp.sort_values(ascending=False).head(8).to_string()}")
    T.to_parquet('stock_ml/analysis/meanrev_sleeve/mr_trades.parquet')