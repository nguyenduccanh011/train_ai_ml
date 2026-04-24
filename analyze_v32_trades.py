"""
V32 Trade Analysis — Entry/Exit Timing & Improvement Ideas
"""
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "C:/Users/DUC CANH PC/Desktop/train_ai_ml/portable_data/vn_stock_ai_dataset_cleaned/all_symbols"
TRADES_CSV = "C:/Users/DUC CANH PC/Desktop/train_ai_ml/stock_ml/results/trades_v32.csv"


# ── Load price data ──────────────────────────────────────────────────────────

def load_prices(symbol):
    path = os.path.join(DATA_DIR, f"symbol={symbol}", "timeframe=1D", "data.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    return df[["date", "open", "high", "low", "close", "volume"]].drop_duplicates("date")


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze():
    trades = pd.read_csv(TRADES_CSV, parse_dates=["entry_date", "exit_date"])
    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.normalize()
    trades["exit_date"] = pd.to_datetime(trades["exit_date"]).dt.normalize()

    # Cache price by symbol
    price_cache = {}
    for sym in trades["entry_symbol"].unique():
        p = load_prices(sym)
        if p is not None:
            price_cache[sym] = p.set_index("date")["close"]

    # ── Compute entry/exit alternatives ───────────────────────────────────────
    records = []
    for _, row in trades.iterrows():
        sym = row["entry_symbol"]
        if sym not in price_cache:
            continue
        prices = price_cache[sym]

        entry_dt = row["entry_date"]
        exit_dt = row["exit_date"]
        entry_price = prices.get(entry_dt, np.nan)
        exit_price = prices.get(exit_dt, np.nan)
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
            continue

        def price_n_days_before(dt, n):
            idx = prices.index.get_indexer([dt], method="nearest")[0]
            target = idx - n
            if target < 0:
                return np.nan
            return prices.iloc[target]

        def price_n_days_after(dt, n):
            idx = prices.index.get_indexer([dt], method="nearest")[0]
            target = idx + n
            if target >= len(prices):
                return np.nan
            return prices.iloc[target]

        # Entry alternatives
        entry_m5  = price_n_days_before(entry_dt, 5)
        entry_m10 = price_n_days_before(entry_dt, 10)
        entry_m20 = price_n_days_before(entry_dt, 20)
        entry_m50 = price_n_days_before(entry_dt, 50)

        # Min price in window before entry (best possible entry)
        idx_entry = prices.index.get_indexer([entry_dt], method="nearest")[0]
        min_5  = prices.iloc[max(0, idx_entry-5):idx_entry].min() if idx_entry > 0 else np.nan
        min_10 = prices.iloc[max(0, idx_entry-10):idx_entry].min() if idx_entry > 0 else np.nan
        min_20 = prices.iloc[max(0, idx_entry-20):idx_entry].min() if idx_entry > 0 else np.nan

        # Price improvement if bought earlier (negative = cheaper = BETTER entry)
        def pct_diff(alt, base):
            if pd.isna(alt) or pd.isna(base) or base == 0:
                return np.nan
            return (alt - base) / base * 100

        entry_vs_m5  = pct_diff(entry_m5, entry_price)   # negative → earlier was cheaper
        entry_vs_m10 = pct_diff(entry_m10, entry_price)
        entry_vs_m20 = pct_diff(entry_m20, entry_price)
        entry_vs_m50 = pct_diff(entry_m50, entry_price)
        entry_vs_min5  = pct_diff(min_5, entry_price)
        entry_vs_min10 = pct_diff(min_10, entry_price)
        entry_vs_min20 = pct_diff(min_20, entry_price)

        # Check if this is a "bought at recovery peak" pattern:
        # trend down before → price rose before entry → then fell after entry
        # Look at 10d before entry price change
        price_10d_before = price_n_days_before(entry_dt, 10)
        price_3d_after   = price_n_days_after(entry_dt, 3)
        rise_before_entry = pct_diff(entry_price, price_10d_before) if not pd.isna(price_10d_before) else np.nan
        drop_after_entry  = pct_diff(price_3d_after, entry_price) if not pd.isna(price_3d_after) else np.nan

        # Exit alternatives
        exit_m5  = price_n_days_before(exit_dt, 5)
        exit_m10 = price_n_days_before(exit_dt, 10)
        exit_p5  = price_n_days_after(exit_dt, 5)
        exit_p10 = price_n_days_after(exit_dt, 10)

        # Max price in window around exit (best possible exit)
        idx_exit = prices.index.get_indexer([exit_dt], method="nearest")[0]
        max_before5  = prices.iloc[max(0, idx_exit-5):idx_exit].max() if idx_exit > 0 else np.nan
        max_before10 = prices.iloc[max(0, idx_exit-10):idx_exit].max() if idx_exit > 0 else np.nan
        max_after5   = prices.iloc[idx_exit+1:min(len(prices), idx_exit+6)].max()
        max_after10  = prices.iloc[idx_exit+1:min(len(prices), idx_exit+11)].max()

        exit_vs_m5   = pct_diff(exit_m5,   exit_price)  # negative → earlier was cheaper → missed selling high
        exit_vs_m10  = pct_diff(exit_m10,  exit_price)
        exit_vs_p5   = pct_diff(exit_p5,   exit_price)  # positive → later higher → sold too early
        exit_vs_p10  = pct_diff(exit_p10,  exit_price)
        missed_before5  = pct_diff(max_before5,  exit_price)  # positive → could have sold higher 5d ago
        missed_before10 = pct_diff(max_before10, exit_price)
        missed_after5   = pct_diff(max_after5,   exit_price)  # positive → price rose after exit
        missed_after10  = pct_diff(max_after10,  exit_price)

        records.append({
            **row.to_dict(),
            "entry_price": entry_price,
            "exit_price":  exit_price,
            # Entry alternatives
            "entry_vs_m5":  entry_vs_m5,
            "entry_vs_m10": entry_vs_m10,
            "entry_vs_m20": entry_vs_m20,
            "entry_vs_m50": entry_vs_m50,
            "entry_vs_min5":  entry_vs_min5,
            "entry_vs_min10": entry_vs_min10,
            "entry_vs_min20": entry_vs_min20,
            "rise_before_entry": rise_before_entry,
            "drop_after_entry":  drop_after_entry,
            # Exit alternatives
            "exit_vs_m5":  exit_vs_m5,
            "exit_vs_m10": exit_vs_m10,
            "exit_vs_p5":  exit_vs_p5,
            "exit_vs_p10": exit_vs_p10,
            "missed_before5":  missed_before5,
            "missed_before10": missed_before10,
            "missed_after5":   missed_after5,
            "missed_after10":  missed_after10,
        })

    df = pd.DataFrame(records)
    print(f"Analyzed {len(df)} trades (of {len(trades)} total)\n")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: Overall stats
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 80)
    print("SECTION 1 — TỔNG QUAN V32")
    print("=" * 80)
    wins = df[df["pnl_pct"] > 0]
    losses = df[df["pnl_pct"] < 0]
    total = len(df)
    print(f"  Total trades : {total}")
    print(f"  Win rate     : {len(wins)/total*100:.1f}%")
    print(f"  Avg win      : +{wins['pnl_pct'].mean():.2f}%")
    print(f"  Avg loss     : {losses['pnl_pct'].mean():.2f}%")
    print(f"  Profit factor: {wins['pnl_pct'].sum()/abs(losses['pnl_pct'].sum()):.2f}")
    print(f"  Total PnL    : {df['pnl_pct'].sum():.1f}%")
    print(f"  Max loss     : {df['pnl_pct'].min():.2f}%")
    print(f"  Avg hold     : {df['holding_days'].mean():.1f}d")
    print()
    print("  Exit reasons:")
    for r, cnt in df["exit_reason"].value_counts().items():
        sub = df[df["exit_reason"] == r]
        wr = (sub["pnl_pct"] > 0).mean() * 100
        avg = sub["pnl_pct"].mean()
        print(f"    {r:<30} {cnt:>4} trades  WR={wr:.0f}%  avg={avg:+.2f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: Entry timing analysis — "bought at the top"
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SECTION 2 — PHÂN TÍCH TIMING MUA (ĐU ĐỈNH / MUA XA ĐÁY)")
    print("=" * 80)

    # Classify entry patterns
    def classify_entry(row):
        r = row.get("rise_before_entry", np.nan)
        d = row.get("drop_after_entry", np.nan)
        v5  = row.get("entry_vs_min5",  np.nan)
        v20 = row.get("entry_vs_min20", np.nan)
        if pd.isna(r) or pd.isna(d):
            return "unknown"
        # FOMO/đỉnh: giá đã tăng mạnh 10 ngày trước, sau khi mua thì giảm
        if r > 10 and d < -3:
            return "fomo_top"
        # Đỉnh hồi phục: giá đã phục hồi đáng kể nhưng sau mua lại giảm
        if r > 5 and d < -2:
            return "recovery_peak"
        # Mua xa đáy — min 20 ngày trước thấp hơn nhiều
        if not pd.isna(v20) and v20 < -10:
            return "far_from_bottom"
        return "normal"

    df["entry_pattern"] = df.apply(classify_entry, axis=1)

    # ── 2a: Nếu mua sớm hơn có giá tốt hơn không?
    losing = df[df["pnl_pct"] < 0]
    print(f"\n  >> {len(losing)} lệnh THUA. Phân tích nếu mua trước đó N ngày:")
    print(f"  {'Offset':<12} {'Cheaper %':>10} {'Avg saving':>12} {'Med saving':>12}")
    for offset_col, label in [
        ("entry_vs_m5",  "-5d"),
        ("entry_vs_m10", "-10d"),
        ("entry_vs_m20", "-20d"),
        ("entry_vs_m50", "-50d"),
    ]:
        sub = losing[losing[offset_col].notna()]
        cheaper = (sub[offset_col] < -1).mean() * 100
        avg_saving = sub[sub[offset_col] < -1][offset_col].mean()
        med_saving = sub[sub[offset_col] < -1][offset_col].median()
        print(f"  {label:<12} {cheaper:>9.1f}%  {avg_saving:>+11.2f}%  {med_saving:>+11.2f}%")

    # ── 2b: Best possible entry (min price in window)
    print(f"\n  >> Nếu chọn min giá tốt nhất trong N ngày trước khi vào:")
    for col, label in [("entry_vs_min5", "Min-5d"), ("entry_vs_min10", "Min-10d"), ("entry_vs_min20", "Min-20d")]:
        sub = losing[losing[col].notna()]
        pct_better = (sub[col] < -2).mean() * 100
        avg_save = sub[sub[col] < -2][col].mean()
        print(f"  {label:<14} {pct_better:.0f}% trades có giá thấp hơn ≥2%  avg saving={avg_save:+.2f}%")

    # ── 2c: Pattern breakdown
    print(f"\n  >> Phân loại pattern entry:")
    for pat in ["fomo_top", "recovery_peak", "far_from_bottom", "normal", "unknown"]:
        sub = df[df["entry_pattern"] == pat]
        if len(sub) == 0:
            continue
        wr = (sub["pnl_pct"] > 0).mean() * 100
        avg = sub["pnl_pct"].mean()
        print(f"    {pat:<20} {len(sub):>5} trades  WR={wr:.0f}%  avg={avg:+.2f}%")

    # ── 2d: Top worst "bought at recovery peak" examples
    fomo = df[(df["entry_pattern"].isin(["fomo_top", "recovery_peak"])) & (df["pnl_pct"] < -5)]
    fomo = fomo.sort_values("pnl_pct").head(15)
    print(f"\n  >> Top {len(fomo)} lệnh thua nặng nhất do mua đỉnh hồi phục / FOMO:")
    print(f"  {'Symbol':<8} {'Entry date':<14} {'PnL':>8} {'Rise before':>12} {'Drop after':>12} {'Pattern':<20}")
    for _, r in fomo.iterrows():
        print(f"  {r['entry_symbol']:<8} {str(r['entry_date'])[:10]:<14} {r['pnl_pct']:>+7.2f}%  "
              f"{r.get('rise_before_entry', np.nan):>+10.2f}%  "
              f"{r.get('drop_after_entry', np.nan):>+10.2f}%  {r['entry_pattern']:<20}")

    # ── 2e: High rise before entry — even for winners (bought too late in wave)
    high_rise_entries = df[df["rise_before_entry"].fillna(0) > 15]
    print(f"\n  >> {len(high_rise_entries)} lệnh vào khi giá đã tăng >15% trong 10 ngày trước:")
    wr_late = (high_rise_entries["pnl_pct"] > 0).mean() * 100
    avg_late = high_rise_entries["pnl_pct"].mean()
    print(f"     WR={wr_late:.1f}%  avg={avg_late:+.2f}%  (vs tổng WR={len(wins)/total*100:.1f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: Exit timing analysis
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SECTION 3 — PHÂN TÍCH TIMING BÁN")
    print("=" * 80)

    # ── 3a: Bán sớm quá — giá tiếp tục tăng sau khi bán
    print("\n  >> Lệnh BÁN QUÁ SỚM — giá tăng thêm sau khi bán:")
    for col, label in [("missed_after5", "+5d"), ("missed_after10", "+10d")]:
        sub = df[df[col].notna()]
        missed_sig = (sub[col] > 5).mean() * 100
        avg_missed = sub[sub[col] > 5][col].mean()
        print(f"    {label}: {missed_sig:.1f}% trades giá tăng >5% sau bán  avg missed={avg_missed:+.2f}%")

    # ── 3b: Bán muộn quá — đỉnh đã qua
    print(f"\n  >> Lệnh BÁN QUÁ MUỘN — đỉnh đã qua trước khi bán:")
    for col, label in [("missed_before5", "-5d"), ("missed_before10", "-10d")]:
        sub = df[df[col].notna()]
        missed_sig = (sub[col] > 5).mean() * 100
        avg_missed = sub[sub[col] > 5][col].mean()
        print(f"    {label}: {missed_sig:.1f}% trades đỉnh cao hơn điểm bán >5%  avg missed={avg_missed:+.2f}%")

    # ── 3c: Worst exit timing — sold too late (big loss that had profit before)
    late_exit_loss = df[
        (df["pnl_pct"] < -5) &
        (df["max_profit_pct"] > 5)
    ].sort_values("pnl_pct")
    print(f"\n  >> {len(late_exit_loss)} lệnh: max_profit>5% nhưng kết thúc lỗ >5% (bán muộn)")
    if len(late_exit_loss) > 0:
        print(f"  {'Symbol':<8} {'Entry':>12} {'Exit':>12} {'MaxProfit':>10} {'PnL':>8} {'ExitReason':<25}")
        for _, r in late_exit_loss.head(15).iterrows():
            print(f"  {r['entry_symbol']:<8} {str(r['entry_date'])[:10]:>12} {str(r['exit_date'])[:10]:>12} "
                  f"{r['max_profit_pct']:>+9.2f}% {r['pnl_pct']:>+7.2f}% {r['exit_reason']:<25}")

    # ── 3d: Sold too early — missed big upside
    early_exit_win = df[
        (df["pnl_pct"] > 0) &
        (df["missed_after10"].fillna(0) > 10)
    ].sort_values("missed_after10", ascending=False)
    print(f"\n  >> {len(early_exit_win)} lệnh lời nhưng giá tăng >10% sau 10 ngày bán (bán sớm):")
    if len(early_exit_win) > 0:
        print(f"  {'Symbol':<8} {'Entry':>12} {'Exit':>12} {'PnL':>8} {'After10d':>10} {'ExitReason':<25}")
        for _, r in early_exit_win.head(15).iterrows():
            print(f"  {r['entry_symbol']:<8} {str(r['entry_date'])[:10]:>12} {str(r['exit_date'])[:10]:>12} "
                  f"{r['pnl_pct']:>+7.2f}% {r.get('missed_after10', np.nan):>+9.2f}% {r['exit_reason']:<25}")

    # ── 3e: Bán ở đáy điều chỉnh — giá phục hồi ngay sau bán
    sold_at_dip = df[
        (df["missed_after5"].fillna(0) > 8) &
        (df["pnl_pct"] < 5)   # not a big winner when sold
    ].sort_values("missed_after5", ascending=False)
    print(f"\n  >> {len(sold_at_dip)} lệnh bán ở đáy điều chỉnh (giá tăng >8% trong 5 ngày sau bán):")
    if len(sold_at_dip) > 0:
        print(f"  {'Symbol':<8} {'Exit date':>12} {'PnL':>8} {'Next5d':>8} {'ExitReason':<25}")
        for _, r in sold_at_dip.head(10).iterrows():
            print(f"  {r['entry_symbol']:<8} {str(r['exit_date'])[:10]:>12} "
                  f"{r['pnl_pct']:>+7.2f}% {r.get('missed_after5', np.nan):>+7.2f}% {r['exit_reason']:<25}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: Pattern quantification for v30 changes
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SECTION 4 — SO SÁNH V30 / V31 / V32")
    print("=" * 80)
    for ver in ["v30", "v31", "v32"]:
        try:
            t = pd.read_csv(f"C:/Users/DUC CANH PC/Desktop/train_ai_ml/stock_ml/results/trades_{ver}.csv")
            n = len(t)
            wr = (t["pnl_pct"] > 0).mean() * 100
            tot = t["pnl_pct"].sum()
            avg_w = t[t["pnl_pct"] > 0]["pnl_pct"].mean()
            avg_l = t[t["pnl_pct"] < 0]["pnl_pct"].mean()
            pf = t[t["pnl_pct"] > 0]["pnl_pct"].sum() / abs(t[t["pnl_pct"] < 0]["pnl_pct"].sum())
            hold = t["holding_days"].mean()
            print(f"  {ver.upper()}  trades={n}  WR={wr:.1f}%  avg_win={avg_w:+.2f}%  avg_loss={avg_l:+.2f}%  "
                  f"PF={pf:.2f}  total={tot:+.1f}%  hold={hold:.1f}d")
        except Exception as e:
            print(f"  {ver}: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5: Detailed pattern summary for v33 proposals
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SECTION 5 — QUANTIFICATION CHO ĐỀ XUẤT V33")
    print("=" * 80)

    # Entry quality buckets
    df["entry_quality"] = "ok"
    df.loc[df["rise_before_entry"].fillna(0) > 15, "entry_quality"] = "late_entry_high_rise"
    df.loc[df["entry_vs_min20"].fillna(0) < -10, "entry_quality"] = "far_from_20d_bottom"
    df.loc[(df["rise_before_entry"].fillna(0) > 10) & (df["drop_after_entry"].fillna(0) < -3), "entry_quality"] = "bought_at_peak"

    print("\n  >> Entry quality distribution vs performance:")
    print(f"  {'Quality':<28} {'N':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10}")
    for q in ["ok", "late_entry_high_rise", "far_from_20d_bottom", "bought_at_peak"]:
        sub = df[df["entry_quality"] == q]
        if len(sub) == 0:
            continue
        wr = (sub["pnl_pct"] > 0).mean() * 100
        avg = sub["pnl_pct"].mean()
        tot = sub["pnl_pct"].sum()
        print(f"  {q:<28} {len(sub):>5}  {wr:>5.1f}%  {avg:>+7.2f}%  {tot:>+9.1f}%")

    # Summarize "sold at bottom" vs "sold at top distributed"
    print(f"\n  >> Exit pattern analysis:")
    late_sell = df[df["missed_before5"].fillna(0) > 5]
    early_sell = df[df["missed_after5"].fillna(0) > 5]
    print(f"    Bán muộn (đỉnh đã qua 5d trước >5%): {len(late_sell)} lệnh  "
          f"avg PnL={late_sell['pnl_pct'].mean():+.2f}%")
    print(f"    Bán sớm (giá tăng >5% trong 5d sau): {len(early_sell)} lệnh  "
          f"avg PnL={early_sell['pnl_pct'].mean():+.2f}%")

    # signal_hard_cap breakdown
    hc = df[df["exit_reason"] == "signal_hard_cap"]
    print(f"\n  >> signal_hard_cap ({len(hc)} trades):")
    print(f"     WR={( hc['pnl_pct']>0).mean()*100:.0f}%  avg={hc['pnl_pct'].mean():+.2f}%  "
          f"total={hc['pnl_pct'].sum():+.1f}%")
    print(f"     max_profit>5% rồi đóng âm: "
          f"{((hc['max_profit_pct']>5)&(hc['pnl_pct']<0)).sum()} trades")
    print(f"     max_profit>10% rồi đóng âm: "
          f"{((hc['max_profit_pct']>10)&(hc['pnl_pct']<0)).sum()} trades")

    # Hap_preempt stats
    hap = df[df["exit_reason"] == "v32_hap_preempt"]
    print(f"\n  >> v32_hap_preempt ({len(hap)} trades):")
    print(f"     WR={( hap['pnl_pct']>0).mean()*100:.0f}%  avg={hap['pnl_pct'].mean():+.2f}%  "
          f"total={hap['pnl_pct'].sum():+.1f}%")

    return df


if __name__ == "__main__":
    df = analyze()
    out = "C:/Users/DUC CANH PC/Desktop/train_ai_ml/stock_ml/results/v32_timing_analysis.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
