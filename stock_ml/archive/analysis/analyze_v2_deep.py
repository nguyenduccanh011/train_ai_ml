"""
Deep Analysis V2 Model - Phân tích chi tiết mô hình mới
"""
import pandas as pd
import numpy as np
import os

def main():
    # Load V2 trades
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    df = pd.read_csv(os.path.join(results_dir, "v2_trades_20260419_120806.csv"))
    
    wins = df[df["is_win"] == True]
    losses = df[df["is_win"] == False]
    n = len(df)
    
    print("=" * 100)
    print("🔬 DEEP ANALYSIS - V2 MODEL (forward_risk_reward + leading features + GBM)")
    print("=" * 100)
    print(f"\n📊 TỔNG QUAN: {n} trades, {len(wins)} thắng ({len(wins)/n*100:.1f}%), {len(losses)} thua ({len(losses)/n*100:.1f}%)")
    print(f"   Avg Return: {df['trade_return_pct'].mean():+.2f}%")
    print(f"   Median Return: {df['trade_return_pct'].median():+.2f}%")
    print(f"   Total Return: {df['trade_return_pct'].sum():+.1f}%")
    print(f"   Avg Win: {wins['trade_return_pct'].mean():+.2f}%")
    print(f"   Avg Loss: {losses['trade_return_pct'].mean():+.2f}%")
    
    gross_profit = wins['trade_return_pct'].sum()
    gross_loss = abs(losses['trade_return_pct'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    print(f"   Profit Factor: {pf:.2f}")
    
    # ═══════════════════════════════════════════
    # 1. ĐIỂM MUA NẰM Ở ĐÂU TRONG SÓNG?
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("📍 1. ĐIỂM MUA NẰM Ở ĐÂU TRONG SÓNG? (0=đáy, 1=đỉnh)")
    print(f"{'─'*100}")
    
    bins = [(0, 0.2, "🟢 Gần đáy (0-20%)"), (0.2, 0.4, "🟡 Dưới giữa (20-40%)"), 
            (0.4, 0.6, "🟠 Giữa sóng (40-60%)"), (0.6, 0.8, "🔴 Trên giữa (60-80%)"), (0.8, 1.01, "⛔ Gần đỉnh (80-100%)")]
    
    print(f"   {'Vị trí mua':<30} {'Trades':>7} {'%':>7} {'Win%':>7} {'AvgRet':>9} {'AvgDD':>9} {'GaveBack':>9}")
    for lo, hi, label in bins:
        mask = (df["entry_wave_pos"] >= lo) & (df["entry_wave_pos"] < hi)
        s = df[mask]
        if len(s) > 0:
            wr = s["is_win"].mean() * 100
            print(f"   {label:<30} {len(s):>7} {len(s)/n*100:>6.1f}% {wr:>6.1f}% {s['trade_return_pct'].mean():>+8.2f}% {s['max_dd_pct'].mean():>8.2f}% {s['gave_back_pct'].mean():>8.2f}%")
    
    print(f"\n   📈 Entry wave trung bình: {df['entry_wave_pos'].mean():.2f}")
    print(f"      Wins: {wins['entry_wave_pos'].mean():.2f} | Losses: {losses['entry_wave_pos'].mean():.2f}")
    
    # ═══════════════════════════════════════════
    # 2. SAU KHI MUA, LỖ TỐI ĐA BAO NHIÊU?
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("📉 2. SAU KHI MUA, LỖ TỐI ĐA (MAX DRAWDOWN)")
    print(f"{'─'*100}")
    
    dd_bins = [(-100, -20, "⛔ DD > -20% (thảm họa)"), (-20, -10, "🔴 DD -10% đến -20% (nặng)"), 
               (-10, -5, "🟠 DD -5% đến -10%"), (-5, -2, "🟡 DD -2% đến -5%"), 
               (-2, 0.001, "🟢 DD 0% đến -2% (nhẹ)"), (0.001, 100, "✅ Không lỗ")]
    
    print(f"   {'Mức DD':<35} {'Trades':>7} {'%':>7} {'Win%':>7} {'AvgRet':>9}")
    for lo, hi, label in dd_bins:
        mask = (df["max_dd_pct"] >= lo) & (df["max_dd_pct"] < hi)
        s = df[mask]
        if len(s) > 0:
            print(f"   {label:<35} {len(s):>7} {len(s)/n*100:>6.1f}% {s['is_win'].mean()*100:>6.1f}% {s['trade_return_pct'].mean():>+8.2f}%")
    
    print(f"\n   Max DD trung bình: {df['max_dd_pct'].mean():.2f}%")
    print(f"   Max DD tệ nhất: {df['max_dd_pct'].min():.2f}%")
    print(f"   Max DD (wins): {wins['max_dd_pct'].mean():.2f}%")
    print(f"   Max DD (losses): {losses['max_dd_pct'].mean():.2f}%")
    
    # Trades với DD nặng
    heavy_dd = df[df["max_dd_pct"] < -10]
    print(f"\n   ⚠️  Trades DD > -10%: {len(heavy_dd)} ({len(heavy_dd)/n*100:.1f}%)")
    if len(heavy_dd) > 0:
        print(f"      Win rate trong nhóm này: {heavy_dd['is_win'].mean()*100:.1f}%")
        print(f"      Avg return: {heavy_dd['trade_return_pct'].mean():+.2f}%")
        print(f"      Nếu loại bỏ nhóm này, avg return còn lại: {df[df['max_dd_pct'] >= -10]['trade_return_pct'].mean():+.2f}%")
    
    # ═══════════════════════════════════════════
    # 3. SAU KHI BÁN, GIÁ CÓ TĂNG TIẾP KHÔNG?
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("📈 3. SAU KHI BÁN (EXIT), GIÁ CÓ TIẾP TỤC TĂNG KHÔNG?")
    print(f"{'─'*100}")
    
    after_5d = df["after_exit_5d_pct"]
    after_up_5 = (after_5d > 2).sum()
    after_down_5 = (after_5d < -2).sum()
    after_flat = n - after_up_5 - after_down_5
    
    print(f"   Sau bán 5 ngày:")
    print(f"     Tăng >2%: {after_up_5} ({after_up_5/n*100:.1f}%) → BÁN SỚM, bỏ lỡ lợi nhuận")
    print(f"     Giảm >2%: {after_down_5} ({after_down_5/n*100:.1f}%) → BÁN ĐÚNG")
    print(f"     Đi ngang: {after_flat} ({after_flat/n*100:.1f}%) → Trung lập")
    print(f"   Avg return sau bán 5d: {after_5d.mean():+.2f}%")
    
    # Phân tích bán sớm ở lệnh thắng
    premature = wins[wins["after_exit_5d_pct"] > 2]
    print(f"\n   🔴 BÁN SỚM (lệnh thắng + giá tăng >2% sau 5d): {len(premature)}/{len(wins)} wins ({len(premature)/max(len(wins),1)*100:.1f}%)")
    if len(premature) > 0:
        print(f"      Avg profit đã chốt: {premature['trade_return_pct'].mean():+.2f}%")
        print(f"      Avg tiếp tục tăng: {premature['after_exit_5d_pct'].mean():+.2f}%")
        print(f"      → Missed thêm: {premature['after_exit_5d_pct'].mean():.2f}%")
    
    # Phân tích bán đúng
    correct_exit = df[df["after_exit_5d_pct"] < -2]
    print(f"\n   ✅ BÁN ĐÚNG (giá giảm >2% sau 5d): {len(correct_exit)}/{n} ({len(correct_exit)/n*100:.1f}%)")
    
    # ═══════════════════════════════════════════
    # 4. GAVE BACK - LỢI NHUẬN ĐÃ TRẢ LẠI
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("💸 4. GAVE BACK - LỢI NHUẬN ĐÃ TRẢ LẠI")
    print(f"{'─'*100}")
    
    print(f"   Max profit TB trong trade: {df['max_profit_pct'].mean():+.2f}%")
    print(f"   Actual return TB: {df['trade_return_pct'].mean():+.2f}%")
    print(f"   Gave back TB: {df['gave_back_pct'].mean():.2f}%")
    print(f"   Gave back (wins): {wins['gave_back_pct'].mean():.2f}%")
    print(f"   Gave back (losses): {losses['gave_back_pct'].mean():.2f}%")
    
    big_gb = df[df["gave_back_pct"] > 5]
    print(f"\n   ⚠️  Gave back >5%: {len(big_gb)} trades ({len(big_gb)/n*100:.1f}%)")
    if len(big_gb) > 0:
        print(f"      Max profit TB: {big_gb['max_profit_pct'].mean():+.2f}%")
        print(f"      Actual return TB: {big_gb['trade_return_pct'].mean():+.2f}%")
        print(f"      → Nếu trailing stop ở 50% max profit, sẽ chốt: {big_gb['max_profit_pct'].mean()*0.5:+.2f}% thay vì {big_gb['trade_return_pct'].mean():+.2f}%")
    
    # ═══════════════════════════════════════════
    # 5. PHÂN TÍCH THEO ACTUAL TARGET
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("🎯 5. MODEL ACCURACY - DỰ ĐOÁN ĐÚNG BAO NHIÊU?")
    print(f"{'─'*100}")
    
    # actual_target = 1 means it was actually a good buy point
    correct_buy = df[df["actual_target"] == 1]
    wrong_buy = df[df["actual_target"] == 0]
    
    print(f"   Mô hình dự đoán BUY → Actual là BUY point thật: {len(correct_buy)}/{n} ({len(correct_buy)/n*100:.1f}%)")
    print(f"   Mô hình dự đoán BUY → KHÔNG phải buy point: {len(wrong_buy)}/{n} ({len(wrong_buy)/n*100:.1f}%)")
    
    if len(correct_buy) > 0:
        print(f"\n   Correct predictions (actual_target=1):")
        print(f"      Win rate: {correct_buy['is_win'].mean()*100:.1f}%")
        print(f"      Avg return: {correct_buy['trade_return_pct'].mean():+.2f}%")
        print(f"      Avg DD: {correct_buy['max_dd_pct'].mean():.2f}%")
    
    if len(wrong_buy) > 0:
        print(f"\n   Wrong predictions (actual_target=0):")
        print(f"      Win rate: {wrong_buy['is_win'].mean()*100:.1f}%")
        print(f"      Avg return: {wrong_buy['trade_return_pct'].mean():+.2f}%")
        print(f"      Avg DD: {wrong_buy['max_dd_pct'].mean():.2f}%")
    
    # ═══════════════════════════════════════════
    # 6. PHÂN TÍCH THEO SYMBOL
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("📋 6. PHÂN TÍCH THEO MÃ CỔ PHIẾU")
    print(f"{'─'*100}")
    
    print(f"   {'Symbol':<8} {'Trades':>7} {'Win%':>7} {'AvgRet':>9} {'TotalRet':>10} {'AvgDD':>8} {'Wave':>6}")
    for sym in sorted(df["symbol"].unique()):
        s = df[df["symbol"] == sym]
        wr = s["is_win"].mean() * 100
        print(f"   {sym:<8} {len(s):>7} {wr:>6.1f}% {s['trade_return_pct'].mean():>+8.2f}% {s['trade_return_pct'].sum():>+9.1f}% {s['max_dd_pct'].mean():>7.2f}% {s['entry_wave_pos'].mean():>5.2f}")
    
    # ═══════════════════════════════════════════
    # 7. PHÂN TÍCH THEO WINDOW (THỜI GIAN)
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("📅 7. PHÂN TÍCH THEO THỜI GIAN (WINDOW)")
    print(f"{'─'*100}")
    
    print(f"   {'Window':<70} {'Trades':>7} {'Win%':>7} {'AvgRet':>9}")
    for w in sorted(df["window"].unique()):
        s = df[df["window"] == w]
        print(f"   {w:<70} {len(s):>7} {s['is_win'].mean()*100:>6.1f}% {s['trade_return_pct'].mean():>+8.2f}%")
    
    # ═══════════════════════════════════════════
    # 8. PHÂN TÍCH BUY PROBABILITY
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("🎲 8. PHÂN TÍCH THEO XÁC SUẤT MUA (buy_prob)")
    print(f"{'─'*100}")
    
    prob_bins = [(0.5, 0.55, "50-55%"), (0.55, 0.6, "55-60%"), (0.6, 0.65, "60-65%"), 
                 (0.65, 0.7, "65-70%"), (0.7, 0.8, "70-80%"), (0.8, 1.01, "80-100%")]
    
    print(f"   {'Prob range':<15} {'Trades':>7} {'Win%':>7} {'AvgRet':>9} {'AvgDD':>8}")
    for lo, hi, label in prob_bins:
        mask = (df["buy_prob"] >= lo) & (df["buy_prob"] < hi)
        s = df[mask]
        if len(s) > 0:
            print(f"   {label:<15} {len(s):>7} {s['is_win'].mean()*100:>6.1f}% {s['trade_return_pct'].mean():>+8.2f}% {s['max_dd_pct'].mean():>7.2f}%")
    
    # ═══════════════════════════════════════════
    # 9. GIAO DỊCH THẢM HỌA - PHÂN TÍCH
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("💀 9. TOP 10 GIAO DỊCH TỆ NHẤT - TẠI SAO?")
    print(f"{'─'*100}")
    
    worst = df.nsmallest(10, "trade_return_pct")
    for _, t in worst.iterrows():
        print(f"\n   {t['symbol']} | Return: {t['trade_return_pct']:+.2f}% | DD: {t['max_dd_pct']:.2f}% | Wave: {t['entry_wave_pos']:.2f} | Prob: {t['buy_prob']:.3f}")
        issues = []
        if t['entry_wave_pos'] > 0.7:
            issues.append("Mua gần đỉnh sóng")
        if t['max_dd_pct'] < -15:
            issues.append(f"DD thảm họa {t['max_dd_pct']:.1f}%")
        if t['actual_target'] == 0:
            issues.append("Model sai - không phải buy point thật")
        if t['buy_prob'] > 0.7:
            issues.append(f"High confidence sai: prob={t['buy_prob']:.2f}")
        if not issues:
            issues.append("Thị trường giảm mạnh bất ngờ")
        print(f"      Nguyên nhân: {' | '.join(issues)}")
    
    # ═══════════════════════════════════════════
    # 10. TOP 10 GIAO DỊCH TỐT NHẤT
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("🏆 10. TOP 10 GIAO DỊCH TỐT NHẤT")
    print(f"{'─'*100}")
    
    best = df.nlargest(10, "trade_return_pct")
    for _, t in best.iterrows():
        print(f"   {t['symbol']} | Return: {t['trade_return_pct']:+.2f}% | Wave: {t['entry_wave_pos']:.2f} | MaxProfit: {t['max_profit_pct']:.2f}% | GaveBack: {t['gave_back_pct']:.2f}%")
    
    # ═══════════════════════════════════════════
    # 11. SIMULATION: NẾU ÁP DỤNG STOP-LOSS & TRAILING STOP
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("🔧 11. MÔ PHỎNG: NẾU ÁP DỤNG STOP-LOSS & TRAILING STOP")
    print(f"{'─'*100}")
    
    # Sim 1: Stop-loss at -7%
    for sl in [-5, -7, -10]:
        sim = df.copy()
        stopped = sim["max_dd_pct"] < sl
        sim.loc[stopped, "sim_return"] = sl  # assume stopped at exactly SL
        sim.loc[~stopped, "sim_return"] = sim.loc[~stopped, "trade_return_pct"]
        sim_wr = (sim["sim_return"] > 0).mean() * 100
        print(f"   Stop-loss {sl}%: AvgRet={sim['sim_return'].mean():+.2f}%, WinRate={sim_wr:.1f}%, TotalRet={sim['sim_return'].sum():+.1f}%")
    
    # Sim 2: Trailing stop at X% of max profit
    print()
    for trail_pct in [0.3, 0.5, 0.7]:
        sim = df.copy()
        # If max_profit > 3% and gave_back > trail%, assume exit at max_profit * (1-trail)
        high_profit = sim["max_profit_pct"] > 3
        gave_too_much = sim["gave_back_pct"] > sim["max_profit_pct"] * trail_pct
        improved = high_profit & gave_too_much
        sim.loc[improved, "sim_return"] = sim.loc[improved, "max_profit_pct"] * (1 - trail_pct)
        sim.loc[~improved, "sim_return"] = sim.loc[~improved, "trade_return_pct"]
        sim_wr = (sim["sim_return"] > 0).mean() * 100
        print(f"   Trailing stop {int(trail_pct*100)}% of max profit: AvgRet={sim['sim_return'].mean():+.2f}%, WinRate={sim_wr:.1f}%, TotalRet={sim['sim_return'].sum():+.1f}%, Improved={improved.sum()} trades")
    
    # Sim 3: Combined SL + trailing
    print()
    sl = -7
    trail = 0.5
    sim = df.copy()
    stopped = sim["max_dd_pct"] < sl
    sim["sim_return"] = sim["trade_return_pct"]
    sim.loc[stopped, "sim_return"] = sl
    high_profit = sim["max_profit_pct"] > 3
    gave_too_much = sim["gave_back_pct"] > sim["max_profit_pct"] * trail
    improved = high_profit & gave_too_much & ~stopped
    sim.loc[improved, "sim_return"] = sim.loc[improved, "max_profit_pct"] * (1 - trail)
    sim_wr = (sim["sim_return"] > 0).mean() * 100
    sim_pf = abs(sim[sim["sim_return"]>0]["sim_return"].sum() / sim[sim["sim_return"]<=0]["sim_return"].sum()) if sim[sim["sim_return"]<=0]["sim_return"].sum() != 0 else 0
    print(f"   🏆 COMBO SL={sl}% + Trail={int(trail*100)}%: AvgRet={sim['sim_return'].mean():+.2f}%, WinRate={sim_wr:.1f}%, TotalRet={sim['sim_return'].sum():+.1f}%, PF={sim_pf:.2f}")
    
    orig_pf = pf
    print(f"\n   So sánh: Original PF={orig_pf:.2f} → Combo PF={sim_pf:.2f}")
    print(f"   So sánh: Original Total={df['trade_return_pct'].sum():+.1f}% → Combo Total={sim['sim_return'].sum():+.1f}%")
    
    # ═══════════════════════════════════════════
    # 12. FILTER ANALYSIS: NẾU FILTER BỚT TRADES XẤU
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("🔍 12. FILTER ANALYSIS - NẾU LỌC BỚT TRADES XẤU")
    print(f"{'─'*100}")
    
    filters = [
        ("Wave < 0.5 (chỉ mua nửa dưới)", df["entry_wave_pos"] < 0.5),
        ("Wave < 0.7 (tránh đỉnh)", df["entry_wave_pos"] < 0.7),
        ("Prob > 0.6 (high confidence)", df["buy_prob"] > 0.6),
        ("Prob > 0.65", df["buy_prob"] > 0.65),
        ("Wave<0.5 & Prob>0.6", (df["entry_wave_pos"] < 0.5) & (df["buy_prob"] > 0.6)),
        ("Wave<0.7 & Prob>0.6", (df["entry_wave_pos"] < 0.7) & (df["buy_prob"] > 0.6)),
    ]
    
    print(f"   {'Filter':<35} {'Trades':>7} {'Win%':>7} {'AvgRet':>9} {'TotalRet':>10} {'PF':>6}")
    print(f"   {'(no filter)':<35} {n:>7} {len(wins)/n*100:>6.1f}% {df['trade_return_pct'].mean():>+8.2f}% {df['trade_return_pct'].sum():>+9.1f}% {pf:>5.2f}")
    for name, mask in filters:
        s = df[mask]
        if len(s) > 0:
            sw = s[s["is_win"]]
            sl = s[~s["is_win"]]
            f_pf = abs(sw["trade_return_pct"].sum() / sl["trade_return_pct"].sum()) if len(sl) > 0 and sl["trade_return_pct"].sum() != 0 else 0
            print(f"   {name:<35} {len(s):>7} {s['is_win'].mean()*100:>6.1f}% {s['trade_return_pct'].mean():>+8.2f}% {s['trade_return_pct'].sum():>+9.1f}% {f_pf:>5.2f}")
    
    # ═══════════════════════════════════════════
    # 13. KẾT LUẬN & ĐỀ XUẤT
    # ═══════════════════════════════════════════
    print(f"\n{'═'*100}")
    print("🎯 13. KẾT LUẬN & ĐỀ XUẤT CẢI THIỆN")
    print(f"{'═'*100}")
    
    avg_wave = df["entry_wave_pos"].mean()
    avg_dd = df["max_dd_pct"].mean()
    avg_gb = df["gave_back_pct"].mean()
    pct_heavy_dd = len(df[df["max_dd_pct"] < -10]) / n * 100
    pct_premature_sell = len(wins[wins["after_exit_5d_pct"] > 2]) / max(len(wins), 1) * 100
    accuracy = len(df[df["actual_target"] == 1]) / n * 100
    
    print(f"\n   📊 METRICS HIỆN TẠI:")
    print(f"      Entry wave: {avg_wave:.2f} | DD TB: {avg_dd:.2f}% | Gave back TB: {avg_gb:.2f}%")
    print(f"      DD nặng (>10%): {pct_heavy_dd:.1f}% trades | Bán sớm: {pct_premature_sell:.1f}% wins")
    print(f"      Model accuracy: {accuracy:.1f}%")
    
    issues = []
    recommendations = []
    
    if avg_dd < -5:
        issues.append(f"❌ DRAWDOWN LỚN: TB {avg_dd:.1f}%, cần stop-loss")
        recommendations.append("→ Stop-loss -7%: cắt lỗ sớm, tránh DD thảm họa >20%")
    
    if pct_heavy_dd > 15:
        issues.append(f"❌ QUÁ NHIỀU DD NẶNG: {pct_heavy_dd:.0f}% trades DD >10%")
        recommendations.append("→ Filter wave_pos < 0.7: tránh mua gần đỉnh")
    
    if avg_gb > 4:
        issues.append(f"❌ GAVE BACK NHIỀU: TB {avg_gb:.1f}%")
        recommendations.append("→ Trailing stop 50% max profit: lock profit khi đã lãi")
    
    if pct_premature_sell > 30:
        issues.append(f"⚠️  BÁN SỚM: {pct_premature_sell:.0f}% lệnh thắng mà giá còn tăng")
        recommendations.append("→ Dynamic hold: nếu trend mạnh, giữ thêm 5-10 ngày")
    
    if accuracy < 40:
        issues.append(f"⚠️  ACCURACY THẤP: chỉ {accuracy:.0f}% dự đoán đúng buy point")
        recommendations.append("→ Cần thêm features hoặc ensemble model")
    
    prob_high = df[df["buy_prob"] > 0.65]
    if len(prob_high) > 0 and prob_high["is_win"].mean() > df["is_win"].mean():
        recommendations.append(f"→ Tăng prob threshold lên 0.65: chỉ trade khi confident (hiện {len(prob_high)} trades, WR={prob_high['is_win'].mean()*100:.0f}%)")
    
    print(f"\n   🔴 VẤN ĐỀ:")
    for issue in issues:
        print(f"      {issue}")
    
    print(f"\n   🟢 ĐỀ XUẤT CẢI THIỆN:")
    for i, rec in enumerate(recommendations, 1):
        print(f"      {i}. {rec}")
    
    # Priority actions
    print(f"\n   🏆 ƯU TIÊN TRIỂN KHAI:")
    print(f"      1. STOP-LOSS -7% (giảm DD thảm họa)")
    print(f"      2. TRAILING STOP 50% max profit (lock lợi nhuận)")  
    print(f"      3. FILTER wave < 0.7 & prob > 0.6 (giảm trades xấu)")
    print(f"      4. SMART EXIT: kết hợp ATR-based trailing stop")
    print(f"      5. ENSEMBLE: thêm XGBoost voting với GBM + RF")

if __name__ == "__main__":
    main()
