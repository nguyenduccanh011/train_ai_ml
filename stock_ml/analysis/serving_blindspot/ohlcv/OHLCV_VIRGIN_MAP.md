# BẢN ĐỒ THÔNG TIN CHƯA KHAI THÁC TRONG OHLCV — champion 2646

Ngày: 2026-07-09. Câu hỏi: sau 16 thế hệ, kênh thông tin OHLCV nào champion CHƯA nhìn,
và kênh nào còn tín hiệu THẬT (IC trực giao > null) đáng mở vòng label/head mới?

- Champion: template **2646** `n2_2643_wavestruct_la05_lamp02`, strategy
  `regression_dual_ml_recombine_decoupled`, universe `vn_stock_default` (61 mã).
- Screening: `screen_ic.py` (thư mục này) — không train model, không đốt leaderboard run.
- Kết quả máy: `screen_results.json` / `screen_results.csv`. Causality audit (rebuild-truncated): **PASS** (3 dates, 0 mismatch).

---

## Phần 1 — Kiểm kê feature đang cấp cho champion

### Bản đồ head → feature set (từ Postgres `strategy_templates`/`component_slots` id 2646)

| Head | Target | Feature set | # feat |
|---|---|---|---|
| Entry chính (ml_component 82) | triple_barrier pt=0.15 sl=0.08 h=30 | `entry_lvup126_recov` | 45 |
| zE1 `entry_ensemble` | reversal_entry_regression h10 | (dùng chung entry) | 45 |
| zE2 `entry_ensemble2` | continuation_entry_regression h10 | (dùng chung entry) | 45 |
| zE3 `entry_ensemble3` | mfe_regression h20 | entry + breadth `pct_above_ma50`,`adv_pct` | 47 |
| zE4 `entry_ensemble4` | forward_return_penalized_regression h20 | (dùng chung entry) | 45 |
| Exit (ml_component 78) | velocity_exit_regression h20 u8 volnorm40 | `exit_vol_downpress` | 19 |

`entry_lvup126_recov` = `_LEADING_V2` (37) + lowvol_rank_60, nearhigh_rank, comp_nh_lv,
comp_lowvol_uptrend, comp_nh_x_lv126, dist_10d_low, range_pos_20, recov_setup.
`exit_vol_downpress` = `_EXIT_VOL_DIST` (16) + down_vol_intensity_5, down_vol_count_10, updown_vol_20.
Không có entry_ensemble5 / exit_ensemble. Union toàn stack = **58 feature duy nhất**
(nguồn: `stock_ml/src/features/catalog.py` + bảng `feature_def`).

### Tỷ trọng kênh input (58 feature duy nhất)

| Kênh | # | % | Feature |
|---|---|---|---|
| **Chỉ Close** (kể cả CSRank/market trên close) | **37** | **63.8%** | ret_1/5/10/20d, sma_5/20/50_ratio, ema_10_ratio, sma5_cross_sma20, rsi_7/14, macd_line/hist, roc_10, bb_width/pct_20, bb_squeeze, realized_vol_10, vol_percentile_60, dist_52w_high/low, dist_63d_high, dist_10d_low, range_pos_20 (min/max của CLOSE), recov_setup, lowvol_rank_60, nearhigh_rank, comp_nh_lv, comp_lowvol_uptrend, comp_nh_x_lv126, momentum_rank, volatility_rank, ma5_accel, market_trend, market_volatility_regime, pct_above_ma50, adv_pct |
| Có High/Low | 12 | 20.7% | adx_14, plus_di_14, minus_di_14, atr_14_ratio, atr_regime, high_low_pct, high_low_pct_5d, mfi_14, is_limit_lock, upper/lower_wick_ratio, body_ratio |
| Có Volume | 10 | 17.2% | volume_ratio_5/20, obv_slope_10, mfi_14, is_limit_lock, dist_day_25, dist_day_vol20_25, down_vol_intensity_5, down_vol_count_10, updown_vol_20 |
| Có Open | 4 | 6.9% | close_to_open, upper_wick_ratio, lower_wick_ratio, body_ratio — **tất cả chỉ 1 bar t, không có tổng hợp k-bar** |

### Kênh trống hoàn toàn (0 feature trong champion)

1. **Gap qua đêm** (`open_t` vs `close_{t-1}`): không feature nào vượt ranh giới phiên. `close_to_open` là intraday.
2. **Phân rã return overnight vs intraday** (ai đẩy giá: ATC/gap hay trong phiên).
3. **Candle anatomy đa bar** (bóng nến/CLV trung bình k-bar) — chỉ có snapshot 1 bar.
4. **HL-range volatility estimator** (Parkinson/GK) — có trong catalog (`lowvol_park_rank`) nhưng champion không dùng.
5. **Illiquidity/microstructure volume**: Amihud, dollar-volume, CV(volume), corr(giá, volume) liên tục — chỉ có counts và OBV slope.
6. **Tần suất trần/sàn k-bar** (đặc thù VN ±7%): `is_limit_lock` chỉ bắt full-lock (high==low) tại bar t.
7. **Nén range nhanh NR7-style** (`atr_regime` là 14 vs 50 — chậm) và **path efficiency theo True Range**.

---

## Phần 2 — 25 ứng viên (công thức trong `screen_ic.py::sym_features`, tất cả rolling kết thúc tại t, causal audit PASS)

| Kênh | Ứng viên | Công thức tóm tắt |
|---|---|---|
| Gap | c_gap_ret1 | clip(O/C₋₁−1, ±7.5%) |
| Gap | c_gap_abs_mean10 | mean(\|gap\|,10) |
| Gap | c_gap_freq15_10 | mean(\|gap\|>1.5%,10) |
| Gap | c_on_drift_20 | Σ log(1+gap), 20 bar (drift qua đêm) |
| Gap | c_gap_follow_10 | mean(sign(gap)·(C/O−1),10) — follow-through vs fade |
| Candle | c_clv | ((C−L)−(H−C))/(H−L) |
| Candle | c_clv_mean_10 | mean(CLV,10) |
| Candle | c_uwick_mean_10 / c_lwick_mean_10 | mean(bóng trên/dưới ÷ range,10) |
| Range | c_nr_pos_7 | TR/max(TR,7) — nén NR7 |
| Range | c_tr_ratio_5_20 | mean(TR,5)/mean(TR,20) |
| Range | c_park_vs_close_20 | ParkinsonVol20 / std(ret,20) — thông tin HL ngoài close-vol |
| Range | c_range_pos_hl_20 | (C−minL20)/(maxH20−minL20) (bản HL của range_pos_20) |
| Range | c_path_eff_10 | \|ret10\| / Σ(TR/C₋₁,10) — hiệu suất đường đi |
| Volume | c_amihud_20 | log mean(\|ret1\|/(C·V),20) — illiquidity |
| Volume | c_vol_cv_20 | std(V,20)/mean(V,20) |
| Volume | c_volu_ma5_20 | mean(V,5)/mean(V,20) |
| Volume | **c_pv_corr_10** | corr(ret1, Δlog V, 10) — đồng biến giá–khối lượng |
| Volume | c_signed_vol_10 | (upV−downV)/ΣV, 10 |
| Volume | c_eff_result_10 | mean(V·CLV,10)/mean(V,10) — effort vs result |
| VN | c_limit_up_cnt_20 / c_limit_dn_cnt_20 | # phiên ret1 ≥ +6.5% / ≤ −6.5% trong 20 |
| VN | c_bigmove_freq_60 | mean(\|ret1\|≥5%, 60) |
| Health | c_range_spike_freq_20 | mean(TR/mean(TR,60) > 1.8, 20) |
| Health | c_zero_ret_freq_20 | mean(\|ret1\|<1e-4, 20) — staleness |

---

## Phần 3 — IC screening (OOS folds 2022/23/24/25/26H1, 1 146 ngày, 61 mã/ngày)

Phương pháp: IC = Spearman cross-sectional mỗi ngày → mean theo fold.
**IC trực giao** = residual rank-OLS per-date của ứng viên trên (a) `close10` = 10 feature
Close chủ lực của stack (ret_5d, ret_20d, sma_20/50_ratio, rsi_14, macd_hist,
realized_vol_10, dist_52w_high, range_pos_20, dist_10d_low); (b) `strict16` = close10 + 6
feature HL/O/V stack ĐÃ có (atr_14_ratio, high_low_pct_5d, upper_wick_ratio,
volume_ratio_20, updown_vol_20, mfi_14) → residual strict16 ≈ thông tin thật sự mới.
**Null** shuffle-within-date: band 95% = **±0.0076** (analytic sd 0.0039, empirical
200-shuffle xác nhận: sd 0.0039, band −0.0070..+0.0089). `t_cons` = t bảo thủ với
N_eff = ngày/21 (chỉnh overlap horizon). Target (b) `vexit` = label exit của champion
tái tạo CHÍNH XÁC (velocity_exit h20 u8 volnorm40; **IC dương = feature cao → sắp giảm**).

### Top theo |IC trực giao strict16| — target fwd_ret_21 (entry proxy)

| Ứng viên | IC thô | resid close10 | **resid strict16** | t_cons | folds 22/23/24/25/26 | sign |
|---|---|---|---|---|---|---|
| **c_pv_corr_10** | −0.028 | −0.031 | **−0.035** | **−2.09** | −.053/−.038/−.043/−.026/+.010 | 4/5 |
| c_amihud_20 | +0.009 | +0.021 | +0.025 | 1.11 | +.078/−.019/+.104/−.072/+.039 | 3/5 |
| c_on_drift_20 | −0.049 | −0.023 | −0.021 | −1.14 | −.017/−.014/−.049/−.034/+.050 | 4/5 |
| c_range_pos_hl_20 | +0.022 | +0.009 | +0.012 | 0.73 | +.008/+.026/+.008/+.022/−.025 | 4/5 |
| c_clv | −0.004 | −0.017 | −0.010 | −0.57 | 5 fold cùng dấu âm | 5/5 |
| (20 ứng viên còn lại) | | | \|resid16\| ≤ 0.010 | \|t\| < 0.6 | | |

### Top theo |IC trực giao strict16| — target vexit (label exit champion)

| Ứng viên | IC thô | resid close10 | **resid strict16** | t_cons | folds 22/23/24/25/26 | sign |
|---|---|---|---|---|---|---|
| **c_pv_corr_10** | +0.033 | +0.042 | **+0.041** | **+2.34** | +.054/+.048/+.033/+.043/+.003 | **5/5** |
| c_amihud_20 | −0.014 | −0.024 | −0.025 | −1.21 | −.059/+.011/−.081/+.035/−.035 | 3/5 |
| c_bigmove_freq_60 | −0.018 | −0.012 | −0.022 | −1.20 | −.033/−.022/+.019/−.056/−.018 | 4/5 |
| c_clv | +0.015 | +0.030 | +0.019 | 1.02 | +.012/+.023/+.012/+.030/+.017 | **5/5** |
| c_park_vs_close_20 | +0.031 | +0.027 | +0.016 | 0.95 | +.018/−.018/+.037/+.039/−.006 | 3/5 |
| c_gap_ret1 | −0.019 | −0.016 | −0.016 | −0.94 | −.042/−.023/+.007/−.008/−.015 | 4/5 |
| c_nr_pos_7 | +0.009 | +0.013 | +0.012 | 0.65 | 5 fold cùng dấu dương | 5/5 |
| (18 còn lại) | | | \|resid16\| ≤ 0.011 | \|t\| < 0.6 | | |

### Nhận xét trung thực

- **23/25 ứng viên CHẾT trên entry proxy** sau trực giao hóa (|resid16| ≤ null band hoặc
  fold không nhất quán). Nhiều IC thô trông được (c_on_drift_20 −0.049, c_gap_abs_mean10
  −0.029, c_bigmove_freq_60 −0.028) teo ≥50% sau residualize = phần lớn là vol/momentum cũ đội lốt.
- **c_pv_corr_10 là ngoại lệ duy nhất đạt chuẩn cả 2 target**: IC trực giao −0.035 (fwd21)
  / +0.041 (vexit), CAO HƠN IC thô (bị feature cũ che chứ không phải ăn theo), 4/5 và 5/5
  fold cùng dấu, |t_cons| > 2 cả hai chiều, vượt null band ~5σ. Nghĩa kinh tế nhất quán:
  giá và khối lượng đồng biến chặt trong 10 phiên (rally hút volume kiểu đu đỉnh) → forward
  return thấp hơn VÀ downside bất thường sắp tới cao hơn. Đây là kênh "đồng chuyển động
  giá–khối lượng liên tục" mà stack chỉ nhìn qua counts rời rạc (dist_day, down_vol_count).
- **c_amihud_20** (illiquidity premium): |resid16| ~0.025 cả 2 target, hướng nhất quán
  (kém thanh khoản → fwd return cao hơn, ít downside chuẩn hóa hơn) nhưng fold ĐẢO DẤU
  mạnh (2022 +0.078 / 2025 −0.072) — tín hiệu regime-dependent, chưa đạt chuẩn promote,
  đáng theo dõi như conditioning feature hơn là alpha.
- **c_clv** (vị trí close trong range ngày): nhỏ (|IC| 0.01–0.02) nhưng **cùng dấu cả
  10/10 fold × 2 target** — close sát high hôm nay → mai kém + sắp giảm (contrarian intraday).
  Dưới ngưỡng t nhưng là bias thật, rẻ, đáng nhét vào exit set nếu mở vòng mới.
- Caveat: (1) universe 61 mã → IC cross-sectional ồn, t bảo thủ max chỉ 2.3; (2) giá raw
  không back-adjust → kênh gap nhiễm dividend (đã clip ±7.5%, vẫn có thể giết tín hiệu gap
  thật); (3) vexit label chuẩn hóa vol → ứng viên tương quan vol (bigmove_freq) có thể ăn
  cấu trúc normalization thay vì alpha.

---

## Phần 4 — Xếp hạng & Verdict

### Xếp hạng ứng viên (theo IC trực giao + ổn định fold)

1. **c_pv_corr_10** — ĐẠT chuẩn screening (resid IC −0.035/+0.041, |t|>2, 9/10 fold đúng dấu). Ứng viên duy nhất đủ điều kiện mở thí nghiệm head/feature-set.
2. c_amihud_20 — biên độ đủ nhưng đảo dấu theo năm → theo dõi, không promote.
3. c_clv (+ c_nr_pos_7 cùng profile) — nhỏ, ổn định tuyệt đối về dấu, chỉ đáng làm feature kèm cho exit head, không gánh nổi head riêng.
4. c_bigmove_freq_60, c_park_vs_close_20, c_gap_ret1, c_on_drift_20 — một chiều target, fold lỗ chỗ hoặc nghi artifact.
5. 17 ứng viên còn lại (toàn bộ nhóm candle-mean, gap-frequency, VN-limit-counts, health, volume-trend) — **chết sau trực giao hóa**.

### Verdict trung thực

**OHLCV chưa cạn tuyệt đối, nhưng đất trống còn lại rất hẹp.** Toàn bộ các kênh "trống
hoàn toàn" nghe hứa hẹn (gap qua đêm, candle anatomy k-bar, nén range, tần suất trần/sàn,
health) đều KHÔNG mang thông tin trực giao dùng được trên universe này — phù hợp quy luật
16 thế hệ rằng đất dễ đã bị cày hết. Kênh duy nhất sống sót là **microstructure khối
lượng–giá liên tục** (pv_corr, và họ hàng amihud/CLV yếu hơn): mức IC trực giao 0.035–0.04
là NHỎ so với cú nhảy composite ≥+5 điển hình cần label mới + thông tin mới, nhưng nó (a)
vượt null ~5σ, (b) mạnh nhất đúng ở phía EXIT (vexit +0.041, 5/5 fold) — khớp với lịch sử
là mọi cú nhảy đều đến từ exit-label + thông tin mới.

**Đề xuất bước tiếp (rẻ → đắt):** một thí nghiệm duy nhất đáng đốt — clone feature set
`exit_vol_downpress` + {pv_corr_10, clv, nr_pos_7} (và biến thể entry set + pv_corr_10)
theo pattern `rq_probe.py`, seed 42 trước. KHÔNG mở chiến dịch rộng trên các kênh khác;
số liệu ở đây nói chúng không có gì để khai.

*Files: screen_ic.py (harness), screen_results.json/.csv (số đầy đủ), rank_cands.py (bảng xếp hạng).*
