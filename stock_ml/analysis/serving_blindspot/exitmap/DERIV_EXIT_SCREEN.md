# SÀNG LỌC TÍN HIỆU PHÁI SINH VN30F1M/F2M CHO EXIT 2024+ — HỒ SƠ ĐÓNG KÊNH

Ngày: 2026-07-10. Scripts + dữ liệu: `exitmap/dx00_peek.py … dx06_shiftnull.py`,
`dx04_deriv_ic.csv`, `dx04_t3_rallied.csv`. Offline thuần — không run leaderboard, không sửa engine.

## 0. Chẩn đoán "vì sao velocity head im lặng từ 2024" (dx03)

Ba mảnh chứng gián tiếp (không có model scores lưu lại):

**C — Pre-emption bởi force gates (cơ học, mạnh nhất).** % exit có force flag (dl12/nonbull/lowbreadth)
tại decision bar theo năm exit: 2020 **55.6%** → 2023 76.4% → 2024 **94.3%** → 2025 **99.5%** → 2026 **100%**;
hold_med rơi về 8.5–11 bar từ 2022. Không gian residual cho head (label `head_signal` = exit không flag)
co về 0 một cách cơ học: lệnh chết vào tay price-rule trước khi head kịp có vai trò closer.

**A — Thông tin nền của head luôn mỏng và trùng với force gates.** IC cross-sectional (486 mã, panel daily)
của 8 feature price/vol lõi vs chính label `velocity_exit_regression(h20,u8,volnorm40)` theo năm:
|IC| ≤ 0.044 ở MỌI năm, dấu lật cả bảng năm 2021 (melt-up: âm toàn bộ); 2024 hỗn hợp
(ret_5d +0.031, dist_10d_low +0.042, rsi_14 −0.029). Không có "sụp đổ 2024" — thông tin chưa bao giờ dày,
và 2 feature khỏe nhất (dist_10d_low, ret_5d = độ sâu pullback) chính là thông tin downleg/zigzag mà
force gates đã hành động trực tiếp.

**B — Cơ hội exit vẫn còn.** Phân phối label vexit: daily q90 2024/2025/2026H1 = 5.45/5.46/5.67
vs pooled 2020–23 = 5.25 → không phải "hết sự kiện đáng exit".

**Kết luận bước 0: (b) là chẩn đoán đúng, ở nghĩa rộng.** Head im không phải vì phân phối score trôi
khỏi threshold cố định (recalibration ≤+1.5 không đổi được gì khi force-gate coverage 94–100% và
phần residual head bắn được năm 2024 cho WR 8% — anti-signal). Feature price/vol không còn/không từng có
thông tin ĐỘC LẬP với force gates cho label này; head exit mới bắt buộc cần **thông tin ngoài price/volume
daily** — đúng câu hỏi kênh phái sinh dưới đây.

## 1. Phát hiện data-integrity: INTRADAY KHÔNG TỒN TẠI (dx00–dx02)

Câu "market.duckdb có intraday 1m/5m VN30F1M/F2M từ 2018-08" trong NICHE_LOSS_MAP là **sai về granularity**:

- `ohlcv` có PK `(symbol, timeframe, date)` với `date` kiểu **DATE không có giờ** → mọi timeframe intraday
  (1m/5m/15m/30m/1H) bị upsert đè còn **đúng 1 bar tùy tiện/ngày** (1m: 1797 row/8 năm; close bar "1m"
  2026-05-05 = 2014.4 vs close 1D = 2038.8 — bar giữa phiên ngẫu nhiên, không phải bar cuối).
- `market.duckdb.bak` (76MB): sập y hệt. `market_raw_api.duckdb`: chỉ daily stocks 488 mã. Không có
  VN30 spot index ở bất kỳ nguồn nào (kể cả stock-serving ohlcv.db).
- Hệ quả: mọi feature intraday thật (RV trong phiên, momentum 30–60' cuối, imbalance sáng/chiều) **bất khả thi
  với dữ liệu hiện có**. Muốn mở lại phải refetch từ API với schema có cột time — việc data-engineering,
  ngoài scope offline.

Kênh khả dụng duy nhất còn lại: **daily VN30F1M/F2M OHLCV+volume** (2017-08-10 → 2026-06-16, ~2200 phiên) —
vẫn là thông tin hệ chưa từng thấy (giá futures, term structure, gap, volume phái sinh). Screening tiến hành
trên kênh này, với thay thế trung thực:
- basis LEVEL (F1M − spot) bất khả thi (không có VN30 spot) → **basis-change proxy** = ret(F1M) − ret(EW proxy 61 mã) (1/5/10d);
- backwardation streak dùng calendar spread **F2M < F1M**;
- Timing: bar daily t chốt ATC 14:45 cùng lúc cổ phiếu; engine quyết định bar t, fill close(t+1) → **không lookahead**.

13 feature: slope (F2M−F1M)/F1M, slope_chg5, gap1, gap_abs5, range_c, range_r520, rv10, clv5, co5,
volz20, backwd_streak, f1_ret5, f1_ret10 (winsorize 0.5/99.5% — hấp rollover artifact; caveat: chuỗi F1M
là front-month thô, ngày đáo hạn nhiễu ~1 lần/tháng).

## 2. Thiết kế IC screening (dx04–dx06)

Tín hiệu MARKET-LEVEL (1 chuỗi) → IC = **time-series Spearman theo fold năm** (2022, 2023, 2024, 2025, 2026H1),
khác harness cross-sectional `ohlcv/screen_ic.py`. Trực giao = residual rank-OLS của feature trên 8 control
hệ ĐÃ thấy: proxy_ret5/20, dist_ma20/50 (EW proxy), breadth pct_above_ma50 full-univ, SNR20 universe,
drop5_z (market-drop gate), proxy_rv10. **Null = circular-shift 300–500 lần** (giữ autocorrelation cả feature
lẫn target — trung thực hơn shuffle-within-date vốn chỉ đúng cho cross-section), band 2.5/97.5.

Target đúng chỗ đau (từ `gbx08_enriched2.csv`, 1378 lệnh):
- **T1 open_dd10**: mean trên các lệnh ĐANG MỞ tại t của forward-10-bar drawdown symbol lệnh ("sóng sắp gãy" tầng danh mục); n=1060 ngày ≥2022.
- **T1b open_gb5**: tỷ lệ lệnh mở rớt ≥5% trong 10 bar tới.
- **T2 univ_dd10**: forward-10-bar drawdown của EW proxy.
- **T3 rallied**: event-level tại decision bar — tách cohort sold-then-rallied (663; rate 0.43 ở ≥2022) vs sold-đúng.

## 3. Kết quả

### T1/T2 (time-series, IC trực giao pooled ≥2022 / pooled 2024+ vs null 95%)

Top theo |IC trực giao| (T1 open_dd10) — TẤT CẢ nằm TRONG null band:

| feature | ort ≥2022 | null ≥2022 | ort 2024+ | null 2024+ | raw ≥2022 |
|---|---|---|---|---|---|
| f1_ret10 | +0.085 | (−0.111, +0.123) | +0.084 | (−0.179, +0.171) | −0.046 |
| co5 | +0.080 | (−0.090, +0.098) | +0.084 | (−0.149, +0.141) | −0.061 |
| gap_abs5 | +0.067 | (−0.109, +0.115) | +0.082 | (−0.182, +0.158) | +0.113 |
| slope | −0.061 | (−0.142, +0.222) | −0.129 | (−0.240, +0.152) | −0.053 |
| backwd_streak | +0.058 | (−0.219, +0.168) | +0.127 | (−0.158, +0.225) | +0.032 |

(T1b/T2 cùng cấu trúc — xem `dx04_deriv_ic.csv`.) Feature persistent (slope, streak, vol-family) có null band
rất rộng — đúng bản chất: vài episode contango/backwardation ≈ vài quan sát độc lập. Raw IC của range_c
(+0.144…+0.157) vượt null NHƯNG residual còn +0.054/+0.026 → là proxy_rv10/breadth mặc áo mới — đúng bài học pv_corr.

### T3 rallied (event-level)

- Null permute-trong-năm (lạc quan): 3 feature "vượt" ở pooled ≥2022 (gap_abs5 −0.074, volz20 −0.077, co5 −0.071)
  — nhưng ở lát 2024+ cả 3 tụt vào trong null (gap_abs5 −0.079 vs lo −0.084; volz20 lật dấu +0.027; co5 −0.079 vs lo −0.096),
  và lát 2024+ lại "vượt" bằng 2 feature KHÁC (slope +0.097 — nhưng 2023 = **−0.186** lật dấu mạnh; backwd_streak −0.078).
  Không feature nào vượt ở CẢ hai lát; 13 feature × 2 lát ở mức 95% kỳ vọng ~1.3 false positive/lát — khớp số quan sát.
- **Null circular-shift theo ngày (trung thực — giữ clustering thời gian của cả feature lẫn label): KHÔNG feature nào
  ra ngoài band** ở cả ≥2022 lẫn 2024+ (vd slope 2024+ +0.097 vs (−0.135, +0.127); gap_abs5 ≥2022 −0.074 vs (−0.209, +0.146)).

### Đối chiếu ngưỡng đã lập (chuẩn pv_corr)

Ngưỡng sống: |IC trực giao| ≥ ~0.04 VÀ đúng chiều ≥4/5 fold VÀ ngoài null. Vài feature qua được 2 điều kiện đầu
ở một lát cắt, **không feature nào ngoài null trung thực**; tham chiếu: pv_corr đạt +0.041 với 5σ mà vẫn chết ở
decision layer — các ứng viên ở đây thậm chí không đạt 1σ theo shift-null.

## 4. VERDICT

1. **Kênh phái sinh (dạng daily khả dụng): CHẾT ở screening.** Không có thông tin trực giao dùng được cho
   exit 2024+ trên cả 3 loại target đau (giveback-sắp-tới của lệnh mở, drawdown universe, tách cohort
   sold-then-rallied). Không đề xuất head/gate/probe nào.
2. **Kênh intraday thật: chưa từng được test — vì data không tồn tại** (bị collapse tại ingestion, cả bản .bak).
   Đây là phát hiện data-integrity cần sửa vào NICHE_LOSS_MAP ("còn mở (b)" thực chất là "chưa có data").
   Nếu muốn mở: refetch intraday với schema có cột time (sửa PK `ohlcv`) — quyết định data-engineering riêng,
   KHÔNG phải nợ nghiên cứu; xác suất trước (prior) không cao vì bản daily của cùng kênh vừa chết sạch.
3. **Velocity head im lặng = (b) + pre-emption cơ học**: force gates phủ 94–100% decision bar từ 2024,
   residual của head là noise (WR 8%). Recalibration threshold không cứu; head exit mới đòi hỏi nguồn thông tin
   mới — sau hồ sơ này, trong dữ liệu đang có KHÔNG còn nguồn nào chưa đào. Exit stack hiện tại (pure price-rule)
   là trạng thái bão hòa với toàn bộ data hiện hữu.

Cùng NICHE_LOSS_MAP và EXIT_ATTRIBUTION_MAP, hồ sơ này đóng nốt kênh dữ liệu cuối: alpha còn lại nằm ở
**sizing/multi-position** (đã có giá niêm yết +63.7u, chờ serving) — không nằm ở tín hiệu mới từ data cũ.
