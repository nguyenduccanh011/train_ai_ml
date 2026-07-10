# P3 SCREEN V2: PHÁI SINH INTRADAY THẬT CHO EXIT 2024+ — HỒ SƠ GO CÓ ĐIỀU KIỆN

Ngày: 2026-07-10. Scripts + dữ liệu: `p3_probe/p3_ingest_intraday.py`, `p3_probe/p3b_intraday_features.py`,
`p3_probe/p3c_deriv_screen_v2.py`, output `p3_probe/p3_ingest_report.json`, `p3b_intraday_features.parquet`,
`p3c_ic_v2.csv`, `p3c_t3_rallied_v2.csv`. Tiếp nối `exitmap/DERIV_EXIT_SCREEN.md` (vòng daily proxy: 0/13 chết)
và `P3_INTRADAY_PROBE.md` (nguồn Entrade). KHÔNG đụng DB/bảng production; KHÔNG sửa engine.

## 1. Phase A — Ingest `market_data/market_intraday.duckdb` (DB MỚI)

Bảng `ohlcv_intraday` PK `(symbol, timeframe, ts TIMESTAMP UTC)` — sửa đúng lỗi PK-DATE gốc; kèm `_fetch_manifest`.
18 request (1 năm/request), toàn bộ OK, < 2 phút.

| symbol | rows | ts_min (UTC) | ts_max (UTC) |
|---|---|---|---|
| VN30F1M | **446.950** | 2018-08-13 02:00 | 2026-07-10 03:18 |
| VN30F2M | **281.075** | 2018-08-13 02:00 | 2026-07-10 03:17 |

**Verify close 1m-cuối-ngày vs close 1D production (read-only):** F1M khớp **99,56% exact / 100% ≤50bp**
(med lệch 0 bp); F2M **98,90% exact / 99,95% ≤50bp**. → Chuỗi intraday và daily production cùng nguồn gốc, tin được.

**Bar/ngày:** F1M median 243 (241 từ 2025 — đổi vi cấu trúc phiên), chỉ 4 partial day < 200 bar bị loại.
F2M median 117–191 (chỉ có bar khi khớp lệnh — đúng probe).

**Map lỗ hổng coverage (ngày có 1D production nhưng thiếu intraday):** F1M thiếu 377 ngày =
(i) 2017-08 → 2018-08-10: **trước khi nguồn Entrade bắt đầu** (2018-08-13) — không vá được;
(ii) **2023-03 (7d), 2023-04→08 (~100d), 2023-09 (13d)** — đúng lỗ ~6 tháng đã biết;
(iii) lẻ tẻ: 2020-07 (3d), 2024-03 (3d). Vùng đau exit **2024+ phủ ~99%**. Xử lý lỗ 2023: feature = NaN,
fold 2023 chỉ tính trên ~113–126 ngày có dữ liệu (ghi n rõ trong CSV).

## 2. Phase B — 15 feature intraday thật (từ 1m F1M; slope dùng thêm F2M)

**Timing chống lookahead:** mọi feature của ngày t chỉ dùng bar 1m BÊN TRONG phiên t, giờ chốt = **ATC close 14:45**
(bar cuối) — dùng cho quyết định cuối phiên t, fill close(t+1), đồng nhất timing với dx04.

`rv_park` (Parkinson 5m), `rv_5m` (std ret 5m), `mom_30`/`mom_60`/`mom_pm` (momentum 30'/60'/cả buổi chiều tới ATC),
`vol_imb_pm` (chênh volume chiều−sáng), `gap_open` (open 09:00 vs ATC hôm trước), `range_c_id`, `range_comp`
(nén biên độ vs 20d), `chop5` (số đảo chiều 5m), `maxdd_15m` (tốc độ rơi 15' tệ nhất), `vwap_dev` (ATC vs VWAP),
`atc_volshare` (tỷ trọng volume 14:30→ATC), `slope_id` (mean trong phiên (F2M−F1M)/F1M), `backwd_id` (streak slope_id<0).
1.841 ngày feature; winsorize 0.5/99.5% như dx04.

## 3. Phase C — IC screen v2 (tái dùng nguyên harness dx04/dx06)

Time-series Spearman theo fold năm 2022→2026H1; trực giao = residual rank-OLS trên **8 control daily hệ ĐÃ thấy**
(proxy_ret5/20, dist_ma20/50, breadth_ma50, SNR20 universe, drop5_z, proxy_rv10); null = **circular-shift**
(300 vòng vòng chính; ứng viên top chạy lại **1000 vòng, seed khác**). Target y hệt dx04: T1 open_dd10 /
T1b open_gb5 (giveback lệnh mở), T2 univ_dd10, T3 rallied event-level (shift-null 200).

### 3a. Bảng chính — IC trực giao pooled (T1/T1b), null 1000 vòng seed 7

| feature | target | ort ≥2022 | null ≥2022 | p2t | ort 2024+ | null 2024+ | p2t | fold-sign |
|---|---|---|---|---|---|---|---|---|
| **mom_pm** | T1_open_dd10 | **+0.064** | (−0.064,+0.060) | .046 | **+0.096** | (−0.070,+0.068) | **.003** | **5/5 +** |
| **mom_pm** | T1b_open_gb5 | **+0.077** | (−0.066,+0.060) | .018 | **+0.097** | (−0.065,+0.063) | **.002** | **5/5 +** |
| **mom_60** | T1_open_dd10 | **+0.062** | (−0.050,+0.053) | .016 | **+0.070** | (−0.066,+0.063) | .041 | **5/5 +** |
| mom_60 | T1b_open_gb5 | +0.074 | (−0.050,+0.061) | .014 | +0.058 | (−0.072,+0.065) | .105 trong | 5/5 + |
| vwap_dev | T1b_open_gb5 | +0.062 | (−0.059,+0.057) | .039 | +0.061 | (−0.068,+0.059) | .060 sát | 5/5 + |
| atc_volshare | T1_open_dd10 | −0.082 | (−0.160,+0.135) trong | .466 | **−0.161** | (−0.102,+0.138) | **.002** | 5/5 − |
| atc_volshare | T1b_open_gb5 | −0.081 | (−0.167,+0.148) trong | .523 | **−0.167** | (−0.110,+0.149) | .011 | 5/5 − |

Fold-detail mom_pm×T1: 2022 +0.105 / 2023 +0.037 / 2024 +0.133 / 2025 +0.105 / 2026H1 +0.002 (n=933 pooled, n24=580).

Các feature còn lại: slope_id (ort −0.166) và backwd_id (+0.157) biên độ lớn nhất nhưng **trong** null band rộng
(persistent — vài episode ≈ vài quan sát độc lập, y hệt bài học slope daily); rv_park/rv_5m/range_c_id raw lớn
(+0.14…+0.18) nhưng residual còn 0.02–0.07 trong band → chủ yếu là proxy_rv10 mặc áo mới (đúng bài pv_corr);
mom_30, gap_open, chop5, maxdd_15m, range_comp, vol_imb_pm: chết hẳn.

### 3b. T3 rallied (event-level, shift-null giữ clustering)

≥2022 (n=1042): `mom_60` **−0.098 NGOÀI** null (−0.089,+0.109); mom_pm −0.099 sát lo (−0.142). 2024+ (n=560):
mom_60 −0.153 / mom_pm −0.110 đúng chiều nhưng trong band; chỉ vol_imb_pm +0.110 ngoài (đơn lát → coi là noise).
**Chiều T3 NHẤT QUÁN với T1**: PM-momentum futures ÂM mạnh tại decision bar → xác suất sold-then-rallied CAO hơn
(bán đúng đáy cú xả cuối phiên); PM-momentum DƯƠNG mạnh → giveback 10 bar tới của lệnh mở cao hơn (fade cú kéo).

### 3c. Đối chiếu chuẩn kill/go + caveat trung thực

Chuẩn: |IC trực giao| ≥ ~0.04 VÀ đúng chiều ≥4/5 fold VÀ ngoài shift-null. **mom_pm đạt CẢ BA ở cả hai lát
(≥2022 VÀ 2024+), trên cả hai target giveback, p2t 2024+ = 0.002–0.003** — Bonferroni thô ×15 feature vẫn < 0.05
ở lát 2024+. Khác biệt CHẤT so với vòng daily (0/13, không gì ngoài null trung thực ở bất kỳ lát nào) và so với
pattern "mỗi lát vượt bằng feature khác nhau" của dx04-T3. mom_60/vwap_dev cùng họ, yếu hơn → coi cả cụm là
**MỘT tín hiệu: PM-exhaustion của futures**. Caveat: (1) biên độ 0.06–0.10 chỉ ngang pv_corr (+0.041, 5σ) —
pv_corr từng chết ở decision layer, screen-pass ≠ decision-value; (2) atc_volshare chỉ vượt lát 2024+ (band ≥2022
rộng do persistence) — ứng viên phụ, chưa đủ chuẩn hai lát; (3) 2023 fold mỏng (~113 ngày); (4) target T1/T1b
xây từ trades gbx08 — selection theo hệ hiện tại.

**Check confound ret1 (ĐÃ CHẠY, không chờ bước kế):** confound lớn nhất là "mom_pm = return ngày t mặc áo mới"
(bộ control gốc chỉ có proxy_ret5, chưa có ret1). Residualize lại trên **11 control** (thêm mret1 EW proxy,
f1_co1 thân nến daily F1M, f1_ret1) + shift-null 1000 vòng seed 11: mom_pm **vẫn NGOÀI null cả 4 ô**, và ở lát
2024+ IC còn TĂNG: T1 +0.133 (p=0.001), T1b +0.108 (p=0.008) — tín hiệu nằm ở HÌNH DẠNG trong phiên
(kéo/xả buổi chiều), không phải độ lớn return ngày t. mom_60: 3/4 ô ngoài.

## 4. VERDICT: **GO CÓ ĐIỀU KIỆN** — kênh intraday phái sinh MỞ, đúng 1 tín hiệu

1. Kênh intraday thật KHÁC kênh daily proxy: daily chết 0/13, intraday cho ra **1 họ tín hiệu sống qua null
   trung thực ở chính vùng force-rules độc quyền 2024+**: `mom_pm` (đại diện họ; mom_60/vwap_dev là biến thể).
2. Nghĩa kinh tế: cú kéo (xả) cuối phiên của VN30F1M — thành phần TRỰC GIAO với toàn bộ price/vol daily hệ đã thấy —
   báo giveback (hồi phục) 10 bar tới của danh mục lệnh mở. Đây đúng loại thông tin force gates KHÔNG có
   (gates chỉ thấy daily close).

### Thiết kế bước kế (chưa code engine — theo bài học SCORE_AUDIT: kênh độc lập, KHÔNG blend)

- **Dạng dùng: risk-gate hai phía "PM-exhaustion"** cạnh force gates, KHÔNG cộng vào score nào:
  (a) phía tighten: cuối phiên t, nếu mom_pm (residual online hoặc raw) > **quantile riêng** q90 trailing 252 phiên
  → siết trailing/defer add cho lệnh mở; (b) phía hold: nếu mom_pm < q10 VÀ lệnh đang chạm exit-rule mềm
  → defer exit 1 bar (chống sold-then-rallied — chiều T3).
- **Ngưỡng quantile riêng theo chính phân phối mom_pm** (rolling), không tái dùng threshold hệ cũ; walk-forward
  2024/2025/2026H1 phải giữ hit-rate ổn định.
- **Validation kill-gate kế tiếp (offline replay, trước khi đụng engine):**
  (i) event study phân phối forward giveback_u của lệnh mở theo bucket mom_pm (q90+ vs giữa vs q10−) theo năm;
  (ii) replay trên 1378 trades gbx08: defer-exit khi mom_pm < q10 → Δu và Δrallied-rate cohort 2024+;
  tighten khi > q90 → Δgiveback_u; chuẩn sống: cải thiện ≥2024 không làm giảm eff — fail thì kill như pv_corr;
  (iii) ablation chọn 1 đại diện họ (mom_pm vs mom_60), tránh đếm trùng;
  (iv) kiểm tra cơ chế beta-noise: **ĐÃ CHẠY** (mục 3c) — mom_pm sống, thậm chí mạnh hơn, sau control ret1;
  còn lại: tách hẳn thành phần basis (mom_pm F1M − mom_pm proxy intraday nếu sau này có spot 1m).
- Ứng viên phụ theo dõi cùng replay (không tự đứng): `atc_volshare` thấp bất thường 2024+ → giveback cao.

KHÔNG commit git. DB mới `market_intraday.duckdb` chỉ phục vụ research; production không đổi.
