# HỒ SƠ CÔNG TỐ — r2c_oxt04_p42 (t2900, hệ 0-ML fc_rule2 + pb4.2/50 + overext_trail 0.04 + snr g12)
Ngày: 2026-07-10. Công tố viên adversarial, tiền lệ 6 REJECT + 1 lật khung đo.
Thước: `nh_nav2.py` (K25, R0.6, lag2+advance 0.08%/vòng, shuffle-mean±sd 20 perm) — đã tái lập
ĐÚNG từng chữ số các số công bố: full ×16.838±0.754 (+22.2%±6.6, 3.4sd), f22 ×3.758±0.086 (+9.4%±3.6, 2.6sd),
f23 ×2.605 (+5.7%±3.5, 1.6sd), MaxDD −13.28% mọi perm (gb −15.28/worst −15.61). Mọi delta dưới đây = delta sạch.
Scripts/bằng chứng: `pr3/pr3_00_cfg.py, pr3_10_selection.py (+pr3_10_scores.csv/out), pr3_11_heldout.py,
pr3_lib.py (verify series=run từng chữ số), pr3_40_loyo.py, pr3_30_trail.py, pr3_21_droptop.py, pr3_60_stress.py` (+ *_out.txt).

## TỘI 1 — Selection inflation (nặng nhất): THÀNH LẬP MỘT PHẦN (phần pick-premium), tuyến gỡ được
Quét TOÀN BỘ 56 config đã thử 3 vòng (r2_/r2b_/r2c_) trên đúng thước chọn (K25 adv):
- **p42 = rank 56/56 ở CẢ HAI khung** — headline đúng nghĩa là MAX của search. Phân phối delta vs gb:
  full min −22.1 / q25 +2.4 / **med +5.5** / q75 +10.5 / max +22.2(=p42); f22 min −18.2 / **med +1.0** / q75 +4.5 / max +9.4(=p42).
- Lát KHÔNG dùng khi chọn (n=20 perm):
  - **f21**: p42 +24.8%±5.2 (**4.8sd**) — sống.
  - **f24**: p42 +7.7%±2.5 (**3.1sd**) — sống, NHƯNG p42 là điểm **TỆ NHẤT cả họ** (p41 +9.8/4.0sd, oxt03 +9.1, c2 +9.0, base +9.0): phần premium riêng của knob 4.2 (+22.2 vs plateau +19.6) KHÔNG tái lập ngoài tập chọn — đó là noise selection.
  - **f25**: p42 +1.1%±1.8 (**0.6sd**) — TOÀN TUYẾN chết về noise ở 18 tháng gần nhất (mọi config +0.7..+1.9%, ≤1.2sd); lợi thế DD cũng biến mất (p42 −11.9 vs gb −11.4).
- Kết luận: **alpha của TUYẾN là thật ngoài tập chọn lọc (f21 4.8sd, f24 3.1sd), nhưng con số 22.2/9.4 là đuôi max**; expected value đúng = mức plateau/họ hàng: **full ~+17..+20%, f22 ~+6..+7%** (bào chữa đã tự khai chiết khấu này ở R3 §dè-chừng-1 — ghi nhận thành khẩn).

## TỘI 2 — Episode concentration: GỠ (kèm ghi chú 2021)
- Yearly delta NAV (điểm %): 2020 −0.7 / **2021 +25.2** / 2022 +3.6 / 2023 +7.6 / 2024 −0.9 / 2025 +5.0 / 2026 −1.6 — 2021 gánh quá nửa headline full-frame; monthly log-delta: 47/78 tháng dương, top-3 tháng dương = 60% tổng, episode 60d mạnh nhất (kết thúc 2021-04-19) = 58% tổng — CÓ concentration thời-đoạn, nhưng hai chiều (tháng âm lớn nhất 2020-03 −27% tổng).
- LOYO bỏ 2021 vẫn **+9.3%±4.3 (2.2sd)** → không sụp khi rút episode lớn nhất.
- **Drop-top-trade CẢ HAI bên (đòn quyết định)**: bỏ top-10 → delta full **+34.1% (4.2sd)**, f22 +7.6% (2.1sd); bỏ top-20 → **+46.9% (7.7sd)**, f22 +12.0% (3.3sd). Lợi thế p42 TĂNG khi cắt đuôi — **gb mới là hệ sống nhờ top-trade** (PF 6.0, avg +9.3%/lệnh), p42 trải đều 2012 lệnh. Tội không thành lập.

## TỘI 3 — Cơ chế overext_trail 4%: GỠ (không lookahead; caveat regime)
Đọc engine.py (2085-2088, 2176-2179, 2312-2313, 2405-2411) + forensic OHLC sqlite trên 400 lệnh overext_trail:
- Trigger: `lows[i] <= peak_high*(1−0.04)`; **fill = close phiên i+1** (`close_next`) — KHÔNG fill tại mức trail, KHÔNG dùng giá intraday để fill → không có kênh lookahead giá. Tín hiệu quan sát được cuối phiên trigger, đặt lệnh phiên sau = thực thi được.
- Intrabar ambiguity (peak cập nhật bằng high CÙNG bar trước khi so low): 122/194 lệnh tái dựng được trigger trên bar vừa lập peak — giả định high-trước-low làm engine bán SỚM hơn (hướng bất lợi cho knob ride, không phải thổi phồng). 206/400 lệnh không tái dựng được do khác basis điều chỉnh giá CSV-vs-DB (trung tính, đã ghi nhận).
- Gap risk có trong giá: fill vs mức trail mean +1.49% nhưng p10 −3.84%, 35% lệnh fill DƯỚI trail, 27/194 gap-through >2% — phân phối thực tế, đã nằm trong pnl backtest.
- Vì sao sống ở đây mà chết ở gb: ở gb "siết trail 8%→/arm15%" là SIẾT kênh exit mặc định (cắt cụt đuôi winner ML −22.8u); ở đây trail04 là NỚI — thay lệnh bán thẳng `overext` bằng ride có khóa (+4.0đ trên 359 lệnh, dồn hết vào cohort winner<10%, loser không đổi). Cơ chế thật khác nhau. CAVEAT: d_pnl của knob theo năm = trend-year knob (2024 −0.11, 2026 −0.16) — đồng dạng với điểm yếu lát 2024/2026.

## TỘI 4 — Robustness dữ liệu: GỠ PHẦN LỚN (LOYO 7/7); phai gần đây = ĐIỀU KIỆN
- **LOYO trên NAV (per-perm, 20 perm): 7/7 năm PASS ≥2sd** — yếu nhất bỏ 2021: +9.3%±4.3 (2.2sd); mạnh nhất bỏ 2024: +23.1% (4.0sd). Bỏ 2020+2021 cùng lúc: +9.7%±3.8 (2.5sd) — khớp f22 độc lập (+9.4, 2.6sd).
- Xu hướng thời gian KHÔNG đơn điệu: f21 4.8sd → f22 2.6sd → f23 1.6sd → **f24 3.1sd** → **f25 0.6sd**. Không phải phai tuyến tính, nhưng **lát tươi nhất (18 tháng) = không phân biệt được gb, và DD-edge = 0 ở lát này**. Cửa sổ quá ngắn để kết tội (sd ~1.8%), đủ dài để bắt đặt tripwire.

## TỘI 5 — Serving/parity: THÀNH LẬP 1 LỖ HỔNG CỤ THỂ (sửa được, fail to tiếng)
- Wheel production `stock_ml_core 0.3.3`: CÓ overext_trail_pct, entry_pullback_pct/window, exit_snr_extend_threshold/window/min_gain, nonbull_*, downtrend_hard_stop_pct, market gates, `model_mode=rule_only` (0-ML chạy được). **THIẾU `exit_snr_defer_min_giveback`** (config t2900 đặt 0.08; engine repo có tại engine.py:329). Wheel dựng config bằng `EngineConfig(cost=cost, **engine_cfg)` strict-dataclass → bundle sẽ **CRASH to tiếng**, không lệch ngầm. Impact key nhỏ (vòng 1: g12 vs g12_nogb lệch 0.01 NAV) nhưng BẮT BUỘC bump wheel 0.3.4 (hoặc bỏ key + re-verify trades bit-level).
- Tiền lệ live: champion wavestruct ĐANG chạy live `overext_trail_pct 0.04` + pullback keys + market gates + nonbull → 5/7 knob có tiền lệ. **snr_extend (4 key) CHƯA TỪNG live**; dtstop −6% và pb 4.2/50 chỉ khác GIÁ TRỊ key đã live.
- Warmup: cần ~60-120 bar (MA20/40, z_lookback 60, snr w20, pb w50) — serving giữ lịch sử nhiều năm, đủ; 0-ML không cần retrain. Điều kiện: chạy leakage-auditor 4-check trên bundle trước promote (thủ tục chuẩn).

## TỘI 6 — Composite counter-check + friction: THÀNH LẬP ĐIỀU KIỆN THỰC THI (bằng chứng nặng nhất phiên tòa)
- Giải phẫu comp 588 vs gb 735: per-trade edge p42 **+5.19%/lệnh** (PF 3.50, med +2.92%, hold 8d, 2012 lệnh) vs gb **+9.32%/lệnh** (PF 6.02, hold 14d, 1378 lệnh) — p42 thắng NAV thuần bằng VÒNG QUAY, edge/lệnh mỏng gần nửa gb → mọi friction per-order ăn thẳng vào alpha.
- Phí roundtrip: R0.7 → f22 +8.0% (2.2sd) sống; R0.8 → +6.9% (**2.0sd**) mấp mé; **R0.9 → +5.5% (1.6sd) CHẾT chuẩn 2sd** (full còn 2.1sd). Trần sống: R ≤ 0.8%.
- **Entry lag +1 phiên (fill close ngày kế, áp CẢ HAI bên): LẬT KÈO** — p42 ×10.45 vs gb ×10.97: full **−4.8%±4.3 (−1.1sd)**, f22 **−5.3%±2.4 (−2.2sd)** = gb thắng ngược có ý nghĩa. p42 mất 38% NAV vì trễ 1 phiên, gb chỉ mất 20%. Toàn bộ lợi thế của bị cáo nằm ở việc mua ĐÚNG close phiên kế sau tín hiệu pullback; với ~2000 lệnh/6.5 năm (~1.2 lệnh/ngày, K25 slot) đây là yêu cầu kỷ luật thực thi nghiêm ngặt, không phải auto-đạt với retail.
- (Ứng trước 0.08%/vòng là điều-kiện-tồn-tại đã khai ở NAV_V2_FRONTIER — tái xác nhận, không truy thêm.)

## PHÁN QUYẾT: **CONDITIONAL** (không REJECT, không SURVIVED sạch)
Không có tội tử hình: không lookahead (T3 gỡ), không concentration lệnh (T2 gỡ — còn tốt hơn gb), LOYO 7/7 ≥2sd (T4),
alpha tuyến sống ở lát held-out f21/f24 (T1). Nhưng 4 điều kiện ràng buộc, vi phạm cái nào thì lợi thế vs gb biến mất hoặc âm:
1. **Chiết khấu số công bố về plateau**: expected value = full **+17..+20%**, f22 **+6..+7%** (KHÔNG dùng 22.2/9.4 — đó là max của 56 config; premium riêng knob 4.2 âm ở f24).
2. **Kỷ luật thực thi = điều kiện sống mới** (bổ sung vào cost model cùng advance-fee): fill trong phiên close T+1; nếu vận hành thực tế trễ +1 phiên thường xuyên → gb thắng ngược (−5.3% f22, 2.2sd). Cost thật phải ≤0.8% roundtrip (chết 2sd ở 0.9%).
3. **Serving**: bump wheel ≥0.3.4 thêm `exit_snr_defer_min_giveback` (hoặc bỏ key + re-verify bit-level); snr_extend chưa từng live → leakage-auditor + shadow-run trước khi cấp vốn.
4. **Tripwire recency**: f25-tới-nay đang noise (+1.1%, 0.6sd) và DD-edge = 0; nếu rolling-18-tháng vs gb < −1sd trong 2 quý liên tiếp → demote về gb.

## BẢNG HỘI ĐỒNG CUỐI — r2c_oxt04_p42 vs gb_x08 (nh_nav2 K25 adv R0.6, mean±sd 20 perm)
| tiêu chí | r2c_oxt04_p42 | gb_x08 | ghi chú công tố |
|---|---|---|---|
| NAV full 2020- | ×16.84±0.75 (+22.2%, 3.4sd) | ×13.78±0.42 | max-of-56; EV chiết khấu +17..+20% |
| NAV f22 | ×3.76±0.09 (+9.4%, 2.6sd) | ×3.43±0.08 | EV chiết khấu +6..+7% (plateau) |
| f23 / f24 / f25 | +5.7% (1.6sd) / +7.7% (3.1sd) / **+1.1% (0.6sd)** | ref | f24 held-out PASS; f25 noise → tripwire |
| LOYO | 7/7 ≥2sd (min +9.3% bỏ 2021) | ref | robust dữ liệu |
| MaxDD full (mean/worst) | **−13.28/−13.28%** | −15.28/−15.61% | edge DD = 0 ở f25 (−11.9 vs −11.4) |
| drop-top-20 trade | +46.9% (7.7sd) | ref | p42 ÍT tựa top-trade hơn gb |
| per-trade edge | +5.19%/lệnh, PF 3.5, 2012 lệnh, hold 8d | +9.32%, PF 6.0, 1378 lệnh, hold 14d | p42 = turnover engine, mỏng/lệnh |
| phí: R0.7/0.8/0.9 (f22) | 2.2sd / 2.0sd / **1.6sd chết** | ref | trần cost 0.8% |
| **entry lag +1 phiên** | **−4.8% full/−5.3% f22 → THUA gb** | thắng | điều kiện kỷ luật thực thi |
| điều kiện tồn tại | retail R≤0.8 + ứng trước 0.08%/vòng + fill T+1 close | không cần ứng, chịu trễ tốt | |
| serving | 0-ML rule_only OK; wheel thiếu 1 key (crash to tiếng); snr chưa live | đang chạy dạng ML | bump wheel + shadow-run |
