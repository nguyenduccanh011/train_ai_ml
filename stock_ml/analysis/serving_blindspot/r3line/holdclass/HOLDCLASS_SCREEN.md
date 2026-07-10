# HOLDCLASS_SCREEN — Phân lớp trần tại bar entry (recycle 16 vs runner 40/60)

**Ngày:** 2026-07-10. **Câu hỏi:** train model dự báo cả trung hạn lẫn dài hạn — dạng khả thi duy nhất còn mở là phân lớp TẠI BAR ENTRY: lệnh nào đáng trần 16 (recycle), lệnh nào đáng trần 40-60 (runner) — có ăn thêm không?

**Cohort:** 1.027/2.451 lệnh r3_mh16_s42 chạm trần (`exit_reason=max_hold`, holding 17 bar, mean pnl +8.62%), entry 2020-04→2026-06.

**Verdict: CHẾT — oracle ceiling khổng lồ nhưng separator ex-ante = null (cùng kết cục án runaway).** Trần chung 16 + re-entry là tối ưu khả thi; câu hỏi multi-horizon khép bằng số.

## Phương pháp (hc_00/01/02/10/11)

- **Counterfactual giữ-tiếp:** exit stack tối giản = trần mới, close tại bar i1+24 (~trần 40 phiên) / i1+44 (~trần 60). KHÔNG tái lập signal/trailing trong extension (bảo thủ về phía đơn giản; oracle bù bằng chọn lệnh hindsight-perfect). Giá: `new_exit = exit_price × close[j]/close[i1]` — hợp lệ vì db back-adjusted, exit engine ≈ close×(1−S0) (ratio1≈0.9995, hc_00). Giữ tiếp không tốn thêm roundtrip; recycle tốn 0.6%/vòng.
- **Đối chứng chi phí cơ hội (control chain):** vốn giải phóng tại exit, T+2, vào lệnh THỰC TẾ kế tiếp (tie-break alphabet như sim), chain đến hết cửa sổ extension, leg cuối pro-rate theo bar, gap = cash. Net return chuẩn nh_nav2 (R=0.006).
- **NAV:** nh_nav2 K25, 2 chế độ (A lag2 noadv; B adv 0.08%), shuffle-mean 20 perm.

## (1) Bước 1 — Oracle ceiling

**Per-trade:** giữ-tiếp-thắng-recycle chỉ **47.4%** (487/1027) ở trần 40, **48.6%** ở trần 60. Edge mean **ÂM**: −1.24% (40), −2.86% (60) — recycle vô điều kiện đã tốt hơn giữ vô điều kiện (khớp R3 round 1: mù nâng mh40 làm NAV tụt ×21.77→×17.94). Edge dương chỉ ở 2023 (+3.3%/+6.5%) và 2026; âm nặng 2022/2025 (−5%/−10%).

**NAV oracle (hindsight-perfect, chỉ giữ lệnh giữ-tiếp-thắng):**

| variant | full adv | full noadv | f22 adv | f22 noadv | DD |
|---|---|---|---|---|---|
| mh16 base | ×21.77 | ×19.62 | 4.27 | 4.10 | −13.8% |
| oracle 16/40 | ×37.80 **(+74%)** | ×35.99 **(+83%)** | 5.64 (+32%) | 5.55 (+36%) | −14.1% |
| oracle 16/60 | ×38.51 (+77%) | ×33.57 (+71%) | 5.26 | 5.20 | −17.2% |
| oracle 16/40/60 argmax | ×49.63 (+128%) | ×43.45 (+121%) | 6.23 | 6.03 | −16.9% |

Oracle ≫ +5% → ý tưởng có trần kinh tế lớn, BẮT BUỘC xử bước 2. Nhưng lưu ý cấu trúc: label gần coin-flip (47%), giá trị nằm trọn ở việc CHỌN ĐÚNG — đúng loại bài toán mà 2 án trước (rem_MFE giữa-lệnh, runaway separator) đã fail.

## (2) Bước 2 — Separator ex-ante tại bar entry

14 feature chỉ dùng dữ liệu ≤ bar entry: ru5/20/60, dma20/60, vol20, snr60, volr (vol 5/20), rngpos20, dd252, fillgap (vị trí fill vs close), breadth, mkt_ru20, mkt_vol20. Label = hold_win40. n=1027, base rate 47.4%.

**Per-feature AUC (overall):** tất cả 0.50–0.53 — không feature nào tách. Vài ô per-year nhấp nháy (mkt_vol20 2022: 0.716, volr 2023: 0.616) nhưng không lặp lại năm khác — đúng kỳ vọng null nhiều phép thử.

**Logistic LOYO (leave-one-year-out):**

| năm | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | pooled | pooled ≥22 |
|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.610 | 0.479 | 0.597 | 0.591 | 0.641 | **0.429** | **0.441** | 0.532 | 0.530 |

Permutation null (n=200, trong-năm): p=0.005 — nhưng tín hiệu ~0.53 KHÔNG ổn định: đổi dấu 3/7 năm, sụp đúng 2025-2026 (giai đoạn gần nhất). Decile lift không đơn điệu: chỉ decile 9 dương (+1.5%), decile 8 âm (−2.3%).

**GBM (phi tuyến, LOYO):** pooled **0.496** (= random), ≥2022: **0.486**; 2022: 0.380, 2026: 0.421. Top-quintile trong năm: edge âm 5/7 năm (2022 top-quintile: −10.4%, TỆ hơn mù). Phi tuyến không cứu — fit noise.

**Đối chiếu án runaway (RUNAWAY_AUTOPSY.md):** cùng kết cục — feature tại bar tín hiệu không mang thông tin về hành vi giá 24-44 bar SAU trần 16. Đây là lần thứ 3 họ separator dài-hạn-trên-feature-ngắn-hạn ra null (xl_ ts40, runaway, holdclass).

## (3) Bước 3 — KHÔNG chạy

Điều kiện AUC ≥0.58 ổn định theo năm: FAIL rõ (0.53 linear sign-flipping / 0.50 GBM). Không sim 2-trần, không thiết kế head/label, không train.

## Verdict cuối

**Multi-horizon hold-class: CHẾT.** Con số quyết định:
1. Recycle vô điều kiện > giữ vô điều kiện (edge −1.24%/lệnh capped; mù mh40 mất ~4 NAV-x).
2. Oracle +74-83% NAV là hindsight thuần — muốn ăn phải phân lớp đúng ~gần-coin-flip 47/53.
3. Thông tin phân lớp KHÔNG tồn tại tại bar entry: AUC OOF 0.53 linear (đổi dấu 3/7 năm, chết 2025-26), 0.496 GBM, decile lift không đơn điệu.

→ **"Trần chung 16 + re-entry" là tối ưu khả thi** với thông tin có tại bar entry. Nếu muốn mở lại án này, phải có nguồn thông tin MỚI (không phải OHLCV-derived tại entry) — không phải feature engineering thêm trên cùng dữ liệu.

## File
- `hc_00_check.py` — verify giá CSV vs db (extension hợp lệ)
- `hc_01_oracle.py` → `hc_capped_ext.csv`, `hc_oracle_{h40,h60,best}_trades.csv`
- `hc_02_nav.py` — NAV K25 2 chế độ, shuffle-mean
- `hc_10_separator.py` → `hc_sep_features.csv` (per-feature AUC, LOYO logistic, null perm, decile)
- `hc_11_gbm.py` — GBM LOYO + top-quintile theo năm
