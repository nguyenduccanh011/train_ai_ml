# BẢN ĐỒ NGÁCH THUA CỦA TOP-1 (gb_x08) + FALSIFICATION "BỎ PULLBACK LẦN 2"

Ngày: 2026-07-09. Seed: 42. Scripts + dữ liệu: `signalq/nicheloss/` (np_depth_ladder.py, nl_map.py, nl_np_peryear.py, gb_x08_s42_trades.csv, *.log). Không đụng template 2646/2730/2783 — mọi biến thể là CLONE mới (np_* = 2786-2788, op_* = 2789-2790).

## PHẦN A — "Exit nhạy hơn rồi, bỏ pullback được chưa?" → **BÁC, lần thứ 4, chốt vĩnh viễn**

Ladder depth trên **exit stack MỚI** (clone gb_x08/2783: snr_extend 0.8/20/0.27 + defer_min_giveback 0.08) so với **exit CŨ** (clone champion 2646) — cùng depth, cùng window 40, seed 42. Bảng cũ chỉ có pb02_w10 (613.7) nên chạy thêm 2 control op_* để so cùng-toạ-độ.

| depth (w40) | exit CŨ (base 2646 = 729.6) | Δcũ | exit MỚI (base 2783 = 735.0) | Δmới | Δmới−Δcũ |
|---|---|---|---|---|---|
| 4.5% (gốc) | 729.6 | 0 | 735.0 | 0 | — |
| 3.0% | op_pb03 **696.1** (pf 4.883, mdd .2157, tr 1436) | **−33.5** | np_pb03 **702.3** (pf 4.945, mdd .2154, tr 1428) | **−32.7** | +0.8 |
| 2.0% | op_pb02 **651.6** (pf 4.261, mdd .2509, tr 1465) | **−78.0** | np_pb02 **657.4** (pf 4.311, mdd .2509, tr 1456) | **−77.6** | +0.4 |
| 0% at-market | a2_nopb **492.3** (pf 3.082, mdd .3363, tr 1528) | **−237.3** | np_atmkt **498.6** (pf 3.103, mdd .3354, tr 1522) | **−236.4** | +0.9 |

**Kết luận cấu trúc: độ dốc đường chảy máu Y NGUYÊN đến <1 điểm trên mọi nấc** (0.4–0.9 trên gap 33–237). Exit stack mới chỉ tịnh tiến cả đường lên ~5–6 điểm (đúng bằng edge của gb_x08 trên champion) rồi chảy máu theo đúng hàm cũ. PF/MDD degrade cũng trùng khít (pf 6.02→3.10, mdd .175→.335 ở at-market — giống hệt 5.96→3.08, .175→.336 của exit cũ). **Đệm giá 4.5% tại entry là alpha TRỰC GIAO với mọi cải tiến exit** — snr_extend/giveback vận hành trên lệnh ĐÃ có basis rẻ, không cứu nổi basis đắt. Câu "bỏ pullback" đóng vĩnh viễn, kể cả khi exit head còn tiến hóa tiếp.

Per-year pnl (u, theo năm entry) — chảy máu KHÔNG phải hiện tượng regime, nhưng nặng nhất ở năm xấu:

| năm | gb_x08 | np_pb03 | np_pb02 | np_atmkt | ghi chú |
|---|---|---|---|---|---|
| 2020 | 45.44 | 48.32 | 48.62 | 46.09 | melt-up: depth nông/at-market KHÔNG thua, thậm chí nhỉnh hơn |
| 2021 | 26.44 | 26.57 | 26.63 | 24.75 | ~hoà |
| 2022 | 11.34 | 9.81 | 8.04 | **2.87** | **bear: mất 8.5u — đệm 4.5% là bảo hiểm regime xấu** |
| 2023 | 18.04 | 17.37 | 16.85 | 14.66 | −3.4 |
| 2024 | 2.35 | 1.18 | 0.70 | **−1.81** | sideways chết: lật sang âm |
| 2025 | 25.42 | 26.03 | 25.30 | 23.09 | −2.3 |
| 2026 | −0.55 | −0.92 | −1.66 | −3.57 | −3.0 |

Chi tiết mới so với hồ sơ lineA: phần lớn cái giá của việc bỏ pullback nằm ở **2022/2024/2026 (năm xấu, −8.5/−4.2/−3.0u)**, còn 2020 melt-up thì at-market ngang ngửa. Đệm giá 4.5% về bản chất là position-level insurance chống regime xấu — đúng loại rủi ro mà exit stack (vốn phản ứng SAU khi đã vào lệnh đắt) không thể hoàn lại.

## PHẦN B — Bản đồ ngách thua của gb_x08 (trade-level, offline)

Nguồn: gb_x08 dump từ run_trades `template/gb_x08-32a8dfee` (1378 lệnh, 128.50u — khớp leaderboard); champ 2646 (1384, 127.25); dsb60 `vx_dsb60_trades.csv` (1494, 134.13); sx_w60/sx_w40 (1305/1332, 134.96/131.99); pyramid `week0/best_trades/trades_w0_pyr_u10_r02.csv` (1384, 192.18). Join khoá symbol+entry_date.

### PNL theo năm entry (u)

| năm | gb_x08 | champ | dsb60 | sx_w60 | sx_w40 | pyramid |
|---|---|---|---|---|---|---|
| 2020 | 45.44 | 44.34 | 48.43 | **62.26** | 54.93 | 69.72 |
| 2021 | 26.44 | 27.35 | **33.30** | 17.42 | 20.74 | 38.28 |
| 2022 | **11.34** | 11.17 | 8.59 | 11.18 | 11.10 | 14.47 |
| 2023 | 18.04 | 17.97 | **19.82** | 17.92 | 18.23 | 27.81 |
| 2024 | **2.35** | 2.35 | 1.75 | 2.35 | 2.35 | 2.52 |
| 2025 | **25.42** | 24.61 | 23.27 | 24.38 | 25.19 | 39.95 |
| 2026 | **−0.55** | −0.55 | −1.04 | −0.55 | −0.55 | −0.58 |

### Ngách từng ghost + phân loại

**sx_w60 (defer window 60)** — d_total +6.46u vs gb_x08, nhưng: matched Δ 2020 = **+16.5u** dồn vào entries 2020-08/09/10 (+5.4/+7.7/+4.2); 43 lệnh adv hold median 138 vs 88 bar, mã NKG/VND/LPB/HSG/DGC/NVL (thép–chứng khoán–bank sóng 2020H2). Trả giá ngay trong cùng melt-up: entries 2021-02/03 **−4.3/−5.1** (gãy pha 1 đầu 2021) + knock-on 2021 **−9.6u** (79 entry bị chặn vì slot bị lệnh defer chiếm). Mọi năm ≥2022 ≤ 0 (−0.17/−0.12/0/−1.04). **Phân loại: (i) regime-chết thuần** — máy bơm composite bằng đúng 3 tháng 2020, tự thua trong phần còn lại của chính melt-up. Khớp verdict autopsy cũ, giờ có toạ độ tháng.

**sx_w40** — bản nhẹ của w60: 2020 +9.5 (2020-09/10 +5.0/+2.7), 2021 −5.7 (knock-on), ≥2022 ~0 trừ 2025-04 +2.6 đơn lẻ. **Phân loại: (i).**

**dsb60 (dip-size-boost bull-mask MA60)** — d_total +5.63u, 2 nguồn: (a) matched hold-dài 2020 +5.7 (cùng họ w60); (b) **438 lệnh vonly +14.8u nhưng +14.0 nằm trọn 2021** (dip-buy melt-up: tháng 2021-05/08/02 +5.5/+4.9/+2.1); vonly các năm ≥2022 cộng lại **−2.1u**. Ngách sống duy nhất: 2023 +1.78u (vonly +2.6, tháng 2023-09/12) — nhỏ, không có discriminator đi trước (xem probe dưới), không đủ trả chi phí điều kiện hóa. **Phân loại: (i) là chính; ngách 2023 KHÔNG điều kiện hóa an toàn được** với các trục hiện có (bull-mask MA60 trễ pha và crash-brake ngày-1-gap đã chết, không đề xuất lại).

**pyramid (w0_pyr_u10_r02)** — **KHÔNG phải ngách regime**: thắng gb_x08 ở MỌI năm (+24.3/+11.8/+3.1/+9.8/+0.2/+14.5), matched adv 250 lệnh +72.7u, hold ~không đổi (57 vs 60), entries gần trùng 100% (1377/1378 matched). Đây là sizing alpha trực giao thuần — **+63.7u**, gấp 2.5 lần toàn bộ trần regime-switching và không phụ thuộc 2020-21. Đã deferred vì đòi multi-lot serving; con số này là opportunity cost đo được của quyết định đó.

### Probe tín hiệu regime NHANH (offline, duckdb universe 488 mã)

- Composite day-t thử: `(tỷ lệ mã trần 5 phiên ≥2%) & (tval_z60 > 0.5)`. Nhanh hơn MA60 thật (không lag 60 bar): bật 82–100% số ngày trong 2020-09..2021-01 (đúng lõi ngách w60), ~0% trong 2022-04..07 và 2024H2. NHƯNG false-positive dày: 2023-05..08 (62–70%), 2025-02/07/08 (75–86%) — các cửa sổ mà w60/dsb60 flat-to-âm (2025: w60 −1.0, dsb60 −2.2). Kỳ vọng dương CHỈ khi melt-up kiểu 2020 lặp lại — 6 năm chưa lặp. Theo chuẩn công tố (regime test ≥2022 bắt buộc): **REJECT điều kiện hóa bằng breadth-thrust/value-shock.**
- "Intraday breadth": **không khả thi với dữ liệu hiện có** — market.duckdb chỉ có intraday cho 2 symbol VN30F1M/F2M (1m/5m từ 2018-08). Trục intraday duy nhất còn mở về mặt DATA là tín hiệu từ phái sinh (basis/gap/range VN30F1M) — chưa từng thử, chỉ được phép phân tích offline nếu quay lại hướng này.
- "Số mã sàn ngày t" (pct_floor daily đã tính được từ duckdb): vẫn là tín hiệu biết-cuối-ngày-t hành-động-t+1 — cùng cấu trúc trễ 1 phiên đã giết crash-brake; không mở lại.

### Oracle ceiling (mỗi năm entry chọn model tốt nhất trong {gb_x08, dsb60, sx_w60, champ})

| năm | pick | oracle | Δ vs gb_x08 |
|---|---|---|---|
| 2020 | sx_w60 | 62.26 | +16.82 |
| 2021 | dsb60 | 33.30 | +6.86 |
| 2022 | gb_x08 | 11.34 | 0 |
| 2023 | dsb60 | 19.82 | +1.78 |
| 2024–26 | gb_x08 | — | 0 |

**Oracle = 153.95u vs gb_x08 128.50u → trần regime-switching = +25.46u**, nhưng **23.68u (93%) nằm ở 2020-21** — regime chết, và oracle này còn được biết-trước-tương-lai miễn phí. **Trần phần SỐNG (entries ≥2022) chỉ +1.78u** — dưới cả noise seed-to-seed (~±2u). (Tham chiếu: thêm pyramid vào pool → oracle 192.2u, pick pyramid mọi năm — nhắc lại rằng alpha còn lại nằm ở sizing, không phải switching.)

## VERDICT TỔNG

1. **Chốt tử vĩnh viễn: bỏ/nông pullback.** Đường chảy máu depth bất biến qua 2 thế hệ exit stack (sai khác <1 điểm/nấc). Đệm 4.5% = alpha trực giao với exit, tập trung bảo hiểm năm xấu (2022 −8.5u nếu bỏ). Không thử lại dù exit head có tiến hóa nữa.
2. **Chốt tử: regime-switching giữa các model hiện có.** Trần biết-trước-tương-lai = +25.5u nhưng phần sống ≥2022 = +1.78u; mọi ghost đều là máy bơm 2020-21; discriminator nhanh nhất tìm được (breadth-thrust+value-shock) vẫn false-positive ở 2023/2025. Không đầu tư thêm.
3. **Ngách 2023 của dsb60 (+1.78u)**: ghi nhận tồn tại, không hành động — quá mỏng, không có tín hiệu đi trước.
4. **Còn mở:** (a) **sizing/pyramid** — +63.7u đều mọi năm, trực giao regime, chỉ chờ multi-lot serving (đã deferred, giờ có giá niêm yết của sự chờ); (b) tín hiệu regime từ **phái sinh VN30F1M intraday** (data 1m từ 2018-08) — trục duy nhất chưa thử còn dữ liệu, chỉ offline; (c) lever giveback-at-defer đã có trong gb_x08 — không đụng.

Templates np_atmkt/np_pb02/np_pb03 (2786-88), op_pb02/op_pb03 (2789-90) giữ trên leaderboard làm kết quả âm trung thực (đều dưới champion, không nhiễu ranking).
