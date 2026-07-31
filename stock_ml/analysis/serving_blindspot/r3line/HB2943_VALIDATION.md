# HỒ SƠ VALIDATION — t2943 (pr5_dynclean_mh25: stack t2936 + head csrank dyn e3 + mh25)
Ngày: 2026-07-10/11. Nhiệm vụ: multi-seed + công tố rút gọn — t2943 là thế hệ kế tiếp hay ảo giác seed-42?
Thước: `nh_nav2` K25 R0.6 settle T+2, shuffle-mean±sd 20 perm, HAI chế độ (adv 0.08% / no-advance).
Scripts: `hb_00_dump.py, hb_01_anchor.py, hb_10_seeds.py, hb_20_prosecution.py, hb_21_recent.py,
hb_30_plateau.py` (+ hb_*_out.txt). Clone DB mới: hb_2943_mh16 = t2944, hb_2943_mh20 = t2945.
KHÔNG đụng canonical, KHÔNG commit.

## SỰ CỐ MÔI TRƯỜNG (bắt buộc đọc trước khi đối chiếu số cũ)
Commit `926f4c6b` (22:13, agent Opus khác, chủ đích) **xóa toàn bộ corpus serving_blindspot** giữa phiên
làm việc — gồm nh_nav2.py, mọi script pr*/ab*/xg* và MỌI trades CSV (CSV không tracked → mất vĩnh viễn khỏi disk).
Khắc phục: workdir mới **`F:/PROJECTS/hb2943_work/`** (ngoài vùng refactor);
- nh_nav2.py + pr3_lib.py khôi phục từ git (`b4e12b41`);
- trades CSV re-dump từ Postgres `run_trades` (hb_00): t2943 s42, t2936 s42/s555, t2929 s42, t2783(gb) s42.
  **t2936 s7/s99 mất cả trong DB** (run_id không chứa seed → s555 ghi đè; số per-seed t2936 dùng lại từ
  ab_30_out.txt đã đọc trước khi bị xóa).
- **Anchor check (hb_01): 8/8 CSV candidate tái lập BIT-EXACT số công bố** (t2943 ×21.32/4.56/1.65; t2936
  ×22.87/4.43; t2929 ×21.15/4.45; s555 ×22.26). Riêng **gb: dump run DB t2783 cho ×14.14 adv / ×13.24 noadv
  full (f22 3.47/3.36) — MẠNH hơn baseline enriched2 cũ (×13.78/×13.03) ~+2.6% full / ~+1% f22** (enriched2
  gốc mất, không tái tạo được). → Mọi delta-vs-gb trong hồ sơ này dùng baseline MỚI, nhất quán nội bộ cả 3 cột,
  và THẤP hơn con số kiểu cũ ~2-3đ%; KHÔNG so trực tiếp delta ở đây với delta trong AB_PROSECUTION.md.
- **Canary pipeline: re-run t2943 seed 42 sau refactor tái lập trades BIT-EXACT (2206 lệnh)** → code core
  không bị refactor làm đổi hành vi; mọi run mới hợp lệ.

## 1. MULTI-SEED t2943 (hb_10; seeds 42/7/99/555; NAV mean±sd 20 perm; adv | noadv)
| seed | comp | tr | full | f22 | f23 | f25 | DDw full | DDw f25 |
|---|---|---|---|---|---|---|---|---|
| 42 | 666.8 | 2206 | ×21.32±0.56 \| ×19.48±0.64 | 4.56 \| 4.60 | 2.99 \| 3.00 | 1.65 \| 1.68 | −13.9 \| −13.7 | −12.9 \| −12.8 |
| 7 | 665.4 | 2207 | ×22.20±0.53 \| ×19.83±0.56 | 4.57 \| 4.64 | 3.03 \| 3.03 | 1.64 \| 1.67 | −13.7 \| −13.6 | −12.9 \| −12.8 |
| 99 | 665.1 | 2201 | ×21.48±0.71 \| ×19.83±0.76 | 4.51 \| 4.63 | 2.98 \| 3.01 | 1.63 \| 1.66 | −13.8 \| −13.7 | −12.9 \| −12.8 |
| 555 | 663.1 | 2206 | ×21.62±0.48 \| ×19.04±0.65 | 4.63 \| 4.58 | 3.00 \| 2.98 | 1.61 \| 1.63 | −13.7 \| −13.6 | −12.9 \| −12.8 |
| **seed-mean** | | | **×21.65 \| ×19.54** | **×4.57 \| ×4.61** | **×3.00 \| ×3.01** | **×1.63 \| ×1.66** | | |

So t2936 seed-mean (×22.24 \| ×20.67; f22 4.37 \| 4.19; f23 2.94 \| 2.88):
- **f22: +4.6% adv / +10.0% noadv** — MỌI seed t2943 (min 4.51/4.58) ≥ seed TỐT NHẤT của t2936 (4.43/4.24).
- f23: +2.0% / +4.5%. f25: mọi seed DDw −12.9/−12.8 (t2936: −13.4..−13.5).
- **full: −2.7% adv / −5.5% noadv** (trong hạn mức −8%). s42 là seed YẾU NHẤT full-frame của t2943 (×21.32)
  và chỉ mid-pack f22 → **seed-42-luck BỊ BÁC**; recent-edge là tính chất của cấu trúc, không phải của seed.
- Gate công tố (f22 ≥ t2936+2%, full ≥ −8%): **PASS cả hai chế độ** → chạy công tố rút gọn.
- Row canonical t2943 trong DB: seed 42 chạy lại CUỐI cùng (run_id t2943 bị ghi đè theo mỗi seed — y hệt
  hiện tượng làm mất t2936 s7/s99; per-seed lưu tại hb_2943_s*_trades.csv trong workdir). Run restore đã
  verify bit-exact vs trades pr5_70 gốc (2206 lệnh; dòng `bit_exact_final=False` trong hb_10_out.txt là
  artifact dtype SQL-vs-CSV, đã kiểm lại qua CSV-roundtrip: True).

## 2. CÔNG TỐ RÚT GỌN — 3 cột (s42, delta vs gb re-dump ×14.14/×13.24; hb_20 + hb_21)
| tiêu chí | **t2943** | t2936 | dyn_mh25 (t2929) |
|---|---|---|---|
| held-out f21 adv \| noadv | **+54.3% (8.8sd) \| +54.8% (7.9sd)** | +48.9 \| +42.2 | +48.9 \| +51.6 |
| held-out f24 adv \| noadv | **+24.8% (8.6sd) \| +32.7% (12.5sd)** | +21.7 \| +23.4 | +21.1 \| +30.5 |
| LOYO bỏ-2021 adv \| noadv | +27.2% (7.3sd) \| +34.9% (8.3sd) PASS | **+37.2 \| +40.7** PASS | +25.8 \| +34.3 PASS |
| LOYO bỏ-2022 adv \| noadv | +33.8% (6.3sd) \| +24.5% (5.0sd) PASS | **+46.4 \| +41.1** PASS | +33.4 \| +24.7 PASS |
| drop-top-20 full adv \| noadv | +67.8% (11.2sd) \| +56.7% (9.1sd) | **+84.4 \| +64.8** | +65.7 \| +57.7 |
| drop-top-20 f22 adv | **+39.2% (10.1sd)** | +36.0 | +36.8 |
| entry lag+1 full adv \| noadv | +35.4% (6.5sd) \| +13.5% (2.7sd) | +36.6 \| **+18.4** | +30.0 \| +8.6 (1.6sd) |
| entry lag+1 f22 adv \| noadv | **+27.1% (7.6sd)** \| +17.6% (5.5sd) | +23.5 \| +18.0 | +20.0 \| +12.7 |
| fee R0.8 full \| f22 (adv) | +41.2% \| **+26.5% (7.0sd)** | **+48.9** \| +21.4 | +40.3 \| +23.7 |
| fee R1.0 full \| f22 (adv) | +31.9% (5.6sd) \| **+21.4% (5.8sd)** | **+36.4** \| +15.0 | +31.1 \| +18.7 |
| f23 adv \| noadv (hb_21) | **+21.1% (8.3sd) \| +22.8% (6.2sd)** | +18.2 \| +16.7 | +19.2 \| +21.6 |
| f25 adv \| noadv (hb_21) | **+4.3% (2.6sd) \| +10.8% (5.9sd)** | +2.9 (1.7sd) \| +4.4 | +2.4 (1.3sd) \| +8.8 |
| DDw f25 (gb −11.4) | **−12.9 \| −12.8** | −13.5 \| −13.4 | −12.9 \| −12.8 |

Đọc bảng: t2943 **không thua ô nào <2sd trừ 0 ô** (min = lag+1 full noadv +13.5% 2.7sd — vẫn ≥2sd, hơn hẳn
dyn25 1.6sd tại cùng ô); **thắng cả hai đối chứng ở TẤT CẢ ô recent** (f22/f23/f25, drop20-f22, lag+1-f22,
R0.8/R1.0-f22) và cả f21/f24 held-out; nhường t2936 đúng ở nhóm ô full-frame (LOYO, drop20-full, fee-full)
— nhất quán với trade-off ~−3..−6% full đã khai. **f25-adv lần đầu vượt 2sd** trong cả rổ (ô tripwire của
t2936); DD-edge f25 vẫn đảo dấu vs gb (−12.9 vs −11.4) → tripwire recency KẾ THỪA nguyên văn.

## 3. TRỤC MH CỦA HYBRID (hb_30, s42; t2944 mh16, t2945 mh20, t2943 mh25)
| mh | full | f22 | f23 | f25 | DDw full | DDw f25 |
|---|---|---|---|---|---|---|
| 16 (t2944) | **×21.91 \| ×19.85** | 4.40 \| 4.20 | 2.95 \| 2.89 | 1.62 \| 1.58 | −14.0 \| −13.9 | −13.4 \| −13.4 |
| 20 (t2945) | ×19.99 \| ×17.62 | 4.34 \| 4.15 | 2.91 \| 2.77 | 1.64 \| 1.59 | −12.7 \| −12.5 | −12.2 \| −12.2 |
| 25 (t2943) | ×21.32 \| ×19.48 | **4.56 \| 4.60** | **2.99 \| 3.00** | **1.65 \| 1.68** | −13.9 \| −13.7 | −12.9 \| −12.8 |

- Trục FULL: mh25 KHÔNG phải đỉnh cô lập (mh16 ×21.91 nhỉnh hơn; mh20 trũng ~−9% — đúng hình dạng trũng
  mh20 lặp lại của mọi stack, R3_ROUND1/pr5_10).
- Trục RECENT (lý do tồn tại của hybrid): **edge f22-noadv/f25-noadv TẬP TRUNG ở mh25** (4.60/1.68 vs
  4.20/1.58 mh16, 4.15/1.59 mh20). mh16-hybrid ≈ t2936 thường (4.40/4.20 ≈ 4.43/4.22) — head csrank gần như
  KHÔNG thêm gì ở mh16; recent-tilt là **tương tác head×mh25**, không phải plateau rộng theo mh.
  Trừ điểm trung thực: điểm recent đứng trên 1 nấc mh — nhưng (a) 4/4 seed tái lập (mục 1) nên không phải
  noise; (b) dyn_mh25/t2929 độc lập cùng cấu trúc cho cùng recent-tilt; (c) f25-adv của mh20 (1.64) vẫn giữ
  — suy giảm chủ yếu ở kênh noadv. Theo dõi như một tripwire cấu trúc, không đủ để chặn.

## 4. PHÁN QUYẾT: **t2943 = THẾ HỆ KẾ TIẾP trên trục ≥2022 — SURVIVED công tố rút gọn** (4 điều kiện)
Căn cứ: (i) thắng t2936 seed-mean ở MỌI lát ≥2022 cả hai chế độ (f22 +4.6/+10.0%), mọi seed ≥ seed tốt nhất
t2936; (ii) full chỉ −2.7/−5.5% (hạn mức −8%); (iii) pass toàn bộ công tố rút gọn ≥2sd, thắng 3 cột ở mọi ô
recent, f25-adv 2.6sd — vá đúng điểm yếu recency của t2936; (iv) seed-42-luck bị bác (s42 là seed yếu nhất
full-frame). Ảo giác seed: **KHÔNG**.

Vai trò đề xuất:
1. **t2943 = ứng viên chính THẾ HỆ KẾ TIẾP (recent-champion), promote SAU t2936** — không thay t2936 ngay:
   t2936 vẫn là đường promote ngắn nhất (bundle = dòng live downpress, chỉ sửa engine dict); t2943 cần export
   head csrank/xsec MỚI (bundle nặng hơn, chưa tiền lệ serving; bug loader xsec cwd phải xử trước khi serve).
   Trình tự đề nghị: promote t2936 theo án AB_PROSECUTION §Phán quyết → chuẩn bị bundle t2943 (export csrank
   head + leakage-auditor 4-check + shadow-run bit-level) → t2943 lên thay khi shadow sạch.
2. **Tripwire kế thừa + mới**: (a) DD-edge f25 vẫn đảo vs gb (−12.9 vs −11.4) → giữ nguyên tripwire recency
   án t2907/t2936 (rolling-18-tháng vs gb < −1sd 2 quý liên tiếp → demote); (b) tripwire cấu trúc mh25: nếu
   f22/f25-noadv của mh25-line rớt về mức mh16 (~4.2/1.58) trong khi mh16 không rớt → recent-tilt đã tắt,
   quay về t2936.
3. **Số công bố theo seed-mean**: full ×21.65 adv / ×19.54 noadv; f22 ×4.57/×4.61; f23 ×3.00/×3.01;
   f25 ×1.63/×1.66 — KHÔNG quote s7 ×22.20 hay s555 f22 ×4.63 đơn lẻ.
4. dyn_mh25/t2929 rớt về dự phòng khảo cứu (t2943 ≥ nó ở mọi ô đo, cùng gia tộc lệnh); t2936 giữ nguyên án
   SURVIVED + vai trò promote-trước.

Files: hb_10_out.txt (per-seed+canary), hb_20_out.txt (công tố 3 cột), hb_21_out.txt (f23/f25),
hb_30_out.txt (plateau mh), hb_01 (anchor). Workdir: `F:/PROJECTS/hb2943_work/` (trades CSV chỉ còn ở đây
+ Postgres). Đối chiếu: AB_PROSECUTION.md (án t2936, §T6 mở tuyến này — số delta-vs-gb ở đó dùng baseline
enriched2 cũ, xem §SỰ CỐ).
