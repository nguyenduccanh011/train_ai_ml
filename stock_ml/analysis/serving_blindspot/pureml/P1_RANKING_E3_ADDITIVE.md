# P1_RANKING_E3 (additive) — PRE-CHECK offline: rank làm nguồn ứng viên cho champion — **KILL**

Ngày: 2026-07-10. Offline thuần (0 run engine). Script: `p1e3_precheck.py`.
Artefacts: `xr_xr_k20_s42_trades.csv` (rank membership top-20, P1-E2), `xr_xr_smoke_champ2646_s42_trades.csv`
(champion 2646 canonical, 1384 lệnh, tái tạo 729.6). unit-weight = pnl_pct.

## Bối cảnh
P1-E2 đã KILL ranking đứng-một-mình (max 138 « bar 625.8) nhưng defer hướng ADDITIVE
(rank làm NGUỒN ỨNG VIÊN cho timing head champion, không phải gate/standalone) sang "vòng
chiến lược". Đây là pre-check rẻ trước khi viết engine path union: **rank có mang winner
TRỰC GIAO mà champion KHÔNG có không?** Nếu trùng hết → fill-bound/redundant (giống án
continuation-h6 corr 0.849, +0.4) → không đáng viết code.

## Kết quả (764 lệnh rank vs 1384 lệnh champion, s42)
- **Overlap 85.1%**: lệnh rank gần như luôn ở cùng mã champion cũng đang giữ trong cửa sổ
  [entry,exit]. Chỉ **14.9% rank-only (trực giao)**.
- Cohort **rank-only NET ÂM**: ALL −1.40u; **≥2022 −0.48u**.
- Winner rank-only ≥2022: chỉ **+2.72u** gross (26 lệnh), mã top nhỏ vụn: AAV +0.63, DIG +0.40,
  HSG +0.19, NT2 +0.17... — không có runner trực giao nào đáng kể.
- Đối chiếu: cohort trùng-champion đóng góp gần trọn giá trị (ALL +34.2u, ≥2022 +13.7u).

## Verdict
**KILL P1-E3 additive.** Cận trên alpha additive champion đang bỏ lỡ = +2.72u gross ≥2022
(và cohort trực giao NET ÂM). Dưới noise ±2u; capacity 1-slot còn bào tiếp (85% redundant =
champion đã chiếm slot). Đây là lần đo thứ 3 cùng một sự thật: tín hiệu selection top-K
**gần như trùng hoàn toàn** tập mã champion đã giao dịch — phần trực giao 15% không sinh lời.
Không viết engine path union.

Caveat: overlap định nghĩa lỏng (giao cửa sổ thời gian bất kỳ trên cùng mã), nên 85% là cận
trên overlap; nhưng số quyết định (winner rank-only ≥2022 = +2.72u, cohort net âm) không phụ
thuộc định nghĩa lỏng đó — kể cả tính rộng rãi nhất cho additive thì vẫn dưới ngưỡng.

## Hệ quả frontier
Cùng với family champions (pure-ML/no-pullback/rule-only ĐÓNG), P1 standalone KILL, P2 cancel
KILL, exit-timing daily 0/60+: **mọi tuyến từ tín hiệu/họ hiện có trên daily-stock OHLCV nay
đã falsify.** Trục chưa-bác duy nhất còn lại = **thông tin MỚI**: P3 intraday phái sinh (khả
thi 7.3 năm, P3_INTRADAY_PROBE) hoặc P5 foreign-flow (chưa screen). Cả hai là data-eng.
