# DUAL-LAYER ENTRY — ĐỊNH GIÁ TRONG KHUNG NAV (2026-07-09)

Câu hỏi: cohort runaway có kỳ vọng dương tự thân nhưng âm trong 1-slot vì occupancy
(RUNAWAY_AUTOPSY.md). Nếu cho nó **pool vốn riêng** (dual-layer: sleeve A = core limit,
sleeve B = at-market trên tín hiệu runaway) thì portfolio được gì — trong 2 ràng buộc vốn
đã chốt: 100% NAV cố định, hoặc +50% NAV margin lãi 15%/năm?

Scripts: `dl_00_sleeveB.py` (tái tạo sleeve B trail10 có ngày entry/exit, verify khớp autopsy
1.009 lệnh / +205,8u / hold med 51 / WR 0,83) → `dl_01_dual_sim.py` (sim NAV daily, tái dùng
cơ chế `week0/portfolio_sim.py`: leg neo 2 đầu vào pnl CSV, size = NAV_sleeve/K, cost model
slippage 1.0015/0.9985 + fee 0.004 đã nằm trong pnl net). Metrics: `dl_sim_metrics.csv`.

## Thiết kế

- **Sleeve A** = 1.378 trades thật gb_x08 seed-42 (`signalq/nicheloss/gb_x08_s42_trades.csv`),
  K=25, cơ chế hệt chuẩn cũ. BASE tái lập trên trades gb_x08 (chuẩn cũ ×14.06 là trên trades
  wavestruct serving — cùng thang: ×14.44 vs ×14.06).
- **Sleeve B** = cohort runaway at-market: entry close[bar tín hiệu+1]×1.0015, exit **trail10**
  (peak-close −10%, scheme tốt nhất autopsy §2), sequential per-symbol đã dedup → 1.009 lệnh
  (+20,4%/lệnh, WR 0,83, hold med 51 bar). 9 lệnh end_of_data (exit sau 08/07/2026) bị loại → 1.000.
  Pool slot **RIÊNG** K_B=10 (demand concurrency của cohort: mean 39 / med 37 / max 94 → pool
  luôn đầy, khớp first-come).
- **DL_xx**: chia vốn tĩnh tại t0, 2 sổ riêng, không rebalance, không vay chéo — sleeve B lấy vốn
  TỪ sleeve A đúng nghĩa.
- **DL_MG**: sổ A giữ nguyên 100% NAV (y hệt BASE, không bị đụng cash); sổ B equity₀=0, mua bằng
  margin, trần dư nợ 50%×NAV tổng, size = 50%×NAV/K_B, slot cap cứng K_B; lãi 15%/năm cộng dồn
  hàng ngày trên dư nợ thật (trả từ cash_B, thiếu thì vốn hoá vào nợ); tiền thoát lệnh B trả nợ
  trước, phần thừa giữ lại cash_B và dùng trước khi rút nợ mới (B tự tài trợ dần bằng lãi giữ lại).

## Kết quả (NAV₀=1.0, 2020-01-02 → 2026-07-08, 6,51 năm)

| Kịch bản | NAV cuối | CAGR | MaxDD | Lãi vay (NAV₀ / %LN) | Lệnh B khớp/bỏ |
|---|---|---|---|---|---|
| **BASE** (gb_x08 K25) | **14.44** | **50.67%** | **−15.31%** | – | – |
| DL_90_10 | 17.03 | 54.54% | −14.09% | – | 201/799 |
| DL_80_20 | 19.63 | 57.94% | −12.89% | – | 201/799 |
| DL_70_30 | 22.23 | 60.98% | −11.77% | – | 201/799 |
| **DL_MG** (A 100% + B margin 50%, K_B=10) | **34.60** | **72.31%** | **−13.47%** | 0.250 / 0.74% | 212/788 |
| B_ONLY K10 (chẩn đoán) | 40.40 | 76.45% | −10.92% | – | 201/799 |
| _tham chiếu: CHAMPION K25 chuẩn cũ_ | _14.06_ | _50.05%_ | _−13.76%_ | – | – |
| _tham chiếu: PYR_AGGR_MARGIN K25_ | _16.84_ | _54.26%_ | _−18.51%_ | _0.110 / 0.69%_ | – |

Sensitivity K_B (DL_80_20): K_B=5 → ×19.82 (58.18%, −13.22%, 99 lệnh); K_B=15 → ×19.63
(57.94%, −13.15%, 291 lệnh) — kết quả gần như phẳng theo K_B, không phải artifact chọn slot.

Theo năm (DL_MG): 2020 +145,6% · 2021 +196,0% · 2022 +37,3% · 2023 +41,7% · 2024 +31,0% ·
2025 +57,4% · 2026 +18,7% — vượt BASE **mọi năm**,
kể cả 2022 bear (+37,3 vs +31,9). Sleeve A trong mọi kịch bản: 1.021 khớp / 357 bỏ (không đổi —
pool riêng đúng nghĩa, A không mất lệnh nào).

Double-exposure (leg-days sleeve B trùng mã đang mở ở sleeve A): 17% (2.022/12.186 splits;
2.271/12.893 margin) — B thường cưỡi đúng con sóng mà A cũng đang giữ.

## Trả lời 3 câu hỏi

**(a) Chia vốn tĩnh có đáng không khi per-unit B "yếu hơn" A?**
Tiền đề per-unit cần đính chính: theo slot-velocity thì B ex-post KHÔNG yếu hơn — B +20,4%/lệnh
/ 63 bar ≈ 0,32%/bar, A +9,3%/lệnh / 30 bar ≈ 0,31%/bar, và WR B 0,83 (trail10 cắt −10%) cho DD
thấp hơn. Vì pool riêng **xoá đúng kênh occupancy −197u** (kênh giết ý tưởng trong 1-slot), sim
cho CAGR tăng đơn điệu theo tỷ trọng B (50,7→61,0%) và MaxDD còn giảm. **Nhưng đó là câu trả
lời cho cohort ex-post.** Bản THỰC THI ĐƯỢC duy nhất đã đo (trigger cross-X, rw_03/rw_trigger_X05)
chỉ đạt **+4,9%/lệnh gross** (84,2u/1.702 fires, WR 0,62, hold 33 bar ≈ 0,15%/bar = **một nửa
velocity của A**) — với per-unit đó, chia vốn tĩnh là DILUTION thuần: rút vốn khỏi máy 50,7%/năm
để nuôi máy yếu hơn → kém BASE. **Kết luận: KHÔNG đáng với mọi cơ chế ex-ante hiện có** (separator
đã null, rw_02 §3); chỉ đáng nếu tìm được entry rule tách được ≥~2/3 chất lượng cohort ex-post.

**(b) Margin 15%/năm có ăn hết edge sleeve B không?**
KHÔNG — lãi vay không phải nút chặn. Trong DL_MG lãi tổng chỉ 0,250 NAV₀ = **0,74% tổng lợi
nhuận** (peak margin 49,4% NAV, 0 ngày kẹt trần), vì B tự tài trợ bằng lợi nhuận giữ lại sau
~năm đầu, dư nợ thực giảm dần. Ngay cả neo ex-ante +4,9%/lệnh × ~7,5 vòng/năm/slot ≈ +37%/năm
gross trên vốn B vẫn > 15% danh nghĩa. Nút chặn thật là per-unit ex-ante quá mỏng so với whipsaw
(WR 0,62, mua cú pop rồi chỉnh) + 17% double-exposure cùng sóng với A — không phải chi phí vốn.

**(c) Dual-layer hay pyramid xứng đáng suất margin hơn?**
Trên giấy DL_MG (×34,6 / 72,3% / −13,5%) đè bẹp PYR_AGGR_MARGIN (×16,84 / 54,3% / −18,5%), lại
DD nông hơn. Nhưng hai con số KHÔNG cùng đẳng cấp bằng chứng: pyramid là lệnh add **thực thi
được, đo trên trades thật** (+4,2 điểm CAGR, lãi vay 0,69% LN); sleeve B là cohort **ex-post
look-ahead kép** (phải biết trước limit 40 bar không khớp VÀ giá lên). Phiên bản thực thi được
của dual-layer chưa từng dương ở khung per-unit đủ dày (+4,9%/lệnh chưa qua sim NAV, và trong
1-slot toàn bộ họ cơ chế này âm 7/7 năm). **Suất margin hôm nay thuộc về PYRAMID.**

## VERDICT — xếp hạng docket multi-position

1. **PYR_AGGR_MARGIN** — giữ suất margin: số đo thật, +4,2 điểm CAGR, chi phí lãi không đáng kể;
   cái giá là MaxDD −18,5%.
2. **Dual-layer runaway pool riêng** — XẾP TRÊN wave-start, DƯỚI pyramid. Giá trị thật của sim
   này: chứng minh bằng số rằng **multi-position xoá đúng kênh occupancy** — kênh đã giết mọi
   biến thể runaway trong 1-slot — và upper bound (+7 đến +22 điểm CAGR, MaxDD không xấu đi)
   đủ lớn để giữ chỗ trong docket. Điều kiện mở lại (đã có sẵn dữ liệu, không cần run leaderboard):
   sim NAV cho bản trigger cross-X pool riêng từ `rw_trigger_X05.csv` (per-unit +4,9%/lệnh);
   nếu bản đó không cộng được ≥1 điểm CAGR sau lãi vay thì đóng hẳn nhánh này.
   **→ ĐÃ THI HÀNH 2026-07-09 (dl_02): xem §"Bản thực thi được — phán quyết cuối" — ĐÓNG HẲN.**
3. **Wave-start** — cuối docket: đã 4 vòng falsification ở 1-slot, chưa có cận trên định lượng
   ở khung NAV như dual-layer.

## Caveat bắt buộc

- **Sleeve B từ cohort ex-post + exit sim = CẬN TRÊN LẠC QUAN.** Định nghĩa cohort cần biết
  trước (i) limit treo 40 bar không khớp, (ii) close sau 40 bar > close tín hiệu. Trail10 chọn
  hậu nghiệm là scheme tốt nhất. **Mọi con số ở đây chỉ dùng để XẾP HẠNG ý tưởng trong docket
  multi-position — không phải con số promote, không đại diện cho bất kỳ cơ chế chạy được nào.**
- Pool B khớp first-come khi đầy (201/1.000 = 20% cohort; demand concurrency 39 vs 10 slot) —
  quy tắc chọn khác cho kết quả khác.
- 17% leg-days B trùng mã với A: DD thật sẽ tương quan hơn sim (đường MTM neo pnl đã biết làm
  mượt drawdown trong-lệnh); MaxDD các kịch bản DL bị ước lượng non.
- Mô hình margin: sổ B giữ lợi nhuận làm collateral riêng, không sweep về A; khác mô hình tích
  hợp của PYR_AGGR_MARGIN (mọi inflow trả nợ trước) — so sánh hợp lệ ở mức NAV tổng, không so
  từng dòng lãi vay. Không sim force-liquidation (min equity ròng sổ B ≥ 0 suốt kỳ).
- Sleeve A = seed 42 đơn lẻ của gb_x08; 9 lệnh B chưa đóng tại 08/07/2026 bị loại (thiên lệch nhẹ
  xuống). 2026 là năm chưa hoàn chỉnh.

## Bản thực thi được (trigger cross-X) — PHÁN QUYẾT CUỐI (2026-07-09, dl_02)

Thi hành đúng điều kiện mở lại/đóng hẳn ở VERDICT mục 2. Script `dl_02_trigger_sim.py`
(+ `dl_02b_regime.py`, metrics `dlt_trigger_metrics.csv`): tái dùng nguyên khung dl_01
(cùng cost model, sleeve A gb_x08 s42 K=25, K_B=10, DL_MG margin 50% @15%/năm lãi theo
ngày dư nợ thật), thay sleeve B ex-post bằng lệnh trigger cross-X thật từ
`rw_trigger_X05.csv` (và X03 đối chứng). Exit scheme = **champion (sell-head)** — scheme
DUY NHẤT tồn tại trong CSV trigger (trail10 chỉ có cho cohort ex-post, không có bản
trigger). Tái tạo ngày từ DB: entry = bar(signal)+cross_lag+1, exit = entry+hold,
fill = close×1.0015, pnl = tw_pnl (đã net). Vì pool riêng nên limit sleeve A KHÔNG bị
hủy → lấy toàn bộ fires (add + cancel_fill, đã dedup per-symbol): X05 = 1.690 lệnh sau
loại 12 end_of_data, **+4,96%/lệnh, WR 0,46** (WR 0,62 ghi trước đây là số khác — theo
tw_pnl>0 thực tế là 0,46).

| Kịch bản (X05) | NAV cuối | CAGR | Δ vs BASE | MaxDD | Lãi vay (NAV₀ / %LN) | Lệnh B khớp/bỏ | CAGR ≥2022 |
|---|---|---|---|---|---|---|---|
| **BASE** (gb_x08 K25) | 14.44 | 50.67% | – | −15.31% | – | – | **31.01%** |
| DLT_90_10 | 13.36 | 48.89% | **−1.78** | −15.03% | – | 449/1.241 | 29.68% |
| DLT_80_20 | 12.29 | 46.98% | **−3.69** | −14.73% | – | 449/1.241 | 28.19% |
| **DLT_MG** (A 100% + B margin 50%) | 15.84 | 52.83% | **+2.16** | **−19.47%** | **1.618 / 10,9%** | 481/1.209 | **24.15%** |
| _DLT_MG X03 (đối chứng)_ | _17.71_ | _55.46%_ | _+4.79_ | _−18.44%_ | _1.382 / 8,3%_ | _521/1.472_ | _24.12%_ |

Theo năm (DLT_MG X05 vs BASE): 2020 +101,7 vs +92,6 · 2021 **+195,8 vs +121,6** ·
2022 +21,1 vs +31,9 · 2023 +32,8 vs +35,4 · 2024 +11,3 vs +21,2 · 2025 +45,1 vs +42,7 ·
2026 +2,2 vs +9,5 — **thua BASE 4/5 năm ≥2022**, thắng gần như duy nhất nhờ 2020–21.

Đọc số:
- **Điều kiện chữ nghĩa (≥1 điểm CAGR toàn kỳ sau lãi vay): ĐẠT về mặt số học** (+2,16 điểm).
  Nhưng phân rã regime lật ngược: growth 2020–21 = ×5,97 vs ×4,27 BASE, còn **CAGR ≥2022
  = 24,15% vs 31,01% (−6,9 điểm)**, MaxDD sâu hơn 4,2 điểm (−19,5% vs −15,3%). Toàn bộ
  phần cộng là **ghost 2020/21** — đúng mẫu hình đã khiến stack dsb60 bị REJECT; chuẩn
  regime test ≥2022 là BẮT BUỘC và cơ chế này fail cả 3 mặt (CAGR, MaxDD, từng năm).
- Chia vốn tĩnh (DLT_90_10/80_20): **dilution thuần đúng như dự báo §(a)** — âm 1,8–3,7
  điểm CAGR, đơn điệu theo tỷ trọng B.
- Khác hẳn sleeve B ex-post, bản trigger **không bao giờ tự tài trợ**: WR 0,46 + per-unit
  mỏng → dư nợ treo ~50% NAV gần suốt kỳ (peak 50%), lãi vay 1,618 NAV₀ = 10,9% tổng lợi
  nhuận (ex-post chỉ 0,74%). Nhận định §(b) "lãi vay không phải nút chặn" chỉ đúng cho
  cohort ex-post; ở bản thực thi được, lãi vay ăn đáng kể và whipsaw ăn phần còn lại.
- X03 cho tổng cao hơn (+4,79) nhưng regime ≥2022 y hệt (24,12%) — mọi biến thể X cùng
  một cấu trúc ghost.

**VERDICT CUỐI — ĐÓNG HẲN DUAL-LAYER.** Bản thực thi được duy nhất chỉ "đạt" điều kiện
+1 điểm CAGR bằng đòn bẩy trên hai năm bull 2020–21, trong khi thua BASE −6,9 điểm CAGR
ở regime ≥2022 với MaxDD sâu hơn và 10,9% lợi nhuận nộp cho lãi vay. Không có entry rule
ex-ante nào tách được chất lượng cohort ex-post (separator null, rw_02 §3), nên upper
bound ex-post không có đường thành cơ chế chạy được. Docket multi-position còn:
**pyramid (giữ suất margin) + wave-start**.

## Thiết kế 2-lô all-signal (đề xuất user) — kết quả (2026-07-09, dl_03)

Đề xuất: lô pullback giữ nguyên (sleeve A = gb_x08) + lô market vào NGAY tại bar tín hiệu
cho MỌI tín hiệu — **ex-ante thuần**, không cần nhận biết runaway. Lý luận: giá giảm 4,5%
thì lô market "trở thành" vị thế pullback (chịu thuế basis), runaway thì bắt được sóng.

Scripts: `dl_03_dump_npatmkt.py` (dump 1.522 lệnh run `np_atmkt` — bản at-market toàn tín
hiệu exit stack mới, leaderboard composite 498,6 / PF 3,08 — từ Postgres `run_trades`,
run_id `template/np_atmkt-32a8dfee`; loại 7 lệnh 'open' → 1.515) + `dl_03_twolot_sim.py`
(nguyên khung dl_01: cùng cost model pnl đã net, sleeve A gb_x08 s42 K=25, pool B riêng
K_B=10, TL_MG margin 50% @15%/năm lãi theo ngày dư nợ thật). Metrics:
`dl_twolot_metrics.csv`. Per-unit sleeve B: **+6,97%/lệnh, WR 0,44, hold med 23 bar**;
demand concurrency mean 27 / max 61 → pool K_B=10 luôn đầy, khớp first-come.

| Kịch bản | NAV cuối | CAGR | Δ vs BASE | MaxDD | Lãi vay (NAV₀ / %LN) | Lệnh B khớp/bỏ | CAGR ≥2022 |
|---|---|---|---|---|---|---|---|
| **BASE** (gb_x08 K25) | 14.44 | 50.67% | – | −15.31% | – | – | **30.97%** |
| TL_90_10 | 13.75 | 49.54% | **−1.13** | −15.15% | – | 421/1.094 | 29.85% |
| TL_80_20 | 13.06 | 48.37% | **−2.30** | −14.98% | – | 421/1.094 | 28.67% |
| **TL_MG** (A 100% + B margin 50%, K_B=10) | 18.65 | 56.71% | **+6.04** | −18.83% | 1.917 / 10,9% | 464/1.051 | **28.76%** |
| _B_ONLY K10 (chẩn đoán)_ | _7.57_ | _36.44%_ | _−14.23_ | _−24.84%_ | – | _421/1.094_ | _16.19%_ |
| _tham chiếu: PYR_AGGR_MARGIN K25_ | _16.84_ | _54.26%_ | _+3.59_ | _−18.51%_ | _0.110 / 0,69%_ | – | _**34.84%**_ |

(CAGR ≥2022 của PYR tính từ yearly `portfolio_sim_metrics.csv` cùng công thức dl_02b;
cross-check BASE cùng công thức = 31,01% vs neo-NAV 30,97% — sai khác ≤0,05 điểm.)

Theo năm (TL_MG K10 vs BASE): 2020 +136,8 vs +92,6 · 2021 +151,5 vs +121,6 ·
**2022 +11,8 vs +31,9** · 2023 +33,4 vs +35,4 · 2024 +21,5 vs +21,2 · 2025 +51,0 vs +42,7 ·
2026 +14,6 vs +9,5 — cùng mẫu hình ghost 2020/21, và **chảy máu đúng năm bear**: lô market
mua ngay cú pop cho mọi tín hiệu, trong bear tín hiệu chỉnh −4,5% thì lô market ăn trọn
đoạn giảm mà limit pullback được thiết kế để né.

Sensitivity K_B (TL_MG): 5 → ×22,65 (61,46%, ≥22 **34,66%**, MaxDD **−23,08%**) · 6 → 32,84%/−23,3 ·
7 → 32,35%/−22,9 · 8 → 31,76%/−19,4 · 10 → 28,76%/−18,8 · 12 → 29,30%/−18,6 · 15 → 28,74%/−18,7.
Đơn điệu theo độ tập trung: K_B nhỏ kéo ≥2022 lên nhưng trả bằng MaxDD −23% (sâu hơn cả
pyramid) + fill lottery 240/1.515 = 16% first-come; và **mọi K_B đều thua BASE năm 2022**
(+11,8…+21,0 vs +31,9). Tại mức rủi ro ngang pyramid (MaxDD ≈ −18,5/−19), TL_MG chỉ đạt
28,7–31,8% ≥2022 < PYR 34,84%.

Double-exposure: **55%** leg-days B trùng mã A đang mở (7.456/13.424 ở TL_MG; 6.701/12.213
ở splits) — gấp 3,2 lần mức 17% của sleeve runaway. Đúng chủ ý thiết kế (cưỡi cùng sóng),
nhưng nghĩa là hơn nửa exposure B chồng thẳng lên vị thế A: DD thật tương quan hơn sim,
MaxDD các kịch bản TL bị ước lượng non.

**Phân tích dominance vs pyramid.** Sleeve B về bản chất = bản sao của A với basis đắt hơn
~4,5% (mua pop thay vì chờ pullback): per-unit +6,97%/lệnh WR 0,44 vs A +9,3%/lệnh WR cao
hơn; B_ONLY thua BASE 14 điểm CAGR và **âm −13,8% năm 2022**. Cùng một suất margin 50%:
PYR nộp lãi 0,69% LN và cho ≥2022 = 34,84% / MaxDD −18,5; TL_MG nộp lãi 10,9% LN (dư nợ
treo gần trần suốt kỳ vì B không tự tài trợ nổi, peak 47,9%) và cho ≥2022 = 28,76% /
MaxDD −18,8. **TL_MG thua pyramid ở MỌI điểm trên frontier rủi ro, và thua cả BASE ≥2022
ở spec chuẩn K_B=10 → dominance: có vốn thêm thì nhồi A (pyramid) luôn tốt hơn chạy bản
sao kém của A.** Điểm duy nhất TL_MG "thắng" (tổng ×18,65 > PYR ×16,84) là ghost 2020–21
thuần — đúng tiêu chí đã REJECT dsb60 và trigger cross-X.

Caveat: sleeve B từ trade list **1-slot** np_atmkt = xấp xỉ — pool B thật (K_B slot chạy
song song) sẽ sinh chuỗi lệnh khác hẳn (1-slot bỏ tín hiệu khi đang giữ lệnh); first-come
khi pool đầy (fill 28–31%) là quy tắc chọn tùy định. Con số chỉ dùng để XẾP HẠNG trong
docket, không phải con số promote. 2026 chưa hoàn chỉnh.

**VERDICT 2-LÔ: KHÔNG GIỮ trong docket multi-position.** Chia vốn tĩnh = dilution thuần
(−1,1…−2,3 điểm CAGR). Bản margin chỉ cộng bằng đòn bẩy 2020–21, thua BASE ≥2022 và bị
pyramid dominate với cùng suất vốn ở mọi mức rủi ro. Docket còn: **pyramid + wave-start**.
