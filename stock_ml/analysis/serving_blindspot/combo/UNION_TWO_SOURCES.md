# UNION TWO SOURCES — hợp nhất kênh entry ML (gb_x08) + kênh rule price-gate (fc_rule2)
Ngày: 2026-07-10. Giả thuyết: hợp 2 nguồn entry (overlap trades 39%) vào MỘT sổ 1-slot/mã,
chung stack exit gb → lấp slot trống 2024-26 → cổng NAV K25 vượt cả ×14.44 (gb) lẫn ×14.78 (fc_rule2).
Rủi ro số 1 (án F1/bs_/bch_): occupancy displacement.

## 1. Cơ chế — phát hiện nền tảng: "2 nguồn" KHÔNG khác nhau ở tầng signal

Đọc engine_config t2005/t2835 (fc_rule2) vs t2783 (gb_x08) + code recombine
(`src/pipeline/experiment.py` `_recombine_dual_ml_signals`, nhánh `rule_only_no_ml` dòng ~756):

| tầng | fc_rule2 (t2005) | gb_x08 (t2783) |
|---|---|---|
| buy signal | score HẰNG 1 + `entry_raw_threshold: 0` → buy MỌI bar, lọc bởi `entry_gate: upleg_abovema20` | decoupled zE > **−1.9** (lỏng, ~97% bar) & CÙNG `entry_gate: upleg_abovema20`; ∪ 4 ensemble head (z 0.9/0.7/0.7/0.7) |
| market gate | z −1.1 / w5 / abs_floor −4% | **Y HỆT** |
| fill | pullback-limit 4.5% CỐ ĐỊNH / window 50 | pullback 4.5% conv-scaled (mult ∈ [0.6,1] → depth 2.7–4.5%, DỄ fill hơn) / window 40 |
| exit | overext12/trail8-act15/downleg12/nonbull40-3/dtstop−6% → hold median 7d | full stack champion (snr defer, hold legage/legamp/rs, trail act27 struct…) → hold dài |

⇒ **Buy mask của kênh rule ⊂≈ buy mask gb** (cùng gate; gb chỉ loại ~3% bar zE ≤ −1.9, và fill
của gb dễ hơn ở mọi mức conviction, chỉ thua ở dip rơi vào bar 41–50 của window).
"Nguồn entry thứ 2" thực chất KHÔNG tồn tại ở tầng signal — cái làm fc_rule2 có 2.012 lệnh
vs gb 1.378 và overlap chỉ 39% là **TURNOVER**: exit nhanh (7d) giải phóng slot cùng-mã để
vào lại, trong khi gb đang ôm 1 lệnh dài. Union chung stack exit gb thì kênh rule mất đúng
cơ chế sinh lệnh đó.

Cách union SẠCH nhất (config-only, không cần code/parity knob): clone gb_x08, hạ
`entry_threshold` −1.9 → −99 ⇒ main buy = mọi bar qua upleg gate = đúng buy set kênh rule,
∪ ensemble heads, chung engine. Biến thể điều tiết: + `entry_pullback_window: 50` (trả cho
kênh union cửa sổ fill của rule — nguồn fill-thêm chân thật duy nhất còn lại).

## 2. Dry-run offline (un01_dryrun.py, s42 trades: gb 1378 vs fc_rule2 2012)

Additive thật = lệnh rule không trùng gb (fuzzy ±5d) VÀ slot gb cùng-mã đang RẢNH
(ngoài [entry, exit]+4 phiên cooldown):

| year | n_rule | ov_fuzzy | slot_busy | **additive** | additive_pnl (exit rule) |
|---|---|---|---|---|---|
| 2020 | 331 | 127 | 267 | 64 | −1.17 |
| 2021 | 384 | 131 | 363 | 17 | +0.83 |
| 2022 | 354 | 218 | 310 | 38 | −0.64 |
| 2023 | 299 | 161 | 273 | 21 | +0.10 |
| 2024 | 206 | 115 | 191 | **12** | +0.12 |
| 2025 | 294 | 160 | 269 | **17** | +0.57 |
| 2026 | 144 | 77 | 126 | **13** | +0.51 |
| **Σ** | **2012** | 989 (49%) | 1799 | **182 (9%)** | **+0.32** |

- 85% lệnh rule-không-trùng rơi vào lúc slot gb cùng-mã ĐANG BẬN → không thể thêm vào sổ hợp nhất.
- Phần thêm 2024-26 = **42 lệnh / +1.19u** (pnl theo exit stack rule — proxy).
- **Displacement (first-order)**: 28/182 additive fill chặn gb entry trong khoảng hold —
  gb_pnl bị chặn **3.42u** vs rule_pnl thay thế 1.28u ⇒ **net toàn khung ≈ −3.1u**;
  riêng 2024-26 net ≈ −0.2u (1.19 thêm − 1.40 chặn).
- Đối chiếu "gb đói 2024-26": book-level gb vẫn 19–25 vị thế đồng thời (med 14–20;
  35–44% ngày ≥25) — chỗ trống K25 có, nhưng kênh rule KHÔNG có tín hiệu nào để lấp mà
  gb chưa có (cùng gate!). Slot trống của gb 2024-26 = do PULLBACK KHÔNG KHỚP/tín hiệu
  gate thưa, và rule cũng chịu đúng ràng buộc đó.

⇒ Chuẩn "phần thêm ~0 thì dừng sớm" ĐÃ CHẠM về bản chất. Vẫn đốt 1 vòng seed-42 config-only
rẻ để xác nhận bằng run thật (mọi hiệu ứng bậc cao: domino slot, fill-window, conv).

## 3. Sweep seed-42 `un_` (un02_sweep.py) + occupancy autopsy (un03_autopsy.py)

| variant | knob | comp s42 | Δ gb 735.0 | pnl | trades | NEW (pnl) | LOST gb (pnl) |
|---|---|---|---|---|---|---|---|
| **un_full** | entry_threshold −1.9→−99 (union đủ) | **735.4** | **+0.4** | 128.6 | 1386 | 13 (+0.28) | 5 (0.23) |
| un_w50 | union đủ + pullback_window 50 | 731.0 | −4.0 | 128.1 | 1371 | 65 (+1.35) | **72 (2.49)** |

- **un_full = gb_x08 trade-for-trade**: 1373/1378 lệnh match fuzzy ±5d, matched-cohort pnl
  128.34 vs 128.26. Mở buy từ zE>−1.9 ra TOÀN BỘ bar qua gate chỉ đẻ thêm **13 lệnh/6.5 năm**
  (+0.28u), toàn cụm 2022-09/2023-02 pnl ~0. Xác nhận thực nghiệm: kênh rule KHÔNG có tín hiệu
  nào gb chưa có — union đủ ~ no-op (+0.4 comp = noise).
- **un_w50 = bài học displacement thu nhỏ** (đúng án bs_/bch_): kéo dài cửa sổ fill → fill trễ
  chiếm slot → MẤT 72 lệnh gb premium mang 2.49u (REE +43.8%, DGC +32.3%, NKG +19.4%,
  HPG +14.4%…) đổi lấy 65 lệnh mới +1.35u → comp −4.0. Kênh "thêm" trả giá bằng lệnh premium.
- Biến thể "rule chỉ vào khi slot-idle": TỰ THÂN đã là cơ chế của engine 1-slot/mã —
  un_full chính là biến thể đó (kênh rule chỉ fill khi slot rảnh). Không cần wire riêng.

## 4. Ba cổng — biến thể tốt nhất (un_full)

| cổng | un_full | baseline | verdict |
|---|---|---|---|
| composite s42 | 735.4 | gb 735.0 | ngang (noise); 5-seed KHÔNG chạy — un_full trùng gb 99.6% lệnh, per-seed sẽ bám per-seed gb (735.0/736.7/728.1/735.7/733.3), không thêm thông tin |
| subframe ≥2022 (sv_subframe) | 368.4 / ≥2023: 307.1 / ≥2024: 203.4 | gb 368.5 / 306.3 / 201.9 | ngang |
| **NAV K25 100%** (pr2_04_navsim) | **×14.37 / CAGR 50.55% / MaxDD −15.31%** / uw 120d; avg_open 15.6/25, idle cash 34.7% | gb ×14.44/−15.31%; **fc_rule2 ×14.78/−12.25%** | **FAIL mục tiêu tồn tại** — không vượt gb, càng không chạm ×14.78 |

Cổng NAV còn cho đáp án cấu trúc: sổ K25 của gb trống thật (avg 15.6/25 slot, 34.7% cash idle,
2024-26 med 14-20 vị thế) — NHƯNG kênh rule không lấp được vì nó KHÔNG CÓ tín hiệu riêng để lấp
(§1: cùng gate). Slot trống 2024-26 là do CHÍNH cái gate + pullback chung của cả hai họ.

## 5. Verdict

**GIẢ THUYẾT TỔ HỢP BỊ BÁC — "2 nguồn entry" là ẢO GIÁC phân loại.** Overlap trades 39% giữa
fc_rule2 và gb_x08 không phải bằng chứng của 2 nguồn alpha entry khác nhau: cả hai dùng CÙNG
price-gate entry (upleg_abovema20 + market z −1.1); main-buy của gb (zE>−1.9) đã phủ ~97% buy
mask kênh rule, và fill conv-scaled của gb dễ hơn fill 4.5% cố định của rule. 61% lệnh
"khác" của fc_rule2 sinh ra từ **turnover** (exit 7d giải phóng slot cùng-mã để vào lại) —
một thuộc tính của EXIT STACK, không thể union vào sổ dùng exit stack gb (giữ exit gb thì mất
turnover; giữ exit rule thì thành fc_rule2). Bằng chứng 3 lớp khớp nhau:
1. Dry-run un01: additive thật 182/2012 lệnh (9%), net sau displacement **−3.1u**; 2024-26 net −0.2u.
2. Run thật un_full: +13 lệnh/+0.4 comp (no-op); un_w50: displacement −4.0 comp.
3. NAV K25 un_full ×14.37 ≤ gb ×14.44 < fc_rule2 ×14.78.

Không có ứng viên. Giá trị của fc_rule2 (NAV ngang-hơn, MaxDD thấp, deterministic) nằm ở
**portfolio-frame như một SỔ RIÊNG chạy song song** (diversifier throughput — đúng kết luận
FAMILY_CHAMPIONS §4), không phải nguồn entry ghép vào sổ champion. Nếu còn theo đuổi lấp slot
2024-26, hướng phải là NỚI GATE/PULLBACK CHUNG (đã có án: fc_r2_mgf*, np_atmkt âm) hoặc nguồn
tín hiệu thực sự khác cơ chế — không phải tái tổ hợp 2 họ con của cùng một gate.

## Files
- un00_dump.py (engine_config t2783/t2005/t2835); un01_dryrun.py (+un01_additive_fills.csv);
  un02_sweep.py (+un02_sweep.log, un02_sweep2.log); un03_autopsy.py
- Trades: un_full_s42_trades.csv, un_w50_s42_trades.csv
- Clones: un_full t2846 (run template/un_full-32a8dfee), un_w50 t2847 (template/un_w50-32a8dfee)
- Gates: sv_subframe.py (signalq), pr2_04_navsim.py (pureml) — chạy trực tiếp trên trades CSV
