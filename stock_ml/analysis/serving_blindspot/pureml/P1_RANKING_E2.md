# P1_RANKING_E2 — Tuyến CROSS-SECTIONAL RANKING qua engine/leaderboard: ĐÓNG TUYẾN đứng-một-mình

Ngày: 2026-07-10. Scripts: `xr_01_make_templates.py` → `xr_02_run.py` → `xr_03_report.py`.
Artefacts: `xr_xr_k20_s{42,7,99,555}_trades.csv`, `xr_xr_k10_s{42,7,99,555}_trades.csv`,
`xr_xr_smoke_champ2646_s42_trades.csv`. Templates: xr_k20 (id 2825), xr_k10 (id 2826),
xr_smoke_champ2646 (id 2827). Leaderboard: `template/xr_k20-64dda345`,
`template/xr_k10-64dda345` (run_id ổn định theo template — DB giữ seed chạy cuối, per-seed
đầy đủ trong bảng dưới + CSV).

**Bar đăng ký trước (E1 §thiết-kế-E2):** composite ≥ 625.8 (trần pure-ML) = thành công;
**< 400 sau multi-seed = đóng tuyến đứng-một-mình.**

## 0. Implement (append-only — đúng ràng buộc)

Strategy key mới `xsec_rank_topk` trong `stock_ml/src/pipeline/experiment.py`, 4 điểm chèn,
mọi điểm gate bằng `cfg.strategy == "xsec_rank_topk"` (early-return / no-op cho mọi đường cũ):

1. **`train_fold`** (nhánh dispatch đầu, return sớm): LGBMRanker objective=**lambdarank**,
   group = NGÀY, relevance = quintile per-date của target `forward_return_regression h20`
   (quintile per-date bất biến với demean-theo-ngày ⇒ tái tạo đúng label fwd20_dm của E1);
   params = đúng `p1_01_train.py` (leaves 15, lr .03, 600 trees, label_gain 0..31), seed từ
   template. Walk-forward theo `split_config` chuẩn champion (wf-year, train 2y, gap 85,
   test 2020→2025 + tail 2026H1). Emit score-only (signal=0) + OHLCV carry.
2. **`build_feature_frame`**: append 2 cột breadth 488-univ (`pct_above_ma50`, `adv_pct`)
   vào entry head qua cơ chế `_breadth_append` sẵn có ⇒ đúng bộ **47 cột** của E1
   (`entry_lvup126_recov` 45 + 2 breadth).
3. **`recombine_signals`** (return sớm) → `_xsec_rank_membership_signals`: mỗi **20 bar**
   rank universe theo score; membership hysteresis đúng scheme 'hyst' của p1_03 — vào khi
   lọt top-K, ở lại khi rank ≤ K+10, SELL khi rớt; BUY re-emit cho mọi member mỗi kỳ (để
   tên bị phanh cắt giữa kỳ tự vào lại nếu còn giữ slot). **Force-exit downleg12 GIỮ
   NGUYÊN cơ chế champion** (`_exit_force_mask` — wire được, không phải signal-exit thuần).
4. **`run_experiment`**: 1 dòng `engine_cfg.pop("xsec_rank", None)` (key mới không tồn tại
   ở template cũ nào ⇒ no-op tuyệt đối trên đường cũ). Mọi key mới nằm gọn trong
   `engine.xsec_rank`, parse duy nhất trong nhánh mới.

Engine: `entry_bar_fill_type=close_next`, **không pullback-limit** (điều kiện sống E1 §3),
exit_priority `["signal"]`, hard_stop None, max_hold 10000; universe/cost/scoring GIỮ NGUYÊN
leaderboard (61 mã `vn_stock_default`, roundtrip 0.7%, composite chuẩn `scoring.py`).

**Chứng minh không đụng đường cũ:**
- `tests/regression/test_champions.py`: **1 passed / 12 skipped** (đúng chuẩn).
- Smoke clone nguyên văn champion 2646 (template mới 2827, retrain từ đầu, fold-cache mới):
  seed 42 → **composite 729.6** — khớp canonical từng số (pnl 127.3, pf 5.96, tr 1384).
- Không re-run 2646/2730/2783 canonical; không commit git.

## 1. Kết quả per-seed (composite chuẩn leaderboard; so trần 625.8 / kill 400)

| template | seed | **composite** | total_pnl | pf | mdd/sym | trades | wr | hold |
|---|---|---|---|---|---|---|---|---|
| xr_k20 | 42 | **138.3** | 32.8 | 2.07 | 0.286 | 764 | .488 | 28.3 |
| xr_k20 | 7 | **123.9** | 32.5 | 1.99 | 0.312 | 773 | .475 | 28.0 |
| xr_k20 | 99 | **136.0** | 32.8 | 2.04 | 0.286 | 775 | .484 | 28.5 |
| xr_k20 | 555 | **134.2** | 32.5 | 2.06 | 0.291 | 759 | .490 | 28.4 |
| xr_k10 | 42 | **108.3** | 18.6 | 2.06 | 0.192 | 450 | .513 | 24.7 |
| xr_k10 | 7 | **102.4** | 18.0 | 2.03 | 0.192 | 437 | .506 | 25.1 |
| xr_k10 | 99 | **105.9** | 17.8 | 2.02 | 0.184 | 445 | .501 | 25.4 |
| xr_k10 | 555 | **103.8** | 18.4 | 1.99 | 0.201 | 449 | .486 | 25.1 |

Mốc so sánh cùng cửa sổ/universe/cost: **champion 2646** = 729.6/731.5/722.5/730.4
(s42/7/99/555), **gb_x08** = 735.0/736.7/728.1/735.7, trần pure-ML (pm2_hs10_zx25 + đối
chứng) = **625.8**. xr_k20 mean 4 seed = **133.1**, xr_k10 = **105.1**.

## 2. Per-year pnl (sum pnl_pct theo năm entry — chuẩn regime ≥2022 bắt buộc)

xr_k20:

| seed | 2020 | 2021 | **2022** | **2023** | **2024** | **2025** | **2026H1** | Σ≥2022 |
|---|---|---|---|---|---|---|---|---|
| 42 | +10.60 | +8.91 | +1.79 | +2.80 | +0.66 | +8.70 | −0.69 | +13.26 |
| 7 | +10.72 | +9.48 | +1.42 | +3.08 | +0.89 | +7.59 | −0.67 | +12.31 |
| 99 | +11.83 | +8.76 | +1.19 | +3.50 | +0.66 | +7.66 | −0.77 | +12.24 |
| 555 | +11.29 | +8.92 | +0.90 | +4.18 | +0.80 | +7.23 | −0.84 | +12.27 |

xr_k10 cùng dạng (Σ≥2022 = +7.7…+9.0). Đọc regime:

- **Không có năm gấu âm** (2022 dương mỏng +0.9…+1.8 — spread tương đối sống qua bear đúng
  như E1 robust), 2023 **dương** +2.8…+4.2 (lát yếu nhất của lambdarank E1 hóa ra không âm
  khi qua engine), 2024 mỏng gần 0, 2026H1 **âm nhẹ** −0.7…−0.8.
- Nhưng **phần lớn pnl nằm ở 2020–2021** (60% tổng), đúng cái E1 cảnh báo: biên robust
  ≥2022 chỉ ~+2.4–2.6%/trade net — sống, đều, mà QUÁ MỎNG cho thước tổng-pnl.

## 3. Cấu trúc trades (cơ chế chạy đúng thiết kế)

- Entry chỉ rơi vào **81 ngày** (lưới rebalance+1, close_next), concurrent tối đa 33 ≈
  k_exit 30 (+ chồng lệnh tại ngày swap) — membership là slot-allocator đúng nghĩa.
- Turnover: ~9.4 entries/kỳ trên ~30 slot ≈ **31%/tháng** (E1 hyst tháng: ~0.2–0.3/kỳ ✓).
- Exit tách 2 lớp (xr_k20 s42): **membership-drop 365 lệnh, avg +12.2%** (alpha nằm ở đây)
  vs **downleg12 force 393 lệnh, avg −3.0%** — phanh làm đúng việc phanh (cắt phần gãy,
  nhận lỗ nhỏ), toàn bộ alpha do rank giữ-vị-thế mang lại. Hold trung vị 20 bar, p75 40.
- **Overlap champion** (đo trên vị-thế-đang-mở, s42): 25.1% entry champion nằm trong
  membership xr đang mở (26.1% ≥2022) — khớp E1 (~1/3, không re-skin); chiều ngược 50.4%
  entry xr trùng vị thế champion đang mở.
- avg +4.3%/trade, pf ~2.0 — per-trade LÀNH MẠNH; chết ở đâu: per-bar 0.15%/bar (champion
  0.31%), total_pnl/symbol 0.54 (champion 2.09), mdd/sym 0.29 (champion 0.175), 2026H1 âm
  → composite nghiền cả 4 trục.

## 4. Verdict theo kill đã đăng ký

| Tiêu chí | Kết quả | Trạng thái |
|---|---|---|
| composite ≥ 625.8 (trần pure-ML) | max 138.3 | **FAIL** (cách trần ~4.5×) |
| composite < 400 sau multi-seed | 4/4 seed k20 = 124–138; 4/4 seed k10 = 102–108 | **KILL kích hoạt** |

**VERDICT: ĐÓNG TUYẾN ranking-đứng-một-mình.** Không mơ hồ: 8/8 run < 140, chưa tới 1/4
ngưỡng đóng tuyến, variance seed ±7 điểm — không có đường tối ưu knob nào lấp nổi khoảng
cách 5× (và bar E2 cấm tối ưu knob dưới trần). Đúng kỳ vọng trung thực E1 ghi trước
("gần như chắc chắn KHÔNG chạm trần khi đứng một mình").

**Cái tuyến này CHỨNG MINH được (giá trị giữ lại):** qua engine thật + cost thật + composite
thật, tín hiệu membership top-K tháng vẫn (a) dương mọi năm 2020–2025 kể cả bear 2022,
(b) per-trade +4.3% / pf 2.0 với CHỈ một đầu rank + một phanh giá, (c) overlap champion ~25%
— alpha selection tương đối là thật và khác họ timing. Nó chết vì THÔNG LƯỢNG (238 lệnh/năm
nhưng 0.15%/bar) và đuôi rủi ro (mdd/sym 0.29), không phải vì tín hiệu giả.

**E3 (rank làm selector ứng viên cho timing head champion):** bar viết thiết kế E3 là
xr_k20 ≥ 625.8 — KHÔNG đạt, nên không phác thiết kế ở đây. Hướng đã đăng ký sẵn trong E1
§thiết-kế-E2 (điểm 3): giá trị chuyển sang dạng ADDITIVE — rank làm NGUỒN ỨNG VIÊN cho head
champion, không phải gate; dữ kiện mới từ E2 củng cố: phần alpha nằm ở membership-hold
(+12.2%/lệnh), phần lỗ nằm đúng chỗ champion đã có công cụ xử (downleg/trailing).
Quyết định mở E3 hay không thuộc vòng chiến lược, không thuộc hồ sơ này.

## 5. Caveat

- Số 2020–2021 chịu survivorship của universe 61 mã như MỌI số leaderboard — chỉ đọc
  TƯƠNG ĐỐI vs champion cùng cửa sổ.
- `run_id` leaderboard ổn định theo template ⇒ hàng DB của xr_k20/xr_k10 hiện giữ seed 555
  (seed cuối); per-seed đầy đủ ở bảng trên + trades CSV per-seed trong thư mục này.
- Force-exit và membership-exit cùng exit_reason='signal' trong engine; tách 2 lớp ở §3
  bằng lưới ngày rebalance (off-grid = downleg12 force).
- k10 dùng k_exit = K+10 = 20 (đúng thiết kế E2 "rớt top-(K+10)"), khác 1.5K=15 của E1 —
  không ảnh hưởng verdict (mọi seed cùng cách trần >4×).
