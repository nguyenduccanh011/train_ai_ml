# PURE-ML CHAMPION — khai quật + autopsy
Ngày: 2026-07-10. Nguồn: Postgres stockml (2.587 templates, phân loại pm01_classify.py), trades run_trades, counterfactual pm06_gap.py. So chuẩn: champion 2646 (729.6 s42) và gb_x08 t2783 (735.0 s42).

## 1. Phân loại mức-độ-rule (pm01_classified.csv)

Classifier (key thật trong engine_config, đối chiếu engine.py + experiment.py):
- FORCE-SELL rule: `exit_force_gate`, `exit_force_gate_nonbull`, `exit_force_gate_lowbreadth`, `exit_force_gate_vn30`, overext-sell (`overext_ma_window>0` + 'overext' trong exit_priority), macd/rsi shield, structural/downtrend/csr stop, top_reversal/top_exit/exit_rs_drop.
- SUPPRESS: `exit_force_suppress`, `exit_market_gate_enabled` (market_drop suppress).
- TRAILING/STOP đơn giản: trailing_stop/atr/struct-donch/tier2/pop_lock/…, hard_stop trong exit_priority.
- ENTRY rule-gate: `entry_gate`(upleg), `entry_market_gate`, `entry_breadth_gate`, `entry_skip_nonbull`, bchannel, z_low.
- Pullback-limit fill (`entry_pullback_*`) = cơ chế khớp lệnh, KHÔNG tính là rule.

| class | định nghĩa | n | có run | best composite |
|---|---|---|---|---|
| a PURE-ML | exit_priority=["signal"(+max_hold)], không force/suppress/trailing/stop, không entry rule-gate | 798 | 564 | **619.1** |
| b ML-DOMINANT | + trailing/hard-stop đơn giản hoặc modifier nhẹ, không force-gate, không suppress | 80 | 66 | **625.8** |
| c RULE-HYBRID | có force-sell rule / regime-suppress (họ champion) | 1.709 | 1.481 | 735.0 |

Sanity: gb_x08/cv_parity/x2_parity đều class c với đúng flags (downleg12+nonbull+lowbreadth+overext+market_gate).

### Top-3 nhóm (a) — tất cả cùng 1 gia đình "no-DL" (campaign 2026-06-08/09, thời kỳ tiền-velocity, entry_lvup126_lean era ~gen 0-3 lineage)
| # | tmpl | name | comp (seed, năm) | mô tả |
|---|---|---|---|---|
| 1 | 1058 | n2_sx_rr_h10_thr2p5 | 619.1 (s42, 2026-06) | Entry: continuation_entry_regression h6/pen1.0/tw70, feat entry_lvup126_lean, zE −1.2, fill pullback-limit 3%/25 phiên (close_next). Exit: reward_risk_regression h10, feat exit_vol_market, zX ≥ 2.5. exit_priority=["signal"], max_hold 10000, không stop/trailing/gate. LGBM cả 2 đầu. |
| 2 | 1029 | n2_noDL_volm_thr2p0 | 579.3 (s42) | Cùng base, zX thr 2.0 ("nothing masks the ML"). |
| 3 | 903 | n2_lx_rr_h10_exit_vol_market_nodl | 574.8 (s42) | Cùng head, KHÔNG pullback fill (mua close_next thẳng). |

### Top-3 nhóm (b)
| # | tmpl | name | comp | mô tả |
|---|---|---|---|---|
| 1 | 1053 | n2_rc_hstop15 | 625.8 (s42) | = t1029 + hard_stop −15% (exit_priority [hard_stop, signal]). Đây chính là "entry_lvup126_lean best 625.8" trong ARCHAEOLOGY. |
| 2 | 1052 | n2_rc_hstop20 | 605.7 | hard_stop −20%. |
| 3 | 1054 | n2_rc_hstop25 | 595.7 | hard_stop −25%. |

Nhận xét cấu trúc: trần PURE-ML của leaderboard nằm trọn trong 1 gia đình duy nhất (continuation-h6 entry + reward_risk-h10 exit); hard_stop −15% cộng thêm ~+7..+47 tùy thr. Chưa ai thử pure-ML trên base HIỆN ĐẠI (4-head ensemble + velocity exit) — mọi template ≥700 đều class c.

## 2. Autopsy ứng viên #1 = t1058 (clone pm_sxrr25 = t2808, chạy 2026-07-10)

### Per-seed (composite)
| seed | pm_sxrr25 | champ 2646 | gb_x08 | Δ champ | Δ gb |
|---|---|---|---|---|---|
| 42 | 644.8 | 729.6 | 735.0 | −84.8 | −90.2 |
| 7 | 630.1 | 731.5 | 736.7 | −101.4 | −106.6 |
| 99 | 629.7 | 722.5 | 728.1 | −92.8 | −98.4 |
| 555 | 625.1 | 730.4 | 735.7 | −105.3 | −110.6 |
| 123 | 622.7 | 728.3 | 733.3 | −105.6 | −110.6 |
| **mean** | **630.5** | 728.5 | 733.8 | **−98.0** | **−103.3** |

Ổn định seed (622–645, spread 22), pnl 175–183, mdd 0.49–0.51, ~580 trades, hold TB ~149 phiên. (Clone s42 644.8 > run gốc 619.1 do data snapshot mới hơn.)

### Per-year (trades s42 gốc, n=591, pnl tổng 175.7 vs gb_x08 128.5)
| entry-year | n | pnl | WR | PF | worst |
|---|---|---|---|---|---|
| 2020 | 61 | **+142.0** | .967 | 428 | −.19 |
| 2021 | 32 | +7.4 | .781 | 12.7 | −.31 |
| 2022 | 296 | +6.2 | **.375** | 1.22 | **−.61** |
| 2023 | 5 | −0.4 | .40 | 0.43 | −.37 |
| 2024 | 20 | +0.5 | .55 | 1.37 | −.30 |
| 2025 | 117 | +20.3 | .513 | 3.79 | −.41 |
| 2026 | 60 | −0.3 | .367 | 0.90 | −.28 |

Regime test ≥2022: KHÔNG ĐẠT chuẩn hiện hành — 2022/2023/2024/2026 ≈ 0 hoặc âm, chỉ 2025 dương thực. 81% pnl từ cohort COVID-2020 hold ~440 phiên.

### So cấu trúc với gb_x08 (join symbol+entry_date, s42)
- Overlap entry: 16.1% exact (95/591), 28.3% fuzzy ±5d — hai model gần như KHÁC HẲN tập entry (single-head continuation vs 4-head ensemble TB).
- Trên 95 lệnh chung: t1058 +38.8 vs gb +14.7 (**Δ +24.1**, chủ yếu 2020 +22.7); hold dài hơn +198d.
- t1058-only (496): +137.0, PF 4.85. gb-only (1.283): +113.8, PF 5.64, trải đều mọi năm (2023 +18.1 mà t1058 chỉ có 5 lệnh).
- Hold: median 47d/p90 480d vs gb 14d/86d. Tail: p05 −29.3% vs −8.9%.
- Exit reason: 534/591 'signal' (ML thật), 57 'open' — đúng nghĩa exit thuần ML.

### Bản đồ điểm chết
1. **Không tail-cut**: 2022 vào 296 lệnh giữa bear, WR .375, chuỗi −30..−61% (DIG −61%/680d, AAV −55%/714d, HPG −44%…). p05 −29%.
2. **Nghẽn sổ (capacity starvation)**: trung bình 48–61/61 mã LUÔN có vị thế mở → 2023 chỉ 5 entry, 2024 20 entry; mất trọn alpha 2023-24; đồng thời chỉ 591 trades → confidence multiplier ~0.74 vs gb ~0.93 (mất ~90đ quality theo cơ chế shrinkage).
3. **Tập trung 1 sóng**: yearly_consistency 1.82 (phạt −51.6đ×1000-scale) vs gb 0.81; mdd_per_symbol 0.511 (phạt −150đ) vs 0.175 (−43đ).

### Ngách nó THẮNG gb_x08 (nguyên liệu tuyến mới)
- **Runner-riding**: 35 lệnh hold >200d thoát 2021-11..2022-03 = +109.6u = 62% tổng pnl. ML zX (reward_risk h10, thr 2.5) TỰ BẮT ĐỈNH sóng 2022-01 (VND +932%, BCG +814%, NKG +794%, HSG +654%… đều exit 12/2021–02/2022) — không cần force gate nào.
- Trên lệnh chung, gb clip runner ở 35–71d (+15–37%) trong khi t1058 ăn +200–800%: BCG 8.14 vs 0.37, DPM 3.65 vs 0.20, MBB 2.25 vs 0.15.
- Runner ngoài-2020 vẫn có: LPB 2022-11 +469% (gb +53%), VTP 2022-12 +331% (gb MISS), BSR 2025 +188% (gb +56%) → khả năng "ôm trọn sóng" không phải chỉ là may mắn COVID.

## 3. Chẩn đoán gap (~103đ multi-seed; phân rã composite s42, pm06_gap.py)

Decompose quality (×1000): t1058 total-pnl 762 > gb 558, nhưng thua ở mdd (−150 vs −43), yr (−52 vs −16), và confidence haircut (591 trades).

| nguồn | counterfactual | Δcomposite | kết luận |
|---|---|---|---|
| (i) thiếu force-exit tail-cut | clip mọi lệnh tại −12% | **+140.4** → 759.4 (vượt cả gb 735) | NGUỒN CHÍNH — over-explains toàn bộ gap. Lưu ý: clip lý tưởng hoá (không slippage gap-down, và force-gate thật cũng chém winner → thực tế < +140). |
| (ii) thiếu trailing | gb chỉ khớp trailing 3/1378 lệnh | ~0–10đ | Không đáng kể first-order. |
| (iii) entry regime (bear-2022) | bỏ entry 2022-03..11 | +63.1 riêng lẻ; khi đã có clip12 thì cộng gộp CHỈ 740.7 (< 759.4 clip-alone, vì mất trade count) | Bị (i) nuốt gần hết — vấn đề không phải mua trong bear, mà là KHÔNG CẮT khi sai. |
| (iv) entry/label/feature (ensemble 4-head, velocity exit, conv fill) | phần dư sau (i) | ≈ 0 về composite first-order, NHƯNG là nguồn của 1.378 vs 591 trades: exit velocity h20 quay vòng vốn (hold 14d), mở khoá 2023-25 (+38u gb-only ở 2023+2025) và confidence | Quan trọng cho throughput/consistency, không phải cho quality thô. |

**Kết luận hạt giống**: tín hiệu ML hai đầu của t1058 tự đứng được (PF 5.2, Sortino 2.35, tự bắt đỉnh 2022-01); thứ nó thiếu KHÔNG phải entry tốt hơn mà là (1) cơ chế cắt đuôi lỗ — chỉ cần hard-stop/ML-risk-head thay vì force-gate giá là composite đã ~715–760 (t1053 hstop15 = 625.8 mới chỉ là thr2.0 + stop thô; chưa ai chạy **thr2.5 + hstop12–15**, ô trống rõ nhất), và (2) cơ chế nhả sổ (exit nhanh trade chết để quay vòng vốn). Tuyến mới đề xuất: base t1058 + hard_stop −12% + (tuỳ chọn) ML risk-exit head thay stop cứng, giữ nguyên khả năng runner-riding.

## Files
- pm01_classified.csv (toàn bộ 2.587 template + class + flags), pm_t1058_s42_trades.csv
- Scripts: pm00_schema_peek.py, pm01_classify.py, pm02_detail.py, pm03_check_runs.py, pm04_run.py (+log), pm05_autopsy.py, pm06_gap.py
- Clone: template 2808 `pm_sxrr25`, runs seeds 42/7/99/555/123 (run_id template/pm_sxrr25-2af419c6)
