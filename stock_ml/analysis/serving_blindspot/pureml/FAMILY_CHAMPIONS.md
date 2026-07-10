# FAMILY CHAMPIONS — rule-only & no-pullback
Ngày: 2026-07-10. Vòng "vô địch từng họ còn lại" sau khi pure-ML bị REJECT (PM2_PROSECUTION). Chuẩn so: gb_x08 t2783 (5-seed 733.8; NAV K25 ×14.44 / CAGR 50.7% / MaxDD −15.3%). Classifier: fc01_classify.py (mở rộng pm01, đọc component_slots + entry_pullback_pct). Runs: fc02/fc05–fc09. NAV: pr2_04_navsim.py (K25, 100% NAV). Subframe: sv_subframe.py.

## 1. Phân loại họ (2.609 template, fc01_classified.csv)

| họ | định nghĩa | n | có run | best composite |
|---|---|---|---|---|
| ml_signal + pullback (họ champion) | ML head cho tín hiệu, fill pullback-limit | 1.353 | 1.180 | **735.0** (gb_x08) |
| **no-pullback** (ml_signal, at-market) | ML head cho tín hiệu, `entry_pullback_pct` off → fill close_next | 1.163 | 849 | **574.8** (t903) |
| **rule-only** (không ML head nào cho tín hiệu) | strategy/model_mode `rule_only` HOẶC `rule_only_no_ml=true` (ML bypass) | 84 | 67 | **568.3** (t2005) |
| rule-entry + ML-head phụ (không thuộc rule-only) | vd hybrid_rule_entry_ml_downside | 9 | 9 | 371.3 |

LƯU Ý phân loại: nhóm `rule_only_no_ml=true` (5 template n3_*) giữ nguyên ml_component_id trong slot nhưng pipeline BỎ QUA ML fit — pm01/fc01 bucket thô xếp nhầm sang "có ML"; đã kiểm tra tay engine_config + desc ("TRUE rule-only, no ML fit"). Rule-only thật = 79 (model_mode rule_only, slot rule thuần) + 5 (n3 rule_only_no_ml) = 84.

### Top-3 họ RULE-ONLY
| # | tmpl | name | comp | mô tả 1 dòng |
|---|---|---|---|---|
| 1 | 2005 | n3_dtstop06 | 568.3 | Nguyên stack engine champion nhưng tín hiệu = price-gate rule (recombine gates, KHÔNG ML): entry_gate upleg_abovema20 + entry_market_gate z −1.1 + pullback-limit 4.5%/50; exit = downleg12 force + nonbull belowma20p2 + overext 12% + trailing 8%/act15 + market-gate z −1.75; + downtrend_hard_stop −6%. close_next. |
| 2 | 2004 | n3_dtstop08 | 567.4 | = t2005, dtstop −8%. |
| 3 | 1787 | n3_rule_champ | 566.1 | = t2005 không dtstop (gốc "rule champ"). |
| — | 1441/1442 | r17_pb | 428.3 | Nhánh rule ngây thơ: entry OR(52w-high −2% / roc10>6% / mfi>80), exit AND(macd_hist<0, sma20_ratio<0), pullback 3%/25, hold≤20. (16 biến thể r17: stop/trail/ox/act/pb-depth đều ≤ 428.) |

### Top-3 họ NO-PULLBACK (ML signal, fill at-market close_next)
| # | tmpl | name | comp | mô tả 1 dòng |
|---|---|---|---|---|
| 1 | 903 | n2_lx_rr_h10_exit_vol_market_nodl | 574.8 | Họ "no-DL" (anh em t1058): entry continuation_entry_regression h6/pen1.0/tw70 (entry_lvup126_lean, LGBM), zE −1.2; exit reward_risk_regression h10 (exit_vol_market), zX 2.0; exit_priority=["signal"], không stop/gate; fill close_next thẳng (KHÔNG pullback). |
| 2 | 728 | n2_xb2_pre_sz25 | 540.9 | Cùng entry head (h8); exit zigzag_pivot peak pre, SELL z 2.5. |
| 3 | 722 | n2_nd6_ezn12_xh60 | 511.2 | Cùng entry head; exit risk_exit_regression h60. |
| — | 2786 | np_atmkt | 498.6 | Đối chiếu thế hệ mới: gb_x08 hiện đại BỎ pullback → mất −236đ (NICHE_LOSS_MAP verdict 1: "chảy máu depth"). |

## 2. Vô địch từng họ — 5 seed + subframe ≥2022 + NAV K25

### 2a. RULE-ONLY: clone `fc_rule2` = t2835 (t2005)

**5 seed: 568.3 × 5 (deterministic tuyệt đối — không ML fit)**, mean Δ gb_x08 = **−165.5**. pnl 100.3, PF 3.45, mdd_sym 0.213, 2.012 trades, WR .615, hold TB 14.8 phiên.
(Chú thích: clone thăm dò đầu tiên `fc_ruleonly` t2830 = r17_pb 428.3→435.1, cũng deterministic 435.1×5; giữ làm datapoint nhánh rule ngây thơ.)

Per-year (s42, n=2012, pnl 100.3 vs gb 128.5):
| entry-year | n | pnl | WR | PF | worst |
|---|---|---|---|---|---|
| 2020 | 331 | +25.8 | .725 | 6.0 | −.20 |
| 2021 | 384 | +30.9 | .732 | 7.0 | −.15 |
| 2022 | 354 | +7.9 | .511 | 1.8 | −.20 |
| 2023 | 299 | +14.2 | .619 | 3.2 | −.16 |
| 2024 | 206 | +4.6 | .539 | 2.0 | −.16 |
| 2025 | 294 | +15.9 | .605 | 4.0 | −.17 |
| 2026 | 144 | +1.0 | .431 | 1.2 | −.24 |

KHÔNG ghost: dương cả 7 năm entry-year, 2023-24-26 đều dương (gb 2024 chỉ +2.35, rule2 +4.57 THẮNG; 2026 +1.01 vs −0.55 THẮNG; 2021 +30.9 vs +26.4 THẮNG).

- **Subframe ≥2022** (s42): 275.5 vs gb 368.5 (**−93**); ≥2023: 249.0 vs 306.3 (−57); ≥2024: 166.4 vs 201.9 (−36). Thua nhưng cùng bậc, KHÔNG sập kiểu nopb/pm2.
- **NAV K25 100%** (khung y hệt, verify lại baseline cùng ngày): fc_rule2 ×**14.78** / CAGR **51.21%** / MaxDD **−12.25%** (2025-04) / underwater dài nhất 101d — **so gb_x08 ×14.44 / 50.67% / −15.31% / 120d: NGANG (nhỉnh hơn cả final NAV lẫn MaxDD)**. Yearly NAV: rule2 thắng 2021 (+133.7 vs +121.6), 2023 (+40.6 vs +35.4), 2024 (+23.4 vs +21.2), 2025 (+49.5 vs +42.7); gb thắng 2020/2022/2026. Cơ chế: 2.012 lệnh hold ngắn (median 7d) quay vòng vốn nhanh (1.688 fills vs gb 1.021) bù cho pnl/lệnh nhỏ — đúng chỗ mù thứ 3 của composite ("mù vốn" lần này theo chiều NGƯỢC: composite phạt rule2 vì total_pnl thô 100<128 + PF thấp, nhưng NAV thực bằng nhau).
- Exit mix: trailing_stop 457 lệnh +65.9u, overext 401 +58.5u (2 kênh chốt lời rule), signal 1.072 −18.4u (kênh xả rác), downtrend_stop 78 −6.1u. Losers lành: p05 −9.8%, worst −23.5%.
- Cấu trúc vs gb_x08: overlap exact 39.2% (fuzzy 49.2%) — chung gần nửa cohort; trên 788 lệnh chung rule2 +38.7 vs gb +74.9 (**Δ −36.2**, đều do gb ôm winner lâu hơn +13.4d); rule2-only 1.224 lệnh +61.6 (PF 3.5) rải đều 7 năm.
- **Điểm chết duy nhất đáng kể: winner bị chém sớm** — p95 +23.5% / p99 +30.2% / max +54% vs gb +55.5%/+103.9%; toàn bộ gap composite nằm ở đây (winner-uplift counterfactual +415). Tail-loss KHÔNG phải vấn đề (clip12 chỉ +22 lý tưởng).
- Ngách thắng gb: mua sớm hơn ở nhịp PVS/BSR/PVD 2021-01 (gb MISS cả cụm), 2024 + 2026 dương khi gb đói lệnh/âm; turnover cao → composite confidence multiplier không bị haircut (2.012 trades).

Gap-analysis (fc04, s42):
| counterfactual | Δcomp | kết luận |
|---|---|---|
| clip loser −12% | +22.1 → 590.4 | tail đã lành, ít còn thịt |
| bỏ entry bear 2022-03..11 | +22.8 → 591.1 | gate hiện có (z −1.1 + abs_floor −4%) đã ăn phần lớn |
| clip12 + no-bear2022 | +38.5 → 606.8 | TRẦN config-only ≈ 607 — không chạm được 735 |
| winner-uplift (pnl gb cùng mã ±10d) | **+415.4** → 983.7 | gap thật = ML selection + để winner chạy (velocity exit h20 của gb giữ trend mạnh, còn rule chốt cứng overext 12%/trail 8%) |

### 2b. NO-PULLBACK: clone `fc_nopb` = t2831 (t903)

Per-seed composite (run 2026-07-10, data snapshot mới):
| seed | fc_nopb | gb_x08 | Δ |
|---|---|---|---|
| 42 | 593.7 | 735.0 | −141.3 |
| 7 | 594.2 | 736.7 | −142.5 |
| 99 | 587.5 | 728.1 | −140.6 |
| 555 | 579.9 | 735.7 | −155.8 |
| 123 | 597.6 | 733.3 | −135.7 |
| **mean** | **590.6** | 733.8 | **−143.2** |

Ổn định seed (spread 18), pnl 149–153, mdd_sym 0.60–0.63, ~990 trades, hold TB ~88 phiên.

Per-year (s42, n=989, pnl 152.7 vs gb 128.5):
| entry-year | n | pnl | WR | PF | worst |
|---|---|---|---|---|---|
| 2020 | 68 | **+110.6** | .956 | 3053 | −.02 |
| 2021 | 123 | +19.9 | .691 | 7.8 | −.33 |
| 2022 | 430 | +5.2 | **.416** | 1.15 | **−.60** |
| 2023 | 11 | +0.5 | .545 | 1.9 | −.30 |
| 2024 | 54 | −0.4 | .444 | 0.88 | −.32 |
| 2025 | 191 | +18.7 | .476 | 2.7 | −.37 |
| 2026 | 112 | −1.8 | .393 | 0.69 | −.30 |

- **Ghost 2020**: 72% pnl từ 68 lệnh COVID-cohort (fill at-market 2020-04-03..14 bắt trọn đáy V mà pullback-limit của gb MISS — top-15 winner: 12/15 là cụm này, NKG +827%, BCG +726%, HSG +683%…, hold 300–500 phiên).
- **Subframe ≥2022** (s42): comp **−36.5** vs gb **+368.5** (d = −405); ≥2023: 39.4 vs 306.3; ≥2024: 36.7 vs 201.9. FAIL chuẩn regime.
- **NAV K25 100%**: ×**4.40** / CAGR 25.6% / **MaxDD −49.5%** (2022-11-15), underwater dài nhất 824 phiên; sổ nghẹt: 62% số ngày full ≥24/25 slot, 2022 −37.9%. So gb ×14.44 / 50.7% / −15.3% → cùng bệnh NAV với pm2 (×5.61) — thậm chí nặng hơn.
- Cấu trúc trades vs gb_x08: overlap exact 2.0% (fuzzy ±5d 16.9%) — tập entry gần như rời nhau; gb-only 1.358 lệnh +125.8 PF 6.0 trải đều mọi năm; nopb-only +147.9 dồn 2020. Tail: p05 −26.9% vs gb −8.9%; p95 +105.7% vs +55.5% (đúng nghĩa "ôm cả 2 đuôi").
- Điểm chết: (1) không tail-cut (2022: 430 lệnh WR .416, chuỗi −40..−60%, DIG −60%/649d); (2) sổ nghẹt → 2023 chỉ 11 entry (mất trọn alpha 2023); (3) yearly_consistency 1.63 (phạt −44.9đ) + mdd_sym 0.611 (phạt kịch khung −150đ).
- Ngách thắng gb thật sự: fill at-market bắt sóng chạy-thẳng-không-pullback (VTP 2022-12 +416% trong khi gb −0.01; LPB 2022-11 +368% vs gb +53) — nhưng chỉ đếm được ~2 runner ngoài-2020 / 6 năm.

Gap-analysis (fc04, s42, first-order, giả định clip không slippage):
| counterfactual | Δcomp | ghi chú |
|---|---|---|
| clip loser −12% | **+172.0** → 765.7 | over-explains gap — NHƯNG pm2 đã chứng minh: hiện thực hoá tail-cut trong chính họ này (pm2 grid) → composite ~760 mà NAV vẫn ×5.61 REJECT (sổ vẫn nghẹt, alpha vẫn dồn 2020). |
| bỏ entry bear 2022-03..11 | +122.5 | bị clip nuốt phần lớn |
| clip12 + no-bear2022 | +218.7 → 812.4 | trần lý tưởng; không đổi bệnh NAV |
| winner-uplift (thay pnl bằng gb cùng mã ±10d nếu lớn hơn) | +183.0 | thiếu selection + exit velocity của 4-head |

## 3. Bản đồ nâng cấp + sweep đã chạy

### RULE-ONLY (nhánh r17 — trước khi tìm ra n3)
Counterfactual (fc04 trên r17_pb): clip12 +33.0; no-bear2022 +46.5; cả hai +58.6 → trần lý tưởng 493.7. Winner-uplift +470 → gap thật nằm ở ML-selection + winner size (p95 +28% vs gb +56%), KHÔNG phải tail (p05 −9% đã lành).
Sweep lever rẻ duy nhất chưa thử trong 79 template (gate regime rule-based, config-only, cơ chế khác selection-wall ML):
| variant | comp s42 | Δ base 435.1 |
|---|---|---|
| fc_ro_mg (entry_market_gate w5 −3%) | 431.5 | −3.6 |
| fc_ro_chop (ER gate w20 0.30) | 276.8 | −158.3 |
| fc_ro_mgchop | 256.1 | −179.0 |
→ Gate thực tế KHÔNG ăn được counterfactual (+46.5 lý tưởng): rule OR-entry mua rải quanh năm, gate cắt cả lãi (cùng hình selection-wall nhưng ở họ rule). Nhánh r17 đóng.

### RULE-ONLY (nhánh n3 = vô địch thật) — sweep seed-42 (fc07/fc08, deterministic)
Lever theo counterfactual, config-only:
| variant | knob | comp s42 | Δ base 568.3 |
|---|---|---|---|
| fc_r2_hs12 | hard_stop −12% (clip12 proxy, ideal +22) | 550.1 | **−18.2** |
| fc_r2_mgf30 | abs_floor −4%→−3% (bear-gate, ideal +23) | 565.2 | −3.1 |
| fc_r2_mgf25 | abs_floor −2.5% | 563.1 | −5.2 |
→ Tail-cut và siết gate đều ÂM thực tế (stop chém lệnh sắp hồi — cùng hình DEEP_STOP của họ gb, nay xác nhận cả ở họ rule; gate hiện có đã tối ưu). Trần config-only 568.3 đứng vững.

Lever winner-riding (đúng nguồn gap +415, NỚI kênh chốt lời — ngược chiều trục "siết trail" đã chết nên được thử):
| variant | knob | comp s42 | Δ base |
|---|---|---|---|
| **fc_r2_ox16** | overext 12%→16% | **576.3** | **+8.0** |
| fc_r2_trail12 | trailing 8/15→12/20 | 566.2 | −2.1 |
| fc_r2_tskip50 | treo trail khi > SMA50 đang lên | 560.8 | −7.5 |
| fc_r2_ox14 | overext 14% | 567.4 | −0.9 |
| fc_r2_ox18 | overext 18% (đỉnh cong) | **577.9** | **+9.6** |
| fc_r2_ox20 | overext 20% | 575.3 | +7.0 |

NAV-check ox16/ox18 (bài học bắt buộc — composite mù vốn): ox16 ×14.38/−12.29%, ox18 ×14.14/−12.29% — composite +8..+10 nhưng NAV đều ≤ base (×14.78); subframe ≥2022 y hệt (276–278 vs 275.5). → Nới overext chỉ đổi shape composite (pnl/lệnh to hơn, ít lệnh hơn), KHÔNG tạo giá trị vốn thật. Plateau composite họ rule-only chốt ở ~570–578, cách gb 157–165.

### NO-PULLBACK
Mọi lever kề cận đều đã có án tử TRONG CÙNG cơ chế:
1. Tail-cut (hard-stop/zx) — pm2 grid ĐÃ chạy đúng họ này (t1058 cùng entry/exit head): composite lên ~762 nhưng **NAV ×5.61 / MaxDD −50.3% REJECT** (PM2_PROSECUTION). fc_nopb NAV gốc còn tệ hơn (×4.40) → không chạy lại.
2. Thêm pullback-limit — thì thành họ champion (rời họ; và chính là hướng NICHE_LOSS_MAP đã chốt: pullback 4.5% là cấu phần sống còn, bỏ = −8.5u ở 2022).
3. Chuyển base hiện đại (4-head + velocity exit) rồi bỏ pullback = np_atmkt 498.6 (−236 vs gb) — kết quả âm trung thực đã có trên leaderboard.
4. Gate entry regime — bị (1) nuốt (fc04: no-bear2022 +122 nhưng clip12 đã +172; cùng kết luận pm06: "vấn đề không phải mua trong bear mà là không cắt khi sai") và không sửa được sổ nghẹt.
→ Không có lever sống. Họ đóng.

## 4. Verdict

| họ | vô địch | 5-seed | ≥2022 vs gb | NAV vs gb | verdict |
|---|---|---|---|---|---|
| no-pullback | t903 (fc_nopb t2831) | 590.6 (−143.2) | −36.5 vs 368.5 FAIL | ×4.40/−49.5% vs ×14.44/−15.3% FAIL | **ĐÓNG** — bản sao bệnh án pm2 + ghost 2020 nặng hơn; giá trị duy nhất (bắt sóng không-pullback) ~2 runner/6 năm, đã được NICHE_LOSS_MAP định giá là không bù nổi chảy máu depth. |
| rule-only | t2005 (fc_rule2 t2835) | 568.3 ×5 (−165.5) | 275.5 vs 368.5 thua nhưng cùng bậc | ×14.78/−12.25% **NGANG-HƠN gb** (K15 cũng hơn: 15.77 vs 14.55; K50 thua: 6.81 vs 8.50) | **ĐÓNG cho mục tiêu vượt 735 composite** (trần config-only ≈ 578 [ox18], mọi lever composite-dương đều NAV-trung tính/âm; gap = ML selection + winner size). **NHƯNG ghi nhận đặc biệt**: NAV-frame K15–25 ngang/hơn champion với MaxDD thấp hơn, 0 ML, deterministic, dương cả 7 năm — giá trị nằm ở portfolio-frame (diversifier/throughput), không phải leaderboard. |

Điều kiện tuyên bố ứng viên (comp 5-seed ≥ gb+2 VÀ subframe ≥2022 không thua VÀ NAV không tệ hơn đáng kể): **không họ nào đạt cả 3** — không có tuyên bố.

## Files
- fc01_classify.py + fc01_classified.csv; fc02_run.py(+log); fc03_autopsy.py (fc03_ruleonly/nopb/rule2.log); fc04_gap.py; fc05/07/08/09_sweep.py(+log); fc06_run.py(+log)
- Trades: fc_fc_ruleonly_s42_trades.csv (r17), fc_fc_nopb_s42_trades.csv, fc_rule2_s42_trades.csv, fc_r2_ox16_s42_trades.csv
- NAV: pr2_nav_ruleonly_s42.csv, pr2_nav_nopb_s42.csv, pr2_nav_rule2_s42*.csv, pr2_nav_gbx08_verify.csv, pr2_nav_r2ox16_s42.csv
- Clones: fc_ruleonly t2830, fc_nopb t2831, fc_ro_mg/chop/mgchop t2832-34, fc_rule2 t2835, fc_r2_hs12/mgf30/mgf25 + fc_r2_ox16/trail12/tskip50 + fc_r2_ox14/ox18/ox20 (fc07-09)
