# BẢN ĐỒ ĐIỂM MÙ CHAMPION & ĐỀ XUẤT TUYẾN MODEL MỚI

**Đối tượng:** bundle production `bundle_n2_2643_wavestruct_la05_lamp02_top150_2025-01-01_wf` (champion sau nhiều vòng research, composite ~730, template 2646).
**Dữ liệu:** 3.813 lệnh engine thật (3.781 đóng + 32 mở), top150, 2020-01-02 → 2026-07-08, store serving (`C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db`), tổng PnL đã đóng **+287,6u** (unit-weight), PF 4,53. Engine-parity đã assert bằng instrumented copy (0/3.813 lệch); mọi con số dưới đây đã qua vòng verify độc lập (65 claim, 65 verified — sai lệch chỉ ở mức làm tròn/nhãn, đã sửa trong báo cáo này).
**Artefacts tái lập:** `trades_metrics.csv`, `events.csv`, `unfilled_signals.csv`, `daily_activity.csv`, `depths.parquet`, `headline_stats.json` + các script `tmp_*` và `tmp_verify_*` cùng thư mục này.

**Caveats chung (áp cho mọi con số):** (a) pnl unit-weight/lệnh, engine KHÔNG có sizing/ràng buộc vốn (~54 vị thế mở đồng thời 2025+); (b) giá store chưa back-adjust (~1,4% drift bar cũ; 0/59 big loss dính ex-date lỗi — đuôi sạch); (c) các counterfactual là overlay tĩnh per-trade, chưa qua re-sim leaderboard; (d) các metric "oracle/runup" nhìn trước tương lai, chỉ dùng để định cỡ leak, không phải chiến lược.

---

## PHẦN 1 — TRẢ LỜI TRỰC DIỆN 3 PHÀN NÀN

### Phàn nàn #1: "Hàng khỏe không bao giờ chỉnh 4,5% → lỡ nguyên sóng" — **ĐÚNG, và là leak lớn nhất hệ thống**

- **66,5% limit đã đặt không bao giờ khớp**: 7.582 miss / 11.395 limit đặt (fill-rate 33,5%; `events.csv` do engine tự ghi, không re-sim).
- Cohort không khớp **tốt hơn cohort đã khớp**: at-market next-bar-close ret21 mean **+11,1% gross / +10,4% net** (median +8,2%, **91,5% dương**), ret42 +16,3% — so với +7,61%/lệnh thực nhận. Tổng lý thuyết **+768,3u net @21 bar ≈ 2,7× toàn bộ PnL thực (+287,6u)**. *Cận trên*: cohort được định nghĩa bằng việc-không-chạm-limit trong 40 bar tương lai (~3,3pp của +11,1% là selection; at-market trên TOÀN BỘ 11.145 signal đặt được là +7,84%) — nhưng ngay cả cận dưới đó vẫn ngang cohort đã fill mà không tốn 40 phiên treo lệnh.
- **Fill-rate NGHỊCH chu kỳ**: 26,9% năm momentum (2020, 2024) vs **56,1% năm gấu 2022** — cơ chế pullback khớp nhiều nhất đúng lúc thị trường rơi (nạp dao rơi, nuôi phàn nàn #3) và bỏ lỡ nhiều nhất khi sóng chạy.
- Quy mô sóng bỏ lỡ: 7.582 miss gom thành **1.143 cụm sóng**, **343 cụm (30%) có runup 42-bar ≥ +30%**. Ví dụ: VOS 2021-07-29 **+218%**, FTS 2021-05-05 +152%, GEE 2025-01-17 **+122,9%** (đang chảy máu ở giai đoạn serving thực), OIL 2026-01-09 +110%, MSR 2026-01-06 +104%.
- **NHƯNG miss không phải "suýt khớp"**: giá trong window chỉ chạm sâu median **1,53%** vs limit ~3,9-4,5%; shortfall median **2,41pp**, chỉ 15,6% cách limit ≤1pp. Hạ depth về 3%/2% chỉ bắt lại 14,4%/36,9% số miss, và **full re-sim lịch sử đã chấm shallower THUA** (cp_pb03 653,3 vs control 679,4, PF 4,80→3,93, mdd 0,190→0,225; n2_v11_pb02 506,5 — Postgres leaderboard_runs, verify khớp). Mua đuổi window-end cũng chết cả hai đầu: premium≤2% → net **−1,4%**/21bar (momentum đã tắt); premium>4% (83% số miss) chạy tiếp +8,6%/42bar nhưng phải đu đỉnh +20,8% không buffer — đúng "catastrophe −400" đã ghi ở `engine.py:459-479`.
- Forensic cũ (fw_forensic_out.txt, univ 60): 86/538 sóng >30% bị miss, **99% sóng miss bắt đầu DƯỚI MA20** — thủ phạm chính là entry_gate `upleg_abovema20` chặn chân sóng, chỉ 36% là runaway-không-chỉnh. Trên top150 xác nhận cùng cơ chế: **94,1% fill có giá low rẻ hơn ngay trong 10 bar TRƯỚC tín hiệu** — độ trễ nằm ở tầng gate/tín hiệu, không phải tầng pullback.

**Kết luận #1:** phàn nàn đúng; nhưng lời giải KHÔNG phải chỉnh depth (đã bác 2 lần bằng re-sim) mà là **mở kênh vào lệnh thứ hai** ở chân sóng dưới MA20 + re-entry (Tuyến A, Phần 3).

### Phàn nàn #2: "Hàng chục limit sống 40 phiên, người thường không vận hành nổi" — **ĐÚNG, định lượng xong, và KHÔNG lọc được bằng conviction**

- Trung bình **213,5 limit sống/ngày** (median 154, p90 466, **max 1.102 ngày 2025-03-13**); >10 limit trên 95,1% số ngày. 2025+: mean 242,4. Ngay cả đếm chặt kiểu engine-placed (hủy stack khi có vị thế): **107,2 lệnh/ngày trên 27,6 mã** (~3,9 lệnh chồng/mã), phiên 2026-07-08: **157 lệnh trên 18 mã** (serving `pending_orders()` báo 171 do semantics flat-4,5%). Mỗi ngày phải đặt **22,3 lệnh MỚI** (p90 50); fill nổ bất kỳ lúc nào (62,1% số ngày có fill, max 37 fill/phiên 2021-02-08). Bán thì ngược lại dễ: quyết ở close D, khớp phiên D+1 (`pending_exits` cảnh báo trước 1 ngày) — đúng bất đối xứng user mô tả.
- **Không nén được bằng điểm mô hình**: giữ top-10 limit/ngày theo score chỉ giữ **27,5% pnl**; theo score3 còn tệ hơn. Xếp theo **PROXIMITY** (gần giá kích hoạt nhất) là ít-tệ-nhất: top-20 giữ 80,2% fill / **68,4% pnl / 72,7% big_win** — vẫn mất ~1/3 lợi nhuận. Score tại signal bar phẳng với pnl thực (lift ~1,0×) — khớp prior "FLAT primary score" (`engine.py:695-703`).
- Đòn giảm tải rẻ nhất có giá niêm yết: **cap window 40→20** = mất 13,35u (4,6% pnl), bỏ 25/717 big_win, **−45,6% ngày-lệnh-treo** (333.555 → 181.596 limit-days), −20,1% churn, đồng thời bỏ 6/59 big loss kiểu stale-limit. (Prior sweep w5-w25 từng giảm composite dưới scoring cũ — đây là TRADEOFF vận hành, cần re-run dưới scoring 2026-06.)

**Kết luận #2:** tải lệnh là bản chất của alpha dàn-lưới-mỏng — bài toán là **tự động hóa (broker API/OCO) + đổi kiến trúc vào lệnh** (kênh at-market có stop cấu trúc cho tuyến mới), cộng 3 quick-win serving: sort pending theo proximity, dedup ~3,9× stack/mã, chấm điểm `entry_pullback_cancel_below_ma` (1 run config-only ~20s, đồng thời trả lời #3).

### Phàn nàn #3: "Khớp dao rơi sau tăng nóng, một cú −20/−30% thổi bay nhiều năm lãi" — **ĐÚNG về cơ chế, SAI về quy mô ở unit-weight; hard stop đã bị bác bằng số liệu, rủi ro thật nằm ở tầng sizing**

- **Cơ chế knife là có thật và mang tính cấu trúc**: limit đóng băng tại bar tín hiệu, không re-check gì trong 40 bar chờ (kể cả khi model đã bật −1); signal-exit — cửa cắt lỗ DUY NHẤT (100% big loss thoát bằng signal; trailing 0/59 lần arm vì cần MFE +27% mà max MFE cohort này chỉ 23%) — bị **treo đúng lúc washout** (gate z ≤ −1,75). BID: limit đặt 2020-01-06, khớp sau **đúng 40 phiên** giữa crash COVID, −24,3%. **42,4% big loss dồn vào 3 tháng sập index** (2020-03: 11, 2022-10: 9, 2022-06: 5).
- **Quy mô nhỏ hơn nỗi sợ**: 59 lệnh ≤−15% chỉ **−10,77u = 2,9% gross profit**; lệnh tệ nhất TIG −27,1%; **không có lệnh đóng ≤−30%**; worst-10 cộng lại chỉ phá 0,62% gross. "Hot-chase knife" (VGT 2026 −22,5% sau pre_ret20 +28,7%) là đuôi hiếm: ô hot_run & fill-nhanh thực ra là ô TỐT NHẤT (+17,55%/lệnh, bigloss 0,83%).
- **Mọi tail-cutter cơ học đều LỖ RÒNG trên chính 3.781 lệnh này** (sim per-trade với OHLCV thật, fill next-bar-close, cost khớp engine, verify 100%): stop −8%: **−8,82u**; stop −10/−12%: −5,2/−4,07u và còn làm bucket ≤−15% TỆ HƠN (59→66/73 lệnh) vì **gap sàn VN**: 44/472 fill của stop −8% đáp ≤−15%, MSR 2025-04 từ −16,65% (engine chờ hồi) thành **−25,57%** (2 phiên sàn giữa breach và fill); time-stop 15: −7,99u (giết VGI +261%); breakeven-lock 8%: **−32,15u** (giết 88 big_win: KBC +178%→+1,7%). Signal-exit hiện tại thoát chỉ cách đáy ~2,2pp và 45,8% big loss bật >+5% trong 21 bar sau bán — mọi stop bán sớm hơn chỉ khớp sâu hơn trong hố. Tái lập độc lập leaderboard cũ: template hard-stop 656/541/âm vs champion 730,4.
- **Chỗ chưa có câu trả lời**: unit-weight nói "một lệnh không thổi bay gì", nhưng user cảm nhận ở tầng danh mục — với ~54 vị thế đồng thời không ràng buộc vốn, một vị thế over-size −25% vẫn nguy hiểm. Cần `_tmp_analysis/capital_portfolio_sim.py` dịch −10,77u đuôi thành %NAV (chưa chạy — việc bắt buộc của Tuyến B).

**Kết luận #3:** không thêm hard stop phẳng (đã bác 2 tầng: sim tĩnh + leaderboard). Đường giảm đuôi hợp lệ: (a) hủy limit đang treo khi washout kích hoạt / `cancel_below_ma` (knife fill nhanh đầu cú sập = 80% đuôi, median wait chỉ 3 bar); (b) stop phẫu thuật theo cohort đã build sẵn nhưng OFF (`csr_hard_stop_pct`, `hard_stop_atr_mult`, `downtrend_hard_stop_pct` — engine.py:393-401/436-441/653-658) với điều kiện "không fill trong washout"; (c) **giải bằng sizing** (Tuyến B) — size nhỏ khi vào, chỉ tăng khi lệnh tự chứng minh.

---

## PHẦN 2 — BẢN ĐỒ ĐIỂM MÙ (RANKED THEO TÁC ĐỘNG PnL KỲ VỌNG)

| # | Điểm mù | Định cỡ (đã verify) | Bằng chứng chốt |
|---|---------|---------------------|------------------|
| **1** | **Kênh vào lệnh khuyết ở chân sóng** — gate `upleg_abovema20` + pullback 4,5% đứng ngoài 66,5% limit đặt; miss không phải suýt-khớp | Cận trên **+768u net/21bar** (2,7× realized); cận dưới: all-placed at-market +7,84%/21bar ~ ngang cohort fill; 343 cụm sóng ≥30% bị bỏ | `unfilled_signals.csv`; VOS +218%, GEE 2025 +122,9%; fw_forensic: 99% sóng miss start dưới MA20 |
| **2** | **Bán hớ có điều kiện (runner bị signal-exit cắt trước khi trail kịp làm việc)** — KHÔNG phải blanket-hold (đã bác: +10bar chỉ +22,1u, −27u riêng 2022) | 863 lệnh mfe≥27% thoát bằng signal: giveback **−100,8u**, 64,1% chạy tiếp; trail chỉ nổ **7 lần/6 năm**; 402 round-trip winner (+72u trần); oracle bán-đỉnh +333u (53,7% giá trị đỉnh bị trả lại) | ORS +52,9% sau mfe +86,9% rồi giá chạy thêm +93%; DDV exit +7,2% rồi +115%/21bar; 80% cohort bán-hớ rebuy giá cao hơn +5,19% (+4,88% bóc slippage) |
| **3** | **Thông tin hậu-entry không được quy đổi thành SIZE** — winner tự khai báo trong vài bar đầu nhưng mọi lệnh đều 1 unit | Cohort mae≥−2%: 36,8% lệnh, **WR 87,1%, +225,4u (78% pnl)**; prior: up-by-bar-3 ⇒ 93% WR; **pyramid sweep +129..+248 composite (seed 42) — delta lớn nhất archive, CHƯA từng multi-seed/promote** | `pyramid_sweep_result.json`; engine.py:294-302; đồng thời là lời giải sizing cho phàn nàn #3 |
| **4** | **Regime-idle: engine chạy không tải ~40% thời gian** — năm phẳng không cháy nhưng không sản xuất | Bigwin-rate 2020/2021: 32,6/29,3% vs **2024: 5,4%, 2026 YTD: 5,1%**; PF 9,0/9,3 vs 1,81/0,83; cohort non-bigwin ~0 MỌI năm (−6,1..+3,7u) — sàn lỗ đã được giữ, trần nằm ở sản xuất big_win | Trùng gốc với #1: fill-rate thấp nhất đúng năm momentum; 2026: 32 lệnh mở MTM +1,49u chưa tính — không phải "edge chết" |
| **5** | **Tải vận hành + churn** — chi phí uy tín/slot vốn, không phải chi phí pnl | 213,5 limit sống/ngày (max 1.102); churn 34,2% số lệnh chỉ −1,84u nhưng đốt ~9u friction + slot; hold≤8 bar: 45,4% số lệnh, −33,1u | Không lọc được bằng score (top-10 giữ 27,5% pnl); proximity top-20 giữ 68,4% |
| **6** | **Đuôi knife + lỗ hổng tầng portfolio** | Đuôi đã đóng chỉ −10,77u (2,9% gross) và được exit machinery quản gần tối ưu cục bộ (mọi stop sim đều âm); NHƯNG %NAV drawdown 2020-03/2022-10 với 54 vị thế đồng thời **chưa được đo** | MSR mae −33,9% (đáy 2025-04-09) vẫn đóng −16,65% tốt hơn mọi stop; câu hỏi mở duy nhất = sizing |

**Luật bất biến rút ra (đã xác nhận lần 3, trên top150):** *SELECTION WALL* — gần như mọi cohort đều net-dương, nên mọi GATE entry đều cắt vào lợi nhuận (14/14 filter thử đều âm: green-only −144,4u, skip hot −95,5u, conviction floor −103,9u...), và mọi cơ chế THOÁT-SỚM cơ học đều amputate đuôi big_win (4/4 sim âm). Tín hiệu yếu chỉ được dùng làm **MODULATOR** (độ sâu fill, độ dài hold, SIZE) — đây là điều kiện biên của mọi tuyến mới.

---

## PHẦN 3 — CORE EDGE PHẢI BẢO TOÀN (guardrail = acceptance test cho mọi tuyến mới)

Hệ thống là **nhà máy một sản phẩm**: 717 big_win (19% số lệnh, mean +42%, hold median 63 bar) = **104,8% toàn bộ PnL**; 3.064 lệnh còn lại net −14u (phí quyền chọn). Mọi tuyến mới phải chứng minh KHÔNG làm rơi các cohort sau trước khi so composite:

1. **Bigwin-rate ≥19% và bigwin-pnl-share ~100%** — không tối ưu WR/mean của cohort giữa.
2. **Kiên nhẫn giữ lệnh**: cohort hold>20 bar = 39,6% số lệnh, **WR 84,2%, mean +21,05%, 109,6% pnl**.
3. **Fast-shallow-fill slice** (wait≤4 bar & fill tại/trên MA20): 30,4% lệnh, **48,0% pnl, dương TỪNG năm 2020-2025** — hình mẫu CEO 2021-10-08 +290% (wait 1, trên MA20).
4. **Conv shallow-fill hoạt động**: quintile fill nông nhất +11,94%/lệnh vs full-depth +7,52%; fill nông nhất (≤3,0%) WR 80%, +27,2%, 0 big loss.
5. **Không gate màu nến / momentum nóng**: cohort nến đỏ = **+144,4u (50,2% pnl)** dù WR thấp hơn; decile pre_ret20 nóng nhất là decile TỐT NHẤT (+16,5%/lệnh, +62,4u); hot&fast +17,55%/lệnh.
6. **Breadth cấu trúc**: 149/150 mã net ≥0 (duy nhất SAB −0,24u); top-10 lệnh chỉ 7,3% pnl — edge không phụ thuộc stock-picking.
7. **Giữ nguyên market-washout exit gate** trong mọi thiết kế: bằng chứng 2025-04 — trail nổ trong washout bán đúng đáy (25 big_loss MỚI, cluster entry 2025-04-03 fill sim 04-09 = đáy, trong khi signal-exit chờ hồi thoát 04-14).
8. **Signal-exit đúng trong gấu**: tỷ lệ bán-hớ 2022 chỉ 48% vs 67% 2021; blanket hold 2022 = −27u — mọi can thiệp exit phải regime-conditional và chỉ áp cho winner.

---

## PHẦN 4 — BA TUYẾN MODEL MỚI

> Nguyên tắc chung: champion là sản phẩm của ~2.600 template; mọi trục "hiển nhiên" (shallower pullback, fill-if-missed, confirm-reversal, hard stop, no-pullback, blanket hold, structural deepening) ĐỀU đã bị bác bằng leaderboard — bảng đối chiếu prior đặt trong từng tuyến. Ba tuyến dưới đây nhắm vào 3 seam CHƯA từng được chấm: kênh-vào-lệnh thứ hai, sizing, và handoff-exit có điều kiện.

### TUYẾN A — "WAVE-START B-CHANNEL": kênh vào lệnh thứ hai ở chân sóng, at-market có xác nhận + stop cấu trúc
**Khai thác:** điểm mù #1 + #4 (+768u trần; 99% sóng miss start dưới MA20; fill-rate 26,9% đúng năm momentum) và giảm điểm mù #5 (kênh B không treo limit).

**Cơ chế:**
- Giữ NGUYÊN kênh A (pullback champion, không đụng).
- Kênh B chỉ kích hoạt khi kênh A "mù": tên có entry-signal/score dương nhưng **dưới MA20** (vùng gate chặn) hoặc vừa bị trailing/signal-exit thoát mà cấu trúc còn nguyên. Vào lệnh **at-market close-next CÓ XÁC NHẬN** (V-bottom confirm: close xanh vượt high bar trước + volume; grading bằng exit-head z — prior đã đo splitter này tách "+9,7% bottoms khỏi +1,6% knives"), stop = **swing-low cấu trúc** (`structural_stop_lookback` — đã build, OFF), KHÔNG đặt limit chờ.
- Nhánh re-entry: bật `resume_reentry_win` + `reentry_max_premium_pct` (cap 2-4%) — prior đo 80% trailing-exit resume trong ~3 bar, re-entry mean **+3,10%/+1.186u** (engine.py:524-532); đồng thời vá luôn vết "bán xong mua lại +5,2% premium" (80% cohort bán-hớ).

**Đối chiếu prior (tại sao lần này khác):**
- `fill_if_missed`/premium-cap: bác (−400, fm_p2 net −1,4%) — kênh B KHÔNG chase window-end, nó vào ở CHÂN sóng trước khi premium tồn tại.
- Shallower/structural-deepening limit: bác (cp_pb03 −26 comp; deepen 486→116) — kênh B không đụng trục depth.
- No-pullback line: bác (429-454 comp) vì exit machinery không quản nổi at-market KHÔNG stop — kênh B có structural stop riêng, và chỉ chạy trên subset dưới-MA20 thay vì thay toàn bộ entry.
- `early_entry_reversal` từng sweep (npe_ee_*, không adopt) — kết quả chi tiết không còn trên disk; **việc đầu tiên là re-run dưới scoring 2026-06** (SCORE_PNL_W 0,45 + Sortino — scoring cũ phạt full-wave capture, chính là lý do nhiều lever "ride" chỉ thắng sau khi đổi scoring). Lưu ý mở: key `early_entry_reversal` không nằm trong EngineConfig hiện tại — xác minh nó ở tầng pipeline trước khi clone.

**Rủi ro chính:** (1) knife ở đáy — splitter exit-head-z phải giữ IC OOS; (2) tăng số lệnh → shrink composite + tải vận hành (kênh B phải bị cap số slot/ngày); (3) 2022: chân-sóng-giả liên tiếp — bắt buộc giữ washout gate cho cả entry kênh B.

**Thí nghiệm #1 (config-only nếu key còn ở pipeline; ~20s/label với cached predictions):** clone template 2646 theo đúng pattern `stock_ml/scripts/deploy_wavestruct.py` (BASE_TMPL=2646, `run_template_experiment`), axes: `resume_reentry_win ∈ {2,3,5}` × `reentry_max_premium_pct ∈ {0.02,0.04}` (nhánh re-entry chạy được NGAY, không cần code mới); song song 1 sweep `early_entry_*` nếu key tồn tại, ngược lại thêm knob nhỏ theo tiền lệ volhold. Multi-seed 42/7/99/555, so mean vs 2646 (~730).

### TUYẾN B — "REVEAL-AND-ADD" SIZING LINE: từ chọn-lệnh sang phân-bổ-vốn
**Khai thác:** điểm mù #3 (mae≥−2%: WR 87,1%, +225,4u; up-by-bar-3 ⇒ 93% WR; **pyramid sweep +129..+248 composite — delta lớn nhất archive chưa từng promote**) và là lời giải ĐÚNG TẦNG cho phàn nàn #3 (đuôi −25% nguy hiểm vì size, không phải vì tần suất — 59 big loss chỉ 2,9% gross).

**Cơ chế:**
- Entry y hệt champion nhưng **half-size**; add unit khi lệnh tự chứng minh: giá up-by-bar-3 / mae chưa thủng −2% / vượt +X% (lever `pyramid_add_units` đã build). Đuôi knife tự động chỉ ăn half-size → max loss/lệnh cắt ~50% mà KHÔNG gate (né selection wall).
- Nhánh đối xứng: **partial-exit** — bán 1 phần khi tín hiệu đỉnh (`mfe_act_k`: score4 dự báo fwd-20-bar peak, rank-IC +0,225, "lever duy nhất nối exit với dự báo đỉnh chưa dùng"), giữ phần còn lại theo Donchian-80 — cách duy nhất "bán sớm hơn" không amputate big_win (mọi exit toàn phần đã bác ở Phần 1/#3).
- Bắt buộc kèm **portfolio sim** (`_tmp_analysis/capital_portfolio_sim.py`) trên trades_raw: dịch unit-weight → %NAV, trả lời câu "một lệnh thổi bay nhiều năm lãi" bằng số danh mục thật (chưa ai đo).

**Đối chiếu prior:** pyramid chưa từng bị bác — nó bị BỎ QUÊN (không multi-seed, lo ngại pnl_pct semantics đổi khi có size). Đây là lý do nó là ứng viên #1: kết quả seed-42 đã có (+129 u05_t02 / +248 u10_t02, mdd 0,203-0,241 vs base 0,19), engine hook sẵn, chi phí thử = 4 run config-only.

**Rủi ro chính:** (1) comparability — composite trên pnl_pct đổi nghĩa khi size không đều → phải báo song song composite + portfolio-frame (total_pnl vs MDD như struct_trail từng làm); (2) add-in-washout — cấm add khi market-drop gate active; (3) vốn: 54 vị thế đồng thời + add units → ràng buộc vốn bind sớm hơn, portfolio sim quyết định số slot.

**Thí nghiệm #1:** clone 2646 + `pyramid_add_units ∈ {0.5, 1.0}` × trigger `{up-by-bar-3, mfe≥4%}`; multi-seed 42/7/99/555 + audit seed 123; sau đó capital_portfolio_sim với cap vị thế {20, 30, 54}. Toàn bộ config-only.

### TUYẾN C — "RUNNER HANDOFF": chuyển giao exit có điều kiện cho structure-trail + thu hồi bằng re-entry
**Khai thác:** điểm mù #2 — 863 runner (mfe≥27%, ĐÃ vượt ngưỡng arm trail) vẫn thoát bằng signal-exit: giveback **−100,8u**, 64,1% chạy tiếp sau bán (runup21 +11,9%), trong khi Donchian-80 trail — thiết kế riêng cho cohort này — chỉ nổ **7 lần/6 năm** vì signal-exit luôn bắn trước ("signal exit sells the mean, pre-empts overext" — forensic vol_adaptive đã ghi trong engine).

**Cơ chế:**
- Khi `peak_gain ≥ trailing_activate_pct (0,27)`: **suppress signal-exit, trao toàn quyền cho Donchian-80** (structure break = close < đáy 80 bar). Tiền lệ kỹ thuật đã có: volhold là đúng pattern "suppress one exit, drop no trade" và Donchian ride từng nhân đôi runner return (15,6%→28,3%, template 2482 "MDD −8%, PF +19%").
- Van an toàn regime (bắt buộc, vì 2022 signal-exit ĐÚNG): handoff chỉ khi ngoài washout (market-drop gate inactive) và score3 z ≥ 0 — tái dùng máy volhold đang ON.
- Nhánh phụ giá rẻ: siết `signal_exit_protect_release_drop_k` 2,5 → {3,0, 3,5} (protect đang thả trên nhịp chỉnh: 59,6% cohort bán-hớ là cut-on-dip, band mfe 10-27% còn −77,1u giveback residual) — 2 run config-only.
- Recapture: với exit đã xảy ra mà trend còn nguyên, bật resume re-entry premium-capped (chung lever với Tuyến A) — nhắm thẳng "80% mua lại giá cao hơn +5,2%".

**Đối chiếu prior (quan trọng — trục này đầy xác chết):** blanket hold +10bar (+22,1u, −27u/2022) — bác; %-trail 6/8/10% (−124,9/−74,6/−36,7u, big_loss 59→66) — bác; breakeven-lock (−12,5u, giết KBC +178%) — bác; giveback_sweep candidates (vol_spike −34,8, stale_8 −66,1) — bác. **Cái CHƯA test là handoff có điều kiện mfe≥27%**: khác về bản chất vì nó không thoát sớm hơn mà thoát MUỘN hơn có cấu trúc, trên đúng cohort mà oracle nói còn +11,9%/21bar, và chỉ 863 lệnh (không đụng 2.918 lệnh còn lại).

**Rủi ro chính:** (1) Donchian giveback khi nổ là −18,7% mean từ đỉnh — có thể chỉ đổi dạng giveback; cluster Apr-2024 cho thấy cả 7 lần trail nổ đều vẫn bán hớ; (2) 2022: runner cầm qua gấu — van score3/washout phải giữ; (3) knob mới = 1 dòng engine (theo tiền lệ volhold), mất bit-parity wheel → phải rebuild wheel theo quy trình §11a.

**Thí nghiệm #1:** (bước 0, config-only) sweep `release_drop_k ∈ {3.0, 3.5}` + `mfe_act_k` (score4) ∈ {0.5, 1.0} + `hold_runscore_scale` (runscore.parquet đã train sẵn ở `results/_research_2429/`); (bước 1) thêm knob `signal_exit_defer_above_act: bool`, clone 2646, multi-seed. Kỳ vọng định lượng: thu hồi >0 từ pool −100,8u mà không làm 2022 xấu đi >5u.

**Không đề xuất lại (đã bác, chỉ liệt kê để đóng hồ sơ):** shallower/deeper pullback, fill_if_missed ± premium cap, confirm_reversal (green-only), hard stop phẳng mọi mức, time-stop, breakeven-lock as-is, blanket hold, top-K theo score.

---

## PHẦN 5 — LỘ TRÌNH RESEARCH & TIÊU CHÍ PROMOTE

### 5.1 Trình tự (rẻ trước, đắt sau)
1. **Tuần 0 — config-only trên harness cache (~20s/label):** Tuyến B sweep pyramid (4 run); Tuyến C bước 0 (release_drop_k, mfe_act_k, hold_runscore_scale — 6 run); Tuyến A nhánh resume re-entry (6 run); ops: 1 run `entry_pullback_cancel_below_ma`, 1 run window 40→20 dưới scoring mới. Tất cả clone template 2646 theo pattern `stock_ml/scripts/deploy_wavestruct.py` + `run_template_experiment`, đăng ký Postgres leaderboard_runs.
2. **Tuần 1-2 — multi-seed + audit:** ứng viên nào Δmean > 0 ở bước 1 → chạy đủ seeds 42/7/99/555, audit unseen-seed 123 (`audit_champion_newseeds.py`) chống overfit-engine-param.
3. **Tuần 2-3 — portfolio frame:** capital_portfolio_sim cho mọi ứng viên sống sót (bắt buộc với Tuyến B); báo song song composite + (total_pnl, %NAV MDD, tail/lệnh).
4. **Tuần 3+ — code mới (nếu cần):** knob handoff (Tuyến C) / kênh B (Tuyến A) → rebuild wheel, verify không drift bundle đang chạy (§11 quy trình a).
5. **Shadow serving:** export bundle ứng viên (`export_bundle.py --replicate-last-fold`), chạy song song trên serving ledger (signal_log append-only — baseline hiện mới 6 phiên confirmed) tối thiểu 4-6 tuần trước khi swap.

### 5.2 Tiêu chí promote — apples-to-apples với champion (khắt khe, vì champion = ~2.600 template đã thử)
**Trục hiệu năng (không đổi so với protocol hiện hành):**
- Composite `scoring.py` bản 2026-06 (SCORE_PNL_W 0,45, Sortino, MDD convex), walk-forward 2020-2025, costs 0,0015+0,0015+0,001, universe pin `univ_2429.txt` (60 mã).
- **Multi-seed MEAN (42/7/99/555) > champion mean ~729-730**, VÀ giữ trên unseen seeds; đủ lệnh (~≥1.300/seed như champion 1.386 — shrink sqrt đã phạt thiếu lệnh, không được "ít lệnh điểm cao").
- Nếu tuyến đổi khung so sánh (sizing/portfolio): phải khai báo TRƯỚC objective thay thế (total_pnl vs %NAV-MDD) và vẫn báo composite tham chiếu.

**5 trục PARITY train↔serving** (bắt buộc, theo `C:/Users/DUC CANH PC/Desktop/stock-serving/CLAUDE.md` §11 — đã có case study wavestruct PF 5,94→5,25 chỉ vì lệch scope): (1) PROVENANCE — export từ train, không self-train; (2) PRICE — feature nhạy giá tuyệt đối (ext_atr, zigzag leg, structural stop của Tuyến A/C!) phải chạy trên giá back-adjust khớp train; (3) UNIVERSE pin đúng universe chấm điểm; (4) METRIC/SEED — so aggregate cùng cửa sổ OOS, pin seed khi so PF; (5) **SCOPE feature/breadth** — trục lớn nhất: breadth/CSRank tính trên đúng số mã của run (61-scope PF 5,84 vs 488-scope 5,25), ghi `feature_scope` vào manifest.

**Trục VẬN HÀNH (MỚI — điều kiện cần để promote, baseline đo từ báo cáo này):**
| Metric | Baseline champion | Yêu cầu tuyến mới |
|---|---|---|
| Limit sống/ngày (median / p90 / max) | 154 / 466 / 1.102 | Không tăng; Tuyến A kênh B phải GIẢM (at-market) |
| Lệnh mới phải đặt/ngày (mean / p90) | 22,3 / 50 | ≤ baseline |
| Fill-rate limit theo năm | 26,9%-56,1% (nghịch chu kỳ) | Báo cáo bắt buộc; fill-rate tăng vọt = alarm regime |
| Max loss 1 lệnh đã đóng | −27,1% (TIG) | Không xấu hơn; Tuyến B đo thêm %NAV |
| Số lệnh ≤−15% / 1.000 lệnh | 15,6 (59/3.781) | Không tăng khi thêm kênh entry |
| %NAV MDD (portfolio sim, cap 30 vị thế) | **chưa đo — phải thiết lập baseline trước khi promote bất kỳ tuyến nào** | ≤ baseline |

**Guardrail core-edge (Phần 3, chạy như acceptance test trên trade-frame của ứng viên):** bigwin-rate ≥19%; slice wait≤4 & dist≥0 giữ ~48% pnl và dương từng năm; cohort hold>20 giữ WR ~84%; không rơi cohort nến đỏ (+144u) / hot-run (+95u); ≥95% mã net dương; washout gate còn nguyên.

### 5.3 Việc serving làm ngay không cần model mới (đã có giá niêm yết)
1. `pending_orders` thêm cột distance-to-limit, sort mặc định theo proximity (nếu buộc cắt còn 20 lệnh: giữ 68,4% pnl / 72,7% big_win vs 41,1% nếu cắt theo score).
2. Dedup stack ~3,9 lệnh/mã (157 lệnh/18 mã phiên 2026-07-08) — cần 1 leaderboard run đo cost trước khi áp.
3. Giám sát 4 metric vận hành (bảng trên) + bigwin-rate rolling làm health metric; đừng diễn giải 2026 YTD PF 0,83 là edge chết (32 lệnh mở MTM +1,49u; mẫu giống năm phẳng 2024).
