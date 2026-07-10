# TUYẾN WAVE-START — HỒ SƠ THIẾT KẾ HỘI TỤ (2026-07-09)

4 mũi research (W1 label, W2 integration, W3 anatomy, P1-P5+F1-F2 probes; ~40 run leaderboard + 2 pipeline offline). Tổng hợp:

## Sự thật đã chốt
1. **Gate chất lượng chân sóng HỌC ĐƯỢC** (W1): label bottom-turn (dưới MA20, dd60≤−12% → reclaim MA20 ≤15 bar + ≥10%/42 bar trước khi thủng đáy−3%): OOS IC +0.13–0.18 cả 4 fold 2022-26, null-check sạch, leakage audit PASS. Model chấm chân sóng BỊ LỠ cao hơn chân sóng đã bắt (55% vs 42% top-decile). Edge = phân loại xác suất reclaim, không phải biên độ. Top-5% ≈ 0.5–1.3 tín hiệu/ngày.
2. **Chân sóng chỉ mời dip ~2%** (W3): median retest −2% dưới thrust-close; đệm 4.5% chỉ có ở 30% (21% trên sóng ≥30% bị lỡ). Limit thô −2% khớp 86% đáy giả (−8.5% avg) → BẮT BUỘC gate. Mua xác nhận vượt đỉnh thrust: capture 89%, +10.6%/21bar, false-fill giảm nửa — dự phòng tốt.
3. **Champion bị nghẽn ở FILL/OCCUPANCY, không thiếu tín hiệu** (P1): union head thứ 5 tăng buy-bars 17.5k→43.9k nhưng trades chỉ +6. Mọi cách đổi fill đều lỗ: anchor swing-low sâu hơn −298 (P5), mua vét cuối window −14.7 (F1, đúng cohort 86% đáy giả + chiếm chỗ lệnh tốt), deepen −4.8 (P2). `bot_ride_k` chết (trail không arm kịp trước signal-exit); `bot_deepen` bị clip sàn 1.0 — score6 KHÔNG thể làm NÔNG fill bằng config.
4. Đường đã đóng vĩnh viễn: union thêm head (vô ích khi fill nghẽn), re-param reversal (noise ±0.3), RS gate hậu-union (−103), anchor/deepen/miss-capture (âm cả ba), at-market toàn cục (−237, A2), depth/window toàn cục (−116..−191, A3).

## Thiết kế còn lại — "QUALITY-GATED SHALLOW FILL" (chưa từng test, mọi mảnh đều có số chống lưng)
Cơ chế: score6 (head bottom-structure) CAO → thu NÔNG độ sâu limit của chính tín hiệu đó (4.5% → ~2% đúng anatomy W3); score6 thấp → giữ nguyên 4.5% (giữ knife-filter). Phản chiếu nhánh conv-shallow có sẵn (engine.py:1636-1639) nhưng điều kiện theo bottom-head.

Engine knob mới (gated default-off, ~15-25 LOC, 1 file):
- `bot_shallow_k: float = 0.0` — `_depth *= clip(1 − k·max(z6,0), bot_shallow_floor, 1.0)` tại khối tính depth (cạnh bot_deepen :1645-1650, dùng chung z của score6)
- `bot_shallow_floor: float = 0.4` — sàn nhân depth (0.44 ≈ fill 2% khi depth gốc 4.5%)
- (tùy chọn v2) `bot_shallow_window: int|None` — rút window riêng cho tín hiệu shallowed (anatomy: 8 bar)
Config kèm: `entry_ensemble5 = bottom_structure_entry_regression (h8, penalty 1.5, dip_window 50, require_turn, park 20/0.5)`; 2 biến thể: có union (z_threshold 0.9 — thêm buy chân sóng, P1 đo gần miễn phí −2.2) và score-only (chỉ shallow tín hiệu sẵn có).

Rủi ro chính & guard: (a) occupancy displacement — fill mới chiếm chỗ lệnh core (bài học F1) → theo dõi cấu trúc trades + per-year, guardrail core-edge; (b) knife qua fill nông — chỉ shallow khi z6 vượt ngưỡng cao; (c) parity — knob off ⇒ bit-identical (test A/B template 2646 + regression test_champions.py).

## KẾT QUẢ "QUALITY-GATED SHALLOW FILL" (bs_*, templates 2715-2721) — BÁC, ĐÓNG NHÁNH
- Knob `bot_shallow_k/floor` đã implement (engine.py:915-926, :1108-1119, :1676-1682; parity PASS byte-identical, knob giữ lại gated-off). 7 config đều âm, monotone theo liều: tốt nhất −27.4 (k05/f66), tệ nhất −49.9.
- **Nguyên nhân gốc (quan trọng hơn kết quả): OCCUPANCY RESHUFFLE.** Limit nông khớp SỚM HƠN → chiếm slot 1-vị-thế/mã sớm hơn → xáo trộn cả chuỗi lệnh: 447 lệnh mới (WR 53.2%, +35.8u) đổi chỗ 427 lệnh core (WR 54.1%, +33.3u) = swap ngang giá hơi lỗ; 468 lệnh chung bị mua đắt hơn +0.82% (−4.09u); MDD 0.175→0.202. Fill mới ở đúng năm điểm mù gần như 0 (2022 +0.04, 2024 +0.004, 2026 −0.005).
- Hệ quả thiết kế: **KHÔNG thể monetize chân sóng bằng cách ĐỔI GIÁ pending book hiện tại** (mọi biến thể repricing đều đã bác: anchor, deepen, shallow, miss-capture). Cửa duy nhất còn lại = cơ chế fill CỘNG THÊM khi slot đang rảnh: **confirmation-breakout B-entry** (W3-A4: mua khi close vượt đỉnh thrust, capture 89%, +10.6%/21bar, false-fill 41% — EV/placement cao nhất bảng anchor), gate bằng score6, stop cấu trúc riêng. Khác biệt cấu trúc với A2 (−237): A2 reprice TOÀN BỘ tín hiệu; B-entry chỉ thêm lệnh tại vùng dưới-MA20 nơi kênh chính không hoạt động (slot rảnh) — displacement tối thiểu về nguyên lý, phải kiểm chứng bằng số.
- Kill-criterion đã định trước: nếu cohort B-trades net âm HOẶC displacement lại chiếm ưu thế → đóng toàn tuyến wave-start ở tầng engine hiện tại, viết post-mortem; hướng còn lại khi đó là kiến trúc multi-position (đã hoãn cùng tuyến sizing).

## KẾT QUẢ "CONFIRMATION-BREAKOUT B-ENTRY" (bch_*, templates 2722-2728) — BÁC, ĐÓNG TOÀN TUYẾN
- Knob `entry_bchannel_*` implement (engine.py, gated default-off; parity byte-identical, test_champions 1 passed/12 skipped). 7 config seed-42 đều âm, monotone theo SỐ LỆNH B trên cả 2 trục (z gate & break_lookback): tệ nhất −46.8 (z09, 371 B), tốt nhất −3.4 (z15_lb8, 46 B) → ngoại suy về 0 chỉ khi knob off, y hệt bs_.
- Giả thuyết idle-slot ĐÚNG mà vẫn thua: 100% B-entry nằm ngoài mọi holding interval của champion, ~66% không có core entry trong ±10 bar — nhưng (a) cohort B chỉ dương nhờ 2020 (ex-2020 âm ở MỌI config), năm điểm mù 2022/24/26 ≈ 0; (b) displacement net âm mọi config (−0.38..−3.23): 34% B-fire chiếm slot SỚM vài ngày trước lệnh core premium (WR 0.6-0.7, hold 25d) và thay bằng scratch WR 0.45/3 ngày; (c) nút thắt MỚI lộ ra ở EXIT: sell head của champion đóng lệnh dưới-MA20 ngay lập tức (100% B exit 'signal', median hold 3 bar) — B-trade không bao giờ được ride tới +10.6%/21bar như anatomy; stop cấu trúc B vô nghĩa (5% fires).
- Kill-criterion định trước THỎA ở mọi config → **đóng tuyến wave-start ở tầng engine hiện tại** (1-slot + exit stack champion). Hướng còn lại (đã hoãn): kiến trúc multi-position + exit head riêng cho wave-start. Chi tiết: bch_impl_notes.md.

## KIỂM ĐỊNH CUỐI "B-ONLY EXIT OVERRIDE" (offline sim, e_sim.py + runs/*/exit_sim.csv) — BÁC, XÁC NHẬN ĐÓNG TUYẾN
Mô phỏng 4 exit stack riêng cho cohort B trên OHLCV (fill engine thực, cost tái lập, không lookahead): câu hỏi mở được trả lời — exit cấu trúc (stop = min-low-10-trước-signal, trail Donchian-20 sau +10%, horizon 60) đưa cohort z09 lên +19.0u tổng / **+11.4u ex-2020** (E0 champion chỉ +5.6), tức B-trade ĐƯỢC ride thì có edge thật đúng anatomy. Nhưng chính việc ride là án tử: hold median 3→45 bar biến idle-slot thành occupied-slot, 309 lệnh core premium (WR 0.64) có entry rơi vào window B đang giữ → core foregone **−24.1u ex-2020** (E2 21-bar ngắn nhất vẫn −17.5u; mọi scheme tương tự), cộng displacement entry-time −3.23u đã đo → net **−15.9u**, hoà vốn đòi realization blocking ≤34% — phi thực tế vì đó là đúng cohort premium từng lộ ở autopsy bs_/bch_. Năm điểm mù cũng âm ròng (B 2022 +0.9 vs chặn core 2022 −7.4). z12_lb8 âm ex-2020 ở MỌI scheme trước cả displacement. Kết luận cấu trúc: **edge/slot-ngày của kênh core (~0.10u/25d, WR .64) gấp ~3 lần B (~0.03u/45d, WR .48) — trong kiến trúc 1-vị-thế/mã, wave-start không thể monetize bằng bất kỳ tổ hợp entry/exit nào; điều kiện cần là slot pool riêng (multi-position).** Đóng tuyến ở tầng engine hiện tại.
(Đính chính narrative: claim cũ "cohort B ex-2020 âm mọi config" là số set-diff lẫn core-reshuffle; số cohort sạch qua join b_entries là +5.6u dưới exit champion — không đổi verdict.)

## Trạng thái
- Bước tiếp: implement knob (nhánh riêng, không commit khi chưa duyệt) + parity check + sweep seed-42: {union on/off} × bot_shallow_k {0.5, 1.0} × floor {0.44, 0.55}. Nếu dương → multi-seed 42/7/99/555 + seed 123 + công tố.
- Artifacts: wavestart/ (W1: wavestart_proto.py, dataset.parquet, oos_scores.parquet; W3: 02_anatomy.py, anchor_*.csv; probes: probes_bchannel.py, logs/; W2 blueprint trong transcript + file này). Templates probe 2707-2714, lineA 2682-2706 — kết quả âm giữ nguyên leaderboard.
