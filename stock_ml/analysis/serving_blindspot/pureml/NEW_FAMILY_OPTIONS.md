# NEW_FAMILY_OPTIONS — Bản đồ không gian phương án tuyến model KHÁC HỌ

Ngày: 2026-07-10. Mục tiêu: vượt top-1 gb_x08 (template 2783, mean-5 = 733.76, seed-42 = 735.0,
họ `regression_dual_ml_recombine_decoupled` + pullback-limit 4.5%/40 + exit rule-dominant).
Ràng buộc: kiến trúc 1-vị-thế/mã; dữ liệu hiện có = daily OHLCV VN; cùng universe/cost trên
leaderboard; compute thoải mái. Thuần phân tích — chưa chạy experiment nào cho hồ sơ này.

---

## 0. Kiểm kê năng lực hạ tầng (chạy ngay vs cần code)

### 0.1 Strategy classes (`src/pipeline/experiment.py`)
| Nhóm | Biến thể | Trạng thái |
|---|---|---|
| Recombine family | `regression_dual_ml_recombine` + `_zexit`, `_zexit_both`, `_ema`, `_xrise{,_ema,_raw}`, `_decoupled{,_ema}` (9) | Chạy ngay — họ champion |
| Action classifier | `single_ml_action_classifier`, `_exitout` (SMAC) | Chạy ngay — trần đã đo 259.6, chết |
| Legacy | `v19_3`, `rule_only` | Chạy ngay |
| **Cross-sectional rank làm lõi** | — | **KHÔNG có.** csrank chỉ tồn tại dạng feature (`entry_csr`, norm `csrank`, `tier2_csr_threshold`) — cần code mode mới hoặc adapter tín hiệu |
| Portfolio top-K rebalance | — | KHÔNG có trong engine (engine = signal-driven, pending-limit, 1 slot/mã) |

### 0.2 Model classes
- `src/models/registry.py`: entry/exit LGBM, XGB, RandomForest, MLP, rule; LSTM = stub `NotImplementedError`. Regression: LGBM/XGB/RF/MLP.
- `src/components/models/registry.py`: lightgbm, xgboost, **catboost**, random_forest, **gru** (`gru_seq.py`), ensemble.
- → Sequence model (GRU) đã có xương; TCN/transformer cần code mới. LGBM ranker (lambdarank) chỉ cần objective param — gần như free.

### 0.3 Label families (`src/data/target.py`)
Có sẵn: `trend_regime` (dual_ma/hhll), `return_classification`, `return_regression`,
`forward_risk_reward` (gain/loss/rr barrier — chính là dạng first-passage thô), `early_wave`,
`early_wave_v2`, `early_wave_dual` (+`target_sell`), `generate_exit_labels` độc lập.
KHÔNG có: label cross-sectional (demeaned-by-date), label điều-kiện-fill (đo outcome từ giá
limit-fill giả định thay vì giá signal bar), label imitation từ trade log.

### 0.4 Split/đánh giá
`YearSplitter` walk-forward + `PurgedKFoldSplitter` (`src/data/splitter.py`). Leaderboard +
multi-seed 42/7/99/555 + chuẩn regime ≥2022 bắt buộc + protocol clone-template append-only,
knob gated default-off + golden parity (NEXT_LINE_IMPLEMENTATION_PLAN §0.4).

### 0.5 Trần pure-ML thực đo (pm01_classified.csv)
Class-a (pure-ML exit) tốt nhất **619.1** (tpl 1058), class-b **625.8** (tpl 1053) — đều decoupled.
→ exit rule stack đóng góp ~110+ điểm composite. Bài học cho MỌI tuyến mới: đừng vứt execution
layer (pullback buffer + exit stack) khi thay não; và đừng kỳ vọng tuyến mới chạm 735 ngay ở E1.

---

## 1. Ma trận án tử — phạm vi chính xác (để không re-skin đồ chết)

| Án | Con số | Phạm vi án | KHÔNG áp cho |
|---|---|---|---|
| Bỏ/nông pullback 4.5% | −237 (A2 492.3); ladder depth chảy máu y hệt mọi exit | **Vĩnh viễn, mọi thế hệ exit** | Tuyến giữ nguyên pullback làm execution layer |
| Exit-timing daily OHLCV | 8 lever âm; 0/60+ sim dương; quantile/OR/z tĩnh đều chết | Mọi config/head exit trên **data hiện có** | Exit từ data MỚI (intraday/flow) |
| Threshold 1-tham-số | zE pass 97.5-99.6%; calibration đảo chiều theo năm | Họ ngưỡng trên chuỗi score hiện tại | Kiến trúc quyết định khác (rank ordinal, cancel-policy) |
| Label mới → head cũ | exit2 OR −3.3; reward_risk h10 đảo calibration; pv_corr −0.8/−1.5 | Label mới **bơm vào chuỗi quyết định z-threshold hiện tại** | Label mới + chuỗi quyết định mới |
| Regime-switching | oracle ≥2022 = +1.78u < noise ±2u | Switch giữa **các model hiện có cùng họ** | (trần có thể khác với họ thật sự khác — nhưng prior thấp, xếp sau) |
| Conviction repricing | chết (scoreaudit + conv_scaling) | Trục scale depth/threshold theo conviction | Quyết định rời rạc cancel/giữ lệnh treo |
| SELECTION WALL | mọi entry gate cắt lãi; feature set thay thế thua 13-19 pts | Gate **trừ-bớt** trên tín hiệu timing; feature-injection vào head cũ | **Ranking làm LÕI chọn-mã** (chưa từng bác); nguồn tín hiệu thay thế hoàn toàn |
| Non-LGBM algos | RF 60.9, MLP 211.3, GRU 185.0 | Algo khác trên **label + feature tay hiện tại** | Sequence model trên OHLCV THÔ với label/nhiệm vụ khác |
| Runaway/takeover, re-entry, wave-start, tuyến A | occupancy −197u; oracle re-entry −0.9u; wavestart net −15.9u | Mọi kênh entry thứ hai trong 1-slot với exit stack champion | (đóng thật — không mở lại trong 1-slot) |
| Coverage/head entry mới | 97.8% miss = SELL-VETO, true-silence 0% | Trục "model mù" — union threshold/head phụ | Selection giữa các tín hiệu ĐÃ trên ngưỡng |
| Derivatives daily | 13/13 feature trong null band | Feature phái sinh daily cho exit | Intraday (chưa tồn tại — trục "chưa thử" duy nhất được nêu tên trong NICHE_LOSS_MAP) |

---

## 2. Không gian phương án

### P1 — Cross-sectional ranking làm lõi quyết định (chọn-mã tương-đối)
- **Cơ chế**: mỗi tuần, LGBM (regression trên fwd-return demeaned-by-date, hoặc lambdarank)
  rank toàn universe. Mã vào top-K ⇒ tín hiệu BUY; rớt khỏi top-K′ (hysteresis K′≈2K) ⇒ SELL.
  Execution GIỮ NGUYÊN pullback-limit 4.5%/40 + phần exit rule tối thiểu (hard-stop/washout) —
  chỉ thay NÃO chọn-mã, không đụng alpha đệm giá.
- **Khác nguồn alpha**: champion trả lời "mã X bây giờ có đáng mua không?" (timing tuyệt đối);
  P1 trả lời "trong 488 mã, mã nào ĐÁNG NHẤT tuần này?" (selection tương đối). Cross-sectional
  demeaning loại bỏ beta thị trường khỏi label — thứ mọi label hiện tại đều nhiễm.
- **Vì sao án cũ không áp**: SELECTION WALL bác gate trừ-bớt và csrank-làm-feature/trigger-norm;
  chưa ai chạy ranking làm objective + portfolio construction. "Label mới chết" xử label bơm vào
  chuỗi z-threshold; ở đây chuỗi quyết định là thứ tự ordinal + membership top-K — không có
  threshold một-tham-số nào để chết theo kiểu cũ. 1-slot/mã tương thích tự nhiên (top-K = K mã
  khác nhau, mỗi mã 1 slot).
- **Chi phí xây**: E1 offline thuần pandas+LGBM (0 code engine). E2 = signal-adapter nhỏ
  (membership → cột signal ±1) đi qua engine hiện tại ⇒ so sánh leaderboard công bằng tuyệt đối
  (cùng universe/cost/composite). Knob gated default-off + parity.
- **Rủi ro chính**: (a) rank cao thường = momentum đang chạy ⇒ pullback 4.5% không fill (bài học
  runaway); cần đo fill-rate của cohort top-K ngay tại E1; (b) turnover rebalance ăn cost;
  (c) alpha selection VN có thể trùng phần lớn với momentum head champion đã có (resid-IC check).
- **Thí nghiệm quyết định rẻ nhất**: offline — walk-forward YearSplitter, rank-IC + spread
  top-20 vs universe EW net cost, lát ≥2022, và fill-rate giả lập limit 4.5% trên cohort top-K.
  Kill bar: spread ≥2022 ≤ 0 hoặc rank-IC < 0.03 hoặc fill-rate < ~50% ⇒ đóng. ~1-2 ngày.

### P2 — Label điều-kiện-fill / first-passage trên sổ lệnh treo (order-book policy)
- **Cơ chế**: dựng dataset mọi limit ĐÃ ĐẶT (11,395 lệnh: trades_raw + unfilled_signals +
  pending_orders_ref có sẵn trong serving_blindspot/). Label = first-passage P(chạm +g trước −l
  trong H bar) TÍNH TỪ GIÁ FILL (limit price), không phải giá signal bar; features tại thời điểm
  giá tiệm cận limit (tốc độ rơi, breadth, SNR, tuổi lệnh). Model quyết định HỦY/GIỮ lệnh treo.
- **Khác nguồn alpha**: champion đặt lệnh rồi mù 40 bar; mọi label hiện tại đo outcome từ signal
  bar — không ai từng hỏi "cú rơi 4.5% NÀY là chiết khấu hay là dao?". BLINDSPOT #2 đo được:
  score tại signal bar PHẲNG với pnl trên sổ lệnh treo (lift ~1.0×), proximity thắng score —
  tức khoảng trống thông tin có thật, chưa model nào chiếm.
- **Vì sao án cũ không áp**: "hard stop chết" xử lệnh ĐÃ fill; "conviction repricing chết" xử
  scale depth/threshold; đây là quyết định rời rạc cancel/keep TRƯỚC fill — surface chưa có án.
  Knife-cohort (59 lệnh ≤−15%) và 2022/2024 (năm pullback = dao) là target tự nhiên.
- **Chi phí xây**: E1 hoàn toàn offline trên artefact có sẵn (~1 ngày). E2 = knob engine
  `pending_cancel` (nhỏ, cùng cỡ giveback_guard đã làm: 2 điểm chạm engine.py) + parity.
- **Rủi ro chính**: (a) hủy lệnh tốt = mất big_win (717 lệnh = 104.8% PnL — bất đối xứng chết
  người, model phải cực kỳ precision-first); (b) trần thấp: knife chỉ −10.77u gross, phần lớn
  giá trị phải đến từ né fills xấu 2022/2024 (−8.5u/lật âm theo NICHE phần A).
- **Thí nghiệm quyết định rẻ nhất**: offline oracle — fit LGBM walk-forward trên fills, đo PnL
  thật của decile dự-đoán-xấu ≥2022 và oracle ceiling của cancel-worst-decile.
  Kill bar: ceiling ≥2022 < +3u (so noise ±2u) ⇒ đóng. ~1 ngày.

### P3 — Refetch intraday (data-engineering option)
- **Cơ chế**: refetch OHLCV intraday (1m/5m hoặc tối thiểu 15m/1h) cho universe + VN30F1M/F2M,
  schema có cột time (DERIV_EXIT_SCREEN xác nhận data cũ collapse 1 bar/ngày do PK thiếu time —
  lỗi data-integrity, không phải nợ nghiên cứu). Sau đó: exit head 2024+ từ tín hiệu intraday
  aggregate, intraday breadth, vi-cấu-trúc quanh fill.
- **Khác nguồn alpha**: án "exit bão hòa" tuyên rõ điều kiện *"với daily price/volume"* — đây là
  cách duy nhất đổi tiền đề của án thay vì lách án. NICHE_LOSS_MAP nêu đích danh regime-signal
  từ VN30F1M intraday là "trục duy nhất chưa thử".
- **Vì sao án cũ không áp**: mọi án exit/derivatives đều xử trên data hiện có; intraday chưa
  từng tồn tại để bị xử.
- **Chi phí xây**: data-eng thuần (API vnstock/SSI, độ sâu lịch sử KHÔNG chắc — rủi ro lớn nhất:
  vendor chỉ cho vài năm gần → không đủ so sánh 2020-26 công bằng). Không đụng engine.
- **Rủi ro chính**: lịch sử ngắn ⇒ chỉ dùng được cho lát ≥202X, khó so composite full-period;
  chất lượng tick VN; công sức pipeline mới.
- **Thí nghiệm quyết định rẻ nhất**: probe API — fetch thử 1 mã + VN30F1M, xác định ĐỘ SÂU LỊCH
  SỬ tối đa và schema. Nếu < 4 năm ⇒ hạ ưu tiên (không đủ chuẩn ≥2022 + train). ~nửa ngày.

### P4 — Sequence/deep model trên OHLCV thô (TCN/transformer nhỏ, multi-task)
- **Cơ chế**: encoder chuỗi 60-120 bar OHLCV chuẩn hóa (pooled universe), multi-task head dự báo
  quantile phân phối fwd-return / first-passage prob. Dùng score làm nguồn entry THAY THẾ head
  LGBM (không phải gate thêm), execution giữ pullback-limit.
- **Khác nguồn alpha**: học biểu diễn thay 58 feature tay — điều kiện thắng duy nhất là tồn tại
  cấu trúc phi tuyến/chuỗi mà feature tay không mã hóa.
- **Vì sao án cũ không áp (một phần)**: án GRU 185.0 xử GRU trên label+feature hiện tại; đây là
  raw-window + label khác + nhiệm vụ khác. NHƯNG OHLCV_VIRGIN_MAP (23/25 ứng viên chết sau trực
  giao hóa, kênh sống pv_corr chỉ IC 0.041 và vẫn không đổi được quyết định) là bằng chứng trần
  thông tin daily OHLCV rất thấp — prior yếu, compute rẻ không đổi được prior.
- **Chi phí xây**: train code mới (PyTorch, ~vài ngày) nhưng thí nghiệm quyết định chỉ là số đo
  offline, không đụng engine.
- **Rủi ro chính**: đào lại đúng momentum/mean-reversion mà LGBM đã có (resid-IC ≈ 0); overfit
  regime 2020-21.
- **Thí nghiệm quyết định rẻ nhất**: train 1 TCN pooled, đo **resid-IC** (trực giao hóa với score
  champion) theo chuẩn null-band 5σ của OHLCV_VIRGIN_MAP, per-fold ≥2022. Kill bar: không vượt
  chuẩn pv_corr ⇒ đóng ngay, không sim engine. ~2-3 ngày compute.

### P5 — Data mới: foreign flow / fundamental (option data-eng thứ hai)
- **Cơ chế**: fetch mua/bán ròng khối ngoại daily per-symbol (+ fundamental quý) — kênh thông
  tin NGOÀI price/volume, thoát mọi án OHLCV-saturation. Dùng làm feature cho P1 (ranking) hoặc
  screen resid-IC độc lập.
- **Vì sao án cũ không áp**: chưa có nguồn ngoài-OHLCV nào từng được screen (trừ derivatives
  daily đã chết). Cùng universe/cost ⇒ công bằng leaderboard.
- **Chi phí/rủi ro**: API sẵn (vnstock có foreign flow); rủi ro độ sâu lịch sử + survivorship.
- **Thí nghiệm rẻ nhất**: fetch flow cho universe, resid-IC screen theo chuẩn 5σ null-band.
  ~1 ngày sau khi có data.

### Loại có lý do (không đưa vào bảng xếp hạng)
- **Meta-labeling / imitation-veto trên trade champion**: veto = gate trừ-bớt ⇒ đâm thẳng
  SELECTION WALL (mọi gate cắt lãi, đã đo nhiều lần). Phần "học từ trade champion" chỉ sống nếu
  đầu ra là *thay thế* lệnh (= ranking, đã gộp vào P1) hoặc *quản lý lệnh treo* (= P2).
- **Optimal-execution-path / exit family mới trên daily**: án exit-saturation là direct hit —
  mọi label exit mới trên daily OHLCV đều đi qua đúng surface đã 0/60+. Chỉ sống lại dưới P3.
- **Ensemble-of-families / chọn lệnh theo confidence tương đối**: cấu trúc y hệt regime-switching
  (trần ≥2022 +1.78u) + conviction repricing (chết). Chỉ xét lại SAU KHI P1/P4 sinh ra một model
  khác họ thật sự sống — hiện chưa có vế thứ hai để ensemble. Defer, không phải option độc lập.

---

## 3. Chấm điểm & xếp hạng

Thang: P(alpha mới thật) / Chi phí đến thí-nghiệm-quyết-định đầu tiên / Công-bằng-leaderboard.

| # | Phương án | P(alpha mới) | Chi phí E1 | Công bằng LB | Ghi chú quyết định |
|---|---|---|---|---|---|
| 1 | **P1 Cross-sectional ranking lõi** | TB-cao (trục chưa có án; demeaned label = thông tin champion chưa dùng) | Thấp (~1-2 ngày, offline) | Cao (adapter vào engine hiện tại, cùng universe/cost) | Khởi động ngay |
| 2 | **P2 Fill-conditional label / cancel-policy lệnh treo** | TB (khoảng trống đo được: score phẳng trên sổ treo; nhưng trần bị chặn bởi bất đối xứng big_win) | Rất thấp (~1 ngày, artefact có sẵn) | Hoàn hảo (cùng engine + 1 knob) | Khởi động ngay |
| 3 | P3 Refetch intraday | Cao NẾU data đủ sâu (trục "chưa thử" duy nhất được nêu tên) | Probe rẻ (nửa ngày) nhưng pipeline đầy đủ đắt | TB (lịch sử ngắn khó so full-period) | Probe API ngay ở background; quyết định sau khi biết độ sâu |
| 4 | P5 Foreign flow/fundamental | TB (kênh ngoài-OHLCV, chưa screen bao giờ) | Thấp-TB | Cao | Làm chung batch data-eng với P3; feed P1 |
| 5 | P4 Sequence/deep raw OHLCV | Thấp (OHLCV_VIRGIN_MAP ép prior xuống) | TB (2-3 ngày, compute free) | Cao | 1 phát resid-IC với kill bar cứng, không sa lầy |
| — | Ensemble-of-families | phụ thuộc P1/P4 | — | — | Defer |
| — | Meta-label veto; exit family mới trên daily | ~0 (re-skin án cũ) | — | — | Loại |

---

## 4. Khuyến nghị: khởi động P1 + P2 (P3 probe chạy nền)

### P1 — Lộ trình
1. **E1 Screen offline (kill/go, 1-2 ngày)**: dataset weekly, label fwd21/fwd42 demeaned-by-date;
   LGBM walk-forward (YearSplitter, cùng universe 488). Đo: rank-IC, spread top-20 net cost,
   lát ≥2022/≥2024, multi-seed nhanh, resid-corr với entry score champion, **fill-rate giả lập
   limit 4.5%/40 trên cohort top-K**. Kill: spread ≥2022 ≤0 ∨ IC<0.03 ∨ fill<~50%.
2. **E2 Signal-adapter (1 tuần)**: membership top-K/hysteresis → cột signal; clone template
   append-only, strategy mới `xsec_rank_topk` gated; giữ pullback 4.5%/40 + hard-stop washout;
   golden parity cho đường cũ. Chạy leaderboard multi-seed 42/7/99/555 + chuẩn ≥2022.
   Bar giai đoạn: composite ≥ 650 (vượt trần pure-ML 626 = não mới tự đứng được) trước khi tối ưu.
3. **E3 Lai ghép nếu E2 sống**: rank làm bộ CHỌN ỨNG VIÊN (nguồn tín hiệu thay thế — additive,
   không phải gate), timing head champion làm trigger đặt limit. Vòng công tố đầy đủ §0.4.

### P2 — Lộ trình
1. **E1 Oracle offline (kill/go, ~1 ngày)**: dataset 11,395 limit đã đặt từ artefact
   serving_blindspot; label first-passage từ GIÁ FILL; features tại thời điểm tiệm cận limit.
   LGBM walk-forward. Đo: PnL thật decile dự-đoán-xấu per-year, oracle ceiling cancel-worst-decile,
   riêng cohort knife + 2022/2024. Kill: ceiling ≥2022 < +3u.
2. **E2 Knob engine (nếu E1 sống, ~3 ngày)**: `pending_cancel_score` gated default-off + parity
   tuyệt đối (mẫu giveback_guard); sweep ngưỡng precision-first (ưu tiên không đụng big_win —
   ràng buộc: recall trên 717 big_win ≥ 99%); multi-seed + ≥2022.
3. **E3**: nếu cả P1-E2 và P2-E2 sống → thử chồng (cancel-policy phục vụ luôn sổ lệnh của tuyến rank).

### P3 probe (chạy nền, nửa ngày)
Fetch thử intraday 1 mã VN30 + VN30F1M qua API hiện dùng, xác định độ sâu lịch sử + schema có
cột time. ≥4 năm ⇒ mở dự án data-eng đầy đủ (mở khóa lại toàn bộ trục exit đang bị án treo);
< 4 năm ⇒ ghi nhận, hạ ưu tiên.

### Nguyên tắc chung cho mọi tuyến (thừa kế protocol)
Một thí nghiệm = một clone template append-only; knob gated default-off + golden parity; mọi run
đăng ký leaderboard kể cả xấu; chuẩn regime ≥2022 bắt buộc; multi-seed trước khi tin bất kỳ delta
nào < ±2u; không đụng pullback 4.5% và không bơm label/feature mới vào chuỗi z-threshold cũ.
