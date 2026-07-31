# Nâng cấp Engine: chạy Universe Động point-in-time trong 1 run

> Mục tiêu: model universe-động (top-N thanh khoản point-in-time, khác symbols mỗi fold)
> chạy trong **1 run walk-forward liền mạch** — thay vì 6 run per-year ghép offline.
> Ngày lập: 2026-07-28. **Bổ sung 2026-07-29: §7 — 4 yêu cầu phía ĐĂNG KÝ** (bắt buộc đọc
> cùng 5 bước engine; đăng ký overlay xem thêm
> [refactor/PORTFOLIO_REGISTRATION_PIPELINE.md](refactor/PORTFOLIO_REGISTRATION_PIPELINE.md)).
> **Sửa 2026-07-29 (review đối chiếu code)**: Bước 3 (splitter nhận dict đã resolve),
> Bước 4 (đảo — default full-panel), thêm Bước 4b (recombine stateful), thu gọn Bước 5.
>
> 🆕 **Cập nhật 2026-07-31 — trạng thái & số (đồng bộ `refactor/ENGINE_UPGRADE_AND_LEGACY_RESTRUCTURE.md` §14).**
> Phần **engine (bước 1–4b) ĐÃ thực thi** (`universe_resolver.py`, `splitter.py`, audit 3185 stateful → re-baseline);
> đây là **nền tảng của bundle động dyn300/dyn900** (main doc §9.9 "hình mẫu chuẩn"). NHƯNG **mọi con số ở đây
> (205% tham chiếu, 195.3% 1-run) có TRƯỚC resync 901-mã** — chính caveat "store NAV 193 mã" ở §5. Số hiện hành
> phải chấm lại trên **market_panel full-901** (§13.3) + **CA tại nguồn sieutinhieu** (§9.7/§13.7). §7 (đăng ký)
> phải khớp **thước mới `stock_ml.portfolio`** (không nh_nav2 — Q4/§14.4) và **`config_hash` từ `resolved.json`**
> (§13.2). ⇒ Coi số dưới đây là **đối chiếu độ lớn lịch sử**, không phải số official.

---

## 1. Vì sao cần

**Hiện trạng**: `run_template` nhận **1 danh sách symbols cố định** cho cả run.
- Universe động → phải chạy 6 run riêng (`_dyn900_2020..2025`) rồi ghép trades offline + overlay portfolio.
- Hệ quả: KHÔNG map vào 1 champion row leaderboard; `score_nav_leaderboard.py` chấm per-fold không phản ánh model ghép.

**Sau nâng cấp**: 1 run 2020-2025, mỗi fold tự chọn universe point-in-time → đăng ký leaderboard chuẩn như champion 61-mã.

**Bằng chứng cần point-in-time** (không phải tĩnh): static-900 (universe cố định ADV toàn kỳ) CAGR 301% vs dynamic-900 (point-in-time) 205% — chênh 96pp là **look-ahead bias** (static biết trước mã nào thanh khoản tương lai). Nâng cấp phải giữ point-in-time để số hợp lệ.

---

## 2. Kiến trúc hiện tại (điểm chạm)

```
run_template.py
  → get_pipeline_symbols()            # resolve 1 danh sách symbols cố định
  → run_experiment(cfg, symbols)
       → loader.load_many(requested)  # load OHLCV cho symbols cố định  [experiment.py:3206]
       → build_feature_frame(...)      # feature per-symbol + xsec/breadth đọc thẳng duck
       → splitter.split(df)            # walk-forward, mask THEO THỜI GIAN  [splitter.py:135-144]
            for w in windows():
              train_mask = time in [train_start, train_end)   # KHÔNG lọc universe
              test_mask  = time in [test_start, test_end)
       → predict per fold → concat signal_frames              [experiment.py:3349]
       → recombine_signals(signals_all)  # gate xsec + op stateful, 1 LẦN trên frame ghép [experiment.py:3356]
```

**3 chỗ cần sửa universe-aware**: (a) resolve universe per-fold, (b) union-load + mask train/test theo universe fold, (c) universe vào cache-key (§7.2).
**1 chỗ phải kiểm chứng, không sửa code**: `recombine_signals` (bước 4b).
**KHÔNG sửa mặc định**: cross-sectional rank `_load_xsec_features` — default full-panel (bước 4, §7.3).

---

## 3. Kế hoạch sửa (5 bước)

### Bước 1 — Config: `universe_policy` thay symbols cố định

**File**: `stock_ml/src/pipeline/experiment.py` (`ExperimentConfig`, dòng ~39-372) + `strategy_templates.universe_slug`.

Thêm trường:
```python
universe_policy: dict | None = None
# vd: {"mode": "dynamic_topn", "n": 900, "metric": "adv",
#      "lookback": "prior_year", "min_sessions": 100,
#      "exclude": ["index", "derivative"]}
```
- `None` → giữ hành vi cũ (symbols cố định), **backward-compatible**.
- Khi set → engine resolve universe per-fold (bước 2).
- Persistence DB-first (policy có danh tính trong `universe_sets` + snapshot per-fold): theo §7.1,
  không thiết kế lại ở đây.
- **Bẫy**: có 2 bản `get_pipeline_symbols` (`src/utils/config_loader.py:246` và
  `src/config_loader.py:227`) — xác định bản hệ sống đang dùng rồi sửa đúng bản đó.

### Bước 2 — Resolver universe point-in-time

**File mới**: `stock_ml/src/data/universe_resolver.py`

```python
def resolve_universes(policy: dict, test_years: list[int], duck: str) -> dict[int, list[str]]:
    """top-N tradable equities by prior-year ADV, per test-year. Causal (chỉ dùng data < test_year).
    Loại index + derivative. KHÔNG ép champion (fully dynamic).
    Resolve 1 LẦN cho mọi fold — dict dùng chung cho union-load + splitter (bước 3) + snapshot (§7.1)."""
    # với dynamic_topn + lookback=prior_year:
    #   ADV = avg(volume*close) trong năm (test_year - 1)     <-- ⛔ SAI, xem cảnh báo dưới
    #   filter: count(*) >= min_sessions, not is_nonstock(symbol)   <-- ⛔ SAI
    #   return top-N by ADV (SAU khi loại nonstock)
```
Port logic từ `_dynuniverse_clean.py::uni()` (causal — phần đó vẫn đúng). File đó đang
**untracked**, dễ mất khi dọn rác. Nonstock = VNINDEX/HNX30/VN30/HNXINDEX/UPINDEX/VNXALL +
regex `F\d+M$` + prefix VN30F/VN100F.

> ## ⛔ HAI VẾ CỦA CÔNG THỨC TRÊN LÀ SAI — đo 2026-07-31
>
> Đừng "chép nguyên văn" nữa. Cài đặt hiện tại (`stock_ml/src/data/universe_resolver.py:113`) sai cả
> hai vế theo hợp đồng dữ liệu của nguồn:
>
> **1. `avg(volume * close)` ≠ giá trị giao dịch.** `ohlcv_data` back-adjust **giá** về hiện tại
> nhưng **không** back-adjust khối lượng ⇒ tích thấp hơn giá trị thật đúng bằng hệ số điều chỉnh
> luỹ kế, méo khác nhau tuỳ mã (mã pha loãng mạnh méo nhất — đúng nhóm cần lọc). Và **không tách
> được thoả thuận**: DVG `close×volume` 0.60 tỷ vs khớp lệnh thật 0.03 tỷ (thổi **20×**).
> ⇒ Dùng ADTV khớp lệnh (`basis=matched`) từ `market_flow_1d`.
>
> **2. `count(*)` ≠ số phiên giao dịch.** Bảng phát sinh dòng cho cả phiên mã **không khớp lệnh** —
> đo trên `market_data/market.duckdb`: **22,1%** bar có `volume=0`. Hệ quả lên `min_sessions=100`:
> **11-31% mã qua cổng mỗi năm là mã đã CHẾT** (2019: 255/811 = 31% · 2020: 247/836 = 30% ·
> 2024: 181/893 = 20%).
> ⇒ Dùng `sessions_traded`, không phải `sessions_counted`.
>
> **Hệ quả đã khép kín:** dyn900 có `n:900` là trần chưa bao giờ chạm (thực tế 775) ⇒ ràng buộc thật
> là `min_sessions` ⇒ mà nó đếm phiên ma ⇒ **"dyn900 penny thổi phồng CAGR" là lỗi định nghĩa
> universe, không phải hiện tượng mô hình.**
>
> **3. Bar ma còn bóp méo FEATURE, và top-N tự lọc — đo 2026-07-31.** 97,2% bar `volume=0` phẳng
> tuyệt đối ⇒ hạ thấp vol/ATR/biên độ; pipeline **không lọc** chúng (chỉ cắt đoạn phẳng đầu chuỗi).
> Nhưng đo ra thì **chỉ dyn900 bị**: bar ma chiếm 1,6% ở dyn300 vs **20,6%** ở dyn900, và mức bóp
> méo `vol20` median trên dyn300 = **1,000×** (không đáng kể).
> Đếm mã bẩn *bên trong* chính chính sách top-N: **top-300 dính 0-2 mã/năm (0,0-0,7%)**, còn
> **top-900 dính 73-98 mã/năm (9-11%)** — vì mã thứ 900 có ADTV **≈ 0,00-0,06 tỷ**.
> ⇒ Xếp hạng theo ADV **tự nó đã là bộ lọc**; không cần thêm sàn cho `n` hợp lý. Vấn đề là
> **`n:900` quá sâu** — phải hạ `n`, đặt sàn thật, hoặc cho tier đó nghỉ.
>
> **Nguồn đã có sẵn** `GET /symbols/universe` làm đúng chính sách này cho nhiều `as_of` một lần gọi
> (6 mốc/4 giây), point-in-time thật (trả được ADTV 2020 của ROS/FLC/ITA dù sau này huỷ niêm yết).
> ⇒ `resolve_universes` nên thành **wrapper mỏng quanh endpoint đó**, không phải bản cài đặt thứ hai.
>
> ⚠️ Sửa xong **universe đổi** (trùng 89-96% với bản đã bake, tăng dần theo năm) ⇒ kéo theo
> re-baseline toàn bộ. Chốt trước, đừng để xảy ra như tác dụng phụ.
> Chi tiết + số đo: `docs/refactor/ENGINE_UPGRADE_AND_LEGACY_RESTRUCTURE.md` §13.9.

Khác bản gốc 2 fix chủ đích:
- **Tiebreaker** `ORDER BY adv DESC, symbol` — bản gốc thiếu: ADV bằng nhau ở biên top-N có
  thể đổi thứ tự giữa các lần chạy, vi phạm yêu cầu determinism (§5).
- Parameterized query thay f-string interpolation.

### Bước 3 — Splitter universe-aware (nhận dict đã resolve, KHÔNG query DB)

**File**: `stock_ml/src/data/splitter.py` (`YearSplitter.split`, dòng 135-144) + `experiment.py:3206`.

Splitter hiện **pure** (df vào → mask ra) — giữ nguyên tính chất đó. Resolve universe 1 lần
ở experiment.py, truyền dict xuống:
```python
# experiment.py (trước load):
universe_by_year = resolve_universes(policy, [w.test_year for w in windows], duck)
requested = sorted(set().union(*universe_by_year.values()))   # union-load đủ bars mọi fold

# splitter.py:
def split(self, df, date_col="date", universe_by_year=None):
    for w in self.windows():
        train_mask = (dates >= w.train_start) & (dates < w.train_end)
        test_mask  = (dates >= w.test_start)  & (dates < w.test_end)
        if universe_by_year is not None:
            sym_in = df["symbol"].isin(universe_by_year[w.test_year])
            train_mask &= sym_in; test_mask &= sym_in   # CHỈ universe của fold
        yield w, df.loc[train_mask].copy(), df.loc[test_mask].copy()
```
(Bản nháp cũ cho splitter tự gọi resolver + nhận `duck` — bỏ: trộn DB-access vào splitter,
và resolve ở 2 chỗ dễ lệch nhau. Cũng giữ `.copy()` như code hiện tại.)

**Union-load an toàn cho train/predict** (đã xác minh khi review): feature build là per-symbol,
xsec/breadth đọc thẳng duck — không có bước nào TRƯỚC split phụ thuộc panel loaded, nên
train/test rows sau mask byte-giống per-year run cũ.

### Bước 4 — Cross-sectional rank: DEFAULT full-panel (đảo so với bản nháp cũ)

Bản nháp cũ đề xuất rank `_load_xsec_features` theo universe fold — **đảo lại** sau review:

1. Số tham chiếu 205% được sinh với `_load_xsec_features` đọc TOÀN duck panel (docstring
   `experiment.py:1681` "RANK vs ALL symbols"), bất kể symbols của run — đổi định nghĩa rank
   thì không thể "verify byte-close vs `_dyn900` cũ" như chính bảng test §4 đòi hỏi.
2. Feature trả lời "mã này mạnh thứ mấy so với TOÀN thị trường" — không phụ thuộc tập được
   phép mua. Rank theo fold làm cùng mã cùng ngày có feature khác nhau tùy N → input model
   dịch chuyển mỗi lần đổi universe; serving phải tái tạo đúng universe từng ngày mới khớp.
3. Full-panel rank vẫn causal (chỉ dùng data CÙNG NGÀY của mã khác) và đã kiểm chứng nhất
   quán train↔serving (dyn400 cố ý chọn full-panel, serving sieutinhieu khớp).

Theo §7.3: chính sách panel thành field `xsec_panel: full | fold_universe`, **default `full`**
(không sửa code rank). Chỉ `full` mới so được với số tham chiếu; chọn `fold_universe` = model
MỚI — phải re-baseline và sửa cache key (`_XSEC_CACHE` key chỉ có metrics, `_BREADTH_CACHE`
chỉ (metric, ma_win) — không chứa universe lẫn duck, §7.2).

**Giữ nguyên, KHÔNG thuộc bước này** (bản nháp cũ liệt kê nhầm): `_load_market_breadth` (:1641
— cố ý full-universe, chỉ báo sức khỏe THỊ TRƯỜNG) và `_load_regime_index` (:1625 — 1 series
chỉ số VN30F1M, không liên quan universe). Conviction `cs5_ma50` (Stage-2) nằm ở TẦNG
PORTFOLIO (`stock_ml/portfolio`), không phải experiment.py — pin panel ở đó theo §7.3 + doc
PIPELINE, ngoài phạm vi engine.

### Bước 4b — Kiểm chứng `recombine_signals` (điểm chạm bản nháp cũ bỏ sót)

`recombine_signals` chạy **1 lần trên frame ghép mọi fold** (`experiment.py:3349-3356`):
- Gate cross-sectional per-date (`crashx` :1615, `csrank` :2361, `entry_xs_mom_pct` :2384,
  `breadthq` :1599): mỗi date chỉ thuộc đúng 1 fold → cross-section per-date tự khớp universe
  fold đó — tương thích union-load, **không sửa** (đã soi code khi review).
- Op **stateful xuyên fold**: `breadthq` rolling-252-percentile, `_causal_zscore_by_symbol`
  (norm zscore, window 252) trên lịch sử score. 6-run ghép: lịch sử reset mỗi năm;
  1-run liền mạch: lịch sử liên tục → giá trị KHÁC nhau giữa 2 cách chạy.
- **Việc cần làm**: soát engine_config template 3185. Không dùng op stateful → 1-run tương
  đương tuyệt đối 6-run ghép; có dùng → chấp nhận sai khác + re-baseline (ghi chú vào test bước 5).

**KẾT QUẢ AUDIT (2026-07-29, `_audit_3185_stateful.py`)**: template 3185 (`x2_struct_to`,
strategy `regression_dual_ml_recombine_decoupled`) **CÓ op stateful** → kết luận = **re-baseline**:
- 4 ensemble head `z_threshold` (0.9/0.7/0.7/0.7) không set `*_norm` → default `zscore` =
  `_causal_zscore_by_symbol` window 252/min 60 trên LỊCH SỬ SCORE của signals frame.
- `entry_market_mode`/`exit_market_drop_mode`/`overext_bull_mode` đều `zscore`
  (lookback 60/60/40) — z trên lịch sử market trong frame.
- `entry_gate=upleg_abovema20` — leg-state + MA20 từ close trong frame (warmup đầu năm khác).
- KHÔNG stateful: `exit_force_gate_lowbreadth`/`entry_ensemble3.breadth_features` đọc duck
  (full history, giống nhau ở cả 2 cách chạy).
- Cấu trúc sai khác còn sâu hơn warmup: run per-year cũ có test window [t → data-end] rồi
  lọc yr==t offline (z-history = score CÙNG model từ đầu năm t), còn 1-run mỗi fold chỉ
  [t, t+1) nên z tại năm t nhìn ngược vào score của MODEL FOLD TRƯỚC → chuỗi z trộn model.
  ⇒ số 1-run là ĐỊNH NGHĨA MỚI (chuẩn engine-native, giống champion 61-mã), so với 205%
  chỉ để đối chiếu độ lớn, không kỳ vọng byte-match.

### Bước 5 — Score NAV (thu gọn — overlay đã có pipeline riêng)

Bản nháp cũ đề xuất thêm chế độ chấm overlay vào `score_nav_leaderboard.py` — **đã bị vượt**:
cột `cagr_overlay/maxdd_overlay/overlay_k/overlay_note` đã có trong `leaderboard_nav` + board
(commit 596aafb0), tầng đăng ký overlay do doc PIPELINE own (B3 `register_overlay.py`). Còn lại:
- Run động 1-liền-mạch chấm `cagr_standard` (K25 fair) qua `score_nav_leaderboard.py` như mọi
  run — **không sửa gì**.
- `cagr_overlay` của run động đi qua pipeline đăng ký overlay chính thức (PIPELINE B3-B5) —
  không thiết kế lại ở doc này.

**KẾT QUẢ RE-BASELINE (2026-07-29, `_dyn900_onerun.py` + `_dyn900_onerun_overlay.py`)**:
1-run N=900 fold 2020-2025 (engine mới, template 3185, seed 42, universe point-in-time
807→889 mã/năm, audit PASS, chạy **4.1 phút** vs nhiều giờ của 6-run ghép):
- Signal quality 2020-25: **3.26% / wr 55.2%** (30 300 trades) ≈ tham chiếu ghép 3.07% ✓
- Overlay K10 + dl63 0.13 (CÙNG config tham chiếu): **CAGR 195.3% / DD −29.4% / NAV ×1157**
  vs ghép-offline 205.6% / −17.6% → CAGR đúng độ lớn (−10.3pp, giải thích bằng 4b:
  z-history liền mạch + trộn-model); **DD sâu hơn đáng kể** (−29.4 vs −17.6) do trade
  composition/timing đổi — nhất quán pattern đã biết "universe rộng DD sâu" (penny/beta
  mid-small). Số 1-run là baseline engine-native chính thức từ đây.

---

## 4. Thứ tự thực hiện + test

| Thứ tự | Bước | Test bắt buộc |
|---|---|---|
| 1 | Bước 2 (resolver) — độc lập, dễ test | Unit: `resolve_universes(..., [2024], ...)` = top-N ADV 2023, causal, không index/deriv; chạy 2 lần = nhau (kể cả tie ADV) |
| 2 | Bước 1 (config) — backward-compat | `universe_policy=None` → snapshot 61-mã byte-identical |
| 3 | Bước 3 (splitter) | Snapshot: run universe cố định qua path mới = kết quả cũ byte-identical |
| 4 | Bước 4b (recombine audit) | **ĐÃ XONG 2026-07-29**: 3185 CÓ op stateful (z ensembles + market-z + upleg_abovema20) → kết luận **re-baseline** |
| 5 | Bước 5 (score) | Chấm run động so 205% chỉ để đối chiếu ĐỘ LỚN (không byte-match — xem kết quả audit 4b); số 1-run là chuẩn engine-native mới |

**Nguyên tắc**: mỗi bước có **snapshot guard** — `universe_policy=None` phải cho kết quả byte-identical với engine hiện tại (không phá champion 61-mã / dl63size). Xem cách làm ở `tests/test_baseline_snapshot.py`.

---

## 5. Rủi ro & lưu ý

- **Rủi ro lớn nhất đã chuyển chỗ** (sau khi Bước 4 thành default-không-sửa): (a) Bước 4b —
  tương đương 1-run vs 6-run ghép phụ thuộc op stateful trong config; (b) cache-key §7.2
  (đã từng dính bug ghi đè cache thật). Kết quả tham chiếu để so: `_dyn900` signal 3.07%,
  CAGR overlay 205%.
- **Data union**: load union mọi fold-universe có thể lớn (900+ mã × 8 năm). Kiểm memory (dataset static-900 = 2.66M bars OK).
- **Backward-compat tuyệt đối**: `universe_policy=None` = hành vi cũ. Champion 61-mã / dl63size KHÔNG được đổi 1 trade.
- **Determinism**: giữ (LGBM `deterministic:True`); resolver deterministic nhờ tiebreaker
  `ORDER BY adv DESC, symbol` (bước 2).
- **Ước lượng**: ~1-1.5 ngày engine (bước 1-3 code ~0.5-1 ngày; bước 4b audit config ~0.5 ngày;
  bước 4/5 không còn code phải viết). §7 riêng ~0.5-1 ngày.
- **[2026-07-29] Caveat store NAV 193 mã (phát hiện audit tiền-production)**: thước official
  (`stock_ml.portfolio` + NAV store serving `ohlcv.db`) chỉ có giá 193 mã;
  `portfolio/api.py:91-92` bỏ LẶNG LẼ leg thiếu giá → mọi số `cagr_overlay` của run dyn
  hiện = "signals dyn ∩ 193 mã thanh khoản" (dyn300 fill 192/439 mã, dyn900 193/873,
  61a2hy 77/84). Không phải leak nhưng claim độ phủ sai ở tầng khớp lệnh. Fix + số chấm
  lại trên store full 901 mã (resync sieutinhieu): xem
  `Desktop/stock-serving/DEPLOY_DYN_TIERS.md` §3-B1, §5.
- **Audit tiền-production 2026-07-29**: causality universe chứng minh bằng truncation vật lý
  18/18 PASS; review độc lập 17 mục không leak thật; chi tiết trong memory
  `preprod-audit-2026-07-29-clean`.

---

## 6. Tham chiếu (đã kiểm chứng)

- Logic universe clean causal: `_dynuniverse_clean.py::uni()` (root repo, **untracked** —
  chép nguyên văn khi port, bước 2).
- Kết quả dynamic-900 mục tiêu tái tạo: CAGR 205.6% / DD −17.6% (K10, full-price, NAV≤1).
- Snapshot pattern: `stock_ml/tests/test_baseline_snapshot.py`.
- Cross-sectional hiện tại: `experiment.py:1681` `_load_xsec_features` (docstring: "RANK vs
  ALL symbols") — cơ sở default full-panel (bước 4).
- Recombine 1-lần-trên-frame-ghép: `experiment.py:3349-3356` — cơ sở bước 4b.
- Tầng overlay/đăng ký: `refactor/PORTFOLIO_REGISTRATION_PIPELINE.md` (B3-B5).

---

## 7. Bổ sung 2026-07-29 — 4 yêu cầu phía ĐĂNG KÝ (để lên leaderboard chuẩn)

5 bước §3 mới lo phía ENGINE. Để một run universe-động đăng ký chính thức (leaderboard +
tab danh mục + tái lập được), cần thêm:

### 7.1 Universe = POLICY có danh tính + snapshot vật chất hóa từng fold
- `strategy_templates.universe_slug` trỏ vào một POLICY (row trong `universe_sets` với
  `selector` JSON = `{mode, n, metric, lookback, min_sessions, exclude}`), KHÔNG phải danh
  sách mã cứng.
- Mỗi run: sau khi resolver chạy, **materialize danh sách đã resolve của TỪNG fold** vào
  `universe_versions`/`universe_symbols` (key theo `(slug, fold_year)`) + hash danh sách.
  Lý do: data sau này được vá CA/refill → ADV đổi → resolve lại có thể lệch mã; snapshot làm
  run tái lập byte-stable và audit được "fold 2023 gồm đúng những mã nào". Serving đọc CÙNG
  snapshot (hoặc resolve live với policy y hệt — causal nên hợp lệ, nhưng phải log hash để đối chiếu).

### 7.2 Universe PHẢI vào cache-key (bug đã từng dính — bắt buộc)
- Sự cố thật 2026-07-26: universe không nằm trong cache-key → run PIT **ghi đè cache của base
  61-mã** (phải khôi phục). Feature/prediction cache key = hash(config + universe policy +
  resolved fold lists). Test guard: chạy run 61-mã rồi run dyn200 rồi CHẠY LẠI run 61-mã —
  kết quả byte-identical lần đầu.

### 7.3 Chốt tường minh chính sách panel cross-sectional (2 tầng, train↔serving khớp)
- Có 2 tầng rank chịu ảnh hưởng panel: **Stage-1** (`_load_xsec_features` — RS ranks vào model)
  và **Stage-2** (conviction `cs5_ma50` trong `stock_ml/portfolio` — `ctx.market_frame`).
- Hai lựa chọn hợp lệ: rank trên **universe của fold** (model MỚI — phải re-baseline, xem
  bước 4) hay trên **full-panel** (**default** — khớp số tham chiếu 205% + serving; dyn400
  trước đây cố ý chọn full-488 và serving khớp). Không hard-code —
  thành field template `xsec_panel: fold_universe | full`, ghi vào cả
  `overlay_config_hash` (doc PIPELINE), và train↔serving BẮT BUỘC cùng giá trị.
- Đã đo: panel là siêu-tham-số (CAGR dịch hàng chục pp khi nới panel); SKIP gate causal
  panel-adaptive trong overlay được thiết kế để hấp thụ — nhưng chỉ khi policy nhất quán.

### 7.4 Hiển thị & so sánh trên leaderboard
- **ĐÍNH CHÍNH fairness (user, 2026-07-29)**: phán quyết cũ "mở universe = ăn gian" chỉ đúng
  thời xếp hạng theo tổng-%-cộng-dồn-lệnh. Bảng nay xếp theo **CAGR trên NAV, tổng vốn ≤ 1,
  K-slot** → universe rộng không cộng return máy móc, chỉ mở tập lựa chọn cạnh tranh cùng vốn
  = so sánh CÔNG BẰNG mọi cỡ universe trên cột NAV (`cagr_overlay`/`cagr_nav`). KHÔNG cần
  phân lớp xếp hạng.
- Còn lại 2 ghi chú: (a) cột linear cũ (`pnl_pct`/`total_pnl`/composite) vẫn nhạy số-mã —
  đừng dùng so chéo universe; (b) hiện `universe_slug`/`n_symbols` trên bảng làm metadata
  DIỄN GIẢI risk-profile (universe rộng → beta mid/small, DD sâu hơn — đọc kèm Calmar/DD),
  không phải rào so sánh.

**Ước lượng thêm cho §7**: ~0.5-1 ngày (7.1 nửa ngày; 7.2 vài giờ nhưng test cẩn thận;
7.3 là quyết định + field; 7.4 chỉ hiển thị). Thứ tự khuyến nghị: làm SAU B1-B5 của
PORTFOLIO_REGISTRATION_PIPELINE để `register_overlay.py` có sẵn chỗ ghi config hash.
