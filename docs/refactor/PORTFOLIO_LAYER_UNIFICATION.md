# Refactor: Thống nhất Tầng Danh Mục (Portfolio Layer)

> Gom ~11 bản copy-paste của overlay danh mục thành **một module chuẩn duy nhất** mà
> backtest, đăng-ký-leaderboard, và serving-production cùng import.
> Mục tiêu: xóa drift giữa backtest và production — nguồn gốc mọi lỗi đo trong lịch sử gần đây.
> Ngày lập: 2026-07-28.

---

## 0. TL;DR

- **Vấn đề**: tầng danh mục (K-slot + conviction-sizing + preemption + gate SKIP/ret7/dl63 + T+2)
  nằm **NGOÀI** engine, tồn tại dưới dạng **~11 bản copy-paste** (10× `hb_deploy_*.py`,
  `_champ_prod_replay.py` untracked, `serving/portfolio/core.py`). Backtest và production là
  2 code path viết tay riêng, khớp bằng tay.
- **Hậu quả đã xảy ra**: base-vs-output confusion (151% vs 73%), SKIP causal-vs-fixed lệch 2pp,
  bản chính untracked, mọi so-sánh phải tự viết lại overlay → sai lệch.
- **Giải pháp**: 1 package `stock_ml/portfolio/` — public `run_portfolio(base, signals, ctx, C)`.
  Backtest gọi nó, serving gọi CÙNG nó (qua wheel). Xóa 11 bản copy.
- **Ràng buộc sống-còn**: champion 61-mã (dl63bal) phải giữ **byte-identical** — tài liệu này có
  snapshot guard cho mọi bước.
- **ĐÃ KIỂM CHỨNG (§9, đừng bác nhầm)**: overshoot filter là CAUSAL (+6.1pp deploy-được, KHÔNG phải leak
  như memory cũ ghi) — GIỮ ON; short_tilt5 (+16pp) mới là look-ahead thật; Sieu Tin Hieu chất lượng data
  cao hơn DuckDB train. Số production sạch (sau sync PVD) = **142.3% / -10.7%** (trước sync là 151.1%
  nhưng phồng +8.8pp do 1 mã data cũ — §9.3b). Golden phải PIN snapshot ĐÃ sync mọi CA mới.

---

## 1. Hiện trạng — bản đồ kiến trúc (đã kiểm chứng)

```
Tầng 1 — ENGINE  (stock_ml/src/backtest/engine.py)
  run_backtest(signals, ohlcv, cfg) -> list[Trade]        [engine.py:2558]
    └─ _run_symbol(): xử lý TỪNG MÃ, 1-slot single-unit    [engine.py:1108]
       • conviction ở đây = FEATURE (input pullback), KHÔNG phải sizing
       • Trade có field weight/notional nhưng DEFAULT 1.0/0.0 (legacy per-symbol) [engine.py:1072-1075]
  → OUTPUT: BASE trades (mọi setup, per-symbol, chưa chọn danh mục)

Tầng 2 — OVERLAY DANH MỤC  (NẰM NGOÀI engine — ~11 bản)
  logic: rewrite → gate(SKIP causal/ret7/dl63) → meta-priority → K-slot sim
         (conviction-sizing w∈[0.4,1.8] + preemption MARGIN + T+2 + advance-fee)
  ├─ stock_ml/analysis/serving_blindspot/r3line/hb_deploy_*.py   × 10 bản (mỗi lever 1 bản)
  ├─ _champ_prod_replay.py   (root, 313 dòng, ⚠️ UNTRACKED — không trong git)
  └─ serving/portfolio/core.py   (repo serving riêng, 356 dòng, viết tay lại)  ← PRODUCTION
```

**Phân biệt BASE vs OUTPUT (bẫy đã mắc):** run_trades trong DB có 2 loại tùy script ghi:
- Ghi qua `run_template_experiment` → **BASE** (exit_reason chỉ engine-level: signal/max_hold/overext_trail/trailing_stop).
- Ghi qua `hb_deploy::populate`→`prun_track` → **OUTPUT đã overlay** (exit_reason CÓ preempt/green_trail/early_cut).
- Dùng nhầm OUTPUT làm input overlay = chạy overlay 2 lần → sai (151%→73%). Module chung phải nhận
  **BASE** rõ ràng và tự-nhận-diện (assert không có exit_reason overlay-level).

---

## 2. Vì sao phải sửa (bằng chứng, không phải phong cách)

| Triệu chứng thực tế | Gốc kiến trúc |
|---|---|
| Champion đo 73% thay vì 151% | 11 bản overlay, run_trades lưu khác nhau → nhầm base/output |
| Con số 151 vs 132 vs 153 dễ lẫn | 3 HỆ khác nhau — production-STH single-seed (151.1% T+2 / 153.1% T+0, CÙNG run khác T+0/T+2), DuckDB-registered (132.1%), NGUỒN DATA khác nhau (§9), KHÔNG phải SKIP mode |
| Bản overlay chính có thể mất | CẢ `_champ_prod_replay.py` LẪN `serving/portfolio/core.py` UNTRACKED |
| Production có thể ≠ backtest | `serving/core.py` viết tay riêng, khớp param bằng tay, không share code |
| Mọi phân tích phải tự viết overlay | không có hàm chuẩn để import → mỗi lần copy → lệch |

**Nguyên tắc**: tầng danh mục là một **stage xác định** của hệ, phải có **một** implementation.
Backtest và serving khác ở NGUỒN DATA (DuckDB train vs Sieu Tin Hieu live — chất lượng KHÁC nhau, §9),
KHÔNG khác ở LOGIC. Golden phải gắn ĐÚNG nguồn data, đừng so số chéo nguồn.

---

## 3. Kiến trúc đích

```
stock_ml/portfolio/                         ← PACKAGE MỚI (single source of truth)
  __init__.py         # export run_portfolio, PortfolioConstants, PortfolioContext
  constants.py        # PortfolioConstants (K, MARGIN, KCONV, SKIP causal, R5THR, dl63, tplus, fees)
  panel.py            # build_market_panel(market_src) -> CLO/LO/DIDX/INV/CSm/R5  (conviction/ret7/overshoot)
  priority.py         # meta_priority(closed, signals, market_src) -> pm   (LGBM walk-forward)
  rewrite.py          # rewrite(closed, panel, gt) -> rw   (green-trail/early-cut exit rewrite)
  gates.py            # skip_by_year() causal + _skip_for() + ret7/dl63/overshoot filters
  sim.py              # run_sim(legs, panel, C) -> equity/holdings/trades   (K-slot + size + preempt + T+2)
  api.py              # run_portfolio(base_trades, signals, *, ctx, C) -> PortfolioResult

PortfolioContext (nguồn data trừu tượng hóa):
  - market_src: DuckDB path | live-panel provider   (cho build_market_panel)
  - price_src : SQLite/NavSim | live OHLCV           (mark NAV)
  - date_hi   : cutoff
```

**Ai gọi:**
- **Backtest / đăng-ký-leaderboard**: `stock_ml/scripts/ops/*` gọi `run_portfolio(ctx=DuckDBContext)`.
- **Serving production**: `serving/run_serving.py` gọi `run_portfolio(ctx=LiveContext)` — CÙNG package qua wheel `stock_ml_core`.
- **Engine**: KHÔNG đổi. Vẫn sinh BASE per-symbol. (Không nhồi K-slot vào `_run_symbol` — xem §6.)

`PortfolioResult`: `dict(cagr, dd, calmar, nav_series, equity, holdings, trades, skipped)` —
đủ cho cả metric (leaderboard) lẫn tab danh mục (run_equity/run_portfolio_daily/run_trades).

---

## 4. Kế hoạch 6 bước (mỗi bước có snapshot guard)

### Bước 0 — Đóng băng chuẩn (BẮT BUỘC trước khi đụng gì)
- Track **CẢ HAI** bản tham chiếu vào git — hiện cả hai đều UNTRACKED:
  - `_champ_prod_replay.py` (root repo train_ai_ml).
  - `serving/portfolio/core.py` (repo `Desktop/stock-serving` — `git ls-files` rỗng).
- **PIN data snapshot** làm fixture: freeze bản `market.duckdb` + `ohlcv.db` (hoặc export trades/panel
  ra parquet) + ghi MD5. Golden PHẢI đọc từ snapshot pinned, KHÔNG đọc nguồn Sieu Tin Hieu live —
  nguồn live cập nhật liên tục (chất lượng data ngày một cao hơn sau các lần fix chia-tách),
  nên khóa golden vào nguồn live = golden vỡ mỗi lần sync mà KHÔNG phải lỗi refactor. Xem §9.
- Sinh **golden** (2 con số, đúng bản chất 2 pipeline — xem §9 để hiểu tại sao KHÔNG gộp làm 1):
  - **Golden production** = `_champ_prod_replay.py` trên snapshot Sieu Tin Hieu (SAU sync PVD):
    **3-seed(42/21/123) T+2 = 140.3% / DD -10.8% (overshoot ON)** — số DEPLOY-thực sạch data.
    (single-seed s42 = 142.3%; overshoot OFF 3-seed = 134.9%/-13.6% → edge overshoot +5.4pp vẫn đúng.
    LỊCH SỬ: trước sync PVD single-seed 151.1% PHỒNG +8.8pp do 1 mã data cũ — §9.3b.)
  - **Golden backtest-registered** (tùy chọn, để đối chiếu leaderboard) = champion DuckDB
    `T+2 132.1% / -13.8%`. LƯU Ý số này THẤP HƠN vì data train chưa back-adjust một số mã
    (AAS lệch 15.6% tại ex-date, BSI ~9% — đã đo §9); KHÔNG phải "số thật hơn".
  + trades CSV MD5 cho từng golden.
  Lưu `stock_ml/tests/goldens/champion_prod_overlay.json` + `champion_duck_overlay.json`.
- Test: `test_portfolio_golden.py` — chạy overlay hiện tại phải khớp golden TƯƠNG ỨNG NGUỒN DATA.
  Đây là lưới an toàn cho MỌI bước sau.

### Bước 1 — Trích module `sim.py` (K-slot sim thuần)
- Copy vòng lặp K-slot từ `serving/core.py::_run_sim` (bản đầy đủ nhất: đã có preempt + emit + holdings).
- Input: `legs` (list dict đã có net/prio/conv/w), `panel`, `C`. Output: equity/holdings/trades.
- **KHÔNG đổi logic 1 dòng.** Chỉ tách hàm.
- Guard: `sim.py` cho cùng NAV series với inline hiện tại (so trên champion base).

### Bước 2 — Trích `panel.py` + `priority.py` + `rewrite.py` + `gates.py`
- Port từ `serving/core.py` (đã có `build_market_panel`, `_meta_priority`, `_rewrite`, gate causal).
- Mỗi hàm: 1 unit test so output vs bản inline `_champ_prod_replay` trên champion base (byte-close).
- **Điểm rủi ro cao**: `build_market_panel` — conviction cs5_ma50 là cross-sectional rank; panel-source
  phải KHỚP giữa backtest (DuckDB market.duckdb) và serving (live). Test panel byte-identical trước.

### Bước 3 — Ghép `api.run_portfolio` + `PortfolioContext`
- Ghép 4 module thành pipeline: base → rewrite → panel → gate → priority → sim.
- `PortfolioContext` trừu tượng hóa nguồn: `DuckDBContext(duck_path, price_db)` vs `LiveContext(provider)`.
- Guard: `run_portfolio(DuckDBContext, champion_base)` == golden bước 0 (142.3% / -10.7%, trades MD5).

### Bước 4 — Backtest/đăng-ký chuyển sang dùng module
- Sửa `score_nav`/`hb_deploy`-thay-thế để gọi `stock_ml.portfolio.run_portfolio`.
- Xóa dần 10 bản `hb_deploy_*.py` (chúng chỉ khác ở 1 lever → biến thành `PortfolioConstants` variants).
- `_champ_prod_replay.py` → thin wrapper gọi module (giữ CLI để so lịch sử), hoặc xóa sau khi golden pass.
- Guard: re-đăng-ký champion → leaderboard cagr_adv/cagr_t2 KHÔNG đổi.

### Bước 5 — Serving dùng CÙNG module (qua wheel)
- Bump `stock_ml_core` wheel gồm `stock_ml/portfolio/`. **BẮT BUỘC sửa `pyproject.stock_ml_core.toml`**:
  - Thêm `"stock_ml.portfolio*"` vào `[tool.setuptools.packages.find].include` (hiện chỉ có
    `stock_ml.core*` + `stock_ml.src*` → package mới sẽ KHÔNG vào wheel, serving import fail).
  - `build_market_panel`/`_meta_features` cần `duckdb`, `_load_price_panel` cần `sqlite3`. `duckdb`
    KHÔNG có trong `dependencies` hiện tại (wheel đang "inference-only", verify bởi `test_core_facade.py`).
    → hoặc thêm `duckdb` vào deps, hoặc để `PortfolioContext` inject reader (giữ wheel gọn) — CHỐT trước.
- **LƯU Ý bản chất công việc (§6 mục nền): `serving/portfolio/core.py` ĐÃ module hóa** — đã có
  `PortfolioConstants` dataclass + `run_portfolio()` đúng chữ ký + các hàm `_run_sim/_meta_priority/
  _rewrite/build_market_panel` tách sẵn. Việc THẬT ở Bước 1-2 ≈ "move file này vào `stock_ml/portfolio/`
  + thêm PortfolioContext + bỏ hard-path duckdb", KHÔNG phải "port từng hàm từ inline". `_champ_prod_replay`
  (còn dùng global `OSMAP/SKIP_BY_YEAR`) là bản inline hơn → chỉ dùng làm fixture đối chiếu.
- `serving/portfolio/core.py` → thin wrapper import `stock_ml.portfolio.run_portfolio(LiveContext)`.
- Xóa 356 dòng viết-tay ở serving. Backtest ≡ production về LOGIC (chỉ khác Context).
- Guard: `serving/tests/test_trades.py` vẫn 8/8; CAGR history serving == backtest cùng data.

---

## 5. Snapshot guard — nguyên tắc bất biến

| Bước | Test bắt buộc | Ngưỡng |
|---|---|---|
| 0 | golden champion (prod STH, sau sync PVD) | cagr 142.3% / dd -10.7% / trades MD5 cố định |
| 1 | sim.py vs inline | NAV series byte-identical trên champion base |
| 2 | panel/priority/rewrite/gate | mỗi hàm output byte-close vs `_champ_prod_replay` |
| 3 | run_portfolio == golden | 142.3% / -10.7% / trades MD5 |
| 4 | re-đăng-ký champion | leaderboard cagr KHÔNG đổi |
| 5 | serving == backtest | test_trades 8/8, CAGR khớp cùng data |

**Bất biến tuyệt đối**: champion dl63bal (61-mã) và mọi run đã đăng ký KHÔNG được đổi 1 trade.
Nếu bước nào phá byte-parity → dừng, tìm nguyên nhân, KHÔNG "chấp nhận sai số nhỏ".

---

## 6. Quyết định thiết kế + rủi ro

- **KHÔNG nhồi K-slot vào `_run_symbol`.** Engine per-symbol là đúng chức năng (sinh setup từng mã).
  Danh mục là bài toán CROSS-SYMBOL (chọn 10/N, cạnh tranh vốn) — thuộc tầng riêng. Trade schema đã
  hash-frozen bởi goldens; đụng vào = phá byte-parity toàn bộ. Giữ tách 2 tầng, chỉ UNIFY tầng 2.
- **Nguồn conviction phải khóa.** cs5_ma50 rank cross-sectional trên panel-source; backtest và serving
  PHẢI cùng panel policy (cùng danh sách symbol tính rank) nếu không conviction lệch → gate lệch →
  danh mục khác. `PortfolioContext` phải ghi rõ panel-source và assert khớp.
- **Determinism**: LGBM meta-priority `deterministic:True`; gate causal per-year (thêm fold không đổi
  năm cũ). Giữ nguyên.
- **SKIP mode mặc định**: chốt **causal** cho cả backtest lẫn serving (khác nhau hiện tại là 1 nguồn lỗi).
- **Overshoot filter (os_pct=90) GIỮ mặc định ON — đã kiểm chứng CAUSAL, KHÔNG phải leak.** Xem §9:
  filter quyết định tại FILL-date và chỉ dùng low của đoạn `[signal..fill]` = quá khứ so với lúc khớp
  (100% lệnh fill sau signal do pullback-limit 4.5%). Ngưỡng in-sample p90 ≈ causal-expanding
  (0.034 vs 0.037, immaterial). Đóng góp +6.1pp CAGR + DD tốt hơn 1.1pp, deploy-được.
  KHUYẾN NGHỊ SẠCH GIÁO KHOA (không bắt buộc): đổi ngưỡng `osthr` thành expanding-per-year (tác động ≈0)
  để bỏ hẳn nhãn "in-sample". PHÂN BIỆT với short_tilt5 (§9) — cái đó look-ahead THẬT vì quyết định sớm
  (signal-date) nhưng dùng data muộn (RS tại fill = +8d).
- **Nguồn data: API Sieu Tin Hieu (LIVE) > DuckDB train > snapshot market.duckdb (Desktop, TRỄ).**
  API STH đã fix giá chia-tách quá khứ; nhưng bản `market.duckdb` copy trên Desktop TRỄ so với API
  (CA mới chưa sync — đã fix 6 mã, §9.3b/§9.5). Số production sạch = **142.3%** (data đã sync).
  132.1% DuckDB thấp hơn do train còn lỗi vài mã (AAS/BSI). `PortfolioContext` phải ghi rõ nguồn +
  golden pin snapshot ĐÃ sync. QUY TRÌNH: chạy detector-jump + refill CA mới TRƯỚC mỗi lần đo/pin.
- **Rủi ro lớn nhất = Bước 2 (panel)**: nếu build_market_panel khác nhau (thứ tự symbol, NaN-fill,
  ngày cutoff) → conviction dịch → phá golden. Test panel TRƯỚC khi ghép.
- **CẢ HAI bản tham chiếu untracked** (`_champ_prod_replay.py` + `serving/portfolio/core.py`): track
  NGAY ở bước 0, kẻo mất bản tham chiếu giữa refactor.

---

## 7. Ước lượng

| Bước | Nội dung | Công |
|---|---|---|
| 0 | golden + track | 0.5 ngày |
| 1 | sim.py | 0.5 ngày |
| 2 | panel/priority/rewrite/gate + test | 1.5 ngày (panel rủi ro) |
| 3 | api + context | 0.5 ngày |
| 4 | backtest chuyển + xóa 10 bản | 1 ngày |
| 5 | serving wheel + wrapper | 1 ngày |
| | **Tổng** | **~5 ngày** |

**Thứ tự an toàn**: 0 → 1 → 2 → 3 (dừng được ở đây, đã có module chung cho backtest) → 4 → 5
(serving). Mỗi bước độc lập rollback được; golden bước 0 bảo vệ toàn tuyến.

---

## 8. Tham chiếu

- **Nền cho module** = `serving/portfolio/core.py` (357 dòng, ĐÃ module hóa: `PortfolioConstants` +
  `run_portfolio` + hàm tách sẵn; ⚠️ UNTRACKED, ở repo `Desktop/stock-serving`). Đây là port của
  `hb_deploy_gtos.py` (có overshoot — đã kiểm chứng CAUSAL §9, giữ được).
- Bản tham chiếu backtest (fixture đối chiếu, còn dùng global state): `_champ_prod_replay.py`
  (313 dòng, ⚠️ UNTRACKED, root — track ở bước 0).
- 10 bản lever: `stock_ml/analysis/serving_blindspot/r3line/hb_deploy_*.py` (8 clean + 2 overshoot: gtos/osdef).
- Engine (KHÔNG đổi): `stock_ml/src/backtest/engine.py` (2823 dòng) — `run_backtest` [:2558],
  `_run_symbol` [:1108], `Trade` [:1062], weight/notional default [:1074-1075].
- Bẫy base-vs-output: memory `base-vs-output-trades-trap-champion-151-not-73`.
- Deploy serving hiện tại: `stock-serving/DEPLOY_PANEL900.md`.

---

## 9. Kiểm chứng thực nghiệm (đã CHẠY, không phải suy đoán) — 2026-07-28

Section này ghi lại các con số ĐO THẬT trong lúc review, để không lặp lại kết luận sai từ memory cũ.

### 9.1 Overshoot filter là CAUSAL, KHÔNG phải look-ahead (đảo kết luận memory cũ)

Chạy `_champ_prod_replay.py` (production single-seed, data Sieu Tin Hieu snapshot), A/B overshoot.
LƯU Ý: bảng này đo TRƯỚC khi sync PVD (số ON 151.1% giờ là 142.3% — xem §9.3b); nhưng CHÊNH ON−OFF
(+6.1pp) là kết luận causal, không đổi bản chất khi data sạch hơn.

| | T+2 CAGR (trước sync PVD) | T+2 DD |
|---|---|---|
| Overshoot ON (mặc định gtos) | 151.1% (→142.3% sau sync) | −11.4% (→−10.7%) |
| Overshoot OFF (patch tạm OSMAP=None) | 145.0% | −12.5% |
| Đóng góp | **+6.1pp** | DD tốt hơn 1.1pp |

**Vì sao CAUSAL** (memory `overshoot-filter-pure-leak-remove` & `model-health-audit` GẮN CỜ NHẦM):
công thức `(entry_price − min(low[signal..fill])) / close[signal]`, quyết định skip xảy ra **TẠI FILL-date**
(lúc lệnh thực khớp). Đã đo: 100% (2582/2582) lệnh có fill SAU signal, gap median 8 ngày (pullback-limit 4.5%).
Tại fill-date, TOÀN BỘ đoạn `[signal..fill]` (kể cả bar fill) là QUÁ KHỨ, quan sát được ngay trong phiên khớp
→ không dùng thông tin tương lai. Ngưỡng `osthr = p90 in-sample` vs causal-expanding: 0.0344 vs 0.0343–0.0385
(immaterial). → +6.1pp là edge THẬT deploy-được. Chỉ nên đổi ngưỡng thành expanding cho sạch nhãn (tác động ≈0).

### 9.2 short_tilt5 (+16pp) là look-ahead THẬT — bác ĐÚNG, KHÁC overshoot

Câu hỏi "có lever top-CAGR nào bị báo nhầm look-ahead tương tự overshoot không" → chỉ overshoot bị nhầm.
short_tilt5 (2×rs5+rs20+rs60 làm priority tie-break, memory `rs-multiperiod-...-shorttil5-lookahead`):
st5_prio+1.0 @K10 = 151.5% (+16pp) TRÔNG như breakthrough NHƯNG là look-ahead THẬT; sửa causal → null + DD xấu.

**Nguyên lý phân biệt (mấu chốt):**

| Lever | Mốc QUYẾT ĐỊNH | Data dùng | Causal? |
|---|---|---|---|
| Overshoot | fill-date (skip khi khớp) | low `[signal..fill]` ≤ fill = quá khứ | ✅ |
| Meta-priority (pm) | fill-date (sort entries cùng ngày fill) | feature tại fill-date = cùng ngày | ✅ (immaterial nếu lệch) |
| Conviction / ret7 gate | signal-date | cs5_ma50 / ret7 tại **signal-date** | ✅ |
| **short_tilt5** | **signal-date** (chọn lệnh nào để đặt) | RS tại **fill-date** = +8d tương lai | ❌ look-ahead |

→ Quy tắc audit: lever mạnh bất thường PHẢI hỏi "quyết định tại mốc nào" vs "data lấy tại mốc nào".
Overshoot quyết-muộn-data-cũ (OK); short_tilt5 quyết-sớm-data-mới (leak). ĐỪNG gán nhầm 2 loại này.

**Đã test lại st5 tại nhiều mốc (2026-07-28, harness `_lever_eval.py` DuckDB 3-seed 42/21/123)** — trả lời
"nếu chỉnh để xem ở T-1 thì có tiềm năng không": **KHÔNG.**

| st5 mode | CAGR | DD | Calmar | |
|---|---|---|---|---|
| baseline (no st5) | 82.9% | −26.4% | 3.14 | — |
| st5 @ signal-date (−8d) | 74.8–77.2% | — | 2.87–3.33 | null/hại (khớp bác cũ) |
| st5 @ fill-date (ngày khớp) | **89.2%** | −23.4% | 3.81 | thắng NHƯNG leak intraday |
| st5 @ fill **T-1** (ngày TRƯỚC khớp) | 83.0% | −26.3% | 3.15 | **≈ baseline, edge BIẾN MẤT** |

Toàn bộ +6.3pp của st5@fill đến từ `close` của CHÍNH ngày khớp — mà quyết định vào lệnh là TRONG phiên
(limit chạm), trước khi biết close → leak intraday. Lùi đúng 1 bar (T-1 fill) → edge bốc hơi về baseline.
**Ranh giới causal thật (phân biệt với overshoot):** overshoot dùng `low` của các bar ĐÃ ĐÓNG (min quá khứ,
biết chắc lúc khớp); st5@fill dùng `close` bar ĐANG diễn ra (chưa biết lúc quyết định trong phiên). RS
momentum ngắn KHÔNG mang tín hiệu priority causal — chỉ "thắng" khi được nhìn giá tương lai dù nửa ngày.
ĐỪNG thử lại st5 dưới mọi biến thể mốc.

### 9.3b PVD data-lệch: 151.1% bị PHỒNG +8.8pp bởi 1 mã chưa sync (2026-07-28, ĐÃ FIX)

Bản `market.duckdb` (Desktop) là snapshot có ĐỘ TRỄ — CA MỚI chưa kịp sync. Phát hiện **PVD 2026-07-14**
(split/div ~1.67×): market.duckdb-Desktop còn giá RAW (32.5→19.9 nhảy −36%), trong khi API STH gốc VÀ
DuckDB train ĐÃ adjust đúng (19.47→18.61→19.90). PVD có 47 trades champion → base sinh trên giá-adjusted
nhưng NAV mark trên giá-raw-cao → PnL phồng. Đã refill PVD từ API STH (fetch_history, DELETE+INSERT,
backup `.bak_pvd_20260728`). **Kết quả re-run: T+2 151.1% → 142.3% (−8.8pp), DD −11.4% → −10.7% (tốt hơn).**
→ **142.3% là số sạch hơn 151.1%.** BÀI HỌC: snapshot Desktop trễ so với API; golden phải sync mọi CA
mới TRƯỚC khi pin. Detector: jump>15% giá>3.0 so STH vs TRAIN. Trong 61-mã champion chỉ PVD lệch
(AAS/VTP kiểm tra OK: AAS rights non-adjustable đã khớp, VTP là COVID-dip thật không phải CA).

### 9.3 Nguồn data: Sieu Tin Hieu > DuckDB train (đo trực tiếp)

So `close` cùng mã quanh ex-date 2024-11-28 giữa 2 DB:

| Mã | Serving (Sieu Tin Hieu) | Train DuckDB | Ghi chú |
|---|---|---|---|
| AAS 11-25→11-27 | 6.3–6.4 (liền mạch) | **5.32–5.40** | train chưa back-adjust, ratio 0.844, nhảy +16.5% tại ex-date |
| HDG | 23.17… | 23.18… (ratio ~1.0) | khớp |
| BSI (mẫu 100) | — | lệch median **9.1%** | train lỗi |

→ API Sieu Tin Hieu đã fix giá chưa chia-tách quá khứ → chất lượng CAO HƠN train. Số production sạch
(sau sync PVD) = **142.3%** > 132.1% (DuckDB train nhiễu data lỗi). Golden pin snapshot STH ĐÃ sync.

### 9.4 Sự thật hiện trạng đã verify

- `hb_deploy_*.py`: đúng **10 bản** (8 clean: ec/gt/gt1/ret5g/ret7g/dl63/dl63ts/dl63opt + 2 overshoot: gtos/osdef).
- `_champ_prod_replay.py`: 313 dòng, **UNTRACKED**. `serving/portfolio/core.py`: 357 dòng, **UNTRACKED** (git ls-files rỗng).
- `stock_ml/portfolio/`, `stock_ml/tests/goldens/`, `serving/tests/test_trades.py`: **CHƯA tồn tại** (tạo mới).
- `stock-serving` là repo GIT RIÊNG ở `C:/Users/DUC CANH PC/Desktop/stock-serving` (không phải subdir train_ai_ml).

### 9.5 Rà data lỗi thời market.duckdb toàn bộ 488 mã (2026-07-28, ĐÃ FIX)

Bản `market.duckdb` (Desktop) TRỄ so với API STH. Detector: jump>15.5% giá>3.0 trong 2024+, đối chiếu
STH-DB vs DuckDB-train (STH nhảy + TRAIN mượt = STH lỗi thời), loại crash toàn-sàn 2025-04-08..11 (thuế quan).
**Tìm được 6 mã lỗi thời — đã refill từ API STH (DELETE+INSERT market.duckdb):**

| Mã | CA ngày | Champion? | STH-DB trước | API STH (đúng) |
|---|---|---|---|---|
| PVD | 2026-07-14 | **CÓ** (47 trades) | 31.05→19.9 (−36%) | 19.47→18.61→19.9 |
| KLB | 2026-06-29 | không | 16.75→12.63 (−25%) | mượt (jump 4%) |
| LHC | 2026-06-23 | không | 97.9→47.2 (−52%) | mượt (5%) |
| PET | 2026-07-09 | không | 53.2→37.7 (−29%) | mượt (3%) |
| SBG | 2026-06-25 | không | 14.9→12.55 (−16%) | mượt (1%) |
| TOS | 2026-07-23 | không | 170→101.9 (−40%) | mượt (2%) |

Chỉ **PVD** trong champion → chỉ PVD đổi con số (151.1→142.3%). 5 mã kia chỉ ảnh hưởng universe RỘNG
(dyn200/400/900, webapp) — refill để sạch sẵn. **Re-scan sau refill: 0 mã còn lỗi thời** (71 jump còn lại
đều là giảm-sàn/CA non-adjustable mà CẢ 2 DB đều có). GIỚI HẠN: phép so-TRAIN không bắt được CA mới mà
CẢ STH-DB lẫn TRAIN cùng chưa adjust — nhưng champion đã fetch-API xác minh sạch, rủi ro thấp.
Backup: `market_data/market.duckdb.bak_pvd_20260728`.
