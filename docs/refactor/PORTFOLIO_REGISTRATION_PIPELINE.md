# Refactor: Chuẩn hóa Đăng ký & Tab Danh mục (phần còn lại sau UNIFICATION)

> Phần tiếp theo của [PORTFOLIO_LAYER_UNIFICATION.md](PORTFOLIO_LAYER_UNIFICATION.md) (đã xong 0-5).
> Module `stock_ml/portfolio` đã là single-source-of-truth cho LOGIC; tài liệu này chuẩn hóa nốt
> tầng GHI (đăng ký kết quả overlay vào DB) và tầng ĐỌC (tab danh mục) — nơi bẫy base-vs-output
> vẫn còn sống ở mức dữ liệu. Ngày lập: 2026-07-29.

---

## 0. TL;DR

- **Vấn đề 1 — `run_trades` lẫn 2 tầng dữ liệu**: 19 run lưu trades ĐÃ-overlay (OUTPUT, có
  preempt/green_trail/early_cut), ~10 run lưu BASE. Consumer nào đọc `run_trades` không phân biệt
  sẽ dính đúng bẫy 151%→73%. Guard trong module chỉ chặn phía đọc — gốc bệnh ở phía ghi chưa xử.
- **Vấn đề 2 — tab danh mục không nhất quán**: `runs.py` đọc trades thẳng từ `run_trades`;
  với run lưu BASE (vd champion dl63size) tab đang ghép BASE-trades với overlay-equity trên cùng
  màn hình. Với run lưu OUTPUT thì `/runs/{id}/trades` lại trả trades đã overlay như thể là BASE.
- **Vấn đề 3 — overlay không có danh tính**: `run_equity`/`run_portfolio_daily` chỉ key theo
  `run_id`, không ghi overlay sinh từ config nào (stat_mode/panel/date_hi/version). Số chính thức
  (module 0.4.1 causal) và số script cũ không phân biệt được ngoài text note.
- **Giải pháp**: bảng mới `run_trades_overlay` + bất biến **`run_trades` = BASE-only** (migrate 1 lần
  19 run OUTPUT) + script đăng ký chính thức idempotent theo `overlay_config_hash` + tab đọc đúng tầng.
- Ước lượng: **~2.5 ngày** (B1-B5 lõi 1 ngày; B6 dl63 port 1 ngày; B7 view theo-mã 0.5 ngày).

---

## 1. Hiện trạng — đã kiểm chứng 2026-07-29 (đừng tin lại memory, số đo trực tiếp DB)

**Provenance `run_trades` theo run** (quét `exit_reason`; OUTPUT = có preempt/green_trail/early_cut):

- **BASE (10 run họ x2_struct)**: `x2_struct_to` (2485), `..._dl63size` (2470, CHAMPION),
  `k12preempt_cssize`, `k16meta`, `k16preempt`, `k16prio`, `k16size`, `k25preempt`, `k25size`,
  `rfentry` (đều ~2490-2506 — bản đăng ký qua `run_template_experiment`).
- **OUTPUT (19 run)**: `k10_cs5ma50` (1405), `r7earlycut` (1304), `dl63aggr/bal/opt`
  (1297/1245/1277), `gt_s1` (1370), `gtos` (1253), `gtrail` (1326), `osdef` (1232),
  `t2ret5g` (1156), `t2ret7g` (1223), `k10c2m005` (1402), `k10c2m005_cs5` (1392),
  `k10preempt_cssize` (1390), `k12size2` (1521), `k16preempt_cssize` (1776),
  `k8preempt_cssize` (1221), `_static900-69338138` (2923), `dyn900_champion` (2100).

**Tab danh mục**: `run_equity` + `run_portfolio_daily` có data cho 20 run (populate bởi script
hb_deploy cũ — kết quả GIÁN TIẾP, không phải module). Ba query per-day trades trong
`stock_ml/api/routes/runs.py` (~L285/L315/L324) đọc thẳng `run_trades`.

**Đã có sẵn**: `leaderboard_nav` có cột `cagr_overlay/maxdd_overlay/overlay_k/overlay_note`
(API + dashboard đã hiển thị, commit 596aafb0); 2 run đã có số chính thức: champion 131.4%/−14.1
và gtos 123.5%/−12.1 (module 0.4.1 causal, panel train-duck 901, NAV serving ohlcv,
date_hi 2026-07-08). ORM: `db/models/portfolio.py` (RunEquity/RunPortfolioDaily/RunSkipped/
RunPending). Alembic head = **0028** → migration mới là **0029**.

**Module trả đủ nguyên liệu tab**: `run_portfolio()` → `equity` (date/nav/cash/exposure/
n_positions), `holdings` (date/symbol/weight/pnl/entry_dt/age/is_new/conv/prio), `trades`
(symbol/entry_date/entry_price/exit_date/exit_price/holding_days/pnl_pct/exit_reason/conv/prio),
`skipped` (symbol/entry_date/reason), `rewritten`.

**`run_signals` đầy đủ mọi mã mọi ngày** (champion 95.770 dòng) — chi tiết mã đã xem được full
tín hiệu; cái thiếu là view gộp theo-mã đối chiếu tín-hiệu ↔ số-phận-danh-mục (B7).

**Vận hành**: container `stock-ml-api` bake code vào image (KHÔNG mount source) — đổi
`stock_ml/api/**` phải `docker compose build api` (hoặc hot-patch `docker cp` + restart tạm).
Dashboard mount trực tiếp — chỉ cần Ctrl+F5.

---

## 2. Nguyên tắc thiết kế

1. **Bất biến dữ liệu `run_trades` = BASE-only.** OUTPUT có nhà riêng (`run_trades_overlay`).
   Giải bẫy base-vs-output tận GỐC ở tầng data; assert trong `api.run_portfolio` chỉ còn là
   defense-in-depth. (Đây là dạng mạnh của quyết định "provenance phía ghi" trong doc UNIFICATION §6.)
2. **Overlay phải có danh tính**: mỗi lần đăng ký ghi `overlay_config_hash` = md5(JSON config:
   PortfolioConstants + market_db + ohlcv_db + date_hi + wheel version + base run_id). Idempotent
   như pattern `score_nav_leaderboard` (cùng hash → skip trừ `--force`).
3. **Module thuần, writer mỏng**: `stock_ml/portfolio` KHÔNG biết DB. Toàn bộ ghi nằm trong
   1 script ops. Không nhét SQLAlchemy vào wheel.
4. **Ghi nguyên tử**: mỗi run 1 transaction (delete+insert 4 bảng + update leaderboard_nav).
   Không bao giờ để tab nửa cũ nửa mới.
5. **Không phá số lịch sử**: các dòng `cagr_t2` gián tiếp giữ nguyên làm đối chiếu; số chính thức
   sống ở `cagr_overlay`. Tab của run nào populate lại mới đổi.

---

## 3. Kiến trúc đích

```
run_template_experiment ──> run_trades (BASE, bất biến) + run_signals (full)
                                   │
       stock_ml/scripts/ops/register_overlay.py  <── PortfolioConstants + nguồn data
                                   │   (gọi stock_ml.portfolio.run_portfolio — module thuần)
                                   ▼ 1 transaction / run
   run_equity | run_portfolio_daily | run_skipped | run_trades_overlay   ← tab danh mục đọc
   leaderboard_nav.cagr_overlay/maxdd_overlay/overlay_k/overlay_note/overlay_config_hash
```

**Bảng mới `run_trades_overlay`** (migration 0029, ORM thêm vào `db/models/portfolio.py`):
`id, run_id (idx), symbol, entry_date, entry_price, exit_date, exit_price, holding_days,
pnl_pct, exit_reason, conv, prio, created_at`. Thêm cột `overlay_config_hash TEXT` vào
`leaderboard_nav` (ALTER trong cùng migration — bảng này vốn do script tạo, không ORM;
migration phải `IF NOT EXISTS`-safe).

---

## 4. Kế hoạch từng bước (mỗi bước có guard)

### B1 — Migration 0029 + ORM (0.5h)
- Tạo `run_trades_overlay` + index `(run_id)`; ALTER `leaderboard_nav` ADD `overlay_config_hash`.
- Guard: `alembic upgrade head` chạy sạch; bảng rỗng không ảnh hưởng gì đang chạy.

### B2 — Migrate 1 lần: 19 run OUTPUT rời khỏi `run_trades` (0.5h)
- Detector: run có bất kỳ `exit_reason IN ('preempt','green_trail','early_cut')` → OUTPUT.
  (Đã enumerate ở §1 — script phải TỰ QUÉT lại lúc chạy, không hard-code danh sách.)
- 1 transaction: INSERT các dòng đó vào `run_trades_overlay` (conv/prio = NULL vì bản cũ không
  lưu) rồi DELETE khỏi `run_trades`.
- Guard: (a) tổng dòng chuyển = tổng dòng xóa, per-run count khớp bảng §1; (b) sau migrate:
  `SELECT count(*) FROM run_trades WHERE exit_reason IN (overlay-set)` = **0** — bất biến
  BASE-only xác lập; (c) chạy lại detector → 0 run OUTPUT còn sót.
- LƯU Ý: sau bước này run_trades của 19 run đó RỖNG (chúng chưa từng lưu BASE) — đúng thực tế;
  `/runs/{id}/trades` xử lý ở B4.

### B3 — `stock_ml/scripts/ops/register_overlay.py` (0.5 ngày)
- CLI: `--run-id <board_row> [--base-run <run_id>] [--stat-mode causal] [--market-db ...]
  [--ohlcv-db ...] [--date-hi 2026-07-08] [--force]`. `--base-run` mặc định = chính run đó
  (cho run lưu BASE); bắt buộc chỉ định khi board-row không có BASE (vd gtos → base `x2_struct_to`).
- Luồng: đọc BASE trades + signals từ Postgres → `run_portfolio(ctx=DuckDBContext(...))` →
  transaction: replace `run_equity`/`run_portfolio_daily`/`run_skipped`/`run_trades_overlay`
  theo run_id → update `leaderboard_nav` (cagr/dd/k/note/config_hash).
- Idempotent: hash trùng → skip (log), `--force` để ghi lại.
- Guard: chạy champion 2 lần liên tiếp — lần 2 skip; số ghi vào bảng == `cagr_overlay` đã có
  (131.4%/−14.1 — vì cùng config với lần chấm 596aafb0-era). Trades overlay của champion phải có
  đủ conv/prio (không NULL).

### B4 — Tab đọc đúng tầng (0.5h + rebuild api)
- `runs.py`: 3 query per-day trades (L285/L315/L324) → `run_trades_overlay`.
- `/runs/{id}/trades`: ưu tiên `run_trades_overlay` nếu có dòng, fallback `run_trades`
  (giữ tương thích run chưa từng có overlay); response thêm field `layer: overlay|base`
  để UI ghi rõ đang xem tầng nào.
- Rollout: `docker compose build api && docker compose up -d api` (hot-patch cp chỉ là tạm).
- Guard: tab champion — NAV cuối trong `run_equity` khớp `cagr_overlay` (131.4%); panel trades
  ngày bất kỳ chỉ hiện exit thuộc tầng overlay; `/trades` của dl63bal trả `layer=overlay`
  (data migrate B2), của run BASE thuần trả `layer=base`.

### B5 — Populate chính thức đợt đầu (0.5h)
- Chạy `register_overlay.py` cho: champion `dl63size` (base = chính nó) và `gtos`
  (base = `x2_struct_to`). Các run OUTPUT khác GIỮ tab cũ (kết quả gián tiếp lịch sử) đến khi
  có nhu cầu — trừ dl63-family cần B6 trước.
- Guard: hai run có tab nhất quán module-số; `RUN_PORTFOLIO_GOLDEN=1` suite vẫn 11/11
  (đăng ký không đụng module).

### B6 — (tùy chọn, mở khóa dl63-family) Port dl63 gate (1 ngày)
- Lấy diff `hb_deploy_dl63*.py` tại git **9e37efd7** (~150 dòng): gate dist-low-63 (dl63 0.13
  như note `_static900`) + time-stop (dl63ts). Port thành field trong `PortfolioConstants`
  (vd `dl63_thr: float | None = None`) + gate trong `gates.py`; default None = OFF →
  **golden hiện tại phải pass nguyên vẹn** (guard bắt buộc trước khi dùng).
- Fixture đối chiếu: tái lập số gián tiếp dl63bal **134.7% T+2 trên serving-duck 488-panel**
  (đúng data gốc của số cũ) trước, RỒI mới chấm số chính thức train-duck 901 — tách bạch
  "port đúng" khỏi "khác do panel/data".
- Sau đó `register_overlay.py --run-id dl63bal --base-run x2_struct_to` (+opt/aggr).

### B7 — (tùy chọn) View theo-mã: tín hiệu ↔ số phận danh mục (0.5 ngày)
- Endpoint mới `/runs/{id}/symbol/{sym}/portfolio-fate`: join `run_signals` (signal==1) ×
  `run_trades` BASE (thành setup?) × `run_trades_overlay` (được fill) × `run_skipped` (lý do loại)
  → mỗi tín hiệu 1 dòng: `signal_date, score, became_base_trade, filled, skip_reason`.
- UI: thêm section trong model-details (mã đang xem): "N tín hiệu → n1 vào BASE → n2 vào danh mục;
  loại: conv_skip x, ret7 y, overshoot z, hết-slot w".
- Guard: đếm đối chiếu — filled + skipped + không-thành-base = tổng signal của mã.

---

## 5. Bảng guard tổng

| Bước | Test bắt buộc | Ngưỡng |
|---|---|---|
| B1 | alembic upgrade head | sạch, không đụng bảng cũ |
| B2 | quét exit_reason toàn `run_trades` | 0 dòng overlay-level còn lại; per-run count khớp §1 |
| B3 | đăng ký champion 2 lần | lần 2 skip (hash); số == cagr_overlay hiện có |
| B4 | tab champion + /trades layer flag | NAV cuối == cagr_overlay; dl63bal layer=overlay |
| B5 | golden suite | RUN_PORTFOLIO_GOLDEN=1 → 11/11 pass, không đổi |
| B6 | dl63 OFF golden + fixture 488-panel | golden nguyên vẹn; tái lập 134.7% trước khi chấm mới |
| B7 | đếm đối chiếu per-symbol | filled+skipped+non-base = tổng signal |

---

## 6. Rủi ro + quyết định mở

- **Consumer ngoài API đọc `run_trades`**: trước B2 phải grep toàn repo (`analysis/`, scripts
  nghiên cứu untracked) — script nào đọc run_trades của 19 run OUTPUT sẽ đổi hành vi (thành rỗng).
  Đã biết an toàn: leaderboard stats đọc qua `leaderboard_runs` (không đụng), meta-training của
  module đọc BASE (đúng mong muốn). Còn lại: kiểm kê lúc thực thi.
- **`leaderboard_nav` không có ORM** (script-created): migration ALTER phải chịu được cả DB
  chưa có bảng (CREATE IF NOT EXISTS trước, như DDL trong `score_nav_leaderboard.py`).
- **2 run ngoài họ x2_struct** (`_static900`, `dyn900_champion`) là OUTPUT — migrate cùng B2;
  số của chúng vốn có cảnh báo look-ahead/nghiên cứu, không populate lại chính thức.
- **Quyết định mở 1**: tab cũ của 17 run OUTPUT không populate lại — GIỮ (sử liệu, note trong
  leaderboard_nav vẫn ghi gián-tiếp) hay XÓA? Đề xuất: GIỮ.
- **Quyết định mở 2**: 2 hệ `cagr_adv` lẫn nhau trên leaderboard (66.3% CAGR-NAV mới vs ~135%
  hệ cũ) — ngoài phạm vi doc này nhưng nên chốt một đợt riêng (thêm cột `cagr_basis` hoặc
  re-score toàn bộ bằng 1 hệ).

## 7. Ngoài phạm vi

Nghiên cứu exposure cấp danh mục (regime-sizing / breadth-exit / VN30F-hedge) — chạy TRÊN
pipeline này sau khi B1-B5 xong: mỗi thí nghiệm = 1 `PortfolioConstants` variant qua
`register_overlay.py`, có tab + số + truy vết config tự động.
