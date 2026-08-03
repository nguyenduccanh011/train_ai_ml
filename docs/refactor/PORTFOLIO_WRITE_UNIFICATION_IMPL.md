# Triển khai: Hợp nhất TẦNG GHI danh mục + dọn rác (B3 lõi)

> Nối tiếp [PORTFOLIO_REGISTRATION_PIPELINE.md](PORTFOLIO_REGISTRATION_PIPELINE.md) (B1/B2/B4/B7 xong;
> **B3/B5 chưa**) và [SERVING_PANEL_TASKS.md](SERVING_PANEL_TASKS.md) (panel 1477 đã chốt).
> Ngày lập: 2026-08-02. Mục tiêu: **mọi model đăng ký overlay là có tab danh mục ĐẦY ĐỦ tự động**,
> qua MỘT writer hợp nhất map thẳng output của wheel `stock_ml.portfolio.run_portfolio` → DB —
> không còn nhờ script `hb_*` ad-hoc. Kèm dọn rác (~2.5GB local + 198 script research tracked).

---

## 0. Bối cảnh + phát hiện đã xác minh trực tiếp code (2026-08-02)

**P1 — Registrar tính đủ danh mục rồi VỨT ĐI.** `run_portfolio` trả sẵn
`equity / holdings / trades / skipped` ([portfolio/api.py:338-363](../../stock_ml/portfolio/api.py#L338-L363)),
nhưng cả `register_overlay.py` và `rescore_deploy_tiers.py` chỉ lấy `cagr/maxdd/conv_miss_frac/
offpanel_frac` UPSERT vào `leaderboard_nav` — bỏ hết `equity/holdings/trades/skipped`
([register_overlay.py:173-178](../../stock_ml/scripts/ops/register_overlay.py#L173-L178)).

**P2 — Writer tab danh mục = 5 script `hb_portfolio_*` gần trùng** (`daily/structure/new/top/fix`;
`compute_pending` xuất hiện ở `fix/top/new` + `hb_entry_scenarios`), dùng engine CỤC BỘ `prun()`
+ `import hb_112_meta_target`, và ghi thẳng `run_trades` (vi phạm bất biến BASE-only 0033). Đây
KHÔNG phải nguồn logic tái dùng — trích ra là kéo lại engine phân kỳ đã bị thay. Nguyên liệu
đúng nằm ở **dict output của wheel**.

**P3 — Không writer nào trong `scripts/ops/` ghi `run_equity/run_portfolio_daily/run_trades_overlay/
run_skipped/run_pending`.** Chỉ có: các script `hb_*` (r3line, ~20 run cũ) + migration một lần
[0033](../../stock_ml/db/migrations/versions/0033_run_trades_base_only.py) (chỉ chuyển
`run_trades_overlay`). ⇒ model mới có số `cagr_overlay` nhưng **tab danh mục rỗng**;
[runs.py](../../stock_ml/api/routes/runs.py#L398-L410) nuốt lỗi `except → return []` nên rỗng
trông như "không data" chứ không phải "chưa populate".

**P4 — Lệch tiền xử lý:** `is_nonstock` drop leg ETF/CCQ chỉ có ở
[rescore_deploy_tiers.py:107](../../stock_ml/scripts/ops/rescore_deploy_tiers.py#L107), KHÔNG có ở
`score_overlay` (đường board) ⇒ hai đường chấm khác tập leg (board cho CCQ conviction giả 0.5).

**P5 — `overlay_config_hash` băm danh sách field TAY** (`_HASH_FIELDS` +
"Keep in sync with the sandbox inputs", [overlay_scoring.py:120-126](../../stock_ml/db/overlay_scoring.py#L120-L126))
— drift risk; doc pipeline §2 yêu cầu rút từ `resolved.json`.

**P6 — `pending` (sổ chờ pullback) KHÔNG có trong output wheel.** `compute_pending` bản cũ là
**tái dựng để hiển thị** (ORM ghi rõ outcome/result_date POST-HOC,
[db/models/portfolio.py:133-136](../../stock_ml/db/models/portfolio.py#L133-L136)), dùng hằng
**hardcode champion** `PB_PCT=0.045, PB_WIN=40`. `sim.py::pend` là cash-settlement T+, không liên
quan. ⇒ muốn tab đầy đủ 100% phải thêm knob pullback vào `PortfolioConstants` rồi suy diễn trong wheel.

**P7 — Rác.** Local đã gitignore (~2.5GB: `.out` 800M+, `results/cache` 190M, `build/`, `whlchk/`,
`dist/`). Tracked: **198 file `hb_*.py`** trong r3line (không import bởi prod — coupling
`catalog.py`→r3line chỉ là comment provenance), + `nh_nav2k.py` (thước cũ), `tmp_verify_reg.py` (tạm).

---

## 1. Kiến trúc đích

```
run_template_experiment ──> run_trades (BASE, bất biến) + run_signals (full, chi tiết mã)
                                   │
  register_overlay.py / rescore_deploy_tiers.py
        └─ score_overlay(...)  → stock_ml.portfolio.run_portfolio(...)   [wheel THUẦN]
              trả: metrics + equity/holdings/trades/skipped/pending/held_by_date
                                   │  1 transaction / run
        └─ persist_overlay(conn, run_id, result, C, cfg_hash, panel_fp)  [writer MỎNG, MỚI]
                                   ▼
  run_equity | run_portfolio_daily | run_skipped | run_pending | run_trades_overlay
  leaderboard_nav.{cagr,maxdd,k,note,config_hash,panel_fp,conv_miss_frac,offpanel_frac}
                                   ▲
                     runs.py (tab danh mục + chi tiết mã) đọc đúng tầng
```

**Bất biến giữ nguyên:** `run_trades` = BASE-only; overlay trades → `run_trades_overlay`;
`run_signals` = tín hiệu FULL của chiến lược (trang chi tiết mã KHÔNG bị overlay lọc — overlay chỉ
chú giải qua `filled/skip_reason`). Wheel KHÔNG biết DB (writer mỏng riêng ở `stock_ml/db`).

---

## 2. Giai đoạn 0 — Dọn rác (độc lập)

⚠️ **`git clean -Xd` KHÔNG an toàn** (đính chính 2026-08-02): dry-run cho thấy nó xóa cả `.claude/`,
`.env` (secrets), `.vscode/`, `logs/`, `market_data/`, `portable_data/` — cấu hình/bí mật local,
KHÔNG phải rác build. Chỉ xóa TỪNG thư mục artifact đã biết, không blanket-clean.

| # | Việc | Lệnh / đối tượng | Guard |
|---|---|---|---|
| 0.1 | Xem trước | `git clean -Xnd` | ĐỌC kỹ danh sách; loại trừ `.claude/.env/.vscode/logs/market_data/portable_data` |
| 0.2 | Xóa artifact đích | `rm -rf build/ dist/ whlchk/ _champ_src/ **/__pycache__ **/.pytest_cache **/.ruff_cache`; `.out`/`results/cache` theo tên | `git status` sạch (0 tracked mất), `pytest` xanh |
| 0.3 | Xóa file chết tracked ✅ ĐÃ LÀM | `git rm .../r3line/nh_nav2k.py .../r3line/tmp_verify_reg.py` | grep xác nhận không ai import (chỉ self-ref) |

**Nghiệm thu:** artifact build/cache biến mất; `.env`/`.claude` CÒN NGUYÊN; test suite không đổi số pass.

---

## 3. Giai đoạn 1 — Wheel emit `pending` + `held_by_date` (phương án a)

**Đính chính so với thảo luận:** KHÔNG phải "surface thuần" — vì `pending` cần tham số pullback mà
wheel chưa có. Nhưng vẫn là ADD compute nhỏ, **sim.py KHÔNG đổi ⇒ golden metric byte-identical**.

**HỢP ĐỒNG DATA (quan trọng — quyết gốc "vì sao 2 bên lệch").** Wheel có 2 mặt phẳng data tách biệt
CÓ CHỦ ĐÍCH: (1) **rank/panel** = `market_frame` → `bundle CLO/LO/DIDX/CSm/R5` (1477 declared, CÓ
`low`); (2) **execution/NAV** = `price_frame` → `sym_close` (mã đã trade, chỉ `close`). `hb` cũ KHÔNG
dùng `PortfolioContext`: nó tự `SELECT ... FROM ohlcv` riêng cho `compute_pending`
([hb:128](../../stock_ml/analysis/serving_blindspot/r3line/hb_portfolio_fix.py#L128)) — **mặt phẳng
thứ 3 ad-hoc**, đó là chỗ bẩn khiến 2 bên không thống nhất. `pending` là câu hỏi KHÔNG-GIAN-TÍN-HIỆU
(limit chờ khớp trên vũ trụ được rank) ⇒ thuộc mặt phẳng **panel**, PHẢI tái dùng `bundle CLO/LO/DIDX`
(data đã có sẵn). Không load ohlcv ad-hoc, không mở rộng `price_frame`. Tín hiệu off-panel → không
có dòng pending (đúng hợp đồng declared-universe; các dòng off-panel pending của `hb` là **bug tiềm
ẩn** — pending mã hb không rank/size nổi — KHÔNG tái lập).

**1.1** Thêm knob vào [PortfolioConstants](../../stock_ml/portfolio/constants.py):
```
pull_pct: float = 0.045   # pullback limit = close[signal] * (1 - pull_pct)   (mặc định = champion)
pull_win: int   = 40      # số phiên giao dịch cửa sổ chờ fill
```
Mặc định = số champion cũ (`PB_PCT/PB_WIN`) ⇒ tái lập y hệt bản hb khi cùng chiến lược.
**Đồng thời thêm `pull_pct/pull_win` vào tập hash (xem §5.2)** — knob vào identity, tránh cache va chạm.

**1.2** [api.py](../../stock_ml/portfolio/api.py#L330): port `compute_pending` (~30 dòng) thành hàm
thuần trong wheel, dùng ĐÚNG mặt phẳng panel: `signals` (đã là input), `low/close` từ **`bundle["LO"]`
/ `bundle["CLO"]` + `bundle["DIDX"]`** (map date→bar per-symbol; KHÔNG load ohlcv riêng), và
`held_by_date` (giá trị thứ 4 `run_sim` trả — hiện bắt vào `hbd` rồi bỏ). Mã signal ngoài panel
(`sym not in bundle["DIDX"]`) → bỏ qua (off-panel, đúng hợp đồng). Thêm 2 key vào dict return:
```
held_by_date=hbd,
pending=pending_rows,   # None khi emit_pending=False (chưa tính); [] = tính-nhưng-rỗng.
                        # persist_overlay(detail=True) RAISE nếu pending is None (chống xóa ngầm run_pending)
```
**⚠️ Cổng qua flag `emit_pending` (mặc định False).** `compute_pending` là vòng lặp
`mọi symbol × mọi ngày` (nested O(S×D), [hb_portfolio_fix.py:189-210](../../stock_ml/analysis/serving_blindspot/r3line/hb_portfolio_fix.py#L189-L210));
sandbox endpoint gọi `run_portfolio` TƯƠNG TÁC + golden chạy mỗi commit ⇒ KHÔNG được trả giá
pending trên hot path. Chỉ writer persist bật `emit_pending=True`. Golden path giữ `pending=[]`
⇒ metric byte-identical, không phát sinh chi phí.

**1.3** Bump `stock_ml_core` version (pin `requirements.txt` + `Dockerfile` như wheel 0.4.4);
re-pin golden fixtures.

**Rủi ro & guard:**
- `sim.py` không đụng ⇒ `nav/cagr/maxdd/holdings/trades` **không đổi**.
- **Guard bắt buộc:** `RUN_PORTFOLIO_GOLDEN=1 pytest stock_ml/tests/test_portfolio_golden.py` —
  metric byte-identical (chỉ thêm key, pending mặc định rỗng). Thêm assert riêng chạy
  `emit_pending=True`: `pending` shape khớp 9 cột `run_pending`.
- `low` KHÔNG còn là rủi ro: `bundle["LO"]` đã có sẵn ([panel.py:39](../../stock_ml/portfolio/panel.py#L39)) —
  không mở rộng bundle, không đụng interface `PortfolioContext`.
- **Fixture pending so trên TẬP PANEL, không byte-match toàn cục với hb — và đó là ĐÚNG.** hb đọc
  full ohlcv nên pending cả mã off-panel (bug tiềm ẩn); bản wheel chỉ pending mã panel. Guard:
  lọc pending cả hai vế về `sym ∈ bundle["DIDX"]` rồi so 1:1; ghi chú các dòng off-panel của hb là
  bug được sửa, KHÔNG phải hồi quy. **Materialize + commit fixture này TRƯỚC khi G4 archive r3line.**

---

## 4. Giai đoạn 2 — Writer hợp nhất `persist_overlay` (lõi B3)

**4.1 File mới** `stock_ml/db/overlay_persist.py` — một hàm, một transaction:
```
def persist_overlay(conn, run_id, result: dict, C, cfg_hash, panel_fp, note, *, detail: bool) -> None
```
**⚠️ `detail` phân tầng (thiếu ở bản thảo — chốt trước khi code).** Ghi ĐỦ 5 bảng cho MỌI run
board sẽ phình DB gấp nhiều lần: `run_portfolio_daily` = ~1600 ngày × K vị thế × HÀNG NGHÌN run
(`register_overlay --only-missing` quét cả board), phần lớn run rác không ai mở tab. Quy ước:
- `detail=False` (mặc định board rộng): CHỈ UPSERT `leaderboard_nav` (metrics) — giữ hành vi hiện tại.
- `detail=True` (run `pinned` / deploy-tier): đủ 5 bảng detail + metrics + `emit_pending=True` ở §1.2.

**⚠️ Sở hữu transaction.** `register_overlay` loop hiện tự `con.commit()/con.rollback()` mỗi run
([register_overlay.py:171-178](../../stock_ml/scripts/ops/register_overlay.py#L171-L178)).
`persist_overlay` KHÔNG được commit nội bộ (nếu không double-commit / phá rollback-on-error của
loop). Quy ước: hàm chỉ `DELETE`+`INSERT`+`UPSERT` trên `conn`, **caller commit**. Nếu lỗi giữa
chừng → exception nổi lên, caller `con.rollback()` (loop đã có).

**⚠️ PHÁT HIỆN KHI CODE (2026-08-02): `holdings`/`skipped` của wheel KHÔNG map trực tiếp — đã sửa
tại NGUỒN (wheel), không vá ở writer.** Sim cũ emit `holdings` 9-field (thiếu `entry_weight`,
không có dòng `is_exit`, dư `prio`); `skipped` 3-field (thiếu `signal_date/pnl_pct/conv`). Đã làm
giàu wheel để emit ĐÚNG schema DB (display-only, NAV không đổi — golden 6/6 byte-identical):
- `sim.py`: `holdings` → 11-field khớp `run_portfolio_daily` + dòng `is_exit` (exit_today: "signal"/
  "preempt"/"risk_off"); `entry_weight=invested/nav_at`.
- `api.py`: `skipped` → 6-field (thêm `sigd`, `pnl_pct=leg.net`, `conv`); mọi lý do skip
  (conv/ret7/overshoot/liqcol) đều ghi — `run_skipped` nay là SUPERSET của hb (hb chỉ conv_skip).
⇒ `persist_overlay` là mapper 1:1 thuần (`_detail_rows`), không tính toán/engine.

Trong 1 transaction do caller commit (`detail=True`: 5×`DELETE WHERE run_id` → 5×`execute_values
INSERT` → UPSERT `leaderboard_nav`; `detail=False`: chỉ UPSERT). Cột lấy CHÍNH XÁC từ ORM
[db/models/portfolio.py](../../stock_ml/db/models/portfolio.py):

| Bảng | Nguồn `result[...]` | Cột ghi |
|---|---|---|
| `run_equity` | `equity` (df) | `date, nav, cash, exposure, n_positions` |
| `run_portfolio_daily` | `holdings` | `date, symbol, weight, entry_weight, unreal_pnl, entry_date, days_held, is_new, is_exit, exit_reason, conv` |
| `run_trades_overlay` | `trades` (df) | `symbol, entry_date, entry_price, exit_date, exit_price, holding_days, pnl_pct, exit_reason, conv, prio` |
| `run_skipped` | `skipped` | `symbol, signal_date, entry_date, pnl_pct, conv, skip_reason` |
| `run_pending` | `pending` | `date, symbol, signal_date, days_waiting, limit_price, ref_price, pct_to_limit, outcome, result_date` |
| `leaderboard_nav` | metrics | UPSERT `cagr_overlay, maxdd_overlay, overlay_k, overlay_note, overlay_config_hash, overlay_panel_fp, conv_miss_frac, offpanel_frac, computed_at` |

- **`trades` → `run_trades_overlay`** (KHÔNG `run_trades`; giữ bất biến 0033). `conv/prio` lấy từ df
  `trades` (wheel trả có, khác các dòng migrate legacy NULL).
- Gộp **chỉ `_UPSERT` SQL** (đang copy ở [register_overlay.py:40-55](../../stock_ml/scripts/ops/register_overlay.py#L40-L55)
  và [rescore_deploy_tiers.py:61-72](../../stock_ml/scripts/ops/rescore_deploy_tiers.py#L61-L72)) về đây — một định nghĩa.
  `build_note` KHÔNG gộp được: rescore build note giàu hơn (tier label + `constants` dict + panel + base,
  [rescore:110-113](../../stock_ml/scripts/ops/rescore_deploy_tiers.py#L110-L113)) khác `build_note(C)` của register.
  ⇒ `note` là THAM SỐ (caller tự dựng), hai builder giữ nguyên tại caller.

**4.2 ✅ ĐÃ LÀM.** 2 script bỏ `_UPSERT` riêng, gọi `persist_overlay(...)` rồi `con.commit()`;
`score_overlay` thêm `emit_pending`. `register_overlay`: `--detail/--no-detail`, mặc định
`detail = run đơn hoặc --pinned` (batch board = metrics-only). `rescore_deploy_tiers`: luôn
`detail=True` + `emit_pending=True`.

**Rủi ro & guard:**
- Ghi DB thật. Idempotent theo `overlay_config_hash` (caller-side skip). **✅ Smoke đã chạy
  (rollback, không mutate):** run pinned mẫu → 5 bảng populate (equity 1641 / daily / trades_overlay /
  skipped 610 / pending 20332), NAV cuối `run_equity` == `result.nav`, `run_trades` overlay-reason
  leak = 0, hash nhất quán. Golden module-parity 6/6 byte-identical (writer + wheel-emit không đụng NAV).
- ⚠️ **pending có thể RẤT lớn** (run mẫu degenerate = 20332 dòng/run) → tái khẳng định `detail=False`
  cho board rộng là bắt buộc; chỉ pinned/tier ghi detail.

---

## 5. Giai đoạn 3 — Thống nhất tiền xử lý + danh tính

**5.1** Đẩy `is_nonstock` drop vào `score_overlay`
([overlay_scoring.py:168-182](../../stock_ml/db/overlay_scoring.py#L168-L182)) — MỘT chỗ, cả board
lẫn deploy-tier cùng tập leg. Bỏ đoạn drop lặp
[rescore_deploy_tiers.py:107](../../stock_ml/scripts/ops/rescore_deploy_tiers.py#L107).
- **Guard:** in delta số board — chỉ đổi ở run vốn có leg CCQ; ghi lại (re-baseline có chủ đích + sha).

**5.2 ✅ ĐÃ LÀM — đính chính hướng:** `resolved.json` là artifact ENGINE/serving
(`build_resolved_config`), KHÔNG có trong đường overlay (`score_overlay` nhận `PortfolioConstants C`
trực tiếp) ⇒ không rút hash từ đó được. Cách clean đúng: `overlay_config_hash` nay derive từ
**`dataclasses.asdict(C)` (TOÀN BỘ field)** thay `_HASH_FIELDS` tay — knob mới tự vào identity, hết
drift vĩnh viễn. Phụ: bắt được cả `margin` (ảnh hưởng preempt/NAV) vốn bị `_HASH_FIELDS` bỏ sót =
latent bug đã sửa.
- **Guard đã chạy:** hash ổn định 2 lần cùng config; đổi `k`/`margin`/bất kỳ knob → hash đổi; 32-char.
- **⚠️ Re-baseline 1 lần:** đổi công thức hash → `config_hash` toàn board đổi (điểm số KHÔNG đổi vì
  default không đổi); lần `register_overlay` tới re-score rồi ổn định (chủ ý §9).

---

## 6. Giai đoạn 4 — Fail-loud tầng đọc + archive research

**6.1 ✅ ĐÃ LÀM.** [runs.py](../../stock_ml/api/routes/runs.py) `portfolio/day` + `portfolio/equity`:
thêm `logger`; success trả cờ `"populated": bool(rows)`; `except` nay `logger.exception(...)` +
`raise HTTPException(500)` thay vì `return []` mù (0-rows = chưa populate; exception = lỗi thật).
Inner `_has_ov` probe giữ nguyên (graceful fallback run_trades). Compile + ruff xanh.
- **Còn lại:** rebuild api (`docker compose build api`) + FE đọc cờ `populated` (ngoài phạm vi code).

**6.2 ✅ ĐÃ LÀM (git rm).** Xóa toàn bộ `stock_ml/analysis/serving_blindspot/r3line/` (198 file) —
**rút 0 logic** (writer lấy nguyên liệu từ wheel). Khôi phục qua git history nếu cần:
`git checkout fbf7a456 -- stock_ml/analysis/serving_blindspot/r3line/` (sha ngay trước khi xóa).
Provenance khác: `variants.py` docstring trỏ git 9e37efd7 cho `hb_deploy_*`.
- **Fixture pending:** KHÔNG chặn — pending đã verify bằng unit test deterministic
  (`test_portfolio_pending.py`, 5 ca) + DB smoke (20332 dòng đúng schema); nguồn hb vẫn trong git
  history. Chọn không dựng fixture-đối-chiếu-hb (đòi full research env), thay bằng port faithful +
  unit test.
- **Guard đã chạy:** không module nào import `hb_*`/r3line (chỉ docstring `catalog.py`/`variants.py`;
  `nh_nav2` external, không trong r3line); `pytest --collect-only` = 339 test OK; fast tests xanh.

---

## 7. Bảng guard tổng

| Bước | Test bắt buộc | Ngưỡng |
|---|---|---|
| G0 | `git clean -Xnd` dry-run + `pytest` | 0 file tracked bị xóa; test không đổi |
| G1 | golden (pending=[]) + assert `emit_pending=True` | metric byte-identical; `pending` (scope `sym∈panel`) khớp hb champion 1 lần |
| G2 | đăng ký champion `detail=True` 2 lần + đọc tab; board `detail=False` | lần 2 skip; NAV cuối == `cagr_overlay`; board chỉ ghi metrics; run_trades không lẫn overlay |
| G3 | delta board + hash ổn định | chỉ run có CCQ đổi; hash bền qua 2 lần chấm |
| G4 | grep import + api rebuild | 0 import runtime r3line; endpoint có cờ `populated` |

---

## 8. Thứ tự & phụ thuộc

`G0 (độc lập) → G1 → G2 → G3 → G4`. G1 chặn G2 (writer cần key `pending`/`held_by_date`).
Rủi ro cao nhất = G1 (đụng wheel) nhưng đã cô lập là ADD (sim không đổi) ⇒ golden bảo vệ.
G2 ghi DB thật → dựa golden + idempotent `config_hash` (theo quyết định: không backup thủ công).

**Ước lượng:** G0 ~15ph · G1 ~0.5 ngày (port pending + re-pin golden) · G2 ~0.5 ngày ·
G3 ~2h · G4 ~1.5h.

---

## 9. Quyết định mở / ngoài phạm vi

- **pull_pct/pull_win per-strategy:** mặc định = champion (0.045/40). Nếu chiến lược khác dùng
  pullback khác, phải set qua `PortfolioConstants` trong config chiến lược đó (mỗi `config_hash`
  một bộ). Chiến lược không-pullback: `pending` rỗng — tab degrade sạch.
- **17 run OUTPUT tab cũ (populate bởi hb):** GIỮ làm sử liệu đến khi re-register qua writer mới;
  không xóa tab.
- **Hợp đồng data pending (CHỐT):** pending sống trên mặt phẳng panel (`bundle CLO/LO/DIDX`), off-panel
  → không pending. Không săn champion `offpanel≈0`; fixture G1 so trên tập panel (xem §3).
- **`detail` flag (còn phải chốt lúc code `main()`):** khuyến nghị — luật ngầm `--run-id`/`--pinned`
  → `detail=True`, batch board (`--only-missing`/`--limit`) → `detail=False`; THÊM `--detail/--no-detail`
  override để chặn ca `--pinned --limit` lớn phình `run_portfolio_daily`. Chốt theo thói quen batch thực tế.
- **Ngoài phạm vi:** nghiên cứu exposure cấp danh mục (regime-sizing / VN30F-hedge) — chạy TRÊN
  pipeline này sau G0-G4, mỗi thí nghiệm = 1 `PortfolioConstants` variant qua `register_overlay`.
