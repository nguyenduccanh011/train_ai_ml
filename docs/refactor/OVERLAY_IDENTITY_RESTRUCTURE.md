# Overlay Identity Restructure — một sổ Stage-2 phải tự khai nó là chiến lược nào, trên panel nào

> Tách **RUN (tín hiệu Stage-1)** khỏi **OVERLAY (sổ danh mục Stage-2)**. Hôm nay mọi bảng lưu
> overlay đều khoá theo `run_id` một mình, nên chấm lại một run dưới chiến lược thứ hai **xoá**
> chiến lược thứ nhất, và trang danh mục chỉ hiện được một sổ cho mỗi run.
>
> Trạng thái: **ĐANG TRIỂN KHAI** (bắt đầu 2026-08-03). Liên quan:
> `CONVICTION_UNIVERSE_UNIFICATION.md` (panel), `PORTFOLIO_LAYER_UNIFICATION.md` (một engine),
> `PORTFOLIO_REGISTRATION_PIPELINE.md` (BASE-only invariant).

---

## 1. Bằng chứng (đo trên board 2026-08-02/03)

| Sự việc | Số đo |
|---|---|
| Run có số overlay | **2551** (2549 đã re-score trên panel 1477, `overlay_panel_fp=4421162:2026-07-31:…`) |
| Run có **sổ chi tiết** (`run_equity`) | **30** — 1,2% |
| Thời điểm ghi sổ vs ghi số | sổ **12/07–29/07**, số **02/08 00:27–12:13** ⇒ **sổ thuộc panel CŨ** |
| Ví dụ của chủ dự án | `template/x2_struct_to_k10_cs5ma50_t2ret7g-69338138`: nav 02/08 00:32, sổ 26/07 19:24 |
| Nhân bản Stage-1 | mỗi biến thể `x2_struct_to_*` giữ **96.951 dòng `run_signals`** riêng dù tín hiệu y hệt; `run_signals` = **96 GB** |
| Biến thể mồ côi | `x2_struct_to_k16preempt_cssize`: `run_trades`=0 nhưng vẫn còn sổ |

Nguyên nhân trực tiếp của "sổ khác panel với số": `register_overlay.py` mặc định
`detail=False` cho batch rộng → chỉ upsert `leaderboard_nav`, **không** ghi lại 5 bảng chi tiết.
Không cột nào trên bảng chi tiết nói nó sinh ra từ panel/cấu hình nào, nên API không thể phát
hiện, và trang vẽ im lặng.

## 2. Khuyết tật gốc

Một sổ Stage-2 được xác định bởi **(run_id, cấu hình overlay, panel)**. Hệ **đã biết** bộ ba này —
sandbox cache khoá `(run_id, config_hash)`, `overlay_config_hash`/`overlay_panel_fp` là cột thật —
nhưng **mọi bảng lưu đều khoá `run_id`**:

- `leaderboard_nav` PK `run_id`, ghi `ON CONFLICT (run_id) DO UPDATE`
- 5 bảng chi tiết ghi `DELETE FROM {tbl} WHERE run_id=%s` rồi INSERT (`overlay_persist.py`)

⇒ `overlay_config_hash` chỉ là nhãn "ai ghi cuối cùng", không phải khoá. Vì khoá sai, hệ đẻ ra
**4 cơ chế song song** cho cùng một nhu cầu, không cái nào là chuẩn:

| | cơ chế | vấn đề |
|---|---|---|
| A | 1 **template run** cho mỗi chiến lược (`…_k10_cs5ma50_t2ret7g`) | nhân bản toàn bộ Stage-1 (96 GB); chiến lược lẫn vào bảng xếp hạng tín hiệu |
| B | **run con `overlay/…`** + `parent_run_id`, mượn BASE qua `base_run` | đúng hình dạng nhưng dùng `leaderboard_runs` làm bảng chiến lược |
| C | `overlay_config_hash` + endpoint sandbox | định danh ĐÚNG nhưng không lưu, chỉ số tổng |
| D | serving `portfolio.db` khoá `bundle_id = tier_id + "_live"` | nối chuỗi; `summary` không có panel_fp/config_hash |

## 3. Thiết kế đích

### 3.1 Hai định danh, đừng trộn

| tên | công thức | dùng để |
|---|---|---|
| `overlay_key` | `md5(asdict(PortfolioConstants))` — **KHÔNG** gồm panel | **CHIẾN LƯỢC là cái gì.** Khoá chính, ổn định qua mọi lần sync data, là thứ hiện trên dropdown |
| `scoring_hash` | `md5(asdict(C) + panel_fp)` | **LẦN CHẤM này là gì.** Idempotency: đã chấm đúng cấu hình NÀY trên panel NÀY ⇒ bỏ qua |
| `panel_fp` | `rows:max_date:Σclose:Σvolume` | **Provenance + phát hiện cũ.** Row có `panel_fp != panel hiện hành` ⇒ **stale**, API phải nói ra |

⚠️ **Không đưa panel vào khoá chính.** `panel_fp` đổi mỗi ngày sync ⇒ nếu nó nằm trong PK thì mỗi
ngày đẻ một hàng mới, dropdown đầy bản trùng, sổ cũ thành mồ côi. Panel là **vintage dữ liệu**, không
phải danh tính chiến lược. Hàm `overlay_config_hash(C, panel_fp=None)` hiện có đã hỗ trợ cả hai.

### 3.2 Nhà của hàm định danh = wheel

`overlay_config_hash` đang ở `stock_ml/db/overlay_scoring.py` — **ngoài** wheel, nên serving không
import được và phải tự chế. Chuyển vào `stock_ml/portfolio/identity.py` (cùng nhà với
`PortfolioConstants`), `stock_ml.db` giữ lại một dòng re-export. ⇒ wheel **0.4.5**; hai repo tính ra
cùng một `overlay_key` cho cùng một chiến lược, không cần thoả thuận gì thêm.

### 3.3 Schema

```sql
CREATE TABLE run_overlay (              -- MỘT hàng = một (run, chiến lược)
  run_id       text  REFERENCES leaderboard_runs(run_id) ON DELETE CASCADE,
  overlay_key  char(32),
  label        text,          -- tên người đọc được ("K10 · sàn 5 tỷ")
  config       jsonb,         -- override đầy đủ, KHÔNG chỉ hash: đọc lại được, tái lập được
  base_run     text,          -- mượn BASE trades từ run này (cơ chế B), NULL = tự có
  cagr, maxdd, nav, years double precision, n_trades, k integer,
  scoring_hash char(32), panel_fp text,
  conv_miss_frac, offpanel_frac double precision,
  has_detail   boolean,       -- 5 bảng chi tiết có dữ liệu cho key này không
  computed_at  timestamptz,
  PRIMARY KEY (run_id, overlay_key)
);
```
5 bảng chi tiết (`run_equity`, `run_portfolio_daily`, `run_trades_overlay`, `run_skipped`,
`run_pending`) thêm cột `overlay_key char(32) NOT NULL` + index `(run_id, overlay_key, date)`.

**`leaderboard_nav` giữ nguyên vai trò**: nó là **thước xếp hạng của board** — MỘT cấu hình tham
chiếu cho mọi run, để bảng xếp hạng so **chất lượng tín hiệu** chứ không so chính sách danh mục
(`register_overlay.py` docstring). Từ nay chỉ ghi khi `overlay_key == REFERENCE_KEY`; một lần chấm
chiến lược riêng **không còn đè** được số board. Đây là tách vai trò, không phải nợ kỹ thuật.

### 3.4 Hệ quả cho 4 cơ chế

- **A bị xoá**: biến thể chỉ khác chính sách danh mục không được là một `leaderboard_runs` nữa.
  27 biến thể `x2_struct_to_*` phải audit: cái nào có `run_trades` **giống hệt** một run cha thì
  gộp thành overlay của cha; cái nào khác base (engine knob thật) thì **giữ nguyên là run** —
  đo được: `x2_struct_to` 2485 lệnh vs `…_k10_cs5ma50` 2491 ⇒ **không phải** thuần overlay, cấm gộp máy móc.
- **B thành trường hợp riêng**: `base_run` là cột của `run_overlay`, không cần đẻ `leaderboard_runs` row.
- **C được lưu**: sandbox "Lưu chiến lược này" ⇒ ghi một hàng `run_overlay` + sổ.
- **D đồng bộ**: serving ghi `overlay_key`/`scoring_hash`/`panel_fp` vào `portfolio.db.summary`;
  mỗi tier trong `tiers.yaml` CHÍNH LÀ một overlay config ⇒ parity train↔serving so được **theo
  từng chiến lược**, không phải so tay.

## 4. Thứ tự triển khai

| # | Bước | Nghiệm thu |
|---|---|---|
| B1 | Wheel: `stock_ml/portfolio/identity.py` (`overlay_key`, `scoring_hash`) + re-export ở `stock_ml.db.overlay_scoring`; bump **0.4.5** | `overlay_key(C)` == md5 cũ với `panel_fp=None`; golden 11/11 vẫn PASS |
| B2 | Migration `0035_run_overlay`: tạo bảng + thêm `overlay_key` vào 5 bảng chi tiết (default `'legacy…'`) + index | `alembic upgrade head` sạch; hàng cũ mang key `legacy` |
| B3 | `overlay_persist.persist_overlay(..., overlay_key, label, base_run)`: upsert `run_overlay`; DELETE/INSERT chi tiết theo **(run_id, overlay_key)**; `leaderboard_nav` chỉ ghi khi key == REFERENCE_KEY | chấm 2 chiến lược trên cùng run ⇒ **2 hàng, 2 sổ**, số board không đổi |
| B4 | API: `GET /runs/{id}/overlays`; `portfolio/equity|day|skipped|trades` nhận `?overlay=`; trả `stale` khi `panel_fp` lệch | gọi không tham số vẫn chạy (mặc định: reference, hoặc sổ duy nhất) |
| B5 | UI `portfolio.html`: dropdown chiến lược + banner panel/stale; `model-details.html` truyền `overlay` | trang của chủ dự án hiện đủ chiến lược; sổ lệch panel **báo đỏ** thay vì vẽ im lặng |
| B6 | Ops `rescore_books.py`: chấm lại `--detail` cho tập tier/pinned trên panel hiện hành, rồi **xoá** hàng `legacy` còn sót | 0 hàng `legacy`; mọi sổ có `panel_fp` == panel hiện hành |
| B7 | Data migration: 6 run `overlay/*` → `run_overlay` của run cha (kèm sổ); giữ `leaderboard_runs` row ở trạng thái `superseded` để link cũ không chết | `/runs/template%2F_dyn300_onerun…/overlays` trả **6** chiến lược |
| B8 | Audit 27 `x2_struct_to_*`: báo cáo cái nào gộp được (base trùng) / cái nào là run thật. **KHÔNG tự xoá** | báo cáo + số GB thu hồi được, chờ chủ dự án duyệt |
| B9 | Serving: `portfolio.db.summary` thêm `overlay_key/scoring_hash/panel_fp`; `tiers.yaml` thêm `label` | mỗi sổ khách tự khai chiến lược + vintage panel |

**Bất biến trong suốt quá trình**
- Không xoá dữ liệu trong migration. Xoá là bước ops riêng, có tên, có thể chạy `--dry-run`.
- Sổ SAI nguy hiểm hơn không có sổ: hàng `legacy` (không biết panel/cấu hình) phải **biến mất hoặc
  bị đánh dấu stale**, không được im lặng phục vụ.
- Golden champion phải PASS sau mỗi bước (`RUN_PORTFOLIO_GOLDEN=1`).
- `run_trades` vẫn là **BASE-only** (0033); overlay trades ở `run_trades_overlay`.

## 5. Nhật ký thực thi

| Ngày | Bước | Kết quả |
|---|---|---|
| 2026-08-03 | — | tài liệu này |
| 2026-08-03 | B1 | `stock_ml/portfolio/identity.py` + export; `overlay_config_hash` thành alias; wheel **0.4.5** build + cài ở serving. **Kiểm chéo hai repo**: tier floor-5 full-history ra `5daa7e25…` ở CẢ HAI |
| 2026-08-03 | B2 | `alembic upgrade head` → `run_overlay` + `overlay_key` trên 5 bảng; 1.253.465 dòng sổ cũ mang khoá `legacy` |
| 2026-08-03 | B3 | `persist_overlay` khoá `(run_id, overlay_key)`, trả `overlay_key`; `leaderboard_nav` chỉ ghi cho REFERENCE_KEY; `register_overlay` có `--label`, `already_scored` thay `existing_hash` (metrics-only KHÔNG còn thoả `--detail`) |
| 2026-08-03 | B4 | `GET /runs/{id}/overlays` + `?overlay=` cho equity/day/skipped/trades/portfolio-fate; cờ `stale` = `panel_fp` sổ ≠ panel hiện hành |
| 2026-08-03 | B5 | `portfolio.html`: dropdown chiến lược (giữ trong URL), dải provenance (K/T+/sàn/panel/ngày chấm), banner đỏ khi `stale` hoặc sổ `legacy` |
| 2026-08-03 | B7 | **Dry-run bắt được va chạm khoá**: dùng `reference_config` làm nền thì "Cân bằng — veto big-quiet" trùng khoá với "sàn 5 tỷ" (nền đã pin `liqcol_adv252_ty=0.0`). Nền đúng = `PortfolioConstants` mặc định, **khớp serving**. Sau khi sửa: 6 khoá phân biệt |
| 2026-08-03 | B8 | Audit 27 run `x2_struct_to_*`: **18 run gộp được** (trùng CẢ tín hiệu lẫn base trades) thành 3 nhóm; **3 run mồ côi** (0 base trades) — chờ chủ dự án duyệt, không tự xoá |
| 2026-08-03 | B9 | `portfolio.db.summary` + `overlay_key/scoring_hash/panel_fp` (có `_migrate` ALTER cho DB đang sống); runner đóng dấu mỗi sổ; serving test 93/93 PASS |
| 2026-08-03 | B7 (chạy thật) | 6/6 gộp OK. **CAGR trùng khít số cũ** của các run `overlay/*` (171.5 / 196.6 / 203.4 / 166.5 / 161.7 / 236.0) ⇒ tái cấu trúc **không đổi một con số nào**, chỉ đổi định danh. `leaderboard_nav` của 2 run cha **không bị đụng** (`computed_at` vẫn 02/08) — bất biến "tier không đè số board" hoạt động |
| 2026-08-03 | (phát sinh) | Rebuild + restart image API. Lộ ra `docker-compose.yml` vẫn pin **panel 488 cũ** (`market_golden_pin_20260729.duckdb`, `date_hi 2026-07-08`) cho container trong khi board chấm trên 1477 ⇒ sandbox và board đo trên **hai vũ trụ khác nhau**, và `_panel_now()` trả None nên cờ stale im lặng thành "sạch". Đã trỏ về đúng artifact khai báo; `stale` đổi thành **tri-state** (`None` = không kiểm được ≠ `false` = đã kiểm và sạch), UI có banner riêng cho trạng thái đó |
| 2026-08-03 | (nghiệm thu) | `/overlays` của `_dyn300_onerun` trả **6 chiến lược** (5 tier + tham chiếu, có cờ `is_reference`); `portfolio/equity?overlay=` cho NAV khác nhau (×711 floor5 vs ×1480 K6); `portfolio/day` tách đúng holdings theo từng sổ |
| 2026-08-03 | B6 | Chấm lại 24 sổ `legacy` dưới chiến lược tham chiếu: **19 OK, 5 lỗi "0 base trades"** — run OUTPUT-only mà migration 0033 đã tách BASE ra, **không tái tạo được**. Đáng chú ý: 12 biến thể `x2_struct_to_*` cùng base cho **cùng CAGR 100.6%** dưới cùng cấu hình ⇒ bằng chứng trực tiếp cho kết luận B8 (chúng chỉ khác chính sách danh mục) |
| 2026-08-03 | B6 (purge) | `--purge-legacy` được sửa để **không xoá mù**: chỉ xoá sổ cũ của run đã có bản khai đầy đủ, và nhận ra `overlay/*` đã "chuyển nhà" sang run cha (kiểm bằng chính khoá trong note, không đoán). Xoá **1.043.722** dòng trên 25 run; **giữ** 5 run không tái tạo được — trang hiện banner đỏ "chưa khai chiến lược" cho chúng, trung thực hơn là xoá mất bản ghi duy nhất |
| 2026-08-03 | (nghiệm thu cuối) | URL của chủ dự án (`x2_struct_to_k10_cs5ma50_t2ret7g`) giờ trả 1 chiến lược **có tên**, `panel_fp` = panel hiện hành, `stale=false` **đã kiểm thật** (trước đó là sổ 26/07 trên panel cũ, im lặng). `run_overlay` 25 sổ / 19 run, `leaderboard_nav` vẫn nguyên 2551 dòng |

## 6. Dọn nốt (2026-08-03, đợt 2) — bỏ ngoại lệ ngầm

| Việc | Kết quả |
|---|---|
| **Xoá "khoá ma"** | `legacy` từng là quy ước ngầm: dòng tồn tại trong 5 bảng chi tiết nhưng không có mặt ở bảng chỉ mục nào, nên mọi người đọc phải tự biết. Nay 5 sổ không tái tạo được có **hàng `run_overlay` tường minh** (`label` nói rõ, `config`/`panel_fp` = NULL **có chủ ý**). API bỏ hẳn nhánh fallback: run không có hàng chỉ mục ⇒ trả **rỗng**, không đọc lén sổ vô danh nữa (`populated=false`) |
| **`stale` tri-state** | `null` = KHÔNG kiểm được (panel NULL hoặc không đọc được artifact) ≠ `false` = đã kiểm và sạch. Đã xác nhận trên API sống: sổ vô danh trả `null`, 6 tier trả `false` |
| **14 run trùng → `state='retired'` + `parent_run_id`** | Bảng xếp hạng thôi hiển thị 12 "mô hình" vốn là một (dashboard đã lọc sẵn `state !== 'retired'`; model-details đã đọc Stage-1 từ `parent_run_id`). **Không xoá dòng nào** |

**Hai đính chính quan trọng so với phần trên:**

1. ⚠️ **Không phải chuyện dung lượng.** Đo lại: họ `x2_struct_to_*` chiếm **2.6M/368M dòng `run_signals` = 0,71%** (~0,7 GB trong 96 GB). Ước lượng "gộp lại thu hồi phần lớn 96 GB" trước đó là **sai**. Lý do gộp là **rõ ràng khái niệm**, không phải tiết kiệm chỗ — và vì thế **không được xoá** dữ liệu không tái tạo được để đổi lấy 0,7 GB.
2. ⚠️ **`superseded` là cờ SAI.** Nó do aggregator **tự suy ra** ("bản cũ hơn của cùng tên") và dựng lại mỗi lần tổng hợp, nên ghi nghĩa khác vào đó vừa chồng lấn ngữ nghĩa vừa **bị hoàn tác lặng lẽ**. Cờ đúng là vòng đời `state='retired'` — đã có sẵn, dashboard đã lọc, và vẫn cho phép người xoá thật về sau qua đường bulk-delete có xác nhận. (Lần chạy đầu đã lỡ đặt `superseded`; đã hoàn tác 14 dòng và sửa script.)

### 6.1 Vá hồi quy: link cũ mất tab "Danh mục" (chủ dự án phát hiện)

`model-details.html?run_id=overlay/_dyn900_k16` **không hiện tab Danh mục** sau khi gộp. Đúng như
báo cáo, và là **lỗi do đợt này gây ra**: §5 tuyên bố "link cũ vẫn chạy" nhưng chỉ đúng cho trang
model-details, không đúng cho tab danh mục.

Nguyên nhân: trang quyết định hiện tab bằng cách hỏi `/portfolio/equity` — mà run placeholder giờ
**không còn sổ của riêng nó** (đã chuyển sang run cha rồi purge). Rỗng ⇒ ẩn tab ⇒ đọc như "không có
dữ liệu" trong khi dữ liệu chỉ **đổi chỗ**.

Sửa: hỏi `/overlays` thay cho `/portfolio/equity` — vì một cuốn sổ giờ là `(run_id, overlay_key)`,
hỏi "run này có sổ không" là câu hỏi sai. Endpoint trả thêm `moved_to` khi run không có sổ riêng
nhưng có `parent_run_id` (kèm danh sách sổ của cha), nên link cũ đưa thẳng tới sổ thật. 6 hàng
`overlay/*` cũng chuyển `state='retired'` cho nhất quán với 14 run trùng: **chiến lược không phải
là một run**.

**Bài học:** khi khoá của một thực thể đổi, mọi chỗ hỏi "thực thể này có tồn tại không" bằng khoá
CŨ đều thành câu hỏi sai — và trả lời rỗng, tức im lặng, chứ không lỗi.

### 6.2 Vá UX bảng xếp hạng: không tìm thấy cái đã gộp

Phản hồi tiếp theo của chủ dự án: *"trên leaderboard không tìm được các phần đã gộp để kiểm tra"*.
Đúng — dọn xong mà không nhìn thấy thì không kiểm được, và ba lỗ hổng là thật:

| lỗ hổng | vá |
|---|---|
| Bảng không cho biết model nào có **mấy chiến lược** — model 6 sổ và model 0 sổ trông y hệt nhau. Đây chính là cách 12 dòng vốn là MỘT model tồn tại nhiều tháng mà không ai thấy | Cột **"Danh mục"**: `n_overlays` từ `run_overlay`, bấm vào đi thẳng trang danh mục |
| Dòng `retired` không nói **đi đâu** — "retired" đọc như "đã xoá" | `folded_into` = `parent_run_id` khi state=retired ⇒ hiện *"→ gộp vào &lt;run gốc&gt;"* ngay cạnh tên. **Điều kiện đầu tiên tôi viết sai** (buộc phải "không có sổ riêng"), nên nó ẩn đúng trên 12 dòng cần nó nhất — đã sửa: cờ gộp là `retired + có parent`, không liên quan tới việc còn giữ sổ tham chiếu hay không |
| Placeholder đã gộp có `n_overlays=0` ⇒ trông như "không có danh mục" | Lấy count của run cha làm dự phòng, và link trỏ về đó (`↗`) |

**Đã có sẵn, không cần thêm:** ô Search (client-side, UI nạp `limit=5000` nên phủ hết 2.560 run) và
bộ lọc `State → Retired`. Nên `_dyn300_onerun` tuy xếp hạng 1.677 vẫn tìm được bằng cách gõ "dyn300".
Kiểm tra căn cột: 27 `<th>` khớp 27 `<td>`.

### 6.3 Trang chi tiết mã chỉ thấy lệnh ĐÃ VÀO DANH MỤC

Phản hồi thứ ba: *"trang chi tiết mã hiển thị vài tín hiệu được vào danh mục chứ không đầy đủ tín
hiệu sinh ra cho mã đó"*. Đo trên `_dyn900_onerun` / **SHP**:

| tầng | số lượng |
|---|---|
| tín hiệu mua sinh ra | **1.049** |
| thành lệnh ở engine (`run_trades`, BASE) | **36** |
| vào danh mục — sổ tham chiếu (`run_trades_overlay`) | **1** |

Tín hiệu thì **đủ** (`/signals` trả 1.641 dòng, marker B/S vẽ đủ 1.049). Cái mất là **tầng giữa**:
`/runs/{id}/trades` **ngầm** ưu tiên tầng danh mục ("prefer the PORTFOLIO layer when this run has
one") và không có đường nào xem 36 lệnh gốc. Đúng lớp lỗi đợt này đang dọn: **một lựa chọn ngầm
không nói ra**.

Sửa: `?layer=auto|base|overlay` tường minh, response trả `layer` **thật sự dùng** (không phải cái
được hỏi — `auto` có thể rơi về base). `layer=overlay` mà rỗng thì trả rỗng, **không** âm thầm rơi
về base (36 lệnh engine mà dán nhãn "1 lệnh danh mục" là nói dối). Trang chi tiết có nút gạt
**"Lệnh vào danh mục | Lệnh gốc (engine)"** và thanh trạng thái ghi rõ tầng đang xem.

Nghiệm thu: `auto`→overlay 2.375 (SHP 1) · `base`→base 35.415 (SHP 36) · `overlay`→overlay 2.375 ·
`layer=xxx` → HTTP 400.

### 6.4 Mở placeholder thì lạc sang chiến lược khác (migration 0036)

Phản hồi thứ tư: *"mở `overlay/_dyn300_k6_liqcol` không thấy đủ chi tiết lệnh, có cần chạy lại
không?"* — **không cần chạy lại**. Placeholder này **chưa bao giờ có dữ liệu riêng** (0 dòng ở cả
`run_signals`/`run_trades`/`run_trades_overlay`/`run_equity`); nó luôn mượn lệnh gốc của run cha.
Sổ K=6 của nó (**1.507 lệnh**) nằm trên run cha dưới khoá `8d1c94a9`.

Thiếu là **mối nối ngược**: biết run cha (`parent_run_id`) nhưng không biết placeholder đó ứng với
**chiến lược nào** trong 6 chiến lược của cha ⇒ link cũ thả người đọc vào sổ mặc định (tham chiếu
board, 2.139 lệnh) thay vì sổ K=6 họ vừa bấm. Hai con số đều "đúng", chỉ là **khác chiến lược** —
kiểu sai khó phát hiện nhất.

Sửa bằng **cột**, không bằng chuỗi: migration `0036` thêm `run_overlay.source_run`. Suy ra từ
`overlay_note` thì hôm nay chạy được, mai mục — "chính sách nhét trong một chuỗi" chính là thứ đợt
này đang gỡ. `persist_overlay` ghi từ giờ; 6 sổ gộp trước đó backfill một lần
(`rescore_books --backfill-source`, đọc note lần cuối rồi thôi).

API trả thêm `moved_to_key`; trang chi tiết đổi nút thành **"📊 Danh mục: Tăng trưởng — K=6 + liqcol
5 tỷ →"** và nạp lệnh của **đúng** sổ đó thay vì bảng rỗng.

Nghiệm thu: `overlay/_dyn300_k6_liqcol` → `moved_to_key=8d1c94a9…` "Tăng trưởng — K=6", 1.507 lệnh.
Lệnh theo từng sổ khác nhau thật: K6 **1.507** · tham chiếu **2.139** · Cân bằng **2.156**.

**Chính sách chốt lại — vì sao KHÔNG gộp 18 run thành overlay "đúng nghĩa":** chính sách danh mục của chúng chỉ tồn tại trong **cái tên**, chưa từng được lưu, và sổ cũ đã bị dọn. Chế ra một `config` để gộp cho đẹp là **bịa provenance** — đúng thứ mà cả đợt tái cấu trúc này loại bỏ. Ai cần "K=16 preempt" thì đăng ký nó như một **overlay có tên** trên run gốc; giờ việc đó đã làm được, và đó mới là điểm của thiết kế mới.
