# Hợp đồng nâng cấp engine + tái cấu trúc di sản hệ cũ

> Lập 2026-07-30. Nguồn: 2 đợt audit multi-agent read-only trên repo này (8 + 8 agent: flow sản
> xuất, đường nâng cấp engine, overlay chắp vá, hợp đồng wheel, hiệu suất; rồi khảo cổ di sản
> tiền-DB), mỗi đợt qua 2 agent phản biện đối kháng. Mọi số dưới đây là **đo được**, trừ mục đánh
> dấu ESTIMATE.
>
> Trạng thái: **CHƯA triển khai bước nào trong repo.** Tài liệu này là bản mô tả việc cần làm, không phải
> bản ghi việc đã làm. **Ngoại lệ hạ tầng data (thượng nguồn):** sieutinhieu **đã** phục vụ giá điều chỉnh
> corporate-action (chủ dự án xác nhận 2026-07-31 — §13.7); phần phía repo còn lại (**bỏ clamp loader** +
> kiểm đường cong tính trước/sau bản vá) vẫn chưa làm.

> ## 🧭 ĐỌC TÀI LIỆU NÀY THẾ NÀO
>
> **Phần CHỈ DẪN (làm theo):** §11 (mục tiêu + hợp đồng) · §12 (mô hình 4 đối tượng) · §13 (quyết định
> 2026-07-31) · **§11.6 là bản THỨ TỰ THỰC THI DUY NHẤT**.
>
> **Phần BẰNG CHỨNG (tra cứu, KHÔNG làm theo):** §0–§10. Chúng được viết dưới mục tiêu **đóng băng số**
> mà chủ dự án đã **bác** ngày 2026-07-30. Số đo trong đó vẫn đúng và quý; **chỉ dẫn** trong đó thì
> không. Các câu đã chết đã được **gạch tại chỗ** và đánh dấu ⛔ — nếu bạn thấy một chỉ dẫn ở §0–§10
> **không** có dấu, hãy đối chiếu §11–§13 trước khi làm.
>
> ### 📌 LUẬT BIÊN TẬP — bắt buộc
> **Đổi quyết định thì phải sửa TẠI CHỖ câu cũ, không được chỉ thêm mục mới ở cuối.**
> Bằng chứng: audit 2026-07-31 đo được **7/12** chỗ sửa của vòng trước **không hạ cánh**, và lỗi nặng
> nhất — bật cổng sklearn/numpy **TRƯỚC** khi re-export, tức dừng **6/6 sổ khách** — đã **di cư nguyên
> vẹn** từ §9.9 (đã đóng dấu ⛔) sang §11.6 (bản được coi là chuẩn) trước khi bị phát hiện lại.
> Dấu ⛔ ở **đầu mục** không đủ: người thực thi copy phần **thân** vào task tracker. Gạch ngay tại dòng.

---

## 0. Tài liệu nào chi phối việc gì

| Tài liệu | Phạm vi | Trạng thái |
|---|---|---|
| `docs/refactor/PORTFOLIO_LAYER_UNIFICATION.md` | Hợp nhất logic overlay vào `stock_ml/portfolio` | **Xong 0-5.** Module là single-source-of-truth cho LOGIC. Không còn việc |
| `docs/refactor/PORTFOLIO_REGISTRATION_PIPELINE.md` | Tầng GHI overlay vào DB + tab danh mục | **B1/B4/B7 xong; B2/B3/B5 CHƯA; B6 tuỳ chọn.** Vẫn là kế hoạch đúng cho tầng ghi — thực thi tiếp, **kèm 4 sửa lỗi ở §7 dưới** |
| **Tài liệu này** | P0 vá chảy máu · hợp đồng nâng cấp engine · danh tính run · nhà cho overlay · mối nối di sản | Mới. Chi phối mọi việc **không** nằm trong 2 doc trên |
| `stock-serving/RESTRUCTURE_STRATEGY_LAYER.md` | Phía tiêu thụ: gộp tầng tín hiệu + tầng danh mục thành một đối tượng chiến lược | Mới, repo khác. Các mục CROSS-REPO ở đây khớp với nó |

Thứ tự thực thi tổng: **xem §11.6 — bản thứ tự DUY NHẤT (Phase 0–6)**, hợp nhất và **thay** mọi thứ tự cũ (§9.1/§9.4/§9.9 đã bị đánh dấu ⛔, giữ lại chỉ để truy vết). Mục tiêu nền + cơ chế bảo đảm: **§11** (thay R0/R2/R6 của §2.3). Mô hình đối tượng chuẩn (Engine/Model/Strategy/Bundle, bỏ tier): **§12**. Các quyết định 2026-07-31 (overlay ở khối `portfolio:`, `config_hash` rút từ `resolved.json`, bảng nền thị trường, batch tăng dần): **§13** — được §11.6 gắn vào Phase 1/5 (xem chú thích cuối §11.6).

> 🆕 **Chốt vòng 3 (2026-07-31) — §14 là bản mới nhất.** Danh mục **10 chiến lược** (thay mọi "6 sổ"/"9 chiến lược"); bỏ `nh_nav2` → NAV về `stock_ml.portfolio`; khớp float **exact + Linux-only**; re-export tĩnh **tách wavestruct**; M1–T4. Nơi nào trong §1–§13 nói khác các điểm này thì **§14 thắng**.

---

## 1. P0 — đang sai ngay lúc này, vá trước, không đổi kiến trúc

### 1.1 Bối cảnh thị trường đọc từ FILE theo CWD, bên trong wheel — RỦI RO CAO NHẤT

`stock_ml/src/backtest/engine.py:39-51` — `_load_vnindex(path="portable_data/vn_stock_ai_dataset_cleaned/context_features/symbol=VNINDEX/timeframe=1D/data.csv")`, đường dẫn **tương đối theo CWD**, và **trả `None` im lặng** khi thiếu (`:44-46`). Call site `:1326, 1349, 1376, 1487`, mỗi chỗ là `if _vni is not None:` **không có else**.

Đo được: file tồn tại ở host (320.417 byte, **bar cuối 2026-06-16**). Trong container serving thì `WORKDIR=/app`, Dockerfile COPY chỉ `serving/ webapp/ worker.py run_serving.py`, compose mount chỉ `bundles/ data/ results/ market_data/ serving/ webapp/` ⇒ **`portable_data` không được copy và không được mount**.

Ba bundle bị ảnh hưởng:
- `bundle_n2_2643_wavestruct_la05_lamp02_top150_2025-01-01_wf` (**model active trong `.env` của serving**): `signal_exit_hold_rs_scale=8.0`
- `bundle__dyn300_onerun_2025-01-01_wf` và `bundle__dyn900_onerun_2025-01-01_wf`: 8.0 **cộng** `signal_exit_skip_if_mkt_above_ma=300`

⇒ Cả ba knob **inert trong container, active trong mọi phép đo ở host**. Biểu hiện ra ngoài như phi tất định. **Hệ quả nặng nhất: bằng chứng "0.3.3→0.4.2 không drift" đã được thu trên code path mà production không thực thi.**

Ba tầng suy biến âm thầm cộng thêm:
- `engine.py:1342` `nan=0.0` cho rs; `:1371` `nan=0.0` cho mkt_trend; `mkt_skip` `nan=-1.0` = "không bullish" = **không bao giờ skip**.
- Bar cuối CSV là 2026-06-16 ⇒ ~6 tuần gần nhất **mất modulation thay vì báo lỗi**.
- Hai reader của **cùng một file** có ngữ nghĩa lỗi trái ngược: `engine.py:44-46` trả None im lặng, `experiment.py:1717-1730` **raise KeyError**.

Cùng loại: `_load_runscore` (`engine.py:57-70`, dùng ở `:1394`) đọc `results/_research_2429/runscore.parquet`, cũng CWD-relative.

**Việc cần làm.** Market context phải thành **input tường minh có version**: hoặc materialize VNINDEX vào bundle (hoặc vào `market_data/market.duckdb` — đã được mount), ghi bar cuối + checksum vào `manifest.json`; hoặc **fail loud**. Nguyên tắc: **knob nào có nguồn dữ liệu vắng mặt thì phải raise, không bao giờ suy biến về None/0.0.** Thống nhất ngữ nghĩa lỗi giữa hai reader.

**Guard.** Trong container: `docker exec <web> python -c "from stock_ml.src.backtest.engine import _load_vnindex; print(_load_vnindex())"` phải **raise** (hoặc trả dữ liệu), không được trả None. Sau khi vá, chạy lại so sánh 0.3.3↔0.4.2 **trong container** — kết quả cũ không còn giá trị.

### 1.2 `market.duckdb` là default theo CWD ⇒ "thị trường" là file nào tuỳ repo đang đứng

`experiment.py:1652-1653` (`_load_market_breadth`) và `:1692` (`_load_xsec_features`) đều default `duck="market_data/market.duckdb"`, và cả hai tới được từ `build_feature_frame` (`:3026`) — chính là hàm `stock-serving/serving/build_bundle.py` import từ wheel.

Đo được: hai file khác nhau trả lời cùng đường dẫn — `train_ai_ml/market_data/market.duckdb` **70.791.168 B** vs `stock-serving/market_data/market.duckdb` **23.867.392 B**. Và **3 trong 5 bundle deploy** có `git_sha="serving-build:0.3.3"`, tức self-build trong repo serving ⇒ dựng trên file 23.8 MB.

Universe breadth **không** được ghi vào bundle, dù `feature_scope` thì có (cả 5 bundle ghi `'traded'`). Chính CLAUDE.md §11 của serving xác định scope breadth là trục lệch parity lớn nhất (đo được 488-scope PF 5.25 vs 61-scope 5.84).

**Việc cần làm.** Nguồn breadth/xsec thành tham số tường minh, ghi vào manifest cạnh `feature_scope`, kèm checksum.

### 1.3 Fold-checkpoint: key là tập con thật sự của config đã sinh ra nó, và FILE THẮNG

`experiment.py:3334-3344`: `if fold_cache_path.exists(): pd.read_parquet(...); continue` — **bỏ hẳn `train_fold`**. Key = `(run_id, fold_label)`, với `run_id = f"tmpl_{template_id}_{_fp}"` và `_fp` là sha256 trên **đúng 11 field** (`run_template.py:288-303`): strategy, split, seed, feature_set, entry_features, exit_features, entry_target, exit_target, target, entry_model, exit_model.

Field mà `train_fold` **thật sự đọc** nhưng key **không** có: `cfg.entry_threshold`, `cfg.exit_threshold`, `cfg.signal_threshold`, `cfg.direction`, `cfg.model_mode`, toàn bộ `cfg.engine` (kể cả `engine.entry_ensemble*.features` — thứ chọn feature set của các ensemble head, `:2965-2979`), cộng `universe`, `universe_policy`, `market`, `data_source_dir`, và override `--symbols` (`run_template.py:271-278`).

**Chứng minh thực nghiệm** (read-only, template 1844 nạp từ Postgres live): config hiện tại → `fp = e885031bfc`, và `results/tmpl_1844_e885031bfc/folds/` tồn tại (6 parquet, mtime 2026-06-16 14:20). Đổi `entry_threshold` −1.9→−1.4 **và** `engine.entry_ensemble.features` **và** `universe_policy` → **vẫn ra `e885031bfc`**.

Cộng dồn: `_collect_provenance` (`run_template.py:57-90`, gọi ở `:455`) đóng dấu **git HEAD hôm nay** và `SELECT max(date) FROM ohlcv` của duckdb **hôm nay** ⇒ `leaderboard_runs.data_snapshot_date`/`git_sha` mô tả dữ liệu mà các dự đoán **chưa từng được tính trên**. Frame cũ sau đó chảy tiếp vào `run_signals`/`run_trades` qua `_persist_run_detail` (`:484`).

Phạm vi hiện hữu: `results/tmpl_3524_565c005aa3/folds/` chứa fold predictions của **bundle dyn300 ĐANG DEPLOY** (6 parquet, mtime 29/07 09:25). Trên đĩa có **5043** thư mục fold có fingerprint + **825** thư mục fold trần. `results/_algo_ab/chaseup_diag/folds/` có parquet 19/06, tức **trước** cả đợt resync 901 mã và patch corp-action. Và **85 script nghiên cứu** dưới `results/` gọi `run_experiment` với `run_id` viết tay không fingerprint (vd `results/_research_2429/micro_struct_diff.py:24` `run_id=f"micro_{label}"`).

Comment tại `run_template.py:283-287` cho thấy **đã bị lỗi này một lần** (gap_days 25→85) và chỉ vá đúng field đang nhìn.

**Việc cần làm.** Chọn một: (a) đưa **toàn bộ** config ảnh hưởng dự đoán vào `_fp_src`; hoặc (b) **bỏ read-through cache**, làm resumability tường minh bằng `--resume <run_id>`. Provenance phải đóng dấu từ **input thực dùng**, không phải HEAD hôm nay.

**Guard.** Với một template có fold cache trên đĩa: đổi `entry_threshold` trong DB → `_fp` **phải đổi** → chạy lại **phải** gọi `train_fold`. Assert `leaderboard_runs.data_snapshot_date` khớp `max(date)` của snapshot mà fold thực sự dùng.

### 1.4 `alembic upgrade head` mặc định migrate một SQLite cũ

`alembic.ini`: `sqlalchemy.url = sqlite:///results/leaderboard.db`. `stock_ml/db/migrations/env.py:26-34` chỉ override khi có `DATABASE_URL`, **không import dotenv**. Trong container api thì Postgres thắng (verified `alembic_version = '0029'`). Từ shell host làm đúng theo `docs/DEVELOPMENT.md:41`: **SQLite thắng, im lặng, báo thành công**.

Đã xảy ra rồi: `results/leaderboard.db` 413.696 byte, 19 bảng, `alembic_version = '0025'` — lệch **4 revision** và thiếu `run_equity/run_pending/run_skipped/run_portfolio_daily/run_trades_overlay/leaderboard_nav`.

Cùng loại bẫy: `stock_ml/scripts/ops/import_yaml_templates.py:303` `os.getenv("STOCK_ML_DB_URL", "sqlite+aiosqlite:///./stock_ml.db")` — mà `STOCK_ML_DB_URL` **không được set ở đâu trong cả hai repo** (mọi thứ khác dùng `DATABASE_URL`) ⇒ lệnh import template trong docs tạo một SQLite mới và báo thành công.

**Việc cần làm.** Bỏ fallback `sqlalchemy.url` (hoặc đặt sentinel vô hiệu) để alembic **fail khi thiếu `DATABASE_URL`**; sửa `import_yaml_templates.py` dùng `DATABASE_URL`; sửa lệnh trong `docs/DEVELOPMENT.md` + `docs/CONTRIBUTING.md:183` cho có env var inline.

### 1.5 CI không chạy gì

`.github/workflows/ci.yml` job `test-unit` trỏ `pytest tests/components/`, job `test-regression` trỏ `pytest tests/regression/test_champions.py`. **Cả hai path không tồn tại** (`stock_ml/tests/` có: api cache data execution features fixtures goldens leaderboard model_dashboard signals strategy visualization). Branch là `master`, đúng branch trigger. `pytest tests/ -q --collect-only` thu **319 test trong 4s** — tức có test, chỉ là CI không gọi.

**Việc cần làm.** Trỏ CI vào path thật. Đây là tiền đề của §2 (không có CI thì golden vô nghĩa).

### 1.6 GC báo 0 orphan trên 51 GB — và bẫy khi sửa

`sweep('results')` → referenced 0/0, orphan **0**, 0.0 MB, trên store thật **14.476 parquet / ~51 GB**. Hai nguyên nhân đã chứng minh bằng gọi trực tiếp:
- `find_feature_cache_files` (`garbage_collector.py:95-104`) dùng `root.glob('*/*')` nên chỉ ra **thư mục** `store/<expr_hash>`, bị `is_file()` loại; file thật nằm sâu một cấp (`store/<expr_hash>/<data_version>.parquet`).
- `gather_referenced_keys` (`:60-80`) đọc `results/experiments` — **thư mục không tồn tại**.

Hai lỗi **triệt tiêu nhau** thành con số 0 trông rất an tâm. Và nó **có thật trong production**: route `stock_ml/api/routes/cache.py:32-49` gọi `sweep()` **mỗi request** `/api/v1/cache/stats` (báo 92.9 MB / 0 orphan). Unit test `tests/cache/test_gc.py:18-31` dựng layout **cũ** hai cấp ⇒ test xanh trong khi production mù — **test đang bảo chứng cho layout sai**.

> ⚠️ **BẪY BẤT ĐỐI XỨNG.** Tập referenced đang **rỗng**. Ai sửa **chỉ** cái glob sẽ biến **toàn bộ 14.476 file** thành orphan trong một sweep — và `garbage_collector.py:14-17` **đã chúc phúc trước** cho hành vi đó ("Legacy runs … will be quarantined on first sweep — this is intentional"). Trỏ vào `stock_ml/results` thì code hiện tại **đã** báo 41 file / 8.754 MB orphan.
>
> **Bắt buộc:** sửa nguồn referenced **TRƯỚC**, và làm cho tập referenced rỗng thành **lỗi cứng**, không bao giờ là "mọi thứ đều orphan".

---

## 2. Hợp đồng nâng cấp engine — trọng tâm

### 2.1 Trạng thái hôm nay, nói thẳng: KHÔNG CÓ bảo đảm nào

`load_bundle` (`stock_ml/src/serving/bundle.py:156-234`) bảo đảm **đúng ba điều**: (1) mọi file có trong `manifest['checksums']` byte-identical lúc export; (2) `manifest['format_version']` cùng **MAJOR** với hằng `BUNDLE_FORMAT_VERSION='1.0'` (`src/serving/__init__.py:14`); (3) lightgbm cùng **MAJOR**. Hết.

**Không** bảo đảm, mỗi mục đã kiểm trong code:

| Lỗ | Bằng chứng |
|---|---|
| `manifest.json` **không** được hash | `checksums` nhúng **bên trong** chính nó (`bundle.py:143-152`), trong khi `serving/engine.py:62,75` đọc `universe` và `feature_scope` — hai thứ ảnh hưởng tín hiệu — ra từ đúng file đó. Thêm nữa `:189-198` chỉ lặp **key của** `checksums`, nên file lạ thả vào thư mục bundle không bao giờ bị phát hiện |
| Wheel đọc `config.json` **không được ghi lại** | `_TRACKED_LIBS` (`bundle.py:38`) bỏ `stock_ml_core`; `stock_ml/core/__init__.py:28` đóng băng `__version__ = "0.3.0"` qua **cả 4 wheel** (0.3.3/0.4.0/0.4.1/0.4.2 — kiểm bên trong từng .whl), **zero consumer** |
| `config.engine` fail-loud bên train, fail-**SILENT** bên serving | Train: `EngineConfig(cost=cost, **engine_cfg)` (`experiment.py:3474`) sau pop-list tay. Serving: `{k:v for k,v in engine_cfg.items() if k in known}` (`stock-serving/serving/trades.py:55-56`) ⇒ knob đổi tên = **lỗi ồn ào bên nghiên cứu, đổi chiến lược âm thầm bên production** |
| Drift guard chỉ phủ `entry_feat_cols`, và chỉ so **TÊN** | `inference.py:65-71`. Cả hai bundle dyn còn có `exit_feat_cols` (22 cột) và `entry4_feat_cols` (50 cột) — **không cột nào được so**. **Giá trị** cột không bao giờ được so |
| `export_bundle` bỏ sót head `entry6` mà train **có** train | `export_bundle.py:224-251` không có `entry6_target_col`, `_entry_head_feat` dừng ở 5; nhưng `experiment.py:585-586, 990-1005, 3365-3366` cho thấy `run_experiment` train entry6 |
| sklearn/numpy được ghi lại rồi **bỏ qua** | `bundle.py:204` lặp `for lib in ("lightgbm",)`. Cả hai bundle dyn bake sklearn 1.3.2 / numpy 1.26.3 và chạy dưới pin 1.8.0 của serving — mà comment của chính `requirements.txt:4` ghi 1.9.0 làm lệch dự đoán LGBMRegressor đủ để **đổi lệnh** (VND 40 vs 37) |
| Overlay **hoàn toàn ngoài** artifact | §3.3 |

**Lỗ hổng đáng chú ý nhất: cơ chế version không diễn tả nổi cái bất tương thích DUY NHẤT đang tồn tại.** Cả hai bundle dyn mang `universe_policy`, field mà `ExperimentConfig` chỉ có từ 0.4.2 ⇒ `ExperimentConfig(**bundle.config)` (`inference.py:47`) **TypeError** trên 0.4.0/0.4.1, trong khi `format_version` báo bundle **tương thích**.

### 2.2 Phản biện đã chứng minh: hợp đồng "additive + default-off" là HÌNH THỨC

Agent đóng vai người bump wheel tìm được **10 đường** qua mọi cổng kiểm mà vẫn đổi tín hiệu của model đang phục vụ. Bốn đường quan trọng nhất:

**Đ1 — sửa một default mà KHÔNG bundle nào pin.** `EngineConfig` có **227 field**, **105 field default không trung tính**; các bundle pin 50-77 ⇒ `dyn900` thừa hưởng **78/105**. Cụ thể: cả 5 bundle set `trailing_atr_mult: 2.0`, **không cái nào** set `trailing_atr_floor`/`trailing_atr_cap` (`engine.py:898-899`, default 0.04/0.16) — hai giá trị clamp **mọi** trailing exit tại `engine.py:2091-2093`. Tương tự `overext_atr_lo/hi` (`:815-816`, dùng `:1647-1648`; `overext` có trong `exit_priority` của **mọi** bundle), `overext_ema_span` (`:842`, dùng `:1596`), `signal_exit_hold_volscale_ref` (`:287`, dùng `:2227`).

**Đ2 — sửa một default recombine đọc qua `cfg.engine.get(k, literal)`.** **Không** bundle nào set `z_norm_window` ⇒ cửa sổ z causal 252 bar định nghĩa **mọi dải mua/bán** đến từ literal tại `experiment.py:2810`; tương tự `z_norm_min_periods=60` (`:2814`), `nonbull_ma_win=50` (`:2852`), `nonbull_persist=3` (`:2855`). Đổi 252→260 là mọi tín hiệu của cả 5 bundle dịch. `test_recombine_snapshot.py` chạy 3 mã tổng hợp với score rút từ rng, không có config champion nào.

**Đ3 — sửa biểu thức DSL, giữ nguyên tên cột.** Drift guard so tên. `resolver.py:139-147` + `dsl/engine.py:33-47` gấp `engine_code_fingerprint` + `expr_hash` vào cache key nên **đúng đắn invalidate** 51 GB store ⇒ giá trị mới được tính, phục vụ, giao dịch, **không cảnh báo**. Đây là mặt lộ lớn nhất: **không có golden cấp tín hiệu trên bundle thật ở bất kỳ đâu trong cả hai repo**.

**Đ4 — `prediction_history` seeding bất đối xứng.** Từ **cùng một** `export_bundle.py`: `bundle__dyn900/prediction_history.parquet` có **4 cột** `[symbol,date,score,exit_score]` (1.222.091 dòng) dù ship **6 model**; `bundle_n2_2643_wavestruct/` có **8 cột** gồm `score2..score5`. Nên với dyn900, `score2..score5` được **model production chấm lại mỗi ngày** — trong khi config của dyn900 làm z causal của `score3` **load-bearing cho exit** (`signal_exit_hold_min_score3_z=0.5`, `signal_exit_skip_if_score3_z=1.6`, dùng ở `engine.py:2253-2255, 2291-2293`).

**Đ5 — đường lách vô hiệu hoá mọi golden.** Không có luật bất biến thì mọi golden đều re-pin được: golden fail → "đúng rồi, tôi vừa retune default đó" → regenerate fixture → ship. Tiền lệ **đã có** trong repo: `test_baseline_snapshot.py:129-134` tự tạo fixture rồi `return`.

**Tầng verify hiện tại đo một code path production không dùng.** `test_baseline_snapshot.py:101` chạy `EngineConfig(max_hold_bars=20, hard_stop_pct=-0.08)` — **2 trong 227 field** — trên 4 mã random-walk tổng hợp. `test_portfolio_golden.py:119-120` đọc base trades từ parquet nên **`run_backtest` không bao giờ được gọi**; nó còn md5-pin (assert ở `:57-64`, giá trị trong `tests/goldens/champion_prod_overlay.json['data_md5']`) **hai đường dẫn tuyệt đối trong repo serving production** cộng `F:/PROJECTS/hb2943_work/nh_nav2.py`, và shell-out tới `_champ_prod_replay.py` ở root. Golden output engine duy nhất từng chạy trên bundle thật là `stock-serving/_snap_regression.py` — untracked, một bundle, docstring ghi "Delete after the deploy".

### 2.3 Hợp đồng cần tạo

> ⚠️ **ĐỌC §11 TRƯỚC.** Chủ dự án đã chốt mô hình nền tảng ngày 2026-07-30, và nó **thay thế R0, R2, R6**
> cùng phần "3 giai đoạn REFUSE" của R1 dưới đây. Mục tiêu KHÔNG phải đóng băng hành vi cũ — mục tiêu là
> **hai bản engine (xưởng ↔ tầng phục vụ) cùng dữ liệu cho cùng kết quả**. Phần R3/R4/R5/R7/R8/R9 vẫn giữ
> nguyên giá trị. Đọc §11 để biết mục nào còn, mục nào bỏ.

**R0 — LUẬT BẤT BIẾN — ⛔ ĐÃ BỎ theo §11.4. KHÔNG TRIỂN KHAI. Giữ nguyên văn chỉ để truy vết.**
~~Tuyên bố tập bundle đang deploy (5 thư mục trong `stock-serving/bundles/`) là **tham chiếu đóng băng**. Một wheel làm đổi output của bất kỳ bundle nào trong đó KHÔNG ĐƯỢC SHIP.~~
⛔ Luật này được viết dưới mục tiêu **đóng băng số**, mà chủ dự án đã **bác** ngày 2026-07-30 (§11.1–§11.3): engine mới là **tốt hơn**, số **được phép đổi**. Xây R0 là xây đúng cơ chế chặn bản cải tiến mà cả chương trình tồn tại để ship.
✅ **Thay bằng:** §11.5.2 (ghim wheel chính xác + từ chối khi wheel đang cài ≠ wheel đã sinh số) và §11.5.4 (re-baseline một lệnh có log). Phần *"re-baseline là hành động riêng, có log, chạy lại attestation"* của R0 **vẫn đúng** và đã chuyển vào §11.5.4.

**R1 — `resolved.json`: SNAPSHOT CONFIG ĐÃ RESOLVE, nhúng trong bundle.** Cơ chế duy nhất bắt được Đ1 và Đ2, và **mạnh hơn** kỷ luật `format_version` vì không phụ thuộc con người nhớ bump. **Đã CHỐT 2026-07-30** (phạm vi mở rộng theo số đo trực tiếp ở §2.5, không phải theo báo cáo agent).

**Vì sao cần: bundle KHÔNG đủ để tái sinh tín hiệu.** Đo trên đúng 5 bundle đang deploy:

| Bundle | key trong `config.engine` | pin được / 227 field | **thừa hưởng default từ wheel** |
|---|---|---|---|
| `bundle__dyn300_onerun` | 77 | 65 | **162** |
| `bundle__dyn900_onerun` | 77 | 65 | **162** |
| `bundle_n2_2643_wavestruct…top150` | 71 | 59 | **168** |
| `bundle_n2_consw20…nbpbw_top150` | 59 | 47 | **180** |
| `bundle_n2_velov_univ150c` | 50 | 39 | **188** |

`config.json` có 30 key top-level và **có** làm tốt phần: `universe`, `universe_policy` (bundle dyn), `seed`, `split`, cả ba threshold, `direction`, `model_mode`, `feature_set`, entry/exit features + targets + models, `data_source_dir`, `market`; cộng `universe_by_year` + `feature_scope` + checksum model trong manifest. Bảy thứ **không nằm trong bất kỳ file cấu hình nào**:

1. **162-188 default engine** — nằm trong wheel. Đã đo: cả 5 bundle set `trailing_atr_mult` nhưng **không cái nào** set `trailing_atr_floor`/`trailing_atr_cap`, hai giá trị clamp **mọi** trailing exit.
2. **`z_norm_window` và `z_norm_min_periods` VẮNG MẶT ở cả 5 bundle** (đã kiểm từng file). Cửa sổ z causal 252 bar — thứ định nghĩa **mọi** dải mua/bán — đến từ literal hardcode `experiment.py:2810/2814`. Đây là tham số quan trọng nhất của tầng tín hiệu và nó không ở trong config.
3. **Ngữ nghĩa feature.** `feature_spec.json` chỉ ghi **tên cột**; biểu thức DSL nằm trong `catalog.py` trong wheel. Sửa biểu thức giữ nguyên tên ⇒ giá trị đổi, drift guard so tên nên **không báo**.
4. **Dữ liệu ngoài.** VNINDEX CSV, `runscore.parquet`, breadth `market.duckdb` — CWD-relative, không checksum, không ghi manifest (§1.1, §1.2).
5. **Hợp đồng seeding `prediction_history`** — không ghi ở đâu, và không nhất quán (dyn900 4 cột vs wavestruct 8 cột, cùng exporter).
6. **Runtime** — wheel version không ghi (`_TRACKED_LIBS` bỏ `stock_ml_core`); sklearn/numpy ghi mà không enforce.
7. **Overlay** — 29 field `PortfolioConstants` hoàn toàn ngoài bundle.

⇒ Bundle ghi đủ để biết **đã chọn gì**, **không đủ để tái sinh tín hiệu**. Tái sinh hiện chỉ đúng khi wheel + `catalog.py` + file ngoài **tình cờ giống lúc export**. Đây cũng là lý do mọi bằng chứng "không drift" trước đây không có nền tảng.

**Nội dung `resolved.json`** (checksum vào `checksums.json` của R4):
- **227 field `EngineConfig` sau khi áp default** — không phải 39-65 field pin
- **35 giá trị recombine đã resolve**, kể cả các giá trị đến từ literal hardcode
- **fingerprint `catalog`** — không cần viết mới: `engine_code_fingerprint()` (`dsl/engine.py:33-47`) đã tồn tại và đã được feature store dùng (`resolver.py:139-147`)
- ~~checksum + bar cuối của VNINDEX CSV / `runscore.parquet` / breadth duckdb~~ → **SỬA theo §11.5.1 Q1 + §13.3: KHÔNG nhúng bytes/checksum file; chỉ KHAI danh sách mã + version kỳ vọng. Production resolve từ sieutinhieu, fail-loud nếu thiếu.**
- **wheel version thật** (không phải `core.__version__` đóng băng ở `'0.3.0'`)
- **`PortfolioConstants` đã resolve** cho tầng danh mục
- **hợp đồng seeding `prediction_history`**: cột nào được seed, mask nào

**Triển khai 2 giai đoạn (G1 ghi · G2 cảnh báo) — G3 ĐÃ BỎ theo §11.4.** Lý do chia giai đoạn: hiện **không ai biết** bật "từ chối" ngay sẽ chặn bao nhiêu bundle, và có lệch **đã tồn tại** (sklearn 1.3.2 bake vs 1.8.0 cài).
⛔ **G3 "từ chối load khi `resolved.json` lệch" KHÔNG được triển khai.** Engine mới sẽ resolve khác **một cách chính đáng** (default mới, nút mới) — từ chối vì lệch là chặn đúng thứ ta muốn. Cái "từ chối" **duy nhất còn sống** là §11.5.2: từ chối khi **wheel đang cài ≠ wheel đã sinh ra các đường cong đang lưu**. Hai chuyện khác nhau: một cái so *cấu hình*, một cái so *thước đo*.

| GĐ | Hành vi | Rủi ro | Guard |
|---|---|---|---|
| **G1** | **CHỈ GHI.** Export ghi `resolved.json`; load không kiểm gì | **Bằng không** | Hai wheel khác nhau trên cùng bundle ⇒ diff `resolved.json` phải chỉ ra đúng field đã đổi |
| **G2** | **Ghi + CẢNH BÁO.** Load tính lại, lệch thì log ở mức WARNING + ghi vào audit row, **vẫn chạy** | Không dừng dịch vụ | Đếm thực tế bao nhiêu field lệch trên 5 bundle. ⚠️ Cảnh báo phải vào audit row, **không chỉ log** — `parity.json` đã chứng minh cổng mà không ai đọc thì bằng không (`deploy_log.jsonl` ghi `"parity": "FORCED"`) |
| ~~G3~~ | ⛔ **BỎ (§11.4)** — xem giải thích ngay trên bảng | — | — |

**Backfill 5 bundle đang deploy — ĐÃ CHỐT: sinh bù ngay bằng wheel 0.4.2 hiện tại.** Chạy một lần, ghi lại hiện trạng thật của 5 bundle dưới wheel đang chạy; đây thành **mốc so sánh** cho mọi lần bump sau.
⚠️ Phải đánh dấu **rõ ràng** trong file: `"provenance": "backfill@0.4.2 2026-07-30"` — nó ghi **"hôm nay thế nào"**, KHÔNG phải "lúc export thế nào". Không được để lần đọc sau hiểu nhầm là snapshot tại thời điểm export. Từ bundle export mới trở đi thì `resolved.json` là thật.

**R2 — DẢI RUNTIME (versioned + asserted).** Thêm `stock_ml_core` vào `_TRACKED_LIBS` (`bundle.py:38`). Thêm `min_runtime` vào `manifest_extra` (`export_bundle.py:305-324`). Bump **MINOR** của `BUNDLE_FORMAT_VERSION` mỗi khi field `ExperimentConfig`, key `feature_spec`, hay ngữ nghĩa `model_names` đổi (`universe_policy` đáng ra đã phải làm nó thành 1.1).
⛔ ~~Nới gate `bundle.py:177-182` thành `bundle_minor <= runtime_minor`.~~ **ĐÃ BỊ §11.5.2 THAY bằng GHIM CHÍNH XÁC + từ chối khi khác.** Dải `MINOR ≤` và ghim chính xác là **hai cơ chế ngược nhau**: dải tạo cửa sổ dung sai, tức đúng cái "âm thầm đổi thước" mà §11.5.2 tồn tại để cấm. Ai triển khai R2 theo câu cũ sẽ xây nhầm cơ chế mà vẫn qua được mọi review chỉ đọc §2.3.
✅ **Phần R2 còn sống:** thêm `stock_ml_core` vào `_TRACKED_LIBS` (`bundle.py:38`) và **gate cả sklearn + numpy**, không chỉ lightgbm (`bundle.py:204`).
⛔ **Nhưng gate sklearn/numpy chỉ được bật SAU khi 5 bundle đã re-export** — xem cảnh báo Phase 4 ở §11.6. Bật trước là dừng 6/6 sổ khách.

**R3 — BẢO TOÀN KEY CONFIG: một deserializer dùng chung hai phía.** Chuyển pop-list ra khỏi `experiment.py:3416-3474` thành `engine_config_from_dict(engine_cfg)` trong `stock_ml/src/backtest/engine.py`, export từ `stock_ml/core/__init__.py.__all__`.
⚠️ Nó phải trả **phân hoạch BA chiều** (field `EngineConfig` / key recombine đã biết / **unknown → raise**), **không** phải hai chiều: mỗi bundle production mang **11-12 key không phải field `EngineConfig`** (`costs`, `entry_ensemble`..`entry_ensemble4`, `entry_gate`, `exit_gate`, `exit_force_gate`, `exit_force_gate_lowbreadth`, `exit_force_gate_nonbull`, `nonbull_ma_win`, `nonbull_persist`) — chúng được `recombine_signals` tiêu thụ qua **35** lệnh `cfg.engine.get(...)`. Luật "raise nếu dropped khác rỗng" kiểu hai chiều sẽ **từ chối cả 5 bundle**.
⚠️ Deserializer cũng phải **sở hữu** việc dựng `CostModel` (`experiment.py:3462-3473` nhận **cả** dict `costs` lồng **và** key phẳng commission/tax/slippage — serving chỉ đọc `engine_cfg['costs']`, không đọc key phẳng) và **fail-loud `portfolio.enabled`** (`:3477-3482`), cộng `slippage_model` (`:3472`). Thêm một test khẳng định phân hoạch — hôm nay 3 danh sách tay khớp nhau **do may**: 37 lệnh `pop` trong 3418-3462 (39 trong 3416-3474), 35 key `.get`, 0 key ở mọi bucket lệch, 0 chồng lấn.
⚠️ Đồng thời nhận **`max_hold` mà serving tự chèn** vào `exit_priority` (`stock-serving/serving/trades.py:65-66`) — hiện vô hại (3 bundle thiếu `max_hold` có `max_hold_bars=10000`; 2 bundle dyn đã liệt kê `max_hold` với `max_hold_bars=14`) nhưng là **một sửa chiến lược chỉ tồn tại ở production**.

**R4 — TOÀN VẸN MANIFEST.** Tách `checksums` ra `checksums.json` riêng, **bao gồm** sha256 của chính `manifest.json`, và **đóng cả TẬP file** (file lạ phải bị phát hiện). Định nghĩa `bundle_sha` = sha256 trên danh sách `(path, hash)` đã sort — một chuỗi duy nhất để deploy log, attestation và hàng leaderboard cùng tham chiếu.

**R5 — DRIFT GUARD ĐẦY ĐỦ.** `inference.py:65-71` thành vòng lặp trên **mọi** key `*_feat_cols` có trong `feature_spec` (entry, exit, entry2..entry6). Thêm assert trước `write_bundle` (`export_bundle.py:332`) rằng `set(models)` khớp tập head khai bởi `cfg.engine.entry_ensemble..entry_ensemble5`/`exit_ensemble` — bịt lỗ entry6. Assert `prediction_history` phủ **mọi** cột dự đoán mà các head của bundle sinh ra, và ghi **hợp đồng seeding** (cột nào, mask nào) vào manifest.

**R6 — GOLDEN TOÀN CHUỖI (không phải golden engine).** Commit models — đo được **1.5 MB** cho cả 6 joblib của `bundle__dyn900` — và pin sha256 của **CẢ HAI**: `generate_signals_from_bundle(bundle, slice)` **và** `trades_to_dataframe(run_backtest(...))`, trên một slice OHLCV **đã commit** dưới `stock_ml/tests/goldens/` — không đường dẫn tuyệt đối, không DB ngoài, không `nh_nav2`. Golden chỉ pin `run_backtest` sẽ để hở Đ2, Đ3, Đ4 vì `signals` khi đó là **input fixture**.
⚠️ **Một hash không sống nổi trên hai platform**: CI là ubuntu-latest, bạn làm việc trên Windows 11, và lệch float Windows↔container **đã đo** (VND 40 vs 37). Pin **một fixture mỗi platform**, hoặc tuyên bố Linux là mặt phẳng recompute duy nhất được thừa nhận. Cũng đừng round 8dp như `_snap_regression.py` — đó đúng là độ phân giải làm biến mất một cú lật ở biên clip của Đ1.

**R7 — ATTESTATION THEO WHEEL, không theo bundle.** Viết producer `stock_ml/scripts/ops/attest_bundle.py`: nạp bundle qua wheel, sinh lại tín hiệu trên universe của chính nó, diff với `run_signals` của run được xếp hạng, phát `parity.json` gồm mức khớp theo năm + wheel version + `bundle_sha`; đưa `parity.json` vào `checksums.json` để nó **được ký**, không phải free text.
⚠️ `parity.json` hiện **không tồn tại ở đâu trong repo này** (grep = 0 hit, không bundle nào có), trong khi `stock-serving/serving/deploy.py:74-84` từ chối activate nếu thiếu — và dòng duy nhất trong `deploy_log.jsonl` ghi `"parity": "FORCED"`. `deploy.py` phải **từ chối khi `parity.json`.wheel ≠ wheel đang cài**, nếu không một attestation làm dưới 0.4.2 sẽ pass mãi dưới 0.5.0.
⚠️ Phải quyết **ngay** `attest_bundle` phát gì cho **3 bundle self-train** (`git_sha="serving-build:0.3.3"`, `template_id` None) — chúng **không có** `run_signals` nào để diff, nên lần chạy thật thà đầu tiên sẽ đọc như một sự cố.

**R8 — WHEEL TÁI LẬP ĐƯỢC + PROVENANCE THẬT THÀ.** Hôm nay: `git branch --show-current` = `master` với **116** entry dirty/untracked, gồm ` M stock_ml/src/pipeline/experiment.py`, ` M stock_ml/src/data/splitter.py`, ` M stock_ml/scripts/ops/export_bundle.py` và **untracked** `stock_ml/src/data/universe_resolver.py` — **tất cả đã nằm trong `dist/stock_ml_core-0.4.2-py3-none-any.whl`**, tức wheel production chứa code **không thuộc commit nào**. `git tag` chỉ có `phase-0.2-baseline` và `pre-cleanup-snapshot`. `dist/` của train có {0.2.0, 0.3.0, 0.4.0, 0.4.1, 0.4.2} — **0.3.1-0.3.4, gồm 0.3.3 đã train cả 3 bundle top-150 đang chạy, chỉ tồn tại trong `stock-serving/dist/`**. Build là một dòng README (`stock_ml/README.md:42-45`). `pyproject.stock_ml_core.toml:31-32` là wildcard `"*" = ["*.yaml","*.yml","*.json","*.csv"]` nên ship kèm **5 JSON cache/results bị gitignore**. Wheel **không** byte-reproducible (`core.autocrlf=true`; **85** trong 115 file .py chung khác nhau **chỉ ở EOL** giữa 0.4.1 và 0.4.2) — điều này vô hiệu hoá mọi pin dựa trên hash gắn thêm về sau.
⇒ Viết `stock_ml/scripts/ops/build_wheel.py`: **từ chối tree dirty**, ghi sha256 wheel vào `dist/CHECKSUMS.txt`, tạo tag `wheel-<version>`. Thêm `.gitattributes` `* text eol=lf`. Thay wildcard package-data. `export_bundle.py:64-71` phải ghi HEAD **cộng hash diff của working tree**, hoặc từ chối export.
⚠️ **Provenance của bundle đã ship là sai và không sửa được hồi tố**: cả hai bundle dyn live ghi `git_sha=60a23062…`, một commit **không chứa** code `universe_policy` đã sinh ra chúng. R7 diff với `run_signals` — cũng do chính tree dirty đó sinh ra — nên nó sẽ phát `PASS` và **chứng thực bằng mật mã cho một commit sai**. Cần một dòng ghi rõ: provenance của 5 bundle hiện tại là **không xác thực được**, và điều đó chỉ được sửa từ lần export kế tiếp.

**R9 — FACADE THẬT.** `stock_ml/core/__init__.py` phải export `run_backtest`, `EngineConfig`, `trades_to_dataframe`, `engine_config_from_dict`, `stock_ml.portfolio`, và **`clear_feature_caches()`**. Hôm nay serving đi xuyên facade trên **5 đường** và chọc vào **3 biến private**: `serving/trades.py:53,83` import `stock_ml.src.backtest.engine`; `serving/build_bundle.py:36-38` import **code train** (`build_feature_frame`, `train_fold`, `YearSplitter`); `serving/breadth_store.py:64-74` **ghi vào** `_exp._BREADTH_CACHE` (`experiment.py:1649`), `_exp._XSEC_CACHE` (`:1689`), `_exp._load_regime_index.cache_clear` (`:1636`) sau `hasattr` + `except` trần, tự ghi chú "best-effort no-op nếu wheel đổi". **Đổi tên một biến đó** ⇒ uvicorn sống lâu đóng băng breadth ở lần đọc đầu, bar mới nhận breadth NaN, `exit_force_gate_lowbreadth`/`nonbull_*` đổi trên sổ live, **không log, không exception, không metric**. Serving gọi facade **không có `except` trần** ⇒ wheel đổi tên thành `ImportError` lúc boot. **CROSS-REPO.**

⛔ ~~**Bảo đảm mua được:** sau R0-R6, một bundle đang deploy được bảo đảm rằng nâng wheel **hoặc** để tín hiệu và lệnh của nó **byte-identical**, **hoặc** từ chối load. Không có gì ở giữa.~~
**CÂU TRÊN ĐÃ BỊ BÁC (§11.2)** — nó phát biểu tương thích ngược về **SỐ**, thứ chủ dự án đã bác. Nó cũng là câu dễ trích dẫn nhất tài liệu này, nên nếu để nguyên thì nó sẽ sống lâu hơn mọi đính chính.

✅ **Bảo đảm ĐÚNG, thay thế câu trên:** engine mới **nạp và chạy được** mọi cấu hình còn sống (§11.5.5), và **mọi ô số công bố mang theo engine đã sinh ra nó** (§11.5.2) nên không bao giờ âm thầm đổi thước. **Số ĐƯỢC PHÉP đổi** — nếu là bản sửa/nâng chuẩn thì nó **phải** đổi, và việc đó kết thúc bằng một lần re-baseline có log (§11.5.4).

⚠️ **Phải chọn và ghi rõ đang mua loại tương thích ngược nào.** "Từ chối load" với một sổ khách đang chạy là **sự cố**, không phải tương thích. Tương thích thật cần **hoặc** (a) snapshot config đã resolve ở R1 để bundle cũ giữ nguyên ngữ nghĩa lúc export dưới wheel mới, **hoặc** (b) engine version do bundle khai để code path cũ còn địa chỉ. Hôm nay wheel **không có** lineage version nào để gắn vào.

### 2.4 Overlay cần hợp đồng riêng, cùng hạng

`PortfolioConstants` (`stock_ml/portfolio/constants.py:22-77`) có **29 field**; các tier trong `stock-serving/serving/tiers.yaml` pin **1-3 field mỗi tier**, còn lại **thừa hưởng default từ wheel** ⇒ một bump sửa bất kỳ default nào **đổi 6 sổ live không để lại dấu vết**.

Cần `overlay_config_hash` tính trên **giá trị field đã resolve**, ghi **cả** vào hàng leaderboard **và** vào snapshot bên serving. Cột đã tồn tại (`migration 0029:53-57`) và có **0 writer**.

Và `stock_ml/portfolio/api.py:91-92` có `continue` **trần** cho leg thiếu bar entry/exit trong price panel — **không** append `skipped[]`, **không** counter, trong khi mọi đường loại khác (`:171, 174, 178, 187`) đều append. Đo được: **115 trong 31.817** closed base trade của `template/_dyn900_onerun` bị bỏ như vậy **hôm nay**.

Overlay hiện **không có guard nào chạy được trong CI** (xem `test_portfolio_golden.py` ở §2.2) ⇒ phải copy 2 store đã pin sang một snapshot đóng băng, re-pin theo nó, và commit một golden overlay tự chứa **trước bất kỳ wheel bump nào**.

---

## 3. Danh tính run — chưa từng được mô hình hoá

Đây là lỗi **có từ trước** overlay, không phải do overlay.

`run_id = f"template/{run_name}-{config_hash[:8]}"` (`run_template.py:361-364`), với `config_hash` = sha256 của `summary['config']` — một projection tay mang feature_set / entry_features / exit_features / targets / models / split **cộng đúng SÁU engine key** (`max_hold_bars`, `min_hold_bars`, `hard_stop_pct`, `commission`, `tax`, `slippage`) (`experiment.py:3607-3632`). **Seed, universe, universe_policy, feature scope và ~220 knob còn lại vô hình.**

Đo trên Postgres live: `leaderboard_runs` 3612 hàng, 1052 superseded, state = 3607 `trained` + 5 `pinned`. Hash `2c78c568f85dc826` dùng chung cho **360 run_name khác nhau**; `b8f000e2d1c86eb8` 270; `6933813844a6713e` 146. `cache_key_features=''` trên **3612/3612**.

`run_repo.upsert` (`db/repositories/run_repo.py:71-79`) `ON CONFLICT (run_id) DO UPDATE` **mọi cột**, rồi `mark_superseded` (`:85-95`) bật `superseded=True` cho hàng khác cùng `(bundle, run_name)` — **không đếm xỉa state**. `run_template.py:452` luôn dựng hàng với `state=LifecycleState.trained`. `deploy_wavestruct.py:90-99` lặp SEEDS gọi `run_template_experiment` với `NEW_NAME` **hằng** (`:37`), nên **mọi seed ghi vào cùng một `run_id`** — và `read_row(run_id)` (`:98`) đọc lại **cùng một hàng đã bị ghi đè**, nên bước "xác nhận multi-seed" là **diễn**. (Hàng wavestruct còn lại có `run_seed=42`.)

Workaround đã có mà không ai đọc: `scripts/ops/rescore_multiseed.py:3-14` tự ghi nhận vấn đề và tạo bảng phụ `leaderboard_seed_stats` (`CREATE TABLE IF NOT EXISTS` ở `:35`, không ORM, không migration, 54 hàng) — **API leaderboard không bao giờ đọc nó**.

**Việc cần làm.** `config_hash` phải phủ **toàn bộ** cấu hình ảnh hưởng kết quả, hoặc `run_id` phải mang seed + universe tường minh. `upsert` không được reset `state`/`created_at`/`superseded` do một lần chạy lại. `mark_superseded` phải tôn trọng `state='pinned'`.

**Guard.** Chạy cùng template với 2 seed ⇒ **2 hàng**. `SELECT count(DISTINCT run_name), config_hash FROM leaderboard_runs GROUP BY 2 HAVING count(DISTINCT run_name) > 1` ⇒ rỗng cho run mới.

---

## 4. Mối nối di sản còn lại (ngoài §1)

### 4.1 `leaderboard_nav` — một cột, ~20 writer, ba thước

18 cột **sống** vs **11** cột trong DDL duy nhất của repo (`score_nav_leaderboard.py:60-74`). `cagr_overlay/maxdd_overlay/overlay_k/overlay_note` **không có DDL** trong bất kỳ migration nào; `cagr_t2/maxdd_t2` đến từ ALTER ad-hoc tại `stock_ml/analysis/serving_blindspot/r3line/hb_add_t2_column.py:155-156` — mà `:164-165` còn **ghi lại `cagr_adv`/`maxdd_nav`** mà không đụng `config_hash`.

Writer: **20 file** — 7 one-off ở root (`_audit_rescore_board.py`, `_dyn_multi_overlay.py`, `_finalize_pin.py`, `_note_liqcol.py`, `_official_score_all.py`, `_register_invvol.py`, `_register_overlay_variants.py`), **12** dưới `stock_ml/analysis/serving_blindspot/r3line/`, cộng `scripts/ops/score_nav_leaderboard.py`. `_note_liqcol.py:11-14` ghi provenance bằng **nối chuỗi** vào `overlay_note`.

Tỉ lệ điền: 3445 tổng / `nav_adv` 3439 / `cagr_t2` **20** / `cagr_overlay` **27** / `overlay_config_hash` **0**.

**Chứng cứ số vô lý:** `template/x2_struct_to_k10_cs5ma50_r7ec_gt_dl63size-69338138` có `cagr_adv 0.6626 < cagr_t2 1.3177` — **T+2 thắng T+0 là bất khả về vật lý** ⇒ hai cột do hai thước khác nhau ghi.

Trên **DB mới**: sau `alembic upgrade head` bảng **không tồn tại**; `score_nav_leaderboard` tạo chỉ 11 cột; `api/routes/leaderboard.py:29-31` SELECT `cagr_t2`/`cagr_overlay` ⇒ `UndefinedColumn` ⇒ bị **`except Exception:` nuốt** (`:48-52`, rollback + `logger.debug`, compose đặt `LOG_LEVEL=INFO`) ⇒ **mọi ô CAGR render `—` không một lỗi**.

**Việc cần làm.** Nâng shape 18 cột thành **migration 0030** (`CREATE TABLE IF NOT EXISTS` + `ADD COLUMN IF NOT EXISTS` để an toàn cả trên DB live và DB mới). `score_nav_leaderboard` thành **writer duy nhất**. Bắt buộc `overlay_config_hash` mỗi lần ghi. Bỏ `except Exception` trần ở reader. UI hiện rõ **thước nào**.
Đây chính là **B3 của `PORTFOLIO_REGISTRATION_PIPELINE.md`** (`register_overlay.py` — **chưa tồn tại**, verified `ls stock_ml/scripts/ops/`).

### 4.2 Thước chấm chính thức nằm NGOÀI repo

`scripts/ops/score_nav_leaderboard.py:37-39` import `nh_nav2` từ `F:/PROJECTS/hb2943_work` — **ngoài cả hai repo và không thuộc git repo nào** — với price DB riêng (`DB_PATH`), khác hẳn nguồn dữ liệu của overlay. Không tái lập được chỉ từ repo. Trong khi `stock_ml.portfolio` **đã** vào wheel làm single-source-of-truth ⇒ hai tầng đang ở hai chuẩn quản trị.

**Việc cần làm.** ⚠️ **SỬA vòng-3 (§14.4): BỎ `nh_nav2`, thống nhất NAV về `stock_ml.portfolio`** — code trong wheel mà chính các chiến lược dùng, nguồn giá sieutinhieu, đưa NAV vào khung attestation (§9.6.1b). `nh_nav2` là **codepath NAV thứ hai**, cùng loại landmine `portfolio_engine` #2 (§10.2). Kéo theo: `score_nav_leaderboard.py` thôi import nh_nav2; golden `_champ_prod_replay.py` + `test_portfolio_golden` (md5-pin nh_nav2, §2.2/§5) chuyển sang pin trên `stock_ml.portfolio`.

### 4.3 `state` vs `superseded` — hai cột một khái niệm, và chúng bất đồng trên hàng thật

`src/leaderboard/schema.py:15-25` khai `LifecycleState` là "single source of truth for dashboard visibility". Nhưng `run_repo.list_ranked` (`:106-123`) **luôn** filter `superseded`, còn `state` là filter tuỳ chọn mà API **không bao giờ truyền**. Crosstab đo được: trained/false **2557**, trained/true **1050**, pinned/false **3**, pinned/**TRUE 2**. `retired` **chưa từng** dùng. Migration `0001:128` có partial index `WHERE state='pinned' AND superseded=false` — schema đã giả định giao của hai cột.

⇒ **2 run được operator PIN thủ công đang vô hình trên bảng mặc định.** Và 5 script untracked dưới `r3line/` (`hb_register_{cs5,k8k12,klever,margin}.py`, `hb_csma50_valid.py`) hồi sinh hàng bằng `update leaderboard_runs set state='trained', superseded=false` thô, ngoài mọi bookkeeping.

**Việc cần làm.** Một predicate lifecycle duy nhất: hoặc suy visibility từ `state` (và `mark_superseded` phải set nó), hoặc bỏ `state`.

### 4.4 `fairness` bị bỏ lại ở thế giới CSV, UI vẫn render nó

`src/leaderboard/fairness.py::annotate_rows` chỉ được `aggregator.py:10` và `loader.py:21` import — cả hai thuộc thế giới file đã chết. Cả **6** cột `same_*_as_baseline` **NULL trên 3612/3612**; `api/routes/leaderboard.py:80-131` **không phát** chúng ⇒ JS thấy `undefined`, mọi kiểm `=== false` **fail closed**; `fairness_group_key` là `'template_vn_stock'` suy biến trên **3604/3612** hàng (chỉ 3 giá trị phân biệt tồn tại).

⇒ "Fair mode" (`dashboard/leaderboard.js:255, 307-309, 321-322`) **im lặng so ngang** các run khác universe, khác cửa sổ, khác cost — đúng thứ mà bộ máy này sinh ra để ngăn — và **không cảnh báo nào có thể hiện ra**, không phân biệt được với "mọi so sánh đều công bằng".

**Việc cần làm.** Chọn một: tính flags + group key thật trong đường ghi DB, **hoặc** xoá 6 cột + group key + nhánh JS + module fairness. Nhập nhằng ở đây tệ hơn cả hai lựa chọn.

### 4.5 Danh tính module: `src.*` và `stock_ml.src.*` cùng sống trong một process

`run_template.py:24-32` insert **cả** repo root **và** `repo_root/stock_ml` vào sys.path rồi `from src.pipeline.experiment import ...`; chính module đó lại `import stock_ml.src.*` (`experiment.py:28`). Chỉ **9 file** dưới `stock_ml/src` còn dùng prefix cũ — một cuộc di trú nhỏ, làm xong được.

Hậu quả: **hai class object và hai module cache cho cùng một code** (`LeaderboardRow` đi từ `run_template.py:401`→`:467` là class pydantic **khác** với class `run_repo.py:12` khai; `_VNI_CACHE` của engine và `_BREADTH_CACHE` của experiment mỗi cái tồn tại **hai bản**). Và nó **phá hợp đồng của chính wheel**: `import stock_ml.src.model_dashboard` / `stock_ml.src.config_loader` raise `ModuleNotFoundError: No module named 'src'` khi sys.path chỉ có repo root — **trong khi cả hai nằm trong namelist 0.4.2 mà serving cài**. Máy dev che lỗi này vì entry point insert cả hai root.

Kèm theo: `stock_ml/src/env.py` vs `stock_ml/src/utils/env.py` là **bản fork**, và `utils/env.py:34` tính `dirname(dirname(.../src/utils/env.py))` = `stock_ml/src` nên `get_results_dir()` trả `stock_ml/src/results` **trong khi docstring của chính nó ghi `(stock_ml/results)`**. `seed.py` và `safe_io.py` là **bản trùng md5-identical**. Ba cây `results/` cùng tồn tại: `results/` (**77 GB**, thật), `stock_ml/results` (**8.6 GB**), `stock_ml/src/results` (**93 MB**, tình cờ — và **5 cache manifest của nó bị đóng gói VÀO wheel**).

**Việc cần làm.** Gộp về `stock_ml.src.*`; xoá bản trùng `stock_ml/src/{env,config_loader,seed,safe_io}.py`, giữ bản có path arithmetic đúng; một `results` root duy nhất.

### 4.6 Cuộc di chuyển file chưa hoàn tất trong `scripts/`

**4 file** dưới `stock_ml/scripts/ops/` còn tính `parents[1]`, nay resolve thành `stock_ml/scripts` ⇒ lazy import `from src.…` **luôn raise**: `cache_gc.py:26`, `build_leaderboard.py:7`, `api_server.py:30`, `export_derivatives_ohlcv.py:8`. `cache_gc.py` fail ở **runtime** dòng 116 (`--help` exit 0 vì import lazy). `build_leaderboard.py` cùng hình dạng (failure **UNVERIFIED** — chưa chạy vì nó ghi file).

Và UI live **chỉ operator vào cái server đã chết**: `dashboard/leaderboard.html:295,329` + `leaderboard.js:591` in banner `python -m stock_ml.scripts.api_server` và hint `python -m stock_ml.scripts.build_leaderboard rebuild` — **cả hai module path đã không còn**; `#dataPath` hardcode `../results/leaderboard/leaderboard.json`; `dashboard/serve.py:32-33` in URL `/visualization/*`. `docs/API.md` (`:328, 405, 448, 509, 540, 1069, 1076-1081`) tài liệu hoá **6 endpoint 404/405**, và `DELETE /api/v1/runs/bulk` rơi vào catch-all `{run_id:path}` (`runs.py:652`) như một run **tên là 'bulk'**. Một test **xanh** (`tests/api/test_api_server.py:60`, 9 test) chứng nhận cái server không thể khởi động, vì conftest cấp sẵn sys.path mà CLI thiếu.

**Việc cần làm.** Sửa `parents[1]`→`parents[2]` hoặc xoá 3 script chết; sửa 4 chuỗi lệnh sai trong UI; sửa `docs/API.md`; thêm route DELETE bulk trước catch-all hoặc bỏ khỏi docs.

### 4.7 `FeatureCacheManager` — API cache thời tiền-DB, docstring chỉ sai đường

`stock_ml/src/cache/feature_cache.py`, re-export bởi `src/cache/__init__.py:3,5`. **Zero importer** (grep chỉ ra chính nó, chính __init__, và một docstring ở `src/leaderboard/schema.py:31`). Nhưng **không** được coi là dead-safe: nó **nằm trong wheel production** và `import stock_ml.src.cache` **thành công trong container serving live** (đã verify bằng `docker exec`). Docstring (`:4-7`) khai layout `results/cache/features/<feature_set>/<key>.parquet` — **mâu thuẫn** layout thật của `FeatureStore` ⇒ ai tin docstring sẽ tìm sai chỗ. Nó còn fingerprint CSV dưới `<data_dir>/all_symbols/` (`:40-64`), một layout mà thời DuckDB đã bỏ.

**Việc cần làm.** Xoá `FeatureCacheManager` + re-export, sau khi §1.6 xong.

### 4.8 Cột mang sang mà chưa nối dây

`cache_key_features`/`cache_key_predictions` = `''` trên 3612/3612 · cả 6 `same_*_as_baseline` NULL trên 3612/3612 · `template_config_hash` = `strategy_templates.config_hash` trên **0** của 3604 hàng join được · `jobs` **0 hàng** · `feature_materialization` **0 hàng** trong khi có **14.476** parquet thật.

INFERRED (bằng chứng chỉ có bytecode): một `stock_ml/scripts/seed_features.py` đã seed `feature_def` (86 hàng) / `feature_set` (16 hàng) từ file sang DB — **git chưa từng track nó**, chỉ còn `stock_ml/scripts/__pycache__/seed_features.cpython-312.pyc` (4.517 B, mtime 2026-06-01), trong khi **4 chỗ** vẫn hướng dẫn chạy nó (`migration 0022:15`, `api/routes/features.py:5`, `import_yaml_templates.py:55,68`, `docs/FEATURE_STORE_DSL_DESIGN.md:243`).

### 4.9 Danh tính run-record của dyn: parquet vs Postgres, hai script untracked

`_dyn900_onerun.py:36,62,72,78-80` nhận `N` từ argv, đặt `cfg.name = f"dyn{N}_onerun"` (**không** gạch dưới đầu), `run_id=None` nên **không đăng ký gì**, và ghi `_champ_src/dyn{N}_onerun_{trades,signals}.parquet`. `_dyn_multi_register.py:54,74` lại mint tên DB **có** gạch dưới: `f"_dyn{n}_onerun"`. Verified: template 3524 tên `'_dyn300_onerun'`, và **file `_dyn300_onerun.py` chưa từng tồn tại**.

Hai overlay scorer đọc hai thế giới khác nhau nhưng **ghi cùng một cột** `leaderboard_nav.cagr_overlay`: `_dyn900_onerun_overlay.py:38,45-46` đọc **parquet**, `_dyn_multi_overlay.py` đọc **`run_trades`**. Chỉ parquet của dyn50 và dyn900 còn trên đĩa, còn DB có dyn50/61/100/150/200/300/400/900 ⇒ `_dyn900_onerun_overlay.py 300` **FileNotFound** trong khi `_dyn_multi_overlay.py 300` chạy được — cùng nhiệm vụ, hai thế giới dữ liệu. (Hai nguồn dyn900 đo được **giống nhau hôm nay**: 31.817 dòng cả hai, 0 key chỉ có ở một bên ⇒ đây là **rủi ro drift**, không phải lệch số hiện tại.)

**Việc cần làm.** **Track** các producer (chúng là bản ghi thực thi duy nhất về cách tín hiệu của tier live được tạo), chọn **một** overlay scorer làm authoritative, và thôi đặt tên template DB theo file không tồn tại.

---

## 5. Dọn — danh sách xoá nhỏ đến mức đáng ngạc nhiên

**Xoá được (DEAD_SAFE, đã chứng minh):**

| Đường dẫn | Kích thước | Chứng minh |
|---|---|---|
| `build/lib/` | 1.4 MB / 121 file | Không importer, không trong wheel namelist, không CI/docker/alembic; bản build cũ |
| `whlchk/` | 1.2 MB / 116 file | Như trên |
| `visualization/` (ở **repo root**, khác `stock_ml/visualization`) | 2 thư mục rỗng | Rỗng |
| `stock_ml/tests/visualization/__pycache__/` | 13 KB | Bytecode |
| `stock_ml/analysis/v34_vs_rule_per_symbol.py`, `v34_vs_rule_wave_match.py`, `v35_per_symbol.py` | 3 file | Không referenced |

**Tổng ~2.6 MB.** Nói cách khác: **repo này gần như không có rác an toàn để xoá.**

**TUYỆT ĐỐI KHÔNG XOÁ** (agent phản biện chuyên cứu file đã bảo vệ 2 mục đầu):

| Đường dẫn | Kẻ phụ thuộc |
|---|---|
| `_tmp_analysis/` (1.6 GB) | `stock_ml/scripts/experiments/train_runscore.py:93` (**tracked**) đọc `_tmp_analysis/champ2646_trades.csv` để lấy universe mã champion. `train_runscore.py` là producer của `results/_research_2429/runscore.parquet`, thứ mà `engine.py:57` `_load_runscore` mặc định trỏ vào và `engine.py:1394` nạp làm run-score hold modulator ⇒ xoá là **mất khả năng tái tạo một file âm thầm đổi hành vi model**. Cũng được `docs/RESEARCH_STRATEGY_MAP.md:168,185` và `seed_exit_{toptarget,phase}_1378.py:3` tham chiếu |
| `stock_ml/visualization/` | **506 trong 4.516 file là untracked** (26 thư mục untracked hoàn toàn) ⇒ "git revert phục hồi được" chỉ đúng 4.010/4.516. Nếu xoá thì **phải tar backup trước** |
| `_champ_prod_replay.py` (**tracked**, 313 dòng) | `tests/test_portfolio_golden.py:72` shell-out tới nó và md5-check output |
| `_champ_src/` (47 file, 0 tracked) | `tests/test_portfolio_golden.py:57-64` assert md5 của `_champ_src/_prod_rewritten.csv` |
| `F:/PROJECTS/hb2943_work/nh_nav2.py` | Ngoài cả hai repo, **không thuộc git repo nào**; `_champ_prod_replay.py:17-19` và `score_nav_leaderboard.py:37-39` import nó |
| `_seed_serving_tiers.py`, `_dyn900_onerun.py`, `_dyn_multi_register.py`, `_dynuniverse_clean.py` (đều **untracked**) | `_seed_serving_tiers.py` sinh **7 hàng seed** trong `stock-serving/data/portfolio.db` (26.8 MB). ⛔ Sửa: **KHÔNG phải producer duy nhất** — `serving/portfolio/runner.py:61` và `:107` cũng ghi vào đó (6 hàng `<tier>_live` + 2 hàng replay). Vẫn không được xoá script này: nó là bản ghi duy nhất về cách 7 hàng seed được tạo |
| `results/tmpl_3524_565c005aa3/`, `results/tmpl_3538_*/` | Fold predictions của **hai bundle đang deploy** |
| Hàng Postgres của `template/_dyn300_onerun-69338138` và `template/_dyn900_onerun-69338138` | `_seed_serving_tiers.py:56-59` đọc đúng các run_id này để dựng sổ tier live |
| Bảng `leaderboard_nav` (3445 hàng), `leaderboard_seed_stats` (54 hàng) | Không ORM, không migration create ⇒ drop là **không dựng lại được** |
| `stock_ml/src/model_dashboard/`, `db/init/001_model_dashboard_schema.sql`, `db/model_dashboard.sqlite` | `tests/model_dashboard/test_{store,schema}.py` (9 test) |
| `scripts/ops/api_server.py`, `src/cache/garbage_collector.py`, `src/leaderboard/*` | `tests/api/test_api_server.py:60` (9 test); GC được route live gọi |
| `portable_data/.../VNINDEX/.../data.csv`, `results/_research_2429/runscore.parquet`, `market_data/market.duckdb` | §1.1, §1.2 — di chuyển/xoá là **âm thầm đổi hành vi model** |
| `scripts/__pycache__/seed_features.cpython-312.pyc` | Dấu vết **duy nhất** của script seed `feature_def`/`feature_set` |

---

## 6. Hiệu suất (ưu tiên thấp, ghi để không mất)

`PortfolioContext` (`stock_ml/portfolio/context.py`) **không memoise gì** ⇒ mỗi lần `run_portfolio` dựng lại market panel và fit lại LGBM meta-priority **cho từng config**, nên chi phí **không** amortise giữa các tier dùng chung một bundle.

Neo đã **đo**: `results/` **77 GB** · `results/cache/features` **14.476 file** (~51 GB) · `results/tmpl_*` **5868** thư mục · `strategy_templates` **3222** hàng · `run_signals` **96 GB / 368.490.358 dòng** · DB **98 GB** · `leaderboard_runs` 3612 hàng / 77 cột.

**ESTIMATE, chưa tái lập được** — đừng dùng làm căn cứ quyết định: tỉ lệ 34.467/34.275 fold checkpoint quá cũ (một agent đếm 34.562, không verify được mtime split) · 1338/3222 template có engine key ảnh hưởng tín hiệu (không nêu tiêu chí) · tách 54.5/22.4 GB (đo lại được 51/26) · ~28 GB `run_signals` superseded · `build_market_panel` ~35 s/lần với ~24 s trong dict/strftime.

---

## 7. Sửa lỗi phải áp vào `PORTFOLIO_REGISTRATION_PIPELINE.md` khi thực thi B2/B3/B5

1. **Số writer**: doc nói ~17 script root; thực tế **20 writer**, trong đó **7** ở root và **12** dưới `stock_ml/analysis/serving_blindspot/r3line/`, cộng `score_nav_leaderboard.py`. **`_note_liqcol.py` là writer mà doc không nêu** ⇒ danh sách retire của nó **thiếu**.
2. **`overlay_config_hash`**: doc coi như B1 đã giải quyết. Cột **có** nhưng **0 writer** — B3 phải là nơi ghi nó, và phải tính trên **giá trị field đã resolve**.
3. **`leaderboard_nav` shape**: doc chỉ ALTER thêm `overlay_config_hash`. Thực tế bảng có **18 cột sống vs 11 trong DDL** ⇒ cần migration 0030 nâng toàn bộ shape, `IF NOT EXISTS`-safe cả hai chiều.
4. **`variants.py`**: mô tả đúng là "**1** entry `UNVERIFIED` (`gt1`) + **3** unported (họ `dl63`); chỉ `gtos` được golden-verify", không phải "6 entry UNVERIFIED". File 42 dòng, ship trong wheel, **zero importer**.

Ngoài ra, quyết định sản phẩm bên serving đã thay đổi một tiền đề: chủ dự án chốt **chào khách **cả 10** chiến lược (§14.1)**, nên `tier_dyn900_k16` không còn là "ops-only" (`customer: false`) như `tiers.yaml` đang ghi — xem `stock-serving/RESTRUCTURE_STRATEGY_LAYER.md` §5 D4.

---

## 8. Claim đã bị phản biện bác — KHÔNG dùng lại

Hai agent phản biện mỗi đợt đều trả `needs_fix`; 13 claim bị bác. Ghi lại để không ai trích dẫn như sự thật:

1. "Serving độc lập với train qua đúng 1 wheel" (CLAUDE.md của serving) — **sai**: 5 đường xuyên facade + 3 biến private (xem R9).
2. "`generate_signals_from_bundle` raise nếu cột drift" — **chỉ** `entry_feat_cols`, và **chỉ theo tên**.
3. "`run_portfolio` **luôn** áp rewrite" — `api.py:58` là `rw = rewrite(...) if C.rewrite_on else closed.copy()`.
4. "Hai mô hình chi phí trùng nhau tình cờ" — **không trùng**: `FEE=0.004` đúng bằng `2*0.0015+0.001`, nhưng `S0=0.001` ≠ slippage 0.0015 của bundle ⇒ **mọi sổ đang mang sai số ~0.0005/chiều** ngay hôm nay.
5. "Đã đo overlay-OFF cho NAV thấp hơn rõ rệt" — **chưa hề đo**; `presets.py` chưa tồn tại.
6. "Local nav 1.3737 vs container 1.3628" — không có dấu vết trong repo.
7. "Đường cong tầng 1 của wavestruct ~30% / −3.4%" — nửa `136.4% / −16.4%` là thật (từ `portfolio.db`); nửa tầng 1 không kiểm được.
8. "Tầng 1 báo 10 vị thế mở cùng ngày" — `open_positions()` không được persist.
9. Collision `config_hash`: **146** run_name cho `6933813844a6713e`, lớn nhất là `2c78c568f85dc826` với **360** — không phải 35.
10. Pop-list: **37** lệnh `pop` trong 3418-3462 (**39** trong 3416-3474), phần thêm là `costs` (3462), loop commission/tax/slippage phẳng (3468-3470), `slippage_model` (3472).
11. GC: nguyên nhân là `results/experiments` **không tồn tại** + glob 2 cấp vs layout 3 cấp — **không** phải `CacheKeys` từ DB.
12. `cache_gc.py` fail ở **runtime dòng 116**, không phải lúc import (`--help` exit 0).
13. Citation sai đã sửa: `Dockerfile.api` là `infrastructure/docker/Dockerfile.api:29` (không có file ở root) · md5 pin của golden assert ở `test_portfolio_golden.py:57-64` từ `goldens/champion_prod_overlay.json['data_md5']` (không phải `:99-100`) · `superseded`/`state` ở `leaderboard_adapter.py:67-68` · sort ổn định ở `sim.py:44` (không phải `:39`) · fallback `C.skip` ở `gates.py:55-56` + default 0.40 ở `constants.py:27` · `PortfolioConstants` **29** field · `engine.py` **4360** dòng ở 0.3.4, **2824** ở 0.4.0+ · EOL: **85 trong 115** file .py chung khác nhau chỉ ở EOL · `_champ_prod_replay.py` **có** tracked · không có untracked parquet nào (`.gitignore:80` blanket) — rác thật là ~20 CSV dưới `_champ_src/`.

---

## 9. Bổ sung sau vòng review (2026-07-30)

> Review đối kháng trên chính tài liệu này. Spot-verify lại 2 claim trụ: `engine.py:44-46` đúng là trả `None` im lặng theo CWD; `garbage_collector.py:65-66` trả set rỗng khi `results/experiments` không phải thư mục ⇒ "bẫy bất đối xứng" §1.6 xác nhận. Phần dưới KHÔNG thay đổi chẩn đoán §1–§8, chỉ sửa **thứ tự**, chốt **3 quyết định còn treo**, và ghi **vấn đề mới phát hiện** ngoài §1–§8.

### 9.1 ⛔ ĐÃ XOÁ — thứ tự cũ (§1.5 + R0 + R6 lên P0)

Từng đề xuất đưa §1.5/R0/R6 lên P0. **§11.6 đã đảo ngược** (data-first; R0 bỏ, R6→§11.5.3). Không còn nội dung sống, không có số đo. Thân bài xoá 2026-07-31.

Bản đầu xếp §1 trước rồi mới §2. Nhưng đường tới hạn thật là **phụ thuộc ngược**: các fix §1.1–1.4 đổi hành vi trên đúng code path production, mà "guard" để chứng minh fix đúng lại nằm ở §1.5 (CI phải chạy — hiện trỏ path không tồn tại) và R0/R6 của §2 (luật must-not-ship + golden full-chain trên slice đã commit). Vá §1.1–1.4 khi chưa có ba thứ này = vá mù. ⇒ Nâng **§1.5, R0, R6 lên P0**, làm trước §1.1–1.4 (đã cập nhật dòng thứ tự ở §0).

### 9.2 Ba quyết định còn treo — chủ dự án phải chốt TRƯỚC khi bắt tay

**D1 — Có re-export sạch 5 bundle đang deploy không?** (mâu thuẫn R0 ↔ R8). R0 đóng băng 5 bundle làm tham chiếu; R8 thừa nhận provenance của chúng **sai không sửa hồi tố** (`git_sha=60a23062…` không chứa code `universe_policy` đã sinh ra chúng). Đóng băng một reference mù thì R7 sẽ **chứng thực mật mã cho một commit sai**.
✅ **CHỐT (2026-07-30): CÓ** — re-export sạch cả 5 bundle từ wheel đã tag (sau R8), làm mốc gốc R0. **Bắt buộc:** đối chiếu số của model mới với bản đang chạy; nếu lệch, công bố lại số cho các sổ bị ảnh hưởng **trước khi** thay.

**D2 — Migrate danh tính thế nào khi đổi `config_hash` (§3) + fingerprint fold (§1.3a)?** Cả hai đổi công thức hash ⇒ **mồ côi hàng loạt**: `run_id = template/{run_name}-{config_hash[:8]}` nên đổi hash = đổi danh tính **3612 hàng** `leaderboard_runs` + FK sang `run_signals` (368M dòng) + **5043 fold dir** fingerprinted + `_seed_serving_tiers.py:56-59` đọc run_id CỨNG của dyn300/dyn900 để dựng sổ tier live. §1.3 và §3 là **cùng một bài toán mồ côi**, phải giải một lần.
✅ **CHỐT (2026-07-30): làm bảng phiên dịch `old_run_id ↔ new_run_id`** để leaderboard liền mạch cả cũ lẫn mới — **KHÔNG tính lại hash của hàng cũ**, chỉ ánh xạ. Ràng buộc cứng giữ nguyên: **tuyệt đối không rewrite run_id của 2 bundle tier live** khi `_seed_serving_tiers.py` còn trỏ vào chúng. Chi phí thêm so với "chỉ-run-mới": dựng + kiểm bảng ánh xạ, và một test khẳng định mỗi `old_run_id` map đúng **một** `new_run_id`.

**D3 — R7 attestation phát gì cho 3 bundle self-train?** (`git_sha="serving-build:0.3.3"`, `template_id=None`) — chúng **không có `run_signals`** để diff, nên "attest thật thà" đầu tiên sẽ đọc như một sự cố.
✅ **CHỐT (2026-07-30): cho 3 bundle top-150 chạy qua pipeline chuẩn** để sinh `run_signals`, để **cả 5 bundle attest cùng một kiểu** (diff-vs-ranked-run) — hợp với D1. ⚠️ **Điều kiện tiên quyết phải kiểm TRƯỚC:** pipeline train tái tạo được đúng 3 bundle top-150 (khớp data + config). Nếu KHÔNG tái tạo được thì lùi riêng 3 cái đó về `mode=SELF_BASELINE` (diff bundle-vs-chính-nó dưới cùng wheel), và `deploy.py` chấp nhận `SELF_BASELINE` khác `FORCED`.
🔎 **Kết quả kiểm khả thi (2026-07-30).** Cả 3 template **TỒN TẠI trong Postgres**: wavestruct = `2646 n2_2643_wavestruct_la05_lamp02` (có producer `deploy_wavestruct.py`, đã có run đăng ký), consw20 = `2429 n2_consw20_conv04_vg_combo_hb_nbpbw`, velov = `2234 n2_velov_univ150c` ⇒ `run_template_experiment(template_id=…)` chạy được ngay, **không cần viết producer mới**. NHƯNG đây **không phải tái tạo trung thực**: 3 bundle live được `serving/build_bundle.py` train trên **data API non-back-adjusted của serving** + **universe list phẳng top-150**, còn template train chạy trên **DuckDB back-adjusted** + `universe=vn_stock_default` (docstring `build_bundle.py` tự ghi *"NOT bit-identical… serving-owned"*; list 150 mã **không** nằm trong `config.json`). ⇒ Chạy pipeline sẽ ra model **khác số** (khác data + khác universe) — tức đây là **re-baseline** đúng tinh thần D1, không phải khớp-bit. **Hai việc phải làm trước khi attest có nghĩa:** (1) **pin list phẳng top-150 thành input tường minh có version** (chính là lỗ provenance §1.2 — hiện list này không nằm ở config); (2) ⛔ ~~chốt data basis = DuckDB back-adjusted của train~~ → **SAI theo D6/§13.3**: basis **công bố** = store serving (sieutinhieu, nguồn chuẩn duy nhất); DuckDB local của **cả hai repo** là bản thứ cấp, chỉ dùng train/test, **không dùng để phục vụ**. Vẫn **công bố lại số** cho 3 sổ này (re-baseline). Nếu KHÔNG muốn đổi số của 3 model live thì buộc dùng `SELF_BASELINE` cho chúng.

### 9.3 Vấn đề mới phát hiện (ngoài §1–§8)

1. **§1.1 fail-loud có bán kính nổ ở HOST.** Đổi silent-None→raise sẽ làm **85 script research dưới `results/`** (đang chạy trên host nơi file CÓ mặt) và mọi run host thiếu file **crash ngay**. Fail-loud là đúng cho container, nhưng cần đường opt-in (env `MARKET_CONTEXT_REQUIRED=1`, mặc định bật trong container) hoặc materialize VNINDEX vào `market.duckdb` trước khi bật, nếu không P0 fix tự nó thành sự cố dev.
2. **§1.3 chọn (a) không phải (b).** Bỏ read-through cache (b) = **retrain toàn bộ** — 51 GB fold cache tồn tại chính để tránh việc đó. Chọn (a) đưa toàn bộ config vào `_fp_src`; chấp nhận nó invalidate 5043 fold dir cũ (gộp vào D2).
3. **An toàn migration Postgres 98 GB.** §4.1 (0030), §3 (config_hash), §4.3 (lifecycle) động vào DB live không dựng lại được (§5 ghi `leaderboard_nav`/`leaderboard_seed_stats` **không ORM/không migration create**). Kế hoạch thiếu bước **backup + dry-run trên bản sao** trước mọi migration. Bắt buộc thêm.
4. **Chốt R6 đa nền tảng — đừng để "hoặc".** "Một fixture mỗi platform" nhân đôi bảo trì mà vẫn không phủ container-vs-ubuntu. ⭑ **Chốt: CI/Linux/container là mặt phẳng recompute golden DUY NHẤT; máy Windows là advisory, không bao giờ authoritative** (hệ quả trực tiếp của lệch float VND 40 vs 37 đã đo).
5. **Cross-repo chưa có giao thức khóa version.** R9/R2 và nhiều mục §4 là CROSS-REPO nhưng không định nghĩa cách deploy đồng bộ train↔serving khi đổi facade — đúng vào luận đề "ranh giới một-wheel là hư cấu". Cần release gate: serving pin `stock_ml_core==<version>` chính xác, và đổi facade phải bump MINOR + cập nhật pin trong cùng một PR-cặp.
6. **Không có đường lùi khi wheel xấu đã ship.** R0 nói "must not ship" nhưng tiền lệ `deploy_log.jsonl` ghi `"parity":"FORCED"` cho thấy guard bị bypass dưới áp lực. Cần: `deploy.py` giữ N wheel trước + lệnh `rollback --to <wheel-tag>` một bước, và cấm `FORCED` trừ khi có biến môi trường break-glass có log.
7. **Chương trình này không có ước lượng công / chủ sở hữu.** R0–R9 + §1 + §3 + §4 là một chương trình lớn chạm 2 repo; §6 chỉ có perf-estimate. Trước khi bắt tay nên gắn size (S/M/L) + owner cho từng mục, ít nhất cho tập §9.4.

### 9.4 ⛔ ĐÃ XOÁ — tập con "minimum viable safety"

Bản thứ tự **thứ ba** trong doc này, dựng quanh R0/R6 (mục tiêu cũ). Cả 4 mục của nó đã nằm trong §11.6.
⚠️ Nó còn **mâu thuẫn thật** với §11.6: xếp §1.6 (GC) hạng 2 trong khi §11.6 đặt §1.6 ở Phase 5 — hai chỉ dẫn ngược nhau về cùng một việc. Đã xoá thân bài 2026-07-31; câu suy luận duy nhất §11.6 chưa có đã **chuyển sang §11.6**.

### 9.5 Nit

- **§0**: đã gộp §4.1 = B3 (không liệt kê hai lần); đã sửa dòng thứ tự.
- **Tiêu đề vs nội dung**: §5 kết luận repo gần như không có rác an toàn để xoá (~2.6 MB). Giá trị "tái cấu trúc di sản" thật nằm ở **§4.5/§4.6** (thống nhất danh tính module `src.*`↔`stock_ml.src.*` + path arithmetic), không phải xoá file — cân nhắc đổi tên tài liệu cho khớp.
- **Nâng hạng §4.5**: đây là **lỗi hợp đồng wheel** (`import stock_ml.src.model_dashboard` raise `ModuleNotFoundError` trong môi trường sạch dù module nằm trong namelist 0.4.2 serving cài), chạm production — nên xử cùng hạng §2, không phải "di sản §4".

### 9.6 Nguyên tắc attestation (làm rõ theo yêu cầu chủ dự án, 2026-07-30) — tách "chất lượng data" khỏi "toàn vẹn engine"

> 📌 **Hợp đồng chính thức của test này là §11.5.3 (so tương đương hai chiều, thay R6).** §9.6 dưới đây là **chi tiết kỹ thuật/bằng chứng** (call-graph, 3 seam, điều kiện chống PASS-giả) nuôi cho §11.5.3 — khi lệch, lấy §11 làm chuẩn.

Chủ dự án chốt cách hiểu attestation, sắc hơn cả R7 lẫn D3 ban đầu. **Hai trục phải tách bạch:**

- **Chất lượng data** (vd train trên DuckDB back-adjusted thay vì API non-back-adjusted): số CAGR/NAV **được phép đổi** — đó là cải thiện thật, KHÔNG phải thứ cần "khớp". Re-baseline (D1) công bố lại số là đủ.
- **Toàn vẹn engine qua wheel**: khi xuất engine qua wheel sang production, **không được thiếu/rớt tầng sinh tín hiệu nào**. Trục này phải khớp **bit** — và chỉ đo được khi **giữ data cố định**.

**Định nghĩa test (thay khung "diff-vs-ranked-run" của R7 cho mục đích này), theo từng bundle:**
1. Đóng băng **một ảnh chụp `as_of` từ sieutinhieu** — nguồn chuẩn duy nhất (§13.3/§13.4) — và dùng **CHUNG cho cả hai bên**. ⛔ ~~(dyn → DuckDB back-adjusted của train; 3 top-150 → OhlcvStore serving)~~: hai basis khác nhau thì phép so tương đương **mất nghĩa** ngay từ đầu.
2. Chạy engine phía **train** trên tập đó → `signals_A`.
3. Chạy engine phía **production/serving (qua wheel)** trên **đúng tập bytes đó** → `signals_B`.
4. `signals_A == signals_B` ⇒ QUA. Lệch ⇒ diff chỉ thẳng tầng bị rớt.

**Ba điều kiện — thiếu là PASS-GIẢ:**
1. ⚠️ **Phía production phải chạy TRONG CONTAINER thật, không phải wheel trên host.** Các tầng bị rớt (VNINDEX §1.1, runscore §1.1, breadth/xsec §4.5) chỉ biến mất trong container do thiếu file/mount — chạy trên host (nơi file có mặt) sẽ **PASS giả** trong khi prod vẫn rớt tầng. Test phải phơi được khác biệt môi trường.
2. **Cùng một tập bytes cho cả hai engine.** Hai bên đang đọc 2 store khác nhau; phải có adapter cho một bên đọc dữ liệu bên kia, hoặc commit một slice OHLCV chung (R6), nếu không lại trộn trục data vào và mất tính cô lập.
3. **Chốt một mặt phẳng float chuẩn** (Linux/container); Windows chỉ tham khảo (lệch VND 40 vs 37 đã đo).

**Hệ quả với D3:** cách này **bỏ phụ thuộc vào `run_signals` lịch sử và không cần re-baseline chỉ để attest**. Attestation = "engine-train-path == engine-container-serving-path trên cùng slice + cùng config" cho từng bundle ⇒ 3 bundle self-build không còn phải đăng ký lại vào DB chỉ để có cái đem diff. `SELF_BASELINE` ở D3 nay được hiểu chính là test này (self = cùng data, khác đường engine), không phải một chế độ yếu hơn.

**Test này bắt đúng 7 tầng có thể rớt âm thầm giữa train↔serving:** market-context VNINDEX (§1.1) · runscore modulator (§1.1) · breadth/xsec cache chọc qua biến private (§4.5/R9) · default recombine đọc literal khi bundle không pin (§2.2 Đ2) · config key serving âm thầm drop (§2.1) · head entry6 không export (§2.1/R5) · `max_hold` serving tự chèn (R3).

✅ **Đã tra call-graph runtime serving (2026-07-30) — chốt cặp so sánh A/B.** Cả 5 bundle (kể cả 3 top-150) đi **cùng một đường serving duy nhất**; engine hai bên **là cùng code wheel**, KHÔNG có engine riêng cho top-150 — nên nguồn lệch không bao giờ là "khác engine", chỉ là seam môi trường/wrapper:
- **Tầng tín hiệu:** `serving/engine.py:236` → `generate_signals_from_bundle` (wheel `inference.py:27`) → `build_feature_frame` + `predict_slot_signals` + `recombine_signals` — **cả ba import từ module train** `pipeline.experiment` (đúng §10.1, không có bản serving riêng). Tầng này **không có biến đổi serving-only** ⇒ trên cùng data+config phải **khớp tuyệt đối**; lệch = rớt tầng thật (breadth/xsec đọc trong `build_feature_frame`).
- **Tầng lệnh:** `serving/trades.py:70 derive_trades` → `run_backtest` + `trades_to_dataframe` (wheel `backtest/engine.py`), NHƯNG có **hai biến đổi chỉ-serving**: (i) `trades.py:56` lọc `{k if k in EngineConfig-fields}` → **âm thầm drop** field engine bị đổi tên giữa các wheel; (ii) `trades.py:65-66` **chèn `max_hold`** vào `exit_priority` (R3). `run_backtest` còn đọc VNINDEX + runscore từ file (§1.1).

⇒ **Cách chạy test cho đúng:** tách 2 bước. (1) So `signals` hai bên — phải bit-khớp, vì tầng này không có wrapper serving. (2) So `trades` hai bên **sau khi cho cả hai cùng áp `max_hold` + cùng bộ lọc config** (whitelist đúng 2 biến đổi cố ý); lệch còn lại = rớt tầng thật. Tách signal-diff khỏi trade-diff sẽ localize đúng seam.

⚠️ `backtest/portfolio_engine.py` (§10.2, "engine danh mục #2") **KHÔNG** nằm trên đường serving (importer duy nhất là 1 test) ⇒ không ảnh hưởng test này, nhưng vẫn là landmine toàn-vẹn-wheel phải loại.

#### 9.6.1 Giới hạn của test này — nó là guard chống HỒI-QUY, không phải bằng chứng ĐÚNG-ĐẮN

Đánh giá đối kháng (2026-07-30). Test §9.6 nền tảng đúng, nhưng **có trần và mù ở vài chỗ** — ghi rõ để không cho an toàn giả:

1. **Chỉ chứng minh "serving ≡ train", KHÔNG chứng minh "serving đúng".** Engine tín hiệu là **cùng một hàm** ⇒ bug **đối xứng** ở cả hai bên (lỗi trong `recombine_signals` chung) sẽ PASS vui vẻ. Đây là kính chiếu hậu chống hồi-quy, không phải chứng nhận tín hiệu đúng.
2. **"Bit-khớp" trộn hai thứ:** rớt-tầng (cần bắt) và **runtime-lib-drift** (sklearn 1.3.2 baked vs 1.8.0 pin, đủ đổi lệnh VND 40↔37; float Windows↔Linux). Mọi dung sai ε lại **che được một cú rớt-tầng nhỏ hơn ε** (đúng bài học "round 8dp giấu cú lật biên clip"). Mâu thuẫn trung tâm phải chốt bằng chính sách float.
3. **"Cùng bytes cho hai engine" cần adapter/slice chung** — code mới chưa audit, dễ lệch tz/dtype; và test trên dataset-cầu-nối là **cấu hình không chạy ở prod** ⇒ khe giữa đường-data-test và đường-data-prod.
4. **Whitelist bộ lọc `trades.py:56` = lỗ**, vì đó chính là cơ chế silent-drop §2.1: whitelist nó thì **không phân biệt** "bỏ đúng key recombine-only" với "bỏ nhầm field `EngineConfig` đổi tên". Bắt được vế sau **đòi R1** (config đã resolve). ⇒ **§9.6 KHÔNG thay R1**; hai cái bổ sung nhau.
5. **Độ phủ chỉ bằng slice:** diff end-to-end chỉ bật gate mà data slice kích hoạt (mkt_skip cần VNINDEX>MA+bullish, overext clamp, head entry6…) ⇒ tầng không được kích sẽ **im lặng dù chạy trong container**. Cần slice thiết kế để kích đủ gate, hoặc assert per-seam.
6. **Không phủ overlay/portfolio** — chỉ signal+trade per-symbol; NAV khách nhìn nằm ở tầng overlay, đúng chỗ test này không chạm (và §10.2 ship **hai** portfolio engine).
7. **Rủi ro bị bypass:** test nặng (container-only, Linux-only, cần cầu-nối) ⇒ dưới áp lực dễ `parity=FORCED`. Giá trị phụ thuộc `deploy.py` **từ chối FORCED / wheel-mismatch** (R7).

**Ba điều kiện BẮT BUỘC ghép cùng §9.6, nếu không nó cho an toàn giả:**
- **(a) Vẫn làm R1** (snapshot config đã resolve) — bắt vế "lọc nhầm field" mà §9.6 mù (điểm 4).
- **(b) Test-song-sinh cho overlay** — cùng nguyên tắc trên `stock_ml.portfolio`, và loại engine danh mục #2 khỏi wheel (điểm 6).
- **(c) Chốt chính sách float**: Linux/container là mặt phẳng exact-match **duy nhất**, và **pin cả sklearn + numpy** (R2) để loại lib-drift khỏi diff (điểm 2) — nếu không mỗi bump wheel test đỏ vì lý do sai và sẽ bị phớt lờ.

⇒ Sửa framing: §9.6 **bổ sung** R1/R2/R7 và test overlay, **không thay** chúng. "D3 tan" chỉ đúng ở phần "khỏi cần `run_signals` lịch sử", KHÔNG có nghĩa bỏ được R1.

### 9.7 Chính sách data — nguồn chuẩn duy nhất là sieutinhieu (chốt 2026-07-30, giải nút #1)

Chủ dự án chốt mô hình quản trị data:
- **sieutinhieu (server chính thức) = nguồn chuẩn DUY NHẤT, luôn mới.** Mọi bản khác (train local DuckDB, store serving) là **phái sinh, ĐƯỢC PHÉP cũ** — staleness KHÔNG phải lỗi.
- **Fetch một chiều:** train kéo từ sieutinhieu về local để train/xử lý; local cũ dần theo thời gian, chấp nhận.
- **Lỗi data → BÁO chủ dự án để vá TẠI NGUỒN (sieutinhieu), KHÔNG vá cục bộ.** Vá cục bộ che lỗi nguồn và không lan sang consumer khác; sau khi vá nguồn, bản phái sinh nhận qua lần fetch kế.

⇒ **Thay đề xuất cũ** ("cho serving chạy `back_adjust.py` cục bộ") — vá cục bộ đúng là thứ chính sách này cấm.

**Tension còn treo (một sub-decision):** loader train hiện **clamp/sanitize OHLC-violation + CA-jump ngay trong bộ nhớ lúc load** (đo: HDG 0.909×, AAS 0.844→1.0 tiền-ex-date) trong khi serving đọc **raw** ⇒ chính cái clamp này là **một bản vá cục bộ** đúng loại vừa cấm, và là nguồn lệch 2 mã ~10% giữa train↔serving. Để hai engine hội tụ data, phải chọn cho **phần điều chỉnh corporate-action**:
- (a) đưa điều chỉnh **về nguồn** (sieutinhieu phục vụ giá đã điều chỉnh) → mọi consumer nhất quán tự động; hoặc
- (b) một **bước phái sinh dùng chung, tường minh, có version** mà **cả train và serving cùng chạy y hệt** sau khi fetch raw.
Bar rác thật (VFC −100%) luôn thuộc diện "báo → vá nguồn".

✅ **CHỐT sub-#1 (2026-07-30): (a) SỬA TẠI NGUỒN.** sieutinhieu **phục vụ được giá đã điều chỉnh** ⇒ chọn (a): điều chỉnh corporate-action **tại nguồn**; mọi consumer (train + serving) nhận giá đã-điều-chỉnh nhất quán. **Hệ quả bắt buộc:** BỎ bước clamp/sanitize in-memory của loader train (nay thừa, sẽ **double-adjust** nếu nguồn đã chỉnh) — cả hai bên cùng ăn một nguồn đã chỉnh. "Điều chỉnh tại nguồn" áp đúng taxonomy: chỉ split/dividend thật + bar-rác; **giữ nguyên NON_ADJUSTABLE rights-issue**. Claude **báo mọi lỗi data phát hiện** để chủ dự án vá tại sieutinhieu (chính sách §9.7).

🔎 **Cập nhật 2026-07-31 (chủ dự án xác nhận):** nguồn sieutinhieu **ĐÃ** phục vụ giá điều chỉnh ⇒ nút thượng nguồn đóng. Việc còn lại thuần **phía repo**: (1) **bỏ clamp/sanitize in-memory của loader train** (nay thừa, sẽ double-adjust); (2) kiểm các đường cong đang lưu tính **trước/sau** bản vá — trước thì tính lại. Chi tiết ở §13.7. Claude vẫn kiểm chứng dữ liệu; phát hiện lỗi thì báo để vá tại nguồn.

⚠️ **Taxonomy khi báo lỗi:** chỉ báo lớp **CA-thật-chưa-adjust** và **bar-rác**; KHÔNG báo lại lớp **NON_ADJUSTABLE (rights issue)** — DB đã khớp Fireant+SSI, lớp này từng bị detector gắn cờ nhầm.

**Khớp §9.6:** "đóng băng một dataset feed cả hai engine" chính là một **snapshot phái sinh từ sieutinhieu** tại một thời điểm — nhất quán với chính sách; staleness của snapshot vô hại vì test đo engine-integrity, không đo độ-mới.

### 9.8 Làm rõ kiến trúc engine + tương thích + runtime-lib (chốt 2026-07-30, nút #2/#3/#4)

**#2 — Mục tiêu THẬT của việc tách engine (không chỉ "tách bộ sinh tín hiệu").** Dự án train hàng nghìn chiến lược để lọc ra cái tốt; khi có chiến lược tốt cần **xuất một engine gọn phục vụ được nó** mà KHÔNG phải bê cả dự án + hàng nghìn model không dùng. Đơn vị xuất = **engine (lõi tái dùng) + file cấu hình từng model**, tái tạo **tín hiệu và giao dịch y hệt dự án gốc**. Nâng cấp = thay engine bản mới, giữ file cấu hình; engine mới **cộng thêm tính năng** (vd thêm tầng danh mục cho loại chiến lược mới) nhưng **vẫn chạy được model cũ**.
⇒ **SỬA (dọn theo §11, 2026-07-30):** mục tiêu #2 **đã đạt bằng wheel** — wheel *chính là* engine trích xuất được; hàng nghìn model nằm ở DB/results, **không** trong wheel. Nên **§10.1(a) chỉ là dọn cho sạch, TUỲ CHỌN, làm cuối** (§11.6 bước 8), **KHÔNG phải điều kiện cần** như bản đầu viết. Cái #2 thật sự cần = **bản xuất tự đủ** (§11.5.1: `resolved.json` mang đủ 227 nút engine + 35 recombine + file ngoài + wheel version) để production tái lập đúng cái xưởng chấm; golden tái tạo = **§11.5.3** (không phải R6, đã bị thay).

**#3 — Hợp đồng tương thích.** ⚠️ **Bản chính thức ở §11.2/§11.3/§11.4 — đọc §11, KHÔNG dùng đoạn dưới nếu lệch.** Tóm đúng §11: phân biệt **tương thích CẤU HÌNH (bắt buộc — engine mới nạp+chạy được mọi config sống)** vs **tương thích SỐ (KHÔNG cần — engine tốt hơn thì số phải đổi)**; production chạy **một engine tại một thời điểm**, nâng engine ⇒ **tính lại cả **10** (§14.1)**. Cơ chế = **test tương thích cấu hình (§11.5.5) + ghim-wheel-chính-xác & từ-chối-khi-đổi (§11.5.2) + re-baseline-một-lệnh (§11.5.4)**. **KHÔNG dùng R0** (đã bị §11.4 bỏ — bản đầu tôi viết sai khi dựa vào R0); cũng không "additive + freeze" — chỉ cần *không đổi thước âm thầm*, còn số đổi là chấp nhận được.

**#4 — Runtime-lib version: có vấn đề THẬT, đo được (2026-07-30).**
| | scikit-learn | numpy | lightgbm |
|---|---|---|---|
| bundle **dyn900** baked | **1.3.2** | 1.26.3 | 4.6.0 |
| bundle wavestruct baked | 1.8.0 | 1.26.4 | 4.6.0 |
| serving pin (`requirements.txt`) | **==1.8.0** | >=1.26,<2 | >=4,<5 |

⇒ **dyn900 train bằng scikit-learn 1.3.2 nhưng phục vụ dưới 1.8.0** — lệch 5 minor, **vi phạm chính comment của serving** ("MUST match the bundle's training version"). Model là LightGBM nhưng đường predict qua wrapper sklearn (`LGBMRegressor`); unpickle/chạy chéo version sklearn **không được hỗ trợ chính thức** → có thể đổi dự đoán ở lệnh biên. Đúng ca "model cũ KHÔNG chạy đúng trên môi trường mới" — mâu thuẫn thẳng #2/#3.
⇒ Fix sạch = **R2**: ghi version lib từng model (đã có sẵn `lib_versions` trong manifest) và lúc load **hoặc cấp đúng version hoặc gate/refuse**, như đang gate lightgbm. Chính R2 **làm cho "model cũ vẫn chạy đúng" thành sự thật**, không phải mối lo trừu tượng.
(Phần "float Windows↔Linux" chưa tái lập được tại chỗ → xử như vệ-sinh: chốt một nền chuẩn Linux/container cho golden, không cần chứng minh riêng.)

### 9.9 Bằng chứng universe: truy nguồn danh sách 150 mã + hai chế độ universe

> ⛔ **ĐÃ BỊ §11.6 THAY (2026-07-30).** Thứ tự A→G dưới đây đá nhau với §11.6 (bản chốt data-first, bỏ R0, R6→§11.5.3) — tôi viết nó khi chưa thấy §11.6. **Dùng §11.6.** Riêng phần **truy nguồn 150 mã** + **hai chế độ universe** bên dưới vẫn đúng, đã được dẫn vào Phase 1 của §11.6.

⛔ **Danh sách phase A→G đã XOÁ (2026-07-31).** Đối chiếu từng mục: **không mục nào vắng mặt trong §11.6** — nó là bản sao thuần, và chính đầu mục đã tự ghi "A→G đá nhau với §11.6". Vế **attestation** của "Bất biến xuyên suốt" đã **chuyển sang cuối §11.6**. Phần đo được bên dưới (truy nguồn 150 mã + hai chế độ universe) **giữ nguyên** — §11.6 chỉ trỏ tới chứ không chép.

🔎 **Gỡ blocker phase B — truy nguồn danh sách 150 mã (2026-07-30).** Danh sách **KHÔNG mất**: `_load_universe` (serving `engine.py:39`) đọc `manifest['universe']`, và `build_bundle.py:220` ghi list phẳng vào đó. Đo được: mỗi bundle top-150 có `manifest['universe']` = **150 mã**, `feature_scope='traded'` (dù `config.json` ghi `None` ⇒ **manifest mới là nguồn thật**, và đây cũng là ví dụ config↔manifest lệch mà manifest chưa được hash — §2.1). Có **2 list phân biệt**: **wavestruct ≡ consw20** (150 mã giống hệt), **velov (univ150c) KHÁC** (chỉ chung 109/150 — chọn theo quy tắc "causal" khác). ⇒ Phase B **không phải "khôi phục list đã mất"** mà là: (1) **pin 2 list này NGUYÊN TRẠNG** làm input có version (chúng là ground-truth của cái đã chạy) + **ký/checksum** (R4 — manifest hiện chưa hash nên universe/scope chưa tamper-evident); (2) **ghi lại quy tắc dẫn xuất** (slug `vn_stock_default` → 150: as-of date + tiêu chí ADV) để tái chọn về sau — **KHÔNG re-resolve từ slug** vì universe trôi theo thời gian và top150≠univ150c chứng tỏ hai quy tắc khác nhau.

⚠️ **Chẩn đoán gốc chính xác (đo 2026-07-30): defect KHÔNG phải "slug trôi" mà là FIELD UNIVERSE CỦA CONFIG KHÔNG TRUNG THỰC.** Cả **3** `config.json` mang universe **giống hệt từng chữ** `{"mode":"db","slug":"vn_stock_default"}` — không field phân biệt — nhưng universe thật khác nhau (wavestruct≡consw20 = 150 mã; velov ra 150 mã khác, chung 109/150). ⇒ **Từ config KHÔNG suy ra được velov chạy universe khác wavestruct** — phá thẳng mục tiêu #2 ("config tái tạo model y hệt"). Nhiều khả năng do build tái dùng template config mà quên sửa field universe, để manifest/CLI gánh universe thật (nợ kỹ thuật). **Fix theo #2:** config phải khai universe **xác định + tái tạo được** (velov = "causal-150 + as-of"; top150 = "ADV-150 + as-of"); khi đó `manifest['universe']` chỉ còn là **cache kiểm chứng**, không phải nguồn duy nhất. Đây là điều kiện để "một **chiến lược** = một config đủ tái tạo" (đối tượng = chiến lược, không phải model — một model dùng lại được trong nhiều chiến lược; xem §12 mô hình đối tượng chuẩn).

🔎 **Universe nằm đâu, chuẩn chưa — HAI CHẾ ĐỘ (đo 2026-07-30).**
| Loại | Config khai | Sự thật | Chuẩn? |
|---|---|---|---|
| **Động** (dyn300/dyn900) | slug quy tắc đầy đủ `dyn_topn:n=300,metric=adv,lookback=prior_year,min_sessions=100` + `universe_policy` | `universe_resolver.py` (đã tracked) giải **causal, deterministic, point-in-time** → per-year; manifest cache `universe_by_year` (2020–2025) | ✅ CHUẨN — config đủ tái tạo |
| **Tĩnh** (wavestruct/consw20/velov) | slug `vn_stock_default` | `universe_sets` id=8 `vn_stock_default` = **61 mã**, nhưng bundle chạy **150 mã**; **không có set 150 nào** trong DB | ❌ SAI |

⇒ Bundle **động là hình mẫu chuẩn**; bundle **tĩnh SAI**: `vn_stock_default`=61≠150, nên 150-list **không đến từ config** (gần chắc truyền qua CLI/file lúc build, không ghi lại), chỉ tồn tại ở `manifest.universe` (chưa hash), **không có định nghĩa versioned**. **Tái cấu trúc (kéo tĩnh về khuôn động, hạ tầng `universe_sets`/`universe_versions` đã có sẵn cột `version`/`is_locked`):** (1) universe trong config = tham chiếu **xác định + có version** — rule-slug tái tạo được (`static_topn:n=150,metric=adv,asof=…`; biến thể causal cho univ150c) **hoặc** set đăng-ký-khóa (`vn_top150_adv@v1`, `vn_univ150c@v1`); (2) **sửa slug sai** — 3 config tĩnh thôi ghi `vn_stock_default`; (3) `manifest.universe` = cache có checksum (R4); (4) làm **ngay trong D1 re-export**. Ghi vào phase B của §9.9.

---

## 10. Phân tầng engine — đo trực tiếp 2026-07-30

Khai báo là 2 tầng (tín hiệu + danh mục). Wheel production chứa **bốn** engine:

| Engine | Ở đâu | Quy mô | Đánh giá |
|---|---|---|---|
| **Tín hiệu** (`recombine_signals`) | `stock_ml/src/pipeline/experiment.py` — **module TRAIN** | 35 lệnh `cfg.engine.get(...)` | ❌ sai vị trí; không golden nào trên bundle thật |
| **Lệnh** (`run_backtest`) | `stock_ml/src/backtest/engine.py` | 2824 dòng, config 227 field | ⚠️ thống nhất (wheel == source, đã verify) nhưng phụ thuộc file ngoài theo CWD + suy biến im lặng |
| **Danh mục** (`run_portfolio`) | `stock_ml/portfolio/` (9 module) | `PortfolioConstants` 29 field | ✅ sạch nhất — một implementation, golden pin |
| **Danh mục #2** ⚠️ | `stock_ml/src/backtest/portfolio_engine.py` | 486 dòng | ❌ bản cũ song song, **vẫn được đóng gói** |

### 10.1 Engine tầng tín hiệu nằm trong module TRAIN — lỗi phân tầng, không phải lỗi code

Đo được: `stock_ml/src/serving/inference.py:14-20` import `build_feature_frame`, `predict_slot_signals`, `recombine_signals` từ `stock_ml.src.pipeline.experiment`. `stock_ml/src/serving/` **chỉ có** `bundle.py` + `inference.py`; `stock_ml/src/backtest/` **không có** module tín hiệu nào.

⇒ **Serving không thể sinh một tín hiệu mà không nạp pipeline train.** "Serving độc lập với train, chỉ qua wheel" không bị vi phạm do lập trình cẩu thả — nó **bất khả về cấu trúc** với layout hiện tại. Đây là lý do **R9 (facade) một mình không giải quyết được**: export thêm symbol ra facade không làm `recombine` thôi nằm trong module train.

**Đích — chọn một, và ghi vào hợp đồng:**
- (a) Tách `predict_slot_signals` + `recombine_signals` (+ phần `build_feature_frame` mà inference cần) ra **một module tín hiệu riêng** (vd `stock_ml/src/signal/`) mà **cả** train và serving import. Sạch, nhưng là refactor thật.
- (b) **Thừa nhận** và ghi rõ: wheel `stock_ml_core` **bao gồm cả pipeline train**, và hợp đồng là "một wheel chứa 3 tầng", không phải "serving không chạm code train". Rẻ, trung thực, nhưng bỏ luôn ranh giới.

Không chọn thì mọi tài liệu tiếp theo sẽ tiếp tục khẳng định một ranh giới không tồn tại.

#### 10.1.1 Đặc tả Mức 2 (chủ dự án chốt (a) + phân loại, 2026-07-31)

**Nguyên tắc nền:** *"KHÔNG cần ở serve" ≠ "chết".* Code research/train **đang SỐNG** (dùng để train hàng nghìn chiến lược). Mức 2 = **TÁCH** research khỏi bản production, **KHÔNG xoá**. Không được cắt wheel-gọn-riêng cho production (sẽ thành **hai engine** ⇒ phá "một engine" §11.3); phải sắp-xếp-lại thành **lõi chung** mà cả train lẫn serve import y hệt.

**LÕI ENGINE (ship production — bằng chứng: đường serve chỉ chạm các module này):**
- `core` (facade) · `src/serving/` (bundle, inference) · `src/backtest/engine.py` (**tự chứa, 0 import stock_ml**) · `portfolio/` (9 module, tự chứa).
- Module tính toán lõi cần: `src/features`, `src/models`, `src/signals`, `src/signal_adapter`, `src/data/splitter`, util chung (`env`, `safe_io`, `seed`, `contracts`, `utils`).
- **Phần TÍN HIỆU-LÕI của `src/pipeline/experiment.py`** (`build_feature_frame` / `predict_slot_signals` / `recombine_signals`) — sau khi **XẺ** khỏi phần train.

**HỘP RESEARCH (KHÔNG ship, nhưng GIỮ — đang dùng train):**
| Gói | Bằng chứng |
|---|---|
| `src/model_dashboard` (3) | 0 importer ngoài |
| `src/leaderboard` (6) | 12 importer, 0 trên serve |
| `src/evaluation` (6) | chỉ `experiment.py:3500` (trong hàm train) |
| `src/tracking` (2) · `src/live_sim` (7) | không serve |
| `src/cache` (3) | train/perf + 1 route webapp (verify trước khi động) |
| `src/targets` (22) | sinh nhãn để TRAIN; serve `with_targets=False` |
| phần TRAIN của `experiment.py` | `train_fold` / `run_experiment` |

**HỘP BUILD (công cụ đóng gói, không runtime):** `src/export` (`export_bundle`).

**XOÁ (sai, KHÔNG phải research):** `src/backtest/portfolio_engine.py` (#2, §10.2) + di test giữ nó · `FeatureCacheManager` (§4.7, sau §1.6) · danh sách §5 (~2.6 MB).

**Nút gỡ cốt lõi:** hôm nay `experiment.py` import `src/targets` ở **dòng 36 (mức module)** ⇒ file lõi kéo theo research. Mức 2 = **XẺ `experiment.py`** thành signal-core (deps tối thiểu) + train-part (giữ `targets`/`evaluation`). Xẻ xong, hộp lõi rời hẳn `targets`/`evaluation`, mọi gói research rời bản production.

**TUYỆT ĐỐI KHÔNG XOÁ (bản ghi sống):** scripts `_*.py` gốc (provenance model live, §5) · cache/results 51-77 GB (tính lại được — sửa GC §1.6 trước) · **toàn bộ hộp research** (đang train).

### 10.2 `backtest/portfolio_engine.py` — bản Stage-2 cũ vẫn nằm trong wheel

486 dòng, hardcode **riêng** `HARD_STOP = 0.08` và `ZOMBIE_BARS = 14`, đọc `DEFAULT_PARAMS` từ `.defaults` — tức có **bộ tham số riêng**, không dùng `PortfolioConstants`. Importer **duy nhất**: `stock_ml/tests/execution/test_portfolio_engine.py:3` `from src.backtest.portfolio_engine import backtest_portfolio` — dùng prefix `src.*` đã chết (§4.5).

Xác nhận trong namelist `dist/stock_ml_core-0.4.2-py3-none-any.whl`: `stock_ml/src/backtest/portfolio_engine.py` nằm **cạnh** `stock_ml/portfolio/{api,constants,gates,panel,priority,rewrite,sim,variants,context}.py`.

Phân loại: **MISLEADING_LIVE**, không phải DEAD_SAFE (nó ship, và một test import nó) ⇒ **không** đưa vào danh sách xoá §5 mà chưa xử test.

⇒ `PORTFOLIO_LAYER_UNIFICATION.md` hợp nhất **logic**, nhưng **không loại bản cũ khỏi gói**. Một người đọc wheel hôm nay thấy hai portfolio engine và không có gì nói cái nào đúng. Việc cần làm: chuyển/xoá test rồi xoá file, **hoặc** loại khỏi `packages`/`package-data` của `pyproject.stock_ml_core.toml` (cùng lúc dọn wildcard ở R8).

### 10.3 `config.engine` là HỢP của hai bề mặt config, không tách biệt

Đo trên 5 bundle: 39-65 key là field `EngineConfig`; **11-12 key thì KHÔNG** (`costs`, `entry_ensemble`..`entry_ensemble4`, `entry_gate`, `exit_gate`, `exit_force_gate`, `exit_force_gate_lowbreadth`, `exit_force_gate_nonbull`, `nonbull_ma_win`, `nonbull_persist`) — chúng chỉ được `recombine_signals` đọc qua `cfg.engine.get(...)`.

Một dict phục vụ **hai consumer với luật kiểm khác nhau**: train **raise** khi gặp key lạ (`experiment.py:3474`), serving **im lặng bỏ** (`stock-serving/serving/trades.py:55-56`). Đây chính là lý do **R3 buộc phải phân hoạch BA chiều** — luật "raise nếu dropped khác rỗng" kiểu hai chiều sẽ từ chối cả 5 bundle.

Đích dài hạn: tách `config.engine` thành `config.engine` (chỉ field `EngineConfig`) + `config.recombine` (chỉ key tầng tín hiệu), có schema riêng. Ngắn hạn: `engine_config_from_dict` trả phân hoạch ba chiều + test khẳng định phân hoạch (hôm nay 3 danh sách tay khớp nhau **do may**).

---

## 11. QUYẾT ĐỊNH NỀN TẢNG (chủ dự án chốt 2026-07-30) — thay thế R0, R2, R6

> Mục này **ghi đè** phần nền của §2.3. Toàn bộ §2.3 trước đó được viết dưới một mục tiêu SAI (đóng băng
> hành vi cũ / tương thích ngược về SỐ). Chẩn đoán §1–§10 **không đổi** — chỉ mục tiêu và cơ chế đổi.

### 11.1 Mô hình đúng, bằng lời của chủ dự án

- **Xưởng** = `train_ai_ml`: chứa engine + cấu hình + research, **liên tục được cải tiến**. Bản mới ở xưởng
  là **tốt hơn** (nâng cấp tính năng, sửa lỗi nghiêm trọng, công thức feature chuẩn hơn).
- **Tầng phục vụ** = `stock-serving`.
- Lấy một model top ⇒ **xuất bản engine + cấu hình** để production sinh tín hiệu **như ở xưởng**.
- Xưởng đổi gì (vd công thức feature) thì khi xuất **ưu tiên chạy bản mới**. Mới hơn từ xưởng là **điều tốt**.
- **Mục đích kiểm khớp số**: đảm bảo **hai bản engine** (xưởng ↔ production) **cùng dữ liệu cho cùng kết quả**,
  để production sinh được tín hiệu như ở xưởng.

⇒ Đây là **kiểm chứng TƯƠNG ĐƯƠNG giữa hai bản sao**, KHÔNG phải kiểm chứng bất biến theo thời gian.
Nó khớp với việc **không có khái niệm paper trading** (§ `stock-serving/RESTRUCTURE_STRATEGY_LAYER.md` §5 D5):
chỉ có một đường backtest walk-forward 2020→nay, nên engine tốt hơn ⇒ **tính lại đường cong**, không có
"thành tích live" nào bị phá.

### 11.2 Hai loại tương thích — phân biệt này là cốt lõi

| | Yêu cầu? | Nghĩa |
|---|---|---|
| Tương thích **CẤU HÌNH** | **BẮT BUỘC** | Engine mới phải **nạp và chạy được** mọi thế hệ cấu hình còn trong catalogue, không lỗi |
| Tương thích **SỐ** | **KHÔNG** | Engine mới **được phép** cho số khác trên cùng cấu hình cũ — nếu là bản sửa/nâng chuẩn thì **phải** khác |

Tin tốt: tính chất bắt buộc kia **hôm nay gần như đã đúng** — wheel 0.4.2 đang phục vụ cả 3 bundle top-150
được xuất dưới 0.3.3 (và đợt kiểm 0.3.3↔0.4.2 cho score OOS bit-identical). Cái **thiếu** là một **test** giữ
cho nó đúng, và một **luật** xử lý khi nó sai. Đã có tiền lệ sai đo được: cả hai bundle dyn mang
`universe_policy`, field `ExperimentConfig` chỉ có từ 0.4.2 ⇒ `ExperimentConfig(**bundle.config)`
(`inference.py:47`) **TypeError trên 0.4.1** trong khi `format_version` báo tương thích. Đó chính là một lỗi
tương-thích-cấu-hình.

### 11.3 Chốt: MỘT ENGINE TẠI MỘT THỜI ĐIỂM

Chủ dự án chọn phương án (a). Hệ quả:

1. **Production chạy đúng MỘT bản wheel**, cho **mọi** chiến lược. Không có chuyện mỗi pack một engine riêng.
2. **Nâng engine ⇒ tính lại TẤT CẢ.** Xuất lại / tính lại toàn bộ đường cong của cả **10** chiến lược (§14.1) dưới engine
   mới. Mọi số công bố luôn **cùng một thước** ⇒ **10** chiến lược **so sánh được với nhau** (yêu cầu sản phẩm ở
   `RESTRUCTURE_STRATEGY_LAYER.md` §5 D8).
3. **Không cần** ghim engine theo từng pack, không cần chạy nhiều wheel song song, không cần dải tương thích
   `MINOR ≤`.

### 11.4 Cái gì BỎ khỏi §2.3

| Mục | Xử lý | Lý do |
|---|---|---|
| **R0** (luật bất biến: "wheel làm đổi output của bundle đóng băng thì không được ship") | **BỎ HẲN** | Ngược mục tiêu. Engine mới tốt hơn thì số **phải** được đổi |
| **R1** phần "3 giai đoạn → REFUSE khi lệch" | **BỎ phần REFUSE-khi-lệch** | Engine mới sẽ resolve khác một cách chính đáng (default mới, nút mới). Từ chối vì lệch là chặn đúng thứ mình muốn |
| **R1** phần "dựng engine TỪ `resolved.json` thay vì từ default" | **BỎ** | Không cần nữa: một engine tại một thời điểm ⇒ default hai bên **giống nhau do cấu tạo**. Đây là phức tạp không mua được gì |
| **R2** (dải runtime `MINOR ≤`) | **THAY** bằng **ghim chính xác** | Xem 11.5.2 |
| **R6** (golden đóng băng số của bundle đã deploy) | **THAY** bằng **so tương đương hai chiều** | Xem 11.5.3 |
| R3, R4, R5, R7, R8, R9 | **GIỮ NGUYÊN** | Chúng phục vụ tự-đủ + truy vết, không phục vụ đóng băng |

Hệ quả phụ: hai "giới hạn không giải được" mà panel kiến trúc nêu **biến mất**. (a) Đổi công thức DSL: không
còn là sự cố — xuất lại với công thức mới, công thức mới **tốt hơn**. (b) Wheel sửa lỗi code thật làm đổi số:
**đúng như mong muốn**.

### 11.5 Cái gì THÊM

**11.5.1 — BẢN XUẤT PHẢI TỰ ĐỦ (giữ `resolved.json`, đổi lý do).**
Không còn để đóng băng, mà để (i) cấu hình **đủ** cho production tái lập đúng cái xưởng đã chấm, và (ii) làm
**bằng chứng đối chiếu** khi hai bên lệch. Nội dung như §2.3 R1 đã liệt (227 giá trị engine sau default, 35
giá trị recombine, fingerprint catalog, **khai báo phụ thuộc data** — danh sách series/mã + version kỳ vọng,
KHÔNG nhúng bytes/checksum file (xem Q1 ngay dưới + §13.3) — wheel version thật, `PortfolioConstants` đã
resolve, hợp đồng seeding `prediction_history`).
Đây là **việc lớn nhất còn lại**, vì hôm nay bản xuất **không** mang theo engine: nó mang model + 39-65 trong
227 nút; 161-188 nút còn lại, cửa sổ z 252 bar, công thức DSL, và 3 file ngoài **nằm ở xưởng, không đi theo**.

✅ **Làm rõ chủ dự án chốt 2026-07-30 (Q1 data + Q2 vai trò `resolved.json`):**
- **Q1 — data KHÔNG nhúng, chỉ KHAI BÁO nhu cầu.** `resolved.json` mang **danh sách phụ thuộc data** (series/mã nào + version kỳ vọng: VNINDEX, universe, breadth…), **không** mang bytes data và **không** "checksum + bar cuối file ngoài" kiểu nhúng. Production resolve các phụ thuộc đó từ **nguồn chất lượng sống (sieutinhieu / kho phục vụ)**, **fail-loud nếu thiếu** để bên nguồn chuẩn bị — áp cho **mọi** loại data. Local DuckDB ở **cả hai repo là thứ cấp/lỗi-thời, chỉ train/test**, KHÔNG dùng để phục vụ. (Làm rõ §1.1/§1.2: production không dựa duckdb cũ; nối chính sách §9.7.)
- **Q2 — `resolved.json` là BẢN GHI + để SO, KHÔNG phải input runtime.** Hai việc: (i) tự-mô-tả/audit (đủ mọi nút, không đoán theo mặc định wheel), (ii) "cái cân" — tính lại dưới wheel mới rồi diff với bản đã lưu để **phát hiện đổi mặc định** (§11.5.3). **Không** load nó để ghi đè mặc định lúc chạy (đã loại ở §11.4). Hiệu suất: JSON nhỏ, ghi 1 lần lúc xuất, đọc chỉ khi audit/diff → ~0 chi phí runtime. Đồng thời là **nguồn danh tính**: `config_hash` tính trên nó (nối §3, loại metadata hiển thị §12.3).

**11.5.2 — GHIM CHÍNH XÁC + TỪ CHỐI KHI KHÁC (đây là cái "refuse" DUY NHẤT còn sống).**
Mỗi bộ số công bố ghi kèm **engine version đã sinh ra nó**. Production **từ chối phục vụ** khi wheel đang cài
≠ wheel đã sinh ra các đường cong đang lưu — không phải vì "số đổi là sai", mà vì **không được âm thầm đổi
thước**. Lúc đó chỉ có hai đường: rollback wheel, hoặc **chạy re-baseline** (11.5.4).
Thay thế toàn bộ trò `MINOR ≤` của R2. Vẫn giữ phần R2 thêm `stock_ml_core` vào `_TRACKED_LIBS` và gate
sklearn/numpy (`bundle.py:204` hiện chỉ so lightgbm).

**11.5.3 — SO TƯƠNG ĐƯƠNG HAI CHIỀU (thay R6).**
Cùng pack + **cùng ảnh chụp dữ liệu** → chạy ở xưởng và chạy ở production → so tín hiệu và lệnh. Lệch ⇒
**production sai**, sửa production; **không** hạ chuẩn engine, **không** re-pin fixture.
⚠️ Vẫn giữ nguyên yêu cầu từ §9.3.4: **CI/Linux/container là mặt phẳng recompute DUY NHẤT được thừa nhận**;
máy Windows chỉ tham khảo (lệch float VND 40 vs 37 đã đo).
⚠️ Vẫn giữ nguyên yêu cầu từ §9 (refute): test này phải **không có công tắc** — không env regen, không
auto-create fixture, không skip-on-missing. Ba golden hiện có đều tự vô hiệu
(`test_recombine_snapshot.py:116-120`, `test_baseline_snapshot.py:129-134` thiếu fixture ⇒ tự ghi rồi
`return`; `test_portfolio_golden.py` skip theo đường dẫn tuyệt đối).

**11.5.4 — RE-BASELINE THÀNH MỘT LỆNH, có log.**
Đây là thứ làm phương án (a) **trả được giá**: nâng engine ⇒ một lệnh tính lại toàn bộ **10** chiến lược (§14.1), ghi một
dòng audit (engine cũ → engine mới, ngày, ai chạy, đường cong nào đổi bao nhiêu). **Không** được là một cuộc
chạy tay. Không có nó thì "tính lại tất cả" trở thành lý do để trì hoãn nâng engine — đúng thứ mô hình này
muốn tránh.

**11.5.5 — TEST TƯƠNG THÍCH CẤU HÌNH (bảo đảm ngược THẬT, chạy mỗi lần build wheel).**
Với **mọi** thế hệ cấu hình còn trong catalogue: engine mới phải **nạp và chạy được**, không lỗi. Cụ thể:
`ExperimentConfig(**bundle.config)` + `engine_config_from_dict` + một chu kỳ sinh tín hiệu ngắn cho cả 5 pack
hiện có. Đây là chỗ bắt được đúng lớp lỗi `universe_policy`/0.4.1 ở 11.2.
Kèm theo: luật đọc key lạ phải **giống nhau hai phía** (hôm nay train raise, serving im lặng bỏ — §10.3), và
điều đó do R3 (`engine_config_from_dict`, phân hoạch BA chiều) thực hiện.

### 11.6 Thứ tự bị đổi: dữ liệu giống nhau LÊN ĐẦU

§1.1 (VNINDEX) và §1.2 (`market.duckdb`) trước đây xếp là "P0 vá chảy máu". Dưới mô hình này chúng thành
**điều kiện tiên quyết của toàn bộ chương trình**: không có dữ liệu giống nhau hai bên thì câu "khớp số"
**không có nghĩa**, nên không test tương đương nào chạy được.

Nhắc lại số đo: VNINDEX CSV có ở host, **không** được COPY cũng không mount vào container ⇒ knob
`signal_exit_hold_rs_scale=8.0` của 3/5 pack **sống ở xưởng, chết ở production**. `market_data/market.duckdb`
là **hai file khác nhau** trả lời cùng một đường dẫn tương đối: xưởng 70.791.168 B, serving 23.867.392 B.
**Đó chính là "hai bản engine không khớp", và nó đang xảy ra.**

**THỨ TỰ THỰC THI CHUẨN (hợp nhất §11.6 + §9.9, chốt 2026-07-30 — đây là bản DUY NHẤT; §9.1/§9.4/§9.9 đã bị bản này THAY).**

- **Phase 0a — Chặn mất mát NGAY HÔM NAY (không phụ thuộc gì, làm được trong 10 phút).** Đặt `PRUNE_BUNDLES=0` trong `docker-compose.yml`; backup `data/portfolio.db` + `market_data/market.duckdb`; tar 5 thư mục bundle; backup Postgres 98 GB. **Chỉ có vậy.**
  ⚠️ **Vì sao tách ra:** bản trước gộp việc này với "chỉ xoá khi registry khai `retired:true`" — nhưng registry đó (`serving/strategies.yaml` + `serving/strategies.py`) **CHƯA TỒN TẠI** (kiểm 2026-07-31), và `tiers.yaml` **không có trường `retired`**, chỉ phủ 6/15 `bundle_id` trong bảng `summary`. Nên hành động số 1 của cả chương trình **không có định nghĩa "xong"**. Tách ra thì Phase 0a chạy được ngay, còn phần luật để sau.
  ⚠️ Rủi ro đang chờ: hiện 5 bundle với `KEEP_BUNDLES=9` nên chưa ai chết; **Phase 4 xuất lại 5 cái nữa là 10 > 9** ⇒ prune xoá theo mtime, `rmtree`, **không khôi phục được**.
- **Phase 0b — Prune theo khai báo (sau khi có registry ở Phase 5).** Bỏ hẳn nhánh mtime/`--keep`/`KEEP_BUNDLES`; tập được-xoá = bundle không được bất kỳ chiến lược **không-retired** nào tham chiếu; abort ≠ 0 khi một bundle đang được tham chiếu mà không thấy trên đĩa (§11.8). Dry-run migration trên bản sao (§9.3.3).
- **Phase 1 — Dữ liệu giống nhau hai bên (điều kiện tiên quyết — không có nó thì "khớp số" vô nghĩa).** §1.1 VNINDEX/market-context + §1.2 `market.duckdb` breadth/xsec → input tường minh, có checksum, **giống nhau hai bên**; điều chỉnh corporate-action **tại nguồn sieutinhieu** (✅ nguồn đã phục vụ giá điều chỉnh — xác nhận 2026-07-31, §13.7) ⇒ phía repo chỉ còn **bỏ clamp in-memory của loader** + kiểm đường cong tính trước/sau bản vá (§9.7 sub-#1); **universe**: ~~dyn đã chuẩn (rule-slug + resolver causal)~~ — **SAI, sửa 2026-07-31**: resolver **causal thì đúng**, nhưng **thước thì sai** — `avg(volume*close)` không phải giá trị giao dịch và `count(*)` không phải số phiên khớp lệnh (**§13.9**, đo: 11-31% mã qua cổng mỗi năm là mã đã chết). dyn **cũng phải sửa**, và sửa xong universe đổi ⇒ re-baseline. **Tĩnh phải sửa slug sai + đăng ký set versioned** vào `universe_sets/versions` (finding §9.9 phase B); khai **bảng nền thị trường** `market_panel` + pin mốc `as_of` (§13.3/§13.4).
- **Phase 2 — Cái cân.** §1.5 CI trỏ path thật · §11.5.3 test **so tương đương hai chiều** (signal+trade, chạy TRONG container, Linux là mặt phẳng exact-match duy nhất, **không công tắc**) · §11.5.5 test **tương thích cấu hình** (engine mới nạp+chạy được mọi config sống).
- **Phase 3 — Bản xuất tự đủ + ghim engine.** §11.5.1 `resolved.json` (ghi + đối chiếu, KHÔNG refuse-khi-lệch) · R8 wheel tái-lập + tag + provenance (cần cho việc ghim) · §11.5.2 **ghim wheel chính xác + từ chối khi wheel đang cài ≠ wheel đã sinh số** · R3 deserializer chung (phân hoạch 3 chiều) · R4 toàn vẹn manifest · R5 drift guard · R9 facade.
  ⛔ **Gate sklearn/numpy KHÔNG nằm ở đây** — đã chuyển xuống Phase 4, xem cảnh báo ở đó.
- **Phase 4 — Re-baseline + deploy.** §11.5.4 **re-baseline một lệnh có log** · R7 attestation-per-wheel + `deploy.py` từ chối `FORCED`/wheel-mismatch · **D1** re-export sạch cả 5 (kèm sửa universe tĩnh + đăng ký set) + **D3** (3 top-150 qua pipeline chuẩn nếu tái tạo được, else `SELF_BASELINE`) · **RỒI MỚI** bật gate sklearn/numpy của R2.
  ⛔ **THỨ TỰ TRONG PHASE 4 LÀ BẮT BUỘC: re-export TRƯỚC, gate SAU.** Đo 2026-07-31: `bundle__dyn300` và `bundle__dyn900` bake `scikit-learn 1.3.2` / `numpy 1.26.3` trong `manifest.lib_versions`, còn môi trường chạy pin `scikit-learn==1.8.0` / `numpy 1.26.4`; `serving/tiers.yaml:24-55` cho thấy **cả 6 sổ khách** chạy trên đúng hai pack đó. Bật gate **trước** khi xuất lại ⇒ `load_bundle` **từ chối cả hai pack** ⇒ **6/6 sổ ngừng sinh tín hiệu**. (3 pack top-150 bake đúng 1.8.0/1.26.4 nên sống — chỉ tầng danh mục chết.)
  ⚠️ Lỗi này từng nằm ở `§9.9` phase C/E, được audit yêu cầu sửa, **và đã di cư nguyên vẹn sang bản thứ tự chuẩn này** trước khi được phát hiện lại ngày 2026-07-31. Nếu ai định đảo lại thứ tự, đọc dòng này trước.
- **Phase 5 — Danh tính + di sản.** §1.3 fingerprint đủ + §3 config_hash phủ hết + **D2** bảng phiên dịch `old↔new` · §1.4 alembic · §1.6 GC (sửa nguồn referenced TRƯỚC) · §4.1 migration 0030 (=B3) · §4.2 `nh_nav2` vào repo · §4.3 lifecycle · §4.4 fairness · §4.5/§4.6 thống nhất `stock_ml.src.*` · §4.7 xoá `FeatureCacheManager` · §10.2 loại portfolio-engine-#2 · overlay: §2.4 hợp đồng + test-song-sinh (§9.6.1b).
- **Phase 6 — Tuỳ chọn + dọn.** §10.1(a) tách module tín hiệu (**tuỳ chọn**, dọn cho sạch) · §5 xoá tập nhỏ · doc giao thức release-gate cross-repo.

Khác §11.6 bản đầu: thêm **Phase 0** (prune/backup) làm hàng rào mất-mát; **R8 kéo lên Phase 3** cạnh §11.5.2 (ghim wheel đòi wheel tái-lập/tag); nhét **universe/D1/D2/D3/overlay** vào đúng phase. Nguyên tắc §11.6 giữ nguyên: **dữ liệu-giống-nhau lên đầu**.

**Bất biến xuyên suốt mọi phase** (chuyển từ §9.9 khi xoá danh sách A→G, 2026-07-31): backup Postgres 98 GB + **dry-run migration trên bản sao** trước mọi phase động vào schema (§9.3.3); và **mọi phase động vào sổ live phải chạy lại attestation (§11.5.3) TRƯỚC khi thay** — không có ngoại lệ.

**Nguyên tắc xếp hạng** (chuyển từ §9.4 khi xoá, 2026-07-31): R1–R5, R7–R9 và §1.2/§1.3/§1.4 xếp **sau** vì chúng **tăng độ chặt**, không **tạo khả năng phát hiện**. Thứ gì cho ta *thấy được* drift thì làm trước thứ gì *siết chặt* nó — đó là lý do Phase 2 (cái cân) đứng trước Phase 3 (ghim).

➕ **Tinh chỉnh từ §13 (chốt 2026-07-31) — gắn vào phase tương ứng (§11.6 vẫn là bản thứ tự DUY NHẤT, đây là bổ sung không đảo thứ tự):**
- **Phase 1** nhận **§13.3** (khai `market_panel` = **luật lọc** + `as_of` + `symbols_sha256`, chốt `full_market` cho 5 bundle vì model train theo mẫu số toàn thị trường; **xoá `serving/breadth_store` đọc `breadth_universe.txt` 488-dòng** — panel không tự lớn được) + **§13.4** (phép so tương đương phải pin **một mốc dữ liệu `as_of`**, không phải "cùng đường dẫn tới server") + **§13.9** (sửa thước ADV/phiên của `universe_resolver.py`; đây là điều kiện tiên quyết THẬT của Phase 1 — universe sai thước thì "dữ liệu giống nhau hai bên" chỉ đảm bảo hai bên **cùng sai**).
- **Phase 5** nhận **§13.1** (overlay khai ở khối `portfolio:` **ngang** `engine:`, không lồng dưới `engine:`; giữ bia mộ `engine.portfolio.enabled`) + **§13.2** (`config_hash` rút từ `resolved.json`, dọn **8 cột chết + `leaderboard_seed_stats` + 1 cột trùng khái niệm**) — cùng cụm với §3/§4.1/§4.3 đã ở Phase 5.
- **§13.5 batch tăng dần** (điểm chốt trạng thái cuối ngày) là việc phía **serving**, làm **sau Phase 4** (khi đường xuất→phục vụ đã ổn); điều kiện tiên quyết: sửa "lệnh đang mở tính là đang mở" (`RESTRUCTURE_STRATEGY_LAYER.md` §8.1) + cổng full-span-vs-incremental. §13.6 (lưu trữ theo ngày) **đã chuẩn — giữ nguyên**.

➕ **§11.9 (fold-model) — bổ sung vào danh sách phase, trước đây bị bỏ sót ở đây (phát hiện 2026-07-31):**
§11.9 tự khai ảnh hưởng Phase 1 + Phase 3 nhưng **không có mặt trong thân các phase trên**, nên người đọc danh sách phase sẽ không thấy nó. Gắn chính thức:
- **Phase 1** nhận: đóng gói bundle phải gồm **tất cả fold-model + hợp đồng seeding** (không phải một model + `prediction_history` chép tay). Bundle nặng thêm ~6× model (~9 MB).
- **Phase 3** nhận: **engine phải chọn fold-model theo ngày** khi phục vụ (việc code thật ở tầng `generate_signals_from_bundle`/`predict_slot_signals`); `resolved.json` + so-tương-đương phải khẳng định tín hiệu **tái sinh từ fold-model**; `prediction_history` — nếu còn — chỉ là cache có checksum, **bị loại khỏi mọi phép chứng minh khớp**; attestation §11.5.3 chạy trên **phần tươi, không seed**.

⚠️ **Hệ quả sang doc B, phải xử lý cùng lúc:** `RESTRUCTURE_STRATEGY_LAYER.md` **không nhắc §11.9 một chữ nào** (grep `fold-model`/`prediction_history` = 0 hit) — trong khi đây là sửa **thẳng đường sinh tín hiệu của serving**. Và con số gác cổng của doc B bước S1 (`nav ×825.51 / CAGR 180.4% / DD −21.4% / 1937 lệnh`) được tính **dưới chế độ seed CŨ** mà §11.9 vừa bỏ ⇒ **con số đó sẽ đổi**, không được dùng làm mốc pass/fail sau khi §11.9 land. Doc B phải nhận mục này khi sửa bảng S.

### 11.7 Rủi ro của phương án (a), nói thẳng

- **Mỗi lần nâng engine là tính lại catalogue — nhưng chi phí NẶNG là theo _model_, không theo _chiến lược_.**
  ĐÃ ĐO (M1, 2026-07-31 — cold full-history, 6 fold, single-seed, LGBM `force_col_wise`, Postgres persist,
  máy dev Windows): dyn61 **55s** · dyn300 **208s** · dyn900 **486s** · tĩnh-150 nội suy **~115s**. Thời gian
  bám theo số signal persist (dyn61 96k → dyn900 1,22M), **không** tuyến tính theo số mã. Fold-cache chỉ giảm
  **~20%** (warm dyn61 44s vs cold 55s) vì backtest + ghi signal mới là phần nặng, không phải fit LGBM.
  ⇒ **Ngân sách re-baseline = 6 model cold-train + 10 overlay-replay**, KHÔNG phải "10 × full-history":
  10 chiến lược ánh xạ chỉ **6 model riêng** (5 biến thể dyn300 dùng chung 1 model, §14.1), tầng ML chạy
  **một lần/model** rồi replay overlay (rẻ). Tổng 6 model single-seed ≈ 55+208+486+3×115 = **~19 phút**;
  nếu production đòi 3-seed cho các sổ đã deploy thì ×3 phần đó ≈ **~50 phút**. Overlay-replay chưa đo riêng
  nhưng `PortfolioContext` không memoise (§ chi phí overlay) ⇒ đáng đo trước khi hứa SLA; ước off-peak **<1 giờ**
  cho cả catalogue là an toàn.
- **Nâng engine làm cả **10** đường cong công bố đổi cùng lúc.** Với mô hình "chỉ có backtest, không có paper
  trading" thì hợp lệ, nhưng phải là **sự kiện có chủ đích, có log, có thông báo** — không bao giờ xảy ra
  âm thầm khi restart container. Đó chính là lý do tồn tại của 11.5.2.
- **`prune_bundles` sẽ ăn pack cũ.** Container đang `PRUNE_BUNDLES=1` + `KEEP_BUNDLES=9`, xoá theo mtime,
  chỉ bảo vệ 1 bundle active; mất pack là **không khôi phục được**. Phải tắt/sửa **trước** khi làm gì khác.
- **Nửa "xưởng" của việc thêm chiến lược mới vẫn là script viết tay** (5 tiền lệ
  `stock_ml/scripts/ops/deploy_*.py`). Chương trình này **chỉ** bảo vệ đường xuất→phục vụ; phải nói đúng giá
  đó, hoặc thêm một bước gom 5 script thành một CLI đọc YAML.

### 11.8 Chủ dự án xác nhận thêm 2026-07-30: tính lại là chấp nhận được, và model cũ được NGHỈ

> "Việc chạy lại các chiến lược là bình thường và chấp nhận, vì số lần cập nhật engine khá ít, và mỗi lần
> thay đổi model có thể sẽ tắt bớt model cũ để thay thế cho model mới cao cấp hơn."

Hai hệ quả thiết kế:

**(1) Catalogue KHÔNG phình — nên test tương thích cấu hình (11.5.5) luôn nhỏ.** Nó chỉ phải phủ các thế hệ
cấu hình **còn sống**, không phải mọi thế hệ từng tồn tại. Điều này cũng chặn nỗi lo batch đêm phình theo thời
gian: số chiến lược sống có trần, vì thêm cái mới thì cho cái cũ nghỉ.
Đồng thời nó **hạ giá** của 11.5.4: nâng engine ít lần + tính lại chậm cũng được ⇒ lệnh re-baseline không cần
tối ưu, chỉ cần **đúng và có log**.

**(2) "Cho nghỉ" phải là một KHAI BÁO, không bao giờ là một hiệu ứng phụ.** Đây là chỗ đang có bug:
`serving/prune_bundles.py` xoá bundle **non-active theo mtime** và chỉ bảo vệ đúng `Path(active).name`;
`docker-compose.yml` đang `PRUNE_BUNDLES=1` + `KEEP_BUNDLES=9`. Nên hôm nay hai pack nuôi **toàn bộ** sổ khách
sống sót **chỉ vì** con số 9, và `rmtree` là **không khôi phục được**.

Luật đích:
- Pack chỉ được xoá khi **registry khai `retired: true`** cho mọi card trỏ vào nó. Không bao giờ theo mtime,
  không bao giờ theo số lượng.
- `prune_bundles` đọc registry, **abort ≠ 0** nếu một pack đang được card không-retired tham chiếu mà không
  thấy trên đĩa.
- Card `retired: true` **vẫn ở trong registry** (không biến mất), chỉ ẩn khỏi menu khách — để một người từng
  theo chiến lược đó không gặp trang chết, và để audit trả lời được "tháng 7 mình công bố những gì".
- Vì mô hình là "một đường backtest, tính lại khi nâng engine" (§11.1), pack của một chiến lược **đã nghỉ**
  thật sự **xoá được** — không có thành tích live nào cần nó để giải thích. Nhưng phải là **hành động khai
  báo**, có log, không phải hệ quả của một biến môi trường.

### 11.9 Bundle phải TỰ TÁI SINH tín hiệu — bỏ `prediction_history` seed, ship FOLD-MODEL (Option 1, chủ dự án chốt 2026-07-30)

**Phát hiện (đo được, [inference.py:114-141](stock_ml/src/serving/inference.py#L114-L141)).** `generate_signals_from_bundle` chấm điểm tươi cho mọi bar rồi **GHI ĐÈ điểm quá khứ bằng điểm cũ lưu trong bundle** (`prediction_history.parquet`): `raw.loc[seed, c] = raw.loc[seed, f"{c}_hist"]` (dòng 138); chỉ bar mới giữ điểm model hiện tại (comment dòng 115-119). Hệ quả:
- **Che lỗi:** điểm lịch sử do engine hiện tại tính bị vứt, nên engine mới chấm sai lịch sử **không lộ**.
- **Không chứng minh được tái tạo:** tín hiệu lịch sử = output engine cũ chép sang ⇒ "khớp backtest" là **vòng tròn** (bơm chính output backtest vào z-window).

**Vì sao code làm vậy — WALK-FORWARD.** Bundle chỉ ship model **fold cuối**; trong backtest mỗi năm chấm bởi **model-fold năm đó** (causal). Lấy model cuối chấm lại 2020 thì (a) không khớp backtest, (b) nhìn tương lai. Nên seed điểm per-fold. ⇒ **Với một model duy nhất được ship, "tái tạo lịch sử + khớp gốc" là bất khả về bản chất.**

**Trấn an:** bar **live** vẫn được model hiện tại chấm tươi (dòng 119); seed chỉ ảnh hưởng **baseline z-window của lịch sử** và **cuộn hết sau ~252 phiên live**. Nên nó che khả năng tái-tạo-LỊCH-SỬ, **không** che chất lượng tín hiệu LIVE. Với §11.5.3 (production≡xưởng) seed **không phá** (hai bên seed giống nhau); nó chỉ phá mục tiêu "model tự sinh lại + khớp gốc".

**✅ CHỐT — Option 1: bundle mang TẤT CẢ fold-model, production chạy lại walk-forward.** Bar năm Y chấm bằng **model-fold-Y** ⇒ lịch sử **được tái tạo THẬT**, khớp gốc **do cấu tạo**, `prediction_history` thành **thừa** (nếu giữ thì chỉ là cache dẫn-xuất-được + checksum, **không bao giờ** dùng để chứng minh khớp). Các fold-model **đã tồn tại**: template `_dyn300_2020.._2025`, `_dyn900_2020.._2025`, và `results/tmpl_*/folds/`.

**Chi phí & việc phải làm:**
- Bundle mang ~6× model (~9 MB, chấp nhận) thay vì chỉ fold cuối.
- **Engine phải chọn fold-model theo ngày** khi phục vụ (việc code thật, ở tầng `generate_signals_from_bundle`/`predict_slot_signals`).
- **Attestation (§11.5.3) chạy trên phần TƯƠI, không seed** — để chứng minh model+engine sinh đúng, không vòng tròn.

**Ảnh hưởng thứ tự (§11.6):** (Phase 1) đóng gói bundle nay phải gồm **fold-model + hợp đồng seeding**, không phải một model + `prediction_history` chép tay; (Phase 3) `resolved.json` + so-tương-đương phải khẳng định tín hiệu **tái sinh từ fold-model**, và `prediction_history` — nếu còn — là cache có checksum, bị loại khỏi mọi phép "chứng minh khớp".

---

## 12. Mô hình đối tượng chuẩn (chủ dự án chốt 2026-07-30) — 4 đối tượng, KHÔNG có tier

> Thay phần mô tả tản mạn ở §11.1 và **mọi chỗ nói "tier"**. Đây là danh từ chuẩn cho toàn hệ. Chốt qua Q1/Q2/Q3 với chủ dự án: model để **kho chung**, overlay **gộp vào strategy**, **bỏ hẳn tier**.

### 12.1 Bốn đối tượng

| Đối tượng | Là gì | Định danh | Tái dùng |
|---|---|---|---|
| **Engine** | Code tính toán (wheel `stock_ml_core`): {model + config} → tín hiệu → lệnh → sổ | version + tag wheel | **một bản** cho cả production (§11.3) |
| **Model** | Bộ trọng số ML = **MỘT BỘ fold** (mỗi năm một fold — §11.9), ăn features → điểm | hash nội dung bộ fold | **1 model → N chiến lược** |
| **Strategy (chiến lược)** | Cấu hình giao dịch **hoàn chỉnh, tái tạo được**: model-ref + universe + engine-knobs + **overlay** + phí | `config_hash` (chỉ trên trường **ảnh-hưởng-số**) | — |
| **Bundle** | Gói mang ra production = model (bộ-fold, hoặc con-trỏ tới kho) + config chiến lược + manifest | `bundle_sha` | — |

**TIER: ĐÃ BỎ.** Trước đây tier = model + overlay + nhãn. Nay **overlay thuộc strategy** (Q2), **nhãn + ẩn/hiện là metadata của strategy** (12.3). Mỗi chiến lược = một mặt hàng; **không có lớp trung gian** (chủ dự án xác nhận: không bao giờ cần một chiến lược mang nhiều tên song song).

### 12.2 Quan hệ + nơi cấu hình

- **1 Model → N Strategy:** cùng bộ fold, khác overlay/universe/knob → chiến lược khác. Ví dụ thật: `dyn300_k6_liqcol` và `dyn300_k10_floor5` = **hai chiến lược, một model** dyn300.
- **Model để KHO CHUNG (Q1):** chiến lược trỏ model bằng **hash**, không nhúng-nhân-bản; đóng bundle mới gom model vào.
- **Overlay vào config chiến lược (Q2):** bỏ vai trò cấu-hình của `tiers.yaml`; một chiến lược = **một file** đủ {model-ref + universe + engine-knobs + overlay + phí}.
- **Engine:** một bản, version hoá (§11.3/§11.5.2).

### 12.3 Metadata KHÔNG-ảnh-hưởng-số trên strategy (thay "tier")

- `display_name` — tên hiện cho khách.
- `lifecycle` — **sống / nghỉ** (retired, §11.8); menu khách = các chiến lược đang **sống**.
- **Cả hai KHÔNG vào `config_hash`** ⇒ đổi tên hoặc cho nghỉ **không đổi số, không đổi danh tính** chiến lược. Đây là điều làm việc gộp-tier an toàn.

### 12.4 Việc kéo theo (nối vào §11.6)

- **Di tham số overlay** đang ở `tiers.yaml` (K, floor, liqcol, invvol, sizing…) vào config từng chiến lược, **đảm bảo không đổi số** (đo trước/sau) — thuộc Phase 5 (overlay §2.4).
- **Dựng kho model** + cơ chế bundle trỏ model bằng hash — thuộc Phase 1/3 (đóng gói bundle).
- `config_hash` tính **chỉ** trên trường ảnh-hưởng-số (loại `display_name`/`lifecycle`) — nối §3 + D2 ở Phase 5.

### 12.5 Head/ensemble: ở CONFIG, ship đúng bộ đã kiểm định (luật C — chủ dự án chốt 2026-07-31)

**Head khai ở CONFIG, không phải engine.** Số "giám khảo" (head) = số key `entry_ensemble*` (+ `exit_ensemble`) trong `config.engine`; engine chỉ là code chạy chung. Điểm cuối = **tổng hợp** các head ⇒ **số head đổi → điểm đổi → lệnh đổi**. Ví dụ dyn900: `entry_ensemble`, `entry_ensemble2..4` → entry2..entry5 + head chính = **5 head** (khớp 5 model bundle).

**Luật C:** ship **đúng bộ head đã kiểm định**; export mà thiếu một head đã validate → **báo lỗi, dừng** (không âm thầm). Head thử nghiệm phải bị loại **trước** khi kiểm định, không phải lúc xuất — vì ship khác đội đã validate thì con số đã chứng minh **vô nghĩa**.

**Root cause bỏ sót (đo `export_bundle.py`):** export **viết tay danh sách head, cứng tới entry5** ở HAI chỗ — lời gọi `train_fold` (`:236-243`, liệt kê `entry2..entry5_target_col`, **không có entry6**) và dict `_entry_head_feat = {2,3,4,5}` (`:296-297`) — trong khi `build_feature_frame` (`:166`) trả `entry6_feat_cols` và serving `inference.py` khai `{2,3,4,5,6}`. Comment ghi "generic over head count" nhưng code **hardcode**. ⇒ Anti-pattern **"danh sách tay song song bị lệch"**, **cùng họ** §1.3 / §2.1 / §3: mỗi nơi tự liệt kê, không nguồn-chung, không assert khớp; ai mở rộng lên 6 ở feature+serving nhưng quên export.

**Hiện trạng:** 5 bundle deploy đều khai ≤5 head → **hôm nay không mất head** (lỗi **latent**); nhưng champion 6-head kế tiếp sẽ **rụng âm thầm**, drift-guard không bắt.

**Fix (khớp R5):** export **lặp generic qua ensemble keys của config** (bỏ hardcode 2-5) + **ASSERT `set(models)` ≡ head config khai** + ghi `*_feat_cols`/`*_target` cho **mọi** head. Nguồn-sự-thật-duy-nhất = config.

---

## 13. Quyết định chốt 2026-07-31 — nơi khai overlay, danh tính, bảng nền, tính tăng dần

> Nối tiếp §11 (mục tiêu) và §12 (mô hình đối tượng). Mục này **không** thay §12 — nó trả lời những câu
> §12 để mở: overlay khai **ở đâu trong file**, `config_hash` tính **từ cái gì**, và ba việc mới.

### 13.1 Overlay khai ở khối `portfolio:` CẤP TRÊN CÙNG, không nhét dưới `engine:`

§12.2 đã chốt overlay thuộc config chiến lược. Câu còn lại là đặt ở đâu. **Chốt: một khối `portfolio:`
ngang hàng với `engine:`, không lồng vào trong.**

**Nguyên nhân gốc vì sao nó bị tách ra — đã tìm được, và không phải do quên.** `experiment.py:3477-3482`:

```python
if portfolio_cfg.get("enabled"):
    raise ValueError("engine.portfolio.enabled is no longer supported — the legacy
        alpha->portfolio->execution tier (src/portfolio + src/execution) was removed.
        Only refuted xsec top-k templates (629-631) ever enabled it.")
```

⇒ Chỗ khai trong config **từng tồn tại**, bị một tầng danh mục **thế hệ trước** dùng, tầng đó **bị bác bỏ
và gỡ đi**, và chỗ khai bị biến thành **bia mộ báo lỗi**. Khi tầng overlay MỚI (`stock_ml/portfolio`) ra
đời, cửa vào config **đã bị đóng đinh** ⇒ nó đi ra ngoài config, sống bằng đối tượng Python + script rời.
**Cấu trúc phân mảnh hôm nay là di chứng của một thất bại cũ, không phải của sự cẩu thả.**

**Vì sao KHÔNG lồng dưới `engine:` — ba lý do đo được:**
1. `config.engine` **đã là hợp của hai bề mặt**: 39-65 khoá là field `EngineConfig`, 11-12 khoá chỉ
   `recombine_signals` đọc (§10.3). Hai người đọc, hai luật kiểm (train raise / serving im lặng bỏ).
   Thêm 29 khoá `PortfolioConstants` = **ba người đọc trong một dict**.
2. Đường nạp hiện tại là **pop 37 khoá bằng danh sách viết tay** rồi `EngineConfig(**engine_cfg)`. Lồng
   overlay vào đó là kéo dài đúng danh sách viết tay đang là nguồn lỗi (**cùng anti-pattern §12.5**).
3. Overlay ăn **đầu ra** của engine (lệnh nền), nó là **giai đoạn sau**, không phải tham số của engine. Và
   `stock_ml/portfolio` là tầng **sạch nhất** đang có (§10) — trộn vào `engine.py` 2824 dòng là kéo cái
   sạch vào cái chưa sạch.

**Phân mảnh thật nằm ở đường CHẠY, không ở tên khoá.** Ba việc phải làm, và chỉ ba:

| | Hôm nay | Đích |
|---|---|---|
| Cấu hình | nửa dưới không có chỗ khai | **một file, hai mục ngang hàng** (`engine:` + `portfolio:`) |
| Đường chạy | overlay chạy bằng ~20 script rời | **một lệnh chạy cả hai tầng, trả cả hai kết quả, lên board qua đường chính thức** |
| Danh tính | `config_hash` không phủ nửa dưới | **một ID phủ cả hai** (13.2) |

Sau ba việc đó, `stock_ml/portfolio` **vẫn là package riêng** — đó là **module hoá tốt**, không phải phân
mảnh. Phân mảnh = không có gì nối lại; module hoá = nối bằng hợp đồng rõ ràng.

Giữ nguyên bia mộ cho `engine.portfolio.enabled` để config thế hệ cũ vẫn báo lỗi to; **không tái dùng tên
khoá đã chết** cho tầng mới.

**Kiểm chứng đã đo:** `strategy_templates` có **25 cột, không cột nào** về portfolio/overlay; ở xưởng
**không có file yaml nào** tương đương `tiers.yaml`; `PortfolioConstants` **chỉ được dựng trong code**
(default trong `api.py`, `variants.py`, tests) — **chưa bao giờ đọc từ file cấu hình**.

### 13.2 `config_hash` = hash của CONFIG ĐÃ RESOLVE, không phải của bản tóm tắt

§12.1 chốt "config_hash chỉ trên trường ảnh-hưởng-số". Đây là cách tính.

**Vì sao ID đang trùng — cơ chế:** ID băm từ `summary['config']`, mà bản tóm tắt đó được viết ra để **MÔ TẢ**
một lần chạy (hiển thị/báo cáo), rồi bị **đem dùng làm DANH TÍNH**. Mô tả chỉ cần nêu vài điểm chính; danh
tính phải phủ **mọi thứ làm đổi kết quả**. Nó mang ~10 trường + **6 nút engine** trên tổng **227+**.
⇒ Đo trên DB: một `config_hash` bị **360 tên run khác nhau** dùng chung; hai hash khác là 270 và 146.

**Chốt: ID rút từ `resolved.json`** (§11.5.1) — vật thể đã chứa toàn bộ giá trị đã resolve. Một vật thể,
một hash, một danh tính, **không còn danh sách trường viết tay để lệch**. Loại `display_name`/`lifecycle`
theo §12.3.

**Di trú (trả lời D2 §9.2):** run cũ **không có** `resolved.json` ⇒ giữ ID cũ, đóng băng, **không tính lại**.
Run mới có ⇒ nhận ID mới. Đây là **hệ quả tự nhiên**, không phải mẹo vá — không có bước di trú rủi ro trên
368 triệu dòng. Cần nối lịch sử thì thêm bảng ánh xạ `old_run_id → new_run_id`.
⚠️ **Ràng buộc:** hai bundle dyn đang phục vụ có `run_id` **cứng** trong `_seed_serving_tiers.py:56-59`.
Không đổi ID của chúng khi script đó còn trỏ vào.

**Hệ quả 13.1 + 13.2 giải cho nhau:** hôm nay 5 tier dyn300 dùng **chung một bundle**, chỉ khác tham số
danh mục — mà tham số đó **không nằm trong config** ⇒ chúng băm ra **cùng một ID**. Khi overlay vào config,
chúng thành **5 chiến lược riêng, 5 ID riêng**, xếp hạng cạnh nhau trên board. Đúng mô hình §12.2.

**Dọn kèm — đã đo trực tiếp trên 3.612 hàng `leaderboard_runs`:**

| Cột / bảng | Đo được | Xử lý |
|---|---|---|
| `cache_key_features` | **0/3612** có giá trị | xoá |
| `cache_key_predictions` | **0/3612** | xoá |
| 6 cột `same_*_as_baseline` | **0/3612** mỗi cột | xoá cùng module fairness (§4.4) |
| `fairness_group_key` | 3612/3612 có giá trị nhưng chỉ **3 giá trị phân biệt** | vô dụng, xoá cùng nhóm |
| `leaderboard_seed_stats` | 54 hàng | bảng phụ sinh ra **chỉ vì** ID bỏ qua seed → ID mới phủ seed ⇒ **thừa**, xoá |
| `state` vs `superseded` | 5 pinned / 1052 superseded | hai cột một khái niệm (§4.3), giữ một |

**Tổng: 8 cột chết + 1 bảng phụ + 1 cột trùng khái niệm.**

Và **hai projection viết tay** cùng mô tả "cái gì làm nên một lần chạy" — `summary['config']` (cho ID) và
`_fp_src` (cho fold cache, §1.3) — đều thiếu, và **thiếu khác nhau**. Cả hai phải rút từ **cùng một**
config đã resolve. Không ai duy trì danh sách riêng nữa.

### 13.3 Bảng nền thị trường: khai DANH SÁCH MÃ trong config, không mang DB đi

**Nguyên tắc chủ dự án chốt:** sieutinhieu là **nguồn chuẩn duy nhất**; mọi `market.duckdb` ở hai repo đều
là **bản sao thứ cấp** và đều lỗi thời được. ⇒ **Không mang DB qua biên.** Chỉ mang **danh sách mã yêu cầu**
trong config. Repo nào thiếu mã nào thì **báo lỗi kèm tên mã thiếu** — và lỗi đó chính là tín hiệu để đi lấy
dữ liệu mới. **Lỗi là cơ chế, không phải sự cố.** Về sau production đấu thẳng sieutinhieu, DB local chỉ còn
phục vụ train/test.

**Đây là tập mã THỨ BA, đang bị lẫn với hai tập kia:**

| | Là gì | Trạng thái |
|---|---|---|
| Universe giao dịch | mã chiến lược được mua bán | manifest `universe`/`universe_by_year` ✅ |
| Feature scope | mã được tính feature | manifest `feature_scope` ✅ |
| **Bảng nền thị trường** | **mẫu số** tính breadth + xếp hạng chéo | ❌ **không khai ở đâu** |

**Khối config:**
```yaml
market_panel:
  policy: full_market        # hoặc own_universe
  source: sieutinhieu
  as_of: <ngày>
  # LUẬT LỌC, không phải danh sách phẳng — danh sách phẳng đẻ ra breadth_universe.txt
  filter: { asset_type: stock, min_sessions_traded: 100, sessions: 250 }
  n_symbols: <resolve ra bao nhiêu tại as_of>
  symbols_sha256: <hash danh sách đã sắp xếp>
```

⚠️ **`full_market` = 901 là SAI, sửa 2026-07-31.** Con số 901 là số mã ta *tình cờ có trong DB*, không
phải mẫu số thị trường. Đo từ nguồn (`GET /symbols/`, `GET /symbols/universe`):

| Định nghĩa | Số mã | Dùng khi |
|---|---|---|
| Cổ phiếu **từng tồn tại** (gồm huỷ niêm yết) | **1.981** | panel lịch sử point-in-time, **không survivorship** |
| Đang niêm yết hôm nay | 1.372 | ❌ dùng cho lịch sử = tái tạo bẫy +29pp |
| **Thực sự khớp lệnh** ≥100 phiên năm trước | ~1.055 (2025) | "thị trường giao dịch được" |

Panel 901 của xưởng phủ ~85% phần đang giao dịch; panel 488 của serving phủ 64%. **Chốt ngưỡng nào là
quyết định re-baseline** (conviction là rank trên mẫu số này), không phải tham số vặn sau.
`n_symbols` vì vậy là **kết quả resolve tại `as_of`**, không phải hằng số khai tay.

**Chốt giá trị cho 5 bundle hiện có: `full_market`.** Không phải vì nó hay hơn, mà vì **model đã được HUẤN
LUYỆN với mẫu số toàn thị trường** — đổi sang `own_universe` lúc phục vụ là feed model một feature có phân
bố khác cái nó từng học. Đo được mức lệch hạng khi đổi mẫu số: wavestruct **0,124** · dyn300 **0,099** ·
dyn900 **0,004**.
⚠️ `own_universe` phải định nghĩa là **"universe đang có hiệu lực tại ngày của bar đó"** (theo năm). Ai cài
bằng danh sách phẳng sẽ **tái tạo bẫy survivorship +29pp**, và tệ hơn là nó cho kết quả **đẹp hơn** nên
không ai nghi.
⚠️ Breadth thì **không có lựa chọn** — nó là chỉ báo chế độ thị trường; tính trên chính rổ của mình là tự
soi gương, mất hẳn tính độc lập vốn là giá trị của nó. **Một khai báo dùng chung cho cả hai**, chỉ tách khi
research chứng minh cần.

**Kiểm ở HAI thời điểm, hai câu hỏi khác nhau:**
- **Lúc xuất:** khai báo có khớp cái đã train không? ⚠️ Muốn *kiểm* chứ không phải *khẳng định* thì
  **training phải GHI LẠI mẫu số nó dùng** (bao nhiêu mã, hash, theo năm nếu có). Hôm nay không ghi ⇒ cổng
  lúc xuất chỉ kiểm được **chính tả**.
- **Lúc nạp bundle:** nơi phục vụ có làm nổi cái đã khai không? Thiếu ⇒ báo rõ mã thiếu.

> 📍 **Số đo thiệt hại + nguyên nhân gốc: `RESTRUCTURE_STRATEGY_LAYER.md` §13.1 SỞ HỮU.**
> Gồm: panel serving 488 vs xưởng 901 · dyn900 thiếu **321/775 mã (41,4%)** · **182/383 lệnh (47,5%)** của
> `tier_dyn900_k16_live` mang `conv=0.5`/`prio=-9.9` trong khi hàng seed cùng tier có conv thật ·
> `breadth_universe.txt` đúng 488 dòng là nguyên nhân panel không tự lớn được · quyết định **xoá file đó**.
> Mọi hành động đều ở repo serving. Doc này chỉ giữ **hợp đồng** (khối yaml + ngữ nghĩa + cổng kiểm) ở trên.

### 13.3.1 Hợp đồng resolve data: local là CACHE, sieutinhieu là NGUỒN, thiếu → FETCH (chốt 2026-07-31)

Làm rõ §13.3 + §9.7 + §11.5.1 Q1 (chủ dự án nhấn mạnh): **KHÔNG bao giờ** coi một `market.duckdb`/CSV local là
"nguồn versioned" — mọi bản local (cả hai repo) là **cache thứ cấp, được phép cũ**. **sieutinhieu = nguồn chuẩn
duy nhất, mới nhất, cập nhật thường xuyên** (đã có `serving/ingest.py` → `https://sieutinhieu.vn/api/v1`:
`fetch_history`, `fetch_universe`).

**Endpoint đã có, đã kiểm chạy 2026-07-31** (đường đọc công khai, không cần token; OpenAPI
`https://sieutinhieu.vn/openapi.json` là nguồn sự thật về danh sách endpoint — **đừng chép vào doc**):

| Cần gì | Endpoint | Ghi chú đã đo |
|---|---|---|
| Giá (tín hiệu, mark NAV) | `/ohlcv/` | **đã back-adjust về hiện tại**; nguồn re-sync TOÀN BỘ lịch sử mỗi sự kiện quyền |
| Universe point-in-time | `/symbols/universe?as_of=a,b,c&sessions=&top_n=&min_adtv=` | **nhiều mốc/1 lần gọi** — 6 mốc/4 giây; trả `notes[]` liệt kê mọi thứ bị loại, không cắt im lặng |
| ADV/ADTV một mã tại một mốc | `/symbols/{sym}/adv?as_of=&basis=matched` | từ `market_flow_1d` — **point-in-time, append-only, KHÔNG BAO GIỜ re-sync** |
| Danh sách mã / trạng thái niêm yết | `/symbols/?listing_status=&include_delisted=` | 1.981 cổ phiếu gồm huỷ niêm yết |
| Sàn **tại thời điểm** | tham số `exchange` (tra `symbol_listing_history`) | 369 mã từng chuyển sàn (ACB HNX→HOSE 2020) |

Điểm quyết định: nguồn **point-in-time thật**, không phải danh sách hiện tại — hỏi ADTV của
ROS/FLC/ITA tại 2020 vẫn ra số thật (96.8 / 65.4 / 118.6 tỷ) dù các mã đó sau này huỷ niêm yết, và
top-1 thanh khoản 2020 chính là ROS. Đây là thứ chống survivorship **tại gốc** thay vì vá sau.

**Hợp đồng chuẩn — mỗi lần một bên (xưởng hoặc production) cần data:**
1. **Khai nhu cầu** trong config: danh sách mã/series (VNINDEX, universe, breadth…) + `as_of` (§13.4). KHÔNG nhúng bytes.
2. **Kiểm cache local:** đủ mã + coverage ≥ `as_of`? → dùng luôn.
3. **Thiếu mã / cũ hơn `as_of`:** → **fetch bổ sung từ sieutinhieu** → upsert vào cache local → dùng.
4. **Nguồn không tới + cache không đủ:** → **fail-loud** (guard `MARKET_CONTEXT_REQUIRED`, §1.1). Tuyệt đối không chạy với data thiếu.

"**Cùng data hai bên**" = cả hai **khóa cùng `as_of`** ⇒ cùng ảnh chụp dù mỗi bên tự fetch. Cache cũ vô hại cho
test toàn-vẹn-engine (độ mới là việc của batch hằng ngày). ⇒ **Điều chỉnh §1.1/§1.2/§13.3:** reader KHÔNG "trỏ
vào local duckdb versioned" mà đi qua **resolver có hợp đồng fetch** (local chỉ là cache của resolver). Hai chế
độ: research/train được dùng cache cũ (§9.7); official/serving/attestation phải resolve theo `as_of` (fetch nếu thiếu).

### 13.4 So khớp hai bản phải cố định MỘT MỐC DỮ LIỆU

Nối cả hai bên vào cùng sieutinhieu **chưa đủ để khớp**: server vẫn chạy tiếp, xưởng đọc 9h và production đọc
15h là ra hai kết quả khác nhau mà **cả hai đều đúng**. ⇒ Phép so tương đương (§11.5.3) phải pin **"dữ liệu
tính đến ngày X"**, không phải "cùng đường dẫn tới server". Thiếu điều này thì bài test lúc xanh lúc đỏ mà
không ai hiểu vì sao.

### 13.5 Batch đêm: TÍNH TĂNG DẦN — ràng buộc phía WHEEL

> 📍 **Thiết kế + việc code: `RESTRUCTURE_STRATEGY_LAYER.md` §13.3 SỞ HỮU** (bảng `sim_state`, `runner.py`,
> `store.py`, cổng kiểm — toàn bộ nằm ở repo serving). Mục này chỉ giữ phần thuộc xưởng/wheel.

**Ràng buộc wheel phải sửa:** mô phỏng phải **nhận được trạng thái đầu vào**. Đã đo: `PortfolioConstants` có
**29 field**, **không cái nào** là trạng thái (chỉ `date_lo`, `market_start`). Không sửa chỗ này thì không có
tính tăng dần.

**Quy mô việc đang lãng phí:** mỗi đêm chạy lại từ đầu — dyn300 ~**12.200** lệnh nền, dyn900 ~**31.800**,
nhân 9 chiến lược, chỉ để ra thêm một ngày.

⚠️ **Điều kiện tiên quyết** — quá khứ phải đứng yên trước (`RESTRUCTURE_STRATEGY_LAYER.md` §8.1); nếu không,
kết quả tăng dần **sai**. Nói **một lần** ở đây, không lặp lại chỗ khác.

### 13.6 Lưu trữ danh mục theo ngày: ĐÃ CHUẨN — giữ nguyên

> 📍 **`RESTRUCTURE_STRATEGY_LAYER.md` §13.4 SỞ HỮU** — số đo (112.701 dòng holdings · 1.520-1.613 phiên ·
> khoá `(bundle_id, date, symbol)` · **0,025 ms/truy vấn** · DB 26,8 MB), bốn khuyết tật UX, và cơ hội thanh
> trượt ngày. Mọi thứ đo và sửa đều ở repo serving; **không có việc gì phía xưởng**.

Kết luận cần nhớ: **cách lưu đã chuẩn, không tối ưu gì thêm.** Việc LƯU ổn; chỉ việc TÍNH lại toàn bộ mỗi
đêm là lãng phí (§13.5) — hai chuyện khác nhau, đừng lẫn.

### 13.7 Quyết định vận hành — phần thuộc XƯỞNG

> 📍 **`RESTRUCTURE_STRATEGY_LAYER.md` §13.6 SỞ HỮU** 5 quyết định có hành động ở repo serving:
> `build_bundle` là lối thoát hiểm · `k=150` không công bố biến thể · xoá 7 hàng seed + 2 hàng mồ côi ·
> không thông báo khách khi đổi engine · bỏ từ "tier".

Hai hàng dưới đây có việc **phía xưởng**, nên ở lại doc này:

| | Quyết định |
|---|---|
| **Xuất lại 3 bundle self-train** | ⚠️ **SỬA vòng-3 (§14.2):** cả 3 là chiến lược chào khách ⇒ pin-list + sửa slug **cả 3 ngay**; re-export **consw20+velov ngay** (chưa live), **wavestruct sau** (sự kiện công bố có chủ đích — đang là model active `.env`) |
| **Corp-action ở nguồn** | **Nguồn đã vá** (sieutinhieu phục vụ giá điều chỉnh, xác nhận 2026-07-31). Việc còn lại phía xưởng: **bỏ clamp/sanitize in-memory của loader train** (nay thừa, sẽ double-adjust — §9.7 sub-#1). Vế "kiểm đường cong đang lưu" thuộc doc B |

### 13.8 Thí nghiệm mẫu số xsec (nếu muốn chạy) — 3 bẫy

Sửa nhỏ (~10 dòng): `_load_xsec_features` **không có tham số tập mã**; nó xếp hạng bằng
`piv.pct_change(20).rank(axis=1, pct=True)` trên **mọi cột** của panel. Chỉ cần lọc cột trước khi xếp hạng.
`rs_*` trong cùng hàm tính theo VNINDEX nên **không bị ảnh hưởng**.

1. **`_XSEC_CACHE` khoá theo `tuple(sorted(metrics))`** — không theo đường dẫn, không theo tập mã. Quên mở
   rộng khoá ⇒ lần gọi thứ hai trả kết quả lần đầu ⇒ thí nghiệm báo "hai cách giống hệt", **kết luận sai**.
2. **Universe phải theo TỪNG NĂM** — dùng danh sách phẳng là xếp hạng bar 2020 dựa trên hiểu biết 2026
   ⇒ survivorship, và kết quả sẽ **đẹp giả tạo**.
3. **Fold cache** không chứa thông tin mẫu số ⇒ sửa template cũ tại chỗ sẽ **nạp lại dự đoán cũ**. Phải chạy
   như **template ID MỚI**.

**Chỉ đáng chạy trên nhóm 150 mã và dyn300** (lệch hạng 0,124 / 0,099). **dyn900 vô nghĩa** (0,004 — universe
của nó gần bằng cả thị trường).

---

### 13.9 ⛔ KHUYẾT TẬT ĐỊNH NGHĨA UNIVERSE — `universe_resolver.py` (phát hiện 2026-07-31)

**Đây là lỗi ở XƯỞNG, và nó đã đóng băng vào `universe_by_year` của mọi bundle dyn đang phục vụ.**
Không phải lỗi serving; serving chỉ kế thừa.

`stock_ml/src/data/universe_resolver.py:113`:
```sql
SELECT symbol, year(date) AS y, avg(volume * close) AS adv, count(*) AS c ...
```
rồi `:116` lọc `if c >= min_sessions`. Hai vế đều sai theo hợp đồng dữ liệu của nguồn:

**(1) `avg(volume * close)` không phải giá trị giao dịch.** `ohlcv_data` back-adjust **giá** về hiện
tại nhưng **không** back-adjust khối lượng ⇒ tích của chúng thấp hơn giá trị giao dịch thật đúng bằng
hệ số điều chỉnh luỹ kế, và méo **khác nhau tuỳ mã** — mã pha loãng mạnh méo nhất, tức đúng nhóm
penny mà `n:900` định lọc. Nó cũng **không tách được thoả thuận**: đo DVG `close×volume` 0.60 tỷ vs
khớp lệnh thật 0.03 tỷ (**thổi 20×**), OPC công bố 68,19 tỷ trong đó 68,11 tỷ là thoả thuận.
Trên bar **gần đây** sai số nhỏ (median 1.00×, p10 0.98, p90 1.09 trên 471 mã) — nó **tích luỹ lùi về
quá khứ**, đúng chiều mà walk-forward cần chính xác nhất.

**(2) `count(*)` không phải số phiên giao dịch.** Bảng phát sinh dòng cho cả phiên mã **không khớp
lệnh**. Đo trên `market_data/market.duckdb` của chính xưởng: **22,1%** bar có `volume = 0`. Hệ quả
lên chính cổng `min_sessions=100`:

| năm | mã qua cổng | trong đó là mã **CHẾT** vẫn qua |
|---|---|---|
| 2019 | 811 | **255 (31%)** |
| 2020 | 836 | **247 (30%)** |
| 2021 | 866 | 146 (17%) |
| 2022 | 881 | 174 (20%) |
| 2023 | 884 | 185 (21%) |
| 2024 | 893 | 181 (20%) |
| 2025 | 779 | 82 (11%) |

**Chuỗi nhân quả đã khép kín.** dyn900 có `n: 900` là **trần chưa bao giờ chạm** (thực tế 775) ⇒ ràng
buộc thật là `min_sessions` ⇒ mà `min_sessions` đếm phiên ma ⇒ universe nhận 11-31% mã đã chết mỗi
năm ⇒ **"dyn900 bị penny thổi phồng CAGR" KHÔNG phải hiện tượng mô hình, nó là lỗi định nghĩa
universe.** Nặng nhất đúng ở 2019-2020 (30%) — chính là dữ liệu nuôi universe 2020-2021.

**Sửa:** `sessions_traded` thay `count(*)`; ADTV khớp lệnh (`basis=matched`, `market_flow_1d`) thay
`avg(volume*close)`. Cả hai nguồn đã có sẵn ở endpoint §13.3.1 — **không phải xây mới**.

#### 13.9.1 Bar ma còn làm hỏng FEATURE, không chỉ cổng đếm (đo 2026-07-31)

Nguồn phát sinh bar cho **cả phiên mã không khớp lệnh**, và **97,2%** số bar đó phẳng tuyệt đối
(`open=close=high=low`). Chúng bơm vào mọi cửa sổ trượt những ngày **lợi suất 0, biên độ 0** ⇒ làm
**giảm** độ biến động / ATR / biên độ đo được. Cổ phiếu rác trông **êm ả**, tức *hấp dẫn* với model
ưa cấu trúc giá mượt. Pipeline **không lọc chúng**: `loader.py:59` chỉ `dropna`; `duckdb_loader.py`
có bộ cắt tiền-IPO nhưng docstring ghi rõ *"Volume is deliberately not used"* — nó chỉ cắt đoạn
phẳng **ở đầu chuỗi**, bar ma **giữa lịch sử** đi thẳng vào feature.

**Nhưng đo ra thì đây KHÔNG phải vấn đề toàn hệ — nó là vấn đề của riêng dyn900:**

| bar `volume=0` trong universe đang train | |
|---|---|
| dyn300 | **1,6%** |
| dyn900 | **20,6%** (148 mã có >50% lịch sử là bar ma; 254/775 mã ≥25%) |

Mức bóp méo `vol20` trên chính dyn300 (217/300 mã có bar ma):

| nhóm bar ma | số mã | vol thật / vol đo được | biên độ thật / đo được |
|---|---|---|---|
| <1% | 162 | **1,000×** | 1,001× |
| 1-5% | 30 | 1,010× | 1,023× |
| 5-15% | 19 | 1,025× | 1,089× |
| >15% | 6 | 1,124× | 1,271× |

Median toàn dyn300 = **1,000×** ⇒ **không đáng kể**.

#### 13.9.2 ⇒ SỬA ĐÚNG CHỖ: top-N đã tự lọc, vấn đề là ĐỘ SÂU của dyn900

Đếm mã bẩn (≥25% bar ma) **bên trong chính chính sách top-N**:

| | 2020 | 2022 | 2024 | 2026 | ADTV của mã thứ N |
|---|---|---|---|---|---|
| **top-300** (dyn300) | 1 | 0 | 1 | 1 | 0,48 → 6,79 → 2,50 → 2,64 tỷ |
| **top-900** (dyn900) | 73 | 83 | 80 | 98 | **0,00 → 0,06 → 0,02 → 0,03 tỷ** |

**Xếp hạng theo ADV tự nó đã lọc**: mã nhiều bar ma = mã ít giao dịch = hạng thấp. top-300 dính
0-2 mã bẩn mỗi năm (0,0-0,7%). top-900 dính 9-11% **vì nó với xuống tới mã có ADTV ≈ 0** — `n:900`
không phải "900 cổ phiếu tốt nhất", nó là "gần như mọi mã tồn tại".

⇒ **KHÔNG cần tầng làm sạch bar riêng, và KHÔNG cần thêm sàn ADTV cho dyn300.** Việc phải làm:
1. `sessions_traded` thay `count(*)` — loại mã thật sự đã chết
2. **dyn900: `n:900` là lỗi thiết kế.** Phải chọn: hạ `n`, đặt sàn ADTV thật, hay cho nghỉ hẳn.
   Nhắc lại §14.1: `dyn900 ∩ sàn ADTV` == `dyn300 ∩ sàn` ở mọi ngưỡng ⇒ phương án "dyn900 có lọc"
   trùng khít dyn300.
3. Ghi `tv_source ∈ {measured, reconstructed}` khi `back_adjust.py:214` phải tái dựng (95,19% dòng
   `traded_value` hiện là tái dựng) — để lần sau không ai tưởng nó là số đo.

⚠️ **Sửa xong thì universe ĐỔI, và mọi số đổi theo.** Đã đo mức lệch: universe do nguồn resolve so với
`universe_by_year` xưởng đã bake trùng **89-96%**, tăng dần theo năm (2020: 268/300 → 2025: 289/300).
Đây là **re-baseline có chủ đích**, đúng §11 ("nâng engine ⇒ tính lại tất cả") — nhưng phải chốt
trước, không được để nó xảy ra như một tác dụng phụ của việc port code.

⚠️ **Đừng port resolver sang serving.** Kế hoạch cũ (serving B3) là chép `universe_resolver.py` sang
`stock-serving`; làm vậy là **nhân đôi đúng hai khuyết tật trên**. Serving gọi thẳng
`/symbols/universe`. Bên xưởng thì `universe_resolver.py` nên trở thành **wrapper mỏng quanh cùng
endpoint đó**, không phải bản cài đặt thứ hai của cùng một chính sách.

**Ảnh hưởng lan sang tầng danh mục (serving sở hữu số đo):** gate `liqcol_*` trong
`stock_ml/portfolio/api.py:55` dùng **cùng công thức sai** `tvv = close × volume`, và `:180-187` chỉ
áp gate khi có dữ liệu ⇒ mã ngoài panel **đi thẳng qua, không lọc, không ghi skip** (fail-open: 3 mã
dyn300, **321 mã dyn900**). Đây là định nghĩa của `floor5`/`floor10` — hai sản phẩm tiền thật đang
chào khách. Chi tiết + thứ tự vá: `stock-serving/DEPLOY_DYN_TIERS.md` §3.1 **B11**.

---

## 14. Chốt 2026-07-31 (vòng 3) — danh mục 10 chiến lược · khớp float · thước NAV · re-export tĩnh

> Vòng review đối kháng thứ 3 trên chính tài liệu này, tập trung các câu §11–§13 còn mở. Mục này **ghi đè**
> mọi con số "6 sổ"/"9 chiến lược" rải rác (§2.4/§7/§11.3/§11.5.4/§11.7), quyết định `nh_nav2` (§4.2), và
> dòng "3 self-train" ở §13.7. Chẩn đoán §1–§13 **không đổi** — chỉ chốt các câu còn treo. Chốt qua Q1–Q4
> với chủ dự án + đối chiếu `tiers.yaml` thật của serving.

### 14.1 Danh mục CHÍNH THỨC = 10 chiến lược (thay "6 sổ"/"9 chiến lược")

| # | Chiến lược | Model | Khối `portfolio:` khác biệt | Chào khách |
|---|---|---|---|---|
| 1 | dyn300_floor5 | dyn300 | sàn ADV 5 tỷ (mặc định) | ✅ |
| 2 | dyn300_floor10 | dyn300 | sàn ADV 10 tỷ | ✅ |
| 3 | dyn300_liqcol | dyn300 | cho mã mỏng | ✅ |
| 4 | dyn300_k6 | dyn300 | k=6, tập trung | ✅ |
| 5 | dyn300_invvol15 | dyn300 | cân theo biến động | ✅ |
| 6 | dyn61a2hy | dyn61a2hy | mặc định (vốn lớn, DD thấp) | ✅ |
| 7 | dyn900_k16 | dyn900 | k=16 | ✅ ⚠️ |
| 8 | wavestruct | wavestruct | tĩnh top-150 | ✅ |
| 9 | consw20 | consw20 | tĩnh top-150 | ✅ |
| 10 | velov | velov | tĩnh univ150c | ✅ |

**Đối chiếu số cũ:** "6 sổ" = 6 tier dyn chào khách (dòng 1–6); "9" = ước tính cũ; **đúng là 10**. Catalogue
đã **trôi** kể từ lúc §1–§13 đo (0.4.2 thêm dyn61a2hy + floor5/floor10/invvol ngày 29/07) — bằng chứng sống
cho nhu cầu nguồn-sự-thật-duy-nhất.

**Cấu trúc (§12 làm cụ thể):** bỏ tầng **tier** + đường **`.env`-active**; mỗi chiến lược = **một file config**
có khối `portfolio:` (§13.1), một `config_hash` (§13.2), một dòng board. 5 dòng dyn300 chung **một model** →
5 config khác nhau **đúng ở khối `portfolio:`** → 5 hash riêng, xếp hạng cạnh nhau. **Hiệu suất:** gom theo
model ⇒ tầng tín hiệu (ML) chạy **một lần/model**, chỉ **replay overlay** khác nhau (rẻ) — đây là móc cho
tính-tăng-dần §13.5; model trỏ bằng hash (§12.2), không nhân bản.

⚠️ **dyn900_k16 (#7) chào khách — nhưng CAGR bị PENNY thổi phồng** (đo: ~74% vốn vào mã <10 tỷ ⇒ ~205% ảo vs
gate ADV chuẩn ~66%). Khi xếp cạnh 9 cái kia nó **trông tốt nhất một cách giả tạo**. **Bắt buộc** kèm **nhãn/blurb
cảnh báo penny** — và **KHÔNG làm floor-variant**: đo lại từ nguồn 2026-07-31 (`/symbols/universe`,
`basis=matched`, as_of 2026-01-02), `dyn900 ∩ sàn` **bằng đúng** `dyn300 ∩ sàn` ở mọi ngưỡng —
5 tỷ: **228/228** · 10 tỷ: **179/179** · 20 tỷ: **139/139**, tức **0 mã thêm** ⇒ floored-dyn900
**trùng khít dyn300**. (Số cũ "+2 mã" tính bằng `close×volume` — sai thước, xem §13.9.)
Ghi vào overlay Phase 5.

⚠️ **Nhưng nguyên nhân penny của #7 nằm SÂU HƠN overlay** — nó là **lỗi định nghĩa universe** ở
`universe_resolver.py`, xem **§13.9**. Dán nhãn cảnh báo là vá triệu chứng; sửa `sessions_traded` +
ADTV khớp lệnh mới là sửa gốc, và sửa xong thì universe dyn900 đổi ⇒ hàng #7 phải **re-baseline**,
không chỉ đổi blurb.

**Hệ quả kéo theo:** re-baseline = **6 model × full-history + 10 overlay-replay** (KHÔNG phải "10 × full-history" —
5 biến thể dyn300 chung 1 model; ngân sách đo được ~19 phút single-seed, xem §11.7 M1); test tương thích cấu hình
§11.5.5 lặp trên **10 config**; `config_hash` + config-compat phủ khối `portfolio:`.

### 14.2 Re-export 3 bundle tĩnh — pin-list ngay, re-baseline TÁCH wavestruct (Q2)

- **(i) Pin danh sách 150 mã + sửa slug sai — CẢ 3 NGAY, KHÔNG đổi số.** Config ghi `vn_stock_default` (=61 mã
  trong DB) nhưng bundle chạy **150 mã** (§9.9); list chỉ ở `manifest.universe`, chưa versioned. Đăng ký set
  versioned vào `universe_sets/versions`, checksum (R4).
- **(ii) Re-export dưới wheel đã tag + DuckDB back-adjusted — ĐỔI số công bố:**
  - **consw20 + velov: NGAY** — chưa phục vụ khách nên **không mất gì để công bố lại**.
  - **wavestruct: SAU**, như **sự kiện công bố có chủ đích, có log** — đang là model active trong `.env`, đổi
    số nó là sự kiện chạm khách.
- Động cơ D1 "re-export cả 5 ngay" (làm mốc **R0**) đã **hết hiệu lực** vì R0 bị bỏ (§11.4). §11.6: Phase 1
  nhận (i); Phase 4 tách (ii) — consw20/velov vs wavestruct.

### 14.3 Chính sách khớp float cho test tương đương §11.5.3 (Q3)

- **Tầng tín hiệu: bit-exact TUYỆT ĐỐI**, không dung sai (không có wrapper serving-only — §9.6).
- **Tầng lệnh: bit-exact SAU KHI** cả hai cùng áp `max_hold` + cùng bộ lọc config (whitelist đúng **2** biến
  đổi cố ý — R3/§9.6).
- **Mặt phẳng: Linux/container DUY NHẤT** authoritative; pin **sklearn + numpy + lightgbm** chính xác + ép
  **đơn luồng (OMP/BLAS=1)**; Windows chỉ tham khảo (lệch VND 40 vs 37 đã đo).
- **Điều kiện khóa "no tolerance":** chạy **2 lần trong container, chứng minh 0-diff** TRƯỚC khi khóa chính sách.
- **Fallback pre-commit:** nếu 2-run vẫn lệch (nhiễu BLAS còn sót) → **ε hẹp CHỈ cho tầng lệnh**, ε **nhỏ hơn**
  hiệu ứng rớt-tầng nhỏ nhất từng đo; tầng tín hiệu **vẫn exact**.

### 14.4 Thước NAV — BỎ nh_nav2, thống nhất về `stock_ml.portfolio` (Q4)

- **Official NAV = code trong wheel** (`stock_ml.portfolio`) mà chính các chiến lược dùng, **KHÔNG** phải
  scorer ngoài repo. Nguồn giá về **sieutinhieu** (§9.7).
- **Bỏ `nh_nav2`** (`F:/PROJECTS/hb2943_work`, ngoài mọi git) — nó là **codepath NAV thứ hai**, cùng loại
  landmine `portfolio_engine` #2 (§10.2). Thay đề xuất "đưa nh_nav2 vào repo" ở §4.2.
- **NAV vào khung attestation:** test-song-sinh overlay (§9.6.1b) trên `stock_ml.portfolio` chứng thực **chính
  con số official**. Trước đây NAV nằm **ngoài** mọi cổng khớp (§9.6.1 điểm 6) — nay đóng.
- **Việc:** xác nhận `stock_ml.portfolio` tái tạo đúng số official (hoặc chấp nhận **re-baseline board 1 lần**,
  gộp vào sự kiện re-baseline). Kéo theo: `score_nav_leaderboard.py` (§4.1/§4.2) thôi import nh_nav2; golden
  `_champ_prod_replay.py` + `test_portfolio_golden` (md5-pin nh_nav2 — §2.2/§5) chuyển sang pin trên
  `stock_ml.portfolio`.

### 14.5 M1 (đo) + T1–T4 (thiết kế/quy trình)

- **M1 — ✅ ĐÃ ĐO (2026-07-31).** Cold full-history (6 fold, single-seed): dyn61 **55s** · dyn300 **208s** ·
  dyn900 **486s** · tĩnh-150 nội suy **~115s**; fold-cache chỉ giảm ~20%. **Đính chính then chốt:** re-baseline
  KHÔNG phải "10 × full-history" mà là **6 model cold-train + 10 overlay-replay** (5 biến thể dyn300 chung 1
  model) ⇒ ngân sách ML **~19 phút single-seed** (~50 phút nếu 3-seed các sổ deploy). Chi tiết + đính chính đã
  ghi vào **§11.7** và §14.1. Còn treo: đo riêng overlay-replay (`PortfolioContext` không memoise).
- **T1 — training GHI LẠI `market_panel`.** Ghi tại train-time vào run record (số mã + sha + theo năm) → mang
  vào `resolved.json` lúc xuất. Không thì cổng xuất §13.3 **chỉ kiểm được chính tả**.
- **T2 — engine chọn fold-model theo ngày (§11.9).** Xác nhận **năm live dùng fold-model cuối** (khớp cách
  backtest xử năm cuối); ghi **ranh giới fold** (năm→fold-id) vào `resolved.json`.
- **T3 — giao thức release-gate cross-repo (§9.3.5).** serving pin `stock_ml_core==<exact>`; đổi facade →
  **bump MINOR + cập nhật pin trong CÙNG một cặp PR**; một release-note chung.
- **T4 — size Phase 0–6** (owner: chủ dự án quyết, Claude thực thi): **P0 S · P1 L · P2 M · P3 L · P4 M · P5 L ·
  P6 M**. P1/P3/P5 nặng nhất (data infra cross-repo · bản-xuất-tự-đủ + ghim + facade · danh tính + di sản + overlay).

### 14.6 Còn treo sau vòng 3 (không chặn, ghi để không quên)

- **M1 ✅ đã chạy (2026-07-31)** — mô hình một-engine (§11.3) nay có số thật: ~19 phút single-seed cho 6 model
  (§11.7). Còn treo phụ: đo riêng chi phí overlay-replay tầng portfolio (chưa memoise).
- **`stock_ml.portfolio` có tái tạo đúng số official của nh_nav2 không** — chưa đo; nếu lệch thì §14.4 kéo theo
  một lần re-baseline board (chấp nhận được dưới §11.1, nhưng phải có log + công bố).
- **dyn900_k16 = dùng NHÃN cảnh báo penny** (floor-variant BỊ BÁC — serving D4 đo floored-dyn900 ≈ dyn300); áp ở Phase 5.

---

### 14.7 🔴 CHỐT 2026-07-31 — TÍNH LẠI + TRAIN LẠI TRÊN DATA SẠCH

**Chủ dự án chốt:** *"Tất cả các con số đang chưa chuẩn và clean đều cần phải tính lại, thậm chí
train lại với data sạch hơn."* ⇒ Ràng buộc "không được đổi số đã công bố" **được gỡ hoàn toàn**.

**Hệ quả trực tiếp lên tài liệu này:**
- Mọi knob/cờ/dual-write tồn tại **chỉ để giữ hành vi cũ** là **rác** — không dựng. Chính cơ chế
  tương thích-ngược là thứ đã đẻ ra 4 bản sao sai của cùng một chính sách (§13.9).
- **§13.9 chuyển từ "việc quý sau" lên ĐƯỜNG CHÍNH.** Sửa thước ⇒ universe đổi ⇒ train lại
  walk-forward ⇒ export lại. Đây không còn là hệ quả phải né, nó là mục tiêu.
- Chia nhỏ re-baseline theo đợt để thông báo: **bỏ**. Một sự kiện, có log.
- `customer:false` của dyn900 (§14.1 #7) là cách né hệ quả của lỗi định nghĩa universe. Sau khi
  sửa `n` thì phải **quyết lại từ census mới**, không giữ nguyên nhãn cũ.

**8 điều kiện "data sạch" phải đúng TRƯỚC khi bấm train + thứ tự thực thi:
`stock-serving/DEPLOY_DYN_TIERS.md` §7 SỞ HỮU** (repo đó nắm store + panel + coverage).
Phần thuộc xưởng trong danh sách đó: điều kiện 2/3/4/5 = sửa `universe_resolver.py` (§13.9),
và bước 5 = train lại walk-forward.

⚠️ **Ba thứ phải copy ra hiện vật TRƯỚC khi động vào bất cứ gì** — thông tin không tái tạo được:
`universe_by_year` trong manifest 3 bundle dyn (nguồn **tự viết lại lịch sử**: 397 mã hủy niêm yết
được bổ sung 31/07 ⇒ universe đã train không dựng lại được) · 2 file golden pin (phải **dời khỏi**
`market_data/` trước khi rebuild panel; nghiệm thu **hai chiều**) · `breadth_universe.txt` 488 dòng
(3 bundle tĩnh **không ghi** panel lúc train — xoá là phục vụ chúng bằng breadth khác, im lặng).

⚠️ **`export_bundle.py` phải sinh `parity.json`** trong đợt này: đo 31/07 **0/6 bundle** có file đó
⇒ `gate_parity` của `serving/deploy.py` hôm nay không bao giờ qua được, mọi deploy đi bằng
`--force`. Cổng đang là hình thức.

## 15. Nhật ký triển khai (execution log)

> Cập nhật sau mỗi phase/sub-task hoàn thành. Thứ tự theo §11.6. ✅ xong · 🔄 đang làm · ⏳ gated/hoãn có chủ đích.

### Phase 0 — Chặn mất mát — 🔄 (bắt đầu 2026-07-31)

- ✅ **0a. Auto-prune bundle TẮT.** `stock-serving/docker-compose.yml`: `PRUNE_BUNDLES=1 → 0` (+ comment lý do).
  Xác minh cơ chế: `serving/scheduler.py:181` gate toàn bộ prune bằng `if args.prune_bundles`, mà
  `:154` đọc env `PRUNE_BUNDLES` mặc định `"0"` ⇒ đặt `0` là skip sạch. **Hiện trạng an toàn:** worker đang
  chạy vẫn env cũ `=1` NHƯNG có 6 bundle ≤ `KEEP_BUNDLES=9` nên `remove_dirs` rỗng → KHÔNG prune gì lúc này.
  Thay đổi thành **bền** khi `docker compose up -d serving-worker` tạo lại container (không gấp vì chưa có
  rủi ro tức thời). Prune-theo-registry (`retired:true`, §11.8) hoãn sang **Phase 5** (registry lifecycle chưa tồn tại).
- ⏳ **0b. Backup Postgres + dry-run migration — GATE gắn vào Phase 5, KHÔNG chạy bây giờ.** Lý do: chưa có
  migration nào trước Phase 5; dump 98GB lúc này sẽ **CŨ** (DB đổi giữa nay và Phase 5) → vô ích + tốn đĩa.
  Kỷ luật đúng = backup + dry-run **ngay trước** migration đầu tiên. Thủ tục đã soạn:
  - Backup: `docker exec stock-ml-postgres pg_dump -U stockml -d stockml -Fc -f /tmp/stockml_preP5.dump`
    rồi `docker cp stock-ml-postgres:/tmp/stockml_preP5.dump <host>/backups/` (creds: user/db=`stockml`, pass `stockml_dev`).
  - Dry-run: restore vào DB tạm `stockml_dryrun` → `alembic upgrade head` → assert 0 lỗi + shape đúng → drop.

**Trạng thái Phase 0:** hành động bảo vệ tức thời (prune) XONG; backup là gate có kỷ luật gắn Phase 5. Sẵn sàng sang Phase 1.

### Phase 1 — Dữ liệu giống nhau hai bên — 🔄 (bắt đầu 2026-07-31)

> Phát hiện định hình lại Phase 1: **kích hoạt VNINDEX trong container / đổi nguồn sang duckdb (mới hơn) là
> ĐỔI SỐ production**, không phải hạ tầng thuần. VNINDEX **CÓ** trong `market_data/market.duckdb` (ohlcv, 5614
> dòng, tới 2026-07-27) trong khi CSV cũ dừng 2026-06-16 ⇒ chuyển nguồn = đổi cả train lẫn serving. Cách chuẩn:
> **xây cơ chế (0 đổi số) trước → cutover đổi-số làm có đo, checkpoint với chủ dự án.**

- ✅ **1a-mechanism. Reader VNINDEX/runscore thống nhất + fail-loud env-guard (mặc định TẮT = 0 đổi số).**
  `engine.py`: thêm `_market_context_required()` (đọc env `MARKET_CONTEXT_REQUIRED`, default OFF); `_load_vnindex`
  và `_load_runscore` nay **raise** khi nguồn vắng + env bật, giữ trả None khi env tắt (hành vi cũ). `experiment.py`
  `_load_xsec_features`: bỏ block đọc CSV trùng lặp (1716-1727), gọi chung `_load_vnindex` ⇒ **một reader, một
  ngữ nghĩa lỗi** (hết cảnh engine trả None-im-lặng vs experiment KeyError). **Verify:** parity vs CSV = True
  (2857 dòng, giống hệt), guard raise đúng cho cả 2 reader, `py_compile` OK, `_os` không còn sót.
- ⏳ **1a-cutover (ĐỔI SỐ — checkpoint trước khi làm).** Đưa reader qua **resolver có hợp đồng fetch-on-miss**
  (§13.3.1): khai VNINDEX/rs + `as_of` trong config → cache local (duckdb) thiếu/cũ thì **fetch từ sieutinhieu**
  → chạy; bật `MARKET_CONTEXT_REQUIRED=1` trong container. ⇒ VNINDEX/rs knob **kích hoạt production** (khớp xưởng),
  data về đúng `as_of` (KHÔNG "bê duckdb local cũ"). Bắt buộc chụp baseline champion trước, đo delta sau.
- ⏳ 1b (manifest checksum breadth/xsec — phần metadata 0 đổi số làm được ngay; đổi-nguồn đổi-số checkpoint), 1d
  (universe tĩnh versioned), 1c/1e (đổi số — checkpoint).
- ✅ **1d. Universe tĩnh versioned (0 đổi số).** Đăng ký 2 set **KHÓA** vào `universe_sets`/`versions`/`symbols`
  từ manifest 150-mã: `vn_top150_adv` (id 9 — wavestruct≡consw20, sha `21749fe9…`) + `vn_univ150c` (id 10 —
  velov, sha `0c09a1a2…`); mỗi set 150 mã, `is_locked=t`, sha verified. Artifact tracked:
  `scripts/ops/register_static_universe.py` + `scripts/ops/data/static_universe_sets.json` (idempotent, reversible
  vì set khóa). **CÒN (Phase 4/D1):** 3 config bundle ghi slug sai `vn_stock_default`(=61) → trỏ về set đúng
  (wavestruct/consw20→`vn_top150_adv`, velov→`vn_univ150c`) = đổi-số, gộp vào re-export. Tiếp theo: **1b-metadata**.
- ✅ **§13.9 + §13.3.1 — SỬA THƯỚC UNIVERSE + HỢP ĐỒNG FETCH (ĐỔI SỐ, build trọn trong xưởng — chủ dự án duyệt).**
  Gốc §13.9: `universe_resolver.py` xếp hạng bằng `avg(volume*close)` (giá back-adjust × volume không-adjust →
  méo tới 20× do thoả thuận) + `count(*)` (22% bar volume=0 ma) ⇒ 11-31%/năm universe là mã đã chết. **Fix
  đúng nguồn** (không vá SQL local): resolve từ sieutinhieu `/symbols/universe?basis=matched` — server xếp
  hạng matched-ADTV + gate `min_sessions_traded` thật, point-in-time & append-only (ổn định theo `as_of`).
  - **(1)** `src/data/sieutinhieu.py` — client public (universe + ohlcv), fail-loud, one-shot full-history
    (server đã nâng trần bar 1000→**50000/call**, xác nhận qua OpenAPI + chủ dự án báo).
  - **(2)** `universe_resolver.py` — đổi nguồn thước sang endpoint; **giữ nguyên** logic lookback-min /
    hysteresis / sticky / top_n / is_nonstock. Map cửa-sổ: `sessions=250` (trần server) ≈ 1 năm trước;
    prior_2y = 2 call lấy `min` ADTV. `duck` param còn nhận nhưng KHÔNG dùng (endpoint là nguồn). 12 test
    resolver rewrite sang **mock `fetch_universe`** (deterministic, không network) — pass.
  - **(3)** `duckdb_loader.ensure_symbols_cached` — **fetch-on-miss** (§13.3.1): universe đúng chọn mã
    survivorship (ROS/FLC/mã hủy niêm yết) local cache KHÔNG có ⇒ fetch full-history từ `/ohlcv/` + upsert
    vào `ohlcv` (traded_value=volume×close). Wire vào `experiment.py` (train) + `export_bundle.py` (re-export),
    guard dưới dynamic-path. 2 test mock — pass. Nếu KHÔNG có bước này, mã thiếu bị `[s for s in symbols if s
    in available]` **âm thầm drop** → universe co về survivors, đúng bias §13.9 xoá.
  - **(4) BẰNG CHỨNG đo (new resolver vs manifest cũ, dyn900):** mỗi năm **−260..274 mã chết** ra, **+260..306
    mã** survivorship vào (~30% churn/năm, khớp bảng §13.9). Old-n 807-889 (chưa chạm trần 900 = "trần chưa bao
    giờ chạm"); new chạm 900. 2020 top-1 = **ROS** (thanh khoản nhất 2020 thật, nay hủy niêm yết — thước cũ
    không bao giờ có). ⇒ dyn900 re-baseline sẽ ra số khác HẲN (đúng), penny-CAGR ảo được sửa TẠI GỐC.
  - **Verify:** ruff sạch mọi file đụng (N802 dòng 1306 experiment.py = tiền-tồn, không đụng); full unit
    **313 pass, 12 skip** (0 regression; 13 fail = engine#2 chết §10.2/Phase 6). **⇒ điều kiện tiên quyết
    re-baseline (§13.9) XONG — nay re-baseline 1 lần đúng thước.**

**Cross-repo — serving `RESTRUCTURE_STRATEGY_LAYER.md` (đối chiếu 2026-07-31):** đã đồng bộ vòng-3 (D1/D4 = 10 chiến
lược, dyn900 chào khách, D6 = nguồn sieutinhieu; có S1-S11 = bản thực thi phía serving). Đối chiếu: **S1(i) =
Phase 0a** (`PRUNE_BUNDLES=0`, đã đánh dấu XONG bên đó); **serving D4 tinh chỉnh §14.1** (dyn900 floor trùng dyn300
→ dùng nhãn, đã cập nhật); **serving nhắc:** bật fail-loud "thiếu mã thì dừng" hôm nay = **6/6 sổ khách ngừng**
(dyn900 thiếu 321 mã) ⇒ **1a-cutover phải chờ hợp đồng fetch (§13.3.1) lấp đủ mã trước**. Hai S-plan/Phase-plan
đồng bộ khi tới các bước cross-repo (prune-registry, universe serve-year, strategies.yaml, attestation).

### Phase 2 — Cái cân — 🔄 (bắt đầu 2026-07-31)

> Chọn **hướng B** (chủ dự án ủy quyền "clean/không-nợ"): dựng test validate TRƯỚC khi chạm số production.

- ✅ **2a. Sửa CI (§1.5) — CI chạy test THẬT thay vì path chết.** `.github/workflows/ci.yml`: `typecheck` bỏ
  `mypy src/components/` (đã xóa DB-first); `test-unit` `tests/components/`(không tồn tại)→`tests/` **250 pass**,
  loại 7 file nợ-test (mỗi cái comment lý do); `test-regression` `tests/regression/test_champions.py`(không tồn
  tại)→`test_baseline_snapshot + test_recombine_snapshot` **4 pass** (TODO nâng lên R6/§11.5.3 ở 2b); + quote
  `DATABASE_URL: "sqlite:///:memory:"` (YAML fragile). **Verify:** 250+4 xanh, ci.yml valid YAML (5 job parse).
- 🔎 **Phát hiện: chạy cây test thật lộ 30 fail = nợ TRONG test-suite, KHÔNG phải engine hỏng** (278 pass).
  Phân loại + disposition (đã ghi vào comment CI):
  - 13 `test_portfolio_engine.py` → engine #2 chết (§10.2) → XÓA cùng engine (Phase 6).
  - 8 `model_dashboard/` → schema drift `LeaderboardRow.exit_model_type` → FIX (model_dashboard = hộp research giữ, §10.1.1).
  - 2 `test_per_slot_features.py` → đọc `config/experiments/*.yaml` (hệ YAML đã xóa) → XÓA.
  - 2 `test_loader_layouts.py` → API `DataLoader.symbols` cũ (layout DuckDB-era §4.7) → XÓA.
  - 3 `test_dsl_ops` + 1 `test_resolver` → assert_close so builder cũ quá chặt (values BẰNG NHAU, diff=0) → sửa assert / bỏ builder-parity.
  - 1 `test_run_experiment_e2e` → integration cần data/bundle committed.
  ⇒ Dọn test-debt = follow-up (phần gắn §10.2/Phase 6, phần fix-schema); **KHÔNG che** — mỗi exclusion có lý do trong CI.
- ✅ **2c. config-compat test Level-1 (§11.5.5) — "bảo đảm ngược THẬT".** `tests/test_config_compat.py` +
  6 fixture snapshot (`tests/fixtures/bundle_configs/{dyn300,dyn61a2hy,dyn900,wavestruct,consw20,velov}.json`):
  khẳng định engine **nạp `ExperimentConfig(**config)` cho CẢ 6 bundle sống** không lỗi — bắt đúng lớp
  `universe_policy`/0.4.1 (§11.2). **Verify:** 7 pass; CI test-unit 250→**257**. Level-2 (`engine_config_from_dict`
  3-chiều — R3/Phase 3 + short signal cycle) nối sau. Refresh fixtures khi re-export (Phase 4).
- ✅ **M1. Đo chi phí re-baseline (§11.7/§14.5) — điều kiện chốt mô hình một-engine.** Chạy cold full-history
  (6 fold, single-seed) qua `run_template.py` với `STOCK_DATA_DIR` trỏ `market_data/market.duckdb` (fresh out-dir
  ⇒ cache-miss). **Số đo (máy dev Windows):** dyn61 (61 mã, 96k signal) **55s** · dyn300 (300, 469k) **208s** ·
  dyn900 (901, 1,22M) **486s** · wavestruct-template (chạy 61-mã vì config còn `vn_stock_default`, 96k) 45s ⇒
  tĩnh-150 nội suy **~115s**. Warm-rerun dyn61 **44s** (fold-cache chỉ −20%: backtest + persist signal mới nặng,
  không phải fit LGBM). **Kết luận:** re-baseline = **6 model cold-train + 10 overlay-replay**, KHÔNG phải
  "10×full-history" (5 dyn300-variant chung 1 model) ⇒ **~19 phút single-seed** (~50 phút nếu 3-seed sổ deploy);
  overlay-replay chưa đo riêng. Đã sửa §11.7 + §14.1/§14.5. **Lưu ý:** các run này upsert vào leaderboard dev
  (run_id `template/_dyn*_onerun-*`, deterministic ⇒ khớp số cũ); temp out-dir `results/_m1_cost/` đã dọn.
- ✅ **2b. Golden full-chain KHÔNG-CÔNG-TẮC (§11.5.3) — "cái cân" thật trên production-path.**
  `tests/test_serving_golden.py` + fixture tự chứa `tests/fixtures/serving_golden/` (bundle 500K + slice
  76K + golden 32K + `regen.py`): nạp bundle → `generate_signals_from_bundle` (build_feature_frame →
  predict_slot_signals → recombine_signals) trên slice OHLCV committed, khẳng định **cột signal rời rạc
  khớp byte-for-byte**. **No-switch:** thiếu fixture ⇒ FAIL (không auto-create/skip/regen), không tolerance.
  **Phạm vi (chốt vòng-3):** signal-level, chạy mọi nền (rời rạc ⇒ ổn định Win↔Linux); tầng trade/NAV
  nhạy-float là lớp Linux-only sau. **Tự chứa:** dùng strategy per-symbol thuần (template 112 `sumEX_thr15`,
  `regression_dual_ml_recombine`, features `leading_v2`) — KHÔNG dùng champion vì champion đọc breadth qua
  `market_data/market.duckdb` CWD-relative **bỏ qua STOCK_DATA_DIR** (đúng defect §1.2, chờ hợp đồng data
  Phase 1 mới inject được). Vào CI `test-regression` (thay path tạm 2a) + chạy trong `test-unit`. **Verify:**
  golden tái tạo (chạy 2 lần signal y hệt); CI test-unit 257→**259 pass**; ruff+format sạch; fixture parquet
  thoát blanket `.gitignore` bằng negation tường minh. Refresh sau re-baseline (Phase 4): `regen.py`.
- 🔄 **2d. Dọn test-debt (30 fail phân loại ở 2a) — phần SỬA xong, phần XÓA chờ xác nhận.**
  - ✅ **dsl_ops + resolver (4 fail → sạch).** Gốc thật KHÔNG phải "assert quá chặt" (2a đoán sai): `nanmax`
    **giấu** mismatch NaN-vs-giá-trị. DSL để **NaN warmup honest**, golden legacy **điền 50.0** neutral; phần
    hữu hạn byte-identical (diff 0.0). Sửa `assert_close`: giá trị hữu hạn khớp chặt + chấp nhận DSL từ-chối-bịa
    **warmup đầu chuỗi liền mạch** (NaN giữa chuỗi hoặc DSL-bịa-nơi-golden-NaN = FAIL). Riêng `dist_52w_high`/
    `dist_52w_low` = lệch **min_periods THẬT** (DSL `Max/Min(252)` strict-window vs legacy expanding) lộ ra vì
    fixture chỉ 160 bar < 252; sửa sẽ ĐỔI SỐ production ⇒ skip có-lý-do + đưa vào **cụm đổi-số**, không phải
    test-debt (golden đóng băng 160 bar, builder đã xóa nên không regen dài hơn được). → 28 pass, 1 skip.
  - ✅ **model_dashboard (8 fail → 9 pass).** KHÔNG phải test-debt mà **bug CODE thật**: `model_dashboard/
    schema.py:133,193` đọc `row.exit_model_type` — cột đã bỏ khỏi `LeaderboardRow` ⇒ `model_id_for_row` crash
    mọi lần gọi (không có ID cũ cần giữ). Thay bằng `row.model_mode` (nay mã hóa entry/exit composition).
  - ✅ **XÓA test chết (chủ dự án duyệt "xóa 4 test chết, giữ test sống"):** bỏ 2 test `from_yaml`
    (`test_per_slot_config_backward_compat` + `_from_yaml_new`) khỏi `test_per_slot_features.py` — GIỮ 2 test
    per-slot feature còn sống; bỏ 2 test `loader.symbols` (`test_loader_supports_flat_symbol_layout` +
    `_prefers_all_symbols`) + helper `_write_symbol_csv` khỏi `test_loader_layouts.py` — GIỮ
    `test_vn_derivatives` (dùng `load_symbol`, skip nếu thiếu manifest).
  - ⏳ **Còn ignore có chủ đích (2):** `test_portfolio_engine.py` (13) xóa **cùng engine #2 ở Phase 6**;
    `test_run_experiment_e2e.py` (1) = integration (cần data/bundle committed).
  - **CI:** ignore **7 → 2**. **Verify:** test-unit 259 → 296 (gỡ 3 fix) → **298 pass**, 12 skip (thêm 2
    per_slot/loader sau xóa test chết); ruff+format sạch mọi file đụng.
- **Tiếp theo Phase 2:** Level-2 §11.5.5 (`engine_config_from_dict` khi R3/Phase 3) hoặc sang **Phase 3**
  (bản-xuất-tự-đủ + ghim engine). Phase 2 "cái cân" coi như ĐỦ dùng: config-compat (2c) + serving-golden (2b)
  + CI test-thật gần-sạch (2a/2d).

### Phase 3 — Bản xuất tự đủ + ghim engine — 🔄 (bắt đầu 2026-07-31)

- ✅ **R3. Deserializer CHUNG `engine_config_from_dict` (§11 R3 / §662) — 0-ĐỔI-SỐ.** Tách pop-list 58 dòng
  (experiment.py) thành `engine_config_from_dict(engine_cfg) -> (EngineConfig, portfolio_cfg)` trong
  `engine.py`, với **`_RECOMBINE_KEYS`** = 1 set chuẩn 35 khóa (thay 3 danh-sách-tay "khớp do may"). Phân hoạch
  BA chiều: cost→CostModel · recombine keys→strip · còn lại→EngineConfig; khóa lạ **raise TypeError cả hai phía**
  (§10.3). Export từ facade `core` (kèm `EngineConfig/CostModel/run_backtest/trades_to_dataframe` cho tầng trade
  serving — R9). experiment.py gọi hàm chung. **Verify:** partition test `test_engine_config_partition.py`
  (227 field ∩ 35 key = ∅; khóa lạ raise; input không mutate; cost flat+nested) 5 pass; **0-đổi-số end-to-end**
  chạy dyn61 qua run_experiment ra **2443 trades + config_hash y hệt** M1 (trước R3); test-unit **298→303 pass**.
- ⚠️ **Sự cố + khắc phục (ghi để không lặp): `ruff format` churn cả file.** Chạy `ruff format` trên
  engine.py/experiment.py (HEAD vốn KHÔNG ruff-clean) reformat **1240+841 dòng** — nuốt cả công việc
  dynamic-universe **CHƯA COMMIT** (universe_policy/universe_by_year, pre-session). Khôi phục: restore HEAD, tái
  dựng đúng **9 hunk logic** (dynamic-universe + Phase1-VNINDEX + R3) từ backup đã chuẩn-hóa-format (so diff
  CR-normalized, **0 khác biệt logic**). Diff cuối tối thiểu (engine 73, experiment 143 dòng), old-style, không
  churn. **Bài học: KHÔNG chạy `ruff format` trên file chưa-ruff-clean có thay đổi chưa commit; chỉ `ruff check --fix`
  vùng mình sửa, hoặc format thủ công đoạn thêm.**
- ✅ **§11.5.1 `resolved.json` tự-đủ — G1 CHỈ-GHI (R1).** Module mới `src/serving/resolved.py`
  `build_resolved_config(cfg, engine, universe/slug/as_of)` capture đúng cái `config.json` KHÔNG đủ để
  tái sinh tín hiệu: **227 nút EngineConfig sau default** (162-188 nút thừa-hưởng-wheel), **z-window
  recombine resolved** (z_norm_window 252 / z_norm_min_periods 60 từ literal, hoặc override), **catalog
  fingerprint** (`engine_code_fingerprint` — ngữ nghĩa feature mà tên cột không bắt), **wheel version THẬT**
  (`importlib.metadata` = 0.4.2, không phải `core.__version__` đóng băng 0.3.0), **khai-báo data** (universe
  slug + as_of + n_symbols; VNINDEX/runscore/breadth = DECLARED, KHÔNG nhúng bytes — §13.3 Q1). `write_bundle`
  thêm param `resolved_config` → ghi `resolved.json` + **checksum** (R4); `export_bundle.py` dựng qua **cùng
  deserializer R3** (0-đổi-số). **Load KHÔNG kiểm gì** (G1; refuse là việc §11.5.2, so thước không so config).
  **Verify:** export template 112 ra resolved.json (227 engine + z-window + fp `123dbcf597a8` + wheel 0.4.2 +
  data-declared), có trong manifest checksums, `load_bundle` OK; test `test_resolved_config.py` 3 pass;
  test-unit 303→**306 pass**. Còn treo (incremental, không chặn): full-35 recombine-default resolution,
  `PortfolioConstants` overlay, hợp đồng seeding `prediction_history` (items #5/#7 của R1).
- ✅ **§11.5.2 ghim-wheel-chính-xác + từ-chối-khi-khác (cái "refuse" DUY NHẤT còn sống) + phần R2.**
  `bundle.py`: (A) **gate pickle** mở từ `lightgbm`-only → `_PICKLE_CRITICAL_LIBS = lightgbm/scikit-learn/numpy`
  (major-skew raise); (B) **gate đo-thước** `ENGINE_WHEEL_PIN` (opt-in, mirror `MARKET_CONTEXT_REQUIRED`):
  từ chối khi `stock_ml_core` bundle-sinh-số ≠ runtime (EXACT, không chỉ major) — bundle KHÔNG ghi wheel cũng
  từ chối (không attest được → re-export). Thoát = rollback wheel hoặc re-baseline §11.5.4. Thêm `stock_ml_core`
  vào `_TRACKED_LIBS` (nay vào `manifest.lib_versions` mọi export mới). **Mặc định TẮT** ⇒ dev/backtest/test
  nâng wheel tự do. **Verify:** `test_bundle_gates.py` 5 pass (sklearn major-skew raise; strict_libs=False bỏ qua;
  pin OFF mặc định; pin ON mismatch raise; pin ON thiếu-wheel raise); test-unit 306→**311 pass**; diff bundle.py
  72 dòng localized. **R4** = checksums đã phủ mọi payload gồm `resolved.json`; **R5** drift = `catalog_fingerprint`
  trong resolved.json (guard so-lúc-load là G2, hoãn cùng resolved.json G2).
- **Tiếp theo Phase 3:** còn R8 (wheel tái-lập + tag — hạ tầng packaging, cần cho ghim thành deploy thật) +
  R5-G2 (drift-warning lúc load). Phase 3 lõi (bản-xuất-tự-đủ + ghim) coi như ĐỦ để sang **Phase 4**
  (re-baseline + attestation §11.5.3-full + deploy) khi chủ dự án muốn.

### Phase 4 — Re-baseline + deploy — 🔄 (bắt đầu 2026-07-31, chủ dự án duyệt "chạy thật")

> ⚠️ **CẢNH BÁO GHI-LOG (chủ dự án đã duyệt biết trước):** §13.9 (thước universe sai) CHƯA fix và đang chặn
> cross-repo (§13.3.1 fetch ADTV-matched + 321 mã thiếu dyn900 do repo serving sở hữu). ⇒ **re-baseline lần
> này chạy trên thước universe CŨ, số sẽ phải làm lại khi §13.9/§13.3.1 land.** HẠ TẦNG (re-baseline CLI +
> parity + deploy wheel-gate) là BỀN, không throwaway; chỉ SỐ của lần chạy là tạm.

> Recon môi trường 2026-07-31: **LIVE thật** — `stock-serving-worker` up 40h, `stock-ml-postgres` healthy,
> 6 bundle đang phục vụ khách (`bundle_n2_2643_wavestruct…`, `…consw20…`, `…velov…`, `_dyn300/61a2hy/900_onerun`).
> Deploy = restart container chạm khách ⇒ mọi cổng không-khôi-phục (prune · deploy · bật gate) dừng xin go/no-go.

- ✅ **§11.5.4 — re-baseline MỘT LỆNH có log (công cụ, in-repo, 0 chạm serving).**
  `stock_ml/scripts/ops/rebaseline.py`: cold-train tầng ML của danh mục chính thức (§14.1) = **6 model riêng**
  (10 chiến lược → 6 model; 5 biến thể dyn300 chung 1 model, replay overlay rẻ, KHÔNG train ở đây). Map
  model→template **resolve theo TÊN lúc chạy** (fail-loud nếu tên biến mất — đúng lớp drift §13.9 cảnh báo),
  KHÔNG hardcode id. Bắt before→after mỗi đường cong (composite/pnl/pf/mdd/trades) + ghi **1 dòng audit** vào
  `data/rebaseline_log.jsonl` (wheel + catalog_fingerprint + who/when + delta từng model). Có `--dry-run`
  (đọc DB, train 0 gì), `--seeds` (multi-seed sổ deploy), `--models` (subset). **Verify:** ruff sạch; `--dry-run`
  resolve đúng cả 6 (dyn300=3524 · dyn61a2hy=3531 · dyn900=3538 · wavestruct=2646 · consw20=2429 · velov=2234),
  đọc đúng before-values từ leaderboard. **CHƯA chạy thật** (số throwaway §13.9 + bẩn board → chờ go/no-go
  cùng backup pre-flight).
- ✅ **parity.json emit + deploy.py wheel-gate (R7).** `export_bundle.py._write_parity`: mọi bundle mới ship
  `parity.json` {status, wheel, catalog_fingerprint, bundle_fingerprint, as_of} — **status=PENDING** lúc xuất
  (chỉ attestation §11.5.3 hai-chiều mới flip PASS; export KHÔNG tự chứng). `serving/deploy.py.gate_parity`:
  thêm check `parity.wheel == wheel đang deploy` (từ chối attestation-cũ-dưới-wheel-khác — R7/§11.5.2), + báo
  wheel khi PASS. Sửa cổng "hình thức" (đo 31/07: 0/6 bundle có parity ⇒ mọi deploy đi `--force`). compile OK.
- 🔄 **Re-baseline THẬT single-seed (§13.9 đã fix ⇒ đúng-1-lần, KHÔNG throwaway).** 2 phát hiện + fix trong lúc chạy:
  - **Fetch tolerant:** 30/417 mã dyn900 (CI5/H11/HAT…) trả HTTP 500 ở `/ohlcv/` — có trong `market_flow`
    (qua cổng universe) nhưng KHÔNG có series giá ⇒ **untradeable**. `ensure_symbols_cached` nay per-symbol:
    drop-cảnh-báo mã no-OHLCV (không thể backtest mã không bar), **chỉ abort khi nguồn sập** (all-fail). +2 test.
  - **Panel-fixed-first (§13.3 full_market):** breadth/xsec đọc **toàn** `market.duckdb`; để re-baseline
    deterministic (không phụ thuộc thứ-tự-fetch), **populate TRỌN 463 mã survivorship TRƯỚC** (387 fetch OK +
    30 untradeable drop; cache 901→**1334**) rồi re-baseline cả 6 trên panel cố định. **Đo:** delta full-vs-partial
    panel RẤT NHỎ (dyn300 388.6→388.9, velov 495.6→495.9) ⇒ hiệu ứng panel-size **không đáng kể**; **velov
    Δ−160 lúc đầu là STALE-BEFORE** (dòng board cũ velov 656 từ run xưa), KHÔNG phải panel. Full-panel-first vẫn
    đúng (cần cho mã dyn900 + sạch).
  - **2 bug fix trong lúc chạy:** (a) chunk persist run_signals = **7 cột** không phải 6 → hạ 5000→**4000** (dưới
    cap asyncpg 32767); (b) khi mã missing DUY NHẤT còn lại là 7 untradeable (all HTTP-500), tolerant-code cũ
    tưởng "nguồn sập" → abort dyn900; sửa: phân biệt HTTP-error (drop) vs connection-error (abort). +test.
  - **✅ RE-BASELINE CLEAN XONG (echo-off, 14′, panel 1334, deterministic).** 5 model Δ≈0 (dyn300 388.9 ·
    dyn61a2hy 411 · wavestruct 717.5 · consw20 703.6 · velov 495.9 — velov 656→496 cũ = STALE-BEFORE). **dyn900
    (§13.9): pf 1.88→2.19 · maxDD −44.9%→−37.5% · −3712 lệnh rác · composite 217→193** (composite giảm = bớt
    inflation penny giả, ĐÚNG luận điểm). run_signals dyn900 = **1.22M ghi sạch** (0 lỗi persist). Tối ưu tốc độ:
    tắt echo (`.env DEBUG=True` → chạy `DEBUG=false`) + chunk 4000; single-thread là ràng buộc determinism §14.3.
- ✅ **Backup pre-flight** (`stock-serving/backups/preP4_*`): tar 6 bundle (44M) + portfolio.db + serving
  duckdb + .env. Prune vẫn OFF (Phase 0a). Postgres-backup = gate Phase 5 (re-export/deploy không migrate schema).
- ✅ **Validate export path (dyn900 staging).** `export_bundle.py` re-export ra universe §13.9 đúng: **serving
  2026 = 900 mã** (cũ 775; nay chạm trần vì bỏ mã chết + thêm survivorship), fold 2020=801→2022+=900; 7 mã
  untradeable drop; bundle mang **parity.json (PENDING)** + resolved.json. Cả single-fit lẫn replicate-last-fold OK.
- ✅ **Attestation runner §11.5.3** (`scripts/ops/attest_bundle.py`): chạy bundle qua production-path
  (`generate_signals_from_bundle`) so `signal` rời-rạc vs backtest `run_signals` (Postgres) trên cửa sổ serve;
  100% khớp → flip parity PASS, lệch → giữ PENDING + in diff. **Đo (dyn900 replicate-last-fold vs re-baseline
  run):** 2025+ = 92.3% khớp · 2026+ (phần tươi) = **95.2%** khớp. Runner ĐÚNG (đo chính xác).
- ✅ **§11.9 fold-model shipping — MACHINERY BUILD XONG (hướng sạch, chủ dự án chốt).**
  - `bundle.py`: format `fold_models/<year>/<head>.joblib` + manifest `fold_years`; write/load backward-compat
    (bundle cũ `fold_years` rỗng → `fold_models=None`); test roundtrip.
  - `export_bundle.py --fold-models`: train MỌI fold (2020-2025), ship mỗi model; serve-model = fold cuối.
  - `inference.py`: bỏ seed; mỗi bar chấm bởi fold-model của năm nó (clip vào [min,max] fold), **mask mỗi năm
    về `universe_by_year[Y]`** (khớp per-fold-mask của backtest → z-window per-symbol đúng); feature trên union.
  - `attest_bundle.py`: feed **union** `universe_by_year` (feature-scope parity).
- ⚠️ **Attest dyn900 fold-model = 94.8% (chưa 100%). ROOT-CAUSE ĐO ĐƯỢC:** so raw-score prod vs backtest AAA
  2025-08/09 lệch **~0.01-0.03** (median 0.0109) — **KHÔNG phải float-noise** (float ~1e-8), mà là **model
  fold-2025 do `--fold-models` RE-TRAIN ≠ model fold-2025 backtest đã sinh run_signals** (subtle path/non-determinism
  giữa hai đường train). Lệch nhỏ này flip tín hiệu ở **mép band** z>−1.9 → ~5%.
- ➡️ **HƯỚNG SẠCH ĐÚNG (bounded, phiên sau):** thay vì export re-train, **backtest PERSIST đúng fold-model** (sửa
  fold loop `experiment.py:3341` dump joblib mỗi fold vào `{run_id}/folds/{label}.models.joblib`) → re-baseline sinh
  fold-model → `export_bundle --fold-models-from-run <run_id>` ship CHÍNH model đó ⇒ serving == backtest by-construction
  ⇒ attest 100%. (§11.9 tự khai "fold-model đã tồn tại" — ý là DÙNG model backtest, không re-train.)
- ⏳ **Còn (GẤT irreversible, sau khi attest 100%):** re-export 6 bundle từ fold-model-persisted + attest PASS ·
  deploy serving (activate+restart, chạm 6/6 sổ khách) · bật gate sklearn/numpy SAU re-export.
- ⏳ **Còn:** (a) so board cũ/mới (dyn900 kỳ vọng đổi HẲN) · (b) re-export 5 bundle (parity PENDING) · (c) [GẤT]
  attestation §11.5.3 flip PENDING→PASS · (d) [GẤT] deploy serving · (e) [GẤT] bật gate sklearn/numpy **SAU**
  re-export (bật trước = 6/6 sổ ngừng). Backup bundle trước re-export; Postgres-backup là gate Phase 5 (§0b).

#### 🟢 GIẢI XONG §11.9 + phát hiện lỗi re-baseline (2026-07-31, chiều/tối)
- ✅ **Hướng sạch §11.9 CODE + VERIFY.** `experiment.py` fold loop dump `{run_id}/folds/{label}.models.joblib`
  (gate env `STOCKML_PERSIST_FOLD_MODELS`); `export_bundle --fold-models-from-run <folds_dir>` ship CHÍNH model
  backtest, KHÔNG re-train. dyn900: attest **94.8% → 99.877%** (351/285167 còn lệch). Score nay **byte-identical**
  (entry |Δ|max 1.3e-8, exit 1.4e-6 = float-noise) — xác nhận root-cause 94.8% cũ đúng là export-retrain≠backtest.
- ✅ **351 còn lại = FLOAT-BOUNDARY, code ĐÚNG (chứng minh dứt điểm).** 348/351 = SELL-flip ở ngưỡng cứng
  z(exit)>2.0. Test causality: serving full vs cắt đuôi 2026-05-31, overlap 2,608,219 bar,
  **changed_by_future_bars = 0 (0.0000%)** ⇒ pipeline THUẦN CAUSAL, không leak / non-causal-gate / membership-seam.
  Chênh chỉ do accumulation-order float giữa prod-path vs backtest-path. Theo §14.3 ⇒ **100% bit-exact chỉ Linux/
  container**; attest deploy PHẢI chạy trong container (Windows dừng ở ~99.9% là bình thường).
- 🔴 **LỖI PHÁT HIỆN: re-baseline §11.5.4 KHÔNG train lại — restore checkpoint fold cũ.** `run_experiment`
  (experiment.py) restore `{run_id}/folds/*.parquet` nếu tồn tại; checkpoint key = hash(config), KHÔNG gồm
  engine-wheel/data-fp ⇒ sau nâng engine vẫn dùng lại số cũ. Bằng chứng mtime: dyn* = 29/07, wavestruct/consw20/
  velov = **17–21 THÁNG 6** (trước §13.9). ⇒ 6 số catalogue công bố trước đó **STALE**. Ép retrain dyn900 →
  composite **193.3 → 299.3** (trades 28105→35415, pf 2.19→2.31, DD 0.375→0.298).
- ✅ **FIX:** thêm env `STOCKML_FRESH_FOLDS` (bỏ nhánh restore, retrain+ghi đè); `rebaseline.py` tự set
  `STOCKML_FRESH_FOLDS`+`STOCKML_PERSIST_FOLD_MODELS` khi chạy thật ⇒ re-baseline = recompute thật + sinh fold-model
  sẵn để export. (Ghi chú: re-baseline cần `STOCK_DATA_DIR` trỏ đúng duckdb repo-root — default `stock_ml/market_data`
  sai; bản restore-cũ "chạy được" chính vì restore nên KHÔNG load data.)
- 🔄 **Đang chạy:** redo sạch cả 6 model (fresh + persist). Sau đó: export 6 `--fold-models-from-run` + attest
  **container** PASS + deploy. Liên quan memory: `rebaseline-restored-stale-checkpoints-invalid`,
  `phase4-11-9-fold-model-attestation-state`.

### Phase 5 — Danh tính + di sản + migrations — 🔄 (bắt đầu 2026-08-01, chủ dự án authorize trọn: backup+apply+xóa)

> Kỷ luật: mỗi unit INCREMENTAL + verify riêng. Đảo-được làm trước; identity migration (368M dòng + sổ live)
> để sau cùng, bắt buộc dry-run trên bản-sao-restore + checkpoint trước khi apply live.

- ✅ **§0b. Backup Postgres (safety-net cho MỌI migration).** `pg_dump -Fc` DB 99GB →
  `/f/pg_backups/stockml_preP5_20260801.dump` **6.3GB**, verified (26 bảng, có run_signals + leaderboard_runs).
  Restore-copy để dry-run: `docker exec -i stock-ml-postgres pg_restore -d <copydb> < dump`.
- ✅ **§4.1 Migration 0030 — codify `leaderboard_nav` 18-cột.** Live drift 18 vs 11 DDL (cagr_t2/maxdd_t2 +
  overlay_* thêm ad-hoc, KHÔNG migration nào bắt) ⇒ fresh-DB rebuild từ script hụt 7 cột. `0030_leaderboard_nav_shape.py`:
  CREATE/ADD COLUMN **IF NOT EXISTS** (additive, no-op live). Validate fresh-DB (chuỗi 0001→0030 = 18 cột) rồi
  apply live (head 0029→**0030**). downgrade = no-op có chủ đích (không xoá đường cong NAV đã publish).
- ✅ **§10.2 Xóa engine thực thi #2 chết.** `src/backtest/portfolio_engine.py` (486 dòng) + test (393) — verified
  chết (chỉ 2 self-ref, 0 live importer, `__init__` không export). Suite **319 passed / 0 failed** (hết 13 fail
  biết-trước). `test_pnl_calculator.py` cùng dir độc lập → GIỮ.
- ✅ **§4.5/§4.6 (phần rời-rạc, rủi-ro-thấp) — dọn danh tính module.** Map thực tế: mọi importer LIVE (api, export,
  run_template, cache_gc, evaluation, experiments) + repo serving đều dùng bản **`utils/*`**; root
  `stock_ml/src/{config_loader,env}.py` = **fork CHẾT** (chỉ tự-tham-chiếu + 1 file analysis scratch; serving 0 import;
  `utils/__init__` import bản utils). ⇒ **xoá 2 fork root** + repoint scratch (`src.env`→`src.utils.env`). **§4.6:** sửa
  `parents[1]→parents[2]` cho 4 script ops (cache_gc/api_server/build_leaderboard/export_derivatives_ohlcv) — lazy
  import `from src.*` trước đó **raise runtime** (ROOT=`stock_ml/scripts` thiếu `stock_ml` trên path). **Verify:**
  suite 319/0, smoke import serving-critical (pipeline.experiment/backtest.engine/data.splitter/utils.*) sạch, 3 lazy
  import ops resolve với `stock_ml` on path.
- ✅ **§4.5 Bug env arithmetic — FIX (audit xong, an toàn).** Audit: `STOCK_RESULTS_DIR` **KHÔNG set** ở prod
  (chỉ 1 test monkeypatch) ⇒ bug LIVE không bị che; `stock_ml/results` (8.6GB: experiments/research/alpha_gate/
  live_sim…) = cây THẬT, `stock_ml/src/results` (93MB: chỉ `cache/`) = tình cờ do chính bug đẩy vào; serving KHÔNG
  import `utils.env` ⇒ an toàn. `utils/env.py`: thêm `_stock_ml_dir()` (dirname×**3** = stock_ml, sửa off-by-one di
  sản khi copy `env.py`→`utils/env.py`), thay 2 site. Nay `get_results_dir()→stock_ml/results`,
  `resolve_data_dir("../portable_data/x")→repo_root/portable_data/x`. Suite 319/0. (CÒN: xoá cây orphan
  `stock_ml/src/results/cache` 93MB — gộp vào "một results root" §4.5, là data gitignore nên hoãn cùng convergence.)
- ✅ **§1.6 (phần AN TOÀN) — guard bẫy-bất-đối-xứng + tương tác với env-fix.** Env-fix ở trên (`get_results_dir`
  nay trỏ `stock_ml/results`) đã **sửa NỬA "nguồn referenced" của §1.6**: GC nay quét `stock_ml/results/experiments`
  (TỒN TẠI, có `v22/predictions_meta.json`) thay vì `stock_ml/src/results/experiments` (vắng) ⇒ referenced hết rỗng.
  **An toàn xác nhận:** cả 3 caller `sweep()` đều dry-run mặc định (`cache.py:39` cứng `dry_run=True`; `api_server
  /api/gc/sweep` + `cache_gc --apply` gated explicit) ⇒ env-fix KHÔNG tự quarantine gì. Thêm **guard §1.6** (doc-mandate
  "referenced rỗng thành lỗi cứng"): `sweep()` nay **raise** khi `not dry_run and orphans and referenced rỗng` — chặn
  thảm hoạ "quarantine cả cache" nếu experiments dir mất/di chuyển. +1 test. Suite **320/0**.
- ⏳ **§1.6 (phần NGUY HIỂM — HOÃN, cần thiết kế store-attribution):** glob `find_feature_cache_files` `*/*` tìm cache
  kiểu-cũ `features/<set>/<key>.parquet` (41 file/8.75MB) nhưng MÙ với **FeatureStore** `features/store/<expr_hash>/
  <ver>.parquet` (bulk ~51GB, sâu 1 cấp, key = expr_hash KHÁC hệ `cache_keys.features`). Sửa glob để gồm `store/**`
  = 14.476 file bỗng "orphan" vì referenced (key cũ) không map key store ⇒ **bẫy thật**. Cần: thiết kế attribution
  store→run (hoặc quyết định store do FeatureStore tự quản, GC KHÔNG đụng) + sửa test đang bảo-chứng-layout-cũ. Chưa làm.
- ⏳ **§4.7 FeatureCacheManager — vẫn gated sau §1.6-đầy-đủ.** Đã verify: `feature_cache.py` (229 dòng) = **toàn bộ**
  class `FeatureCacheManager`, 0 code-importer (chỉ re-export `cache/__init__` + 1 docstring `schema.py:31`); serving
  KHÔNG import `stock_ml.src.cache`; `garbage_collector.py` cùng package độc lập. Xoá được về mặt kỹ thuật NHƯNG doc
  gate "sau §1.6" (§1.6 store-fix chưa xong) ⇒ giữ nguyên, xoá cùng lúc dọn `stock_ml/src/results` orphan.
- ✅ **§4.8 dead-col (phần cache_keys) — XOÁ `cache_key_features`/`cache_key_predictions` + Pydantic `CacheKeys`.**
  DB verify: cả 2 cột rỗng `''` trên **3612/3612** (write path DB-first chưa từng set); GC attribute cache từ
  `predictions_meta.json` (nguồn FILE) chứ KHÔNG đọc cột này. Gỡ `CacheKeys` model + field `cache_keys` khỏi
  schema/loader/aggregator/adapter/ORM(run.py)/run_template + 2 test. `api_server._quarantine_run_cache` (script cũ,
  dùng cache_keys THẬT để quarantine cache khi xoá run) refactor **đọc keys từ `predictions_meta.json`** (giống GC)
  thay row đã xoá — feature vẫn chạy, hết phụ thuộc field bỏ. **Migration 0031** drop 2 cột (guarded IF EXISTS,
  downgrade re-add). Bonus: gỡ docstring `CacheKeys` trỏ `FeatureCacheManager` (dead §4.7). Verify: suite **320/0**,
  ruff sạch, apply live head 0030→**0031** (cột biến mất), ORM round-trip live OK. **CÒN §4.8:** `same_*_as_baseline`
  (6, NULL) + `is_baseline` (0 true) + `fairness_group_key` (populated degenerate, load-bearing dedup) = phần fairness,
  gắn §4.4 (xem dưới).
- ✅ **§4.4 fairness-mechanism removal (BACKEND) — commit `ddd52fbf` + migration 0032.** Data toàn chết: `same_*`
  NULL 3612/3612, `is_baseline` false all, `fairness_group_key` populated nhưng degenerate (3 giá trị) + KHÔNG emit
  qua API. Gỡ: **xoá `fairness.py`**, chuyển 3 helper GENERAL (`load_config`/`resolve_market_family`/`backtest_window_key`)
  vào `loader.py` (consumer duy nhất còn lại); drop 8 field khỏi schema/ORM/adapter/aggregator(CSV+annotate per-market/
  family+_summary)/`run_repo.get_by_fairness_group`(0 caller); **rewrite `_row_signature` dedup bỏ `fairness_group_key`**
  (nó là hash của field đã có sẵn trong signature → giữ granularity; chỉ ảnh hưởng CSV rebuild, không phải board DB
  live); xoá `test_fairness.py` (coverage resolve_market_family đã có ở test_loader) + gỡ field khỏi fixture/golden.
  **Migration 0032** drop 8 cột + index `idx_runs_fairness_group`. Verify: suite **316/0**, ruff sạch, apply live head
  0031→**0032**, ORM round-trip live OK.
- ⏳ **§4.4 fair-mode UI (JS/HTML) — HOÃN (cosmetic, đã-hỏng-sẵn, untested/outward-facing).** `dashboard/leaderboard.js`
  + `visualization/leaderboard.js` còn nhánh fair-mode (toggle global/fair, `getFairBaselineGroup`, cột `renderFairness`,
  warnings `same_*===false`, `setScoreMode`) + nút trong 2 HTML. ĐÃ hỏng từ trước (API chưa từng emit field → đọc
  undefined, fail-closed) ⇒ backend removal KHÔNG mới-làm-hỏng. Gỡ trọn = refactor UI (js+html, ~9 site/file) không có
  test browser ⇒ tách pass cosmetic riêng, không barrel ở đuôi unit lớn.
- ⏳ **§4.5 (phần couple còn lại — HOÃN, cần verify chạy thật):**
  - **Convergence trọn `src.*`→`stock_ml.src.*`** (bỏ double-cache class/module): đòi MỌI entry có repo_root trên
    path (đổi run_template sys.path + convert model_dashboard/data/pipeline…) — thay đổi phối hợp, verify bằng chạy
    pipeline thật, không chỉ pytest. **fix-vs-xoá 3 script** cache_gc/api_server/build_leaderboard (có bị `stock_ml/api`
    thay chưa?) = quyết định kiến trúc, hiện chọn "fix" (bảo thủ, đảo được), delete để sau.
- ⏳ **CÒN (liên-kết/nguy hiểm):** drop 8 cột chết `leaderboard_runs` = cơ chế **fairness cũ** §4.4 (ripple ORM/adapter/
  schema/API) · **⚠️ §3 config_hash re-identity** (368M dòng run_signals + 5043 fold dir + sổ tier live — D2 + dry-run
  bản-sao + checkpoint) · §4.7 FeatureCacheManager (partial-file). Liên quan memory: `phase5-progress-and-identity-danger`.
