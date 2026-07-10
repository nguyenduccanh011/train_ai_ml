# Tài liệu triển khai production Train61

## 1. Mục tiêu

Triển khai hệ thống Train61 lên server theo hướng ổn định, có thể kiểm toán lại tín hiệu, không phụ thuộc file cache tạm và không chạy tính toán nặng trực tiếp trong request web.

Mục tiêu chính:

- Kết nối API dữ liệu đầy đủ từ `https://sieutinhieu.vn/api/v1`.
- Lưu dữ liệu thị trường vào database làm nguồn sự thật.
- Tính signal hằng ngày từ model production cố định.
- Giả lập gửi từng nến để kiểm tra replay/live có khớp batch backtest không.
- Web chạy mượt: API chỉ đọc kết quả đã tính sẵn, worker xử lý tác vụ nặng.
- Có versioning rõ cho dữ liệu, model, signal và khả năng rollback.

## 2. Hiện trạng dự án

Runtime hiện tại đang hoạt động theo mô hình file-first:

- `src/data/loader.py` đọc dữ liệu từ CSV layout `symbol=XXX/timeframe=1D/data.csv`.
- `app/serve_train61_model.py` phục vụ UI/API bằng Flask.
- OHLCV cache nằm trong `data/ohlcv/*.json`.
- Signal cache nằm trong `cache/signals/.../*.json`.
- Feature cache nằm trong `cache/features`.
- Model artifact nằm trong `models/*.pkl`.

Các điểm rủi ro hiện tại:

- `_load_ohlcv()` ưu tiên JSON cache nếu file tồn tại, có thể trả nến cũ.
- `/api/signal` ưu tiên memory/disk cache, chưa kiểm cache có khớp dữ liệu mới không.
- Một số model dạng `pooled_global_rerun` có thể chạy pipeline nặng khi user request.
- Dữ liệu local có thể chậm hơn API. Ví dụ kiểm tra ngày 2026-05-15: API đã có ACB tới 2026-05-14, local CSV mới tới 2026-05-08.

## 3. Nguyên tắc kiến trúc

Production cần chuyển sang mô hình:

```text
SieuTinHieu API
  -> ingestion worker
  -> normalize + validate
  -> PostgreSQL/TimescaleDB
  -> data_version
  -> feature/predict/replay worker
  -> predictions/signals/trades tables
  -> web API chỉ đọc kết quả
```

Nguyên tắc bắt buộc:

- Không gọi API dữ liệu trực tiếp trong model inference request.
- Không train/retrain model trong web server.
- Không chạy `Pipeline(...).run()` toàn universe trong request web.
- Không dùng JSON cache làm nguồn dữ liệu chính.
- Mọi signal phải truy ngược được `model_version + data_version + config_version`.
- Model production không retrain hằng ngày. Hằng ngày chỉ ingest dữ liệu mới và chạy inference.

## 4. Docker architecture

Nên triển khai bằng Docker Compose trong giai đoạn đầu. Không nên gom tất cả vào một container.

Giai đoạn hiện tại là **localhost Docker trước**, chưa public server. Mục tiêu là dựng cùng cấu trúc hạ tầng production ngay trên máy local:

```text
localhost
  -> docker compose
      -> db/redis chạy mặc định
      -> api chạy bằng profile riêng
      -> worker/scheduler thêm ở phase sau
```

Repo hiện đã có:

```text
Dockerfile
docker-compose.local.yml
.env.example
.dockerignore
LOCAL_DOCKER_DEPLOYMENT.md
db/init/001_train61_schema.sql
```

Trạng thái local hiện tại:

- `db` chạy bằng `timescale/timescaledb:latest-pg16`, bind `localhost:15432` trên máy local này để tránh xung đột PostgreSQL host đang chiếm `5432`.
- `redis` chạy bằng `redis:7-alpine`, bind `localhost:6379`.
- Schema nền tảng đã có 11 bảng: `instruments`, `market_bars`, `data_versions`, `ingestion_runs`, `data_quality_checks`, `model_artifacts`, `predictions`, `signals`, `trades`, `replay_runs`, `replay_steps`.
- `api` được đặt trong Docker Compose profile `api`, chưa chạy mặc định để tránh bị chặn bởi build image ML còn nặng.
- Flask server đã hỗ trợ `TRAIN61_HOST` và `TRAIN61_PORT`; trong container dùng `TRAIN61_HOST=0.0.0.0`, trên máy host truy cập `http://127.0.0.1:5012`.

Service đề xuất:

```text
db          PostgreSQL hoặc TimescaleDB
redis       queue, job status, hot cache
api         Flask/FastAPI phục vụ UI/API
worker      ingest, feature, predict, replay, reconcile
scheduler   chạy lịch ingest/reconcile
nginx       reverse proxy khi deploy public
```

Sơ đồ:

```text
                user/browser
                    |
                  nginx
                    |
                  api
               /        \
            redis        db
              |          |
          worker <--- scheduler
```

Gợi ý `docker-compose.yml`:

```yaml
services:
  db:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_DB: train61
      POSTGRES_USER: train61
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  api:
    build: .
    command: python app/serve_train61_model.py
    env_file: .env
    depends_on:
      - db
      - redis
    volumes:
      - ./models:/app/models
      - ./cache:/app/cache
      - ./config:/app/config

  worker:
    build: .
    command: python tools/worker.py
    env_file: .env
    depends_on:
      - db
      - redis
    volumes:
      - ./models:/app/models
      - ./cache:/app/cache
      - ./config:/app/config

  scheduler:
    build: .
    command: python tools/scheduler.py
    env_file: .env
    depends_on:
      - worker

volumes:
  pgdata:
```

Biến môi trường cần chuẩn hóa:

```env
DATABASE_URL=postgresql://train61:password@db:5432/train61
REDIS_URL=redis://redis:6379/0
SIEUTINHIEU_API_BASE=https://sieutinhieu.vn/api/v1
MODEL_DIR=/app/models
CACHE_DIR=/app/cache
ACTIVE_MODEL_ID=train61_pooled
DEFAULT_TIMEFRAME=1D
```

### 4.1. Local Docker commands

Chạy hạ tầng local:

```powershell
docker compose -f docker-compose.local.yml up -d
```

Kiểm tra:

```powershell
docker compose -f docker-compose.local.yml ps
docker compose -f docker-compose.local.yml exec -T db pg_isready -U train61 -d train61
docker compose -f docker-compose.local.yml exec -T redis redis-cli ping
docker compose -f docker-compose.local.yml exec -T db psql -U train61 -d train61 -c "\dt"
```

Chạy thêm API container khi đã sẵn sàng build image:

```powershell
docker compose -f docker-compose.local.yml --profile api up -d --build
```

Apply schema thủ công nếu volume DB đã tồn tại trước khi thêm file init:

```powershell
Get-Content db\init\001_train61_schema.sql | docker compose -f docker-compose.local.yml exec -T db psql -U train61 -d train61
```

Lưu ý: hiện build API image có thể lâu vì dependency ML (`lightgbm`, `xgboost`, `catboost`). Trước khi đưa `api` vào profile mặc định, nên tách `requirements-runtime.txt` hoặc tạo image runtime tối ưu để tránh build quá lâu.

## 5. Database schema đề xuất

### 5.1. Instruments

Lưu danh mục mã.

```sql
create table instruments (
    id bigserial primary key,
    provider_symbol_id bigint,
    symbol text not null unique,
    name text,
    exchange text,
    asset_type text,
    data_provider text,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
```

### 5.2. Market bars

Lưu OHLCV chuẩn hóa.

```sql
create table market_bars (
    id bigserial primary key,
    symbol text not null,
    symbol_id bigint,
    timeframe text not null,
    timestamp timestamptz not null,
    open numeric(20, 6) not null,
    high numeric(20, 6) not null,
    low numeric(20, 6) not null,
    close numeric(20, 6) not null,
    volume bigint not null default 0,
    traded_value numeric(30, 6),
    provider text not null,
    provider_bar_id bigint,
    provider_created_at timestamptz,
    ingested_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique(symbol, timeframe, timestamp, provider)
);

create index idx_market_bars_symbol_tf_time
    on market_bars(symbol, timeframe, timestamp desc);
```

Nếu dùng TimescaleDB:

```sql
select create_hypertable('market_bars', 'timestamp', if_not_exists => true);
```

### 5.3. Data versions

Lưu fingerprint hiện tại của từng symbol/timeframe.

```sql
create table data_versions (
    symbol text not null,
    timeframe text not null,
    provider text not null,
    latest_timestamp timestamptz,
    latest_close numeric(20, 6),
    row_count bigint not null default 0,
    version_hash text not null,
    updated_at timestamptz not null default now(),
    primary key(symbol, timeframe, provider)
);
```

Gợi ý `version_hash`:

```text
sha1(symbol|timeframe|provider|latest_timestamp|latest_close|row_count)
```

### 5.4. Ingestion runs

Theo dõi mỗi lần kéo dữ liệu.

```sql
create table ingestion_runs (
    id bigserial primary key,
    provider text not null,
    timeframe text not null,
    mode text not null,
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    status text not null default 'running',
    symbol_count int not null default 0,
    inserted_count int not null default 0,
    updated_count int not null default 0,
    error_count int not null default 0,
    error text
);
```

### 5.5. Model artifacts

Quản lý model production/candidate.

```sql
create table model_artifacts (
    model_id text primary key,
    model_version text not null,
    status text not null,
    artifact_path text not null,
    config_path text,
    config_hash text,
    feature_set text,
    trained_until timestamptz,
    created_at timestamptz not null default now(),
    promoted_at timestamptz
);
```

`status` nên gồm:

```text
candidate
active
retired
failed
```

### 5.6. Predictions, signals, trades

```sql
create table predictions (
    id bigserial primary key,
    model_id text not null,
    model_version text not null,
    symbol text not null,
    timeframe text not null,
    bar_time timestamptz not null,
    data_version_hash text not null,
    y_pred int,
    y_pred_exit int,
    buy_proba numeric(10, 6),
    payload jsonb,
    created_at timestamptz not null default now(),
    unique(model_id, model_version, symbol, timeframe, bar_time, data_version_hash)
);

create table signals (
    id bigserial primary key,
    model_id text not null,
    model_version text not null,
    symbol text not null,
    timeframe text not null,
    bar_time timestamptz not null,
    data_version_hash text not null,
    action text not null,
    reason text,
    confidence numeric(10, 6),
    payload jsonb,
    created_at timestamptz not null default now(),
    unique(model_id, model_version, symbol, timeframe, bar_time, data_version_hash)
);

create table trades (
    id bigserial primary key,
    model_id text not null,
    model_version text not null,
    symbol text not null,
    timeframe text not null,
    entry_date timestamptz,
    exit_date timestamptz,
    entry_price numeric(20, 6),
    exit_price numeric(20, 6),
    pnl_pct numeric(12, 6),
    holding_days int,
    entry_reason text,
    exit_reason text,
    is_open boolean not null default false,
    data_version_hash text,
    payload jsonb,
    created_at timestamptz not null default now()
);
```

### 5.7. Replay

```sql
create table replay_runs (
    id bigserial primary key,
    model_id text not null,
    model_version text not null,
    timeframe text not null,
    start_time timestamptz not null,
    end_time timestamptz not null,
    status text not null default 'running',
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    summary jsonb
);

create table replay_steps (
    id bigserial primary key,
    replay_run_id bigint not null references replay_runs(id),
    symbol text not null,
    bar_time timestamptz not null,
    data_version_hash text,
    y_pred int,
    y_pred_exit int,
    action text,
    position_state jsonb,
    signal_payload jsonb,
    created_at timestamptz not null default now(),
    unique(replay_run_id, symbol, bar_time)
);
```

## 6. API provider integration

Tạo provider riêng cho API Siêu Tín Hiệu.

Module:

```text
src/data/providers/base.py
src/data/providers/sieutinhieu.py
```

Interface:

```python
class MarketDataProvider:
    def list_symbols(self, limit: int = 1000, offset: int = 0) -> dict:
        ...

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1D",
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> dict:
        ...

    def fetch_latest(
        self,
        symbol: str,
        timeframe: str = "1D",
        limit: int = 10,
    ) -> list[dict]:
        ...
```

Endpoint cần dùng:

```text
GET /symbols/?limit=...&offset=...
GET /symbols/search?query=...
GET /ohlcv/?symbol=...&timeframe=...&start_date=...&end_date=...&limit=...&offset=...
GET /ohlcv/latest?symbol=...&timeframe=...&limit=...
GET /ohlcv/symbols/{symbol}/count?timeframe=...
```

Normalize OHLCV:

```text
symbol       = symbol request
symbol_id    = response.symbol_id
timeframe    = response.timeframe
timestamp    = UTC timestamp
open         = Decimal(response.open)
high         = Decimal(response.high)
low          = Decimal(response.low)
close        = Decimal(response.close)
volume       = int(response.volume)
provider     = "sieutinhieu"
provider_id  = response.id
created_at   = response.created_at
```

Lưu ý thực tế:

- API latest trả giá dạng string decimal.
- OHLCV item không luôn có `symbol`, adapter phải tự gắn symbol từ request.
- Pagination cần dùng `limit <= 1000`.
- Daily update nên lấy `latest limit=10`, không chỉ lấy 1 nến, để bắt dữ liệu provider sửa lại vài phiên gần nhất.

## 7. Ingestion strategy

### 7.1. Initial backfill

Mục tiêu: kéo lịch sử đầy đủ cho universe ban đầu.

Luồng:

```text
1. Sync symbols từ /symbols/
2. Lọc asset_type=stock, exchange phù hợp
3. Với từng symbol:
   - gọi /ohlcv/ theo pagination
   - normalize
   - validate
   - upsert market_bars
4. Recompute data_versions
5. Export CSV fallback nếu cần giữ pipeline cũ
```

Command dự kiến:

```powershell
python tools/ingest_sieutinhieu.py --mode backfill --timeframe 1D --symbols config/train61_symbols.json
```

### 7.2. Daily update

Chạy sau khi provider đã có dữ liệu cuối ngày.

Luồng:

```text
1. Với từng symbol active:
   - gọi /ohlcv/latest?limit=10
   - upsert vào market_bars
2. Recompute data_version
3. Nếu data_version thay đổi:
   - enqueue compute signal cho symbol đó
4. Chạy daily reconcile
5. Xuất report
```

Command dự kiến:

```powershell
python tools/ingest_sieutinhieu.py --mode latest --timeframe 1D --latest-limit 10
python tools/reconcile_daily.py --timeframe 1D
```

### 7.3. Intraday

Nếu dùng 15m/30m/1H:

- Lưu riêng theo `timeframe`.
- Không trộn daily và intraday trong cùng feature cache.
- Replay từng nến nên chạy theo timeframe độc lập.
- Lịch scheduler phải chờ nến đóng, không predict trên nến chưa hoàn chỉnh.

## 8. Data quality checks

Mỗi lần ingest cần kiểm:

- Duplicate `(symbol, timeframe, timestamp, provider)`.
- OHLC hợp lệ: `low <= open/close <= high`.
- Volume không âm.
- Timestamp UTC hợp lệ.
- Missing bars so với lịch giao dịch.
- Gap giá bất thường.
- Provider sửa dữ liệu cũ: same timestamp nhưng OHLCV thay đổi.

Bảng log:

```sql
create table data_quality_checks (
    id bigserial primary key,
    ingestion_run_id bigint references ingestion_runs(id),
    symbol text,
    timeframe text,
    timestamp timestamptz,
    check_type text not null,
    severity text not null,
    message text,
    payload jsonb,
    created_at timestamptz not null default now()
);
```

Severity:

```text
info
warning
error
critical
```

## 9. DataLoader migration

Không nên sửa toàn bộ pipeline một lần. Nên chuyển theo từng bước.

### Phase A: DB ingest nhưng pipeline vẫn đọc CSV

- Ingest vào DB.
- Export DB ra layout CSV cũ:

```text
data/vn_stock_ai_dataset_cleaned/all_symbols/symbol=ACB/timeframe=1D/data.csv
```

- Dùng để kiểm tra DB/API có khớp dữ liệu cũ.

### Phase B: DataLoader hỗ trợ DB

Mở rộng `DataLoader`:

```python
DataLoader(
    data_dir=...,
    source="csv" | "db",
    db_url=...
)
```

Các method giữ nguyên:

```python
load_symbol(symbol)
load_all(symbols)
load_all_context()
```

Điều này giúp các module feature/model ít phải đổi.

### Phase C: DB là nguồn chính

- Production đọc DB.
- CSV chỉ còn là export/debug.
- JSON OHLCV cache không còn là source of truth.

## 10. Signal freshness

Mọi OHLCV/signal payload phải gắn:

```json
{
  "model_id": "train61_pooled",
  "model_version": "train61_pooled_v1_20260510",
  "symbol": "ACB",
  "timeframe": "1D",
  "data_version_hash": "...",
  "latest_bar_date": "2026-05-14"
}
```

Khi `/api/signal` nhận request:

```text
1. Lấy current data_version từ DB
2. Lấy signal cache/result hiện có
3. Nếu data_version_hash khớp: trả cache
4. Nếu không khớp:
   - trả 202 missing/stale
   - worker/scheduler tính lại ngoài request web
```

Không được trả signal cũ khi dữ liệu đã đổi.

## 11. Heavy compute strategy

Các phần nặng hiện tại:

- Feature engineering nhiều symbol/lịch sử dài.
- `Pipeline(...).run()` cho pooled/global model.
- `pooled_global_rerun`.
- On-demand signal khi user click symbol.
- Replay/backtest đối chiếu.
- Đọc CSV/JSON nhiều lần bằng pandas.

Giải pháp:

- API web chỉ đọc kết quả đã tính.
- Worker tính trước signal sau ingest.
- `pooled_global_rerun` phải precompute toàn universe sau khi data đổi.
- `/api/symbols` đọc summary/materialized table thay vì loop đọc từng signal file.
- Replay/backtest chạy background, không chạy trong request.
- Redis chỉ dùng cho job status/hot cache; DB vẫn là nguồn chính.

Worker jobs:

```text
ingest_latest
recompute_data_version
compute_symbol_signal
compute_universe_payload
replay_bars
daily_reconcile
export_csv_snapshot
```

## 12. Model versioning policy

Không retrain hằng ngày cho signal production.

Production policy:

```text
active model cố định
+ dữ liệu mới hằng ngày
+ inference hằng ngày
+ retrain candidate định kỳ
+ promote thủ công hoặc bán tự động sau kiểm định
```

Tách rõ:

```text
data_version   dữ liệu mới tới đâu
model_version  model artifact nào đang dùng
signal_version signal sinh từ model_version + data_version
```

Quy trình promote:

```text
1. Train candidate model ở background
2. Chạy walk-forward/backtest/replay
3. So sánh với active model
4. Kiểm tra độ flip signal
5. Nếu tốt hơn rõ ràng, promote
6. Giữ active cũ để rollback
```

Metric so sánh:

- Total return.
- Max drawdown.
- Sharpe/Calmar.
- Win rate.
- Trade count.
- PnL per trade.
- Yearly consistency.
- Số tín hiệu bị flip so với model cũ.
- Số open position thay đổi.

## 13. Replay từng nến

Mục tiêu replay: kiểm tra khi gửi từng nến vào engine, kết quả có khớp batch backtest không.

Thiết kế:

```text
ReplayRunner
  -> đọc market_bars theo timestamp tăng dần
  -> với mỗi bar gọi BarEngine.on_bar()
  -> lưu replay_steps
  -> sau khi xong so sánh với batch trades/signals
```

Interface đề xuất:

```python
class BarEngine:
    def on_bar(self, symbol: str, timeframe: str, bar: dict) -> dict:
        ...
```

Replay command:

```powershell
python tools/replay_bars.py --model-id train61_pooled --timeframe 1D --start 2025-01-01 --end 2026-05-14
```

Kết quả cần lưu:

- y_pred tại từng bar.
- action tại từng bar.
- position state.
- trade state.
- data_version.
- model_version.

Đối chiếu:

```text
batch signal/trades
vs
replay signal/trades từng nến
```

Nếu lệch, report cần chỉ rõ:

- Symbol.
- Bar time.
- Batch action.
- Replay action.
- Feature khác nhau hay prediction khác nhau.
- Nguyên nhân nghi ngờ.

## 14. Daily reconciliation

Chạy sau daily ingest + signal compute.

Checklist:

```text
1. API latest vs DB latest
2. DB latest vs CSV export nếu còn dùng CSV
3. data_version changed symbols
4. signal generated symbols
5. stale signal còn sót
6. open position diff
7. replay sample diff
8. report JSON/HTML
```

Command:

```powershell
python tools/reconcile_daily.py --date 2026-05-14 --timeframe 1D
```

Report output:

```text
reports/daily_reconcile/2026-05-14.json
reports/daily_reconcile/2026-05-14.html
```

## 15. API/web endpoints production

Nên giữ endpoint hiện tại nhưng đổi nguồn đọc từ DB/result cache.

Endpoint hiện tại:

```text
GET /api/models
GET /api/model-info
GET /api/symbols
GET /api/ohlcv/<symbol>
GET /api/signal/<model_id>/<symbol>
GET /api/signal/<model_id>/<symbol>/status
```

Endpoint nên thêm:

```text
GET /api/data-version/<symbol>?timeframe=1D
GET /api/jobs/<job_id>
GET /api/reconcile/latest
GET /api/replay-runs
GET /api/replay-runs/<id>
POST /api/admin/refresh-symbol/<symbol>
POST /api/admin/ingest-latest
```

Admin endpoints cần auth khi public.

## 16. Security

Giai đoạn public server cần:

- Không expose Postgres/Redis ra internet.
- Nginx reverse proxy.
- HTTPS.
- Basic auth hoặc admin token cho admin endpoints.
- `.env` không commit.
- Backup DB hằng ngày.
- Log request và job errors.
- Rate limit nếu endpoint public.

## 17. Monitoring

Metrics cần theo dõi:

```text
ingestion_success_count
ingestion_error_count
latest_bar_lag_days
data_version_changed_count
signal_generated_count
signal_stale_count
worker_queue_depth
worker_job_duration_seconds
api_latency_ms
api_error_count
replay_diff_count
```

Health checks:

```text
/health
  api ok
  db connected
  redis connected
  active model loaded
  latest ingestion status
```

## 18. Backup và rollback

Backup:

- Postgres dump hằng ngày.
- Model artifacts trong `models/`.
- Config resolved YAML.
- Daily reconcile reports.

Rollback model:

```text
1. Set active model_artifacts.status từ active -> retired
2. Set model cũ retired -> active
3. Clear hot cache Redis
4. Không xóa predictions/signals cũ
5. Web đọc active model mới
```

Rollback data:

- Không xóa dữ liệu provider cũ trực tiếp.
- Nếu provider sửa sai dữ liệu, lưu updated row và ghi quality log.
- Nếu cần snapshot theo ngày, tạo table hoặc export `market_bars_snapshot`.

## 19. Lộ trình triển khai

### Phase 1: Freshness cho runtime hiện tại

Mục tiêu: giảm rủi ro trả cache cũ trước khi có DB đầy đủ.

Việc cần làm:

- Thêm helper tính fingerprint source data cho symbol.
- Gắn fingerprint vào OHLCV payload.
- Gắn fingerprint vào signal payload.
- `/api/signal` kiểm fingerprint trước khi trả cache.
- Nếu stale, trả 202 và để worker/scheduler generate lại ngoài request web.

Done khi:

- Append/cập nhật nến mới thì OHLCV và signal không trả cache cũ.

### Phase 2: Provider API + DB schema

Mục tiêu: ingest dữ liệu thật vào DB.

Việc cần làm:

- Tạo `SieuTinHieuProvider`.
- Tạo DB schema.
- Tạo migration SQL.
- Viết `tools/ingest_sieutinhieu.py`.
- Backfill 61 symbols ban đầu.
- Daily update bằng latest limit 10.

Done khi:

- DB có đủ `market_bars`.
- `data_versions` cập nhật đúng.
- ACB trong DB khớp API latest 2026-05-14 hoặc mới hơn khi chạy ngày sau.

### Phase 3: DataLoader đọc DB

Mục tiêu: pipeline có thể dùng DB làm nguồn chính.

Việc cần làm:

- Mở rộng `DataLoader` với `source=db`.
- Giữ interface `load_symbol/load_all`.
- Thêm fallback CSV.
- Test feature/predict với DB data.

Done khi:

- Cùng symbol/date range, output từ DB loader khớp CSV export.

### Phase 4: Worker hóa compute

Mục tiêu: web không chạy tác vụ nặng.

Việc cần làm:

- Tạo worker job queue.
- Job `compute_symbol_signal`.
- Job `compute_universe_payload`.
- Precompute `pooled_global_rerun`.
- API `/api/signal` chỉ đọc DB/result cache hoặc trả 202.

Done khi:

- User click symbol không làm server chạy pipeline dài.

### Phase 5: Replay từng nến

Mục tiêu: kiểm chứng live simulation.

Việc cần làm:

- Tạo `BarEngine`.
- Tạo `ReplayRunner`.
- Lưu `replay_runs/replay_steps`.
- Viết compare batch vs replay.

Done khi:

- Có report chỉ ra signal/trade khớp hoặc lệch ở bar nào.

### Phase 6: Docker production

Mục tiêu: chạy server mượt. Phase này đi theo hướng local-first: trước tiên chạy ổn trên localhost Docker, sau đó mới đưa compose lên server.

Việc cần làm:

- Dockerfile. Đã có bản đầu tiên.
- docker-compose local với db/redis. Đã chạy được.
- DB init schema. Đã có `db/init/001_train61_schema.sql`.
- docker-compose với api/worker/scheduler.
- `.env.example`. Đã có bản đầu tiên.
- Healthcheck.
- Backup script.
- Nginx config.
- Tối ưu API image để build nhanh hơn.

Done khi:

- `docker compose -f docker-compose.local.yml up -d` chạy được hạ tầng local.
- `docker compose -f docker-compose.local.yml --profile api up -d --build` chạy được API.
- Web đọc dữ liệu DB.
- Worker ingest/predict chạy nền.

## 20. Checklist trước khi public

- DB có backup.
- Model active cố định và có version.
- Không retrain hằng ngày trong production.
- Signal có `model_version` và `data_version_hash`.
- `/api/signal` không trả stale cache.
- `/api/symbols` không loop đọc nhiều file JSON chậm.
- Worker xử lý ingest/predict/replay.
- Admin endpoint có auth.
- Log lỗi provider API.
- Reconcile report chạy mỗi ngày.
- Có rollback model.

## 21. Ưu tiên triển khai ngay

Thứ tự nên làm trong repo này:

```text
1. SieuTinHieuProvider (DONE)
2. DB schema + docker compose db/redis (DONE local foundation)
3. ingest latest/backfill (DONE initial script)
4. data_version (DONE after ingest)
5. signal freshness (DONE bước đầu trong API runtime)
6. DB DataLoader (DONE bước đầu)
7. worker compute (DONE bước đầu)
8. API signal precomputed-first + healthcheck (DONE bước đầu)
9. replay từng nến
10. production docker/nginx/backup
```

Điểm quan trọng nhất: đưa dữ liệu và signal vào cơ chế version. Khi có `model_version + data_version_hash`, hệ thống mới đủ tin cậy để giải thích, đối chiếu và rollback.

Cập nhật 2026-05-15:

- Đã thêm provider `SieuTinHieuProvider`.
- Đã thêm `tools/ingest_sieutinhieu.py` cho `latest` và `backfill`.
- Script ghi `ingestion_runs`, upsert `market_bars`, validate OHLC/volume cơ bản, và cập nhật `data_versions`.
- Đã test thành công `latest` với ACB limit 2 trong DB Docker local. Kết quả DB latest là `2026-05-14`, close `22.800000`, row_count `2`.
- Đã chạy `latest-limit 10` cho 61 symbols: `market_bars=610`, `data_versions=61`, `errors=0`. Có 60 symbols latest `2026-05-14`; riêng `BCG` latest `2025-10-08` theo provider.
- Đã chạy backfill lịch sử cho 61 symbols: `market_bars=210785`, `data_versions=61`, date range `2000-07-28` -> `2026-05-15`.
- Latest distribution sau backfill: 59 symbols latest `2026-05-15`, `AAV` latest `2026-05-14`, `BCG` latest `2025-10-08`.
- Có 142 bar lịch sử provider trả OHLC không hợp lệ; script skip các bar này và ghi `data_quality_checks` với `check_type=invalid_ohlcv`.
- Đã thêm signal freshness bước đầu trong `app/serve_train61_model.py`: payload OHLCV/signal có `data_version`, `data_version_hash`, `latest_bar_date`, `latest_close`; `/api/signal` chỉ trả RAM/disk cache khi hash còn khớp.
- Đã thêm `GET /api/data-version/<symbol>` để kiểm tra nguồn version hiện tại.
- Version source ưu tiên DB `data_versions` qua `DATABASE_URL`; nếu API runtime chưa kết nối DB thì fallback fingerprint CSV để bảo vệ cache trong mô hình file-first.
- Đã xử lý xung đột port host: PostgreSQL local đang chiếm `5432`, Docker DB được publish qua `localhost:15432`, host API/helper hiện đọc được `source=db:data_versions`.
- Đã mở rộng `src/data/loader.py` với `source="csv" | "db"`, `db_url`, `provider`, `fallback_csv`; giữ nguyên interface `symbols`, `load_symbol`, `load_all`, `load_all_context`, `summary`.
- Đã test DB loader: `symbols=61`, ACB đọc từ DB có `4843` rows, range `2006-11-21 -> 2026-05-15`; CSV local chỉ `2825` rows, range `2015-01-05 -> 2026-05-08`.
- Compose API local đã truyền `TRAIN61_DATA_SOURCE=db`, `DATA_PROVIDER=sieutinhieu`, `DEFAULT_TIMEFRAME=1D`.
- Đã thêm `tools/worker.py`: `compute-symbol-signal`, `compute-universe-signals`, `run-once-after-ingest`. Worker dùng lại signal engine hiện có, ghi cache `cache/signals/<model_id>/<symbol>.json`, và bỏ qua cache fresh theo `data_version_hash`.
- Đã thêm `tools/scheduler.py`: chạy chu kỳ `ingest latest -> compute signals`, có `--once`, `--skip-ingest`, `--interval-seconds`.
- Đã thêm Compose profile `worker` và `scheduler`.
- Đã test worker với ACB: lần đầu `generated`, `latest_bar_date=2026-05-15`, `data_version_hash=7968269c...`; lần sau `cached`. Scheduler `--once --skip-ingest --symbols ACB` chạy xong một chu kỳ.
- Đã chuyển API `/api/signal` sang precomputed-first: nếu cache fresh thì trả 200; nếu missing/stale thì trả 202 kèm `worker_command`. Mặc định `TRAIN61_API_GENERATE_ON_REQUEST=0`, nên API không spawn compute thread trong request web. Chỉ bật `TRAIN61_API_GENERATE_ON_REQUEST=1` khi debug local.
- Đã thêm `GET /health` để kiểm tra API, DB, Redis và active model; Compose API local đã có healthcheck gọi endpoint này.
- Đã thêm `tools/env.py` và cho worker/scheduler/ingest tự đọc `.env`, tránh trường hợp chạy script từ host nhưng rơi về CSV hoặc sai DB URL.
- Đã thêm `tools/daily_signal_report.py`: tạo danh sách open buy positions, entry mới, prediction cho phiên sau, watchlist xác suất mua cao nhất và thống kê từng ngày trong 5 phiên gần nhất. Mỗi cutoff date chỉ query DB với `market_bars.timestamp <= cutoff_date` để tránh dùng nến tương lai trong prediction.
- Việc tiếp theo: làm replay từng nến; sau đó hoàn thiện backup script, nginx config, auth admin endpoint và tối ưu API image.
