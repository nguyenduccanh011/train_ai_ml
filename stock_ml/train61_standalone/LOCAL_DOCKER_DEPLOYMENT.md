# Local Docker deployment

Mục tiêu của cấu hình này là chạy local giống production nhất có thể, nhưng vẫn bind về localhost để dễ kiểm thử trước khi đưa lên server.

## Trạng thái hiện tại

Đã chạy và kiểm tra thành công:

```text
train61-db-local      healthy, localhost:15432
train61-redis-local   healthy, localhost:6379
```

DB local đã apply schema `db/init/001_train61_schema.sql` và có các bảng nền tảng:

```text
instruments
market_bars
data_versions
ingestion_runs
data_quality_checks
model_artifacts
predictions
signals
trades
replay_runs
replay_steps
```

`api` đang nằm trong Compose profile `api`, không chạy mặc định. Lý do: image API hiện kéo/cài nhiều dependency ML (`lightgbm`, `xgboost`, `catboost`) nên build lần đầu có thể rất lâu. Giai đoạn hiện tại ưu tiên dựng DB/Redis trước để chuẩn bị ingest và schema.

## Thành phần

```text
db      TimescaleDB/PostgreSQL cho dữ liệu thị trường và signal về sau
redis   queue/status/cache nhẹ
api     Flask app hiện tại, port 5012, bật bằng profile api
```

Hiện tại `api` vẫn dùng file dataset/cache sẵn có trong repo. `db` và `redis` được dựng trước để các phase tiếp theo có thể thêm ingestion, worker và scheduler mà không đổi kiến trúc deploy.

Schema DB local nằm tại:

```text
db/init/001_train61_schema.sql
```

File này tự chạy khi Postgres volume được tạo lần đầu. Nếu DB volume đã tồn tại, apply thủ công bằng lệnh ở phần kiểm tra service.

## Chạy lần đầu

```powershell
docker compose -f docker-compose.local.yml up -d
```

Lệnh trên mặc định chỉ chạy hạ tầng `db` và `redis`. Đây là bước nên chạy trước để chuẩn bị database local. Không cần `--build` vì hai service này dùng image public.

Để chạy thêm API container:

```powershell
docker compose -f docker-compose.local.yml --profile api up -d --build
```

Mở:

```text
http://127.0.0.1:5012
```

Nếu muốn đổi port, mật khẩu DB hoặc biến môi trường khác:

```powershell
Copy-Item .env.example .env
```

Sau đó sửa `.env` rồi chạy compose lại.

## Chạy nền

```powershell
docker compose -f docker-compose.local.yml up -d
```

Chạy nền kèm API:

```powershell
docker compose -f docker-compose.local.yml --profile api up -d --build
```

Xem log:

```powershell
docker compose -f docker-compose.local.yml logs -f api
```

Dừng:

```powershell
docker compose -f docker-compose.local.yml down
```

Dừng và xóa volume database local:

```powershell
docker compose -f docker-compose.local.yml down -v
```

## Kiểm tra service

```powershell
docker compose -f docker-compose.local.yml ps
docker compose -f docker-compose.local.yml exec -T db pg_isready -U train61 -d train61
docker compose -f docker-compose.local.yml exec -T redis redis-cli ping
docker compose -f docker-compose.local.yml exec -T db psql -U train61 -d train61 -c "\dt"
```

Apply schema thủ công vào DB đang chạy:

```powershell
Get-Content db\init\001_train61_schema.sql | docker compose -f docker-compose.local.yml exec -T db psql -U train61 -d train61
```

Kiểm tra API:

```powershell
Invoke-RestMethod "http://127.0.0.1:5012/api/models"
Invoke-RestMethod "http://127.0.0.1:5012/api/ohlcv/ACB"
```

## Ghi chú triển khai

- Flask trong container bind `0.0.0.0` qua `TRAIN61_HOST`, còn truy cập từ máy host vẫn là `127.0.0.1:5012`.
- Các thư mục `data`, `models`, `cache`, `config`, `results`, `reports` được mount từ máy host để giữ đúng dữ liệu hiện tại.
- Không dùng `docker compose down -v` nếu muốn giữ dữ liệu Postgres local.
- Khi thêm ingestion/worker, nên thêm service `worker` và `scheduler` vào compose này, không chạy job nặng trong `api`.
- Trước khi bật API container thường xuyên, nên tách dependency runtime nhẹ hơn, ví dụ `requirements-runtime.txt`, để image build nhanh và ổn định.

## Bước tiếp theo

Thứ tự nên làm tiếp:

```text
1. Thêm SieuTinHieuProvider để gọi API dữ liệu. DONE
2. Viết ingest latest/backfill vào market_bars. DONE
3. Cập nhật data_versions sau ingest. DONE
4. Thêm signal freshness dựa trên data_version_hash. DONE bước đầu
5. Thêm DataLoader đọc DB. DONE bước đầu
6. Thêm worker service cho ingest/predict. DONE bước đầu
7. Chuyển API signal sang precomputed-first, không compute nặng trong request mặc định. DONE
8. Tối ưu API image rồi bật profile api thường xuyên.
```

## Cập nhật ingest Siêu Tín Hiệu

Đã thêm:

```text
src/data/providers/base.py
src/data/providers/sieutinhieu.py
tools/ingest_sieutinhieu.py
```

Chạy latest cho một mã từ host nếu port Postgres local trỏ đúng container:

```powershell
python tools\ingest_sieutinhieu.py --mode latest --symbols ACB --latest-limit 10 --database-url postgresql://train61:train61_local_password@localhost:15432/train61
```

Nếu máy host đang có PostgreSQL riêng cùng dùng port 5432, cấu hình local hiện publish Docker DB qua `localhost:15432`. Có thể chạy từ host bằng port này, hoặc chạy script trong network của DB container:

```powershell
docker run --rm --network container:train61-db-local -v "${PWD}:/work" -w /work python:3.12-slim sh -c "pip install -q requests psycopg[binary] && python tools/ingest_sieutinhieu.py --mode latest --symbols ACB --latest-limit 10 --database-url postgresql://train61:train61_local_password@127.0.0.1:5432/train61"
```

Lệnh đã kiểm tra thành công với ACB:

```text
ACB latest-limit 2 -> market_bars inserted=2, data_versions row_count=2, latest_timestamp=2026-05-14
```

Sau đó đã chạy latest cho toàn bộ `config/train61_symbols.json`:

```text
symbols=61, market_bars=610, data_versions=61, errors=0
60 symbols latest_timestamp=2026-05-14
BCG latest_timestamp=2025-10-08 theo dữ liệu provider trả về
```

Đã chạy backfill lịch sử cho 61 mã:

```text
market_bars=210785
data_versions=61
date range=2000-07-28 -> 2026-05-15
latest distribution:
  59 symbols -> 2026-05-15
  AAV -> 2026-05-14
  BCG -> 2025-10-08
```

Backfill phát hiện một số bar lịch sử provider trả OHLC không hợp lệ. Script hiện skip các bar này và ghi vào `data_quality_checks`:

```text
invalid_ohlcv warnings=142
```

## Cập nhật signal freshness

Đã cập nhật `app/serve_train61_model.py`:

```text
GET /api/data-version/<symbol>
GET /api/ohlcv/<symbol>
GET /api/signal/<model_id>/<symbol>
GET /api/symbols
```

Server hiện gắn `data_version`, `data_version_hash`, `latest_bar_date` và `latest_close` vào OHLCV/signal payload. Khi request signal, cache trong RAM/disk chỉ được trả nếu `data_version_hash` còn khớp nguồn dữ liệu hiện tại. Nếu stale hoặc missing, API mặc định trả `202` kèm trạng thái và `worker_command`; worker/scheduler chịu trách nhiệm generate lại.

Nguồn version ưu tiên là bảng DB `data_versions` khi `DATABASE_URL` kết nối được. Nếu chưa cấu hình DB cho API runtime, server fallback sang fingerprint từ CSV local để vẫn phát hiện cache cũ trong runtime file-first hiện tại.

Đã kiểm tra cú pháp và endpoint bằng Flask test client:

```text
python -m py_compile app\serve_train61_model.py
/api/data-version/ACB -> 200, source=csv, latest=2026-05-08
/api/ohlcv/ACB?model_id=train61_pooled -> 200, source=csv, latest=2026-05-08
```

Lưu ý: máy host có PostgreSQL riêng đang chiếm `localhost:5432`, nên Docker DB local đã được publish qua `localhost:15432`. Trong container API, `DATABASE_URL` vẫn trỏ `db:5432` theo Compose network.

## Cập nhật DB DataLoader

Đã mở rộng `src/data/loader.py`:

```python
DataLoader(
    data_dir="data/vn_stock_ai_dataset_cleaned",
    source="csv" | "db",
    db_url="postgresql://train61:train61_local_password@localhost:15432/train61",
    provider="sieutinhieu",
)
```

Interface cũ vẫn giữ:

```text
loader.symbols
loader.load_symbol(symbol)
loader.load_all(symbols=...)
loader.load_all_context()
loader.summary()
```

Mặc định nếu không truyền `source` thì đọc `TRAIN61_DATA_SOURCE`; nếu biến này không có thì vẫn đọc CSV để không phá pipeline cũ. Khi `source=db`, loader đọc `market_bars` và danh sách mã từ `data_versions`; nếu DB không có dữ liệu và `fallback_csv=True` thì fallback CSV.

Compose API local đã truyền:

```text
TRAIN61_DATA_SOURCE=db
DATA_PROVIDER=sieutinhieu
DEFAULT_TIMEFRAME=1D
DATABASE_URL=postgresql://train61:...@db:5432/train61
```

Đã kiểm tra từ host với DB Docker qua `localhost:15432`:

```text
DataLoader(source=db).symbols -> 61 symbols
ACB DB -> 4843 rows, 2006-11-21 -> 2026-05-15, latest close=23.1
ACB CSV -> 2825 rows, 2015-01-05 -> 2026-05-08, latest close=22.85
load_all(['ACB', 'AAV']) -> 6812 rows, max timestamp=2026-05-15
load_all_context() -> 5 CSV context symbols
```

Chênh lệch DB/CSV là kỳ vọng hiện tại vì DB đã backfill mới hơn và dài hơn CSV local.

## Cập nhật worker compute

Đã thêm:

```text
tools/worker.py
tools/scheduler.py
docker-compose.local.yml service worker
docker-compose.local.yml service scheduler
```

Worker dùng lại logic generate signal của `app/serve_train61_model.py`, ghi cache JSON vào cùng `cache/signals/<model_id>/<symbol>.json`, và kiểm tra freshness trước khi generate. Nếu cache còn khớp `data_version_hash`, worker trả `cached`; nếu stale hoặc `--force`, worker generate lại.

Các script `tools/worker.py`, `tools/scheduler.py` và `tools/ingest_sieutinhieu.py` tự đọc `.env` nếu có, nên khi chạy từ host sẽ dùng đúng `DATABASE_URL`, `TRAIN61_DATA_SOURCE`, `DATA_PROVIDER` như cấu hình local.

API `/api/signal/<model_id>/<symbol>` hiện đã chạy theo hướng precomputed-first:

```text
fresh cache -> trả 200 với signal payload
missing/stale cache -> trả 202 với trạng thái và worker_command
```

Mặc định API không tự generate signal trong request web. Biến điều khiển:

```env
TRAIN61_API_GENERATE_ON_REQUEST=0
```

Chỉ bật `TRAIN61_API_GENERATE_ON_REQUEST=1` khi cần debug local. Production nên để `0` và để worker/scheduler tính signal.

Health endpoint:

```text
GET /health
```

Endpoint này kiểm tra API, DB, Redis và active model. Compose API local đã có healthcheck gọi `/health`.

## Daily signal report

Đã thêm report hằng ngày:

```powershell
python tools\daily_signal_report.py --model-id train61_pooled --days 5
```

Report này tạo:

```text
reports/daily_signals/daily_signal_report_<date>.json
reports/daily_signals/open_positions_<date>.csv
reports/daily_signals/new_entries_<date>.csv
reports/daily_signals/next_session_predictions_<date>.csv
reports/daily_signals/watchlist_top_buy_proba_<date>.csv
reports/daily_signals/all_predictions_<date>.csv
```

Với từng ngày cutoff trong 5 phiên gần nhất, script query DB chỉ lấy `market_bars.timestamp <= cutoff_date`, compute feature/predict lại trên dữ liệu tới ngày đó, nên phần prediction không dùng nến tương lai. `open_positions` và `new_entries` được cắt trạng thái theo ngày cutoff từ signal cache/trades đã precompute.

Chạy một symbol từ host:

```powershell
$env:DATABASE_URL='postgresql://train61:train61_local_password@localhost:15432/train61'
$env:TRAIN61_DATA_SOURCE='db'
python tools\worker.py compute-symbol-signal --model-id train61_pooled --symbols ACB
```

Chạy toàn universe:

```powershell
python tools\worker.py compute-universe-signals --model-id train61_pooled --symbols config/train61_symbols.json
```

Chạy scheduler một vòng sau ingest:

```powershell
python tools\scheduler.py --once --model-id train61_pooled --symbols config/train61_symbols.json
```

Chạy scheduler chỉ compute, không ingest:

```powershell
python tools\scheduler.py --once --skip-ingest --model-id train61_pooled --symbols ACB
```

Chạy bằng Docker Compose profile:

```powershell
docker compose -f docker-compose.local.yml --profile worker run --rm worker
docker compose -f docker-compose.local.yml --profile scheduler up -d scheduler
```

Đã kiểm tra:

```text
python tools\worker.py compute-symbol-signal --model-id train61_pooled --symbols ACB
-> generated, latest_bar_date=2026-05-15, data_version_hash=7968269c...

python tools\worker.py compute-symbol-signal --model-id train61_pooled --symbols ACB
-> cached

python tools\scheduler.py --once --skip-ingest --symbols ACB --model-id train61_pooled
-> scheduler cycle completed
```
