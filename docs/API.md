# Stock ML Platform API Documentation

## Overview

Stock ML Platform API provides RESTful endpoints for model management, backtesting, and leaderboard retrieval.

## Base URL

### API Endpoints
```
http://localhost:8000/api/v1
```

Through Nginx proxy:
```
http://localhost/api/v1
```

### Dashboard & UI
```
http://localhost:9001          (direct development access)
http://localhost               (production via Nginx)
```

**Files:**
- `/leaderboard.html` — Model leaderboard with rankings
- `/dashboard.html` — Trading signals chart view
- `/leaderboard.json` — JSON data source for leaderboard

---

## Endpoints

### Health Check

**GET** `/health`

Check API health status.

**Response (200)**:
```json
{
  "status": "healthy",
  "timestamp": "2026-05-30T12:00:00.000000",
  "version": "1.0.0"
}
```

---

### Detailed Health

**GET** `/health/detailed`

Get detailed system metrics.

**Response (200)**:
```json
{
  "status": "healthy",
  "timestamp": "2026-05-30T12:00:00.000000",
  "version": "1.0.0",
  "cpu_percent": 45.2,
  "memory_percent": 62.1,
  "uptime_seconds": 3600
}
```

---

### List Models

**GET** `/api/v1/models`

Get list of all trained models.

**Response (200)**:
```json
{
  "total": 2,
  "models": [
    {
      "name": "Technical Rules",
      "market": "vn_stock",
      "pnl_pct": -41.2,
      "win_rate": 0.405,
      "created_at": "2026-05-30T10:00:00"
    }
  ]
}
```

---

### Get Model Details

**GET** `/api/v1/models/{model_id}`

Get detailed information about a specific model.

**Path Parameters**:
- `model_id` (string): Model identifier

**Response (200)**:
```json
{
  "id": "model_1",
  "name": "Technical Rules",
  "market": "vn_stock",
  "pnl_pct": -41.2,
  "win_rate": 0.405,
  "total_trades": 100,
  "created_at": "2026-05-30T10:00:00"
}
```

**Response (404)**:
```json
{
  "detail": "Model not found"
}
```

---

### Get Leaderboard

**GET** `/api/v1/leaderboard`

Get complete leaderboard of all models.

**Query Parameters**:
- `n` (integer, optional): Number of top models to return (default: all, max: 1000)

**Response (200)**:
```json
{
  "models": [
    {
      "name": "Technical Rules",
      "market": "vn_stock",
      "pnl_pct": -41.2,
      "win_rate": 0.405
    }
  ],
  "summary": {
    "total_models": 1,
    "best_model": "Technical Rules",
    "best_pnl": -41.2
  }
}
```

---

### Get Top Models

**GET** `/api/v1/leaderboard/top/{n}`

Get top N models by PnL.

**Path Parameters**:
- `n` (integer): Number of models (1-1000)

**Response (200)**:
```json
{
  "models": [...],
  "summary": {
    "total_models": 1,
    "best_model": "Technical Rules",
    "best_pnl": -41.2
  }
}
```

**Response (400)**:
```json
{
  "detail": "n must be between 1 and 1000"
}
```

---

### Train Model

**POST** `/api/v1/models/train`

Start training a new model.

**Request Body**:
```json
{
  "model_name": "new_model_1",
  "config": {
    "param1": "value1"
  }
}
```

**Response (200)**:
```json
{
  "message": "Model training started",
  "model_name": "new_model_1"
}
```

**Response (422)**:
```json
{
  "detail": "Validation error"
}
```

---

## Model Lifecycle Management

### List All Runs

**GET** `/api/v1/runs`

List all model runs with optional filtering.

**Query Parameters**:
- `state` (string, optional): Filter by state — `trained`, `pinned`, or `retired`
- `market` (string, optional): Filter by market — `vn_stock`, `vn_derivatives`, `crypto_spot`, etc.

**Response (200)**:
```json
[
  {
    "run_id": "matrix_001/v22#abc12345",
    "bundle": "matrix_001",
    "run_name": "v22",
    "market": "vn_stock",
    "state": "trained",
    "composite_score": 0.58,
    "trades": 420,
    "wr": 0.52,
    "pnl_pct": 12.3,
    "sharpe": 0.85,
    "max_drawdown": -0.18
  }
]
```

---

### Get Run State

**GET** `/api/v1/runs/{run_id}/state`

Get current lifecycle state of a model run.

**Path Parameters**:
- `run_id` (string): Model run identifier (e.g., `matrix_001/v22#abc12345`)

**Response (200)**:
```json
{
  "run_id": "matrix_001/v22#abc12345",
  "state": "trained"
}
```

**Response (404)**:
```json
{
  "detail": "run not found: matrix_001/v22#abc12345"
}
```

---

### Change Run State (Pin/Unpin/Retire)

**PATCH** `/api/v1/runs/{run_id}/state`

Transition a model between lifecycle states: `trained` → `pinned` → `retired`.

**Path Parameters**:
- `run_id` (string): Model run identifier

**Request Body**:
```json
{
  "state": "pinned"
}
```

**State Transitions**:
- `pinned`: Model is exported to dashboard, signals visible as chart overlays
- `trained`: Model on leaderboard only, not on dashboard (default)
- `retired`: Historical reference only, no longer active

**Response (200)**:
```json
{
  "run_id": "matrix_001/v22#abc12345",
  "state": "pinned",
  "dashboard": {
    "pinned": 1
  }
}
```

**Response (400)**:
```json
{
  "detail": "invalid state: 'unknown'"
}
```

**Response (404)**:
```json
{
  "detail": "run not found: matrix_001/v22#abc12345"
}
```

---

### Spawn Retrain Job

**POST** `/api/v1/runs/{run_id}/retrain`

Start async retraining of a model run.

**Path Parameters**:
- `run_id` (string): Model run identifier

**Request Body** (optional):
```json
{
  "device": "cpu"
}
```

**Response (200)**:
```json
{
  "job_id": "abc123def456",
  "run_id": "matrix_001/v22#abc12345",
  "status": "running"
}
```

---

### Delete Run Cache

**DELETE** `/api/v1/runs/{run_id}/cache`

Quarantine feature/prediction cache files for a run (keeps metrics on leaderboard).

**Path Parameters**:
- `run_id` (string): Model run identifier

**Response (200)**:
```json
{
  "run_id": "matrix_001/v22#abc12345",
  "quarantined_cache": [
    "results/cache/_trash/20260530_143022/features/leading_v2/abc123.parquet",
    "results/cache/_trash/20260530_143022/predictions/def456.pkl"
  ]
}
```

---

### Delete Run Entirely

**DELETE** `/api/v1/runs/{run_id}`

Delete run directory and all its cache, rebuild leaderboard (removes from all views).

**Path Parameters**:
- `run_id` (string): Model run identifier

**Response (200)**:
```json
{
  "deleted_run_dir": "results/experiments/matrix_001/v22",
  "quarantined_cache": [...]
}
```

**Response (404)**:
```json
{
  "detail": "run not found: matrix_001/v22#abc12345"
}
```

---

## Bulk Operations

### Bulk Update State

**POST** `/api/v1/runs/bulk-state`

Transition multiple models at once (batch pin/retire/train).

**Request Body**:
```json
{
  "state": "retired",
  "filter": {
    "current_state": "trained",
    "market": "vn_stock"
  }
}
```

**Filter Parameters** (all optional):
- `current_state`: Only update runs currently in this state
- `market`: Only update runs for this market
- `state`: Alias for `current_state`

**Response (200)**:
```json
{
  "updated": 15,
  "run_ids": [
    "matrix_001/v22#abc12345",
    "matrix_001/v23#def67890"
  ],
  "state": "retired"
}
```

**Response (400)**:
```json
{
  "detail": "invalid state: 'unknown'"
}
```

---

### Bulk Delete Runs

**DELETE** `/api/v1/runs/bulk`

Delete all runs in a given state (requires explicit confirmation).

**Request Body**:
```json
{
  "state": "retired",
  "confirm": true
}
```

**Request Parameters**:
- `state` (string): Delete all runs in this state
- `confirm` (boolean): Must be `true` to confirm deletion

**Response (200)**:
```json
{
  "deleted": 12,
  "freed_mb": 245.3,
  "run_ids": [
    "matrix_001/v22#abc12345"
  ]
}
```

**Response (400)**:
```json
{
  "detail": "Must set confirm: true to delete"
}
```

---

## Cache Management

### Get Cache Statistics

**GET** `/api/v1/cache/stats`

Get disk usage and orphan statistics for cache system.

**Response (200)**:
```json
{
  "feature_cache_mb": 142.3,
  "prediction_cache_mb": 48.1,
  "orphan_count": 7,
  "orphan_mb": 23.5,
  "trash_mb": 12.0,
  "referenced_features": 18,
  "referenced_predictions": 18
}
```

---

### Run Garbage Collection Sweep

**POST** `/api/v1/gc/sweep`

Scan cache for orphaned files and optionally quarantine them.

**Request Body**:
```json
{
  "apply": false,
  "purge_older_than_days": null
}
```

**Parameters**:
- `apply` (boolean): If `false`, dry-run (report only). If `true`, move orphans to trash.
- `purge_older_than_days` (float, optional): Also purge trash batches older than N days

**Response (200)**:
```json
{
  "dry_run": true,
  "orphan_count": 7,
  "orphan_mb": 23.5,
  "quarantined": 0,
  "purged": 0
}
```

---

### Purge Old Trash

**POST** `/api/v1/cache/purge-trash`

Delete trash batches older than N days (permanently removes quarantined files).

**Request Body**:
```json
{
  "older_than_days": 7.0
}
```

**Response (200)**:
```json
{
  "purged_dirs": 2,
  "freed_mb": 45.6,
  "older_than_days": 7.0
}
```

---

## Job Tracking

### List Jobs

**GET** `/api/v1/jobs`

List all background jobs (retrain, backtest, etc.).

**Query Parameters**:
- `status` (string, optional): Filter by status — `running`, `done`, or `error`

**Response (200)**:
```json
[
  {
    "job_id": "abc123def456",
    "type": "retrain",
    "run_id": "matrix_001/v22#abc12345",
    "status": "running",
    "started_at": "2026-05-30T14:32:15.123456",
    "log": "logs/retrain_abc123def456.log"
  }
]
```

---

### Get Job Status

**GET** `/api/v1/jobs/{job_id}`

Poll status of a background job.

**Path Parameters**:
- `job_id` (string): Job identifier

**Response (200)**:
```json
{
  "job_id": "abc123def456",
  "status": "done",
  "started_at": "2026-05-30T14:32:15.123456",
  "exit_code": 0,
  "log": "logs/retrain_abc123def456.log"
}
```

**Response (404)**:
```json
{
  "detail": "job not found: abc123def456"
}
```

---

## Error Handling

All endpoints return errors in the following format:

```json
{
  "detail": "Error message"
}
```

### Status Codes
- **200**: Success
- **400**: Bad Request (invalid parameters)
- **404**: Not Found
- **422**: Validation Error
- **429**: Rate Limited (100 requests/minute)
- **500**: Internal Server Error
- **501**: Not Implemented (feature disabled)

---

## Rate Limiting

All endpoints are rate-limited to **100 requests per minute**.

When limit is exceeded:
```json
{
  "detail": "Rate limit exceeded"
}
```

---

## Authentication

Currently, no authentication is required (development mode).

For production, add:
- API Key authentication via `X-API-Key` header
- JWT Bearer token support
- OAuth2 integration

---

## Deployment

### Docker

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# Verify API health
curl http://localhost:8000/health

# View logs
docker-compose logs -f api
```

### Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configure:
- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `DEBUG`: Debug mode (default: False)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `CORS_ORIGINS`: Comma-separated list of allowed origins
- `DB_ENABLED`: Enable database (requires PostgreSQL running)

### Nginx Configuration

Nginx reverse proxy is configured in:
- `infrastructure/nginx/nginx.conf`
- Listens on port 80 (HTTP) and 443 (HTTPS)
- Proxies requests to FastAPI (port 8000)
- Handles CORS and SSL termination

---

## Dashboard Integration

Dashboard files are located in:
- `stock_ml/dashboard/` or `stock_ml/visualization/`

### Serve Dashboard

From `stock_ml/` directory:

```bash
python -m http.server 8181
```

Access:
- Dashboard: http://localhost:8181/visualization/dashboard.html
- Leaderboard: http://localhost:8181/visualization/leaderboard.html

Dashboard loads API config from `api-config.js` and fetches data from:
- `/api/v1/leaderboard` — model leaderboard
- `/api/v1/models` — model list
- `/api/v1/models/{model_id}` — model details

---

## Testing

### Unit Tests

```bash
pytest stock_ml/api/tests/ -v
```

### Integration Tests

```bash
# Start API first
python -m uvicorn stock_ml.api.main:app

# In another terminal
pytest stock_ml/tests/integration/ -v
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/api/v1/models

# Get leaderboard
curl http://localhost:8000/api/v1/leaderboard

# Get top 5 models
curl http://localhost:8000/api/v1/leaderboard/top/5

# Get model details
curl http://localhost:8000/api/v1/models/v22
```

---

## Performance

### Response Times
- Health endpoints: ~1ms
- Models list: ~1ms
- Leaderboard query: ~1-5ms
- Top models: ~1-2ms

### Scalability
- Stateless API (no session state)
- Can handle 100+ concurrent requests
- Rate limited to 100 req/min per endpoint
- Use load balancing for horizontal scaling

---

## Monitoring

### Health Checks

The API provides health check endpoints:

```bash
# Quick health check
curl http://localhost:8000/health

# Detailed health with metrics
curl http://localhost:8000/health/detailed
```

### Logging

Logs are written to:
- Console (stdout/stderr)
- File: `./logs/app.log` (JSON format, rotated daily)

Configure log level in `.env`:
```
LOG_LEVEL=DEBUG
```

### Database Monitoring

If database is enabled, monitor:
- Connection pool status
- Query performance
- Migration status

---

## API Schema

Complete OpenAPI schema available at:
- `/api/openapi.json` — OpenAPI 3.0 spec
- `/api/docs` — Swagger UI
- `/api/redoc` — ReDoc documentation

---

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
python -m uvicorn stock_ml.api.main:app --port 8001
```

### Database Connection Failed
```bash
# Check PostgreSQL is running
psql -U stockml -d stockml -c "SELECT 1"

# Or disable database
DB_ENABLED=false
```

### CORS Errors
Ensure `CORS_ORIGINS` in `.env` includes your dashboard URL:
```
CORS_ORIGINS=["http://localhost:8181","http://localhost:3000"]
```

### Rate Limit Issues
Reduce request rate or increase limit in `stock_ml/api/middleware/rate_limiter.py`

---

## Support

- **Issues**: Check `docs/` folder
- **Logs**: `./logs/app.log`
- **Config**: `.env` and `stock_ml/api/config.py`
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "model_name"],
      "msg": "Value error, Model name must be alphanumeric with underscores only"
    }
  ]
}
```

---

## Error Handling

All errors return JSON with appropriate HTTP status code:

```json
{
  "detail": "Error message"
}
```

### Common Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (validation error) |
| 404 | Not Found |
| 422 | Unprocessable Entity (invalid data) |
| 429 | Too Many Requests (rate limited) |
| 500 | Internal Server Error |

---

## Rate Limiting

- **Limit**: 100 requests per minute (default)
- **Status Code**: 429 (Too Many Requests)

---

## Authentication

Currently no authentication. Production deployment should add:
- API Key authentication
- JWT token validation

---

## CORS

Allowed origins (configured in .env):
- http://localhost:3000
- http://localhost:8000
- http://127.0.0.1:3000
- http://127.0.0.1:8000

---

## Interactive API Documentation

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **OpenAPI Schema**: http://localhost:8000/api/openapi.json

---

## Examples

### Check API Health

```bash
curl http://localhost:8000/health
```

### Get Top 5 Models

```bash
curl http://localhost:8000/api/v1/leaderboard/top/5
```

### Train New Model

```bash
curl -X POST http://localhost:8000/api/v1/models/train \
  -H "Content-Type: application/json" \
  -d '{"model_name": "test_model_1", "config": {}}'
```

### Through Nginx

```bash
curl http://localhost/api/v1/models
```

---

## Support

For issues or questions:
1. Check the logs: `docker-compose logs -f api`
2. Review health endpoint: `GET /health/detailed`
3. Check API docs: `GET /api/docs`
