# Quick Start Guide

Khởi động Stock ML Platform trong 5 phút.

## Yêu cầu

- Docker & Docker Compose
- Git
- Windows PowerShell hoặc bash

## Setup

### 1. Clone & Navigate

```bash
cd stock_ml
```

### 2. Start Services

```bash
docker-compose up -d
```

Services sẽ start:
- **API** (FastAPI) → `http://localhost:8000`
- **Dashboard** → `http://localhost:9001` hoặc `http://localhost` (via Nginx)
- **Database** → SQLite at `./results/leaderboard.db`
- **Nginx** → Proxy `http://localhost`

### 3. Verify Services

```bash
# Check API health
curl http://localhost/health

# Check all containers
docker-compose ps
```

### 4. Access Dashboard

**Recommended: Use Nginx (port 80)**
```
http://localhost/leaderboard.html
http://localhost/template-explorer.html
http://localhost/model-library.html
http://localhost/universe.html
```

**Direct Dashboard (port 9001)**
```
http://localhost:9001/leaderboard.html
```

## Common Tasks

### View Logs

```bash
docker-compose logs -f api
docker-compose logs -f dashboard
```

### Stop Services

```bash
docker-compose down
```

### Reset Database

```bash
# Remove old database
rm ./results/leaderboard.db

# Restart containers (migrations will re-run)
docker-compose up -d
```

### Seed Data

```bash
# Seed universes from YAML
docker exec -w /app stock-ml-api python -m stock_ml.scripts.seed_universes

# Access API to create templates
curl -X POST http://localhost/api/v1/templates/ \
  -H "Content-Type: application/json" \
  -d '{"name":"My Template",...}'
```

## Key Services

| Service | Port | Purpose |
|---------|------|---------|
| API | 8000 | FastAPI endpoints |
| Database | N/A | SQLite (file-based) |
| Dashboard | 9001 | Static files (Node) |
| Nginx | 80 | Proxy + routing |

## API Base URLs

- **Via Nginx**: `/api/v1/model-library/...`, `/api/v1/templates/...`
- **Direct**: `http://localhost:8000/api/v1/model-library/...`

## Next Steps

- 📖 Read [API Documentation](API.md)
- 🏗️ Read [System Architecture](../ARCHITECTURE.md)
- 🔧 Read [Development Guide](DEVELOPMENT.md)
- 🚀 Read [Deployment Guide](DEPLOYMENT.md)

## Troubleshooting

See [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues.
