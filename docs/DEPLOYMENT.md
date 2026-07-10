# Deployment Guide

Production deployment với Docker, Nginx, PostgreSQL.

## Environment Setup

### 1. Create .env File

```bash
cp .env.example .env
```

Configure:
```env
# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/stockml
DB_ENABLED=true

# CORS
CORS_ORIGINS=["https://your-domain.com"]

# Security
API_KEY_ENABLED=true
RATE_LIMIT_ENABLED=true
```

## Deployment Steps

### 1. Start Services

```bash
docker-compose up -d
```

### 2. Run Migrations

```bash
docker exec -w /app stock-ml-api alembic upgrade head
```

### 3. Seed Data

```bash
docker exec -w /app stock-ml-api python -m stock_ml.scripts.seed_universes
```

### 4. Verify

```bash
curl http://localhost/health
docker-compose ps
```

## SSL/HTTPS Setup

### Let's Encrypt

```bash
sudo certbot certonly --standalone -d your-domain.com
```

Update `infrastructure/nginx/nginx.conf` with certificate paths.

## Backups

```bash
# PostgreSQL backup
docker exec stock-ml-postgres pg_dump -U stockml stockml > backup.sql

# Restore
docker exec -i stock-ml-postgres psql -U stockml stockml < backup.sql
```

## Monitoring

- Health checks: `curl https://your-domain.com/health/detailed`
- Logs: `docker-compose logs -f api`
- Resources: `docker stats`

## Security Checklist

- ✅ DEBUG=False
- ✅ Enable API_KEY_ENABLED
- ✅ Setup SSL certificates
- ✅ Configure CORS_ORIGINS
- ✅ Use PostgreSQL (not SQLite)
- ✅ Enable rate limiting
- ✅ Restrict network access
