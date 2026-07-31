# Troubleshooting Guide

Common issues và solutions.

> 🆕 **Cập nhật 2026-07-31:** DB sống là **PostgreSQL** (`stock-ml-postgres`, port 5433). Lệnh `docker exec … alembic upgrade head` bên dưới chạy TRONG container nên Postgres thắng (đúng). ⚠️ Các lệnh `sqlite3 /app/results/leaderboard.db` chỉ đọc **file SQLite CŨ (rev 0025, lỗi thời)** — KHÔNG phải dữ liệu sống; dùng `psql`/API thay thế. (Bug "alembic tạo nhầm SQLite" chỉ xảy ra khi chạy từ shell HOST thiếu `DATABASE_URL` — xem `ENGINE_UPGRADE_AND_LEGACY_RESTRUCTURE.md` §1.4.)

## 429 Too Many Requests

**Problem**: API returns 429 when accessing endpoints

**Causes**:
1. Database migrations not run
2. Exception handler catching all exceptions
3. CORS blocking requests

**Solutions**:

```bash
# Run migrations
docker exec -w /app stock-ml-api alembic upgrade head

# Check API logs
docker-compose logs api | grep ERROR

# Verify CORS config
curl -H "Origin: http://localhost:9001" http://localhost/api/templates/ -v
```

## 404 Not Found

**Problem**: API endpoints return 404

**Common Issues**:
- Wrong API prefix (`/api/v1/` vs `/api/`)
- Route not registered in main.py
- Wrong database tables

**Debug**:

```bash
# List all routes
curl http://localhost:8000/api/openapi.json | grep -i '"path"'

# Check database tables
docker exec stock-ml-api sqlite3 /app/results/leaderboard.db ".tables"

# Test direct API
curl http://localhost:8000/api/v1/model-library/targets
```

## CORS Errors

**Problem**: Browser blocks requests with CORS error

**Solution**:

1. Add origin to `stock_ml/api/config.py`:

```python
cors_origins: list[str] = [
    "http://localhost:9001",
    "http://your-domain.com"
]
```

2. Rebuild Docker:

```bash
docker-compose build
docker-compose up -d api
```

## Database Connection Failed

**Problem**: "no such table" or database locked

**Solutions**:

```bash
# Check migrations status
docker exec stock-ml-api alembic current

# Run migrations again
docker exec -w /app stock-ml-api alembic upgrade head

# Check database file
ls -lh ./results/leaderboard.db

# Reset database (development only!)
rm ./results/leaderboard.db
docker-compose restart api
```

## Empty Database After Restart

**Problem**: Database tables exist but no data

**Cause**: Migrations create empty schema only

**Solution**: Seed data:

```bash
docker exec -w /app stock-ml-api python -m stock_ml.scripts.seed_universes
```

## Nginx 502 Bad Gateway

**Problem**: Nginx cannot reach API

**Debug**:

```bash
# Check if API is running
docker-compose ps | grep api

# Check API logs
docker-compose logs api | grep ERROR

# Test direct connection from Nginx container
docker exec stock-ml-nginx curl http://api:8000/health

# Check Nginx logs
docker exec stock-ml-nginx cat /var/log/nginx/error.log
```

## Docker Container Won't Start

**Problem**: Container exits immediately

**Debug**:

```bash
# Check logs
docker-compose logs api

# Inspect container
docker inspect stock-ml-api

# Check resource limits
docker stats

# Try manual startup
docker run -it train_ai_ml-api /bin/bash
```

## High Memory Usage

**Problem**: Container using 4GB+ memory

**Solutions**:

```bash
# Limit container memory
# Add to docker-compose.yml:
api:
  deploy:
    resources:
      limits:
        memory: 2G

# Restart
docker-compose up -d api

# Monitor
docker stats stock-ml-api
```

## Slow API Responses

**Problem**: API takes >5 seconds to respond

**Debug**:

```bash
# Check database performance
docker exec stock-ml-api sqlite3 /app/results/leaderboard.db ".stats on"

# Monitor resource usage
docker stats stock-ml-api

# Check logs for slow queries
docker-compose logs api | grep "duration"
```

## SSL Certificate Errors

**Problem**: HTTPS returns certificate error

**Solutions**:

```bash
# Check certificate validity
openssl x509 -in /path/to/cert.pem -text -noout

# Verify Nginx has correct paths
docker exec stock-ml-nginx cat /etc/nginx/nginx.conf | grep ssl_

# Renew certificate (Let's Encrypt)
sudo certbot renew --force-renewal

# Reload Nginx
docker exec stock-ml-nginx nginx -s reload
```

## Feature/Template Load Fails

**Problem**: "No templates found", "Failed to load feature sets"

**Causes**:
1. Database not seeded
2. API URL mismatch in dashboard
3. Database schema missing tables

**Solutions**:

```bash
# Seed universes first (prerequisite)
docker exec -w /app stock-ml-api python -m stock_ml.scripts.seed_universes

# Check API returns data
curl http://localhost/api/model-library/feature-sets
curl http://localhost/api/model-library/targets
curl http://localhost/api/templates/

# Check browser console
# Open http://localhost/template-explorer.html
# Look for API_BASE value in console

# Verify API endpoints exist
curl http://localhost:8000/api/openapi.json | grep feature-sets
```

## Dashboard Not Loading

**Problem**: `http://localhost:9001/template-explorer.html` shows 404

**Causes**:
1. Dashboard container not running
2. File not in container
3. Wrong port

**Solutions**:

```bash
# Check dashboard container
docker-compose ps | grep dashboard

# Check files in container
docker exec stock-ml-dashboard ls -la /app/*.html

# Check if file exists locally
ls stock_ml/dashboard/template-explorer.html

# Try via Nginx (port 80)
curl -I http://localhost/template-explorer.html
```

## Port Already in Use

**Problem**: "Address already in use"

**Solution**:

```bash
# Find process on port
lsof -i :8000
netstat -tlnp | grep 8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
api:
  ports:
    - "8001:8000"
```

## Permission Denied Errors

**Problem**: "Permission denied" when accessing files/folders

**Solution**:

```bash
# Fix ownership
sudo chown -R $USER:$USER ./results/
sudo chown -R $USER:$USER ./logs/

# Fix permissions
chmod -R 755 ./results/
chmod -R 755 ./logs/
```

## Migration Conflicts

**Problem**: "Target database is not up to date"

**Solution**:

```bash
# Check current version
docker exec stock-ml-api alembic current

# Check all versions
docker exec stock-ml-api alembic history

# Upgrade to latest
docker exec -w /app stock-ml-api alembic upgrade head

# If still fails, reset (development only!)
docker exec stock-ml-api rm /app/results/leaderboard.db
docker-compose restart api
```

## Getting Help

1. **Check Logs**
   ```bash
   docker-compose logs -f [service] | grep ERROR
   ```

2. **Health Check**
   ```bash
   curl http://localhost/health/detailed
   ```

3. **Docker Stats**
   ```bash
   docker stats
   docker inspect stock-ml-api
   ```

4. **API Docs**
   ```
   http://localhost:8000/api/docs
   http://localhost:8000/api/redoc
   ```

5. **Check Configuration**
   ```bash
   cat .env
   cat stock_ml/api/config.py
   docker-compose config
   ```
