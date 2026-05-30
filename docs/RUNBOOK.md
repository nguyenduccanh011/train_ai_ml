# Stock ML Platform - Operations Runbook

**Last Updated**: 2026-05-30  
**Version**: 1.0.0  
**Owner**: Stock ML Team  
**On-Call**: DevOps Engineer  

---

## 1. Quick Reference

### Health Check
```bash
# Local API
curl http://localhost:8000/health

# Via Nginx
curl http://localhost/health

# Automated script
bash scripts/health_check.sh
```

### Common Commands

```bash
# View logs
docker-compose logs -f api

# Restart services
docker-compose restart

# Scale API service
docker service scale stock_ml_api=3

# Check resource usage
docker stats

# Run tests
pytest tests/ -v

# Generate coverage report
pytest tests/ --cov=stock_ml --cov-report=html
```

---

## 2. System Architecture

```
┌─────────────────────────────────────────────┐
│         Client / Browser / API Client       │
└────────────────┬────────────────────────────┘
                 │ HTTP/HTTPS
┌────────────────▼────────────────────────────┐
│    Nginx Reverse Proxy (Port 80/443)        │
│  - Load balancing                           │
│  - SSL termination                          │
│  - Gzip compression                         │
└────────────────┬────────────────────────────┘
                 │ Internal network
┌────────────────▼────────────────────────────┐
│   FastAPI Application (Port 8000)           │
│  - Health checks                            │
│  - Model management                         │
│  - Leaderboard API                          │
│  - Error handling & logging                 │
└────────────────┬────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
   ┌────▼───┐      ┌─────▼────┐
   │  Logs  │      │   Data   │
   │ (JSON) │      │ (Models) │
   └────────┘      └──────────┘
```

### Components
- **Nginx**: Reverse proxy, load balancing, SSL
- **FastAPI**: REST API server with automatic documentation
- **Docker**: Containerization and orchestration
- **Logging**: Structured JSON logging with rotation
- **Health Checks**: Automated system monitoring

---

## 3. Daily Operations

### Morning Check-in (7 AM)

```bash
# 1. Check all services are up
docker-compose ps

# 2. Check recent logs for errors
docker-compose logs api --since 8h | grep -i "error\|warn"

# 3. Check disk usage
du -sh logs/ data/ results/

# 4. Verify health endpoints
curl -s http://localhost:8000/health/detailed | jq .

# 5. Check resource usage
docker stats
```

### During Business Hours

- Monitor health endpoint every 5 minutes (automated via cron/monitoring)
- Watch for error logs: `docker-compose logs -f api | grep -i error`
- Track API response times in application logs

### End of Day (6 PM)

```bash
# 1. Archive logs older than 7 days
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;

# 2. Check leaderboard was updated
ls -lt results/leaderboard.json | head -1

# 3. Verify no pending errors
docker-compose logs api --since 1h | grep -i error

# 4. Backup critical data
tar -czf backup_results_$(date +%Y%m%d).tar.gz results/
```

---

## 4. Common Tasks

### 4.1 View Application Logs

```bash
# Last 50 lines
docker-compose logs api --tail 50

# Real-time streaming
docker-compose logs -f api

# Last 2 hours
docker-compose logs api --since 2h

# Specific time range
docker-compose logs api --since "2026-05-30 10:00:00" --until "2026-05-30 11:00:00"

# Search for errors
docker-compose logs api | grep -i error
```

### 4.2 Monitor System Performance

```bash
# Docker container stats
docker stats stock-ml-api --no-stream

# Continuous monitoring
watch -n 2 'docker stats stock-ml-api --no-stream'

# Check disk usage
df -h /
du -sh /app/logs /app/data /app/results

# Check memory usage
free -h
```

### 4.3 Restart Services

```bash
# Restart single service
docker-compose restart api

# Restart all services
docker-compose down
docker-compose up -d

# Force rebuild and restart
docker-compose build --no-cache
docker-compose down
docker-compose up -d
```

### 4.4 Update Application

```bash
# 1. Pull latest code
git pull origin main

# 2. Run tests
pytest tests/ -v

# 3. Build new image
docker-compose build

# 4. Deploy with zero downtime
docker-compose up -d

# 5. Verify deployment
curl http://localhost:8000/health
```

### 4.5 Backup and Restore

```bash
# Backup everything
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz data/ results/ logs/ config/

# Backup to S3
aws s3 cp backup.tar.gz s3://bucket/backups/

# Restore from backup
tar -xzf backup.tar.gz

# Verify restoration
ls -la data/ results/
```

---

## 5. Troubleshooting

### 5.1 API Won't Start

**Symptoms**: Container keeps restarting, API not responding

**Investigation**:
```bash
# 1. Check logs
docker-compose logs api

# 2. Check config loading
docker-compose exec api python -c "from stock_ml.api.config import settings; print(settings)"

# 3. Check port availability
lsof -i :8000
```

**Solutions**:
```bash
# Kill process on port 8000
lsof -i :8000 | awk 'NR==2 {print $2}' | xargs kill -9

# Rebuild image
docker-compose build --no-cache

# Full restart
docker-compose down -v
docker-compose up -d
```

### 5.2 High Memory Usage

**Symptoms**: `docker stats` shows memory > 2GB

**Investigation**:
```bash
# Check which files are large
du -sh logs/* | sort -h | tail -5
du -sh data/* | sort -h | tail -5

# Check process memory
docker-compose exec api ps aux --sort=-%mem | head -5
```

**Solutions**:
```bash
# Rotate logs manually
docker-compose exec api python -c \
  "import logging.handlers; h = logging.handlers.RotatingFileHandler('logs/app.log'); h.doRollover()"

# Clear old data
rm -rf data/old_*

# Restart to free memory
docker-compose restart api
```

### 5.3 High CPU Usage

**Symptoms**: `docker stats` shows CPU > 50%

**Investigation**:
```bash
# Check request volume
docker-compose logs api --since 1h | grep "GET\|POST" | wc -l

# Check for slow queries
docker-compose logs api | grep "duration\|slow"

# Top processes
docker-compose exec api top -b -n 1 | head -15
```

**Solutions**:
```bash
# Increase workers (if configured)
# Update docker-compose.yml and rebuild

# Add request rate limiting
# Check rate_limit settings in .env

# Optimize database queries
# Profile slow endpoints with APM tools
```

### 5.4 Disk Space Low

**Symptoms**: `df -h` shows < 1GB free space

**Investigation**:
```bash
# Find large files
find /app -type f -size +100M -exec ls -lh {} \;

# Check log size
du -sh logs/

# Check data size
du -sh data/ results/
```

**Solutions**:
```bash
# Compress old logs
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;

# Archive old data
tar -czf data_archive_$(date +%Y%m%d).tar.gz data/old_*
rm -rf data/old_*

# Move to external storage
aws s3 sync data/ s3://bucket/data/
```

### 5.5 API Returning 500 Errors

**Symptoms**: `curl http://localhost:8000/api/v1/models` returns 500

**Investigation**:
```bash
# Check recent errors
docker-compose logs api --tail 100 | grep -i "error\|traceback"

# Check database connectivity
docker-compose exec api python -c "import sqlite3; sqlite3.connect('stock_ml.db')"

# Check file permissions
docker-compose exec api ls -la data/ results/
```

**Solutions**:
```bash
# Check .env configuration
docker-compose exec api cat .env | head -10

# Recreate data directories
docker-compose exec api mkdir -p data results logs

# Restart API service
docker-compose restart api

# Check logs
docker-compose logs api
```

---

## 6. Performance Optimization

### Response Time Tuning

```bash
# Check endpoint latencies
docker-compose logs api | grep "GET\|POST" | awk '{print $NF}' | sort -n | tail -10

# Identify slow endpoints
docker-compose logs api | grep -E "GET.*[0-9]{3,}ms"
```

### Resource Optimization

```bash
# Limit memory usage in docker-compose.yml
# Add to service:
# mem_limit: 2g
# memswap_limit: 2g

# Limit CPU usage
# cpus: '1.5'
# cpu_shares: 1024
```

### Caching Strategy

- API responses cached at nginx level (headers: `Cache-Control`)
- Model leaderboard cached in application memory
- Feature cache in `stock_ml/src/cache/feature_cache.py`

---

## 7. Monitoring and Alerts

### Health Check Intervals

- **Automatic (Docker)**: Every 30 seconds
- **Manual (Script)**: Run `bash scripts/health_check.sh` every 5 minutes
- **Application**: Logs all requests and responses

### Key Metrics to Watch

1. **API Health**: `GET /health/detailed`
   - `cpu_percent` < 80%
   - `memory_percent` < 80%
   - `uptime_seconds` > 0

2. **Error Rate**: Monitor logs for error count
   - Target: < 1% of requests
   - Alert if: > 5% in 5-minute window

3. **Response Time**: Check request duration logs
   - Target: < 500ms p95
   - Alert if: > 2000ms p95

4. **Disk Usage**: Monitor logs and data directories
   - Alert if: > 80% disk usage

### Setting Up Alerts (Example with cron)

```bash
# Add to crontab
*/5 * * * * bash scripts/health_check.sh || echo "API unhealthy" | mail -s "Alert" ops@company.com
```

---

## 8. Security Maintenance

### Weekly Tasks

```bash
# Update dependencies
pip list --outdated
pip install --upgrade package_name

# Check security vulnerabilities
pip install safety
safety check

# Review access logs
grep "403\|401" logs/app.log | wc -l
```

### Monthly Tasks

```bash
# Rotate API keys (if using authentication)
# Update SSL certificates (if expired soon)
# Review and update firewall rules
# Audit user/service account access
```

### Quarterly Tasks

```bash
# Security vulnerability scan
# Dependency audit
# Access review and cleanup
# Compliance check
```

---

## 9. Disaster Recovery

### Backup Strategy

**Frequency**: Daily  
**Retention**: 30 days  
**Location**: S3 + local backup

```bash
# Create backup
./scripts/backup.sh

# Backup locations
# - Local: /backups/
# - S3: s3://bucket/backups/
# - Archive: /archive/
```

### Recovery Procedure

**Recovery Time Objective (RTO)**: < 1 hour  
**Recovery Point Objective (RPO)**: < 1 hour

```bash
# 1. Stop current services
docker-compose down

# 2. Restore from backup
tar -xzf /backups/backup_YYYYMMDD.tar.gz

# 3. Verify data integrity
ls -la data/ results/
md5sum -c data/checksums.md5

# 4. Start services
docker-compose up -d

# 5. Verify health
curl http://localhost:8000/health

# 6. Run smoke tests
pytest tests/unit/test_health.py -v
```

---

## 10. Contact Information

### On-Call Support

- **Primary**: DevOps Team (devops@company.com)
- **Secondary**: Platform Team (platform@company.com)
- **Escalation**: Engineering Manager (eng-manager@company.com)

### Tools & Access

- **Logs**: Docker logs, Cloudwatch
- **Metrics**: Docker stats, Prometheus
- **Chat**: Slack #stock-ml channel
- **Docs**: This runbook + API.md + DEPLOYMENT.md

### Handoff Checklist

- [ ] Read recent alerts and issues
- [ ] Run morning health check
- [ ] Review error logs from overnight
- [ ] Check disk/memory/CPU usage
- [ ] Verify backup completed successfully

---

## 11. Useful Resources

- **Quick Start**: `QUICKSTART.md`
- **API Documentation**: `docs/API.md`
- **Deployment Guide**: `docs/DEPLOYMENT.md`
- **Configuration**: `.env` file
- **Logs**: `logs/app.log` (JSON format)
- **Code**: `stock_ml/` directory

---

## 12. Change Log

| Date | Change | Owner |
|------|--------|-------|
| 2026-05-30 | Initial runbook created | DevOps Team |
| | Updated for production | Platform Team |

---

**Remember**: When in doubt, check the logs and run `curl http://localhost:8000/health/detailed` first!
