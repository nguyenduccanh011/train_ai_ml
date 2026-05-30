# Deployment Guide

## Quick Start

### Local Development

```bash
# 1. Clone and setup
cd stock_ml
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt -r requirements-api.txt

# 3. Configure environment
cp .env.example .env
# Edit .env as needed for your environment

# 4. Run API
python -m uvicorn stock_ml.api.main:app --reload

# 5. Run tests
pytest tests/ -v
```

### Docker Development

```bash
# Build and run with docker-compose
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f api

# Run tests inside container
docker-compose exec api pytest tests/ -v

# Stop services
docker-compose down
```

---

## Production Deployment

### System Requirements

- Docker & Docker Compose
- Python 3.12+ (for local development)
- 4GB RAM minimum
- 10GB disk space

### Pre-Deployment Checklist

- [ ] All tests pass locally (`pytest tests/ -v`)
- [ ] Environment variables configured (`.env` file)
- [ ] SSL certificates ready (if using HTTPS)
- [ ] Database migrated (if using database)
- [ ] Logs directory writable
- [ ] Health checks responding

---

## Docker Swarm Deployment

### 1. Initialize Swarm

```bash
docker swarm init
```

### 2. Create Config and Secrets

```bash
# Create environment config
docker config create stock_ml_env .env

# Create logging config
docker config create stock_ml_logging config/logging.yaml
```

### 3. Deploy Stack

```bash
# Deploy full stack
docker stack deploy -c docker-compose.yml stock_ml

# Verify services
docker stack services stock_ml
docker service ls
```

### 4. Monitor Deployment

```bash
# View service status
docker service ps stock_ml_api

# View logs
docker service logs -f stock_ml_api

# Scale services
docker service scale stock_ml_api=3
```

---

## Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl create namespace stock-ml
```

### 2. Create ConfigMaps and Secrets

```bash
# Logging config
kubectl create configmap stock-ml-logging \
  --from-file=config/logging.yaml \
  -n stock-ml

# Environment variables
kubectl create secret generic stock-ml-env \
  --from-file=.env \
  -n stock-ml
```

### 3. Deploy Application

```bash
# Apply Kubernetes manifests
kubectl apply -f infrastructure/k8s/namespace.yaml
kubectl apply -f infrastructure/k8s/configmap.yaml
kubectl apply -f infrastructure/k8s/deployment.yaml
kubectl apply -f infrastructure/k8s/service.yaml
```

### 4. Verify Deployment

```bash
# Check pods
kubectl get pods -n stock-ml

# Check services
kubectl get services -n stock-ml

# View logs
kubectl logs -f deployment/stock-ml-api -n stock-ml
```

---

## Configuration

### Environment Variables

Key variables in `.env`:

```bash
# App
APP_NAME="Stock ML Platform"
APP_VERSION="1.0.0"
DEBUG=False

# Server
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/api/v1

# Paths
DATA_DIR=./data
RESULTS_DIR=./results
LOG_LEVEL=INFO

# CORS (whitelist specific origins)
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Rate Limiting
RATE_LIMIT_ENABLED=True
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```

### Logging Configuration

Located in `config/logging.yaml`:
- Console output (DEBUG level)
- File output (INFO level, JSON format)
- Rotating file handler (10MB max, 5 backups)
- Per-module configuration

### Health Checks

The application includes built-in health checks:

```bash
# Simple health check
curl http://localhost:8000/health

# Detailed health metrics
curl http://localhost:8000/health/detailed

# Automated script
bash scripts/health_check.sh
```

---

## Testing

### Run All Tests

```bash
pytest tests/ -v --tb=short
```

### Run Specific Test Suite

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v
```

### Generate Coverage Report

```bash
pytest tests/ --cov=stock_ml --cov-report=html
# Open htmlcov/index.html in browser
```

---

## Monitoring

### View Docker Logs

```bash
# Real-time logs
docker-compose logs -f api

# Last 50 lines
docker-compose logs api --tail 50

# View all service logs
docker-compose logs

# Filter logs by service
docker-compose logs nginx
```

### Monitoring Endpoints

```bash
# API health
GET /health
GET /health/detailed

# Response example
{
  "status": "healthy",
  "timestamp": "2026-05-30T12:00:00Z",
  "version": "1.0.0",
  "cpu_percent": 2.5,
  "memory_percent": 15.3,
  "uptime_seconds": 3600
}
```

---

## SSL/TLS Configuration

### Self-Signed Certificate (Development)

```bash
# Generate certificate
openssl req -x509 -newkey rsa:4096 -nodes \
  -out infrastructure/nginx/ssl/cert.pem \
  -keyout infrastructure/nginx/ssl/key.pem \
  -days 365

# Use in Nginx
server {
  listen 443 ssl;
  ssl_certificate /etc/nginx/ssl/cert.pem;
  ssl_certificate_key /etc/nginx/ssl/key.pem;
}
```

### Production Certificate (Let's Encrypt)

```bash
# Using Certbot with Docker
docker run -it --rm -v $(pwd)/infrastructure/nginx/ssl:/etc/letsencrypt \
  certbot/certbot certonly --standalone \
  -d api.example.com

# Update Nginx configuration
# Use /etc/letsencrypt/live/api.example.com/fullchain.pem
```

### Nginx SSL Configuration

Edit `infrastructure/nginx/nginx.conf`:

```nginx
# Redirect HTTP to HTTPS
server {
  listen 80;
  server_name api.example.com;
  return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
  listen 443 ssl;
  server_name api.example.com;
  
  ssl_certificate /etc/nginx/ssl/cert.pem;
  ssl_certificate_key /etc/nginx/ssl/key.pem;
  ssl_protocols TLSv1.2 TLSv1.3;
  ssl_ciphers HIGH:!aNULL:!MD5;
  
  # Proxy API
  location /api {
    proxy_pass http://api:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }
}
```

---

## Database Migration

### Initialize Database

```bash
# 1. Start PostgreSQL
docker-compose up -d postgres

# 2. Wait for DB to be ready
sleep 10

# 3. Run migrations
alembic upgrade head

# 4. Verify migration
psql -U stockml -d stockml -c "\dt"
```

### Create New Migration

```bash
# Auto-generate migration
alembic revision --autogenerate -m "Add new column"

# Verify generated file
cat stock_ml/db/migrations/versions/*.py

# Apply migration
alembic upgrade head
```

### Seed Database

```bash
# Create initial data
python -c "from stock_ml.db.seed import create_seed_data; create_seed_data()"

# Or use provided seed script
python stock_ml/scripts/seed_leaderboard.py
```

---

## Performance Tuning

### API Optimization

```bash
# Use production ASGI server (gunicorn)
gunicorn stock_ml.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Database Connection Pooling

In `stock_ml/db/engine.py`:

```python
engine = create_async_engine(
  DATABASE_URL,
  echo=False,
  pool_size=20,
  max_overflow=40,
  pool_pre_ping=True,
  pool_recycle=3600
)
```

### Cache Configuration

Enable Redis caching (optional):

```bash
# Start Redis
docker-compose up -d redis

# Configure in .env
REDIS_URL=redis://redis:6379

# Use in code
from stock_ml.api.utils.cache import cached
```

---

## Backup and Recovery

### Backup Database

```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump \
  -U stockml stockml > backup.sql

# Restore from backup
docker-compose exec postgres psql \
  -U stockml stockml < backup.sql
```

### Backup Files

```bash
# Backup results and config
tar -czf backup.tar.gz results/ config/ stock_ml/

# Store in secure location
aws s3 cp backup.tar.gz s3://my-bucket/backups/
```

---

## Troubleshooting

### 1. API won't start

```bash
# Check logs
docker-compose logs api

# Verify Python version
python --version  # Must be 3.11+

# Check dependencies
pip install -r requirements.txt -r requirements-api.txt
```

### 2. Database connection fails

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Test connection
psql -U stockml -h localhost -d stockml -c "SELECT 1"

# Or disable database in .env
DB_ENABLED=false
```

### 3. Port conflicts

```bash
# Find process using port
lsof -i :8000

# Use different port
API_PORT=8001

# Update docker-compose.yml if needed
```

### 4. CORS errors

```bash
# Check browser console for detailed error
# Update .env CORS_ORIGINS to include your domain
CORS_ORIGINS=["http://localhost:8080","https://example.com"]

# Restart API
docker-compose restart api
```

### 5. Rate limiting blocks legitimate traffic

```bash
# Increase limits in .env
RATE_LIMIT_ENABLED=false  # Only for development

# Or adjust in code
stock_ml/api/middleware/rate_limiter.py
```

---

## Security Hardening

### Production Checklist

- [ ] Enable HTTPS/SSL (use Let's Encrypt)
- [ ] Add API key authentication
- [ ] Implement JWT token validation
- [ ] Set strong database passwords
- [ ] Enable firewall rules
- [ ] Use secrets management (AWS Secrets Manager, etc.)
- [ ] Enable database backups
- [ ] Configure logging and monitoring
- [ ] Set up intrusion detection
- [ ] Regular security updates

### Secrets Management

```bash
# Don't commit .env to git
echo ".env" >> .gitignore

# Use environment variables in production
export DATABASE_URL="postgresql://..."
export API_KEY="secret_key"

# Or use AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id stock-ml-api
```

---

## Scaling

### Horizontal Scaling

```bash
# Scale API service (Docker Swarm)
docker service scale stock_ml_api=3

# Load balancing (Nginx)
upstream api {
  server api1:8000;
  server api2:8000;
  server api3:8000;
}
```

### Vertical Scaling

```bash
# Increase container resources
docker update --memory 4G api
docker update --cpus 2 api
```

### Database Scaling

```bash
# Read replicas
# Primary: PostgreSQL master
# Replicas: Standby servers for read-only queries

# Implement read/write splitting
# Write to master, read from replicas
```

---

## Maintenance

### Regular Tasks

- **Daily**: Monitor logs, check health
- **Weekly**: Backup database, review metrics
- **Monthly**: Update dependencies, security patches
- **Quarterly**: Major version upgrades, disaster recovery drill

### Update Process

```bash
# 1. Pull latest code
git pull origin main

# 2. Update dependencies
pip install -r requirements.txt -r requirements-api.txt --upgrade

# 3. Run migrations
alembic upgrade head

# 4. Run tests
pytest tests/ -v

# 5. Restart services
docker-compose down
docker-compose build
docker-compose up -d

# 6. Verify health
curl http://localhost:8000/health
```

---

## Support and Documentation

- **API Documentation**: `/api/docs` (Swagger UI)
- **Source Code**: `stock_ml/` directory
- **Config Files**: `config/` directory
- **Logs**: `logs/app.log`
- **Issues**: Check `docs/` folder
- **GitHub**: For version control and CI/CD

# Show timestamps
docker-compose logs api -t
```

### Monitor System Resources

```bash
# Check container resource usage
docker stats stock-ml-api

# Check disk usage
du -sh logs/
du -sh data/

# Monitor processes
ps aux | grep uvicorn
```

### Health Monitoring

```bash
# Continuous health checks
watch -n 5 'curl -s http://localhost:8000/health'

# Log health metrics
while true; do
  curl -s http://localhost:8000/health/detailed | jq .
  sleep 60
done
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs api

# Rebuild image
docker-compose build --no-cache

# Restart services
docker-compose down
docker-compose up -d
```

### Port Already in Use

```bash
# Check what's using port 8000
lsof -i :8000

# Use different port
docker-compose up -d --publish 8001:8000
```

### Out of Memory

```bash
# Check memory usage
docker stats

# Increase Docker memory limit in settings
# For Mac/Windows: Docker Desktop → Preferences → Resources → Memory

# For Linux: No limit, check system RAM
```

### Application Logging Issues

```bash
# Check log file permissions
ls -la logs/

# Check log file size
du -h logs/app.log

# Rotate logs manually
docker-compose exec api \
  python -c "import logging.handlers; \
  h = logging.handlers.RotatingFileHandler('logs/app.log'); \
  h.doRollover()"
```

---

## Performance Optimization

### API Performance

- Enable uvicorn workers: `--workers 4`
- Use production ASGI server (gunicorn, hypercorn)
- Enable gzip compression in nginx

### Database Performance

- Add indexing for frequently queried fields
- Implement caching layer (Redis)
- Use connection pooling

### System Performance

- Use SSD for logs/data storage
- Enable Docker resource limits
- Use load balancer (nginx, HAProxy)

---

## Security Hardening

### For Production

1. **Enable Authentication**
   - Add API key or JWT validation
   - Implement rate limiting per user

2. **Enable HTTPS**
   - Generate SSL certificates
   - Update nginx config with SSL

3. **Secure Environment**
   - Use secrets manager (Vault, AWS Secrets Manager)
   - Don't commit .env to version control
   - Rotate API keys regularly

4. **Network Security**
   - Use private networks for services
   - Implement firewall rules
   - Use VPN for remote access

### Example: Enable API Key Authentication

```python
# In stock_ml/api/config.py
API_KEY_ENABLED = True
API_KEY = "your-secret-key-here"  # Use environment variable

# In routes, check the header:
from fastapi import Header, HTTPException

@router.get("/models")
async def list_models(x_api_key: str = Header(...)):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of endpoint
```

---

## Backup and Recovery

### Backup Data

```bash
# Backup data directory
tar -czf backup_data.tar.gz data/
tar -czf backup_results.tar.gz results/

# Backup logs
tar -czf backup_logs.tar.gz logs/

# Backup to S3
aws s3 cp backup_data.tar.gz s3://bucket/backups/
```

### Recovery

```bash
# Restore from backup
tar -xzf backup_data.tar.gz

# Verify restoration
ls -la data/

# Restart services
docker-compose down
docker-compose up -d
```

---

## Scaling

### Horizontal Scaling

```bash
# With Docker Swarm
docker service scale stock_ml_api=3

# With Kubernetes
kubectl scale deployment stock-ml-api --replicas=3 -n stock-ml

# With Docker Compose (using load balancer)
# Uncomment load balancer service in docker-compose.yml
docker-compose up -d
```

### Load Balancing

Nginx is already configured as reverse proxy in `infrastructure/nginx/nginx.conf`.

---

## Maintenance

### Regular Tasks

- **Weekly**: Review logs for errors
- **Monthly**: Update dependencies, run security scans
- **Quarterly**: Backup data, test recovery procedures
- **Annually**: Review and update architecture

### Update Procedure

```bash
# 1. Pull latest code
git pull origin main

# 2. Rebuild Docker images
docker-compose build --no-cache

# 3. Run tests
pytest tests/ -v

# 4. Deploy with zero downtime (rolling update)
docker-compose up -d

# 5. Verify health
curl http://localhost:8000/health
```

---

## Support

- **Documentation**: See `docs/API.md`
- **Health Check**: `GET /health/detailed`
- **API Docs**: `GET /api/docs`
- **Logs**: `docker-compose logs -f api`
- **Issues**: Check application logs and system resources
