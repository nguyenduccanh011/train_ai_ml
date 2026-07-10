# Development Guide

Local development setup và workflows.

## Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Git
- VS Code (recommended)

## Local Setup (Without Docker)

### 1. Clone Repository

```bash
git clone <repo>
cd train_ai_ml
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
# or
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements-api.txt
pip install -r requirements-ml.txt    # For ML development
```

### 4. Setup Database

```bash
# Run migrations
alembic upgrade head

# Seed data (optional)
python -m stock_ml.scripts.seed_universes
```

### 5. Start API

```bash
python -m uvicorn stock_ml.api.main:app --reload --port 8000
```

API will be at `http://localhost:8000`

## Docker Development

### Start Services

```bash
docker-compose up -d
```

### View Logs

```bash
docker-compose logs -f api
docker-compose logs -f dashboard
```

### Run Commands in Container

```bash
# Migrations
docker exec -w /app stock-ml-api alembic upgrade head

# Seed data
docker exec -w /app stock-ml-api python -m stock_ml.scripts.seed_universes

# Python shell
docker exec -it stock-ml-api python
```

## Database Development

### Migrations

Create new migration:

```bash
alembic revision --autogenerate -m "Description of change"
```

Review migration in `stock_ml/db/migrations/versions/`

Run migrations:

```bash
alembic upgrade head
```

Rollback:

```bash
alembic downgrade -1
```

### Database Schema

View schema files:
- [Models](../stock_ml/db/models/)
  - `run.py` — Leaderboard runs
  - `template.py` — Strategy templates, components, catalogs
  - `universe.py` — Universe sets and symbols
  - `trade.py` — Trade execution records

## API Development

### File Structure

```
stock_ml/api/
├── main.py                    # FastAPI app, routers
├── config.py                  # Settings
├── middleware/
│   ├── rate_limiter.py
│   ├── logging_middleware.py
│   └── error_handler.py
├── routes/
│   ├── health.py
│   ├── models.py
│   ├── templates.py
│   ├── universes.py
│   ├── model_components.py
│   └── ...
├── schemas/
│   └── models.py              # Pydantic schemas
└── tests/
    └── test_*.py
```

### Add New Endpoint

1. Create route in `stock_ml/api/routes/`:

```python
# stock_ml/api/routes/my_feature.py
from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from stock_ml.db.dependencies import get_db

router = APIRouter(prefix="/api/my-feature", tags=["my-feature"])

@router.get("/")
async def list_my_feature(session: AsyncSession = Depends(get_db)):
    """Get my feature data."""
    # Implementation
    return {}
```

2. Include router in `stock_ml/api/main.py`:

```python
from . import routes
app.include_router(routes.my_feature.router, tags=["my-feature"])
```

3. Add schema if needed in `stock_ml/api/schemas/`:

```python
from pydantic import BaseModel

class MyFeatureResponse(BaseModel):
    id: int
    name: str
    # ...
```

### Testing

```bash
# Run all tests
pytest stock_ml/api/tests/ -v

# Run specific test
pytest stock_ml/api/tests/test_health.py -v

# With coverage
pytest --cov=stock_ml.api stock_ml/api/tests/
```

## Dashboard Development

### File Structure

```
stock_ml/dashboard/
├── *.html              # Page files
├── api-config.js       # API configuration
├── js/
│   ├── nav.js
│   └── ...
└── results/            # Linked from results/
```

### Add New Page

1. Create HTML file:

```html
<!-- stock_ml/dashboard/my-page.html -->
<!DOCTYPE html>
<html>
<head>
    <title>My Page</title>
    <script src="api-config.js"></script>
</head>
<body>
    <script>
        const API_BASE = window.API_CONFIG?.baseUrl || '/api/v1';
        
        async function init() {
            const data = await fetch(`${API_BASE}/my-feature/`)
                .then(r => r.json());
            console.log(data);
        }
        
        init();
    </script>
</body>
</html>
```

2. Access at:
   - http://localhost:9001/my-page.html (direct)
   - http://localhost/my-page.html (via Nginx)

### API Configuration

Dashboard auto-detects API base URL:

```javascript
// api-config.js
window.API_CONFIG = {
    baseUrl: '/api/v1',                    // All environments
}
```

## Code Standards

### Python

- Format: Black
- Type hints: Required for function signatures
- Async: Use AsyncSession for database access
- Logging: Use Python logging, not print

### JavaScript

- No framework constraints
- Use fetch API, not XMLHttpRequest
- Vanilla JS + HTML preferred

### Commits

Format:

```
<type>: <description>

<optional body>

- <optional footer>
```

Types:
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code restructuring
- `docs:` Documentation
- `test:` Tests
- `chore:` Maintenance

Example:

```
feat: Add universe CRUD endpoints

Implements create, read, update endpoints for universe management.
- POST /api/v1/universes/
- GET /api/v1/universes/{slug}
- PUT /api/v1/universes/{slug}

Closes #123
```

## Debugging

### API Logs

```bash
docker-compose logs -f api | grep ERROR
```

### Database Debugging

Connect to SQLite:

```bash
docker exec -it stock-ml-api sqlite3 /app/results/leaderboard.db
```

Query:

```sql
SELECT * FROM strategy_templates LIMIT 5;
SELECT COUNT(*) FROM leaderboard_runs;
```

### API Testing

```bash
# Health check
curl http://localhost/health

# Get templates
curl http://localhost/api/v1/templates/

# Create template
curl -X POST http://localhost/api/v1/templates/ \
  -H "Content-Type: application/json" \
  -d '{"name":"test",...}'
```

## Performance

### Database Optimization

- Migrations create indexes automatically
- Add new indexes in migration files if needed:

```python
op.create_index("idx_templates_market", "strategy_templates", ["market"])
```

### API Caching

Currently no response caching. Add with Redis if needed.

### Asset Optimization

Dashboard assets served as-is. Gzip handled by Nginx.

## Troubleshooting

### Import Errors

Ensure `stock_ml` is on Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Database Locked

SQLite locks when concurrent writes occur:

```bash
# Solution: use PostgreSQL for production
DATABASE_URL=postgresql+asyncpg://...
```

### CORS Errors

Add origin to `stock_ml/api/config.py`:

```python
cors_origins: list[str] = [
    "http://localhost:9001",
    # ...
]
```

## Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLAlchemy Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [Alembic Migrations](https://alembic.sqlalchemy.org/)
- [Docker Compose](https://docs.docker.com/compose/)
