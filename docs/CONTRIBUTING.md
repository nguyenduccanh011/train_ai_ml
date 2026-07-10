# Contributing Guide

Code standards và workflow.

## Getting Started

1. Fork the repository
2. Create feature branch: `git checkout -b feat/my-feature`
3. Make changes
4. Test locally
5. Commit: `git commit -m "feat: description"`
6. Push: `git push origin feat/my-feature`
7. Open Pull Request

## Code Standards

### Python

**Format**: Black
```bash
black stock_ml/
```

**Linting**: Ruff
```bash
ruff check stock_ml/
```

**Type hints**: Required
```python
def process_data(data: list[str]) -> dict[str, int]:
    """Process data and return results."""
    return {}
```

**Async/Await**: Use for I/O
```python
async def fetch_data(session: AsyncSession) -> list:
    result = await session.execute(query)
    return result.scalars().all()
```

### JavaScript

- No frameworks required
- Use fetch API
- Vanilla JS preferred
- Event-driven where appropriate

### Commits

Format:
```
<type>: <description>

Optional longer description explaining the change.

- Optional footer line 1
- Optional footer 2
```

Types:
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code restructuring (no behavior change)
- `docs:` Documentation
- `test:` Add/modify tests
- `chore:` Maintenance, dependencies

Examples:
```
feat: Add universe CRUD endpoints
fix: Handle CORS for port 9001
docs: Add API documentation
test: Add test for rate limiter
```

## Testing

### Run Tests

```bash
# All tests
pytest stock_ml/api/tests/ -v

# Specific test
pytest stock_ml/api/tests/test_health.py::test_health_check -v

# With coverage
pytest --cov=stock_ml.api stock_ml/api/tests/
```

### Write Tests

```python
# stock_ml/api/tests/test_my_feature.py
import pytest
from fastapi.testclient import TestClient
from stock_ml.api.main import app

client = TestClient(app)

def test_my_endpoint():
    response = client.get("/api/my-feature/")
    assert response.status_code == 200
    assert "data" in response.json()
```

### Test Database Changes

```python
@pytest.fixture
async def test_db():
    """Create test database session."""
    engine = create_async_engine("sqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    SessionLocal = sessionmaker(engine, class_=AsyncSession)
    async with SessionLocal() as session:
        yield session
```

## Documentation

### Code Comments

Only comment the WHY, not WHAT:

```python
# Good: explains why this is necessary
# SQLite has concurrent write issues, use FOR UPDATE lock
cursor.execute("SELECT * FROM templates FOR UPDATE")

# Bad: just repeats the code
# Query all templates
templates = session.query(Template).all()
```

### Docstrings

Required for public functions:

```python
async def list_templates(
    market: str | None = None,
    session: AsyncSession = Depends(get_db)
) -> list[dict]:
    """List strategy templates.
    
    Args:
        market: Filter by market (optional)
        session: Database session
        
    Returns:
        List of template dicts
    """
```

### File Headers

Add to new Python files:

```python
"""Module description."""

from __future__ import annotations
```

## Database Changes

### Adding a Column

1. Create migration:
   ```bash
   alembic revision --autogenerate -m "Add new_column to templates"
   ```

2. Review generated migration in `stock_ml/db/migrations/versions/`

3. Test locally:
   ```bash
   alembic upgrade head
   ```

### Adding a Table

Define model first, then migrate:

```python
# stock_ml/db/models/my_model.py
from sqlalchemy import Column, Integer, String
from stock_ml.db.base import Base

class MyModel(Base):
    __tablename__ = "my_models"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
```

Then:

```bash
alembic revision --autogenerate -m "Add my_models table"
```

## API Endpoints

### Adding Endpoint

1. Create/modify route file in `stock_ml/api/routes/`
2. Include router in `stock_ml/api/main.py`
3. Add Pydantic schema if needed
4. Add tests
5. Document in [API.md](API.md)

### Endpoint Checklist

- [ ] Route decorated with HTTP method
- [ ] Async function (for I/O)
- [ ] Type hints on parameters and return
- [ ] Docstring explaining endpoint
- [ ] Error handling for edge cases
- [ ] Tests covering happy path + errors
- [ ] Documentation updated

Example:

```python
@router.get("/templates/", response_model=list[TemplateResponse])
async def list_templates(
    market: str | None = None,
    session: AsyncSession = Depends(get_db)
) -> list[TemplateResponse]:
    """List templates, optionally filtered by market.
    
    Returns empty list if no templates found.
    """
    repo = StrategyTemplateRepository(session)
    templates = (
        await repo.list_by_market(market) 
        if market 
        else await repo.list_all(is_active=True)
    )
    return [_to_response(t) for t in templates]
```

## Dashboard Components

### HTML Best Practices

- Keep HTML semantic (use `<button>`, `<nav>`, etc.)
- Use `api-config.js` for API base URL
- Load CSS/JS from relative paths
- Add event listeners, not inline handlers

Good:
```html
<button id="load-btn">Load Templates</button>
<script>
    document.getElementById('load-btn').addEventListener('click', async () => {
        const data = await fetch(`${API_BASE}/templates/`);
    });
</script>
```

Bad:
```html
<button onclick="loadTemplates()">Load Templates</button>
```

## Performance Considerations

### Database

- Use indexes for frequently queried columns
- Lazy-load relationships when needed
- Avoid N+1 queries

### API

- Cache static responses (if not real-time)
- Paginate large result sets
- Validate input early

### Dashboard

- Lazy-load large lists
- Debounce search inputs
- Cache API responses in localStorage

## Security

### Input Validation

Always validate and sanitize:

```python
from pydantic import BaseModel, Field

class CreateTemplate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    market: str = Field(..., pattern="^[a-z_]+$")
```

### SQL Injection Prevention

Use ORM or parameterized queries:

```python
# Good: ORM
await session.execute(
    select(Template).where(Template.market == market)
)

# Bad: String concatenation
query = f"SELECT * FROM templates WHERE market = '{market}'"
```

### CORS

Whitelist specific origins, never use wildcard in production.

## Dependency Updates

```bash
# Check for updates
pip list --outdated

# Update package
pip install --upgrade package-name

# Test after update
pytest stock_ml/api/tests/

# Commit with explanation
git commit -m "chore: Update package-name to 2.0"
```

## Code Review

### For Reviewers

- Is it correct? (Does it do what it claims?)
- Is it understandable?
- Are there tests?
- Performance impact?
- Security issues?

### For Authors

- Keep PRs small and focused
- Write clear commit messages
- Respond to feedback promptly
- Run tests before requesting review

## Release Process

1. Update version in `stock_ml/api/config.py`
2. Create release branch: `git checkout -b release/v1.2.0`
3. Update changelog
4. Run full test suite
5. Merge to `main`
6. Create git tag: `git tag -a v1.2.0 -m "Release 1.2.0"`
7. Push tag: `git push origin v1.2.0`
8. Build Docker image: `docker build -t stockml:1.2.0 .`
9. Push to registry

## Questions?

Check:
- [API Documentation](API.md)
- [Development Guide](DEVELOPMENT.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
