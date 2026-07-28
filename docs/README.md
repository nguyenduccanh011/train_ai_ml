# Documentation Index

Stock ML Platform - Complete system documentation.

## Getting Started

**New to the project?** Start here:
1. [Quick Start](QUICK_START.md) — Setup trong 5 phút
2. [Development Guide](DEVELOPMENT.md) — Local development
3. [API Documentation](API.md) — API endpoints & examples

## Documentation Map

### Core Documentation
| Document | Purpose |
|----------|---------|
| [Quick Start](QUICK_START.md) | 5-minute setup guide |
| [API Documentation](API.md) | All endpoints, examples, schemas |
| [Portfolio Layer Unification](refactor/PORTFOLIO_LAYER_UNIFICATION.md) | Tầng danh mục Stage-2 thống nhất + golden guard |
| [Research Strategy Map](RESEARCH_STRATEGY_MAP.md) | Bản đồ nghiên cứu: champion, lever đã bác, hướng đi |
| [Troubleshooting](TROUBLESHOOTING.md) | Common issues & solutions |

### Feature Guides
| Document | Purpose |
|----------|---------|
| [Universe Management](UNIVERSE_MANAGEMENT_GUIDE.md) | Tạo & quản lý symbol set |
| [Model Library](MODEL_LIBRARY_GUIDE.md) | Tạo feature set / target / component |
| [Template Submission](TEMPLATE_SUBMISSION_GUIDE.md) | Build & submit strategy template |
| [Component Slots Architecture](guides/COMPONENT_SLOTS_ARCHITECTURE.md) | Dual ML+Rule per decision slot |
| [Component Slots Quick Start](guides/COMPONENT_SLOTS_QUICKSTART.md) | How to create hybrid strategies |
| [Extensibility Guide](EXTENSIBILITY_GUIDE.md) | Extending models, features, targets |
| [Feature Store + DSL Design](FEATURE_STORE_DSL_DESIGN.md) | **(PLANNED)** Expression DSL + per-feature feature store, normalized DB, steps |

## For Developers

| Document | Purpose |
|----------|---------|
| [Development Guide](DEVELOPMENT.md) | Local setup, workflows, debugging |
| [Contributing Guide](CONTRIBUTING.md) | Code standards, testing, PRs |
| [Extensibility Guide](EXTENSIBILITY_GUIDE.md) | How to add models, features, targets |
| [API Documentation](API.md) | Endpoint specs & schemas |

## For DevOps/Deployment

| Document | Purpose |
|----------|---------|
| [Deployment Guide](DEPLOYMENT.md) | Production setup with Docker/Postgres |
| [Troubleshooting](TROUBLESHOOTING.md) | Production issues & monitoring |
| [Quick Start](QUICK_START.md) | Docker setup reference |

## Key Files

```
stock_ml/
├── api/                           # FastAPI application
│   ├── main.py                   # App & routers
│   ├── config.py                 # Settings
│   ├── routes/                   # Endpoints
│   ├── middleware/               # CORS, rate limiting, logging
│   └── schemas/                  # Pydantic models
├── db/
│   ├── models/                   # SQLAlchemy models
│   ├── migrations/               # Alembic migrations
│   └── repositories/             # Data access layer
├── dashboard/                     # HTML/JS UI
│   ├── *.html                   # Pages
│   └── api-config.js            # API configuration
└── scripts/
    ├── seed_universes.py        # Seed universe sets
    └── import_yaml_templates.py # Import YAML templates → DB (feature sets, targets, components)
```

## Quick Reference

### Start Development

```bash
docker-compose up -d
docker-compose logs -f api
```

### Run Tests

```bash
pytest stock_ml/api/tests/ -v
```

### Deploy

```bash
docker-compose -f docker-compose.yml up -d
docker exec -w /app stock-ml-api alembic upgrade head
```

### Check Status

```bash
curl http://localhost/health
docker-compose ps
```

## API Endpoints

**Health Check**
```
GET /health
```

**Universes**
```
GET /api/v1/universes/
POST /api/v1/universes/
```

**Templates**
```
GET /api/v1/templates/
POST /api/v1/templates/
```

**Model Library**
```
GET /api/v1/model-library/feature-sets
GET /api/v1/model-library/targets
GET /api/v1/model-library/components
```

**Feature Store** _(PLANNED — see [Feature Store + DSL Design](FEATURE_STORE_DSL_DESIGN.md))_
```
GET  /api/v1/features/definitions
POST /api/v1/features/validate
GET  /api/v1/features/sets
```

See [API Documentation](API.md) for complete list.

## Key Concepts

### Universe
A set of trading symbols (e.g., "vn_stock_default" = VIC, VNM, etc.)

### Template
Strategy configuration: market, universe, components, signals, etc.

### Component
Reusable model or rule (entry/exit/regime/size slots)

### Target
ML target variable definition (forward return, volatility, etc.)

### Run
Backtest execution result with metrics (PnL, Sharpe, etc.)

## Architecture

```
Internet
   ↓
Nginx (port 80)
   ├→ API Server (port 8000)
   └→ Dashboard (port 9001)
   ↓
   Database (PostgreSQL — container stock-ml-postgres)
```

Kiến trúc backtest/portfolio: xem `stock_ml/README.md` +
[refactor/PORTFOLIO_LAYER_UNIFICATION.md](refactor/PORTFOLIO_LAYER_UNIFICATION.md).

## Support

**Something broken?**
1. Check [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Review logs: `docker-compose logs api`
3. Check health: `curl http://localhost/health/detailed`

**Want to contribute?**
Read [Contributing Guide](CONTRIBUTING.md)

**Need more details?**
Check [Development Guide](DEVELOPMENT.md)

---

**Last Updated**: 2026-05-31
**Version**: 1.0.0
