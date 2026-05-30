"""FastAPI Main Application - Production-ready configuration"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
import logging.config
from pathlib import Path
import yaml

from .config import settings
from .middleware.error_handler import error_exception_handler
from .middleware.logging_middleware import LoggingMiddleware
from .middleware.rate_limiter import limiter, rate_limit_error_handler
from . import routes

# Configure logging from YAML or fallback to basicConfig
logging_config_path = Path(__file__).parent.parent.parent / "config" / "logging.yaml"
if logging_config_path.exists():
    with open(logging_config_path) as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
else:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(settings.log_file),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Attach limiter to app for use in routes
app.state.limiter = limiter

# Add CORS middleware (FIX: whitelist specific origins, not wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add exception handlers
app.add_exception_handler(Exception, error_exception_handler)
app.add_exception_handler(Exception, rate_limit_error_handler)

# Include routers
app.include_router(routes.health.router, tags=["health"])
app.include_router(routes.models.router, prefix=settings.api_prefix, tags=["models"])
app.include_router(routes.leaderboard.router, prefix=settings.api_prefix, tags=["leaderboard"])
app.include_router(routes.runs.router, tags=["runs"])
app.include_router(routes.jobs.router, tags=["jobs"])
app.include_router(routes.experiments.router, tags=["experiments"])

# Mount static files (dashboard) if exists
dashboard_dir = Path(__file__).parent.parent / "dashboard" / "build"
if dashboard_dir.exists():
    app.mount("/", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")
    logger.info(f"Dashboard mounted at /")


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Rate limiting: {'enabled' if settings.rate_limit_enabled else 'disabled'}")
    logger.info(f"CORS origins: {settings.cors_origins}")

    if settings.db_enabled:
        try:
            from sqlalchemy import text
            from stock_ml.db.engine import async_engine
            async with async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection verified")
        except Exception as e:
            logger.error(f"Database connection failed: {e}. Running without DB.")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info(f"Shutting down {settings.app_name}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )
