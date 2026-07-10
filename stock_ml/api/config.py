"""FastAPI Configuration - Production-ready settings"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "Stock ML Platform"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # Data paths (resolve relative to project root)
    _root_dir: Path = Path(__file__).parent.parent.parent  # Points to project root
    data_dir: Path = _root_dir / "data"
    results_dir: Path = _root_dir / "results"
    config_dir: Path = _root_dir / "config"

    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("./logs/app.log")

    # CORS - whitelist specific origins (FIX: not wildcard)
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:9001",  # Dashboard direct port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:9001",
    ]

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds

    # Database — Postgres is the metadata source-of-truth. Override via DATABASE_URL
    # in .env (default targets the docker postgres published on host port 5433).
    # SQLite is kept only for test fixtures, which set their own URL.
    database_url: str = "postgresql+asyncpg://stockml:stockml_dev@localhost:5433/stockml"
    db_enabled: bool = True

    # OHLCV source-of-truth: DuckDB OLAP store (read-only). Override via
    # STOCK_DATA_DIR in .env. Default resolves to <repo>/market_data/market.duckdb,
    # matching stock_ml/config/markets/vn_stock.yaml.
    stock_data_dir: str = str(_root_dir / "market_data" / "market.duckdb")

    # Security
    api_key_enabled: bool = False
    api_key: str = ""  # Set in .env for production

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.log_file.parent.mkdir(exist_ok=True, parents=True)


settings = Settings()
