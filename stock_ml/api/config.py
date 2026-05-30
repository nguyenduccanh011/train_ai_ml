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
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds

    # Database
    database_url: str = "postgresql+asyncpg://stockml:stockml@localhost:5432/stockml"
    db_enabled: bool = False

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
