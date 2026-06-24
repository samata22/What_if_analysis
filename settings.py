"""
config/settings.py
Central configuration — loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Anthropic ──────────────────────────────────────────────────────────
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"

    # ── SQL Server (MSSQL) ─────────────────────────────────────────────────
    db_server: str          # e.g. "my-server.database.windows.net"
    db_name: str
    db_user: str
    db_password: str
    db_driver: str = "ODBC Driver 18 for SQL Server"
    db_port: int = 1433

    # ── Agent behaviour ────────────────────────────────────────────────────
    max_sql_retries: int = 3          # self-correction attempts
    max_rows_returned: int = 5000     # safety cap on query results
    query_timeout_seconds: int = 30

    # ── API ────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
