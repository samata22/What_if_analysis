"""
tools/db_connection.py
Manages the SQL Server connection pool via SQLAlchemy.
Provides a single engine instance reused across the app.
"""
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from functools import lru_cache
import structlog

from config.settings import get_settings

log = structlog.get_logger()


def _build_connection_url() -> str:
    s = get_settings()
    # SQLAlchemy MSSQL connection string using pyodbc
    return (
        f"mssql+pyodbc://{s.db_user}:{s.db_password}"
        f"@{s.db_server}:{s.db_port}/{s.db_name}"
        f"?driver={s.db_driver.replace(' ', '+')}"
        f"&TrustServerCertificate=yes"
        f"&timeout={s.query_timeout_seconds}"
    )


@lru_cache
def get_engine() -> Engine:
    """
    Returns a cached SQLAlchemy Engine with connection pooling.
    Called once; reused for the lifetime of the application.
    """
    url = _build_connection_url()
    engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,       # validate connections before use
        pool_recycle=1800,        # recycle connections every 30 min
        echo=False,               # set True for SQL debug logging
    )

    # Log on first successful connect
    @event.listens_for(engine, "connect")
    def on_connect(dbapi_conn, conn_record):
        log.info("db.connected", server=get_settings().db_server)

    return engine


def test_connection() -> bool:
    """Quick health-check — returns True if DB is reachable."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.error("db.connection_failed", error=str(e))
        return False
