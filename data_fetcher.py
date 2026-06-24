"""
tools/data_fetcher.py

Executes SQL queries against MSSQL and returns results as
pandas DataFrames + serializable dicts for downstream agents.
"""
from __future__ import annotations

from typing import Any
import pandas as pd
from sqlalchemy import text
import structlog

from tools.db_connection import get_engine
from config.settings import get_settings

log = structlog.get_logger()
settings = get_settings()


class QueryResult:
    """Wraps a query result with convenience methods."""

    def __init__(self, df: pd.DataFrame, sql: str, row_count: int):
        self.df = df
        self.sql = sql
        self.row_count = row_count
        self.columns = list(df.columns)
        self.truncated = row_count >= settings.max_rows_returned

    def to_dict(self) -> dict[str, Any]:
        """Serializable representation for API responses."""
        return {
            "columns": self.columns,
            "rows": self.df.to_dict(orient="records"),
            "row_count": self.row_count,
            "truncated": self.truncated,
            "sql": self.sql,
        }

    def to_summary_text(self) -> str:
        """
        Compact text summary injected into LLM prompts for analysis.
        Avoids sending thousands of rows to the model.
        """
        if self.df.empty:
            return "Query returned no rows."

        lines = [f"Query returned {self.row_count:,} rows."]
        if self.truncated:
            lines.append(f"⚠ Results truncated at {settings.max_rows_returned:,} rows.")

        lines.append(f"Columns: {', '.join(self.columns)}")

        # Numeric summary for numeric columns
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            lines.append("\nNumeric summary:")
            desc = self.df[numeric_cols].describe().round(2)
            lines.append(desc.to_string())

        # First 10 rows preview
        lines.append(f"\nFirst {min(10, len(self.df))} rows:")
        lines.append(self.df.head(10).to_string(index=False))

        return "\n".join(lines)


def execute_query(sql: str) -> tuple[QueryResult | None, str | None]:
    """
    Executes a SQL SELECT query.

    Returns:
        (QueryResult, None)         on success
        (None, error_message_str)   on failure
    """
    log.info("data_fetcher.executing", sql_preview=sql[:120])
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)

        result = QueryResult(df=df, sql=sql, row_count=len(df))
        log.info("data_fetcher.success", rows=result.row_count, cols=len(result.columns))
        return result, None

    except Exception as e:
        error_msg = str(e)
        log.error("data_fetcher.failed", error=error_msg[:300])
        return None, error_msg


def execute_query_as_callback(sql: str) -> tuple[Any, str | None]:
    """
    Adapter so data_fetcher can be passed as execute_fn to sql_agent.
    Returns (QueryResult | None, error_str | None).
    """
    result, error = execute_query(sql)
    return result, error
