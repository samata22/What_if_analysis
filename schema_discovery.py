"""
tools/schema_discovery.py

Discovers and caches the DB schema (tables, columns, PKs, FKs, row counts).
This metadata is injected into every SQL generation prompt so the LLM
knows exactly what it can query.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Any

from sqlalchemy import text
import structlog

from tools.db_connection import get_engine

log = structlog.get_logger()


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ColumnMeta:
    name: str
    data_type: str
    nullable: bool
    max_length: int | None = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: str | None = None   # "other_table.other_column"


@dataclass
class TableMeta:
    schema: str
    name: str
    row_count: int = 0
    columns: list[ColumnMeta] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"[{self.schema}].[{self.name}]"


@dataclass
class SchemaCatalog:
    tables: list[TableMeta] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """
        Renders the schema as a compact text block suitable for LLM prompts.
        Example output:
            [dbo].[locations] (1234 rows)
              - location_id  INT  PK
              - region       NVARCHAR(100)
              - revenue      DECIMAL(18,2)
        """
        lines: list[str] = []
        for t in self.tables:
            lines.append(f"\n{t.full_name}  ({t.row_count:,} rows)")
            for c in t.columns:
                flags = []
                if c.is_primary_key:
                    flags.append("PK")
                if c.is_foreign_key:
                    flags.append(f"FK→{c.references}")
                flag_str = "  " + ", ".join(flags) if flags else ""
                nullable_str = "" if c.nullable else "  NOT NULL"
                lines.append(f"  - {c.name:<35} {c.data_type}{nullable_str}{flag_str}")
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self.tables]


# ── Discovery queries (MSSQL / T-SQL) ────────────────────────────────────────

_COLUMNS_SQL = text("""
SELECT
    s.name                          AS schema_name,
    t.name                          AS table_name,
    c.name                          AS column_name,
    tp.name                         AS data_type,
    c.is_nullable                   AS is_nullable,
    c.max_length                    AS max_length,
    CASE WHEN pk.column_id IS NOT NULL THEN 1 ELSE 0 END AS is_primary_key,
    CASE WHEN fk.parent_column_id IS NOT NULL THEN 1 ELSE 0 END AS is_foreign_key,
    OBJECT_NAME(fk.referenced_object_id) AS fk_ref_table,
    fk_col.name                     AS fk_ref_column
FROM
    sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    JOIN sys.columns c ON c.object_id = t.object_id
    JOIN sys.types tp ON c.user_type_id = tp.user_type_id
    -- Primary keys
    LEFT JOIN (
        SELECT ic.object_id, ic.column_id
        FROM sys.index_columns ic
        JOIN sys.indexes i ON ic.object_id = i.object_id AND ic.index_id = i.index_id
        WHERE i.is_primary_key = 1
    ) pk ON pk.object_id = c.object_id AND pk.column_id = c.column_id
    -- Foreign keys
    LEFT JOIN sys.foreign_key_columns fk
        ON fk.parent_object_id = c.object_id AND fk.parent_column_id = c.column_id
    LEFT JOIN sys.columns fk_col
        ON fk_col.object_id = fk.referenced_object_id AND fk_col.column_id = fk.referenced_column_id
WHERE
    t.is_ms_shipped = 0   -- exclude system tables
ORDER BY
    s.name, t.name, c.column_id
""")

_ROW_COUNT_SQL = text("""
SELECT
    s.name   AS schema_name,
    t.name   AS table_name,
    p.rows   AS row_count
FROM
    sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    JOIN sys.partitions p ON p.object_id = t.object_id AND p.index_id IN (0, 1)
WHERE
    t.is_ms_shipped = 0
""")


# ── Main discovery function ───────────────────────────────────────────────────

def discover_schema(schemas: list[str] | None = None) -> SchemaCatalog:
    """
    Introspects the connected MSSQL database and returns a SchemaCatalog.

    Args:
        schemas: Optional list of schema names to restrict to (e.g. ['dbo', 'cpg']).
                 If None, all non-system schemas are included.
    """
    log.info("schema_discovery.start")
    engine = get_engine()
    catalog = SchemaCatalog()
    tables: dict[str, TableMeta] = {}

    with engine.connect() as conn:
        # ── Row counts ────────────────────────────────────────────────────
        row_counts: dict[str, int] = {}
        for row in conn.execute(_ROW_COUNT_SQL):
            key = f"{row.schema_name}.{row.table_name}"
            row_counts[key] = int(row.row_count)

        # ── Columns + keys ────────────────────────────────────────────────
        for row in conn.execute(_COLUMNS_SQL):
            if schemas and row.schema_name not in schemas:
                continue

            key = f"{row.schema_name}.{row.table_name}"
            if key not in tables:
                tables[key] = TableMeta(
                    schema=row.schema_name,
                    name=row.table_name,
                    row_count=row_counts.get(key, 0),
                )

            ref = None
            if row.is_foreign_key and row.fk_ref_table:
                ref = f"{row.fk_ref_table}.{row.fk_ref_column}"

            tables[key].columns.append(ColumnMeta(
                name=row.column_name,
                data_type=row.data_type,
                nullable=bool(row.is_nullable),
                max_length=row.max_length if row.max_length != -1 else None,
                is_primary_key=bool(row.is_primary_key),
                is_foreign_key=bool(row.is_foreign_key),
                references=ref,
            ))

    catalog.tables = list(tables.values())
    log.info("schema_discovery.complete", table_count=len(catalog.tables))
    return catalog


# ── Cached version (refreshed per process start) ──────────────────────────────

@lru_cache(maxsize=1)
def get_cached_schema(schemas_key: str = "") -> SchemaCatalog:
    """
    Returns a cached schema catalog.
    Pass schemas_key as a comma-separated string of schema names to filter,
    e.g. get_cached_schema("dbo,cpg").
    """
    schemas = [s.strip() for s in schemas_key.split(",") if s.strip()] or None
    return discover_schema(schemas)


def invalidate_schema_cache() -> None:
    """Call this if tables are added/modified and cache needs refresh."""
    get_cached_schema.cache_clear()
    log.info("schema_discovery.cache_cleared")
