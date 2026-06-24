"""
agents/sql_agent.py

Uses Claude to translate a natural language question + schema context
into a safe, executable T-SQL SELECT query.

Includes:
  - Prompt construction with full schema injection
  - Response parsing
  - Self-correction loop (re-prompts with the error on failure)
"""
from __future__ import annotations

import re
import structlog
from anthropic import Anthropic

from config.settings import get_settings
from tools.schema_discovery import SchemaCatalog

log = structlog.get_logger()
settings = get_settings()
client = Anthropic(api_key=settings.anthropic_api_key)


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert T-SQL query writer for Microsoft SQL Server.

RULES — follow these exactly:
1. Generate only a single SELECT statement. Never write INSERT, UPDATE, DELETE, DROP, ALTER, or any DDL/DML.
2. Use only tables and columns that exist in the schema provided.
3. Always qualify table names with schema prefix: [schema].[table].
4. For string comparisons use LIKE with % wildcards unless an exact match is needed.
5. Add TOP {max_rows} to every query to prevent runaway result sets.
6. When a question involves aggregation, always include a meaningful ORDER BY.
7. Return ONLY the raw SQL query — no explanation, no markdown code fences, no comments.
8. If the question cannot be answered from the available schema, reply with exactly:
   CANNOT_ANSWER: <one sentence reason>
"""

_USER_PROMPT = """DATABASE SCHEMA:
{schema}

USER QUESTION:
{question}

{scenario_context}

Write the T-SQL SELECT query."""

_CORRECTION_PROMPT = """The previous query failed with this error:
{error}

Previous query:
{previous_sql}

Fix the query and return only the corrected SQL. Same rules apply."""


# ── Core generation function ──────────────────────────────────────────────────

def generate_sql(
    question: str,
    schema: SchemaCatalog,
    scenario_params: dict | None = None,
    previous_error: str | None = None,
    previous_sql: str | None = None,
) -> tuple[str, bool]:
    """
    Generates a T-SQL query for the given question.

    Args:
        question:        Natural language question from the user.
        schema:          SchemaCatalog from schema discovery.
        scenario_params: Optional what-if parameters to inject into the prompt.
        previous_error:  If retrying after a failure, the SQL error message.
        previous_sql:    The failed SQL from the previous attempt.

    Returns:
        (sql_query, is_answerable) — if not answerable, sql_query contains the reason.
    """
    schema_text = schema.to_prompt_text()
    scenario_context = _build_scenario_context(scenario_params)

    system = _SYSTEM_PROMPT.format(max_rows=settings.max_rows_returned)

    if previous_error and previous_sql:
        # Self-correction mode
        user_content = _CORRECTION_PROMPT.format(
            error=previous_error,
            previous_sql=previous_sql,
        )
    else:
        user_content = _USER_PROMPT.format(
            schema=schema_text,
            question=question,
            scenario_context=scenario_context,
        )

    log.info("sql_agent.generating", question=question[:80], retry=bool(previous_error))

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )

    raw = response.content[0].text.strip()

    # Check for unanswerable signal
    if raw.startswith("CANNOT_ANSWER:"):
        reason = raw.replace("CANNOT_ANSWER:", "").strip()
        log.warning("sql_agent.cannot_answer", reason=reason)
        return reason, False

    # Strip any accidental markdown fences
    sql = _clean_sql(raw)
    log.info("sql_agent.generated", sql_preview=sql[:120])
    return sql, True


def _build_scenario_context(params: dict | None) -> str:
    if not params:
        return ""
    lines = ["WHAT-IF SCENARIO PARAMETERS (incorporate these into the query):"]
    for k, v in params.items():
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines)


def _clean_sql(raw: str) -> str:
    """Remove markdown fences and leading/trailing whitespace."""
    # Strip ```sql ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


# ── Retry wrapper (used by the orchestrator graph) ───────────────────────────

def generate_sql_with_retry(
    question: str,
    schema: SchemaCatalog,
    scenario_params: dict | None = None,
    execute_fn=None,          # callable: sql -> (result, error_str | None)
) -> dict:
    """
    Attempts SQL generation + execution up to max_sql_retries times.
    On execution failure, sends the error back to Claude for self-correction.

    Returns a dict with keys:
        sql        : final SQL string
        success    : bool
        error      : error message if all retries failed
        attempts   : number of attempts made
    """
    sql, answerable = generate_sql(question, schema, scenario_params)

    if not answerable:
        return {"sql": None, "success": False, "error": sql, "attempts": 0}

    if execute_fn is None:
        return {"sql": sql, "success": True, "error": None, "attempts": 1}

    previous_sql = None
    previous_error = None

    for attempt in range(1, settings.max_sql_retries + 1):
        if attempt > 1:
            # Self-correct with the previous error
            sql, answerable = generate_sql(
                question, schema, scenario_params,
                previous_error=previous_error,
                previous_sql=previous_sql,
            )
            if not answerable:
                return {"sql": None, "success": False, "error": sql, "attempts": attempt}

        result, error = execute_fn(sql)

        if error is None:
            log.info("sql_agent.success", attempts=attempt)
            return {"sql": sql, "success": True, "result": result, "error": None, "attempts": attempt}

        log.warning("sql_agent.execution_error", attempt=attempt, error=error[:200])
        previous_sql = sql
        previous_error = error

    return {
        "sql": sql,
        "success": False,
        "error": f"Failed after {settings.max_sql_retries} attempts. Last error: {previous_error}",
        "attempts": settings.max_sql_retries,
    }
