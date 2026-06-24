"""
api/main.py
FastAPI application — exposes the What-If Analysis agent via REST endpoints.

Endpoints:
  GET  /health          — DB connectivity + schema check
  GET  /schema          — Return discovered schema metadata
  POST /analyze         — Main analysis endpoint (lookup + what-if)
  POST /sensitivity     — Sweep one parameter across a range
  POST /schema/refresh  — Invalidate schema cache and rediscover
"""
from __future__ import annotations

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config.settings import get_settings
from tools.db_connection import test_connection
from tools.schema_discovery import get_cached_schema, invalidate_schema_cache
from tools.data_fetcher import execute_query_as_callback
from agents.scenario_engine import sensitivity_sweep
from agents.sql_agent import generate_sql_with_retry
from graph.orchestrator import run_agent
from api.models import (
    AnalysisRequest,
    AnalysisResponse,
    SensitivityRequest,
    SensitivityResponse,
    SensitivityPoint,
    ScenarioDeltaResponse,
    MetricComparison,
    HealthResponse,
)

log = structlog.get_logger()
settings = get_settings()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api.startup", host=settings.api_host, port=settings.api_port)
    # Warm up schema cache on startup
    try:
        get_cached_schema()
        log.info("api.schema_warmed_up")
    except Exception as e:
        log.warning("api.schema_warmup_failed", error=str(e))
    yield
    log.info("api.shutdown")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="What-If Analysis Agent",
    description="Agentic system for natural language data analysis and what-if scenario modelling.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check DB connectivity and schema availability."""
    db_ok = test_connection()
    try:
        schema = get_cached_schema()
        table_count = len(schema.tables)
    except Exception:
        table_count = 0

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db_connected=db_ok,
        schema_tables=table_count,
    )


@app.get("/schema", tags=["System"])
def get_schema(schemas: str = ""):
    """
    Return the current schema catalog.
    Pass ?schemas=dbo,cpg to filter to specific schemas.
    """
    try:
        catalog = get_cached_schema(schemas)
        return {"tables": catalog.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/schema/refresh", tags=["System"])
def refresh_schema():
    """Invalidate the schema cache and trigger a fresh discovery."""
    invalidate_schema_cache()
    try:
        catalog = get_cached_schema()
        return {"status": "refreshed", "table_count": len(catalog.tables)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
def analyze(request: AnalysisRequest):
    """
    Main analysis endpoint. Accepts a natural language question with an
    optional what-if scenario block and returns a full analytical answer.

    Examples:
    - Simple: { "question": "What is revenue by region?" }
    - What-if: { "question": "Impact of closing NE locations?",
                 "scenario_params": { "remove_rows": {"region": "Northeast"},
                                      "metric_columns": ["revenue"] } }
    """
    log.info("api.analyze.request", question=request.question[:80])

    # Convert Pydantic model to plain dict for the graph
    scenario_dict = request.scenario_params.model_dump() if request.scenario_params else None

    try:
        state = run_agent(
            question=request.question,
            scenario_params=scenario_dict,
            target_schemas=request.target_schemas,
        )
    except Exception as e:
        log.error("api.analyze.unhandled_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    # Build scenario delta response if available
    scenario_response = None
    if state.scenario_delta:
        d = state.scenario_delta
        scenario_response = ScenarioDeltaResponse(
            scenario_name=d.scenario_name,
            baseline_rows=d.baseline_rows,
            scenario_rows=d.scenario_rows,
            row_change=d.scenario_rows - d.baseline_rows,
            metrics=MetricComparison(
                baseline=d.metrics_baseline,
                scenario=d.metrics_scenario,
                delta=d.metrics_delta,
                delta_pct=d.metrics_delta_pct,
            ),
            narrative_hint=d.narrative_hint,
        )

    return AnalysisResponse(
        question=request.question,
        answer=state.final_answer,
        is_whatif=state.is_whatif,
        sql_used=state.generated_sql,
        sql_attempts=state.sql_attempts,
        row_count=state.query_result.row_count if state.query_result else None,
        data_truncated=state.query_result.truncated if state.query_result else False,
        scenario_delta=scenario_response,
        error=state.error_message or state.fetch_error,
    )


@app.post("/sensitivity", response_model=SensitivityResponse, tags=["Analysis"])
def sensitivity_analysis(request: SensitivityRequest):
    """
    Sweep one column multiplier across a range and see the effect on a metric.

    Example: Vary 'budget' from 70%–130% and track 'coverage_pct'.
    """
    log.info("api.sensitivity.request", question=request.question[:80], column=request.column)

    # Step 1: fetch baseline data
    schemas_key = ",".join(request.target_schemas)
    schema = get_cached_schema(schemas_key)

    result_dict = generate_sql_with_retry(
        question=request.question,
        schema=schema,
        execute_fn=execute_query_as_callback,
    )

    if not result_dict["success"] or not result_dict.get("result"):
        raise HTTPException(
            status_code=422,
            detail=result_dict.get("error", "Could not fetch baseline data.")
        )

    baseline = result_dict["result"]

    # Step 2: run sensitivity sweep
    try:
        sweep_results = sensitivity_sweep(
            baseline=baseline,
            param_name=request.column,
            column=request.column,
            multipliers=request.multipliers,
            metric_col=request.metric_col,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sensitivity sweep failed: {e}")

    points = [
        SensitivityPoint(
            multiplier=r["multiplier"],
            label=r["label"],
            metric_value=r[request.metric_col],
            delta_pct=r["delta_pct"],
        )
        for r in sweep_results
    ]

    return SensitivityResponse(
        question=request.question,
        column=request.column,
        metric_col=request.metric_col,
        points=points,
    )
