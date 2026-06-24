"""
api/models.py
Pydantic models for FastAPI request and response contracts.
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class ScenarioParamsRequest(BaseModel):
    """
    Optional what-if mutation block.
    Include only the fields relevant to your scenario.

    Example — CPG location optimisation:
    {
        "scenario_name": "Close underperforming NE locations",
        "remove_rows":    {"region": "Northeast", "status": "underperforming"},
        "adjust_columns": {"budget": 1.20},
        "metric_columns": ["revenue", "coverage_pct", "budget"]
    }
    """
    scenario_name: str = "What-If Scenario"
    remove_rows:      dict[str, Any] | None = None
    add_rows:         list[dict[str, Any]] | None = None
    modify_rows:      dict[str, Any] | None = None
    modify_values:    dict[str, Any] | None = None
    adjust_columns:   dict[str, float] | None = None
    override_columns: dict[str, Any] | None = None
    metric_columns:   list[str] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    """
    Main request body for /analyze.

    Simple lookup:
        { "question": "What is total revenue by region?" }

    What-if:
        {
            "question": "What happens to coverage if we close NE locations?",
            "scenario_params": {
                "scenario_name": "Close NE locations",
                "remove_rows": {"region": "Northeast"},
                "metric_columns": ["coverage_pct", "revenue"]
            },
            "target_schemas": ["dbo"]
        }
    """
    question: str = Field(..., min_length=5, description="Natural language question")
    scenario_params: ScenarioParamsRequest | None = None
    target_schemas: list[str] = Field(
        default_factory=list,
        description="DB schemas to restrict to (empty = all schemas)"
    )


class SensitivityRequest(BaseModel):
    """Request body for /sensitivity — sweeps one column multiplier over a range."""
    question: str
    column: str = Field(..., description="Column to vary, e.g. 'revenue'")
    metric_col: str = Field(..., description="Metric to track, e.g. 'coverage_pct'")
    multipliers: list[float] = Field(
        default=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        description="Multiplier values to sweep over"
    )
    target_schemas: list[str] = Field(default_factory=list)


# ── Response ──────────────────────────────────────────────────────────────────

class MetricComparison(BaseModel):
    baseline: dict[str, float]
    scenario: dict[str, float]
    delta:    dict[str, float]
    delta_pct: dict[str, float]


class ScenarioDeltaResponse(BaseModel):
    scenario_name:  str
    baseline_rows:  int
    scenario_rows:  int
    row_change:     int
    metrics:        MetricComparison
    narrative_hint: str


class AnalysisResponse(BaseModel):
    question:        str
    answer:          str
    is_whatif:       bool
    sql_used:        str | None
    sql_attempts:    int
    row_count:       int | None
    data_truncated:  bool
    scenario_delta:  ScenarioDeltaResponse | None
    error:           str | None


class SensitivityPoint(BaseModel):
    multiplier: float
    label: str
    metric_value: float
    delta_pct: float


class SensitivityResponse(BaseModel):
    question:   str
    column:     str
    metric_col: str
    points:     list[SensitivityPoint]


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    schema_tables: int
