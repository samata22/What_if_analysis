"""
agents/scenario_engine.py

The What-If Scenario Engine.

Responsibilities:
  1. Take base data (QueryResult) + scenario parameters
  2. Apply in-memory mutations (add/remove/modify locations, shift demand, etc.)
  3. Compute delta vs. baseline
  4. Return structured comparison ready for the narrative agent

All mutations happen in-memory on pandas DataFrames — nothing is written to the DB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import pandas as pd
import numpy as np
import structlog

from tools.data_fetcher import QueryResult

log = structlog.get_logger()


# ── Scenario parameter models ─────────────────────────────────────────────────

@dataclass
class ScenarioParams:
    """
    Describes a what-if scenario. Fields are flexible — not all need to be set.

    Example for CPG location optimisation:
        ScenarioParams(
            scenario_name="Close 3 NE locations",
            remove_rows={"region": "Northeast", "status": "underperforming"},
            adjust_columns={"revenue": 1.15},   # +15% revenue for remaining
            add_rows=[{"location": "Atlanta", "region": "Southeast", "revenue": 500000}],
        )
    """
    scenario_name: str = "What-If Scenario"

    # Row-level mutations
    remove_rows: dict[str, Any] | None = None       # filter condition for rows to drop
    add_rows: list[dict[str, Any]] | None = None    # new rows to append
    modify_rows: dict[str, Any] | None = None       # {filter_col: val} rows to modify
    modify_values: dict[str, Any] | None = None     # {col: new_val} applied to matched rows

    # Column-level mutations (applied to entire column)
    adjust_columns: dict[str, float] | None = None  # {col: multiplier}, e.g. {"revenue": 1.1}
    override_columns: dict[str, Any] | None = None  # {col: fixed_value}

    # Custom mutation function (escape hatch for complex scenarios)
    custom_mutator: Callable[[pd.DataFrame], pd.DataFrame] | None = None

    # Metric columns to compare in the delta report
    metric_columns: list[str] = field(default_factory=list)


# ── Delta report ──────────────────────────────────────────────────────────────

@dataclass
class ScenarioDelta:
    scenario_name: str
    baseline_rows: int
    scenario_rows: int
    metrics_baseline: dict[str, float]
    metrics_scenario: dict[str, float]
    metrics_delta: dict[str, float]           # absolute change
    metrics_delta_pct: dict[str, float]       # % change
    baseline_df: pd.DataFrame
    scenario_df: pd.DataFrame
    narrative_hint: str = ""                  # filled in by scenario engine

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "baseline_rows": self.baseline_rows,
            "scenario_rows": self.scenario_rows,
            "row_change": self.scenario_rows - self.baseline_rows,
            "metrics_baseline": self.metrics_baseline,
            "metrics_scenario": self.metrics_scenario,
            "metrics_delta": self.metrics_delta,
            "metrics_delta_pct": self.metrics_delta_pct,
            "narrative_hint": self.narrative_hint,
        }

    def to_summary_text(self) -> str:
        lines = [
            f"SCENARIO: {self.scenario_name}",
            f"Baseline rows: {self.baseline_rows:,}  →  Scenario rows: {self.scenario_rows:,}",
            "",
            "METRIC COMPARISON:",
        ]
        for col in self.metrics_baseline:
            b = self.metrics_baseline[col]
            s = self.metrics_scenario[col]
            d = self.metrics_delta[col]
            dp = self.metrics_delta_pct[col]
            direction = "▲" if d > 0 else ("▼" if d < 0 else "─")
            lines.append(
                f"  {col:<30}  baseline={b:,.2f}  scenario={s:,.2f}  "
                f"delta={d:+,.2f} ({dp:+.1f}%) {direction}"
            )
        if self.narrative_hint:
            lines.append(f"\nKey insight: {self.narrative_hint}")
        return "\n".join(lines)


# ── Engine ────────────────────────────────────────────────────────────────────

class ScenarioEngine:
    """
    Applies what-if mutations to a baseline DataFrame and produces a ScenarioDelta.
    """

    def run(
        self,
        baseline: QueryResult,
        params: ScenarioParams,
    ) -> ScenarioDelta:
        log.info("scenario_engine.start", scenario=params.scenario_name)

        baseline_df = baseline.df.copy()
        scenario_df = baseline_df.copy()

        # ── Apply mutations in order ─────────────────────────────────────
        scenario_df = self._apply_remove_rows(scenario_df, params)
        scenario_df = self._apply_add_rows(scenario_df, params)
        scenario_df = self._apply_modify_rows(scenario_df, params)
        scenario_df = self._apply_adjust_columns(scenario_df, params)
        scenario_df = self._apply_override_columns(scenario_df, params)

        if params.custom_mutator:
            scenario_df = params.custom_mutator(scenario_df)

        # ── Determine metric columns ─────────────────────────────────────
        metric_cols = params.metric_columns or self._auto_detect_metrics(baseline_df)

        # ── Compute delta ────────────────────────────────────────────────
        b_metrics = self._aggregate_metrics(baseline_df, metric_cols)
        s_metrics = self._aggregate_metrics(scenario_df, metric_cols)
        delta, delta_pct = self._compute_delta(b_metrics, s_metrics)

        hint = self._generate_hint(delta_pct, metric_cols)

        result = ScenarioDelta(
            scenario_name=params.scenario_name,
            baseline_rows=len(baseline_df),
            scenario_rows=len(scenario_df),
            metrics_baseline=b_metrics,
            metrics_scenario=s_metrics,
            metrics_delta=delta,
            metrics_delta_pct=delta_pct,
            baseline_df=baseline_df,
            scenario_df=scenario_df,
            narrative_hint=hint,
        )

        log.info("scenario_engine.complete", row_change=result.scenario_rows - result.baseline_rows)
        return result

    # ── Mutation helpers ─────────────────────────────────────────────────

    def _apply_remove_rows(self, df: pd.DataFrame, params: ScenarioParams) -> pd.DataFrame:
        if not params.remove_rows:
            return df
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in params.remove_rows.items():
            if col not in df.columns:
                log.warning("scenario_engine.remove_rows.col_not_found", col=col)
                continue
            if isinstance(val, list):
                mask &= df[col].isin(val)
            else:
                mask &= df[col] == val
        removed = mask.sum()
        log.info("scenario_engine.rows_removed", count=int(removed))
        return df[~mask].reset_index(drop=True)

    def _apply_add_rows(self, df: pd.DataFrame, params: ScenarioParams) -> pd.DataFrame:
        if not params.add_rows:
            return df
        new_rows = pd.DataFrame(params.add_rows)
        log.info("scenario_engine.rows_added", count=len(new_rows))
        return pd.concat([df, new_rows], ignore_index=True)

    def _apply_modify_rows(self, df: pd.DataFrame, params: ScenarioParams) -> pd.DataFrame:
        if not params.modify_rows or not params.modify_values:
            return df
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in params.modify_rows.items():
            if col in df.columns:
                mask &= df[col] == val
        for col, new_val in params.modify_values.items():
            if col in df.columns:
                df.loc[mask, col] = new_val
        log.info("scenario_engine.rows_modified", count=int(mask.sum()))
        return df

    def _apply_adjust_columns(self, df: pd.DataFrame, params: ScenarioParams) -> pd.DataFrame:
        if not params.adjust_columns:
            return df
        for col, multiplier in params.adjust_columns.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col] * multiplier
                log.info("scenario_engine.column_adjusted", col=col, multiplier=multiplier)
        return df

    def _apply_override_columns(self, df: pd.DataFrame, params: ScenarioParams) -> pd.DataFrame:
        if not params.override_columns:
            return df
        for col, val in params.override_columns.items():
            if col in df.columns:
                df[col] = val
        return df

    # ── Metric helpers ───────────────────────────────────────────────────

    def _auto_detect_metrics(self, df: pd.DataFrame) -> list[str]:
        """Falls back to all numeric columns if none specified."""
        return df.select_dtypes(include="number").columns.tolist()

    def _aggregate_metrics(self, df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
        result = {}
        for col in cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                result[col] = float(df[col].sum())
        return result

    def _compute_delta(
        self,
        baseline: dict[str, float],
        scenario: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        delta, delta_pct = {}, {}
        for col in baseline:
            b, s = baseline[col], scenario.get(col, 0.0)
            delta[col] = s - b
            delta_pct[col] = ((s - b) / b * 100) if b != 0 else 0.0
        return delta, delta_pct

    def _generate_hint(self, delta_pct: dict[str, float], metric_cols: list[str]) -> str:
        if not delta_pct:
            return ""
        biggest = max(delta_pct, key=lambda k: abs(delta_pct[k]))
        pct = delta_pct[biggest]
        direction = "increase" if pct > 0 else "decrease"
        return f"Largest impact is on '{biggest}': {abs(pct):.1f}% {direction} under this scenario."


# ── Sensitivity sweep ─────────────────────────────────────────────────────────

def sensitivity_sweep(
    baseline: QueryResult,
    param_name: str,
    column: str,
    multipliers: list[float],
    metric_col: str,
    scenario_name_prefix: str = "Scenario",
) -> list[dict[str, Any]]:
    """
    Runs multiple scenarios varying one column multiplier across a range.
    Useful for 'what happens to revenue if we vary coverage by 10%–50%?'

    Returns a list of dicts with (multiplier, metric_value) for charting.
    """
    engine = ScenarioEngine()
    results = []
    for m in multipliers:
        params = ScenarioParams(
            scenario_name=f"{scenario_name_prefix} {param_name}={m:.0%}",
            adjust_columns={column: m},
            metric_columns=[metric_col],
        )
        delta = engine.run(baseline, params)
        results.append({
            "multiplier": m,
            "label": f"{m:.0%}",
            param_name: m,
            metric_col: delta.metrics_scenario.get(metric_col, 0),
            "delta_pct": delta.metrics_delta_pct.get(metric_col, 0),
        })
    return results
