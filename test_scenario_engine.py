"""
tests/test_scenario_engine.py
Unit tests for the ScenarioEngine — no DB required (uses in-memory DataFrames).
"""
import pytest
import pandas as pd

from tools.data_fetcher import QueryResult
from agents.scenario_engine import ScenarioEngine, ScenarioParams, sensitivity_sweep


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_cpg_baseline() -> QueryResult:
    """
    Simulates the baseline CPG location dataset as it would come from a DB query.
    Models: location_id, region, status, revenue, coverage_pct, headcount
    """
    data = {
        "location_id":  [1, 2, 3, 4, 5, 6, 7, 8],
        "location_name":["Boston","Hartford","Albany","Atlanta","Charlotte","Dallas","Denver","Phoenix"],
        "region":       ["Northeast","Northeast","Northeast","Southeast","Southeast","South","West","West"],
        "status":       ["active","active","underperforming","active","active","active","active","active"],
        "revenue":      [1_200_000, 980_000, 320_000, 1_500_000, 1_100_000, 1_800_000, 950_000, 1_050_000],
        "coverage_pct": [88.0, 82.0, 61.0, 91.0, 89.0, 93.0, 85.0, 87.0],
        "headcount":    [45, 38, 22, 55, 42, 60, 35, 40],
    }
    df = pd.DataFrame(data)
    return QueryResult(df=df, sql="SELECT * FROM locations", row_count=len(df))


# ── Tests: row mutations ──────────────────────────────────────────────────────

class TestRemoveRows:
    def test_remove_by_region(self):
        baseline = make_cpg_baseline()
        params = ScenarioParams(
            scenario_name="Close NE locations",
            remove_rows={"region": "Northeast"},
            metric_columns=["revenue", "coverage_pct"],
        )
        delta = ScenarioEngine().run(baseline, params)

        assert delta.scenario_rows == 5          # 8 total - 3 NE
        assert delta.baseline_rows == 8
        assert delta.metrics_delta["revenue"] < 0   # revenue drops

    def test_remove_by_status(self):
        baseline = make_cpg_baseline()
        params = ScenarioParams(
            scenario_name="Remove underperforming",
            remove_rows={"status": "underperforming"},
            metric_columns=["revenue"],
        )
        delta = ScenarioEngine().run(baseline, params)

        assert delta.scenario_rows == 7          # 1 underperforming removed
        assert delta.metrics_delta["revenue"] < 0

    def test_remove_nonexistent_column_is_safe(self):
        baseline = make_cpg_baseline()
        params = ScenarioParams(
            remove_rows={"nonexistent_col": "value"},
            metric_columns=["revenue"],
        )
        delta = ScenarioEngine().run(baseline, params)
        # No rows should be removed — engine warns and skips
        assert delta.scenario_rows == delta.baseline_rows


class TestAddRows:
    def test_add_new_location(self):
        baseline = make_cpg_baseline()
        params = ScenarioParams(
            scenario_name="Open Miami location",
            add_rows=[{
                "location_id": 9, "location_name": "Miami", "region": "Southeast",
                "status": "active", "revenue": 1_300_000, "coverage_pct": 90.0, "headcount": 48,
            }],
            metric_columns=["revenue", "coverage_pct"],
        )
        delta = ScenarioEngine().run(baseline, params)

        assert delta.scenario_rows == 9
        assert delta.metrics_delta["revenue"] > 0


class TestAdjustColumns:
    def test_revenue_uplift(self):
        baseline = make_cpg_baseline()
        params = ScenarioParams(
            scenario_name="20% revenue uplift",
            adjust_columns={"revenue": 1.20},
            metric_columns=["revenue"],
        )
        delta = ScenarioEngine().run(baseline, params)

        expected_base = baseline.df["revenue"].sum()
        assert abs(delta.metrics_scenario["revenue"] - expected_base * 1.20) < 1.0
        assert abs(delta.metrics_delta_pct["revenue"] - 20.0) < 0.1


# ── Tests: composite what-if ──────────────────────────────────────────────────

class TestCompositeScenario:
    def test_close_ne_and_open_southeast(self):
        """
        CPG scenario: Close all NE locations, open 2 new SE locations,
        and apply 15% revenue uplift to remaining South/West locations.
        """
        baseline = make_cpg_baseline()
        params = ScenarioParams(
            scenario_name="NE closure + SE expansion",
            remove_rows={"region": "Northeast"},
            add_rows=[
                {"location_id": 9,  "location_name": "Miami",   "region": "Southeast",
                 "status": "active", "revenue": 1_300_000, "coverage_pct": 91.0, "headcount": 50},
                {"location_id": 10, "location_name": "Orlando", "region": "Southeast",
                 "status": "active", "revenue": 1_100_000, "coverage_pct": 88.0, "headcount": 42},
            ],
            adjust_columns={"revenue": 1.15},
            metric_columns=["revenue", "coverage_pct", "headcount"],
        )
        delta = ScenarioEngine().run(baseline, params)

        # 8 - 3 NE + 2 new = 7
        assert delta.scenario_rows == 7
        # Revenue should be net positive after expansion + uplift
        assert delta.metrics_scenario["revenue"] > delta.metrics_baseline["revenue"]


# ── Tests: sensitivity sweep ──────────────────────────────────────────────────

class TestSensitivitySweep:
    def test_sweep_produces_correct_count(self):
        baseline = make_cpg_baseline()
        multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
        results = sensitivity_sweep(
            baseline=baseline,
            param_name="budget_factor",
            column="revenue",
            multipliers=multipliers,
            metric_col="revenue",
        )
        assert len(results) == len(multipliers)

    def test_sweep_monotonically_increases(self):
        baseline = make_cpg_baseline()
        multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
        results = sensitivity_sweep(
            baseline=baseline,
            param_name="budget_factor",
            column="revenue",
            multipliers=multipliers,
            metric_col="revenue",
        )
        values = [r["revenue"] for r in results]
        assert values == sorted(values), "Revenue should increase as multiplier increases"

    def test_baseline_multiplier_has_zero_delta(self):
        baseline = make_cpg_baseline()
        results = sensitivity_sweep(
            baseline=baseline,
            param_name="factor",
            column="revenue",
            multipliers=[1.0],
            metric_col="revenue",
        )
        assert abs(results[0]["delta_pct"]) < 0.01


# ── Tests: delta report ───────────────────────────────────────────────────────

class TestDeltaReport:
    def test_summary_text_contains_metric(self):
        baseline = make_cpg_baseline()
        params = ScenarioParams(
            remove_rows={"region": "Northeast"},
            metric_columns=["revenue"],
        )
        delta = ScenarioEngine().run(baseline, params)
        summary = delta.to_summary_text()
        assert "revenue" in summary
        assert "baseline" in summary.lower()

    def test_to_dict_serializable(self):
        baseline = make_cpg_baseline()
        params = ScenarioParams(metric_columns=["revenue"])
        delta = ScenarioEngine().run(baseline, params)
        d = delta.to_dict()
        assert isinstance(d, dict)
        assert "metrics_baseline" in d
        assert "metrics_scenario" in d
