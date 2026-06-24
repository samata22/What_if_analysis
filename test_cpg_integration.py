"""
tests/test_cpg_integration.py
End-to-end integration test for the CPG location strategy use case.
Mocks the DB layer so no live connection is needed.

Run with: pytest tests/test_cpg_integration.py -v
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from tools.data_fetcher import QueryResult
from agents.scenario_engine import ScenarioEngine, ScenarioParams


# ── Shared CPG dataset ────────────────────────────────────────────────────────

CPG_DATA = pd.DataFrame({
    "location_id":   [1, 2, 3, 4, 5, 6, 7, 8],
    "location_name": ["Boston","Hartford","Albany","Atlanta","Charlotte","Dallas","Denver","Phoenix"],
    "region":        ["Northeast","Northeast","Northeast","Southeast","Southeast","South","West","West"],
    "status":        ["active","active","underperforming","active","active","active","active","active"],
    "revenue":       [1_200_000, 980_000, 320_000, 1_500_000, 1_100_000, 1_800_000, 950_000, 1_050_000],
    "coverage_pct":  [88.0, 82.0, 61.0, 91.0, 89.0, 93.0, 85.0, 87.0],
    "headcount":     [45, 38, 22, 55, 42, 60, 35, 40],
    "cost_per_loc":  [280_000, 260_000, 210_000, 310_000, 275_000, 330_000, 255_000, 265_000],
})

BASELINE = QueryResult(df=CPG_DATA.copy(), sql="SELECT * FROM [dbo].[locations]", row_count=8)
ENGINE   = ScenarioEngine()


# ── Scenario 1: Close underperforming NE ─────────────────────────────────────

def test_scenario_close_underperforming_northeast():
    """
    Business question:
    'What happens to total revenue and coverage if we close underperforming NE locations?'
    """
    params = ScenarioParams(
        scenario_name="Close underperforming NE locations",
        remove_rows={"region": "Northeast", "status": "underperforming"},
        metric_columns=["revenue", "coverage_pct", "headcount", "cost_per_loc"],
    )
    delta = ENGINE.run(BASELINE, params)

    # Albany removed (only underperforming NE location)
    assert delta.scenario_rows == 7

    # Revenue drops by Albany's amount
    assert abs(delta.metrics_delta["revenue"] - (-320_000)) < 1.0

    # Cost also drops (good)
    assert delta.metrics_delta["cost_per_loc"] < 0

    print("\n" + delta.to_summary_text())


# ── Scenario 2: Full NE closure + SE reinvestment ─────────────────────────────

def test_scenario_ne_closure_se_expansion():
    """
    Business question:
    'If we close all NE locations and invest in 2 new SE locations,
     what's the net revenue and coverage impact?'
    """
    params = ScenarioParams(
        scenario_name="Close NE, Expand SE",
        remove_rows={"region": "Northeast"},
        add_rows=[
            {
                "location_id": 9, "location_name": "Miami", "region": "Southeast",
                "status": "active", "revenue": 1_400_000, "coverage_pct": 92.0,
                "headcount": 52, "cost_per_loc": 300_000,
            },
            {
                "location_id": 10, "location_name": "Orlando", "region": "Southeast",
                "status": "active", "revenue": 1_200_000, "coverage_pct": 89.0,
                "headcount": 45, "cost_per_loc": 285_000,
            },
        ],
        metric_columns=["revenue", "coverage_pct", "headcount", "cost_per_loc"],
    )
    delta = ENGINE.run(BASELINE, params)

    # 8 - 3 NE + 2 new = 7 locations
    assert delta.scenario_rows == 7

    # NE revenue lost: 1.2M + 0.98M + 0.32M = 2.5M
    # SE revenue gained: 1.4M + 1.2M = 2.6M → net positive
    assert delta.metrics_delta["revenue"] > 0, "Net revenue should be positive after SE expansion"

    print("\n" + delta.to_summary_text())


# ── Scenario 3: Budget reallocation with demand growth ───────────────────────

def test_scenario_demand_growth_west():
    """
    Business question:
    'If demand grows 25% in the West, what's the revenue uplift?'
    """
    def west_uplift(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df["region"] == "West", "revenue"] *= 1.25
        df.loc[df["region"] == "West", "coverage_pct"] = df.loc[
            df["region"] == "West", "coverage_pct"
        ].clip(upper=100.0)
        return df

    params = ScenarioParams(
        scenario_name="25% West demand growth",
        custom_mutator=west_uplift,
        metric_columns=["revenue", "coverage_pct"],
    )
    delta = ENGINE.run(BASELINE, params)

    # Denver (950k) + Phoenix (1.05M) × 25% uplift = +500k
    expected_delta = (950_000 + 1_050_000) * 0.25
    assert abs(delta.metrics_delta["revenue"] - expected_delta) < 1.0
    assert delta.metrics_delta_pct["revenue"] > 0

    print("\n" + delta.to_summary_text())


# ── Scenario 4: Headcount optimisation ───────────────────────────────────────

def test_scenario_headcount_reduction():
    """
    Business question:
    'If we reduce headcount by 10% across all locations, what's the cost impact?'
    """
    params = ScenarioParams(
        scenario_name="10% headcount reduction",
        adjust_columns={"headcount": 0.90, "cost_per_loc": 0.92},  # cost drops slightly less
        metric_columns=["headcount", "cost_per_loc", "revenue"],
    )
    delta = ENGINE.run(BASELINE, params)

    # Headcount should drop by ~10%
    assert abs(delta.metrics_delta_pct["headcount"] - (-10.0)) < 0.1
    # Revenue unchanged
    assert delta.metrics_delta["revenue"] == 0.0

    print("\n" + delta.to_summary_text())


# ── Scenario 5: Multi-step optimisation ──────────────────────────────────────

def test_scenario_full_optimisation():
    """
    Business question:
    'Optimise our portfolio: close underperformers, open best-fit new location,
     apply 10% revenue growth assumption to retained locations.'
    """
    params = ScenarioParams(
        scenario_name="Full portfolio optimisation",
        remove_rows={"status": "underperforming"},
        add_rows=[{
            "location_id": 9, "location_name": "Nashville", "region": "Southeast",
            "status": "active", "revenue": 1_250_000, "coverage_pct": 90.0,
            "headcount": 46, "cost_per_loc": 290_000,
        }],
        adjust_columns={"revenue": 1.10},
        metric_columns=["revenue", "coverage_pct", "cost_per_loc"],
    )
    delta = ENGINE.run(BASELINE, params)

    # Removed Albany + added Nashville = still 8 locations
    assert delta.scenario_rows == 8

    # Net revenue should be positive (removed low performer, added strong one, +10% on rest)
    assert delta.metrics_delta["revenue"] > 0

    # Coverage should improve (removed low-coverage Albany at 61%)
    assert delta.metrics_delta["coverage_pct"] > 0

    print("\n" + delta.to_summary_text())
