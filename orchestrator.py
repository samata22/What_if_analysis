"""
graph/orchestrator.py
Wires all nodes into a LangGraph StateGraph with conditional routing.

Flow:
  load_schema
      ↓
  classify_intent
      ↓
  generate_sql
      ↓
  execute_query  ←─── self-correction loop (handled inside node)
      ↓
  [if whatif] → run_scenario → generate_narrative
  [if lookup] → generate_narrative
      ↓
  END

Any node that sets state.error_message or state.fetch_error routes to handle_error.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
    node_load_schema,
    node_classify_intent,
    node_generate_sql,
    node_execute_query,
    node_run_scenario,
    node_generate_narrative,
    node_handle_error,
)


# ── Edge conditions ───────────────────────────────────────────────────────────

def route_after_schema(state: AgentState) -> str:
    if state.has_error():
        return "handle_error"
    return "classify_intent"


def route_after_classify(state: AgentState) -> str:
    if state.has_error():
        return "handle_error"
    return "generate_sql"


def route_after_sql(state: AgentState) -> str:
    if state.has_error() or not state.sql_answerable:
        return "handle_error"
    return "execute_query"


def route_after_execute(state: AgentState) -> str:
    if state.has_error():
        return "handle_error"
    if state.is_whatif and state.scenario_params:
        return "run_scenario"
    return "generate_narrative"


def route_after_scenario(state: AgentState) -> str:
    if state.has_error():
        return "handle_error"
    return "generate_narrative"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("load_schema",        node_load_schema)
    graph.add_node("classify_intent",    node_classify_intent)
    graph.add_node("generate_sql",       node_generate_sql)
    graph.add_node("execute_query",      node_execute_query)
    graph.add_node("run_scenario",       node_run_scenario)
    graph.add_node("generate_narrative", node_generate_narrative)
    graph.add_node("handle_error",       node_handle_error)

    # Entry point
    graph.set_entry_point("load_schema")

    # Conditional edges
    graph.add_conditional_edges("load_schema",     route_after_schema,   {"classify_intent": "classify_intent", "handle_error": "handle_error"})
    graph.add_conditional_edges("classify_intent", route_after_classify,  {"generate_sql": "generate_sql",       "handle_error": "handle_error"})
    graph.add_conditional_edges("generate_sql",    route_after_sql,       {"execute_query": "execute_query",     "handle_error": "handle_error"})
    graph.add_conditional_edges("execute_query",   route_after_execute,   {"run_scenario": "run_scenario", "generate_narrative": "generate_narrative", "handle_error": "handle_error"})
    graph.add_conditional_edges("run_scenario",    route_after_scenario,  {"generate_narrative": "generate_narrative", "handle_error": "handle_error"})

    # Terminal edges
    graph.add_edge("generate_narrative", END)
    graph.add_edge("handle_error",       END)

    return graph.compile()


# ── Public runner ─────────────────────────────────────────────────────────────

def run_agent(
    question: str,
    scenario_params: dict | None = None,
    target_schemas: list[str] | None = None,
) -> AgentState:
    """
    Main entrypoint. Run the full agent graph for a given question.

    Args:
        question:        Natural language question.
        scenario_params: Optional what-if mutation dict (see ScenarioParams).
        target_schemas:  DB schemas to restrict to, e.g. ["dbo", "cpg"].

    Returns:
        Final AgentState with .final_answer, .query_result, .scenario_delta, etc.
    """
    app = build_graph()

    initial_state = AgentState(
        question=question,
        scenario_params=scenario_params,
        target_schemas=target_schemas or [],
    )

    final_state = app.invoke(initial_state)
    return final_state
