"""
graph/nodes.py
One function per LangGraph node. Each receives AgentState, mutates it, returns it.
"""
from __future__ import annotations

import structlog
from anthropic import Anthropic

from config.settings import get_settings
from graph.state import AgentState
from tools.schema_discovery import get_cached_schema
from tools.data_fetcher import execute_query, execute_query_as_callback
from agents.sql_agent import generate_sql, generate_sql_with_retry
from agents.scenario_engine import ScenarioEngine, ScenarioParams

log = structlog.get_logger()
settings = get_settings()
client = Anthropic(api_key=settings.anthropic_api_key)


# ── Node 1: Load Schema ───────────────────────────────────────────────────────

def node_load_schema(state: AgentState) -> AgentState:
    """Loads (or retrieves from cache) the DB schema."""
    log.info("node.load_schema")
    state.current_node = "load_schema"
    try:
        schemas_key = ",".join(state.target_schemas)
        schema = get_cached_schema(schemas_key)
        state.schema_loaded = True
        log.info("node.load_schema.done", tables=len(schema.tables))
    except Exception as e:
        state.error_message = f"Schema discovery failed: {e}"
        log.error("node.load_schema.failed", error=str(e))
    return state


# ── Node 2: Classify Intent ───────────────────────────────────────────────────

def node_classify_intent(state: AgentState) -> AgentState:
    """
    Determines if the question is:
      (a) a simple data lookup, or
      (b) a what-if / scenario analysis question.
    Sets state.is_whatif accordingly.
    """
    log.info("node.classify_intent")
    state.current_node = "classify_intent"

    prompt = f"""Classify this question as either WHATIF or LOOKUP.

WHATIF: The question involves hypothetical changes, scenarios, comparisons of alternatives,
        optimisation under constraints, or asks "what would happen if...".
        Examples: "If we close 3 locations...", "What's the impact of shifting budget to Southeast",
                  "Compare current vs proposed coverage", "Optimise location strategy if demand grows 20%"

LOOKUP: A direct data question with no hypothetical component.
        Examples: "What is current revenue by region?", "How many locations do we have?",
                  "Show me top 10 locations by sales"

Question: {state.question}

Reply with exactly one word: WHATIF or LOOKUP"""

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    classification = response.content[0].text.strip().upper()
    state.is_whatif = classification == "WHATIF"
    log.info("node.classify_intent.done", is_whatif=state.is_whatif)
    return state


# ── Node 3: Generate SQL ──────────────────────────────────────────────────────

def node_generate_sql(state: AgentState) -> AgentState:
    """Calls the SQL agent to generate a T-SQL query."""
    log.info("node.generate_sql")
    state.current_node = "generate_sql"

    schemas_key = ",".join(state.target_schemas)
    schema = get_cached_schema(schemas_key)

    sql, answerable = generate_sql(
        question=state.question,
        schema=schema,
        scenario_params=state.scenario_params,
    )

    state.sql_answerable = answerable
    if answerable:
        state.generated_sql = sql
    else:
        state.error_message = f"Cannot answer from available data: {sql}"

    return state


# ── Node 4: Execute Query ─────────────────────────────────────────────────────

def node_execute_query(state: AgentState) -> AgentState:
    """Executes the generated SQL against MSSQL."""
    log.info("node.execute_query")
    state.current_node = "execute_query"

    if not state.generated_sql:
        state.fetch_error = "No SQL to execute."
        return state

    # Self-correcting execution via retry wrapper
    schemas_key = ",".join(state.target_schemas)
    schema = get_cached_schema(schemas_key)

    result_dict = generate_sql_with_retry(
        question=state.question,
        schema=schema,
        scenario_params=state.scenario_params,
        execute_fn=execute_query_as_callback,
    )

    if result_dict["success"]:
        state.query_result = result_dict.get("result")
        state.generated_sql = result_dict["sql"]
        state.sql_attempts = result_dict["attempts"]
    else:
        state.fetch_error = result_dict["error"]
        log.error("node.execute_query.failed", error=state.fetch_error)

    return state


# ── Node 5: Run Scenario Engine ───────────────────────────────────────────────

def node_run_scenario(state: AgentState) -> AgentState:
    """
    Applies what-if mutations to the baseline data.
    Only runs if state.is_whatif is True and scenario_params are provided.
    """
    log.info("node.run_scenario")
    state.current_node = "run_scenario"

    if not state.query_result:
        state.error_message = "No query result to run scenario against."
        return state

    # Build ScenarioParams from the dict passed in the request
    raw = state.scenario_params or {}
    params = ScenarioParams(
        scenario_name=raw.get("scenario_name", "What-If Scenario"),
        remove_rows=raw.get("remove_rows"),
        add_rows=raw.get("add_rows"),
        modify_rows=raw.get("modify_rows"),
        modify_values=raw.get("modify_values"),
        adjust_columns=raw.get("adjust_columns"),
        override_columns=raw.get("override_columns"),
        metric_columns=raw.get("metric_columns", []),
    )

    engine = ScenarioEngine()
    state.scenario_delta = engine.run(state.query_result, params)
    log.info("node.run_scenario.done", scenario=params.scenario_name)
    return state


# ── Node 6: Generate Narrative ────────────────────────────────────────────────

def node_generate_narrative(state: AgentState) -> AgentState:
    """
    Uses Claude to synthesise query results (and scenario delta if applicable)
    into a clear, human-readable analytical answer.
    """
    log.info("node.generate_narrative")
    state.current_node = "generate_narrative"

    # Build context for the narrative prompt
    data_summary = ""
    if state.query_result:
        data_summary = state.query_result.to_summary_text()

    scenario_summary = ""
    if state.scenario_delta:
        scenario_summary = f"\n\nSCENARIO ANALYSIS:\n{state.scenario_delta.to_summary_text()}"

    prompt = f"""You are a senior business analyst. Answer the user's question based on the data below.

USER QUESTION:
{state.question}

DATA FROM DATABASE:
{data_summary}
{scenario_summary}

INSTRUCTIONS:
- Give a clear, direct answer to the question.
- For what-if questions: highlight the key deltas and their business implications.
- Quantify everything with numbers from the data.
- Flag any limitations (e.g. data was truncated, scenario assumptions).
- Use bullet points for multi-part comparisons.
- Keep the tone analytical and concise — this is for a business decision-maker.
"""

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    state.final_answer = response.content[0].text.strip()
    state.completed = True
    log.info("node.generate_narrative.done")
    return state


# ── Node 7: Error Handler ─────────────────────────────────────────────────────

def node_handle_error(state: AgentState) -> AgentState:
    """Formats errors into a user-facing message."""
    log.error("node.handle_error", error=state.error_message or state.fetch_error)
    state.current_node = "handle_error"
    error = state.error_message or state.fetch_error or "Unknown error"
    state.final_answer = (
        f"I was unable to complete your request.\n\nReason: {error}\n\n"
        "Please check that your question relates to the available data, "
        "or contact your data team if the issue persists."
    )
    state.completed = True
    return state
