"""
graph/state.py
Defines the shared state object that flows through every LangGraph node.
"""
from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class AgentState:
    # ── Input ──────────────────────────────────────────────────────────────
    question: str = ""                        # original user question
    scenario_params: dict | None = None       # structured what-if params (optional)
    target_schemas: list[str] = field(default_factory=list)  # e.g. ["dbo", "cpg"]

    # ── Schema ─────────────────────────────────────────────────────────────
    schema_loaded: bool = False

    # ── SQL generation ─────────────────────────────────────────────────────
    generated_sql: str | None = None
    sql_attempts: int = 0
    sql_answerable: bool = True               # False = schema can't answer this

    # ── Data fetch ─────────────────────────────────────────────────────────
    query_result: Any | None = None           # QueryResult object
    fetch_error: str | None = None

    # ── Scenario ───────────────────────────────────────────────────────────
    scenario_delta: Any | None = None         # ScenarioDelta object
    is_whatif: bool = False                   # did user ask a what-if question?

    # ── Analysis & narrative ───────────────────────────────────────────────
    analysis_text: str = ""
    final_answer: str = ""

    # ── Control flow ───────────────────────────────────────────────────────
    error_message: str | None = None
    completed: bool = False
    current_node: str = "start"

    def has_error(self) -> bool:
        return bool(self.error_message or self.fetch_error)
