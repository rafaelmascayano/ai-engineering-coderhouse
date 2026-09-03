"""Grafo jerárquico: Supervisor -> {researcher, analyst} -> Supervisor -> END."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from orchestrator.agents._shared import ToolBindableChatModel
from orchestrator.agents.analyst_agent import build_analyst_node
from orchestrator.agents.research_agent import build_research_node
from orchestrator.state import OrchestratorState
from orchestrator.supervisor import (
    SupervisorChatModel,
    build_supervisor_node,
    route_supervisor,
)


def build_orchestrator_graph(
    *,
    supervisor_model: SupervisorChatModel,
    researcher_model: ToolBindableChatModel,
    analyst_model: ToolBindableChatModel,
    max_steps: int,
    retriever: Runnable[str, list[Document]] | None = None,
) -> CompiledStateGraph[OrchestratorState, None, OrchestratorState, OrchestratorState]:
    """Compone el Supervisor y los dos especialistas en un único `StateGraph`.

    Los tres modelos se reciben por separado (aunque en producción suelen ser
    la misma instancia) para que las pruebas puedan inyectar un doble por rol
    sin acoplar el grafo a un único tipo de modelo.
    """

    builder: StateGraph[
        OrchestratorState, None, OrchestratorState, OrchestratorState
    ] = StateGraph(OrchestratorState)
    builder.add_node(
        "supervisor", build_supervisor_node(supervisor_model, max_steps=max_steps)
    )
    builder.add_node(
        "researcher", build_research_node(researcher_model, retriever=retriever)
    )
    builder.add_node("analyst", build_analyst_node(analyst_model))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"researcher": "researcher", "analyst": "analyst", "end": END},
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")

    return builder.compile()
