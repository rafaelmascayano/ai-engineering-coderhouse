"""Grafo del Módulo 7: el jerárquico del Módulo 6 + pausa Human-in-the-loop.

Reutiliza los nodos ya probados en `orchestrator/` (Supervisor, researcher,
analyst) sin modificarlos, e inserta `human_approval` entre el Supervisor y
`analyst`:

    START -> supervisor -- next=researcher --> researcher -> supervisor
                        \\-- next=analyst --> human_approval -- aprobado --> analyst
                        |                                   \\-- rechazado --> END
                        \\-- next=end      --> END
    analyst -> supervisor

El grafo se compila con un checkpointer inyectado (en producción,
`AsyncRedisSaver`) para que `interrupt()` pueda persistir el estado mientras
el trabajo espera aprobación humana, sobrevida a reinicios y sea reanudable
por `job_id` (usado como `thread_id`).
"""

from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from app.hitl import build_human_approval_node, route_after_approval
from orchestrator.agents._shared import ToolBindableChatModel
from orchestrator.agents.analyst_agent import build_analyst_node
from orchestrator.agents.research_agent import build_research_node
from orchestrator.config import OrchestratorSettings
from orchestrator.state import OrchestratorState
from orchestrator.supervisor import (
    SupervisorChatModel,
    build_supervisor_node,
    route_supervisor,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_shared_model(settings: OrchestratorSettings, api_key: str) -> ChatOpenAI:
    """Mismo modelo compartido que `orchestrator.runner.create_openrouter_model`."""

    return ChatOpenAI(
        model=settings.model,
        api_key=SecretStr(api_key),
        base_url=OPENROUTER_BASE_URL,
        temperature=settings.temperature,
        max_retries=2,
    )


def build_hitl_graph(
    *,
    checkpointer: BaseCheckpointSaver,
    settings: OrchestratorSettings | None = None,
    api_key: str | None = None,
    critical_agent: str = "analyst",
    supervisor_model: SupervisorChatModel | None = None,
    researcher_model: ToolBindableChatModel | None = None,
    analyst_model: ToolBindableChatModel | None = None,
    retriever: Runnable[str, list[Document]] | None = None,
) -> CompiledStateGraph[OrchestratorState, None, OrchestratorState, OrchestratorState]:
    """Compone el grafo jerárquico + HITL y lo compila con `checkpointer`.

    Igual que `orchestrator.runner.run_orchestrator`, los tres modelos pueden
    inyectarse por separado -- así las pruebas ensamblan exactamente esta
    topología (incluida la pausa HITL) con dobles deterministas, sin API key
    ni red.
    """

    resolved_settings = (
        settings if settings is not None else OrchestratorSettings.from_env()
    )

    shared_model = None
    if supervisor_model is None or researcher_model is None or analyst_model is None:
        resolved_key = (
            api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
        )
        if not resolved_key.strip():
            raise RuntimeError(
                "Falta OPENROUTER_API_KEY. Configúrala en .env antes de levantar "
                "la API."
            )
        shared_model = create_shared_model(resolved_settings, resolved_key.strip())

    builder: StateGraph[
        OrchestratorState, None, OrchestratorState, OrchestratorState
    ] = StateGraph(OrchestratorState)
    builder.add_node(
        "supervisor",
        build_supervisor_node(
            supervisor_model if supervisor_model is not None else shared_model,
            max_steps=resolved_settings.max_steps,
        ),
    )
    builder.add_node(
        "researcher",
        build_research_node(
            researcher_model if researcher_model is not None else shared_model,
            retriever=retriever,
        ),
    )
    builder.add_node(
        "analyst",
        build_analyst_node(
            analyst_model if analyst_model is not None else shared_model
        ),
    )
    builder.add_node(
        "human_approval", build_human_approval_node(critical_agent=critical_agent)
    )

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"researcher": "researcher", "analyst": "human_approval", "end": END},
    )
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"proceed": "analyst", "end": END},
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")

    return builder.compile(checkpointer=checkpointer)
