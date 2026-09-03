"""Fachada asíncrona: arma el modelo, corre el grafo y devuelve un resultado tipado."""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from orchestrator.agents._shared import ToolBindableChatModel
from orchestrator.config import OrchestratorSettings
from orchestrator.graph import build_orchestrator_graph
from orchestrator.state import AgentContribution
from orchestrator.supervisor import SupervisorChatModel

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True, slots=True)
class OrchestratorResult:
    """Resultado observable de una ejecución: respuesta, aportes y pasos usados."""

    answer: str
    contributions: list[AgentContribution]
    steps: int


def create_openrouter_model(settings: OrchestratorSettings, api_key: str) -> ChatOpenAI:
    """Crea el modelo compartido por el Supervisor y los especialistas."""

    return ChatOpenAI(
        model=settings.model,
        api_key=SecretStr(api_key),
        base_url=OPENROUTER_BASE_URL,
        temperature=settings.temperature,
        max_retries=2,
    )


async def run_orchestrator(
    request: str,
    *,
    settings: OrchestratorSettings | None = None,
    api_key: str | None = None,
    supervisor_model: SupervisorChatModel | None = None,
    researcher_model: ToolBindableChatModel | None = None,
    analyst_model: ToolBindableChatModel | None = None,
    retriever: Runnable[str, list[Document]] | None = None,
) -> OrchestratorResult:
    """Ejecuta un turno completo del orquestador para una única solicitud.

    Sin `settings`/`api_key` usa `OrchestratorSettings.from_env()` y crea un
    único `ChatOpenAI` sobre OpenRouter para los tres roles. Cada modelo
    también puede inyectarse por separado, lo que permiten las pruebas
    offline sin llamar a ninguna API.
    """

    normalized_request = request.strip()
    if not normalized_request:
        raise ValueError("request no puede estar vacío")

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
                "Falta OPENROUTER_API_KEY. Configúrala en .env antes de ejecutar "
                "el orquestador con el modelo real."
            )
        shared_model = create_openrouter_model(resolved_settings, resolved_key.strip())

    graph = build_orchestrator_graph(
        supervisor_model=supervisor_model
        if supervisor_model is not None
        else shared_model,
        researcher_model=researcher_model
        if researcher_model is not None
        else shared_model,
        analyst_model=analyst_model if analyst_model is not None else shared_model,
        max_steps=resolved_settings.max_steps,
        retriever=retriever,
    )

    final_state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=normalized_request)],
            "contributions": [],
            "next_agent": "researcher",
            "instruction": normalized_request,
            "task_completed": False,
            "step_count": 0,
        }
    )

    last_message = final_state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise RuntimeError("El grafo finalizó sin una respuesta del Supervisor")

    return OrchestratorResult(
        answer=str(last_message.content),
        contributions=list(final_state["contributions"]),
        steps=final_state["step_count"],
    )
