"""Fachada asíncrona del agente y gestión del ciclo de vida de SQLite."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from cyclic_agent.config import AgentSettings
from cyclic_agent.graph import (
    AgentState,
    ToolBindableChatModel,
    build_graph,
    graph_config,
)
from cyclic_agent.tracing import records_from_update, write_trace

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class CyclicAgent:
    """Ejecuta turnos persistentes sobre un grafo ya compilado."""

    def __init__(
        self,
        graph: CompiledStateGraph[AgentState, None, AgentState, AgentState],
        *,
        recursion_limit: int,
    ) -> None:
        self._graph = graph
        self._recursion_limit = recursion_limit

    async def ask(
        self,
        prompt: str,
        *,
        thread_id: str,
        trace_path: Path | None = None,
    ) -> str:
        """Procesa un turno, conserva el estado y opcionalmente registra su traza."""

        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("El prompt no puede estar vacío")

        config = graph_config(thread_id, self._recursion_limit)
        events: list[dict[str, Any]] = []
        final_message: AIMessage | None = None

        async for update in self._graph.astream(
            {"messages": [HumanMessage(content=normalized_prompt)]},
            config=config,
            stream_mode="updates",
        ):
            new_records = records_from_update(update, start_sequence=len(events) + 1)
            events.extend(new_records)
            for state_update in update.values():
                if not isinstance(state_update, dict):
                    continue
                for message in state_update.get("messages", []):
                    if isinstance(message, AIMessage) and not message.tool_calls:
                        final_message = message

        if final_message is None:
            raise RuntimeError("El grafo finalizó sin una respuesta del modelo")

        if trace_path is not None:
            await write_trace(
                trace_path,
                thread_id=thread_id,
                prompt=normalized_prompt,
                events=events,
            )
        return str(final_message.content)


def create_openrouter_model(settings: AgentSettings) -> ChatOpenAI:
    """Crea el modelo gratuito mediante la API OpenAI-compatible de OpenRouter."""

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Falta OPENROUTER_API_KEY. Configúrala en .env antes de ejecutar "
            "la demo."
        )
    return ChatOpenAI(
        model=settings.model,
        api_key=SecretStr(api_key),
        base_url=OPENROUTER_BASE_URL,
        temperature=settings.temperature,
        max_retries=2,
    )


@asynccontextmanager
async def create_agent(
    settings: AgentSettings,
    *,
    model: BaseChatModel | ToolBindableChatModel | None = None,
) -> AsyncIterator[CyclicAgent]:
    """Abre el checkpointer SQLite asíncrono y entrega un agente listo para usar."""

    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    selected_model = model if model is not None else create_openrouter_model(settings)
    async with AsyncSqliteSaver.from_conn_string(
        str(settings.database_path)
    ) as checkpointer:
        await checkpointer.setup()
        graph = build_graph(selected_model, checkpointer)
        yield CyclicAgent(graph, recursion_limit=settings.recursion_limit)
