"""Definición del StateGraph cíclico y su contrato de estado."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from cyclic_agent.tools import ORDER_TOOLS

SYSTEM_PROMPT = """Eres un agente de soporte de pedidos preciso y autónomo.
Decide por ti mismo si necesitas herramientas a partir del pedido del usuario.
No inventes información de pedidos. Si una respuesta requiere datos que viven en
herramientas distintas, llama a todas las necesarias antes de concluir. Después de
cada resultado vuelve a evaluar si la información está completa. Ante un error o
status=not_found, corrige y reintenta únicamente si la conversación contiene un ID
alternativo inequívoco; en caso contrario pide una aclaración breve. Puedes usar el
historial del thread para resolver referencias como «ese cliente» o «el último».
Expresa montos como pesos con separador de miles y conserva fechas ISO-8601.
"""


class AgentState(MessagesState):
    """Estado acumulativo; `messages` conserva el reducer de `MessagesState`."""


class ToolBindableChatModel(Protocol):
    """Contrato mínimo para modelos reales y dobles offline de pruebas."""

    def bind_tools(
        self, tools: Sequence[BaseTool]
    ) -> Runnable[object, AIMessage]:
        """Vincula el catálogo que el modelo puede seleccionar autónomamente."""


def _tool_error(error: Exception) -> str:
    """Convierte fallos de herramientas en observaciones accionables para el LLM."""

    return json.dumps(
        {
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
            "instruction": (
                "Revisa los argumentos y reintenta si puedes corregirlos con el "
                "contexto; de lo contrario pide aclaración."
            ),
        },
        ensure_ascii=False,
    )


def build_graph(
    model: BaseChatModel | ToolBindableChatModel,
    checkpointer: BaseCheckpointSaver[str],
    *,
    tools: Sequence[BaseTool] = ORDER_TOOLS,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Construye el ciclo modelo → herramientas → modelo con persistencia."""

    model_with_tools = cast(ToolBindableChatModel, model).bind_tools(tools)

    async def call_model(state: AgentState) -> dict[str, list[AIMessage]]:
        response = await model_with_tools.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        if not isinstance(response, AIMessage):
            raise TypeError("El modelo debe devolver un AIMessage")
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("model", call_model)
    builder.add_node(
        "tools",
        ToolNode(list(tools), handle_tool_errors=_tool_error),
    )
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    return builder.compile(checkpointer=checkpointer)


def graph_config(thread_id: str, recursion_limit: int) -> RunnableConfig:
    """Crea la configuración estable de sesión y aplica el techo del ciclo."""

    normalized_thread_id = thread_id.strip()
    if not normalized_thread_id:
        raise ValueError("thread_id no puede estar vacío")
    return {
        "configurable": {"thread_id": normalized_thread_id},
        "recursion_limit": recursion_limit,
    }
