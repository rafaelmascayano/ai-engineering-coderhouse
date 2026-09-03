"""Construcción reutilizable del ciclo modelo -> herramientas -> modelo.

Cada especialista es una instancia independiente de este mismo ciclo (el que
ya probó el Módulo 5), pero recibe únicamente la instrucción puntual que le
asigna el Supervisor -- nunca el historial completo del orquestador -- para
evitar la "contaminación de contexto" entre agentes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition


class ToolBindableChatModel(Protocol):
    """Contrato mínimo para modelos reales y dobles offline de pruebas."""

    def bind_tools(self, tools: Sequence[BaseTool]) -> Runnable[object, AIMessage]:
        """Vincula el catálogo que el especialista puede seleccionar autónomamente."""


class SpecialistState(MessagesState):
    """Estado aislado del especialista: solo su propia conversación con las tools."""


def build_specialist_agent(
    model: ToolBindableChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
) -> CompiledStateGraph[SpecialistState, None, SpecialistState, SpecialistState]:
    """Compila un ciclo ReAct acotado a un rol y a un catálogo de tools propio."""

    model_with_tools = model.bind_tools(tools)

    async def call_model(state: SpecialistState) -> dict[str, list[AIMessage]]:
        response = await model_with_tools.ainvoke(
            [SystemMessage(content=system_prompt), *state["messages"]]
        )
        if not isinstance(response, AIMessage):
            raise TypeError("El modelo debe devolver un AIMessage")
        return {"messages": [response]}

    builder = StateGraph(SpecialistState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(list(tools)))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    return builder.compile()


async def run_specialist(
    agent: CompiledStateGraph[SpecialistState, None, SpecialistState, SpecialistState],
    instruction: str,
) -> str:
    """Ejecuta el ciclo con una única instrucción y devuelve la respuesta final."""

    result = await agent.ainvoke({"messages": [HumanMessage(content=instruction)]})
    messages = cast(list[BaseMessage], result["messages"])
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return str(message.content)
    raise RuntimeError("El especialista finalizó sin una respuesta del modelo")
