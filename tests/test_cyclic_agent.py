"""Pruebas offline del ciclo, resiliencia y memoria del módulo 5."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from cyclic_agent.agent import CyclicAgent
from cyclic_agent.graph import build_graph


def _current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Aísla el turno actual para que el doble se comporte de forma determinista."""

    last_human = max(
        index
        for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    )
    return messages[last_human:]


def _tool_call(name: str, cliente_id: int, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": {"cliente_id": cliente_id},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


class MultiStepModel:
    """Doble que fuerza las dos observaciones exigidas por la aceptación."""

    def bind_tools(
        self, tools: Sequence[BaseTool]
    ) -> Runnable[object, AIMessage]:
        assert {tool.name for tool in tools} == {
            "buscar_resumen_pedidos",
            "buscar_ultimo_pedido",
        }
        return RunnableLambda(self._respond)

    async def _respond(self, raw_messages: object) -> AIMessage:
        messages = cast(list[BaseMessage], raw_messages)
        current_turn = _current_turn(messages)
        observations = [
            message for message in current_turn if isinstance(message, ToolMessage)
        ]
        if not observations:
            return _tool_call("buscar_resumen_pedidos", 102, "call-summary")
        if len(observations) == 1:
            return _tool_call("buscar_ultimo_pedido", 102, "call-latest")
        return AIMessage(
            content=(
                "El cliente 102 tuvo 3 pedidos por $14.500. El último fue el "
                "pedido 5015 del 2026-08-03 y está en preparación."
            )
        )


class PersistentContextModel:
    """Doble que solo resuelve «el último» si SQLite conservó el cliente."""

    def bind_tools(
        self, tools: Sequence[BaseTool]
    ) -> Runnable[object, AIMessage]:
        return RunnableLambda(self._respond)

    async def _respond(self, raw_messages: object) -> AIMessage:
        messages = cast(list[BaseMessage], raw_messages)
        current_turn = _current_turn(messages)
        prompt = cast(HumanMessage, current_turn[0]).content
        observations = [
            message for message in current_turn if isinstance(message, ToolMessage)
        ]

        if "último" not in str(prompt):
            if not observations:
                return _tool_call("buscar_resumen_pedidos", 102, "call-memory-summary")
            return AIMessage(content="El cliente 102 tuvo 3 pedidos por $14.500.")

        if observations:
            return AIMessage(content="El último fue el pedido 5015.")

        prior_summary = any(
            isinstance(message, ToolMessage)
            and message.name == "buscar_resumen_pedidos"
            and '"cliente_id": 102' in str(message.content)
            for message in messages[: -len(current_turn)]
        )
        if prior_summary:
            return _tool_call("buscar_ultimo_pedido", 102, "call-memory-latest")
        return AIMessage(content="¿De qué cliente necesitas el último pedido?")


class RetryModel:
    """Doble que corrige un ID luego de observar un resultado incompleto."""

    def bind_tools(
        self, tools: Sequence[BaseTool]
    ) -> Runnable[object, AIMessage]:
        return RunnableLambda(self._respond)

    async def _respond(self, raw_messages: object) -> AIMessage:
        current_turn = _current_turn(cast(list[BaseMessage], raw_messages))
        observations = [
            message for message in current_turn if isinstance(message, ToolMessage)
        ]
        if not observations:
            return _tool_call("buscar_resumen_pedidos", 999, "call-invalid")
        if len(observations) == 1:
            assert "not_found" in str(observations[0].content)
            return _tool_call("buscar_resumen_pedidos", 102, "call-retry")
        return AIMessage(content="Reintento exitoso: el cliente 102 tiene 3 pedidos.")


def test_multi_step_trace_contains_two_tool_calls(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = tmp_path / "multi-step.sqlite"
        trace = tmp_path / "react-trace.json"
        async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
            await saver.setup()
            graph = build_graph(MultiStepModel(), saver)
            agent = CyclicAgent(graph, recursion_limit=10)
            response = await agent.ask(
                "¿Cuántos pedidos tuvo el cliente 102, cuál fue el total y el último?",
                thread_id="multi-step-102",
                trace_path=trace,
            )

        assert "3 pedidos" in response
        payload = json.loads(trace.read_text(encoding="utf-8"))
        tool_names = [
            tool_call["name"]
            for event in payload["events"]
            for tool_call in event.get("tool_calls", [])
        ]
        assert tool_names == ["buscar_resumen_pedidos", "buscar_ultimo_pedido"]

    asyncio.run(scenario())


def test_same_thread_remembers_customer_across_turns(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = tmp_path / "memory.sqlite"

        # Primer proceso lógico: escribe el checkpoint y cierra la conexión.
        async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
            await saver.setup()
            graph = build_graph(PersistentContextModel(), saver)
            agent = CyclicAgent(graph, recursion_limit=10)
            await agent.ask("Dame el resumen del cliente 102", thread_id="session-a")

        # Segundo proceso lógico: reabre el archivo y recupera la misma sesión.
        async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
            await saver.setup()
            graph = build_graph(PersistentContextModel(), saver)
            agent = CyclicAgent(graph, recursion_limit=10)
            remembered = await agent.ask("¿Y el último?", thread_id="session-a")
            isolated = await agent.ask("¿Y el último?", thread_id="session-b")

        assert remembered == "El último fue el pedido 5015."
        assert isolated == "¿De qué cliente necesitas el último pedido?"

    asyncio.run(scenario())


def test_incomplete_result_returns_to_model_and_is_retried(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = tmp_path / "retry.sqlite"
        async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
            await saver.setup()
            graph = build_graph(RetryModel(), saver)
            agent = CyclicAgent(graph, recursion_limit=10)
            response = await agent.ask(
                "Busca el cliente; si el primer ID falla usa el 102.",
                thread_id="retry-session",
            )

        assert response.startswith("Reintento exitoso")

    asyncio.run(scenario())
