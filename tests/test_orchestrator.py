"""Pruebas offline del Módulo 6: ruteo del Supervisor y techo de pasos."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any, cast

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool

from orchestrator.agents.analyst_agent import (
    calcular_expresion,
    evaluate_arithmetic,
    score_sentiment,
)
from orchestrator.agents.research_agent import make_knowledge_search_tool
from orchestrator.config import OrchestratorSettings
from orchestrator.runner import run_orchestrator
from orchestrator.supervisor import build_supervisor_node


class ScriptedSupervisorModel:
    """Doble del Supervisor: devuelve decisiones JSON en el orden indicado."""

    def __init__(self, decisions: Sequence[dict[str, str]]) -> None:
        self._decisions = list(decisions)

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        decision = self._decisions.pop(0)
        return AIMessage(content=json.dumps(decision, ensure_ascii=False))


class SingleToolModel:
    """Doble de especialista: pide una tool fija y luego cierra con respuesta fija."""

    def __init__(
        self, tool_name: str, tool_args: dict[str, Any], final_answer: str
    ) -> None:
        self._tool_name = tool_name
        self._tool_args = tool_args
        self._final_answer = final_answer

    def bind_tools(self, tools: Sequence[BaseTool]) -> Runnable[object, AIMessage]:
        return RunnableLambda(self._respond)

    async def _respond(self, raw_messages: object) -> AIMessage:
        messages = cast(list[BaseMessage], raw_messages)
        observations = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        if not observations:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self._tool_name,
                        "args": self._tool_args,
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content=self._final_answer)


def _fake_retriever(documents: list[Document]) -> Runnable[str, list[Document]]:
    async def _retrieve(_query: str) -> list[Document]:
        return documents

    return RunnableLambda(_retrieve)


def test_supervisor_routes_researcher_then_analyst_then_ends() -> None:
    async def scenario() -> None:
        decisions = [
            {"next": "researcher", "instruction": "busca gastos comunes"},
            {"next": "analyst", "instruction": "calcula 45000*3"},
            {"next": "end", "instruction": "Respuesta final combinando ambos aportes."},
        ]
        researcher_model = SingleToolModel(
            "buscar_en_base_conocimiento",
            {"consulta": "gastos comunes"},
            "El reglamento indica que los gastos comunes se pagan mensualmente.",
        )
        analyst_model = SingleToolModel(
            "calcular_expresion",
            {"expresion": "45000*3"},
            "El cálculo da 135000.",
        )
        retriever = _fake_retriever(
            [
                Document(
                    page_content="Los gastos comunes...", metadata={"source": "doc.txt"}
                )
            ]
        )

        result = await run_orchestrator(
            "¿Cuánto se debe pagar por 3 cuotas de gastos comunes de 45000?",
            settings=OrchestratorSettings(max_steps=6),
            supervisor_model=ScriptedSupervisorModel(decisions),
            researcher_model=researcher_model,
            analyst_model=analyst_model,
            retriever=retriever,
        )

        assert result.answer == "Respuesta final combinando ambos aportes."
        assert result.steps == 3
        assert [c["agent"] for c in result.contributions] == ["researcher", "analyst"]
        assert "gastos comunes" in result.contributions[0]["summary"]
        assert "135000" in result.contributions[1]["summary"]

    asyncio.run(scenario())


def test_supervisor_step_cap_prevents_infinite_loop() -> None:
    async def scenario() -> None:
        decisions = [
            {"next": "researcher", "instruction": "busca A"},
            {"next": "researcher", "instruction": "busca B"},
        ]
        researcher_model = SingleToolModel(
            "buscar_en_base_conocimiento",
            {"consulta": "algo"},
            "Hallazgo parcial.",
        )
        retriever = _fake_retriever([])

        result = await run_orchestrator(
            "pregunta que nunca queda satisfecha",
            settings=OrchestratorSettings(max_steps=2),
            supervisor_model=ScriptedSupervisorModel(decisions),
            researcher_model=researcher_model,
            analyst_model=researcher_model,
            retriever=retriever,
        )

        assert result.steps == 3
        assert len(result.contributions) == 2
        assert "límite de pasos" in result.answer

    asyncio.run(scenario())


def test_evaluate_arithmetic_supports_basic_operators() -> None:
    assert evaluate_arithmetic("45000 * 3") == 135000
    assert evaluate_arithmetic("(10 + 5) / 3") == 5
    with pytest.raises(ValueError):
        evaluate_arithmetic("__import__('os')")


def test_calcular_expresion_tool_reports_errors_without_crashing() -> None:
    async def scenario() -> None:
        ok = await calcular_expresion.ainvoke({"expresion": "2 ** 10"})
        assert ok["status"] == "ok"
        assert ok["resultado"] == 1024

        broken = await calcular_expresion.ainvoke({"expresion": "2 / 0"})
        assert broken["status"] == "error"
        assert broken["resultado"] is None

    asyncio.run(scenario())


def test_score_sentiment_classifies_by_lexicon() -> None:
    positivo = score_sentiment("El reglamento garantiza seguridad y transparencia.")
    negativo = score_sentiment(
        "Hubo sanciones, multas y conflictos por incumplimiento."
    )
    neutral = score_sentiment("El artículo 32 define el aviso de cobro.")

    assert positivo["etiqueta"] == "positivo"
    assert negativo["etiqueta"] == "negativo"
    assert neutral["etiqueta"] == "neutral"


def test_knowledge_search_tool_reports_not_found_without_documents() -> None:
    async def scenario() -> None:
        tool = make_knowledge_search_tool(_fake_retriever([]))
        result = await tool.ainvoke({"consulta": "algo inexistente"})
        assert result["status"] == "not_found"
        assert result["resultados"] == []

        tool_with_hits = make_knowledge_search_tool(
            _fake_retriever(
                [Document(page_content="texto", metadata={"source": "a.txt"})]
            )
        )
        found = await tool_with_hits.ainvoke({"consulta": "algo"})
        assert found["status"] == "ok"
        assert found["resultados"][0]["fuente"] == "a.txt"

    asyncio.run(scenario())


class FlakySupervisorModel:
    """Doble que reproduce a `openrouter/free` devolviendo basura antes del JSON."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content=self._replies.pop(0))


def _initial_supervisor_state(request_text: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=request_text)],
        "contributions": [],
        "step_count": 0,
    }


def test_supervisor_retries_after_unparseable_output() -> None:
    async def scenario() -> None:
        good_decision = json.dumps({"next": "researcher", "instruction": "busca algo"})
        model = FlakySupervisorModel(["User Safety: safe", good_decision])
        node = build_supervisor_node(model, max_steps=6)

        update = await node(_initial_supervisor_state("pregunta"))

        assert update["next_agent"] == "researcher"
        assert update["instruction"] == "busca algo"

    asyncio.run(scenario())


def test_supervisor_raises_after_exhausting_retries() -> None:
    async def scenario() -> None:
        model = FlakySupervisorModel(["basura 1", "basura 2", "basura 3"])
        node = build_supervisor_node(model, max_steps=6)

        with pytest.raises(RuntimeError):
            await node(_initial_supervisor_state("pregunta"))

    asyncio.run(scenario())
