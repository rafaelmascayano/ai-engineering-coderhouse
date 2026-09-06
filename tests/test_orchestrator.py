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


class RecordingSupervisorModel:
    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        return AIMessage(content=self.replies.pop(0))


@pytest.mark.parametrize(
    "invalid",
    [
        "Respuesta sin JSON",
        '{"next": "unknown", "instruction": "busca"}',
        '{"next": "researcher", "instruction": ""}',
    ],
)
def test_supervisor_retries_invalid_decision_preserving_context(invalid: str) -> None:
    async def scenario() -> None:
        model = RecordingSupervisorModel(
            [
                invalid,
                json.dumps({"next": "analyst", "instruction": "calcula 45000*3"}),
            ]
        )
        node = build_supervisor_node(model, max_steps=6)
        update = await node(
            {
                "messages": [HumanMessage(content="consulta original")],
                "contributions": [
                    {"agent": "researcher", "summary": "evidencia previa"}
                ],
                "step_count": 1,
            }
        )
        assert update["next_agent"] == "analyst"
        assert update["step_count"] == 2
        assert len(model.calls) == 2
        assert model.calls[1][:2] == model.calls[0]
        assert "consulta original" in str(model.calls[1][1].content)
        assert "evidencia previa" in str(model.calls[1][1].content)

    asyncio.run(scenario())


def test_supervisor_stops_after_three_invalid_decisions() -> None:
    async def scenario() -> None:
        model = RecordingSupervisorModel(["invalid"] * 4)
        node = build_supervisor_node(model, max_steps=6)
        with pytest.raises(RuntimeError, match="3 intentos"):
            await node(
                {
                    "messages": [HumanMessage(content="consulta")],
                    "contributions": [],
                    "step_count": 0,
                }
            )
        assert len(model.calls) == 3
        assert len(model.replies) == 1

    asyncio.run(scenario())


def test_refinement_preserves_evidence_and_isolates_specialist_context() -> None:
    class EvidenceResearcher:
        def __init__(self) -> None:
            self.instructions: list[str] = []

        def bind_tools(self, tools: Sequence[BaseTool]) -> Runnable:
            async def respond(messages: list[BaseMessage]) -> AIMessage:
                instruction = str(messages[1].content)
                observations = [m for m in messages if isinstance(m, ToolMessage)]
                if not observations:
                    assert len(messages) == 2  # sistema + instrucción, sin historial
                    self.instructions.append(instruction)
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "buscar_en_base_conocimiento",
                                "args": {"consulta": instruction},
                                "id": "research-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                return AIMessage(content=str(observations[-1].content))

            return RunnableLambda(respond)

    class EvidenceAnalyst:
        def bind_tools(self, tools: Sequence[BaseTool]) -> Runnable:
            async def respond(messages: list[BaseMessage]) -> AIMessage:
                observations = [m for m in messages if isinstance(m, ToolMessage)]
                if not observations:
                    assert len(messages) == 2
                    assert "garantiza seguridad" in str(messages[1].content)
                    assert "fuente.txt" in str(messages[1].content)
                    assert "not_found" not in str(messages[1].content)
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "analizar_sentimiento",
                                "args": {"texto": "garantiza seguridad"},
                                "id": "analysis-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                return AIMessage(content=str(observations[-1].content))

            return RunnableLambda(respond)

    async def scenario() -> None:
        async def retrieve(query: str) -> list[Document]:
            if query == "busca con fuente":
                return [
                    Document(
                        page_content="garantiza seguridad",
                        metadata={"source": "fuente.txt"},
                    )
                ]
            return []

        supervisor = RecordingSupervisorModel(
            [
                json.dumps(d)
                for d in [
                    {"next": "researcher", "instruction": "busca"},
                    {"next": "researcher", "instruction": "busca con fuente"},
                    {
                        "next": "analyst",
                        "instruction": "Tono: garantiza seguridad (fuente.txt)",
                    },
                    {
                        "next": "end",
                        "instruction": "fuente.txt: tono positivo según el léxico.",
                    },
                ]
            ]
        )
        researcher = EvidenceResearcher()
        result = await run_orchestrator(
            "Investiga y analiza el tono",
            settings=OrchestratorSettings(max_steps=6),
            supervisor_model=supervisor,
            researcher_model=researcher,
            analyst_model=EvidenceAnalyst(),
            retriever=RunnableLambda(retrieve),
        )
        assert result.steps == 4
        assert [c["agent"] for c in result.contributions] == [
            "researcher",
            "researcher",
            "analyst",
        ]
        assert "not_found" in result.contributions[0]["summary"]
        assert "fuente.txt" in result.contributions[1]["summary"]
        assert '"etiqueta": "positivo"' in result.contributions[2]["summary"]
        assert researcher.instructions == ["busca", "busca con fuente"]
        final_context = str(supervisor.calls[-1][1].content)
        assert "not_found" in final_context and "fuente.txt" in final_context
        assert "positivo" in final_context

    asyncio.run(scenario())
