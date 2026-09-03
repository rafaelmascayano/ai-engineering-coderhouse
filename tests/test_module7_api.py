"""Pruebas offline del Módulo 7: pausa HITL, estado en Redis y worker.

El checkpointer de LangGraph se prueba con `InMemorySaver` (mismo contrato
que `AsyncRedisSaver`, sin necesitar Redis con RedisJSON/RediSearch); el
`JobStore` se prueba contra una instancia real de Redis local (solo usa
comandos básicos `GET`/`SET`, no requiere módulos), aislada en su propia base
(`db=15`) y limpiada en cada test. Sigue el mismo patrón que
`tests/test_orchestrator.py`: cada test es una función síncrona que envuelve
su escenario async en `asyncio.run(...)`, sin depender de `pytest-asyncio`.
"""

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
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from redis.asyncio import Redis

from app.graph import build_hitl_graph
from app.jobs import JobStatus, JobStore
from app.worker import run_job
from orchestrator.config import OrchestratorSettings

REDIS_TEST_URL = "redis://localhost:6379/15"


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
        self.call_count = 0

    def bind_tools(self, tools: Sequence[BaseTool]) -> Runnable[object, AIMessage]:
        return RunnableLambda(self._respond)

    async def _respond(self, raw_messages: object) -> AIMessage:
        self.call_count += 1
        messages = cast(list[BaseMessage], raw_messages)
        observations = [m for m in messages if isinstance(m, ToolMessage)]
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


def _initial_state(request: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=request)],
        "contributions": [],
        "next_agent": "researcher",
        "instruction": request,
        "task_completed": False,
        "step_count": 0,
    }


def test_hitl_pauses_before_analyst_and_resumes_on_approval() -> None:
    async def scenario() -> None:
        decisions = [
            {"next": "analyst", "instruction": "calcula 45000*3"},
            {"next": "end", "instruction": "El total es 135000."},
        ]
        analyst_model = SingleToolModel(
            "calcular_expresion", {"expresion": "45000*3"}, "135000."
        )

        checkpointer = InMemorySaver()
        graph = build_hitl_graph(
            checkpointer=checkpointer,
            settings=OrchestratorSettings(max_steps=6),
            supervisor_model=ScriptedSupervisorModel(decisions),
            researcher_model=analyst_model,
            analyst_model=analyst_model,
            retriever=_fake_retriever([]),
        )

        config = {"configurable": {"thread_id": "job-approve"}}
        paused = await graph.ainvoke(
            _initial_state("calcula 3 cuotas de 45000"), config=config
        )

        assert "__interrupt__" in paused
        interrupt_payload = paused["__interrupt__"][0].value
        assert interrupt_payload["agent"] == "analyst"
        assert analyst_model.call_count == 0

        resumed = await graph.ainvoke(Command(resume={"approved": True}), config=config)
        assert "__interrupt__" not in resumed
        assert resumed["messages"][-1].content == "El total es 135000."
        assert analyst_model.call_count == 2  # tool_call + respuesta final del ReAct

    asyncio.run(scenario())


def test_hitl_rejection_ends_without_calling_analyst() -> None:
    async def scenario() -> None:
        decisions = [{"next": "analyst", "instruction": "calcula 45000*3"}]
        analyst_model = SingleToolModel(
            "calcular_expresion", {"expresion": "45000*3"}, "135000."
        )

        checkpointer = InMemorySaver()
        graph = build_hitl_graph(
            checkpointer=checkpointer,
            settings=OrchestratorSettings(max_steps=6),
            supervisor_model=ScriptedSupervisorModel(decisions),
            researcher_model=analyst_model,
            analyst_model=analyst_model,
            retriever=_fake_retriever([]),
        )

        config = {"configurable": {"thread_id": "job-reject"}}
        await graph.ainvoke(_initial_state("calcula 3 cuotas de 45000"), config=config)

        resumed = await graph.ainvoke(
            Command(resume={"approved": False, "comment": "monto sin validar"}),
            config=config,
        )

        assert "__interrupt__" not in resumed
        assert resumed["messages"][-1].name == "human_approval"
        assert "rechazó" in resumed["messages"][-1].content
        assert "monto sin validar" in resumed["messages"][-1].content
        assert analyst_model.call_count == 0

    asyncio.run(scenario())


def test_researcher_only_path_never_pauses() -> None:
    async def scenario() -> None:
        decisions = [
            {"next": "researcher", "instruction": "busca gastos comunes"},
            {"next": "end", "instruction": "Respuesta final."},
        ]
        researcher_model = SingleToolModel(
            "buscar_en_base_conocimiento",
            {"consulta": "gastos comunes"},
            "Hallazgo.",
        )
        retriever = _fake_retriever(
            [
                Document(
                    page_content="Los gastos comunes...", metadata={"source": "doc.txt"}
                )
            ]
        )

        checkpointer = InMemorySaver()
        graph = build_hitl_graph(
            checkpointer=checkpointer,
            settings=OrchestratorSettings(max_steps=6),
            supervisor_model=ScriptedSupervisorModel(decisions),
            researcher_model=researcher_model,
            analyst_model=researcher_model,
            retriever=retriever,
        )

        config = {"configurable": {"thread_id": "job-no-pause"}}
        result = await graph.ainvoke(_initial_state("busca algo"), config=config)

        assert "__interrupt__" not in result
        assert result["messages"][-1].content == "Respuesta final."

    asyncio.run(scenario())


async def _fresh_redis_client() -> Redis:
    client = Redis.from_url(REDIS_TEST_URL, decode_responses=True)
    await client.flushdb()
    return client


def test_job_store_lifecycle() -> None:
    async def scenario() -> None:
        redis_client = await _fresh_redis_client()
        try:
            store = JobStore(redis_client, ttl_seconds=60)

            job_id = await store.create("una solicitud")
            record = await store.get(job_id)
            assert record is not None
            assert record["status"] == JobStatus.PENDING.value

            await store.mark_running(job_id)
            assert (await store.get(job_id))["status"] == JobStatus.RUNNING.value

            await store.mark_waiting_approval(job_id, {"agent": "analyst"})
            waiting = await store.get(job_id)
            assert waiting["status"] == JobStatus.WAITING_APPROVAL.value
            assert waiting["pending_approval"] == {"agent": "analyst"}

            await store.mark_done(
                job_id, answer="listo", steps=3, contributions=[{"agent": "analyst"}]
            )
            done = await store.get(job_id)
            assert done["status"] == JobStatus.DONE.value
            assert done["answer"] == "listo"
            assert done["pending_approval"] is None
        finally:
            await redis_client.flushdb()
            await redis_client.aclose()

    asyncio.run(scenario())


def test_job_store_marks_failed() -> None:
    async def scenario() -> None:
        redis_client = await _fresh_redis_client()
        try:
            store = JobStore(redis_client, ttl_seconds=60)
            job_id = await store.create("otra solicitud")

            await store.mark_failed(job_id, error="boom")
            record = await store.get(job_id)
            assert record["status"] == JobStatus.FAILED.value
            assert record["error"] == "boom"
        finally:
            await redis_client.flushdb()
            await redis_client.aclose()

    asyncio.run(scenario())


def test_get_missing_job_returns_none() -> None:
    async def scenario() -> None:
        redis_client = await _fresh_redis_client()
        try:
            store = JobStore(redis_client, ttl_seconds=60)
            assert await store.get("no-existe") is None
        finally:
            await redis_client.flushdb()
            await redis_client.aclose()

    asyncio.run(scenario())


def test_worker_run_job_marks_failed_on_exception() -> None:
    async def scenario() -> None:
        redis_client = await _fresh_redis_client()
        try:
            store = JobStore(redis_client, ttl_seconds=60)
            job_id = await store.create("solicitud que falla")

            class ExplodingGraph:
                async def ainvoke(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
                    raise RuntimeError("el LLM no respondió")

            await run_job(
                job_id, "solicitud que falla", job_store=store, graph=ExplodingGraph()
            )

            record = await store.get(job_id)
            assert record["status"] == JobStatus.FAILED.value
            assert "no respondió" in record["error"]
        finally:
            await redis_client.flushdb()
            await redis_client.aclose()

    asyncio.run(scenario())


def test_worker_run_job_reaches_waiting_approval() -> None:
    async def scenario() -> None:
        decisions = [{"next": "analyst", "instruction": "calcula 10*10"}]
        analyst_model = SingleToolModel(
            "calcular_expresion", {"expresion": "10*10"}, "100."
        )
        checkpointer = InMemorySaver()
        graph = build_hitl_graph(
            checkpointer=checkpointer,
            settings=OrchestratorSettings(max_steps=6),
            supervisor_model=ScriptedSupervisorModel(decisions),
            researcher_model=analyst_model,
            analyst_model=analyst_model,
            retriever=_fake_retriever([]),
        )

        redis_client = await _fresh_redis_client()
        try:
            store = JobStore(redis_client, ttl_seconds=60)
            job_id = await store.create("calcula 10*10")

            await run_job(job_id, "calcula 10*10", job_store=store, graph=graph)

            record = await store.get(job_id)
            assert record["status"] == JobStatus.WAITING_APPROVAL.value
            assert record["pending_approval"]["agent"] == "analyst"
        finally:
            await redis_client.flushdb()
            await redis_client.aclose()

    asyncio.run(scenario())


def test_job_store_requires_redis() -> None:
    """Marcador explícito: estos tests necesitan `redis-server` en localhost:6379."""

    async def scenario() -> None:
        client = Redis.from_url(REDIS_TEST_URL, decode_responses=True)
        try:
            await client.ping()
        finally:
            await client.aclose()

    try:
        asyncio.run(scenario())
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"Redis no disponible en localhost:6379: {error}")
