"""Patrón worker: corre el grafo en segundo plano y sincroniza el job en Redis.

Se ejecuta como una tarea `asyncio` (`BackgroundTasks` de FastAPI), no como un
hilo bloqueante: el grafo entero es `async` (nodos `async def`, LLM vía
`ainvoke`), así que nunca bloquea el event loop del proceso principal. Toda
excepción no controlada durante la ejecución del grafo se captura aquí y se
persiste como `FAILED` en Redis -- si no, el cliente quedaría haciendo poll
indefinidamente sobre un trabajo que nunca va a progresar.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.jobs import JobStore
from orchestrator.state import OrchestratorState

logger = logging.getLogger(__name__)

HitlGraph = CompiledStateGraph[
    OrchestratorState, None, OrchestratorState, OrchestratorState
]


def _thread_config(job_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": job_id}}


async def _sync_state(job_store: JobStore, job_id: str, result: dict[str, Any]) -> None:
    """Traduce el estado crudo del grafo al estado observable del job."""

    interrupts = result.get("__interrupt__")
    if interrupts:
        payload = interrupts[0].value
        await job_store.mark_waiting_approval(job_id, payload)
        return

    last_message = result["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise RuntimeError("El grafo finalizó sin una respuesta del Supervisor")

    if result.get("next_agent") == "end" and not result.get("task_completed", True):
        # No debería ocurrir: task_completed siempre se fija en "end". Se deja
        # como salvaguarda explícita en vez de asumir silenciosamente "DONE".
        raise RuntimeError("El grafo llegó a 'end' sin marcar task_completed")

    answer = str(last_message.content)
    if last_message.name == "human_approval":
        await job_store.mark_rejected(job_id, answer=answer)
        return

    await job_store.mark_done(
        job_id,
        answer=answer,
        steps=result["step_count"],
        contributions=list(result["contributions"]),
    )


async def run_job(
    job_id: str,
    request: str,
    *,
    job_store: JobStore,
    graph: HitlGraph,
) -> None:
    """Ejecuta un trabajo nuevo de punta a punta (o hasta la primera pausa HITL)."""

    await job_store.mark_running(job_id)
    try:
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=request)],
                "contributions": [],
                "next_agent": "researcher",
                "instruction": request,
                "task_completed": False,
                "step_count": 0,
            },
            config=_thread_config(job_id),
        )
        await _sync_state(job_store, job_id, result)
    except Exception as error:  # noqa: BLE001 - el estado FAILED es el contrato con el cliente
        logger.exception("job %s falló durante la ejecución del grafo", job_id)
        await job_store.mark_failed(job_id, error=str(error))


async def resume_job(
    job_id: str,
    *,
    approved: bool,
    comment: str | None,
    job_store: JobStore,
    graph: HitlGraph,
) -> None:
    """Reanuda un trabajo pausado en `human_approval` con la decisión humana."""

    await job_store.mark_running(job_id)
    try:
        result = await graph.ainvoke(
            Command(resume={"approved": approved, "comment": comment}),
            config=_thread_config(job_id),
        )
        await _sync_state(job_store, job_id, result)
    except Exception as error:  # noqa: BLE001
        logger.exception("job %s falló durante la reanudación HITL", job_id)
        await job_store.mark_failed(job_id, error=str(error))
