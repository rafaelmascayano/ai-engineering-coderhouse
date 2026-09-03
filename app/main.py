"""API FastAPI del Módulo 7: encola tareas del orquestador y expone su estado.

Los endpoints nunca llaman al LLM ni al grafo de forma síncrona: `POST
/tasks` solo crea el registro en Redis y agenda `run_job` como
`BackgroundTask` de FastAPI (una tarea `asyncio`, no un hilo bloqueante); el
event loop queda libre de inmediato y el cliente hace polling sobre `GET
/tasks/{job_id}`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.config import APISettings
from app.graph import build_hitl_graph
from app.jobs import JobStatus, JobStore
from app.observability import init_observability
from app.worker import resume_job, run_job
from orchestrator.config import OrchestratorSettings


class TaskCreateRequest(BaseModel):
    request: str = Field(min_length=1, description="Solicitud en lenguaje natural")


class TaskCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class ApprovalRequest(BaseModel):
    approved: bool
    comment: str | None = None


class TaskStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    request: str
    answer: str | None = None
    steps: int | None = None
    contributions: list[dict[str, Any]] | None = None
    error: str | None = None
    pending_approval: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = APISettings.from_env()
    observability_provider = init_observability()

    redis_client: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
    job_store = JobStore(redis_client, ttl_seconds=settings.job_ttl_seconds)

    async with AsyncRedisSaver.from_conn_string(settings.redis_url) as checkpointer:
        await checkpointer.asetup()
        graph = build_hitl_graph(
            checkpointer=checkpointer,
            settings=OrchestratorSettings.from_env(),
        )

        app.state.settings = settings
        app.state.redis_client = redis_client
        app.state.job_store = job_store
        app.state.graph = graph
        app.state.observability_provider = observability_provider

        try:
            yield
        finally:
            await redis_client.aclose()


app = FastAPI(
    title="Orquestador multi-agente — API de producción",
    description="Módulo 7: colas asíncronas, estado en Redis, HITL y trazas.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    await app.state.redis_client.ping()
    return {
        "status": "ok",
        "observability_provider": app.state.observability_provider,
    }


@app.post("/tasks", response_model=TaskCreateResponse, status_code=202)
async def create_task(
    payload: TaskCreateRequest, background_tasks: BackgroundTasks
) -> TaskCreateResponse:
    job_store: JobStore = app.state.job_store
    job_id = await job_store.create(payload.request)

    background_tasks.add_task(
        run_job,
        job_id,
        payload.request,
        job_store=job_store,
        graph=app.state.graph,
    )
    return TaskCreateResponse(job_id=job_id, status=JobStatus.PENDING)


@app.get("/tasks/{job_id}", response_model=TaskStatusResponse)
async def get_task(job_id: str) -> TaskStatusResponse:
    job_store: JobStore = app.state.job_store
    record = await job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job_id no encontrado")
    return TaskStatusResponse(**record)


@app.post("/tasks/{job_id}/approve", response_model=TaskStatusResponse)
async def approve_task(
    job_id: str, payload: ApprovalRequest, background_tasks: BackgroundTasks
) -> TaskStatusResponse:
    job_store: JobStore = app.state.job_store
    record = await job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job_id no encontrado")
    if record["status"] != JobStatus.WAITING_APPROVAL.value:
        raise HTTPException(
            status_code=409,
            detail=f"job_id no está esperando aprobación (status={record['status']})",
        )

    background_tasks.add_task(
        resume_job,
        job_id,
        approved=payload.approved,
        comment=payload.comment,
        job_store=job_store,
        graph=app.state.graph,
    )
    record["status"] = JobStatus.RUNNING.value
    return TaskStatusResponse(**record)
