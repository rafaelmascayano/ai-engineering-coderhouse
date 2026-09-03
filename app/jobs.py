"""Estado de los trabajos en Redis: el `job_id` es la unidad que ve el cliente.

Se mantiene deliberadamente separado del checkpointer de LangGraph
(`app/graph.py`): el checkpointer persiste el estado *interno* del grafo (para
poder reanudarlo tras un `interrupt()`), mientras que este módulo persiste el
estado *observable* por el cliente HTTP (`PENDING` -> ... -> `DONE`/`FAILED`).
Son dos preocupaciones distintas aunque comparten el mismo Redis.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis


class JobStatus(StrEnum):
    """Ciclo de vida observable de un trabajo encolado."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


_KEY_PREFIX = "orchestrator:job:"


def _key(job_id: str) -> str:
    return f"{_KEY_PREFIX}{job_id}"


class JobStore:
    """CRUD asíncrono de registros de trabajo, con TTL, sobre `redis.asyncio`."""

    def __init__(self, redis: Redis, *, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl_seconds = ttl_seconds

    async def create(self, request: str) -> str:
        job_id = str(uuid.uuid4())
        record = {
            "job_id": job_id,
            "status": JobStatus.PENDING.value,
            "request": request,
            "created_at": time.time(),
            "updated_at": time.time(),
            "answer": None,
            "steps": None,
            "contributions": None,
            "error": None,
            "pending_approval": None,
        }
        await self._save(job_id, record)
        return job_id

    async def get(self, job_id: str) -> dict[str, Any] | None:
        raw = await self._redis.get(_key(job_id))
        if raw is None:
            return None
        return json.loads(raw)

    async def mark_running(self, job_id: str) -> None:
        await self._update(job_id, status=JobStatus.RUNNING, pending_approval=None)

    async def mark_waiting_approval(self, job_id: str, payload: dict[str, Any]) -> None:
        await self._update(
            job_id, status=JobStatus.WAITING_APPROVAL, pending_approval=payload
        )

    async def mark_done(
        self, job_id: str, *, answer: str, steps: int, contributions: list[Any]
    ) -> None:
        await self._update(
            job_id,
            status=JobStatus.DONE,
            answer=answer,
            steps=steps,
            contributions=contributions,
            pending_approval=None,
        )

    async def mark_rejected(self, job_id: str, *, answer: str) -> None:
        await self._update(
            job_id,
            status=JobStatus.REJECTED,
            answer=answer,
            pending_approval=None,
        )

    async def mark_failed(self, job_id: str, *, error: str) -> None:
        await self._update(
            job_id, status=JobStatus.FAILED, error=error, pending_approval=None
        )

    async def _update(self, job_id: str, **fields: Any) -> None:
        record = await self.get(job_id)
        if record is None:
            raise KeyError(f"job {job_id} no existe")
        status = fields.pop("status", None)
        if status is not None:
            record["status"] = JobStatus(status).value
        record.update(fields)
        record["updated_at"] = time.time()
        await self._save(job_id, record)

    async def _save(self, job_id: str, record: dict[str, Any]) -> None:
        await self._redis.set(
            _key(job_id), json.dumps(record, ensure_ascii=False), ex=self._ttl_seconds
        )
