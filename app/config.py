"""Configuración validada de la API (Módulo 7)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class APISettings:
    """Valores de ejecución inmutables cargados desde variables de entorno."""

    redis_url: str = "redis://localhost:6379/0"
    job_ttl_seconds: int = 24 * 60 * 60
    critical_agents: frozenset[str] = frozenset({"analyst"})

    @classmethod
    def from_env(cls) -> APISettings:
        """Carga `.env` y valida la TTL de los registros de trabajo en Redis."""

        load_dotenv()
        job_ttl_seconds = int(os.getenv("API_JOB_TTL_SECONDS", str(24 * 60 * 60)))
        if job_ttl_seconds <= 0:
            raise ValueError("API_JOB_TTL_SECONDS debe ser positivo")

        raw_critical = os.getenv("API_CRITICAL_AGENTS", "analyst")
        critical_agents = frozenset(
            agent.strip() for agent in raw_critical.split(",") if agent.strip()
        )

        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            job_ttl_seconds=job_ttl_seconds,
            critical_agents=critical_agents,
        )
