"""Configuración validada del agente persistente."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class AgentSettings:
    """Valores de ejecución inmutables cargados desde variables de entorno."""

    model: str = "openrouter/free"
    temperature: float = 0.0
    database_path: Path = Path("checkpoints/agent.sqlite")
    recursion_limit: int = 10

    @classmethod
    def from_env(cls) -> AgentSettings:
        """Carga `.env` y valida los límites que protegen el ciclo del agente."""

        load_dotenv()
        recursion_limit = int(os.getenv("AGENT_RECURSION_LIMIT", "10"))
        if not 2 <= recursion_limit <= 50:
            raise ValueError("AGENT_RECURSION_LIMIT debe estar entre 2 y 50")

        temperature = float(os.getenv("AGENT_TEMPERATURE", "0"))
        if not 0 <= temperature <= 2:
            raise ValueError("AGENT_TEMPERATURE debe estar entre 0 y 2")

        return cls(
            model=os.getenv("AGENT_MODEL", "openrouter/free"),
            temperature=temperature,
            database_path=Path(
                os.getenv("AGENT_DB_PATH", "checkpoints/agent.sqlite")
            ),
            recursion_limit=recursion_limit,
        )
