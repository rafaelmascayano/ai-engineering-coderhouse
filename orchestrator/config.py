"""Configuración validada del orquestador."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class OrchestratorSettings:
    """Valores de ejecución inmutables cargados desde variables de entorno."""

    model: str = "openrouter/free"
    temperature: float = 0.0
    max_steps: int = 6

    @classmethod
    def from_env(cls) -> OrchestratorSettings:
        """Carga `.env` y valida el techo de pasos que evita el Supervisor infinito."""

        load_dotenv()
        max_steps = int(os.getenv("ORCHESTRATOR_MAX_STEPS", "6"))
        if not 2 <= max_steps <= 20:
            raise ValueError("ORCHESTRATOR_MAX_STEPS debe estar entre 2 y 20")

        temperature = float(os.getenv("ORCHESTRATOR_TEMPERATURE", "0"))
        if not 0 <= temperature <= 2:
            raise ValueError("ORCHESTRATOR_TEMPERATURE debe estar entre 0 y 2")

        return cls(
            model=os.getenv("ORCHESTRATOR_MODEL", "openrouter/free"),
            temperature=temperature,
            max_steps=max_steps,
        )
