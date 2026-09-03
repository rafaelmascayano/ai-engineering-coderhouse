"""Nodo Human-in-the-loop: pausa obligatoria antes de una delegación crítica.

En este orquestador, el especialista `analyst` es el que ejecuta herramientas
con "costo" real (`calcular_expresion` sobre montos de deuda, por ejemplo):
su salida puede terminar en una decisión financiera. Por eso el Supervisor no
delega en `analyst` directamente: siempre pasa antes por `human_approval`,
que usa `interrupt()` de LangGraph para suspender la ejecución del grafo
-persistiendo el estado en el checkpointer de Redis- hasta que un humano la
apruebe o la rechace vía `POST /tasks/{job_id}/approve`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from orchestrator.state import OrchestratorState

REJECTED_MESSAGE = (
    "La tarea fue detenida: un humano rechazó la ejecución de la herramienta "
    "crítica antes de que el especialista de análisis pudiera correrla."
)


def build_human_approval_node(
    *, critical_agent: str = "analyst"
) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    """Crea el nodo `human_approval`, colocado antes de `critical_agent`.

    `interrupt()` re-ejecuta la función completa al reanudar, así que el
    valor devuelto en la primera invocación (la solicitud de aprobación) no
    debe tener efectos secundarios: solo se usa para decidir el enrutamiento
    posterior una vez que llega la decisión humana.
    """

    async def human_approval_node(state: OrchestratorState) -> dict[str, Any]:
        decision = interrupt(
            {
                "reason": (
                    f"El Supervisor quiere delegar en '{critical_agent}', que "
                    "ejecuta herramientas con costo o efectos secundarios "
                    "(cálculos numéricos que pueden respaldar una decisión "
                    "financiera). Se requiere aprobación humana explícita."
                ),
                "agent": critical_agent,
                "instruction": state["instruction"],
            }
        )

        approved = bool(decision and decision.get("approved"))
        if approved:
            return {}

        comment = (decision or {}).get("comment")
        message = (
            REJECTED_MESSAGE if not comment else f"{REJECTED_MESSAGE} Motivo: {comment}"
        )
        return {
            "next_agent": "end",
            "instruction": message,
            "task_completed": True,
            "messages": [AIMessage(content=message, name="human_approval")],
        }

    return human_approval_node


def route_after_approval(
    state: OrchestratorState,
) -> Literal["proceed", "end"]:
    """Tras `human_approval`: sigue hacia el especialista o corta a `END`."""

    return "end" if state["task_completed"] else "proceed"
