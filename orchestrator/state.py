"""Esquema de estado compartido entre el Supervisor y los especialistas."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState

NextAgent = Literal["researcher", "analyst", "end"]


class AgentContribution(TypedDict):
    """Aporte trazable de un especialista: quién dijo qué, en una sola frase.

    Se acumula (nunca se sobrescribe) para que el Supervisor pueda evaluar
    en cualquier paso qué información ya fue obtenida sin tener que releer el
    historial completo de mensajes de cada especialista.
    """

    agent: Literal["researcher", "analyst"]
    summary: str


class OrchestratorState(MessagesState):
    """Estado del grafo jerárquico.

    `messages` conserva el reducer de `MessagesState` (acumula, no reemplaza).
    `contributions` usa `operator.add` como reducer explícito por el mismo
    motivo: cada nodo especialista solo conoce su propio aporte, y el estado
    compartido es responsable de no perderlo en los saltos asíncronos entre
    nodos. `next_agent`, `instruction`, `task_completed` y `step_count` sí se
    reemplazan en cada turno del Supervisor, porque describen la decisión
    *actual* del router, no un historial.
    """

    contributions: Annotated[list[AgentContribution], operator.add]
    next_agent: NextAgent
    instruction: str
    task_completed: bool
    step_count: int
