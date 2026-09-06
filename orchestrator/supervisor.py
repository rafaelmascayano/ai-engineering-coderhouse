"""Nodo Supervisor: router jerárquico y controlador del techo de pasos."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal, Protocol

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, ConfigDict, Field

from orchestrator.state import AgentContribution, NextAgent, OrchestratorState
from schemas import TextoNoVacio

logger = logging.getLogger(__name__)

_MAX_DECISION_ATTEMPTS = 3

SUPERVISOR_SYSTEM_PROMPT = """Eres el Supervisor de un equipo con dos especialistas:

- researcher: busca evidencia en la base vectorial de pre-entregas anteriores
  (Ley 21.442 de copropiedad inmobiliaria). Solo sabe buscar; no calcula ni
  analiza.
- analyst: analiza datos ya obtenidos (sentimiento de un texto o cálculos
  aritméticos explícitos). No tiene acceso a la base vectorial: si necesita un
  dato, ese dato debe habérselo entregado antes el researcher.

Rúbrica de suficiencia (decide "end" solo si se cumple):
1. Si la solicitud requiere un hecho externo (legal, documental) y todavía no
   hay un aporte de researcher, next="researcher".
2. Si la solicitud pide además un análisis (sentimiento, cálculo, validación)
   y todavía no hay un aporte de analyst para ese análisis, next="analyst".
3. Nunca envíes al mismo especialista dos veces seguidas por la misma tarea ya
   cumplida; si su aporte ya cubre lo pedido, avanza al siguiente paso o a
   "end".
4. Cuando los aportes necesarios ya estén presentes, next="end" y en
   "instruction" escribe la respuesta final para el usuario, combinando en
   prosa (no en lista) lo aportado por researcher y por analyst.
5. Si la solicitud es puramente factual sin ningún cálculo o análisis
   pedido, no delegues en analyst: usa solo researcher y luego "end".

6. Valida la calidad, no solo la presencia de un aporte: researcher debe
   aportar evidencia y fuente; analyst debe responder el análisis solicitado
   con los datos recibidos. Si falta algo, devuelve una instrucción específica
   al especialista responsable indicando qué falta y qué debe corregir.
7. Si dos aportes se contradicen, pide verificar el dato contra la fuente al
   researcher o repetir el cálculo al analyst. Conserva los aportes anteriores
   para trazabilidad; usa la corrección solo si está sustentada. No resuelvas
   un conflicto por mayoría ni por ser el último mensaje recibido.
8. Incluye en la instrucción del analyst el fragmento o los datos concretos
   que debe procesar, junto con su fuente cuando corresponda. No le pidas
   consultar un historial al que no tiene acceso.
9. Si no es posible resolver la falta de evidencia o el conflicto, finaliza
   explicando la limitación; no presentes como validado un resultado incierto.

Responde SIEMPRE con el JSON pedido, sin texto adicional fuera de él.
"""

SUPERVISOR_HUMAN_TEMPLATE = """SOLICITUD ORIGINAL DEL USUARIO:
{user_request}

APORTES RECIBIDOS HASTA AHORA:
{contributions}

FORMATO DE SALIDA:
{format_instructions}
"""


class SupervisorDecision(BaseModel):
    """Decisión de ruteo: a quién delegar ahora, o la respuesta final."""

    model_config = ConfigDict(extra="forbid")

    next: NextAgent = Field(description='"researcher", "analyst" o "end".')
    instruction: TextoNoVacio = Field(
        description=(
            "Instrucción puntual y autocontenida para el especialista elegido "
            '(sin metadata del sistema), o la respuesta final si next="end".'
        )
    )


class SupervisorChatModel(Protocol):
    """Contrato mínimo: recibe mensajes de chat y devuelve un `AIMessage`."""

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage: ...


_PARSER = PydanticOutputParser(pydantic_object=SupervisorDecision)


def format_contributions(contributions: list[AgentContribution]) -> str:
    """Reduce el historial de aportes a texto compacto para el prompt del Supervisor.

    Deliberadamente no incluye los mensajes crudos de cada especialista (con
    sus llamadas a herramientas): solo el resumen final de cada uno, para no
    contaminar el contexto del Supervisor con detalles de implementación.
    """

    if not contributions:
        return "(sin aportes todavía)"
    return "\n".join(
        f"- {contribution['agent']}: {contribution['summary']}"
        for contribution in contributions
    )


def synthesize_fallback(contributions: list[AgentContribution]) -> str:
    """Respuesta de mejor esfuerzo al alcanzar el techo de pasos del Supervisor."""

    if not contributions:
        return (
            "No fue posible completar el análisis dentro del límite de pasos "
            "configurado, y ningún especialista alcanzó a aportar información."
        )
    resumen = " ".join(contribution["summary"] for contribution in contributions)
    return (
        "Se alcanzó el límite de pasos del Supervisor antes de confirmar que "
        f"la tarea estaba completa. Con lo recopilado hasta ahora: {resumen}"
    )


async def _decide(
    model: SupervisorChatModel, base_messages: list[BaseMessage]
) -> SupervisorDecision:
    """Reintenta hasta tres veces una decisión que incumple el esquema.

    Cada intento conserva la solicitud y los aportes originales. Los errores
    de transporte siguen a cargo del cliente; aquí solo se corrige el formato.
    """

    messages = list(base_messages)
    last_error: OutputParserException | None = None

    for attempt in range(1, _MAX_DECISION_ATTEMPTS + 1):
        response = await model.ainvoke(messages)
        raw_content = str(response.content)
        try:
            return _PARSER.parse(raw_content)
        except OutputParserException as error:
            last_error = error
            logger.warning(
                "Supervisor: salida no parseable en el intento %s/%s",
                attempt,
                _MAX_DECISION_ATTEMPTS,
            )
            messages = [
                *base_messages,
                HumanMessage(
                    content=(
                        "Tu respuesta anterior no era JSON válido según el "
                        "formato pedido. Respondé ÚNICAMENTE con el objeto "
                        "JSON, sin ningún texto antes o después."
                    )
                ),
            ]

    raise RuntimeError(
        "El Supervisor no devolvió una decisión JSON válida tras "
        f"{_MAX_DECISION_ATTEMPTS} intentos"
    ) from last_error


def build_supervisor_node(
    model: SupervisorChatModel, *, max_steps: int
) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    """Crea el nodo `supervisor`: decide el próximo especialista o finaliza."""

    async def supervisor_node(state: OrchestratorState) -> dict[str, Any]:
        contributions = state["contributions"]
        step_count = state["step_count"] + 1

        if step_count > max_steps:
            fallback = synthesize_fallback(contributions)
            return {
                "next_agent": "end",
                "instruction": fallback,
                "step_count": step_count,
                "task_completed": True,
                "messages": [AIMessage(content=fallback, name="supervisor")],
            }

        user_request = _first_human_message(state["messages"])
        prompt = SUPERVISOR_HUMAN_TEMPLATE.format(
            user_request=user_request,
            contributions=format_contributions(contributions),
            format_instructions=_PARSER.get_format_instructions(),
        )
        decision = await _decide(
            model,
            [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
        )

        update: dict[str, Any] = {
            "next_agent": decision.next,
            "instruction": decision.instruction,
            "step_count": step_count,
        }
        if decision.next == "end":
            update["task_completed"] = True
            update["messages"] = [
                AIMessage(content=decision.instruction, name="supervisor")
            ]
        return update

    return supervisor_node


def route_supervisor(
    state: OrchestratorState,
) -> Literal["researcher", "analyst", "end"]:
    """Arista condicional: traduce la decisión del Supervisor al nombre del nodo."""

    return state["next_agent"]


def _first_human_message(messages: Sequence[BaseMessage]) -> str:
    for message in messages:
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("El estado no contiene la solicitud original del usuario")
