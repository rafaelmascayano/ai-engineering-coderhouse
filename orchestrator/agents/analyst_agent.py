"""Agente de Análisis: sentimiento léxico y cálculos numéricos, sin red."""

from __future__ import annotations

import ast
import asyncio
import operator as op
from collections.abc import Awaitable, Callable
from typing import Any, Final, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool

from orchestrator.agents._shared import (
    ToolBindableChatModel,
    build_specialist_agent,
    run_specialist,
)
from orchestrator.state import OrchestratorState

ANALYST_SYSTEM_PROMPT = """Eres un agente de análisis autónomo.
Recibes una única instrucción puntual, no el resto de la conversación; toda
la información que necesitas analizar viene incluida en esa instrucción.
Usa `analizar_sentimiento` para evaluar el tono de un texto y
`calcular_expresion` para resolver operaciones aritméticas explícitas. No
calcules de memoria ni estimes sentimiento sin llamar a la herramienta
correspondiente. Si la instrucción no trae datos suficientes para ninguna
herramienta, dilo explícitamente. Responde en español, en un párrafo breve y
autocontenido porque tu respuesta se reenvía a otro agente.
"""

_PALABRAS_POSITIVAS: Final[frozenset[str]] = frozenset(
    {
        "bueno",
        "buena",
        "excelente",
        "correcto",
        "beneficio",
        "beneficios",
        "protege",
        "protección",
        "garantiza",
        "garantía",
        "seguro",
        "seguridad",
        "orden",
        "acuerdo",
        "mejora",
        "positivo",
        "positiva",
        "favorable",
        "eficiente",
        "transparencia",
    }
)
_PALABRAS_NEGATIVAS: Final[frozenset[str]] = frozenset(
    {
        "malo",
        "mala",
        "sanción",
        "sanciones",
        "multa",
        "multas",
        "incumplimiento",
        "incumplimientos",
        "infracción",
        "infracciones",
        "conflicto",
        "conflictos",
        "problema",
        "problemas",
        "riesgo",
        "riesgos",
        "negativo",
        "negativa",
        "desfavorable",
        "deficiente",
        "deuda",
        "deudas",
    }
)

_ALLOWED_BINARY_OPERATORS: Final[
    dict[type[ast.operator], Callable[[float, float], float]]
] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
_ALLOWED_UNARY_OPERATORS: Final[dict[type[ast.unaryop], Callable[[float], float]]] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


class SentimentResult(TypedDict):
    status: str
    etiqueta: str
    puntaje: float
    palabras_positivas: int
    palabras_negativas: int


class CalculationResult(TypedDict):
    status: str
    expresion: str
    resultado: float | None
    error: str | None


def _evaluate_ast(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPERATORS:
        binary_function = _ALLOWED_BINARY_OPERATORS[type(node.op)]
        return binary_function(_evaluate_ast(node.left), _evaluate_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPERATORS:
        unary_function = _ALLOWED_UNARY_OPERATORS[type(node.op)]
        return unary_function(_evaluate_ast(node.operand))
    raise ValueError(f"Expresión no permitida: {ast.dump(node)}")


def evaluate_arithmetic(expresion: str) -> float:
    """Evalúa aritmética básica sin `eval`: solo +, -, *, /, //, %, ** y paréntesis."""

    parsed = ast.parse(expresion, mode="eval")
    return _evaluate_ast(parsed.body)


def score_sentiment(texto: str) -> SentimentResult:
    """Cuenta coincidencias léxicas en español; determinista, sin llamadas externas."""

    palabras = [word.strip(".,;:()[]¡!¿?\"'").lower() for word in texto.split()]
    positivas = sum(1 for word in palabras if word in _PALABRAS_POSITIVAS)
    negativas = sum(1 for word in palabras if word in _PALABRAS_NEGATIVAS)
    total = positivas + negativas
    puntaje = (positivas - negativas) / total if total else 0.0

    if puntaje > 0.15:
        etiqueta = "positivo"
    elif puntaje < -0.15:
        etiqueta = "negativo"
    else:
        etiqueta = "neutral"

    return {
        "status": "ok",
        "etiqueta": etiqueta,
        "puntaje": round(puntaje, 3),
        "palabras_positivas": positivas,
        "palabras_negativas": negativas,
    }


@tool
async def analizar_sentimiento(texto: str) -> SentimentResult:
    """Clasifica el tono de un texto en español como positivo, negativo o neutral.

    Usa esta herramienta cuando la instrucción pida evaluar el tono, la
    connotación o la carga emocional de un fragmento (por ejemplo, uno
    recuperado antes por el investigador). Devuelve `etiqueta`, un `puntaje`
    entre -1 y 1, y el conteo de palabras que sustentan la clasificación. No
    intenta detectar sarcasmo ni matices fuera del léxico conocido.
    """

    await asyncio.sleep(0)
    return score_sentiment(texto)


@tool
async def calcular_expresion(expresion: str) -> CalculationResult:
    """Evalúa una expresión aritmética explícita (+, -, *, /, //, %, **, paréntesis).

    Usa esta herramienta para cualquier cálculo numérico pedido en la
    instrucción, por ejemplo sumar cuotas o convertir montos. No admite
    variables, funciones ni texto: solo números y operadores. Si la expresión
    es inválida devuelve `status=error` con el motivo, en vez de inventar un
    resultado.
    """

    await asyncio.sleep(0)
    try:
        resultado = evaluate_arithmetic(expresion)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as error:
        return {
            "status": "error",
            "expresion": expresion,
            "resultado": None,
            "error": str(error),
        }
    return {
        "status": "ok",
        "expresion": expresion,
        "resultado": resultado,
        "error": None,
    }


ANALYST_TOOLS: Final[list[BaseTool]] = [analizar_sentimiento, calcular_expresion]


def build_analyst_node(
    model: ToolBindableChatModel,
) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    """Crea el nodo `analyst` del grafo, acotado a sus dos herramientas de cómputo."""

    agent = build_specialist_agent(model, ANALYST_TOOLS, ANALYST_SYSTEM_PROMPT)

    async def analyst_node(state: OrchestratorState) -> dict[str, Any]:
        answer = await run_specialist(agent, state["instruction"])
        return {
            "messages": [AIMessage(content=answer, name="analyst")],
            "contributions": [{"agent": "analyst", "summary": answer}],
        }

    return analyst_node
