"""Capa de observabilidad: envía trazas de cada ejecución a LangSmith o Phoenix.

`OBSERVABILITY_PROVIDER` elige el backend:

- `langsmith` (por defecto si hay `LANGCHAIN_API_KEY`): activa el tracing
  nativo de LangChain/LangGraph seteando las variables de entorno que su SDK
  ya inspecciona en cada `ainvoke`. No requiere decorar manualmente las
  llamadas al LLM: todo `Runnable` (incluidos los nodos del grafo y el
  `ChatOpenAI` de cada especialista) queda instrumentado automáticamente.
- `phoenix`: inicializa el colector OpenTelemetry de Arize Phoenix e
  instrumenta LangChain vía OpenInference, con el mismo efecto.
- cualquier otro valor (incluido vacío): no instrumenta nada, para que los
  tests offline y `pytest` no intenten exportar trazas.

`init_observability()` es idempotente: se puede llamar en el `lifespan` de
FastAPI sin riesgo de doble instrumentación.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INITIALIZED = False


def init_observability() -> str:
    """Configura el proveedor de trazas indicado por `OBSERVABILITY_PROVIDER`.

    Devuelve el nombre del proveedor efectivamente inicializado (`langsmith`,
    `phoenix` o `none`), útil para exponerlo en `/health`.
    """

    global _INITIALIZED

    provider = os.getenv("OBSERVABILITY_PROVIDER", "").strip().lower()
    if not provider:
        provider = "langsmith" if os.getenv("LANGCHAIN_API_KEY") else "none"

    if _INITIALIZED:
        return provider

    if provider == "langsmith":
        _init_langsmith()
    elif provider == "phoenix":
        _init_phoenix()
    elif provider != "none":
        logger.warning(
            "OBSERVABILITY_PROVIDER=%s desconocido; no se instrumenta.", provider
        )
        provider = "none"

    _INITIALIZED = True
    return provider


def _init_langsmith() -> None:
    """Activa el tracing nativo de LangChain hacia LangSmith.

    `run_orchestrator` ya construye `ChatOpenAI` (un `Runnable`); con estas
    variables seteadas antes de crear el grafo, LangSmith traza automáticamente
    cada nodo del `StateGraph`, cada llamada al LLM y cada tool call, sin tocar
    el código del orquestador del Módulo 6.
    """

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "orchestrator-module7")
    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.warning(
            "OBSERVABILITY_PROVIDER=langsmith pero falta LANGCHAIN_API_KEY; "
            "las trazas no se enviarán."
        )
    logger.info(
        "Observabilidad: LangSmith activado (proyecto=%s)",
        os.environ["LANGCHAIN_PROJECT"],
    )


def _init_phoenix() -> None:
    """Inicializa el colector de Arize Phoenix mediante OpenInference.

    Instrumenta LangChain a nivel de `Runnable`, por lo que cubre tanto el
    `StateGraph` del orquestador como los `ChatOpenAI` internos de cada
    especialista sin decoradores manuales adicionales.
    """

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError:
        logger.warning(
            "OBSERVABILITY_PROVIDER=phoenix pero faltan las dependencias "
            "arize-phoenix-otel / openinference-instrumentation-langchain."
        )
        return

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    tracer_provider = register(
        project_name=os.getenv("PHOENIX_PROJECT_NAME", "orchestrator-module7"),
        endpoint=f"{endpoint.rstrip('/')}/v1/traces",
        batch=True,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    logger.info("Observabilidad: Arize Phoenix activado (endpoint=%s)", endpoint)
