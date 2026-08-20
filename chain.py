from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError

from llm_clients import EnvironmentLoader, Provider
from schemas import ExtraccionTecnica

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MIN_PIPELINE_MAX_TOKENS = 2048
INCOMPLETE_FINISH_REASONS = {"length", "max_tokens", "content_filter"}
INCOMPLETE_STATUSES = {"incomplete", "failed", "cancelled"}

FORMAT_INSTRUCTIONS = """
Devuelve exactamente los campos definidos por el esquema de salida:
tecnologias, nivel_de_criticidad y resumen_tecnico. No agregues otros campos.
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Eres un analista de sistemas especializado en extracción de entidades técnicas.

Reglas:
- Extrae únicamente tecnologías, frameworks, bases de datos, servicios,
  protocolos o componentes mencionados explícitamente en el texto.
- No inventes tecnologías ausentes.
- Usa criticidad alta para caídas, pérdida de datos, incidentes de seguridad o
  bloqueos severos; media para degradación o errores recuperables; y baja para
  información operativa sin impacto relevante.
- Redacta el resumen técnico en español, de forma breve y factual.
            """.strip(),
        ),
        (
            "human",
            """
Texto de entrada:
{texto}

Instrucciones de formato:
{instrucciones_formato}
            """.strip(),
        ),
    ]
)


class StructuredOutputError(ValueError):
    """La respuesta del modelo no puede aceptarse como salida estructurada."""


def create_model() -> ChatOpenAI:
    """Crea ChatOpenAI usando la API compatible de OpenRouter."""

    config = EnvironmentLoader().get_configuration()
    if config.provider is not Provider.OPENROUTER:
        raise ValueError(
            "El pipeline de extracción requiere PROVIDER=openrouter en el entorno"
        )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing")

    pipeline_max_tokens = max(config.max_tokens, MIN_PIPELINE_MAX_TOKENS)
    if config.max_tokens < MIN_PIPELINE_MAX_TOKENS:
        logger.warning(
            "MAX_TOKENS=%s es insuficiente para modelos con razonamiento "
            "obligatorio; el pipeline usará %s",
            config.max_tokens,
            pipeline_max_tokens,
        )

    return ChatOpenAI(
        model=config.model_name,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=None,
        max_tokens=pipeline_max_tokens,
        timeout=config.timeout,
        max_retries=0,
        extra_body={
            "provider": {"require_parameters": True},
            "reasoning": {"effort": "minimal", "exclude": True},
        },
    )


def _log_attempt(chain_input: dict[str, str]) -> dict[str, str]:
    logger.info("Iniciando intento de extracción estructurada")
    return chain_input


def _validate_structured_output(response: Any) -> ExtraccionTecnica:
    if not isinstance(response, dict):
        logger.warning("El modelo devolvió un contenedor de salida inesperado")
        raise StructuredOutputError("La salida estructurada no es un diccionario")

    raw = response.get("raw")
    metadata = getattr(raw, "response_metadata", None) or {}
    finish_reason = metadata.get("finish_reason")
    status = metadata.get("status")

    if finish_reason in INCOMPLETE_FINISH_REASONS or status in INCOMPLETE_STATUSES:
        logger.warning(
            "Respuesta incompleta del modelo: finish_reason=%r status=%r",
            finish_reason,
            status,
        )
        raise StructuredOutputError(
            "La respuesta del modelo fue truncada o finalizó incompleta"
        )

    parsing_error = response.get("parsing_error")
    if parsing_error is not None:
        logger.warning("Falló el parseo o la validación de la salida estructurada")
        raise StructuredOutputError(
            "El modelo devolvió JSON mal formado o incompleto"
        ) from parsing_error

    parsed = response.get("parsed")
    if not isinstance(parsed, ExtraccionTecnica):
        logger.warning("La respuesta no contiene un objeto ExtraccionTecnica válido")
        raise StructuredOutputError("Falta el objeto Pydantic validado")

    logger.info("Salida estructurada validada correctamente")
    return parsed


def _handle_length_finish_reason(
    structured_model: Runnable[Any, Any],
) -> Runnable[Any, Any]:
    async def invoke(model_input: Any, config: RunnableConfig) -> Any:
        try:
            return await structured_model.ainvoke(model_input, config=config)
        except LengthFinishReasonError as error:
            logger.warning(
                "El parser detectó finish_reason='length'; la respuesta se reintentará"
            )
            raise StructuredOutputError(
                "La respuesta alcanzó el límite de tokens antes de completar el JSON"
            ) from error

    return RunnableLambda(invoke)


def build_chain(
    model: Any | None = None,
) -> Runnable[dict[str, str], ExtraccionTecnica]:
    """Construye Prompt + LLM + parser/validador y agrega un reintento."""

    chat_model = model or create_model()
    structured_model = chat_model.with_structured_output(
        ExtraccionTecnica,
        method="json_schema",
        strict=True,
        include_raw=True,
    )
    resilient_structured_model = _handle_length_finish_reason(structured_model)

    base_chain = (
        RunnableLambda(_log_attempt)
        | prompt
        | resilient_structured_model
        | RunnableLambda(_validate_structured_output)
    )
    return base_chain.with_retry(
        retry_if_exception_type=(StructuredOutputError,),
        wait_exponential_jitter=False,
        stop_after_attempt=2,
    )


@lru_cache(maxsize=1)
def get_chain() -> Runnable[dict[str, str], ExtraccionTecnica]:
    return build_chain()


async def process_text(text: str) -> ExtraccionTecnica:
    """Procesa texto sin estructurar y devuelve un modelo Pydantic validado."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text debe ser un string no vacío")

    logger.info("Comenzando el procesamiento del texto técnico")
    try:
        result = await get_chain().ainvoke(
            {
                "texto": text.strip(),
                "instrucciones_formato": FORMAT_INSTRUCTIONS,
            }
        )
    except StructuredOutputError:
        logger.exception("La extracción falló después del reintento automático")
        raise

    logger.info("Procesamiento técnico completado")
    return result
