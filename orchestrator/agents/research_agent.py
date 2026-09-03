"""Agente de Investigación: consulta la base vectorial de pre-entregas previas."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from orchestrator.agents._shared import (
    ToolBindableChatModel,
    build_specialist_agent,
    run_specialist,
)
from orchestrator.state import OrchestratorState

RESEARCH_SYSTEM_PROMPT = """Eres un agente de investigación autónomo.
Recibes una única instrucción puntual, no el resto de la conversación.
Usa la herramienta `buscar_en_base_conocimiento` para fundamentar tu respuesta
exclusivamente con lo recuperado de la base vectorial (Ley 21.442 de
copropiedad inmobiliaria, indexada en Módulos 3 y 4). No inventes artículos,
cifras ni citas que no estén en los fragmentos recuperados. Si la búsqueda no
encuentra evidencia relevante, dilo explícitamente en vez de responder con
conocimiento propio. Responde en español, en un párrafo breve y autocontenido
porque tu respuesta se reenvía a otro agente sin el contexto de esta
conversación.
"""


class ResultadoFragmento(TypedDict):
    fuente: str
    fragmento: str


class KnowledgeSearchResult(TypedDict):
    status: str
    consulta: str
    resultados: list[ResultadoFragmento]


class BuscarConocimientoInput(BaseModel):
    consulta: str = Field(
        description="Consulta en lenguaje natural sobre la base vectorial"
    )


def make_knowledge_search_tool(
    retriever: Runnable[str, list[Document]],
) -> BaseTool:
    """Envuelve un retriever de LangChain como herramienta del especialista.

    El retriever es una dependencia inyectada (no un singleton global) para
    que las pruebas puedan reemplazar la base vectorial real por un doble
    determinista sin tocar red ni disco.
    """

    async def _buscar(consulta: str) -> KnowledgeSearchResult:
        documents = await retriever.ainvoke(consulta)
        resultados: list[ResultadoFragmento] = [
            {
                "fuente": str(document.metadata.get("source", "fuente-desconocida")),
                "fragmento": document.page_content[:500],
            }
            for document in documents
        ]
        return {
            "status": "ok" if resultados else "not_found",
            "consulta": consulta,
            "resultados": resultados,
        }

    return StructuredTool.from_function(
        coroutine=_buscar,
        name="buscar_en_base_conocimiento",
        description=(
            "Busca en la base vectorial de pre-entregas anteriores (ChromaDB, "
            "Ley 21.442) los fragmentos más relevantes para una consulta en "
            "lenguaje natural. Úsala antes de afirmar cualquier hecho legal."
        ),
        args_schema=BuscarConocimientoInput,
    )


@lru_cache(maxsize=1)
def default_retriever() -> Runnable[str, list[Document]]:
    """Abre perezosamente el mismo índice Chroma que usan los Módulos 3 y 4."""

    from rag_chain import create_retriever
    from rag_config import RAGSettings

    return create_retriever(RAGSettings.from_env(require_api_key=True))


def build_research_node(
    model: ToolBindableChatModel,
    *,
    retriever: Runnable[str, list[Document]] | None = None,
) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    """Crea el nodo `researcher` del grafo, acotado a su propia herramienta."""

    tool = make_knowledge_search_tool(
        retriever if retriever is not None else default_retriever()
    )
    agent = build_specialist_agent(model, [tool], RESEARCH_SYSTEM_PROMPT)

    async def researcher_node(state: OrchestratorState) -> dict[str, Any]:
        answer = await run_specialist(agent, state["instruction"])
        return {
            "messages": [AIMessage(content=answer, name="researcher")],
            "contributions": [{"agent": "researcher", "summary": answer}],
        }

    return researcher_node
