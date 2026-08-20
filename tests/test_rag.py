import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import rag_chain as rag_module
from ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ingest_documents,
    load_source_documents,
    split_documents,
)
from rag_chain import GroundingError, build_rag_chain, get_rag_response
from rag_config import RAGSettings
from schemas import RAGResponse


def retriever_with(documents: list[Document]) -> RunnableLambda:
    async def retrieve(_: str) -> list[Document]:
        return documents

    return RunnableLambda(retrieve)


def model_with(payload: dict[str, Any]) -> RunnableLambda:
    async def respond(_: Any) -> AIMessage:
        return AIMessage(content=json.dumps(payload, ensure_ascii=False))

    return RunnableLambda(respond)


def law_document() -> Document:
    return Document(
        page_content=(
            "Artículo 12.- Para efectos de la administración del condominio se "
            "considerarán los siguientes órganos: asamblea de copropietarios, "
            "comité de administración, administrador y subadministrador."
        ),
        metadata={
            "source": "02_administracion_condominio.txt",
            "chunk_id": "02_administracion_condominio.txt#chunk-001",
        },
    )


class CountingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [[float(len(text) % 17), 1.0, 0.5] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 17), 1.0, 0.5]


def test_grounded_question_returns_answer_and_retrieved_reference() -> None:
    expected_reference = {
        "source": "02_administracion_condominio.txt",
        "chunk_id": "02_administracion_condominio.txt#chunk-001",
    }
    chain = build_rag_chain(
        retriever=retriever_with([law_document()]),
        model=model_with(
            {
                "answer": (
                    "Los órganos son la asamblea, el comité, el administrador "
                    "y el subadministrador."
                ),
                "references": [expected_reference],
            }
        ),
    )

    result = asyncio.run(chain.ainvoke("¿Cuáles son los órganos de administración?"))

    assert isinstance(result, RAGResponse)
    assert result.references[0].model_dump() == expected_reference


def test_trap_question_returns_exact_unknown_without_references() -> None:
    chain = build_rag_chain(
        retriever=retriever_with([law_document()]),
        model=model_with({"answer": "No lo sé", "references": []}),
    )

    result = asyncio.run(
        chain.ainvoke("¿Quién ganó el Premio Nobel de Física en 1921?")
    )

    assert result.answer == "No lo sé"
    assert result.references == []


def test_chain_rejects_reference_not_returned_by_retriever() -> None:
    chain = build_rag_chain(
        retriever=retriever_with([law_document()]),
        model=model_with(
            {
                "answer": "Respuesta sin respaldo real.",
                "references": [
                    {"source": "inventado.txt", "chunk_id": "inventado#001"}
                ],
            }
        ),
    )

    with pytest.raises(GroundingError, match="no recuperadas"):
        asyncio.run(chain.ainvoke("Pregunta"))


def test_get_rag_response_rejects_blank_query_without_building_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_module,
        "get_chain",
        lambda: pytest.fail("No debe construir la cadena para una consulta vacía"),
    )

    with pytest.raises(ValueError, match="no vacío"):
        asyncio.run(get_rag_response("   "))


def test_loader_cleans_pdf_headers_and_splitter_adds_chunk_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OfflineEncoding:
        def encode(
            self,
            text: str,
            *,
            allowed_special: Any = None,
            disallowed_special: Any = "all",
        ) -> list[str]:
            del allowed_special, disallowed_special
            return text.split()

    monkeypatch.setattr(tiktoken, "get_encoding", lambda _: OfflineEncoding())

    source = tmp_path / "ejemplo.txt"
    source.write_text(
        "Ley 21442\n"
        "Biblioteca del Congreso Nacional de Chile - www.leychile.cl - documento "
        "generado el 20-Ago-2026 página 1 de 70\n"
        + ("Este es contenido legal verificable. " * 900),
        encoding="utf-8",
    )

    documents = load_source_documents(tmp_path)
    chunks = split_documents(documents)

    assert "Biblioteca del Congreso" not in documents[0].page_content
    assert len(chunks) > 1
    assert CHUNK_SIZE == 600
    assert CHUNK_OVERLAP == 50
    assert chunks[0].metadata == {
        "source": "ejemplo.txt",
        "chunk_id": "ejemplo.txt#chunk-001",
        "chunk_index": 1,
    }


def test_ingestion_persists_and_skips_unchanged_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OfflineEncoding:
        def encode(self, text: str, **_: Any) -> list[str]:
            return text.split()

    monkeypatch.setattr(tiktoken, "get_encoding", lambda _: OfflineEncoding())
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fuente.md").write_text(
        "# Fuente\n\n" + ("Contenido legal de prueba. " * 800),
        encoding="utf-8",
    )
    settings = RAGSettings(
        data_dir=data_dir,
        vectorstore_dir=tmp_path / "vectorstore",
        collection_name="test_collection",
        embedding_model="fake-embedding",
        chat_model="fake-chat",
        top_k=4,
        api_key="",
    )
    embeddings = CountingEmbeddings()

    first_count = ingest_documents(settings, embeddings=embeddings)
    calls_after_first_ingestion = embeddings.document_calls
    second_count = ingest_documents(settings, embeddings=embeddings)

    assert first_count > 1
    assert second_count == first_count
    assert calls_after_first_ingestion > 0
    assert embeddings.document_calls == calls_after_first_ingestion
