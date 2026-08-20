from __future__ import annotations

import hashlib
import time
from typing import Any

from langchain_core.documents import Document

from cloud_rag.config import CloudRAGSettings, create_embeddings
from cloud_rag.documents import load_documents, split_documents
from cloud_rag.index import ensure_serverless_index


def _vector_id(document: Document) -> str:
    stable_value = f"{document.metadata['chunk_id']}\0{document.page_content}"
    return hashlib.sha256(stable_value.encode("utf-8")).hexdigest()


def _pinecone_metadata(document: Document) -> dict[str, Any]:
    """Incluye el texto original para no depender de una base adicional."""

    metadata = {
        "text": document.page_content,
        "document_id": str(document.metadata["document_id"]),
        "source": str(document.metadata["source"]),
        "page": int(document.metadata["page"]),
        "category": str(document.metadata["category"]),
        "tags": [str(tag) for tag in document.metadata.get("tags", [])],
        "chunk_id": str(document.metadata["chunk_id"]),
        "chunk_index": int(document.metadata["chunk_index"]),
    }
    return metadata


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _wait_for_namespace_empty(
    index: Any,
    namespace: str,
    *,
    timeout_seconds: float = 120,
    poll_seconds: float = 1,
) -> None:
    """Evita que un delete eventualmente consistente borre nuevos upserts."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        stats = index.describe_index_stats()
        namespaces = _field(stats, "namespaces", {}) or {}
        summary = namespaces.get(namespace) if isinstance(namespaces, dict) else None
        if summary is None or int(_field(summary, "vector_count", 0)) == 0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"El namespace '{namespace}' no terminó de vaciarse en "
                f"{timeout_seconds:g} segundos"
            )
        time.sleep(poll_seconds)


def ingest_to_pinecone(
    settings: CloudRAGSettings,
    *,
    client: Any | None = None,
    embeddings: Any | None = None,
    documents: list[Document] | None = None,
    batch_size: int = 64,
    replace_namespace: bool = False,
) -> int:
    """Procesa documentos y sube los vectores al namespace configurado."""

    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor que cero")
    if client is None:
        from pinecone import Pinecone

        client = Pinecone(api_key=settings.pinecone_api_key)

    ensure_serverless_index(client, settings)
    index = client.Index(settings.index_name)
    if replace_namespace:
        index.delete(delete_all=True, namespace=settings.namespace)
        _wait_for_namespace_empty(index, settings.namespace)

    source_documents = documents or load_documents(settings.data_dir)
    chunks = split_documents(source_documents)
    embedding_client = embeddings or create_embeddings(settings)

    uploaded = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedding_client.embed_documents(
            [document.page_content for document in batch]
        )
        if any(len(vector) != settings.embedding_dimension for vector in vectors):
            dimensions = sorted({len(vector) for vector in vectors})
            raise ValueError(
                f"El proveedor devolvió dimensiones {dimensions}; el índice espera "
                f"{settings.embedding_dimension}"
            )
        records = [
            {
                "id": _vector_id(document),
                "values": vector,
                "metadata": _pinecone_metadata(document),
            }
            for document, vector in zip(batch, vectors, strict=True)
        ]
        index.upsert(vectors=records, namespace=settings.namespace)
        uploaded += len(records)
    return uploaded
