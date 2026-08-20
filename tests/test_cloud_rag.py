from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tiktoken
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from cloud_rag.config import (
    OPENROUTER_BASE_URL,
    CloudRAGSettings,
    create_embeddings,
)
from cloud_rag.documents import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    load_documents,
    split_documents,
)
from cloud_rag.evaluation import evaluate_retriever, load_golden_set
from cloud_rag.index import validate_index
from cloud_rag.ingestion import ingest_to_pinecone
from cloud_rag.retriever import RAGSystem, tokenize_lexical


def settings(tmp_path: Path, *, dimension: int = 3) -> CloudRAGSettings:
    return CloudRAGSettings(
        pinecone_api_key="pinecone-test",
        embedding_api_key="openrouter-test",
        embedding_provider="openrouter",
        index_name="test-rag-index",
        namespace="tenant-a",
        cloud="aws",
        region="us-east-1",
        embedding_model="fake-embedding",
        embedding_dimension=dimension,
        data_dir=tmp_path,
        top_k=5,
        candidate_k=10,
        semantic_weight=0.6,
        lexical_weight=0.4,
    )


def test_settings_and_embeddings_use_openrouter_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "pinecone-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")

    configured = CloudRAGSettings.from_env()
    embeddings = create_embeddings(configured)

    assert configured.embedding_api_key == "openrouter-test"
    assert configured.embedding_provider == "openrouter"
    assert embeddings.model == "openai/text-embedding-3-small"
    assert embeddings.openai_api_base == OPENROUTER_BASE_URL
    assert embeddings.check_embedding_ctx_length is False


class FakeEncoding:
    def encode(self, text: str, **_: Any) -> list[str]:
        return text.split()


def test_loader_supports_markdown_json_and_enriched_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _: FakeEncoding())
    (tmp_path / "01_api_python.md").write_text(
        "# API\n\n" + "pinecone namespace vector técnico " * 750,
        encoding="utf-8",
    )
    (tmp_path / "manual.json").write_text(
        json.dumps(
            [
                {
                    "text": "El método upsert inserta vectores.",
                    "page": 7,
                    "category": "api",
                    "tags": ["pinecone", "upsert"],
                }
            ]
        ),
        encoding="utf-8",
    )

    chunks = split_documents(load_documents(tmp_path))

    assert CHUNK_SIZE == 650
    assert CHUNK_OVERLAP == 80
    assert len(chunks) > 2
    required = {
        "document_id",
        "source",
        "page",
        "category",
        "tags",
        "chunk_id",
        "chunk_index",
    }
    assert required.issubset(chunks[0].metadata)
    json_chunk = next(
        chunk for chunk in chunks if chunk.metadata["source"] == "manual.json"
    )
    assert json_chunk.metadata["page"] == 7
    assert json_chunk.metadata["tags"] == ["pinecone", "upsert"]


def test_validate_index_detects_dimension_mismatch(tmp_path: Path) -> None:
    description = SimpleNamespace(dimension=512, metric="cosine")

    with pytest.raises(ValueError, match="Dimensión incompatible"):
        validate_index(description, settings(tmp_path, dimension=1536))


def test_validate_index_rejects_non_serverless_index(tmp_path: Path) -> None:
    description = SimpleNamespace(
        dimension=3,
        metric="cosine",
        spec=SimpleNamespace(pod=SimpleNamespace(environment="us-east1-gcp")),
    )

    with pytest.raises(ValueError, match="no es Serverless"):
        validate_index(description, settings(tmp_path))


class FakeIndex:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.deleted_namespaces: list[str] = []

    def delete(self, *, delete_all: bool, namespace: str) -> None:
        assert delete_all is True
        self.deleted_namespaces.append(namespace)

    def describe_index_stats(self) -> dict[str, Any]:
        return {"namespaces": {}}

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> None:
        self.upserts.append({"vectors": vectors, "namespace": namespace})


class FakePinecone:
    def __init__(self) -> None:
        self.index = FakeIndex()

    def has_index(self, _: str) -> bool:
        return True

    def describe_index(self, _: str) -> SimpleNamespace:
        return SimpleNamespace(
            dimension=3,
            metric="cosine",
            status=SimpleNamespace(ready=True),
        )

    def Index(self, _: str) -> FakeIndex:
        return self.index


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]


def test_ingestion_upserts_text_metadata_and_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _: FakeEncoding())
    client = FakePinecone()
    documents = [
        Document(
            page_content="contenido técnico para recuperar",
            metadata={
                "document_id": "manual.md",
                "source": "manual.md",
                "page": 3,
                "category": "manual",
                "tags": ["python"],
            },
        )
    ]

    count = ingest_to_pinecone(
        settings(tmp_path),
        client=client,
        embeddings=FakeEmbeddings(),
        documents=documents,
        replace_namespace=True,
    )

    assert count == 1
    assert client.index.deleted_namespaces == ["tenant-a"]
    call = client.index.upserts[0]
    assert call["namespace"] == "tenant-a"
    metadata = call["vectors"][0]["metadata"]
    assert metadata["text"] == "contenido técnico para recuperar"
    assert metadata["source"] == "manual.md"
    assert metadata["page"] == 3
    assert metadata["tags"] == ["python"]


class StaticRetriever:
    def __init__(self, sources_by_question: dict[str, list[str]]) -> None:
        self.sources_by_question = sources_by_question

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[Document]:
        sources = self.sources_by_question[query][: (top_k or 5)]
        return [
            Document(page_content=source, metadata={"document_id": source})
            for source in sources
        ]


def test_evaluation_calculates_precision_and_recall_at_five(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps(
            [
                {"pregunta": "q1", "documento_id_esperado": "a.md"},
                {"pregunta": "q2", "documento_id_esperado": "b.md"},
            ]
        ),
        encoding="utf-8",
    )
    golden = load_golden_set(golden_path)
    retriever = StaticRetriever(
        {
            "q1": ["a.md", "x.md", "y.md", "z.md", "w.md"],
            "q2": ["x.md", "y.md", "z.md", "w.md", "v.md"],
        }
    )

    result = evaluate_retriever(retriever, golden, k=5)

    assert result.precision_at_k == pytest.approx(0.1)
    assert result.recall_at_k == pytest.approx(0.5)
    assert result.cases == 2


def test_precision_counts_each_useful_chunk_but_recall_deduplicates_source(
    tmp_path: Path,
) -> None:
    golden = [{"pregunta": "q", "documento_id_esperado": "a.md"}]
    retriever = StaticRetriever({"q": ["a.md", "a.md", "a.md", "x.md", "y.md"]})

    result = evaluate_retriever(retriever, golden, k=5)

    assert result.precision_at_k == pytest.approx(0.6)
    assert result.recall_at_k == pytest.approx(1.0)


def test_rag_system_encapsulates_ensemble_and_returns_top_five(tmp_path: Path) -> None:
    documents = [
        Document(
            page_content=f"Documento técnico {number} sobre artículo {number}.",
            metadata={
                "document_id": f"doc-{number}.md",
                "source": f"doc-{number}.md",
                "chunk_id": f"doc-{number}.md#chunk-1",
            },
        )
        for number in range(1, 7)
    ]
    vector = BM25Retriever.from_documents(documents, k=10)
    bm25 = BM25Retriever.from_documents(documents, k=10)
    system = RAGSystem(
        settings(tmp_path),
        vector_retriever=vector,
        bm25_retriever=bm25,
    )

    results = system.query("artículo 6")

    assert isinstance(system.ensemble_retriever, EnsembleRetriever)
    assert len(results) == 5
    assert any(document.metadata["source"] == "doc-6.md" for document in results)


def test_lexical_tokenizer_normalizes_technical_terms_and_article_numbers() -> None:
    assert tokenize_lexical("Pinecone.Index — Artículo 32.-") == [
        "pinecone",
        "index",
        "artículo",
        "32",
    ]
