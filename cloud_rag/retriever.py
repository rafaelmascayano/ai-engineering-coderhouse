from __future__ import annotations

import re
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_pinecone import PineconeVectorStore

from cloud_rag.config import CloudRAGSettings, create_embeddings
from cloud_rag.documents import load_documents, split_documents
from cloud_rag.index import validate_index


def tokenize_lexical(text: str) -> list[str]:
    """Normaliza mayúsculas y puntuación sin perder términos ni números."""

    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class RAGSystem:
    """Recuperador híbrido: BM25 local + similitud vectorial en Pinecone."""

    def __init__(
        self,
        settings: CloudRAGSettings | None = None,
        *,
        client: Any | None = None,
        embeddings: Embeddings | None = None,
        documents: list[Document] | None = None,
        vector_retriever: BaseRetriever | None = None,
        bm25_retriever: BaseRetriever | None = None,
    ) -> None:
        self.settings = settings or CloudRAGSettings.from_env()
        self.embeddings = embeddings

        if vector_retriever is None:
            if client is None:
                from pinecone import Pinecone

                client = Pinecone(api_key=self.settings.pinecone_api_key)
            description = client.describe_index(self.settings.index_name)
            validate_index(description, self.settings)
            self.embeddings = self.embeddings or create_embeddings(self.settings)
            vectorstore = PineconeVectorStore(
                index=client.Index(self.settings.index_name),
                embedding=self.embeddings,
                text_key="text",
                namespace=self.settings.namespace,
            )
            vector_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.settings.candidate_k},
            )

        if bm25_retriever is None:
            corpus = documents or split_documents(
                load_documents(self.settings.data_dir)
            )
            bm25_retriever = BM25Retriever.from_documents(
                corpus,
                preprocess_func=tokenize_lexical,
            )
            bm25_retriever.k = self.settings.candidate_k

        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[
                self.settings.semantic_weight,
                self.settings.lexical_weight,
            ],
            id_key="chunk_id",
        )

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[Document]:
        """Devuelve los mejores documentos tras fusión por ranking ponderado."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query debe ser un string no vacío")
        limit = self.settings.top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k debe ser mayor que cero")
        return self.ensemble_retriever.invoke(query.strip())[:limit]

    def query(self, query: str) -> list[Document]:
        """Devuelve el Top-k configurado (Top-5 de forma predeterminada)."""

        return self.retrieve(query, top_k=self.settings.top_k)
