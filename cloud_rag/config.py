from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class CloudRAGSettings:
    """Configuración única para inicialización, ingesta y recuperación."""

    pinecone_api_key: str
    embedding_api_key: str
    embedding_provider: str
    index_name: str
    namespace: str
    cloud: str
    region: str
    embedding_model: str
    embedding_dimension: int
    data_dir: Path
    top_k: int
    candidate_k: int
    semantic_weight: float
    lexical_weight: float

    @classmethod
    def from_env(cls, *, require_api_keys: bool = True) -> CloudRAGSettings:
        load_dotenv(PROJECT_ROOT / ".env")

        pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        embedding_provider = (
            os.getenv("EMBEDDING_PROVIDER", "openrouter").strip().lower()
        )
        if embedding_provider not in {"openrouter", "openai"}:
            raise ValueError("EMBEDDING_PROVIDER debe ser 'openrouter' u 'openai'")
        embedding_api_key = (
            openrouter_api_key
            if embedding_provider == "openrouter"
            else openai_api_key
        )
        if require_api_keys and not pinecone_api_key:
            raise ValueError("Falta PINECONE_API_KEY en el archivo .env")
        if require_api_keys and not embedding_api_key:
            variable = (
                "OPENROUTER_API_KEY"
                if embedding_provider == "openrouter"
                else "OPENAI_API_KEY"
            )
            raise ValueError(f"Falta {variable} en el archivo .env")

        top_k = int(os.getenv("RAG_TOP_K", "5"))
        candidate_k = int(os.getenv("RAG_CANDIDATE_K", "10"))
        if top_k < 1:
            raise ValueError("RAG_TOP_K debe ser mayor que cero")
        if candidate_k < top_k:
            raise ValueError("RAG_CANDIDATE_K debe ser mayor o igual que RAG_TOP_K")

        semantic_weight = float(os.getenv("RAG_SEMANTIC_WEIGHT", "0.6"))
        lexical_weight = float(os.getenv("RAG_LEXICAL_WEIGHT", "0.4"))
        if semantic_weight < 0 or lexical_weight < 0:
            raise ValueError("Los pesos del recuperador no pueden ser negativos")
        weight_sum = semantic_weight + lexical_weight
        if weight_sum <= 0:
            raise ValueError("Al menos un peso del recuperador debe ser positivo")

        data_dir_value = os.getenv("RAG_DATA_DIR", "./data")
        data_dir = Path(data_dir_value)
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir

        index_name = os.getenv("INDEX_NAME", "ley-21442-rag").strip()
        namespace = os.getenv("PINECONE_NAMESPACE", "ley-21442").strip()
        default_embedding_dimension = (
            "2048" if embedding_provider == "openrouter" else "1536"
        )
        embedding_dimension = int(
            os.getenv("EMBEDDING_DIMENSION", default_embedding_dimension)
        )
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,43}[a-z0-9])?", index_name):
            raise ValueError(
                "INDEX_NAME debe tener 1-45 caracteres: minúsculas, números o guiones"
            )
        if not namespace:
            raise ValueError("PINECONE_NAMESPACE no puede estar vacío")
        if embedding_dimension < 1:
            raise ValueError("EMBEDDING_DIMENSION debe ser mayor que cero")

        default_embedding_model = (
            "nvidia/nemotron-3-embed-1b:free"
            if embedding_provider == "openrouter"
            else "text-embedding-3-small"
        )

        return cls(
            pinecone_api_key=pinecone_api_key,
            embedding_api_key=embedding_api_key,
            embedding_provider=embedding_provider,
            index_name=index_name,
            namespace=namespace,
            cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
            region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", default_embedding_model
            ).strip(),
            embedding_dimension=embedding_dimension,
            data_dir=data_dir.resolve(),
            top_k=top_k,
            candidate_k=candidate_k,
            semantic_weight=semantic_weight / weight_sum,
            lexical_weight=lexical_weight / weight_sum,
        )


def create_embeddings(settings: CloudRAGSettings) -> OpenAIEmbeddings:
    """Construye el mismo cliente de embeddings para ingesta y consultas."""

    kwargs: dict[str, object] = {}
    if settings.embedding_provider == "openrouter":
        kwargs.update(
            {
                "base_url": OPENROUTER_BASE_URL,
                "check_embedding_ctx_length": False,
                "model_kwargs": {"encoding_format": "float"},
            }
        )

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimension,
        api_key=settings.embedding_api_key,
        max_retries=3,
        request_timeout=30,
        **kwargs,
    )
