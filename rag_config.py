from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class RAGSettings:
    """Configuración compartida por la ingesta y la recuperación."""

    data_dir: Path
    vectorstore_dir: Path
    collection_name: str
    embedding_model: str
    chat_model: str
    top_k: int
    api_key: str

    @classmethod
    def from_env(cls, *, require_api_key: bool = True) -> RAGSettings:
        load_dotenv(PROJECT_ROOT / ".env")

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if require_api_key and not api_key:
            raise ValueError(
                "Falta OPENROUTER_API_KEY. Copia .env.example a .env y configura "
                "la clave."
            )

        top_k = int(os.getenv("RAG_TOP_K", "4"))
        if not 3 <= top_k <= 5:
            raise ValueError("RAG_TOP_K debe estar entre 3 y 5")

        return cls(
            data_dir=Path(os.getenv("RAG_DATA_DIR", PROJECT_ROOT / "data")),
            vectorstore_dir=Path(
                os.getenv("RAG_VECTORSTORE_DIR", PROJECT_ROOT / "vectorstore")
            ),
            collection_name=os.getenv("RAG_COLLECTION", "ley_21442"),
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free"
            ),
            chat_model=os.getenv("RAG_CHAT_MODEL", "openrouter/free"),
            top_k=top_k,
            api_key=api_key,
        )


def create_embeddings(settings: RAGSettings) -> OpenAIEmbeddings:
    """Crea embeddings OpenAI-compatible a través de OpenRouter."""

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key,
        base_url=OPENROUTER_BASE_URL,
        # Los modelos de OpenRouter no necesariamente usan el tokenizer de OpenAI.
        # Los chunks ya están limitados a 600 tokens durante la ingesta.
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )
