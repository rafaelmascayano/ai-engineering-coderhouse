from __future__ import annotations

import time
from typing import Any

from cloud_rag.config import CloudRAGSettings


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def validate_index(index_description: Any, settings: CloudRAGSettings) -> None:
    """Falla pronto ante el error más costoso: dimensión o métrica incorrecta."""

    dimension = int(_field(index_description, "dimension", 0))
    metric = str(_field(index_description, "metric", ""))
    if dimension != settings.embedding_dimension:
        raise ValueError(
            f"Dimensión incompatible: el índice '{settings.index_name}' usa "
            f"{dimension}, pero {settings.embedding_model} está configurado con "
            f"{settings.embedding_dimension}. Crea otro índice o corrige "
            "EMBEDDING_DIMENSION."
        )
    if metric != "cosine":
        raise ValueError(
            f"Métrica incompatible: se esperaba 'cosine' y el índice usa {metric!r}"
        )
    spec = _field(index_description, "spec")
    if spec is not None and _field(spec, "serverless") is None:
        raise ValueError(
            f"El índice '{settings.index_name}' existe, pero no es Serverless"
        )


def ensure_serverless_index(
    client: Any,
    settings: CloudRAGSettings,
    *,
    timeout_seconds: float = 120,
    poll_seconds: float = 1,
) -> Any:
    """Crea el índice Serverless si falta y valida su contrato vectorial."""

    if not client.has_index(settings.index_name):
        from pinecone import ServerlessSpec

        client.create_index(
            name=settings.index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
            deletion_protection="disabled",
            tags={"application": "hybrid-rag", "environment": "education"},
        )

    deadline = time.monotonic() + timeout_seconds
    while True:
        description = client.describe_index(settings.index_name)
        status = _field(description, "status", {})
        if bool(_field(status, "ready", False)):
            validate_index(description, settings)
            return description
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"El índice '{settings.index_name}' no estuvo listo en "
                f"{timeout_seconds:g} segundos"
            )
        time.sleep(poll_seconds)
