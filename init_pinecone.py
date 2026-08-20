from __future__ import annotations

from pinecone import Pinecone

from cloud_rag.config import CloudRAGSettings
from cloud_rag.index import ensure_serverless_index


def main() -> None:
    try:
        settings = CloudRAGSettings.from_env()
    except ValueError as error:
        raise SystemExit(str(error)) from error
    client = Pinecone(api_key=settings.pinecone_api_key)
    description = ensure_serverless_index(client, settings)
    print(
        f"Índice '{settings.index_name}' listo | "
        f"dimensión={description.dimension} | métrica={description.metric} | "
        f"namespace='{settings.namespace}'"
    )


if __name__ == "__main__":
    main()
