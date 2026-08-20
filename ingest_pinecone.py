from __future__ import annotations

import argparse

from cloud_rag.config import CloudRAGSettings
from cloud_rag.ingestion import ingest_to_pinecone


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingesta documentos técnicos en Pinecone Serverless"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--replace-namespace",
        action="store_true",
        help="Elimina primero todos los vectores del namespace configurado.",
    )
    args = parser.parse_args()

    try:
        settings = CloudRAGSettings.from_env()
    except ValueError as error:
        raise SystemExit(str(error)) from error
    count = ingest_to_pinecone(
        settings,
        batch_size=args.batch_size,
        replace_namespace=args.replace_namespace,
    )
    print(
        f"Ingesta completa: {count} chunks en índice='{settings.index_name}', "
        f"namespace='{settings.namespace}'."
    )


if __name__ == "__main__":
    main()
