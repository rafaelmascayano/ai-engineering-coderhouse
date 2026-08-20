from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_config import RAGSettings

CHUNK_SIZE = 600
CHUNK_OVERLAP = 50
SUPPORTED_EXTENSIONS = {".txt", ".md"}
MANIFEST_FILENAME = "ingestion_manifest.json"

PDF_HEADER_RE = re.compile(r"(?m)^Ley 21442\s*$")
PDF_FOOTER_RE = re.compile(
    r"(?m)^Biblioteca del Congreso Nacional de Chile - www\.leychile\.cl - "
    r"documento generado el .*? página \d+ de \d+\s*$"
)


def _clean_text(text: str) -> str:
    """Elimina ruido repetitivo de la exportación del PDF de Ley Chile."""

    text = text.replace("\f", "\n")
    text = PDF_HEADER_RE.sub("", text)
    text = PDF_FOOTER_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_source_documents(data_dir: Path) -> list[Document]:
    """Carga de forma determinista todos los archivos TXT y Markdown."""

    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No hay archivos .txt o .md en {data_dir}")

    documents: list[Document] = []
    for path in paths:
        content = _clean_text(path.read_text(encoding="utf-8"))
        if not content:
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={"source": path.relative_to(data_dir).as_posix()},
            )
        )
    if not documents:
        raise ValueError(f"Los archivos de {data_dir} no contienen texto utilizable")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Fragmenta por tokens y agrega identificadores citables a cada chunk."""

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[Document] = []
    for document in documents:
        source_chunks = splitter.split_documents([document])
        for position, chunk in enumerate(source_chunks, start=1):
            chunk_id = f"{document.metadata['source']}#chunk-{position:03d}"
            chunk.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": position,
                }
            )
            chunks.append(chunk)
    return chunks


def _source_fingerprint(settings: RAGSettings) -> str:
    digest = hashlib.sha256()
    digest.update(settings.embedding_model.encode())
    digest.update(f"{CHUNK_SIZE}:{CHUNK_OVERLAP}".encode())
    for path in sorted(settings.data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            digest.update(path.relative_to(settings.data_dir).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _collection_count(settings: RAGSettings) -> int | None:
    if not settings.vectorstore_dir.exists():
        return None
    client = chromadb.PersistentClient(path=str(settings.vectorstore_dir))
    collections = {collection.name for collection in client.list_collections()}
    if settings.collection_name not in collections:
        return None
    return client.get_collection(settings.collection_name).count()


def _read_manifest(settings: RAGSettings) -> dict[str, Any] | None:
    manifest_path = settings.vectorstore_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_manifest(
    settings: RAGSettings, *, fingerprint: str, chunk_count: int
) -> None:
    manifest = {
        "fingerprint": fingerprint,
        "chunk_count": chunk_count,
        "embedding_model": settings.embedding_model,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "collection_name": settings.collection_name,
    }
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    (settings.vectorstore_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ingest_documents(
    settings: RAGSettings | None = None,
    *,
    force: bool = False,
    embeddings: Any | None = None,
) -> int:
    """Indexa `/data` y devuelve la cantidad de chunks persistidos.

    Si colección y manifiesto coinciden con las fuentes, no vuelve a llamar al
    proveedor de embeddings. Una colección desactualizada exige `force=True`.
    """

    settings = settings or RAGSettings.from_env()
    fingerprint = _source_fingerprint(settings)
    manifest = _read_manifest(settings)
    collection_count = _collection_count(settings)

    if (
        not force
        and manifest is not None
        and manifest.get("fingerprint") == fingerprint
        and collection_count == manifest.get("chunk_count")
        and collection_count
    ):
        return collection_count

    if collection_count is not None and not force:
        raise RuntimeError(
            "La colección existe pero no coincide con los documentos actuales. "
            "Ejecuta `python ingest.py --force` para reconstruirla."
        )

    documents = load_source_documents(settings.data_dir)
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("El proceso de chunking no produjo fragmentos")

    if collection_count is not None:
        client = chromadb.PersistentClient(path=str(settings.vectorstore_dir))
        client.delete_collection(settings.collection_name)

    embedding_client = embeddings or OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key,
    )
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embedding_client,
        persist_directory=str(settings.vectorstore_dir),
    )
    ids = [
        hashlib.sha256(
            f"{chunk.metadata['chunk_id']}:{chunk.page_content}".encode()
        ).hexdigest()
        for chunk in chunks
    ]
    vectorstore.add_documents(chunks, ids=ids)
    _write_manifest(settings, fingerprint=fingerprint, chunk_count=len(chunks))
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta local para el RAG")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconstruye la colección si ya existe.",
    )
    args = parser.parse_args()
    settings = RAGSettings.from_env()
    count = ingest_documents(settings, force=args.force)
    print(
        f"Colección '{settings.collection_name}' lista: {count} chunks en "
        f"{settings.vectorstore_dir}"
    )


if __name__ == "__main__":
    main()
