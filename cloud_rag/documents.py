from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 650
CHUNK_OVERLAP = 80
SUPPORTED_EXTENSIONS = {".json", ".md", ".markdown", ".pdf", ".txt"}

PDF_HEADER_RE = re.compile(r"(?m)^Ley 21442\s*$")
PDF_FOOTER_RE = re.compile(
    r"(?m)^Biblioteca del Congreso Nacional de Chile - www\.leychile\.cl - "
    r"documento generado el .*? página \d+ de \d+\s*$"
)
PAGE_NUMBER_RE = re.compile(r"página\s+(\d+)\s+de\s+\d+", re.IGNORECASE)


def _clean_text(text: str) -> str:
    text = PDF_HEADER_RE.sub("", text)
    text = PDF_FOOTER_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _category_for(path: Path) -> str:
    name = re.sub(r"^\d+[_-]*", "", path.stem.lower())
    return re.sub(r"[^a-z0-9áéíóúñ]+", "-", name).strip("-") or "general"


def _metadata(
    path: Path,
    data_dir: Path,
    *,
    page: int,
    category: str | None = None,
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    source = path.relative_to(data_dir).as_posix()
    normalized_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    inferred_category = category or _category_for(path)
    if not normalized_tags:
        normalized_tags = [inferred_category, path.suffix.lower().lstrip(".")]
    return {
        "document_id": source,
        "source": source,
        "page": page,
        "category": inferred_category,
        "tags": normalized_tags,
    }


def _load_pdf(path: Path, data_dir: Path) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - mensaje de instalación
        raise RuntimeError("Para cargar PDFs instala la dependencia pypdf") from error

    documents: list[Document] = []
    for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata=_metadata(path, data_dir, page=page_number),
                )
            )
    return documents


def _load_text(path: Path, data_dir: Path) -> list[Document]:
    raw_text = path.read_text(encoding="utf-8")
    pages = raw_text.split("\f") if "\f" in raw_text else [raw_text]
    documents: list[Document] = []
    for position, raw_page in enumerate(pages, start=1):
        match = PAGE_NUMBER_RE.search(raw_page)
        page_number = int(match.group(1)) if match else position
        text = _clean_text(raw_page)
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata=_metadata(path, data_dir, page=page_number),
                )
            )
    return documents


def _json_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("documents"), list):
        return payload["documents"]
    return [payload]


def _load_json(path: Path, data_dir: Path) -> list[Document]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents: list[Document] = []
    for position, item in enumerate(_json_items(payload), start=1):
        if isinstance(item, str):
            text = item
            item_metadata: dict[str, Any] = {}
        elif isinstance(item, dict):
            text = str(
                item.get("text")
                or item.get("content")
                or item.get("page_content")
                or ""
            )
            raw_metadata = item.get("metadata", {})
            item_metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            for key in ("page", "category", "tags"):
                if key in item:
                    item_metadata[key] = item[key]
        else:
            continue

        text = _clean_text(text)
        if not text:
            continue
        raw_tags = item_metadata.get("tags")
        tags = raw_tags if isinstance(raw_tags, list) else None
        metadata = _metadata(
            path,
            data_dir,
            page=int(item_metadata.get("page", position)),
            category=str(item_metadata.get("category") or _category_for(path)),
            tags=tags,
        )
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def load_documents(data_dir: Path) -> list[Document]:
    """Carga PDF, Markdown, JSON y TXT conservando metadatos de procedencia."""

    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No hay documentos compatibles en {data_dir}")

    documents: list[Document] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(_load_pdf(path, data_dir))
        elif suffix == ".json":
            documents.extend(_load_json(path, data_dir))
        else:
            documents.extend(_load_text(path, data_dir))
    if not documents:
        raise ValueError("Los archivos encontrados no contienen texto utilizable")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Divide a 650 tokens y añade identificadores deterministas de chunk."""

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[Document] = []
    source_positions: dict[str, int] = {}
    for document in documents:
        source = str(document.metadata["source"])
        for chunk in splitter.split_documents([document]):
            source_positions[source] = source_positions.get(source, 0) + 1
            chunk_index = source_positions[source]
            page = int(chunk.metadata.get("page", 1))
            chunk.metadata.update(
                {
                    "chunk_id": f"{source}#p{page}-chunk-{chunk_index:04d}",
                    "chunk_index": chunk_index,
                }
            )
            chunks.append(chunk)
    if not chunks:
        raise ValueError("El proceso de chunking no produjo fragmentos")
    return chunks
