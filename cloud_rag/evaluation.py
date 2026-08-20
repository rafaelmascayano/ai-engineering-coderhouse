from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int | None = None) -> list[Document]: ...


@dataclass(frozen=True)
class EvaluationResult:
    precision_at_k: float
    recall_at_k: float
    cases: int
    k: int
    details: list[dict[str, Any]]


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("El Golden Set debe ser una lista JSON no vacía")
    for case in payload:
        if not isinstance(case, dict) or not str(case.get("pregunta", "")).strip():
            raise ValueError("Cada caso debe incluir una pregunta no vacía")
        if not case.get("documento_id_esperado") and not case.get(
            "documentos_relevantes"
        ):
            raise ValueError("Cada caso debe declarar al menos un documento relevante")
    return payload


def _relevant_ids(case: dict[str, Any]) -> set[str]:
    values = case.get("documentos_relevantes")
    if isinstance(values, list) and values:
        return {str(value) for value in values}
    return {str(case["documento_id_esperado"])}


def evaluate_retriever(
    retriever: Retriever,
    golden_set: list[dict[str, Any]],
    *,
    k: int = 5,
) -> EvaluationResult:
    """Calcula macro Precision@k y Recall@k sobre documentos fuente."""

    if k < 1:
        raise ValueError("k debe ser mayor que cero")
    if not golden_set:
        raise ValueError("El Golden Set no puede estar vacío")

    precision_values: list[float] = []
    recall_values: list[float] = []
    details: list[dict[str, Any]] = []
    for case in golden_set:
        question = str(case["pregunta"])
        relevant = _relevant_ids(case)
        documents = retriever.retrieve(question, top_k=k)
        retrieved = [
            str(document.metadata.get("document_id") or document.metadata.get("source"))
            for document in documents[:k]
        ]
        # Precision mide cuántos chunks del Top-k pertenecen a fuentes útiles.
        # Recall deduplica por fuente para medir cobertura del Golden Set.
        relevant_chunks = sum(document_id in relevant for document_id in retrieved)
        retrieved_relevant_sources = relevant.intersection(retrieved)
        precision = relevant_chunks / k
        recall = len(retrieved_relevant_sources) / len(relevant)
        precision_values.append(precision)
        recall_values.append(recall)
        details.append(
            {
                "pregunta": question,
                "relevantes": sorted(relevant),
                "recuperados": retrieved,
                "precision_at_k": precision,
                "recall_at_k": recall,
            }
        )

    return EvaluationResult(
        precision_at_k=sum(precision_values) / len(precision_values),
        recall_at_k=sum(recall_values) / len(recall_values),
        cases=len(golden_set),
        k=k,
        details=details,
    )
