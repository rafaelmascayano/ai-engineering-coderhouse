from __future__ import annotations

import argparse
from pathlib import Path

from cloud_rag.evaluation import evaluate_retriever, load_golden_set
from cloud_rag.retriever import RAGSystem

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa Precision@k y Recall@k del recuperador híbrido"
    )
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "golden_set.json",
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    try:
        system = RAGSystem()
    except ValueError as error:
        raise SystemExit(str(error)) from error
    cases = load_golden_set(args.golden_set)
    result = evaluate_retriever(system, cases, k=args.k)

    print(f"\nEvaluación del recuperador híbrido ({result.cases} preguntas)")
    print("=" * 62)
    for number, detail in enumerate(result.details, start=1):
        status = "OK" if detail["recall_at_k"] > 0 else "MISS"
        print(f"{number}. [{status}] {detail['pregunta']}")
        print(f"   Esperados: {', '.join(detail['relevantes'])}")
        print(f"   Top-{result.k}: {', '.join(detail['recuperados'])}")
    print("-" * 62)
    print(f"Precision@{result.k}: {result.precision_at_k:.2%}")
    print(f"Recall@{result.k}:    {result.recall_at_k:.2%}")


if __name__ == "__main__":
    main()
