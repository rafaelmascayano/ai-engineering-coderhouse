from __future__ import annotations

import argparse
import asyncio

from rag_chain import get_rag_response

DEFAULT_QUESTIONS = (
    "¿Cuáles son los órganos de administración de un condominio?",
    "¿Quién ganó el Premio Nobel de Física en 1921?",
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta asíncrona del RAG")
    parser.add_argument(
        "question",
        nargs="?",
        help="Pregunta única. Si se omite, ejecuta los dos casos de prueba.",
    )
    args = parser.parse_args()
    questions = (args.question,) if args.question else DEFAULT_QUESTIONS

    for question in questions:
        response = await get_rag_response(question)
        print(f"\nPregunta: {question}")
        print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
