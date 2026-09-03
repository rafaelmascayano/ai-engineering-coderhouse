"""Demo del Módulo 6: delegación Supervisor -> researcher -> analyst -> END."""

from __future__ import annotations

import argparse
import asyncio

from orchestrator.runner import run_orchestrator

DEFAULT_REQUESTS = (
    "Busca qué dice el reglamento sobre el pago de gastos comunes y calcula "
    "el total de una deuda de 3 cuotas de 45000 cada una.",
    "Busca cómo describe el reglamento las sanciones por incumplimiento y "
    "evalúa si el tono de ese fragmento es positivo o negativo.",
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Demo del orquestador multi-agente")
    parser.add_argument(
        "request",
        nargs="?",
        help="Solicitud única. Si se omite, ejecuta los dos casos de ejemplo.",
    )
    args = parser.parse_args()
    requests = (args.request,) if args.request else DEFAULT_REQUESTS

    for request in requests:
        print(f"\n{'=' * 70}\nSolicitud: {request}\n{'=' * 70}")
        result = await run_orchestrator(request)
        print(f"\nRespuesta final:\n{result.answer}")
        print(f"\nPasos del Supervisor: {result.steps}")
        print("Aportes trazados:")
        for contribution in result.contributions:
            print(f"  - [{contribution['agent']}] {contribution['summary']}")


if __name__ == "__main__":
    asyncio.run(main())
