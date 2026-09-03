"""Interfaz de línea de comandos para una consulta del orquestador."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from orchestrator.runner import run_orchestrator


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orquestador multi-agente (Supervisor + researcher + analyst)."
    )
    parser.add_argument("request", help="Solicitud en lenguaje natural")
    parser.add_argument(
        "--trace",
        type=Path,
        help="Ruta opcional para guardar respuesta y aportes como JSON",
    )
    return parser.parse_args(argv)


async def async_main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = await run_orchestrator(args.request)

    print(result.answer)
    print(f"\n[pasos del Supervisor: {result.steps}]")
    for contribution in result.contributions:
        print(f"  - {contribution['agent']}: {contribution['summary']}")

    if args.trace is not None:
        payload = {
            "request": args.request,
            "answer": result.answer,
            "steps": result.steps,
            "contributions": result.contributions,
        }
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
