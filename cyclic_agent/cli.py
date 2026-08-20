"""Interfaz de línea de comandos para ejecutar un turno persistente."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from cyclic_agent.agent import create_agent
from cyclic_agent.config import AgentSettings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Interpreta argumentos sin mezclar routing del agente con la interfaz."""

    parser = argparse.ArgumentParser(
        description="Agente ReAct de pedidos con memoria SQLite."
    )
    parser.add_argument("prompt", help="Pregunta en lenguaje natural")
    parser.add_argument(
        "--thread-id",
        default="demo-cliente-102",
        help="Identificador estable de la conversación",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        help="Ruta opcional para guardar la traza observable como JSON",
    )
    return parser.parse_args(argv)


async def async_main(argv: Sequence[str] | None = None) -> int:
    """Carga configuración y ejecuta el agente sin bloquear el event loop."""

    args = parse_args(argv)
    settings = AgentSettings.from_env()
    async with create_agent(settings) as agent:
        response = await agent.ask(
            args.prompt,
            thread_id=args.thread_id,
            trace_path=args.trace,
        )
    print(response)
    return 0


def main() -> int:
    """Punto de entrada síncrono mínimo para la CLI asíncrona."""

    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
