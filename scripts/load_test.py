"""Prueba de carga: 5 peticiones concurrentes contra `POST /tasks`.

Uso:

    python scripts/load_test.py --url http://localhost:8000 --concurrency 5

Encola las N peticiones en paralelo con `asyncio.gather`, hace polling de
cada `job_id` hasta que termine (`DONE`/`FAILED`/`WAITING_APPROVAL`) e
imprime la latencia individual y un resumen. El costo por ejecución y la
latencia p95 "de verdad" se leen del dashboard de LangSmith/Phoenix para esta
misma corrida -- ver `screenshots/`.
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx

DEFAULT_REQUESTS = [
    "Busca qué dice el reglamento sobre gastos comunes y calcula 3 cuotas de 45000.",
    "Busca cómo describe el reglamento las sanciones por incumplimiento y evalúa "
    "si el tono de ese fragmento es positivo o negativo.",
    "Busca qué dice el reglamento sobre el fondo de reserva y calcula el 5% de 800000.",
    "Busca los órganos de administración del condominio y calcula 12500 * 3.",
    "Busca las multas por infracciones al reglamento y evalúa el tono de ese "
    "fragmento.",
]


async def _run_one(client: httpx.AsyncClient, request_text: str, index: int) -> None:
    start = time.perf_counter()
    response = await client.post("/tasks", json={"request": request_text})
    response.raise_for_status()
    job_id = response.json()["job_id"]

    status = "PENDING"
    while status in {"PENDING", "RUNNING"}:
        await asyncio.sleep(1.0)
        status_response = await client.get(f"/tasks/{job_id}")
        status_response.raise_for_status()
        status = status_response.json()["status"]

    elapsed = time.perf_counter() - start
    print(f"[{index}] job_id={job_id} status={status} latencia={elapsed:.2f}s")


async def main(base_url: str, concurrency: int) -> None:
    requests = (DEFAULT_REQUESTS * ((concurrency // len(DEFAULT_REQUESTS)) + 1))[
        :concurrency
    ]
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        started = time.perf_counter()
        await asyncio.gather(
            *(
                _run_one(client, request_text, index)
                for index, request_text in enumerate(requests, start=1)
            )
        )
        print(
            f"\nTotal pared-reloj para {concurrency} peticiones concurrentes: "
            f"{time.perf_counter() - started:.2f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.concurrency))
