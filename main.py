import asyncio
import logging

from chain import process_text

SAMPLE_TEXT = """
Una API construida con FastAPI usa Redis como caché y PostgreSQL para
persistencia. Durante picos de tráfico, el pool de conexiones se agota y las
solicitudes terminan con timeout, dejando el servicio fuera de línea.
""".strip()


async def main() -> None:
    """Ejecuta el pipeline correspondiente a la pre-entrega 2."""
    result = await process_text(SAMPLE_TEXT)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    asyncio.run(main())
