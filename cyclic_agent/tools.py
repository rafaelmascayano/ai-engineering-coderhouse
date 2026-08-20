"""Herramientas asíncronas que simulan consultas a una base de pedidos."""

from __future__ import annotations

import asyncio
from typing import Final, TypedDict

from langchain_core.tools import BaseTool, tool


class OrderRecord(TypedDict):
    """Registro interno de un pedido simulado."""

    pedido_id: int
    cliente_id: int
    fecha: str
    total: int
    estado: str


class OrderSummary(TypedDict):
    """Respuesta estructurada de la consulta de resumen."""

    status: str
    cliente_id: int
    pedidos: int
    total: int


class LastOrderFound(TypedDict):
    """Pedido más reciente encontrado."""

    status: str
    cliente_id: int
    pedido_id: int
    fecha: str
    total: int
    estado: str


class OrderNotFound(TypedDict):
    """Observación que permite al modelo recuperarse de un ID desconocido."""

    status: str
    cliente_id: int
    message: str
    suggestion: str


type LastOrderResult = LastOrderFound | OrderNotFound


_ORDERS: Final[tuple[OrderRecord, ...]] = (
    {
        "pedido_id": 5001,
        "cliente_id": 102,
        "fecha": "2026-07-02",
        "total": 3500,
        "estado": "entregado",
    },
    {
        "pedido_id": 5008,
        "cliente_id": 102,
        "fecha": "2026-07-18",
        "total": 4500,
        "estado": "entregado",
    },
    {
        "pedido_id": 5015,
        "cliente_id": 102,
        "fecha": "2026-08-03",
        "total": 6500,
        "estado": "en preparación",
    },
    {
        "pedido_id": 5020,
        "cliente_id": 205,
        "fecha": "2026-08-09",
        "total": 9200,
        "estado": "entregado",
    },
)


@tool
async def buscar_resumen_pedidos(cliente_id: int) -> OrderSummary:
    """Consulta el resumen agregado de pedidos de un cliente por su ID numérico.

    Usa esta herramienta cuando el usuario pregunte cuántos pedidos realizó un
    cliente o cuál es el importe total acumulado. Devuelve exclusivamente
    `status`, `cliente_id`, `pedidos` y `total` (en pesos enteros, sin formato).
    No devuelve fecha, estado ni identificador del pedido más reciente: si el
    usuario también solicita esos datos debes llamar después a
    `buscar_ultimo_pedido`. Si `status` es `not_found`, no inventes datos:
    solicita que el usuario verifique el ID o reintenta solo si el contexto
    aporta otro ID inequívoco.
    """

    await asyncio.sleep(0)
    orders = [order for order in _ORDERS if order["cliente_id"] == cliente_id]
    return {
        "status": "ok" if orders else "not_found",
        "cliente_id": cliente_id,
        "pedidos": len(orders),
        "total": sum(order["total"] for order in orders),
    }


@tool
async def buscar_ultimo_pedido(cliente_id: int) -> LastOrderResult:
    """Busca el pedido cronológicamente más reciente de un cliente por su ID.

    Usa esta herramienta cuando el usuario pida "el último", "el más reciente",
    su fecha, estado, importe individual o número de pedido. La consulta devuelve
    un único registro con `pedido_id`, `fecha` ISO-8601, `total` en pesos enteros
    y `estado`. No calcula cantidad ni gasto acumulado; para esas preguntas usa
    además `buscar_resumen_pedidos`. Si no existe el cliente, devuelve
    `status=not_found` y una sugerencia para pedir aclaración, nunca datos
    fabricados.
    """

    await asyncio.sleep(0)
    orders = [order for order in _ORDERS if order["cliente_id"] == cliente_id]
    if not orders:
        return {
            "status": "not_found",
            "cliente_id": cliente_id,
            "message": "No existen pedidos para el cliente indicado.",
            "suggestion": "Pide al usuario que verifique el cliente_id.",
        }

    latest = max(orders, key=lambda order: (order["fecha"], order["pedido_id"]))
    return {
        "status": "ok",
        "cliente_id": latest["cliente_id"],
        "pedido_id": latest["pedido_id"],
        "fecha": latest["fecha"],
        "total": latest["total"],
        "estado": latest["estado"],
    }


ORDER_TOOLS: Final[list[BaseTool]] = [
    buscar_resumen_pedidos,
    buscar_ultimo_pedido,
]
