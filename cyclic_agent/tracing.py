"""Serialización segura de la traza observable del ciclo ReAct."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def message_record(*, sequence: int, node: str, message: BaseMessage) -> dict[str, Any]:
    """Convierte un mensaje en un evento JSON sin razonamiento interno oculto."""

    record: dict[str, Any] = {
        "sequence": sequence,
        "node": node,
        "message_type": message.type,
        "content": message.content,
    }
    if isinstance(message, AIMessage) and message.tool_calls:
        record["tool_calls"] = [
            {
                "name": call["name"],
                "args": call["args"],
                "id": call["id"],
            }
            for call in message.tool_calls
        ]
    if isinstance(message, ToolMessage):
        record["tool_call_id"] = message.tool_call_id
        record["tool_name"] = message.name
        record["status"] = message.status
    return record


def records_from_update(
    update: Mapping[str, Any], *, start_sequence: int
) -> list[dict[str, Any]]:
    """Extrae mensajes de una actualización `stream_mode=updates`."""

    records: list[dict[str, Any]] = []
    sequence = start_sequence
    for node, state_update in update.items():
        if not isinstance(state_update, Mapping):
            continue
        messages = state_update.get("messages", [])
        if not isinstance(messages, Sequence):
            continue
        for message in messages:
            if isinstance(message, BaseMessage):
                records.append(
                    message_record(sequence=sequence, node=node, message=message)
                )
                sequence += 1
    return records


async def write_trace(
    path: Path, *, thread_id: str, prompt: str, events: Sequence[Mapping[str, Any]]
) -> None:
    """Escribe la traza fuera del event loop mediante una operación atómica."""

    payload = {
        "schema_version": 1,
        "thread_id": thread_id,
        "prompt": prompt,
        "note": (
            "Traza observable de mensajes, llamadas y resultados; no contiene "
            "cadena de pensamiento privada."
        ),
        "events": list(events),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(path)

    await asyncio.to_thread(_write)
