from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from enums import Provider


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelResponse(BaseModel):
    content: str
    provider: Provider
    model: str
    error: str | None = None


class NivelCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


TextoNoVacio = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ExtraccionTecnica(BaseModel):
    """Contrato validado para la extracción de entidades técnicas."""

    model_config = ConfigDict(extra="forbid")

    tecnologias: list[TextoNoVacio] = Field(
        min_length=1,
        description="Tecnologías mencionadas explícitamente en el texto.",
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Impacto técnico estimado: baja, media o alta.",
    )
    resumen_tecnico: TextoNoVacio = Field(
        description="Resumen técnico breve y basado únicamente en el texto.",
    )


class RAGReference(BaseModel):
    """Referencia exacta a un fragmento recuperado desde ChromaDB."""

    model_config = ConfigDict(extra="forbid")

    source: TextoNoVacio
    chunk_id: TextoNoVacio


class RAGResponse(BaseModel):
    """Respuesta grounded con referencias verificables."""

    model_config = ConfigDict(extra="forbid")

    answer: TextoNoVacio = Field(
        description='Respuesta en español o exactamente "No lo sé".'
    )
    references: list[RAGReference] = Field(
        description="Fragmentos recuperados que respaldan la respuesta."
    )

    @model_validator(mode="after")
    def validate_unknown_answer(self) -> RAGResponse:
        if self.answer == "No lo sé":
            if self.references:
                raise ValueError('"No lo sé" no debe incluir referencias')
        elif not self.references:
            raise ValueError("Una respuesta factual debe incluir referencias")
        return self
