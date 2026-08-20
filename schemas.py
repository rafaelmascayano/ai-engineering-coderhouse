from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

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
