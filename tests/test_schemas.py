import pytest
from pydantic import ValidationError

from enums import Provider
from schemas import (
    ChatMessage,
    ExtraccionTecnica,
    ModelResponse,
    NivelCriticidad,
)


def test_chat_message_accepts_valid_input() -> None:
    message = ChatMessage(role="user", content="¿Qué es la entropía?")

    assert message.role == "user"
    assert message.content == "¿Qué es la entropía?"


@pytest.mark.parametrize(
    ("role", "content"),
    [
        ("invalid", "Hola"),
        ("user", ""),
    ],
)
def test_chat_message_rejects_invalid_input(role: str, content: str) -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role=role, content=content)  # type: ignore[arg-type]


def test_model_response_defaults_to_no_error() -> None:
    response = ModelResponse(content="Respuesta", provider="openai", model="test")

    assert response.provider is Provider.OPENAI
    assert response.error is None


def test_model_response_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        ModelResponse(
            content="Respuesta",
            provider="unknown",  # type: ignore[arg-type]
            model="test",
        )


def test_extraccion_tecnica_accepts_and_normalizes_valid_input() -> None:
    result = ExtraccionTecnica(
        tecnologias=[" FastAPI ", "Redis"],
        nivel_de_criticidad="alta",
        resumen_tecnico=" Servicio fuera de línea. ",
    )

    assert result.tecnologias == ["FastAPI", "Redis"]
    assert result.nivel_de_criticidad is NivelCriticidad.ALTA
    assert result.resumen_tecnico == "Servicio fuera de línea."


@pytest.mark.parametrize(
    "data",
    [
        {
            "tecnologias": [],
            "nivel_de_criticidad": "baja",
            "resumen_tecnico": "Sin impacto.",
        },
        {
            "tecnologias": ["   "],
            "nivel_de_criticidad": "baja",
            "resumen_tecnico": "Sin impacto.",
        },
        {
            "tecnologias": ["Redis"],
            "nivel_de_criticidad": "urgente",
            "resumen_tecnico": "Sin impacto.",
        },
        {
            "tecnologias": ["Redis"],
            "nivel_de_criticidad": "media",
            "resumen_tecnico": "   ",
        },
        {
            "tecnologias": ["Redis"],
            "nivel_de_criticidad": "media",
            "resumen_tecnico": "Latencia elevada.",
            "campo_extra": True,
        },
    ],
)
def test_extraccion_tecnica_rejects_invalid_contract(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate(data)
