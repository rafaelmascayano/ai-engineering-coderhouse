import pytest
from pydantic import ValidationError

from schemas import ChatMessage, ModelResponse


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

    assert response.error is None
