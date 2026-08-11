from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from llm_clients.config import Configuration
from schemas import ChatMessage, ModelResponse


class BaseLLMClient(ABC):
    def __init__(self, config: Configuration) -> None:
        self.config = config

    @abstractmethod
    async def chat_completion(
        self, messages: list[ChatMessage], stream: bool = False
    ) -> ModelResponse | AsyncGenerator[str, None]:
        raise NotImplementedError

    @staticmethod
    def _serialize_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [message.model_dump() for message in messages]

    def _error_response(self, error: Exception) -> ModelResponse:
        return ModelResponse(
            content="",
            provider=self.config.provider.value,
            model=self.config.model_name,
            error=f"{type(error).__name__}: {error}",
        )

    @staticmethod
    def _stream_error(error: Exception) -> str:
        return f"[Error: {type(error).__name__}: {error}]"
