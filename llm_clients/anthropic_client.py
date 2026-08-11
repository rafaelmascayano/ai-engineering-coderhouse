import os
from collections.abc import AsyncGenerator

from anthropic import APIError as AnthropicAPIError
from anthropic import AsyncAnthropic

from llm_clients.base import BaseLLMClient
from llm_clients.config import Configuration
from schemas import ChatMessage, ModelResponse


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is missing")
        self.client = AsyncAnthropic(api_key=api_key)

    async def chat_completion(
        self, messages: list[ChatMessage], stream: bool = False
    ) -> ModelResponse | AsyncGenerator[str, None]:
        if stream:
            return self._stream_response(messages)

        try:
            response = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=self._serialize_messages(messages),
            )
            return ModelResponse(
                content=response.content[0].text,
                provider=self.config.provider.value,
                model=self.config.model_name,
            )
        except AnthropicAPIError as error:
            return self._error_response(error)

    async def _stream_response(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        try:
            async with self.client.messages.stream(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=self._serialize_messages(messages),
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except AnthropicAPIError as error:
            yield self._stream_error(error)
