import os
from collections.abc import AsyncGenerator

from openai import APIError as OpenAIAPIError
from openai import AsyncOpenAI

from llm_clients.base import BaseLLMClient
from llm_clients.config import Configuration
from schemas import ChatMessage, ModelResponse


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing")
        self.client = AsyncOpenAI(api_key=api_key)

    async def chat_completion(
        self, messages: list[ChatMessage], stream: bool = False
    ) -> ModelResponse | AsyncGenerator[str, None]:
        if stream:
            return self._stream_response(messages)

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=self._serialize_messages(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return ModelResponse(
                content=response.choices[0].message.content or "",
                provider=self.config.provider,
                model=self.config.model_name,
            )
        except OpenAIAPIError as error:
            return self._error_response(error)

    async def _stream_response(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=self._serialize_messages(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except OpenAIAPIError as error:
            yield self._stream_error(error)
