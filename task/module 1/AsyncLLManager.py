import os
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from anthropic import APIError as AnthropicAPIError
from anthropic import AsyncAnthropic
from openai import APIError as OpenAIAPIError
from openai import AsyncOpenAI

from ConfigurationManager import Configuration, Providers
from schemas import ChatMessage, ModelResponse


class AsyncLLMManager(ABC):
    def __init__(self, config: Configuration):
        self.config = config

    @abstractmethod
    async def chat_completion(
        self, messages: list[ChatMessage], stream: bool = False
    ) -> ModelResponse | AsyncGenerator[str, None]:
        pass

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


class AnthropicClient(AsyncLLMManager):
    def __init__(self, config: Configuration):
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


class OpenAIClient(AsyncLLMManager):
    def __init__(self, config: Configuration):
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
                provider=self.config.provider.value,
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


class LLMFactory:
    @staticmethod
    def create_client(config: Configuration) -> AsyncLLMManager:
        if config.provider == Providers.ANTHROPIC:
            return AnthropicClient(config)
        elif config.provider == Providers.OPENAI:
            return OpenAIClient(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
