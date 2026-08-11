import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from llm_clients.base import BaseLLMClient
from llm_clients.config import Configuration, Provider
from llm_clients.factory import LLMFactory
from llm_clients.openai_client import OpenAIClient
from llm_clients.openrouter_client import OpenRouterClient
from schemas import ChatMessage, ModelResponse


class FakeStream(AsyncIterator[Any]):
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> "FakeStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration as error:
            raise StopAsyncIteration from error


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return FakeStream(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="Hola"))]
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(delta=SimpleNamespace(content=" mundo"))
                        ]
                    ),
                ]
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Respuesta"))]
        )


def make_openai_client() -> tuple[OpenAIClient, FakeCompletions]:
    config = Configuration(provider=Provider.OPENAI, model_name="test-model")
    completions = FakeCompletions()
    client = object.__new__(OpenAIClient)
    BaseLLMClient.__init__(client, config)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_openai_normal_generation() -> None:
    client, completions = make_openai_client()

    async def run() -> ModelResponse:
        response = await client.chat_completion(
            [ChatMessage(role="user", content="Hola")]
        )
        assert isinstance(response, ModelResponse)
        return response

    response = asyncio.run(run())

    assert response.content == "Respuesta"
    assert response.error is None
    assert completions.calls[0]["messages"] == [{"role": "user", "content": "Hola"}]


def test_openai_streaming_generation() -> None:
    client, completions = make_openai_client()

    async def run() -> list[str]:
        stream = await client.chat_completion(
            [ChatMessage(role="user", content="Hola")], stream=True
        )
        assert not isinstance(stream, ModelResponse)
        return [token async for token in stream]

    tokens = asyncio.run(run())

    assert tokens == ["Hola", " mundo"]
    assert completions.calls[0]["stream"] is True


@pytest.mark.parametrize(
    "provider",
    [Provider.OPENAI, Provider.ANTHROPIC, Provider.OPENROUTER],
)
def test_factory_selects_registered_provider(
    provider: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubClient:
        def __init__(self, config: Configuration) -> None:
            self.config = config

    monkeypatch.setitem(LLMFactory._clients, provider, StubClient)  # type: ignore[arg-type]
    config = Configuration(provider=provider, model_name="test-model")

    client = LLMFactory.create_client(config)

    assert isinstance(client, StubClient)
    assert client.config is config


def test_openrouter_uses_openrouter_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_async_openai(**kwargs: str) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("llm_clients.openrouter_client.AsyncOpenAI", fake_async_openai)
    config = Configuration(
        provider=Provider.OPENROUTER,
        model_name="openrouter/free",
    )

    OpenRouterClient(config)

    assert captured == {
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
    }
