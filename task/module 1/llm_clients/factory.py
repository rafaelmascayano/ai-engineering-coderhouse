from llm_clients.anthropic_client import AnthropicClient
from llm_clients.base import BaseLLMClient
from llm_clients.config import Configuration, Provider
from llm_clients.openai_client import OpenAIClient
from llm_clients.openrouter_client import OpenRouterClient


class LLMFactory:
    _clients: dict[Provider, type[BaseLLMClient]] = {
        Provider.ANTHROPIC: AnthropicClient,
        Provider.OPENAI: OpenAIClient,
        Provider.OPENROUTER: OpenRouterClient,
    }

    @classmethod
    def create_client(cls, config: Configuration) -> BaseLLMClient:
        try:
            client_class = cls._clients[config.provider]
        except KeyError as error:
            raise ValueError(f"Unsupported provider: {config.provider}") from error
        return client_class(config)
