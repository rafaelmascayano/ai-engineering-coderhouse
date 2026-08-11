import os

from openai import AsyncOpenAI

from llm_clients.base import BaseLLMClient
from llm_clients.config import Configuration
from llm_clients.openai_client import OpenAIClient


class OpenRouterClient(OpenAIClient):
    def __init__(self, config: Configuration) -> None:
        BaseLLMClient.__init__(self, config)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is missing")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
