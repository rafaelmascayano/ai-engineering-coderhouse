import os
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Providers(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class Configuration(BaseModel):
    provider: Providers
    model_name: str
    timeout: int = Field(default=30, ge=1)
    max_tokens: int = Field(default=1000, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class LoadEnvironment:
    def __init__(self):
        load_dotenv()

    def get_configuration(self) -> Configuration:
        provider_str = os.getenv("PROVIDER", "openai").lower()
        model_name = os.getenv("MODEL_NAME", "gpt-4o")
        timeout = int(os.getenv("TIMEOUT", "30"))

        try:
            provider = Providers(provider_str)
        except ValueError:
            raise ValueError(
                f"Invalid PROVIDER in .env: '{provider_str}'. Must be 'anthropic' or 'openai'"
            )

        return Configuration(
            provider=provider,
            model_name=model_name,
            timeout=timeout,
        )
