import os
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


class Configuration(BaseModel):
    provider: Provider
    model_name: str
    timeout: int = Field(default=30, ge=1)
    max_tokens: int = Field(default=1000, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class EnvironmentLoader:
    def __init__(self) -> None:
        load_dotenv()

    def get_configuration(self) -> Configuration:
        provider_name = os.getenv("PROVIDER", "openai").lower()
        timeout = int(os.getenv("TIMEOUT", "30"))
        max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
        temperature = float(os.getenv("TEMPERATURE", "0.7"))

        try:
            provider = Provider(provider_name)
        except ValueError as error:
            raise ValueError(
                f"Invalid PROVIDER in .env: '{provider_name}'. "
                "Must be 'anthropic', 'openai' or 'openrouter'"
            ) from error

        default_models = {
            Provider.ANTHROPIC: "claude-3-5-haiku-latest",
            Provider.OPENAI: "gpt-4o-mini",
            Provider.OPENROUTER: "openrouter/free",
        }

        return Configuration(
            provider=provider,
            model_name=os.getenv("MODEL_NAME", default_models[provider]),
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
        )
