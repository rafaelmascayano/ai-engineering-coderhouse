from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelResponse(BaseModel):
    content: str
    provider: str
    model: str
    error: str | None = None
