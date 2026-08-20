import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from openai import LengthFinishReasonError
from openai.types.chat import ChatCompletion

import chain as chain_module
from chain import (
    FORMAT_INSTRUCTIONS,
    MIN_PIPELINE_MAX_TOKENS,
    OPENROUTER_BASE_URL,
    StructuredOutputError,
    build_chain,
    create_model,
    process_text,
)
from schemas import ExtraccionTecnica, NivelCriticidad


def extraction() -> ExtraccionTecnica:
    return ExtraccionTecnica(
        tecnologias=["FastAPI", "Redis", "PostgreSQL"],
        nivel_de_criticidad=NivelCriticidad.ALTA,
        resumen_tecnico="API fuera de línea por agotamiento de conexiones.",
    )


def structured_response(
    *,
    parsed: ExtraccionTecnica | None = None,
    finish_reason: str = "stop",
    parsing_error: Exception | None = None,
) -> dict[str, object]:
    return {
        "raw": AIMessage(
            content="",
            response_metadata={"finish_reason": finish_reason},
        ),
        "parsed": parsed,
        "parsing_error": parsing_error,
    }


class StubChatModel:
    def __init__(self, responses: Sequence[dict[str, object] | Exception]) -> None:
        self.responses = responses
        self.calls = 0
        self.schema: object | None = None
        self.options: dict[str, object] = {}

    def with_structured_output(
        self, schema: object, **kwargs: object
    ) -> RunnableLambda:
        self.schema = schema
        self.options = kwargs

        async def respond(_: Any) -> dict[str, object]:
            index = min(self.calls, len(self.responses) - 1)
            self.calls += 1
            response = self.responses[index]
            if isinstance(response, Exception):
                raise response
            return response

        return RunnableLambda(respond)


def invoke(stub: StubChatModel) -> ExtraccionTecnica:
    pipeline = build_chain(stub)
    return asyncio.run(
        pipeline.ainvoke(
            {
                "texto": "FastAPI usa Redis y PostgreSQL.",
                "instrucciones_formato": FORMAT_INSTRUCTIONS,
            }
        )
    )


def test_create_model_configures_chat_openai_for_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "openrouter/free")
    monkeypatch.setenv("TEMPERATURE", "0")
    monkeypatch.setenv("MAX_TOKENS", "321")
    monkeypatch.setenv("TIMEOUT", "12")

    model = create_model()

    assert model.model_name == "openrouter/free"
    assert model.openai_api_base == OPENROUTER_BASE_URL
    assert model.temperature is None
    assert model.max_tokens == MIN_PIPELINE_MAX_TOKENS
    assert model.request_timeout == 12
    assert model.max_retries == 0
    assert model.extra_body == {
        "provider": {"require_parameters": True},
        "reasoning": {"effort": "minimal", "exclude": True},
    }


def test_chain_uses_json_schema_and_returns_validated_model() -> None:
    expected = extraction()
    stub = StubChatModel([structured_response(parsed=expected)])

    result = invoke(stub)

    assert result == expected
    assert stub.calls == 1
    assert stub.schema is ExtraccionTecnica
    assert stub.options == {
        "method": "json_schema",
        "strict": True,
        "include_raw": True,
    }


def test_chain_retries_malformed_json_once_then_recovers() -> None:
    expected = extraction()
    stub = StubChatModel(
        [
            structured_response(parsing_error=ValueError("invalid JSON")),
            structured_response(parsed=expected),
        ]
    )

    result = invoke(stub)

    assert result == expected
    assert stub.calls == 2


def test_chain_retries_truncated_response_before_accepting_parsed_data() -> None:
    expected = extraction()
    stub = StubChatModel(
        [
            structured_response(parsed=expected, finish_reason="length"),
            structured_response(parsed=expected),
        ]
    )

    result = invoke(stub)

    assert result == expected
    assert stub.calls == 2


def test_chain_retries_length_error_raised_by_openai_parser() -> None:
    expected = extraction()
    completion = ChatCompletion.model_validate(
        {
            "id": "test-completion",
            "choices": [
                {
                    "finish_reason": "length",
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                }
            ],
            "created": 0,
            "model": "openrouter/free",
            "object": "chat.completion",
        }
    )
    stub = StubChatModel(
        [
            LengthFinishReasonError(completion=completion),
            structured_response(parsed=expected),
        ]
    )

    result = invoke(stub)

    assert result == expected
    assert stub.calls == 2


def test_chain_raises_after_retry_is_exhausted() -> None:
    stub = StubChatModel(
        [structured_response(parsing_error=ValueError("invalid JSON"))]
    )

    with pytest.raises(StructuredOutputError):
        invoke(stub)

    assert stub.calls == 2


def test_process_text_rejects_blank_input_without_building_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_call() -> None:
        raise AssertionError("get_chain no debe ejecutarse para texto vacío")

    monkeypatch.setattr(chain_module, "get_chain", unexpected_call)

    with pytest.raises(ValueError, match="no vacío"):
        asyncio.run(process_text("   "))


def test_process_text_invokes_async_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = extraction()

    class StubPipeline:
        def __init__(self) -> None:
            self.chain_input: dict[str, str] | None = None

        async def ainvoke(self, chain_input: dict[str, str]) -> ExtraccionTecnica:
            self.chain_input = chain_input
            return expected

    pipeline = StubPipeline()
    monkeypatch.setattr(chain_module, "get_chain", lambda: pipeline)

    result = asyncio.run(process_text("  FastAPI usa Redis.  "))

    assert result == expected
    assert pipeline.chain_input == {
        "texto": "FastAPI usa Redis.",
        "instrucciones_formato": FORMAT_INSTRUCTIONS,
    }
