import pytest
from llm_clients.config import Configuration, EnvironmentLoader, Provider
from pydantic import ValidationError


def test_environment_loader_reads_all_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER", "openrouter")
    monkeypatch.setenv("MODEL_NAME", "openrouter/free")
    monkeypatch.setenv("TIMEOUT", "15")
    monkeypatch.setenv("MAX_TOKENS", "250")
    monkeypatch.setenv("TEMPERATURE", "0.2")

    config = EnvironmentLoader().get_configuration()

    assert config == Configuration(
        provider=Provider.OPENROUTER,
        model_name="openrouter/free",
        timeout=15,
        max_tokens=250,
        temperature=0.2,
    )


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_configuration_rejects_temperature_out_of_range(
    temperature: float,
) -> None:
    with pytest.raises(ValidationError):
        Configuration(
            provider=Provider.OPENAI,
            model_name="test-model",
            temperature=temperature,
        )


def test_environment_loader_rejects_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Invalid PROVIDER"):
        EnvironmentLoader().get_configuration()
