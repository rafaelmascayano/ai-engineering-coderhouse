import asyncio
from collections.abc import AsyncGenerator
from typing import cast

from llm_clients import EnvironmentLoader, LLMFactory
from schemas import ChatMessage, ModelResponse


async def main():
    load_environment = EnvironmentLoader()
    config = load_environment.get_configuration()
    client = LLMFactory.create_client(config)
    messages = [ChatMessage(role="user", content="¿Qué sabe de la vida?")]

    try:
        async with asyncio.timeout(int(config.timeout)):
            print("Respuesta normal:")
            response = cast(
                ModelResponse,
                await client.chat_completion(messages),
            )
            if response.error:
                print(f"Error controlado: {response.error}")
            else:
                print(response.content)

        print("\nRespuesta en streaming:")
        async with asyncio.timeout(int(config.timeout)):
            stream = cast(
                AsyncGenerator[str, None],
                await client.chat_completion(messages, stream=True),
            )
            async for token in stream:
                print(token, end="", flush=True)
            print()
    except TimeoutError as e:
        print(f"La solicitud superó el límite de {config.timeout} segundos: {e}")


if __name__ == "__main__":
    asyncio.run(main())
