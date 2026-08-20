# AI Engineering: clientes LLM y pipeline de extracción técnica

Repositorio del curso con dos entregas compatibles entre sí:

1. Clientes LLM asíncronos para OpenAI, Anthropic y OpenRouter.
2. Pipeline LCEL que transforma texto técnico sin procesar en un objeto
   validado con Pydantic.

## Pre-entrega 2: pipeline de extracción técnica

El pipeline recibe una descripción de arquitectura, incidente o log de error y
devuelve una instancia de `ExtraccionTecnica` con este contrato:

- `tecnologias`: lista no vacía de tecnologías mencionadas en el texto.
- `nivel_de_criticidad`: `baja`, `media` o `alta`.
- `resumen_tecnico`: resumen no vacío, breve y factual.

El esquema rechaza strings vacíos y campos adicionales.

### Arquitectura

La cadena se compone con LangChain Expression Language (LCEL):

```text
ChatPromptTemplate
    | ChatOpenAI.with_structured_output(ExtraccionTecnica)
    | validación de metadatos y objeto Pydantic
    | with_retry(stop_after_attempt=2)
```

`ChatOpenAI` se conecta al endpoint compatible de OpenRouter. La salida usa
JSON Schema estricto y conserva el mensaje crudo para revisar `finish_reason`
antes de aceptar el objeto parseado.

Como `openrouter/free` puede elegir modelos donde el razonamiento es obligatorio,
la solicitud usa esfuerzo `minimal` y omite `temperature`. El pipeline reserva
como mínimo 2048 tokens de salida para dejar espacio tanto al razonamiento como
al JSON final.

Si la respuesta está truncada, el SDK lanza `LengthFinishReasonError`, contiene
JSON inválido o no produce el modelo Pydantic esperado, la cadena convierte el
problema a `StructuredOutputError` y realiza un reintento automático. Después de
dos intentos fallidos propaga el error validado.

### Ejecución del ejemplo asíncrono

Configura `.env` y ejecuta:

```bash
python demo_pipeline.py
```

El script utiliza `asyncio.run()`, llama a `process_text()` mediante `.ainvoke()`
y muestra el resultado con `model_dump_json(indent=2)`.

Ejemplo de salida:

```json
{
  "tecnologias": [
    "FastAPI",
    "Redis",
    "PostgreSQL"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; el agotamiento de conexiones deja el servicio fuera de línea."
}
```

También se puede importar la función directamente:

```python
import asyncio

from chain import process_text


async def main() -> None:
    result = await process_text(
        "FastAPI usa Redis y PostgreSQL; el pool de conexiones se agotó."
    )
    print(result.model_dump())


asyncio.run(main())
```

Los logs informan el inicio del procesamiento, cada intento, la validación, las
respuestas incompletas y el agotamiento de reintentos. No registran el texto de
entrada ni la API key.

## Requisitos e instalación

- Python 3.12
- Una API key de OpenRouter para ejecutar el pipeline real

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Configuración

Copia la plantilla y completa la clave correspondiente:

```bash
cp .env.example .env
```

Configuración recomendada para la pre-entrega 2:

```dotenv
PROVIDER=openrouter
OPENROUTER_API_KEY=tu_api_key
MODEL_NAME=openrouter/free
TEMPERATURE=0.0
MAX_TOKENS=2048
TIMEOUT=30
```

| Variable | Descripción |
| --- | --- |
| `PROVIDER` | Proveedor seleccionado; el pipeline requiere `openrouter` |
| `OPENROUTER_API_KEY` | Clave de OpenRouter, nunca debe subirse al repositorio |
| `MODEL_NAME` | Modelo o router; por defecto `openrouter/free` |
| `TEMPERATURE` | Usada por el Módulo 1; el pipeline la omite por compatibilidad |
| `MAX_TOKENS` | Máximo de salida; el pipeline aplica un mínimo de `2048` |
| `TIMEOUT` | Timeout de cada solicitud en segundos |

`openrouter/free` selecciona modelos gratuitos disponibles y filtra según las
capacidades solicitadas, como Structured Outputs. Su disponibilidad y rate
limits pueden variar. Consulta la
[documentación de Structured Outputs de OpenRouter](https://openrouter.ai/docs/guides/features/structured-outputs).

El archivo `.env` está excluido de Git. Nunca publiques claves reales.

## Entrega del Módulo 1

El ejemplo anterior de clientes asíncronos se conserva. Ejecuta:

```bash
python main.py
```

`main.py` muestra una respuesta normal y otra en streaming usando el proveedor
definido en `PROVIDER`. Los clientes soportados son OpenAI, Anthropic y
OpenRouter, con manejo controlado de errores y timeout.

## Tests y calidad

La suite usa clientes simulados: no realiza llamadas externas ni consume
créditos.

```bash
pytest
```

Las pruebas cubren los esquemas, los clientes del Módulo 1, la composición LCEL,
el uso de JSON Schema estricto, la llamada asíncrona y la recuperación ante JSON
mal formado o respuestas truncadas.

Para ejecutar Ruff mediante pre-commit:

```bash
pre-commit install
pre-commit run --all-files
```

## Estructura principal

```text
.
├── chain.py                    # Prompt, LCEL, salida estructurada y reintento
├── schemas.py                  # Contratos Pydantic de ambos módulos
├── demo_pipeline.py            # Mini-script asíncrono de la pre-entrega 2
├── main.py                     # Ejemplo normal y streaming del Módulo 1
├── llm_clients/                # Clientes OpenAI, Anthropic y OpenRouter
├── tests/                      # Tests unitarios sin llamadas externas
├── requirements.txt            # Dependencias del proyecto
└── .env.example                # Plantilla de configuración sin secretos
```
