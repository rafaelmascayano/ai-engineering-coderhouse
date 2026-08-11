# Clientes LLM asíncronos

Implementación en Python de una interfaz común para consultar OpenAI, Anthropic
y OpenRouter de manera asíncrona. El ejemplo incluye respuestas normales,
streaming de texto, validación con Pydantic y manejo controlado de errores.

## Requisitos

- Python 3.12
- Una API key de OpenAI, Anthropic u OpenRouter

## Instalación

Desde el directorio de este módulo, crea y activa un entorno virtual:

```bash
cd "task/module 1"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows PowerShell, activa el entorno con:

```powershell
.venv\Scripts\Activate.ps1
```

## Configuración

Copia el archivo de ejemplo:

```bash
cp .env.example .env
```

Configura estas variables en `.env`:

| Variable | Descripción | Ejemplo |
| --- | --- | --- |
| `PROVIDER` | Proveedor que se utilizará | `openai`, `anthropic` u `openrouter` |
| `OPENAI_API_KEY` | API key requerida cuando el proveedor es OpenAI | `sk-...` |
| `ANTHROPIC_API_KEY` | API key requerida cuando el proveedor es Anthropic | `sk-ant-...` |
| `OPENROUTER_API_KEY` | API key requerida cuando el proveedor es OpenRouter | `sk-or-...` |
| `MODEL_NAME` | Identificador del modelo | `gpt-4o-mini` |
| `TEMPERATURE` | Aleatoriedad entre 0 y 2 | `0.7` |
| `MAX_TOKENS` | Máximo de tokens de salida, como entero positivo | `1000` |
| `TIMEOUT` | Tiempo máximo de cada modalidad, en segundos | `30` |

Sólo es necesario completar la API key correspondiente al proveedor elegido. El
archivo `.env` está excluido de Git y no debe subirse al repositorio.

Ejemplo para OpenAI:

```dotenv
PROVIDER=openai
OPENAI_API_KEY=tu_api_key
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.7
MAX_TOKENS=1000
TIMEOUT=30
```

Ejemplo para Anthropic:

```dotenv
PROVIDER=anthropic
ANTHROPIC_API_KEY=tu_api_key
MODEL_NAME=claude-3-5-haiku-latest
TEMPERATURE=0.7
MAX_TOKENS=1000
TIMEOUT=30
```

Ejemplo gratuito para OpenRouter:

```dotenv
PROVIDER=openrouter
OPENROUTER_API_KEY=tu_api_key
MODEL_NAME=openrouter/free
TEMPERATURE=0.7
MAX_TOKENS=1000
TIMEOUT=30
```

La clave se crea en [OpenRouter Keys](https://openrouter.ai/settings/keys). El
modelo `openrouter/free` elige automáticamente un modelo gratuito disponible y
puede tener límites de uso o disponibilidad.

## Ejecución

Con el entorno virtual activo y `.env` configurado, ejecuta:

```bash
python main.py
```

El script pregunta «¿Qué es la entropía?» dos veces:

1. En modo normal, esperando la respuesta completa.
2. En modo streaming, mostrando los fragmentos a medida que llegan.

Los errores de conexión, autenticación, cuota o rate limit se presentan de forma
controlada. Cada modalidad también está protegida por el timeout configurado.

## Calidad de código con pre-commit

La configuración ubicada en la raíz del repositorio ejecuta `ruff check --fix` y
`ruff format` sobre los archivos Python de este módulo antes de cada commit.

Instala los hooks una vez, desde la raíz del repositorio:

```bash
pre-commit install
```

También puedes comprobar todos los archivos manualmente:

```bash
pre-commit run --all-files
```

Si Ruff modifica algún archivo, revísalo y vuelve a agregarlo al commit.

## Tests unitarios

La suite usa clientes simulados y no realiza solicitudes externas ni consume
créditos. Desde el directorio del módulo, ejecútala con:

```bash
pytest
```

Los tests cubren esquemas, configuración, fábrica, generación normal, streaming
y configuración del endpoint de OpenRouter.

## Estructura

```text
.
├── llm_clients/
│   ├── __init__.py             # API pública del paquete
│   ├── base.py                 # Interfaz abstracta común
│   ├── config.py               # Configuración y carga del entorno
│   ├── factory.py              # Selección del proveedor
│   ├── anthropic_client.py     # Cliente asíncrono de Anthropic
│   ├── openai_client.py        # Cliente asíncrono de OpenAI
│   └── openrouter_client.py    # Cliente asíncrono de OpenRouter
├── schemas.py                  # Mensajes y respuestas Pydantic
├── main.py                     # Prueba normal y por streaming
├── tests/                      # Tests unitarios sin llamadas externas
├── requirements.txt            # Dependencias fijadas
└── .env.example                # Plantilla de variables de entorno
```
