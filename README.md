# RAG asíncrono con LangChain y ChromaDB

Flujo End-to-End de Retrieval-Augmented Generation (RAG) sobre la Ley chilena
N.º 21.442 de Copropiedad Inmobiliaria. El proyecto ingiere documentos locales,
los fragmenta por tokens, persiste sus embeddings en ChromaDB y responde
preguntas utilizando exclusivamente los fragmentos recuperados.

## Arquitectura

```text
data/*.txt
  -> RecursiveCharacterTextSplitter (600 tokens, overlap 50)
  -> OpenAIEmbeddings vía OpenRouter (Nemotron 3 Embed gratuito)
  -> ChromaDB persistente (./vectorstore)
  -> retriever top_k=4
  -> prompt grounded + ChatOpenAI
  -> PydanticOutputParser(RAGResponse)
```

La cadena se compone con LangChain Expression Language (LCEL). El prompt trata
los documentos como datos no confiables, ignora instrucciones incluidas en
ellos y obliga al modelo a devolver `"No lo sé"` cuando el contexto recuperado
no contiene la respuesta. Una validación posterior rechaza referencias que no
hayan sido devueltas por el retriever.

## Contenido del repositorio

```text
.
├── data/
│   ├── 01_regimen_obligaciones_reglamento.txt
│   ├── 02_administracion_condominio.txt
│   ├── 03_uso_gastos_seguridad.txt
│   └── 04_conflictos_registro_sanciones.txt
├── ingest.py              # carga, limpieza, chunking e indexación persistente
├── rag_chain.py           # retriever y cadena LCEL asíncrona
├── rag_config.py          # configuración compartida
├── demo_rag.py            # consulta real y pregunta trampa
├── main.py                # entrega asíncrona preservada del Módulo 2
├── module1_demo.py        # clientes LLM preservados del Módulo 1
├── schemas.py             # RAGResponse y referencias validadas con Pydantic
├── tests/test_rag.py      # pruebas sin red ni consumo de API
├── requirements.txt
└── .env.example
```

El dataset proviene del PDF oficial de la Biblioteca del Congreso Nacional de
Chile, versión generada el 20 de agosto de 2026. Se dividió en cuatro archivos
por bloques temáticos/páginas. El ingestor elimina los encabezados y pies de
página repetidos antes de fragmentar.

Los módulos de clientes LLM y extracción técnica de las entregas anteriores se
conservan y continúan cubiertos por la suite original.

## Requisitos

- Python 3.12 o superior
- Una API key de OpenRouter para crear embeddings y ejecutar consultas reales

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

En Windows PowerShell, activa el entorno con:

```powershell
.venv\Scripts\Activate.ps1
```

Completa únicamente tu archivo local `.env`:

```dotenv
OPENROUTER_API_KEY=tu_api_key
RAG_CHAT_MODEL=openrouter/free
RAG_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b:free
RAG_COLLECTION=ley_21442
RAG_DATA_DIR=./data
RAG_VECTORSTORE_DIR=./vectorstore
RAG_TOP_K=4
```

`.env` y `vectorstore/` están excluidos de Git. No subas llaves ni la base
vectorial generada localmente.

## 1. Ingesta

Ejecuta una vez:

```bash
python ingest.py
```

El comando carga todos los `.txt` y `.md` de `data/`, crea chunks de 600 tokens
con 50 tokens de solapamiento y los guarda en `./vectorstore`.

El manifiesto local registra el hash de las fuentes, el modelo de embeddings y
la configuración de chunking. Si todo coincide, una segunda ejecución reutiliza
el índice y no vuelve a consumir la API. Si cambian las fuentes o el modelo,
reconstruye explícitamente la colección:

```bash
python ingest.py --force
```

El mismo `RAG_EMBEDDING_MODEL` se usa al indexar y al consultar. La cadena se
detiene con un error claro si el manifiesto indica un modelo distinto.

## 2. Consultas asíncronas

Ejecuta los dos casos requeridos (uno respondible y uno fuera del dataset):

```bash
python demo_rag.py
```

También puedes enviar una sola pregunta:

```bash
python demo_rag.py "¿Cuáles son los órganos de administración de un condominio?"
```

Uso desde Python:

```python
import asyncio

from rag_chain import get_rag_response


async def main() -> None:
    result = await get_rag_response(
        "¿Cuáles son los órganos de administración de un condominio?"
    )
    print(result.model_dump_json(indent=2))


asyncio.run(main())
```

Contrato de salida:

```json
{
  "answer": "Respuesta basada únicamente en el contexto.",
  "references": [
    {
      "source": "02_administracion_condominio.txt",
      "chunk_id": "02_administracion_condominio.txt#chunk-001"
    }
  ]
}
```

Para una pregunta ajena a los documentos, la salida esperada es:

```json
{
  "answer": "No lo sé",
  "references": []
}
```

## Pruebas

La suite usa retrievers y modelos simulados: no necesita una API key, no accede
a internet y no consume créditos.

```bash
pytest
```

Las pruebas del RAG verifican:

- una pregunta cuya respuesta está en el contexto;
- una pregunta trampa que debe devolver exactamente `"No lo sé"`;
- rechazo de referencias inventadas;
- validación de consultas vacías;
- limpieza, chunking y metadatos de los documentos.

## Decisiones de seguridad y calidad

- `top_k` está restringido a 3-5 para evitar contexto excesivo y *Lost in the
  Middle*;
- indexación y consulta comparten el mismo modelo de embeddings;
- la base vectorial se persiste y reutiliza mediante un manifiesto;
- el prompt prohíbe conocimiento externo y trata el contexto como datos;
- `PydanticOutputParser` exige una salida estructurada;
- cada referencia se contrasta con los chunks recuperados antes de devolverla.
