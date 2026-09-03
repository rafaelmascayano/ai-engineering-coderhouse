# Capturas pendientes (Módulo 7)

Esta carpeta debe contener, antes de la entrega final, las capturas del
dashboard de observabilidad que evidencian la corrida de las 5 peticiones
concurrentes (`python scripts/load_test.py`):

1. `traces-overview.png` — listado de trazas del run, una por `job_id`.
2. `trace-detail.png` — una traza abierta mostrando los nodos del grafo
   (`supervisor`, `researcher`/`human_approval`/`analyst`) y qué nodo
   concentra más latencia/tokens.
3. `cost-per-run.png` — costo por ejecución calculado por la plataforma a
   partir de tokens de entrada/salida.
4. `latency-p95.png` — latencia p95 de la corrida de 5 peticiones.

No se incluyen capturas simuladas en este repositorio: deben generarse
ejecutando la API real (con `OPENROUTER_API_KEY` y `LANGCHAIN_API_KEY` o
Phoenix configurados) contra LangSmith o Arize Phoenix, tal como se describe
en el README principal.
