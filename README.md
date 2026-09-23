# Progressive Graph Memory

Reusable Python/FastAPI memory infrastructure with a Neo4j property graph, OpenRouter semantic inference, Jev navigation, and a React/NVL exploration UI. It is independent of the parent PAKT application: no application accounts, SQLite migrations or existing model preferences are changed.

The three graph layers are documents, concepts and sourced assertions. Retrieval combines indexes, graph neighborhoods, bounded concurrent traversal, evidence coverage and optional external retrieval. Sources and contradictory assertions remain attributable.

UNESCO Thesaurus 2026 is the primary live taxonomy. Import the local RDF/TTL before
ingesting documents, or configure `UNESCO_THESAURUS_PATH` for startup import.
See [UNESCO setup, schema and indexing](docs/UNESCO.md) and the
[validated corpus report](docs/unesco-import-report.json). LOCAL concepts extend
UNESCO with explicit parent links; official concepts retain their original URIs.

## Run the offline demo

Python 3.11+ and Node 22 recommended. From this `memory/` directory, in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e '.[test]'
# One-time vocabulary download; afterward the demo requires no model/network access.
.venv\Scripts\python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
$env:MEMORY_MODE='demo'
.venv\Scripts\python -m uvicorn graph_memory.api:create_app --factory --host 127.0.0.1 --port 8010
```

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open **http://127.0.0.1:5174**, select **Load synthetic sources**, then **Explore graph**. The default question deliberately requires a missing practice source. Watch the event timeline, inspect evidence, and replay the session. `python -m graph_memory.demo` runs the same workflow without a browser. Demo inference is a deterministic lexical mock and external retrieval is synthetic; the UI labels that mode explicitly. Demo storage is in memory and disappears on restart.

On Linux/macOS use `.venv/bin/python` and `export MEMORY_MODE=demo`.

## Run with Neo4j and OpenRouter

Copy `.env.example` to `.env`. Set `MEMORY_MODE=live`, `NEO4J_PASSWORD`, `OPENROUTER_API_KEY`, `SEMANTIC_MODEL`, `EMBEDDING_MODEL`, and matching `EMBEDDING_DIMENSIONS`. Semantic and embedding models are intentionally configuration choices; select models supporting structured output and embeddings respectively. Jev defaults to `typesafe/jev-1.13` and uses the dedicated `/api/alpha/decisions` endpoint.

```powershell
docker compose up --build
```

The UI is at http://127.0.0.1:5174, API at http://127.0.0.1:8010 and interactive API documentation at http://127.0.0.1:8010/docs. Alternatively, start only `docker compose up -d neo4j` and run the API/UI commands above with `MEMORY_MODE=live`. Neo4j data uses a Docker volume. Database constraints/indexes are initialized automatically.

When memory cannot cover a question, `EXTERNAL_RETRIEVERS` (default `wikipedia,pubmed,web`, all keyless; empty disables) are searched and relevant results ingested; this applies to `live` and `online_demo`. `OPENROUTER_OPTIONS` (JSON) is merged into every chat request; its default prefers high-throughput providers that honour strict JSON schemas and disables hidden reasoning, since price-weighted routing was observed at 84 s per call instead of 3 s. `ALLOWED_SOURCE_HOSTS` controls URL ingestion separately, with exact HTTPS hostnames. Empty disables direct remote ingestion. Arbitrary application retrievers implement `ExternalRetriever.search()` and are injected into `Memory`.

Use one API worker. The API is single-tenant infrastructure; before exposing it beyond localhost, configure `MEMORY_API_TOKEN`, TLS and application authorization. All Compose ports bind to localhost.

## Docker development with automatic reload

From `memory/`, use the development overlay (same `.env` as above):

```powershell
docker compose -f compose.yaml -f compose.dev.yaml up -d --build
```

Open http://127.0.0.1:5174. Changes to `src/` restart the API automatically;
changes to `frontend/src/` and `frontend/index.html` update the browser through
Vite. The Vite and TypeScript configuration files are also mounted. Polling is
enabled for reliable change detection with Docker Desktop on Windows. Neo4j
keeps using the existing `memory-data` volume.

The first launch builds the images. For subsequent launches, omit `--build`.
Rebuild the affected service after changing Python/npm dependencies or a
Dockerfile, for example:

```powershell
docker compose -f compose.yaml -f compose.dev.yaml up -d --build api
docker compose -f compose.yaml -f compose.dev.yaml up -d --build ui
```

Changes to `.env` require rerunning `up -d` with both Compose files. To return
to the compiled UI and normal API without automatic reload:

```powershell
docker compose up -d --build
```

## Python SDK

```python
from graph_memory import Memory
from graph_memory.models import IngestRequest

memory = Memory.from_settings()
await memory.initialize()
try:
    document = await memory.ingest(IngestRequest(
        title="A note", text="# Systems\n## Reliability\nRetries require idempotency.",
        mime_type="text/markdown",
    ))
    result = await memory.query("Why do retries require idempotency?")
    print(result["answer"], result["coverage"])
finally:
    await memory.close()
```

Binary uploads use base64 bytes with an explicit MIME type; API requests never open caller-provided local paths. Adapters for APIs/databases can submit text/JSON and source metadata. Supported built-in formats are UTF-8 text, Markdown, HTML, JSON, email, text PDFs and DOCX. Scanned PDFs require an external OCR adapter; unsupported formats are rejected.

## Verification

```powershell
.venv\Scripts\python -m pytest -q
cd frontend
npm test
npm run build
```

The optional Neo4j test requires a **dedicated test database**: set `NEO4J_TEST_URI` and `NEO4J_TEST_PASSWORD`, then run `pytest tests/test_neo4j.py`. It creates uniquely identified synthetic nodes and removes those nodes afterward. The test database must use an 8-dimensional vector index; use a separate instance from live embeddings.

Local validation uses mocked OpenRouter transport and synthetic retrieval, not billable live requests. See [validation status](docs/VALIDATION.md) for checks actually run and remaining live checks.

## Documentation

- [Architecture](docs/ARCHITECTURE.md), [graph schema](docs/GRAPH_SCHEMA.md), [API](docs/API.md)
- [Ingestion](docs/INGESTION.md), [retrieval](docs/RETRIEVAL.md), [navigation](docs/NAVIGATION.md)
- [Visualization](docs/VISUALIZATION.md), [security](docs/SECURITY.md), [decisions and limits](docs/DECISIONS.md)
- Prompts and explicit cache-invalidation version: `src/graph_memory/prompts.py`
- Synthetic documents and five example questions: `src/graph_memory/demo.py`

## Example HTTP workflow

```http
POST /memory/ingest
Content-Type: application/json

{"title":"Reliability note","text":"Retries require idempotency."}
```

```http
POST /memory/query
Content-Type: application/json

{"query":"Why do retries require idempotency?","allow_external":true}
```

Use the returned `query_id` with `GET /memory/query/{query_id}/stream` for SSE, then `GET /memory/query/{query_id}` for the grounded answer, evidence, coverage and branch state. `GET /memory/query/{query_id}/events` returns the durable replay log. In demo mode, seed via `POST /memory/demo/seed` and ask “Why are rootless voicings useful and how should I practice them?” to exercise external fallback and automatic ingestion.
