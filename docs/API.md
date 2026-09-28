# API

PAKT queries may set `original_sources_only: true` on `POST /memory/query`.
This restricts evidence to original document/chunk excerpts with public HTTP(S)
source provenance, excluding generated artifacts and extracted assertions before
coverage is evaluated. The default is false, preserving general Atlas exploration.

Interactive OpenAPI documentation: `/docs`. Every endpoint accepts `Authorization: Bearer <MEMORY_API_TOKEN>` when configured. Empty token is intended only for localhost. The module trusts its client with scopes: it isolates private scopes (`user:42`) from each other in retrieval but does not infer PAKT user permissions.

| Method | Path | Result |
|---|---|---|
| GET | `/memory/health` | Mode and startup readiness |
| POST | `/memory/ingest` | Source/document IDs and duplicate flag |
| POST | `/memory/query` | 202 with durable query ID |
| GET | `/memory/query/{id}` | Running/completed/failed query and result |
| GET | `/memory/query/{id}/events?after=0` | Ordered persisted events after sequence |
| GET | `/memory/query/{id}/stream?after=0` | SSE with `id: sequence` and JSON event data |
| GET | `/memory/nodes/{id}` | Node properties, original text, provenance; excludes embedding |
| GET | `/memory/nodes/{id}/neighbors?limit=100` | Bounded nodes/edges for manual inspection |
| GET | `/memory/documents/{id}` | Typed document lookup |
| GET | `/memory/concepts/{id}` | Typed concept lookup |
| POST | `/memory/nodes/{id}/enrich` | Sourced assertions for a retrievable leaf |
| GET | `/memory/metrics` | Aggregate inference counts, tokens and reported costs; `searches`: needs searched online, searches skipped as recent, searches whose sources reached a context |
| POST | `/memory/documents/{id}/forget` | Exclude from retrieval (`hard`: erase) with a reason; its addresses are never fetched again |
| POST | `/memory/documents/{id}/restore` | Undo an exclusion |
| POST | `/memory/forget` | Forget every document from a `url` |
| GET | `/memory/episodes?scope=shared&q=&limit=20` | Recent questions of a scope, or those closest to `q` |
| GET, POST | `/memory/scopes/{scope}/facts` | A private scope's user facts; add a stated one |
| PATCH, DELETE | `/memory/scopes/{scope}/facts/{id}` | Edit, confirm or reject a fact; remove it |
| POST | `/memory/scopes/{scope}/facts/infer` | Propose facts from the scope's recent questions |
| DELETE | `/memory/scopes/{scope}` | Erase a private scope and everything recorded for it |
| POST | `/memory/images` | Keep a fetched or generated image for reuse |
| GET | `/memory/images?scope=shared&url=&q=&origin=&limit=20` | The image fetched from `url`, the candidates closest to `q` (with `similarity`), or the latest |
| GET, DELETE | `/memory/images/{id}?scope=shared` | An image with its base64 `data`; forget it |
| POST | `/memory/spark/sessions` | A finished session (`scope`, `session`, `title`, `kind`, `date`, `turns`, `encounter`) joins the scope's associative memory: one extraction call ([memory](MEMORY.md)) |
| POST | `/memory/spark/recall` | Memories a text activates in the scope (`scope`, `text`, `encounter`, `limit`), with their state, date, session and activation path; no model call |
| GET | `/memory/procedures` | Retrieval know-how: per host and retriever, found/accepted/used/forgotten counts and score |
| POST, DELETE | `/memory/procedures/hosts/{host}/block` | Block (with a reason) or unblock a web host |
| POST | `/memory/demo/seed` | Synthetic dataset, available only in demo mode |

Memory roles (scopes, forgetting, episodes, user facts, retrieval know-how, images) are described in [memory](MEMORY.md).

Ingestion returns `source_id`, `document_id` and `duplicate`; a new document also returns `nodes_created`, `concepts_extracted` (distinct concepts found), `concepts_linked` (concept nodes written, including taxonomy anchors), `concepts_provisional` (provisionally classified concepts linked), `unclassified_concepts` (skipped labels) and `classification_skips` (`{reason: count}`, see [ingestion](INGESTION.md)). Sources ingested during a query are light and return `"light": true` instead of concept counts; they are structured once a later query uses them. The per-concept skip and provisional details are kept in the `classification_review` record of the document.

Ingestion accepts `title`, exactly one of `text`, `content_base64`, `url`, and optional `mime_type`, `filename`, `source_type`, `author`, `published_at`, `metadata`, `scope` (default `shared`; a private scope keeps its own copy). Example DOCX/PDF ingestion: base64 encode bytes client-side, supply the corresponding MIME type and sanitized filename metadata. Unsupported/invalid source formats return 422. Oversized HTTP bodies return 413; limits also apply to decoded content and remote responses.

Query accepts `query`, `allow_external` (default true), `synthesize` (default true; false returns the evidence without writing an answer, whose `answer` is then empty), `mode` (`answer`, the default, or `dossier`: what the sources in memory say about a topic, see [retrieval](RETRIEVAL.md)), `scope` (a private scope read beside the shared memory, whose active facts personalize relevance and the answer) and `reuse` (default true; false answers afresh instead of reusing a recent episode's result, see [memory](MEMORY.md)) and optional `answer_max_words`, which caps this answer's length and overrides `ANSWER_MAX_WORDS`. Capacity exhaustion returns 429. Completed results include `answer`, valid `evidence_ids`, original `evidence`, final `context`, per-need `coverage` (null when no external search was possible, since coverage only decides that search), `branches` and `nodes_explored`. Insufficient coverage is a completed retrieval outcome, not automatically a server error. A failed query record carries `error` (generic) and `reason` (the exception type or the provider error's own message, never a provider response body); a refused API key or credit returns 502 from ingestion.

SSE uses persisted event sequence numbers and supports `Last-Event-ID`. Clients may reconnect without losing events or poll the event endpoint. `event: done` indicates terminal persisted query state. Unknown IDs return 404; malformed cursors return 422. Event records include event UUID, query UUID, timestamp, type, optional node/branch/parent/need IDs and structured metadata.

The API process owns query lifetimes. Clean shutdown cancels tasks and records failure; startup marks interrupted running sessions failed. Historical completed traces remain replayable in Neo4j. Run one API worker; distributed scheduling is outside this MVP.
