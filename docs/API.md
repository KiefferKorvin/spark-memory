# API

PAKT queries may set `original_sources_only: true` on `POST /memory/query`.
This restricts evidence to original document/chunk excerpts with public HTTP(S)
source provenance, excluding generated artifacts and extracted assertions before
coverage is evaluated. The default is false, preserving general Atlas exploration.

Interactive OpenAPI documentation: `/docs`. Every endpoint accepts `Authorization: Bearer <MEMORY_API_TOKEN>` when configured. Empty token is intended only for localhost. The module is single-tenant; it does not infer PAKT user permissions.

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
| GET | `/memory/metrics` | Aggregate inference counts, tokens and reported costs |
| POST | `/memory/demo/seed` | Synthetic dataset, available only in demo mode |

Ingestion returns `source_id`, `document_id` and `duplicate`; a new document also returns `nodes_created`, `concepts_extracted` (distinct concepts found), `concepts_linked` (concept nodes written, including taxonomy anchors), `concepts_provisional` (provisionally classified concepts linked), `unclassified_concepts` (skipped labels) and `classification_skips` (`{reason: count}`, see [ingestion](INGESTION.md)). Sources ingested during a query return `"concepts": "deferred"` instead. The per-concept skip and provisional details are kept in the `classification_review` record of the document.

Ingestion accepts `title`, exactly one of `text`, `content_base64`, `url`, and optional `mime_type`, `filename`, `source_type`, `author`, `published_at`, `metadata`. Example DOCX/PDF ingestion: base64 encode bytes client-side, supply the corresponding MIME type and sanitized filename metadata. Unsupported/invalid source formats return 422. Oversized HTTP bodies return 413; limits also apply to decoded content and remote responses.

Query accepts `query`, `allow_external` (default true) and optional `answer_max_words`, which caps this answer's length and overrides `ANSWER_MAX_WORDS`. Capacity exhaustion returns 429. Completed results include `answer`, valid `evidence_ids`, original `evidence`, final `context`, per-need `coverage`, `branches` and `nodes_explored`. Insufficient coverage is a completed retrieval outcome, not automatically a server error.

SSE uses persisted event sequence numbers and supports `Last-Event-ID`. Clients may reconnect without losing events or poll the event endpoint. `event: done` indicates terminal persisted query state. Unknown IDs return 404; malformed cursors return 422. Event records include event UUID, query UUID, timestamp, type, optional node/branch/parent/need IDs and structured metadata.

The API process owns query lifetimes. Clean shutdown cancels tasks and records failure; startup marks interrupted running sessions failed. Historical completed traces remain replayable in Neo4j. Run one API worker; distributed scheduling is outside this MVP.
