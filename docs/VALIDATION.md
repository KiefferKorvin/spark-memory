# Validation status

Exploration, ingestion and online-fallback fixes (2026-09-23), Windows, Python 3.12, Node 22:

- Backend suite: 34 passed; the Neo4j integration test (extended for the taxonomy content filter and incremental event reads) passed against a disposable Neo4j 5.26 container. The offline seed → query → external fallback → answer cycle also ran on that Neo4j.
- Live keyless retrievers (Wikipedia, PubMed, DuckDuckGo web) fetched and ingested real sources. Findings fixed on the way: DuckDuckGo answers `Accept-Encoding: identity` with a bot challenge (fetcher now accepts bounded gzip); Wikimedia rate-limits request bursts (one request per search); Docker lacked an IPv6 route (IPv4 preferred).
- Real `online_demo` query on an empty memory ("How does altitude change the boiling point of water?"): online search, three relevant sources ingested, 65 nodes explored, 377 events streamed live, grounded answer with 13 citations in 94 s. Before the fixes the same query explored 0 nodes and never searched online, and later timed out at 180 s because OpenRouter's price-weighted routing chose an ~84 s/call provider.
- Frontend replay reducer test, TypeScript check and production build passed; the live graph and replay were checked in a browser against the offline demo.

Earlier status:

Verified locally on Windows, Python 3.12 and Node 22:

UNESCO integration: 25 backend tests passed and the optional live Neo4j test was
skipped. New tests cover RDF/Turtle equivalence, multilingual labels and resource
definitions, polyhierarchy, repeatable import, official relationship protection,
anchored local extensions, uncertain classification rejection and complete
document ingestion with ABOUT links and duplicate detection. Production frontend
build passed. The real local RDF and TTL files produce the same semantic
fingerprint: 4,500 concepts, 95 groups and 15,223 normalized relationships.
The report is `unesco-import-report.json`; its status is **validated**, not imported.
The new UNESCO browser button has been compiled but not visually exercised against
a populated live Neo4j instance. Live Cypher and semantic classification remain
unverified for the environment reasons below.

Earlier baseline validation:

- Backend unit/integration suite: 20 passed; 1 optional Neo4j test skipped because no test database was configured. Covers document hierarchy, Unicode token limits, HTML/JSON/email/DOCX parsing, concept aliases/reuse, duplicate provenance, DAG-cycle rollback, hybrid ranking before candidate limits, branch concurrency, loop prevention, pruning, global budgets, evidence/coverage, external fallback with new-source ingestion, assertion support/contradictions, quote validation, SSRF validation, provider protocol/repair, caching/usage, capacity/timeouts, authenticated API calls and resumable event streaming.
- Offline SDK demo: final coverage SUFFICIENT after external practice-source ingestion; 151 persisted trace events on the initial run. A repeated query can use the enriched memory directly, so its event count differs.
- Frontend TypeScript production build and Node replay-reducer test passed. NVL contributes a large bundle; Vite reports its size warning.
- Browser exercised synthetic seeding, live graph query, covered needs, original evidence/provenance inspection, manual neighbor expansion/collapse, and replay advancing from event 0 to 86 before pause, then restoring the final 127-event enriched-memory session. The small-graph `d3Force` layout replaced an overlapping default-layout result during visual testing.
- npm dependency audit: 0 reported vulnerabilities after applying the `js-cookie` patch override and using Node's built-in test runner.

Environment note: pytest's default cache creation stalled under this workspace's Windows filesystem sandbox. Running `python -m pytest -q -p no:cacheprovider` completed normally. Upstream FastAPI/Starlette test-client deprecation warnings remain; they did not affect these tests.

Not verified live in this environment:

- Neo4j schema/index creation and Cypher execution against a running server: Docker daemon unavailable. The opt-in test is provided in `tests/test_neo4j.py`.
- Paid OpenRouter inference, actual Jev/semantic-model quality, and chosen embedding-model dimensions: no live credentials/models were used. HTTP contract, schema validation, retries, failures and usage accounting were tested with mock transport.
- Live Wikipedia retrieval and Docker image startup. External fallback was verified using a synthetic retriever through the same ingestion/retrieval services.

These are explicit validation gaps, not claims that the external integrations passed. Configure the services and run the documented smoke/integration tests before production deployment.
