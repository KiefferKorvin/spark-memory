# Graph schema

Every semantic node has the `MemoryNode` label, a unique UUID string `id`, a concrete type label and a validated JSON `payload`. Indexed projections include `label`, `label_key`, `text`, `aliases`, `routing_summary`, `embedding`, and `kind`. See [schema.cypher](schema.cypher) for constraints/indexes. Neo4j 5.26 Community is the Compose target.

| Layer | Nodes | Relationships |
|---|---|---|
| Documents | Source, Document, Section, Chunk | PROVIDES, CONTAINS |
| Ontology | Concept, TaxonomyGroup, ExternalConcept | BROADER_THAN, RELATED_TO, HAS_MEMBER, EXACT_MATCH, CLOSE_MATCH |
| Semantic assertions | Assertion (fact/claim/event/decision/preference/procedure) | SUPPORTED_BY, ABOUT, CONTRADICTS |

Documents/sections/chunks can have multiple ABOUT or MENTIONS links. Concepts have multiple parents. BROADER_THAN, CONTAINS and HAS_MEMBER are checked for cycles; related links and contradiction links may connect freely. Endpoint kinds are validated. Assertion support may target an original Chunk or a small Document leaf. UNESCO concepts have original URIs and deterministic URI-derived IDs; LOCAL IDs derive from normalized labels. See [UNESCO](UNESCO.md) for official/import-only relationships, multilingual payloads and local extension rules.

Source identity includes content hash and origin; document identity includes MIME, content hash and parser version. Repeated content reuses a document while a different origin gains a separate Source/PROVIDES relationship. Changed source content produces a new document version and retains the old one. Concept IDs derive from normalized labels; exact labels and aliases precede hybrid/semantic resolution. Assertion IDs include support-node ID and proposition, so conflicting claims from separate sources coexist.

Graph writes are transactional. An internal `MemoryLock` serializes mutation transactions before checking persisted reachability, preventing concurrent DAG-cycle insertion. No graph write is performed until source parsing and all required inference has succeeded. Pydantic validation catches malformed records before writes; Cypher values use parameters and relationship labels use a fixed enum.

`MemoryRecord(category,key,payload)` stores query state, ordered events, inference usage and enrichment markers. These operational records are separate from semantic truth. `navigation_hits`, `successful_retrievals`, and `last_accessed_at` affect navigation only. Provenance walks containment ancestors to documents/sources and includes section paths and contradictory assertion IDs.

Inspect a concept neighborhood:

```cypher
MATCH (c:Concept {label_key:'rootless voicings'})-[r]-(n:MemoryNode)
RETURN c, r, n;
```

Find source support:

```cypher
MATCH (s:Source)-[:PROVIDES]->(d:Document)-[:CONTAINS*0..]->(leaf)
OPTIONAL MATCH (a:Assertion)-[:SUPPORTED_BY]->(leaf)
RETURN s, d, leaf, a LIMIT 50;
```
