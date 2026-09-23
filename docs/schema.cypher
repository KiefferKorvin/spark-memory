// Startup runs equivalent statements. Set vector.dimensions to EMBEDDING_DIMENSIONS.
CREATE CONSTRAINT memory_node_id IF NOT EXISTS FOR (n:MemoryNode) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT memory_record_id IF NOT EXISTS FOR (n:MemoryRecord) REQUIRE (n.category, n.key) IS UNIQUE;
CREATE CONSTRAINT memory_lock_id IF NOT EXISTS FOR (n:MemoryLock) REQUIRE n.id IS UNIQUE;
CREATE INDEX memory_concept_label IF NOT EXISTS FOR (n:Concept) ON (n.label_key);
CREATE CONSTRAINT unesco_uri IF NOT EXISTS FOR (n:UNESCO) REQUIRE n.uri IS UNIQUE;
CREATE FULLTEXT INDEX unesco_text IF NOT EXISTS FOR (n:UNESCOConcept) ON EACH [n.label, n.aliases, n.routing_summary];
CREATE FULLTEXT INDEX memory_text IF NOT EXISTS FOR (n:MemoryNode) ON EACH [n.label, n.text, n.routing_summary, n.aliases];
CREATE VECTOR INDEX memory_vector IF NOT EXISTS FOR (n:MemoryNode) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};
