"""Property graph stores. The in-memory adapter is exclusively for tests/demo."""
import asyncio
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from typing import Protocol

from .models import Edge, Node, Relation, node_from, now


def words(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold()))


# Question words dilute lexical overlap ("what is the..."); English and French cover the configured corpora.
STOP = set("a an and are as at be by can do does for from how i in is it of on or should the them to "
           "what when where which who why with au aux ce de des du en est et la le les un une qui que quoi "
           "comment pourquoi pour par sur dans".split())


def terms(text: str) -> set[str]:
    tokens = words(text)
    return tokens - STOP or tokens


def taxonomy_terms(query, frequencies, size):
    """Lexical taxonomy query: words that label many thesaurus entries ("technique", "non") are dropped unless
    all are, and each kept word also matches its naive singular/plural form ("foods" finds "food")."""
    # ponytail: 0.5% document-frequency cut and English/French "s" plurals; use a stemming analyzer if this misfires.
    tokens = sorted(terms(query), key=lambda t: (frequencies.get(t, 0), t))
    kept = [t for t in tokens if frequencies.get(t, 0) <= max(5, size / 200)] or tokens[:1]
    return sorted({form for t in kept for form in (t, t[:-1] if len(t) > 3 and t.endswith("s") else t + "s")})


def interleave(rankings, limit):
    """Round-robin merge without duplicates, so semantic and lexical candidates both reach the classifier."""
    merged = {}
    for group in itertools.zip_longest(*rankings):
        for node in group:
            if node is not None:
                merged.setdefault(node.id, node)
    return list(merged.values())[:limit]


# Imported taxonomy entries without attached content are dead ends for evidence retrieval.
CONTENT_FILTER = "(coalesce(n.origin,'LOCAL')='LOCAL' OR EXISTS { (n)<-[:ABOUT]-() })"


def cosine(a, b):
    if not a or len(a) != len(b):
        return 0.0
    divisor = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / divisor if divisor else 0.0


def rank(node, query, vector, hits=0, similarity=None):
    tokens = terms(query)
    label = words(node.label + " " + " ".join(getattr(node, "aliases", [])))
    content = words(node.routing_summary + " " + node.summary + " " + node.text)
    lexical = len(tokens & (label | content)) / max(1, len(tokens))
    semantic = cosine(node.embedding, vector) if similarity is None else similarity
    return (lexical * 0.5 + max(0, semantic) * 0.35
            + len(tokens & label) / max(1, len(tokens)) * 0.1
            + min(hits, 20) * 0.0025)


class GraphRepository(Protocol):
    async def initialize(self): ...
    async def close(self): ...
    async def get(self, node_id: str) -> Node | None: ...
    async def put(self, nodes: list[Node], edges: list[Edge]): ...
    async def candidates(self, query: str, vector: list[float], limit: int,
                         parent: str | None = None, kind: str | None = None) -> list[tuple[Node, float]]: ...
    async def neighbors(self, node_id: str, limit: int = 100) -> tuple[list[Node], list[Edge]]: ...
    async def exact_concept(self, label: str, authority: str | None = None) -> Node | None: ...
    async def import_taxonomy(self, nodes, edges, manifest): ...
    async def taxonomy_candidates(self, query: str, limit: int = 12, authority: str = "UNESCO", vector=None): ...
    async def taxonomy_concepts(self, authority: str = "UNESCO") -> list[tuple[Node, str | None]]: ...
    async def set_ontology_embeddings(self, rows: list[tuple[str, str, list[float]]]): ...
    async def taxonomy_roots(self, authority="UNESCO"): ...
    async def taxonomy_ancestors(self, node_id: str, authority: str = "UNESCO"): ...
    async def unesco_ancestors(self, node_id: str): ...
    async def provenance(self, node_id: str) -> dict: ...
    async def record(self, category: str, key: str, data: dict): ...
    async def read_record(self, category: str, key: str) -> dict | None: ...
    async def records(self, category: str, prefix: str = "", after: str | None = None) -> list[dict]: ...
    async def delete_records(self, category: str, prefix: str = ""): ...
    async def stale_embeddings(self, dimensions: int, limit: int) -> list[Node]: ...
    async def update_embeddings(self, nodes: list[Node]): ...
    async def hit(self, node_id: str, success: bool = False) -> int: ...


def validate_edges(nodes: dict, edges: list[Edge]):
    adjacency = defaultdict(set)
    allowed = {
        "PROVIDES": ({"Source"}, {"Document"}),
        "CONTAINS": ({"Document", "Section"}, {"Section", "Chunk"}),
        "ABOUT": ({"Document", "Section", "Chunk", "Assertion"}, {"Concept"}),
        "MENTIONS": ({"Document", "Section", "Chunk"}, {"Concept"}),
        "BROADER_THAN": ({"Concept"}, {"Concept"}),
        "RELATED_TO": ({"Concept"}, {"Concept"}),
        "SEMANTIC_RELATION": ({"Concept"}, {"Concept"}),
        "HAS_MEMBER": ({"TaxonomyGroup"}, {"Concept", "TaxonomyGroup"}),
        "EXACT_MATCH": ({"Concept"}, {"ExternalConcept"}),
        "CLOSE_MATCH": ({"Concept"}, {"ExternalConcept"}),
        "SUPPORTED_BY": ({"Assertion"}, {"Document", "Chunk"}),
        "CONTRADICTS": ({"Assertion"}, {"Assertion"}),
    }
    for edge in edges:
        if edge.source not in nodes or edge.target not in nodes:
            raise ValueError("Dangling edge")
        left, right = allowed[edge.relation.value]
        if nodes[edge.source].kind not in left or nodes[edge.target].kind not in right:
            raise ValueError("Invalid relationship endpoint types")
        if edge.source == edge.target and not (edge.relation == Relation.RELATED_TO and edge.metadata.get("authority") == getattr(nodes[edge.source], "origin", None) and getattr(nodes[edge.source], "origin", "LOCAL") != "LOCAL"):
            raise ValueError("Self links are not supported")
        if edge.relation in (Relation.BROADER_THAN, Relation.CONTAINS, Relation.HAS_MEMBER):
            adjacency[edge.source].add(edge.target)
    # Iterative Kahn traversal permits arbitrary hierarchy depth.
    indegree = {n: 0 for n in nodes}
    for children in adjacency.values():
        for child in children:
            indegree[child] += 1
    ready = [n for n, degree in indegree.items() if not degree]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for child in adjacency[current]:
            indegree[child] -= 1
            if not indegree[child]:
                ready.append(child)
    if visited != len(nodes):
        raise ValueError("Hierarchy cycle rejected")


def check_manifest(previous, current):
    if previous and previous["fingerprint"] != current["fingerprint"]:
        raise ValueError("Ontology snapshot differs; use an explicit taxonomy migration")


def protect_taxonomy(nodes, edges):
    for edge in edges:
        a, b = nodes.get(edge.source), nodes.get(edge.target)
        if edge.relation in (Relation.BROADER_THAN, Relation.RELATED_TO, Relation.HAS_MEMBER, Relation.SEMANTIC_RELATION):
            if getattr(b, "origin", "LOCAL") != "LOCAL" and (edge.relation not in (Relation.RELATED_TO, Relation.SEMANTIC_RELATION) or getattr(a, "origin", "LOCAL") != "LOCAL"):
                raise ValueError("Official taxonomy relationships are writable only by the taxonomy importer")


# Cosine in [-1, 1] computed in the database, so payloads need not carry vectors back to Python.
SIMILARITY = ("CASE WHEN size($vector)>0 AND size(coalesce(n.embedding,[]))=size($vector) "
              "THEN 2*vector.similarity.cosine(n.embedding,$vector)-1 ELSE 0.0 END")
VECTOR_INDEXES = {"memory_vector": "MemoryNode) ON (n.embedding", "ontology_vector": "ImportedConcept) ON (n.ontology_embedding"}


def node_properties(node):
    # The vector lives only in the indexed property: in the payload, 4096 floats would add ~90 KB to every row read.
    return {"payload": node.model_dump_json(exclude={"embedding"}), "kind": node.kind, "label": node.label,
            "label_key": node.label.casefold(), "text": node.text, "routing_summary": node.routing_summary,
            "embedding": node.embedding, "aliases": [a.casefold() for a in getattr(node, "aliases", [])],
            "origin": getattr(node, "origin", None), "uri": getattr(node, "uri", None)}


class InMemoryGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.data = {}
        self.stats = defaultdict(lambda: {"navigation_hits": 0, "successful_retrievals": 0})
        # Taxonomy embeddings live apart from node embeddings, like Neo4j's separate ontology_vector index.
        self.ontology_vectors = {}
        self.lock = asyncio.Lock()

    async def initialize(self):
        pass

    async def close(self):
        pass

    async def get(self, node_id):
        node = self.nodes.get(node_id)
        return node.model_copy(deep=True) if node else None

    async def put(self, nodes, edges):
        async with self.lock:
            protect_taxonomy({**self.nodes, **{n.id: n for n in nodes}}, edges)
            # Copy-on-write: stored nodes are never mutated, so a failed validation leaves the store untouched.
            merged = dict(self.nodes)
            for node in nodes:
                if getattr(node, "origin", "LOCAL") != "LOCAL" and node.id not in self.nodes:
                    raise ValueError("Imported nodes require the taxonomy importer")
                current = merged.setdefault(node.id, node.model_copy(deep=True))
                if node.kind == "Concept" and node.origin == "LOCAL":
                    merged[node.id] = current.model_copy(update={"aliases": sorted(set(current.aliases + node.aliases))})
            links = {**self.edges, **{e.id: e for e in edges}}
            validate_edges(merged, list(links.values()))
            self.nodes, self.edges = merged, links

    async def exact_concept(self, label, authority=None):
        for node in sorted(self.nodes.values(), key=lambda n: getattr(n, "origin", "LOCAL") == "LOCAL"):
            if node.kind == "Concept" and (authority is None or node.origin in (authority, "LOCAL")) and label.casefold() in {
                node.label.casefold(), *(a.casefold() for a in node.aliases)
            }:
                return node.model_copy(deep=True)
        return None

    async def import_taxonomy(self, nodes, edges, manifest):
        async with self.lock:
            previous = self.data.get(("taxonomy", manifest["authority"]))
            check_manifest(previous, manifest)
            merged = {**self.nodes, **{n.id: n.model_copy(deep=True) for n in nodes}}
            links = {**self.edges, **{e.id: e for e in edges}}
            validate_edges(merged, list(links.values()))
            self.nodes, self.edges = merged, links
            await self.record("taxonomy", manifest["authority"], manifest)

    async def taxonomy_candidates(self, query, limit=12, authority="UNESCO", vector=None):
        nodes = [n for n in self.nodes.values() if n.kind == "Concept" and n.origin == authority]
        frequencies = Counter(t for n in nodes for t in words(n.label + " " + " ".join(n.aliases)))
        wanted = " ".join(taxonomy_terms(query, frequencies, len(nodes)))
        lexical = sorted((n for n in nodes if rank(n, wanted, []) > 0), key=lambda n: (-rank(n, wanted, []), n.id))
        semantic = sorted((n for n in nodes if vector and n.id in self.ontology_vectors),
                          key=lambda n: (-cosine(self.ontology_vectors[n.id][1], vector), n.id))
        return interleave([semantic[:limit], lexical[:limit]], limit)

    async def taxonomy_concepts(self, authority="UNESCO"):
        return [(n.model_copy(deep=True), self.ontology_vectors.get(n.id, (None,))[0])
                for n in self.nodes.values() if n.kind == "Concept" and n.origin == authority]

    async def set_ontology_embeddings(self, rows):
        for node_id, key, vector in rows:
            self.ontology_vectors[node_id] = (key, vector)

    async def taxonomy_roots(self, authority="UNESCO"):
        groups = [n for n in self.nodes.values() if n.kind == "TaxonomyGroup" and n.origin == authority and n.group_type == "domain"]
        children = {e.target for e in self.edges.values() if e.relation == Relation.BROADER_THAN}
        return groups or [n for n in self.nodes.values() if n.kind == "Concept" and n.origin == authority and n.id not in children][:100]

    async def taxonomy_ancestors(self, node_id, authority="UNESCO"):
        if node_id not in self.nodes:
            return []
        ids = {node_id}
        while True:
            expanded = ids | {e.source for e in self.edges.values() if e.target in ids and e.relation == Relation.BROADER_THAN}
            if expanded == ids:
                break
            ids = expanded
        return [await self.get(i) for i in sorted(ids) if self.nodes[i].kind == "Concept" and self.nodes[i].origin == authority]

    async def neighbors(self, node_id, limit=100):
        edges = [e for e in self.edges.values() if node_id in (e.source, e.target)][:limit]
        ids = {i for e in edges for i in (e.source, e.target)} - {node_id}
        return [await self.get(i) for i in sorted(ids)], edges

    async def candidates(self, query, vector, limit, parent=None, kind=None):
        nodes = list(self.nodes.values())
        if parent:
            # Fetch the entire test neighborhood before ranking, unlike a first-N slice.
            ids = {e.target if e.source == parent else e.source
                   for e in self.edges.values() if parent in (e.source, e.target)}
            nodes = [n for n in nodes if n.id in ids]
        about = {e.target for e in self.edges.values() if e.relation == Relation.ABOUT}
        nodes = [n for n in nodes if n.kind not in ("Source", "ExternalConcept")
                 and (kind is None or n.kind == kind)
                 and (getattr(n, "origin", "LOCAL") == "LOCAL" or n.id in about)]
        scored = [(n, rank(n, query, vector, self.stats[n.id]["successful_retrievals"])) for n in nodes]
        return sorted(scored, key=lambda pair: (-pair[1], pair[0].id))[:limit]

    async def unesco_ancestors(self, node_id):
        return await self.taxonomy_ancestors(node_id, "UNESCO")

    async def provenance(self, node_id):
        reachable = {node_id}
        for _ in range(len(self.nodes)):
            ancestors = {e.source for e in self.edges.values()
                         if e.target in reachable and e.relation in (Relation.CONTAINS, Relation.PROVIDES)}
            supports = {e.target for e in self.edges.values()
                        if e.source in reachable and e.relation == Relation.SUPPORTED_BY}
            new = ancestors | supports | reachable
            if new == reachable:
                break
            reachable = new
        return {"sources": [n.model_dump(mode="json") for i, n in self.nodes.items()
                            if i in reachable and n.kind == "Source"],
                "documents": [i for i in reachable if self.nodes[i].kind == "Document"],
                "section_path": getattr(self.nodes.get(node_id), "section_path", []),
                "contradictions": [e.target if e.source == node_id else e.source for e in self.edges.values()
                                   if e.relation == Relation.CONTRADICTS and node_id in (e.source, e.target)]}

    async def record(self, category, key, data):
        self.data[(category, key)] = json.loads(json.dumps(data))

    async def read_record(self, category, key):
        data = self.data.get((category, key))
        return json.loads(json.dumps(data)) if data is not None else None

    async def records(self, category, prefix="", after=None):
        return [json.loads(json.dumps(v)) for (c, k), v in sorted(self.data.items())
                if c == category and k.startswith(prefix) and (after is None or k > after)]

    async def delete_records(self, category, prefix=""):
        for key in [k for k in self.data if k[0] == category and k[1].startswith(prefix)]:
            del self.data[key]

    async def stale_embeddings(self, dimensions, limit):
        return [n.model_copy(deep=True) for n in self.nodes.values() if n.embedding and len(n.embedding) != dimensions][:limit]

    async def update_embeddings(self, nodes):
        async with self.lock:
            for node in nodes:
                self.nodes[node.id] = self.nodes[node.id].model_copy(update={"embedding": node.embedding})

    async def hit(self, node_id, success=False):
        self.stats[node_id]["successful_retrievals" if success else "navigation_hits"] += 1
        return self.stats[node_id]["successful_retrievals"]


class Neo4jGraph:
    def __init__(self, settings):
        from neo4j import AsyncGraphDatabase
        self.settings = settings
        # Schema hints (e.g. "CONTRADICTS does not exist yet") would otherwise log a warning per provenance query.
        self.driver = AsyncGraphDatabase.driver(settings.neo4j_uri, auth=(
            settings.neo4j_username, settings.neo4j_password.get_secret_value()), notifications_min_severity="OFF")
        self.frequencies = {}

    async def run(self, cypher, **parameters):
        records, _, _ = await self.driver.execute_query(
            cypher, parameters_=parameters, database_=self.settings.neo4j_database)
        return [dict(row) for row in records]

    async def initialize(self):
        await self.driver.verify_connectivity()
        for statement in [
            "CREATE CONSTRAINT memory_node_id IF NOT EXISTS FOR (n:MemoryNode) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT memory_record_id IF NOT EXISTS FOR (n:MemoryRecord) REQUIRE (n.category, n.key) IS UNIQUE",
            "CREATE CONSTRAINT memory_lock_id IF NOT EXISTS FOR (n:MemoryLock) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX memory_concept_label IF NOT EXISTS FOR (n:Concept) ON (n.label_key)",
            "CREATE CONSTRAINT ontology_uri IF NOT EXISTS FOR (n:ImportedTaxonomy) REQUIRE (n.origin,n.uri) IS UNIQUE",
            "CREATE FULLTEXT INDEX ontology_text IF NOT EXISTS FOR (n:ImportedConcept) ON EACH [n.label,n.aliases,n.routing_summary]",
            "CREATE FULLTEXT INDEX memory_text IF NOT EXISTS FOR (n:MemoryNode) ON EACH [n.label, n.text, n.routing_summary, n.aliases]",
        ]:
            await self.run(statement)
        await self.vector_indexes()
        await self.run("MATCH (n:UNESCO) SET n:ImportedTaxonomy")
        await self.run("MATCH (n:UNESCOConcept) SET n:ImportedConcept")
        await self.run("CALL db.awaitIndexes(60)")

    async def vector_indexes(self, migrate=False):
        """Creates the vector indexes. An index keeps the dimension it was created with, so after an
        EMBEDDING_DIMENSIONS change it is rebuilt only when no stored vector would be orphaned (e.g. a reset graph)
        or during the explicit, paid re-embedding migration (python -m graph_memory.migrate)."""
        dimensions = self.settings.embedding_dimensions
        rows = await self.run("SHOW INDEXES YIELD name, type, options WHERE type='VECTOR' RETURN name, options")
        stale = {r["name"]: r["options"]["indexConfig"]["vector.dimensions"] for r in rows
                 if r["name"] in VECTOR_INDEXES and r["options"]["indexConfig"]["vector.dimensions"] != dimensions}
        if stale and not migrate and await self.run("MATCH (n:MemoryNode) WHERE size(coalesce(n.embedding,[]))>0 "
                                                    "OR n.ontology_embedding IS NOT NULL RETURN n.id LIMIT 1"):
            raise ValueError(f"Vector indexes {stale} do not match EMBEDDING_DIMENSIONS={dimensions}: run "
                             "python -m graph_memory.migrate to re-embed the stored vectors, or use a separate database")
        for name in stale:
            await self.run(f"DROP INDEX {name} IF EXISTS")
        for name, target in VECTOR_INDEXES.items():  # A separate ontology index: thesaurus vectors never crowd content hits.
            await self.run(f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{target}) OPTIONS {{indexConfig: "
                           f"{{`vector.dimensions`: {dimensions}, `vector.similarity_function`: 'cosine'}}}}")

    async def close(self):
        await self.driver.close()

    async def get(self, node_id):
        rows = await self.run("MATCH (n:MemoryNode {id:$id}) RETURN n.payload AS payload", id=node_id)
        return node_from(json.loads(rows[0]["payload"])) if rows else None

    async def put(self, nodes, edges):
        node_map = {n.id: n for n in nodes}
        # Validate endpoint kinds without loading the entire production graph.
        missing = {i for e in edges for i in (e.source, e.target)} - node_map.keys()
        for node_id in missing:
            existing = await self.get(node_id)
            if existing:
                node_map[node_id] = existing
        validate_edges(node_map, edges)
        protect_taxonomy(node_map, edges)

        async def write(tx):
            # ponytail: serialize graph mutations to make DAG checks race-safe; shard locks if write throughput requires it.
            await (await tx.run("MERGE (l:MemoryLock {id:'graph'}) SET l.revision=coalesce(l.revision,0)+1")).consume()
            for node in nodes:
                if node.kind not in {"Source", "Document", "Section", "Chunk", "Concept", "ExternalConcept", "Assertion"}:
                    raise ValueError("Unknown node kind")
                if node.kind == "Concept":
                    result = await tx.run("MATCH (n:Concept {id:$id}) RETURN n.payload AS payload", id=node.id)
                    current = await result.single()
                    if node.origin != "LOCAL" and not current:
                        raise ValueError("Imported nodes require the taxonomy importer")
                    if current:
                        saved = node_from(json.loads(current["payload"]))
                        if saved.origin == "LOCAL":
                            saved.aliases = sorted(set(saved.aliases + node.aliases))
                        node = saved
                props = node_properties(node)
                await (await tx.run(f"MERGE (n:MemoryNode {{id:$id}}) ON CREATE SET n += $props SET n:{node.kind}",
                                   id=node.id, props=props)).consume()
                if node.kind == "Concept":
                    await (await tx.run("MATCH (n:Concept {id:$id}) SET n.payload=$payload,n.aliases=$aliases",
                                       id=node.id, payload=props["payload"], aliases=props["aliases"])).consume()
            for edge in edges:
                relation = edge.relation.value  # enum, never user Cypher
                if edge.relation in (Relation.BROADER_THAN, Relation.CONTAINS):
                    result = await tx.run(f"MATCH (a:MemoryNode {{id:$a}}),(b:MemoryNode {{id:$b}}) "
                                          f"RETURN EXISTS {{ MATCH (b)-[:{relation}*0..]->(a) }} AS cycle",
                                          a=edge.source, b=edge.target)
                    row = await result.single()
                    if row is None or row["cycle"]:
                        raise ValueError("Hierarchy cycle rejected")
                await (await tx.run(f"MATCH (a:MemoryNode {{id:$a}}),(b:MemoryNode {{id:$b}}) "
                                   f"MERGE (a)-[:{relation} {{id:$id}}]->(b)",
                                   a=edge.source, b=edge.target, id=edge.id)).consume()
        async with self.driver.session(database=self.settings.neo4j_database) as session:
            await session.execute_write(write)

    async def exact_concept(self, label, authority=None):
        rows = await self.run("MATCH (n:Concept) WHERE (n.label_key=$label OR $label IN n.aliases) AND ($authority IS NULL OR n.origin=$authority OR coalesce(n.origin,'LOCAL')='LOCAL') RETURN n.payload AS payload ORDER BY CASE WHEN coalesce(n.origin,'LOCAL')<>'LOCAL' THEN 0 ELSE 1 END,n.id LIMIT 1",
                              label=label.casefold(), authority=authority)
        return node_from(json.loads(rows[0]["payload"])) if rows else None

    async def import_taxonomy(self, nodes, edges, manifest):
        from itertools import islice
        if not callable(nodes):
            from .taxonomies import validate_snapshot
            validate_snapshot(nodes, edges, manifest['authority'])
            validate_edges({n.id: n for n in nodes}, edges)
        # RF2 adapters validate their complete staged hierarchy before opening this transaction.
        def batches(values):
            iterator = iter(values() if callable(values) else values)
            while batch := list(islice(iterator, 1000)):
                yield batch
        async def write(tx):
            await (await tx.run("MERGE (l:MemoryLock {id:'graph'}) SET l.revision=coalesce(l.revision,0)+1")).consume()
            result = await tx.run("MATCH (r:MemoryRecord {category:'taxonomy',key:$authority}) RETURN r.payload AS payload", authority=manifest['authority'])
            previous = await result.single()
            check_manifest(json.loads(previous['payload']) if previous else None, manifest)
            if previous:
                return
            for batch in batches(nodes):
                for kind in ('Concept', 'TaxonomyGroup'):
                    rows = [{'id': n.id, 'props': node_properties(n)} for n in batch if n.kind == kind]
                    label = ':ImportedConcept' if kind == 'Concept' else ''
                    await (await tx.run(f"UNWIND $rows AS row MERGE (n:MemoryNode {{id:row.id}}) SET n += row.props SET n:ImportedTaxonomy:{kind}{label}", rows=rows)).consume()
            for batch in batches(edges):
                for relation in (Relation.BROADER_THAN, Relation.RELATED_TO, Relation.HAS_MEMBER, Relation.SEMANTIC_RELATION):
                    rows = [{'id': e.id, 'source': e.source, 'target': e.target, 'metadata': json.dumps(e.metadata)} for e in batch if e.relation == relation]
                    await (await tx.run(f"UNWIND $rows AS row MATCH (a:MemoryNode {{id:row.source}}),(b:MemoryNode {{id:row.target}}) MERGE (a)-[r:{relation.value} {{id:row.id}}]->(b) SET r.metadata=row.metadata", rows=rows)).consume()
            await (await tx.run("MERGE (r:MemoryRecord {category:'taxonomy',key:$authority}) SET r.payload=$payload", authority=manifest['authority'], payload=json.dumps(manifest))).consume()
        async with self.driver.session(database=self.settings.neo4j_database) as session:
            await session.execute_write(write)

    async def taxonomy_candidates(self, query, limit=12, authority="UNESCO", vector=None):
        if authority not in self.frequencies:
            rows = await self.run("MATCH (n:ImportedConcept) WHERE n.origin=$authority RETURN n.label AS label, n.aliases AS aliases", authority=authority)
            self.frequencies[authority] = len(rows), Counter(t for r in rows for t in words(r["label"] + " " + " ".join(r["aliases"] or [])))
        size, frequencies = self.frequencies[authority]
        query = " OR ".join(taxonomy_terms(query, frequencies, size)[:80]) or "__no_terms__"
        rows = await self.run("CALL db.index.fulltext.queryNodes('ontology_text',$query) YIELD node,score WHERE node.origin=$authority RETURN node.payload AS payload ORDER BY score DESC LIMIT $limit", query=query, limit=limit, authority=authority)
        lexical = [node_from(json.loads(r['payload'])) for r in rows]
        semantic = []
        if vector:
            rows = await self.run("CALL db.index.vector.queryNodes('ontology_vector',$k,$vector) YIELD node,score WHERE node.origin=$authority RETURN node.payload AS payload ORDER BY score DESC", k=limit, vector=vector, authority=authority)
            semantic = [node_from(json.loads(r['payload'])) for r in rows]
        return interleave([semantic, lexical], limit)

    async def taxonomy_concepts(self, authority="UNESCO"):
        rows = await self.run("MATCH (n:ImportedConcept) WHERE n.origin=$authority RETURN n.payload AS payload, n.ontology_embedding_key AS key", authority=authority)
        return [(node_from(json.loads(r["payload"])), r["key"]) for r in rows]

    async def set_ontology_embeddings(self, rows):
        await self.run("UNWIND $rows AS row MATCH (n:MemoryNode {id:row.id}) WHERE n:ImportedConcept "
                       "SET n.ontology_embedding=row.vector, n.ontology_embedding_key=row.key",
                       rows=[{"id": i, "key": key, "vector": vector} for i, key, vector in rows])

    async def taxonomy_roots(self, authority="UNESCO"):
        rows = await self.run("MATCH (n:ImportedTaxonomy) WHERE n.origin=$authority AND NOT EXISTS { MATCH (:ImportedTaxonomy)-[:HAS_MEMBER|BROADER_THAN]->(n) } RETURN n.payload AS payload ORDER BY n.label LIMIT 100", authority=authority)
        return [node_from(json.loads(r['payload'])) for r in rows]

    async def taxonomy_ancestors(self, node_id, authority="UNESCO"):
        rows = await self.run("MATCH (n:MemoryNode {id:$id})<-[:BROADER_THAN*0..]-(a:ImportedConcept) WHERE a.origin=$authority RETURN DISTINCT a.payload AS payload", id=node_id, authority=authority)
        return [node_from(json.loads(r['payload'])) for r in rows]

    async def neighbors(self, node_id, limit=100):
        rows = await self.run("MATCH (n:MemoryNode {id:$id})-[r]-(m:MemoryNode) RETURN m.payload AS payload, "
                              "startNode(r).id AS source,endNode(r).id AS target,type(r) AS relation,r.metadata AS metadata LIMIT $limit",
                              id=node_id, limit=limit)
        return list({r["payload"]: node_from(json.loads(r["payload"])) for r in rows}.values()), [
            Edge(source=r["source"], target=r["target"], relation=r["relation"], metadata=json.loads(r["metadata"] or "{}")) for r in rows]

    async def candidates(self, query, vector, limit, parent=None, kind=None):
        lexical = " OR ".join(sorted(terms(query))[:80]) or "__no_terms__"
        # Independent indexes seed candidates; parent filtering also ranks its whole neighborhood in the DB.
        # The content filter runs before LIMIT so thousands of empty taxonomy labels cannot crowd out content.
        rows = await self.run("CALL db.index.fulltext.queryNodes('memory_text',$query) YIELD node AS n,score "
                              f"WHERE {CONTENT_FILTER} RETURN n.payload AS payload,{SIMILARITY} AS similarity ORDER BY score DESC LIMIT $pool",
                              query=lexical, pool=limit*4, vector=vector)
        if vector:
            rows += await self.run("CALL db.index.vector.queryNodes('memory_vector',$pool,$vector) YIELD node AS n,score "
                                   f"WHERE {CONTENT_FILTER} RETURN n.payload AS payload,{SIMILARITY} AS similarity", pool=limit*4, vector=vector)
        ids = None
        if parent:
            neighbors = await self.run(
                "MATCH (:MemoryNode {id:$parent})--(n:MemoryNode) "
                f"WHERE n.kind <> 'Source' AND n.kind <> 'ExternalConcept' AND ($kind IS NULL OR n.kind=$kind) AND {CONTENT_FILTER} "
                "WITH DISTINCT n, size([t IN $terms WHERE toLower(n.label+' '+n.routing_summary+' '+n.text) CONTAINS t]) "
                "+ CASE WHEN size(n.embedding)=size($vector) AND size($vector)>0 THEN vector.similarity.cosine(n.embedding,$vector) ELSE 0 END "
                "+ 0.01*coalesce(n.successful_retrievals,0) AS score "
                f"RETURN n.payload AS payload,score,{SIMILARITY} AS similarity ORDER BY score DESC LIMIT $pool",
                parent=parent, kind=kind, terms=list(terms(query)), vector=vector, pool=limit*4)
            rows += neighbors
            # Restrict index hits to actual neighbors, without sending a huge child list to Python/Jev.
            candidates = [json.loads(r["payload"])["id"] for r in rows]
            allowed = await self.run("MATCH (:MemoryNode {id:$parent})--(n:MemoryNode) WHERE n.id IN $ids RETURN DISTINCT n.id AS id",
                                     parent=parent, ids=candidates)
            ids = {r["id"] for r in allowed}
        nodes, similarity = {}, {}
        for row in rows:
            node = node_from(json.loads(row["payload"]))
            nodes[node.id], similarity[node.id] = node, row["similarity"]
        scored = [(n, rank(n, query, vector, similarity=similarity[n.id])) for n in nodes.values()
                  if n.kind not in ("Source", "ExternalConcept") and (not kind or n.kind == kind)
                  and (ids is None or n.id in ids)]
        return sorted(scored, key=lambda p: (-p[1], p[0].id))[:limit]

    async def unesco_ancestors(self, node_id):
        return await self.taxonomy_ancestors(node_id, "UNESCO")

    async def provenance(self, node_id):
        rows = await self.run(
            "MATCH (n:MemoryNode {id:$id}) OPTIONAL MATCH (n)-[:SUPPORTED_BY]->(support) "
            "WITH n,coalesce(support,n) AS leaf OPTIONAL MATCH (d:Document)-[:CONTAINS*0..]->(leaf) "
            "OPTIONAL MATCH (s:Source)-[:PROVIDES]->(d) "
            "RETURN collect(DISTINCT s.payload) AS sources,collect(DISTINCT d.id) AS documents", id=node_id)
        contradictions = await self.run("MATCH (:MemoryNode {id:$id})-[:CONTRADICTS]-(a) RETURN a.id AS id", id=node_id)
        node = await self.get(node_id)
        return {"sources": [json.loads(s) for s in rows[0]["sources"]], "documents": rows[0]["documents"],
                "section_path": getattr(node, "section_path", []), "contradictions": [r["id"] for r in contradictions]}

    async def record(self, category, key, data):
        await self.run("MERGE (r:MemoryRecord {category:$category,key:$key}) SET r.payload=$payload",
                       category=category, key=key, payload=json.dumps(data))

    async def read_record(self, category, key):
        rows = await self.run("MATCH (r:MemoryRecord {category:$category,key:$key}) RETURN r.payload AS payload", category=category, key=key)
        return json.loads(rows[0]["payload"]) if rows else None

    async def records(self, category, prefix="", after=None):
        # `after` lets stream polling read only new events instead of the whole trace every tick.
        rows = await self.run("MATCH (r:MemoryRecord {category:$category}) WHERE r.key STARTS WITH $prefix "
                              "AND ($after IS NULL OR r.key > $after) "
                              "RETURN r.payload AS payload ORDER BY r.key", category=category, prefix=prefix, after=after)
        return [json.loads(r["payload"]) for r in rows]

    async def delete_records(self, category, prefix=""):
        await self.run("MATCH (r:MemoryRecord {category:$category}) WHERE r.key STARTS WITH $prefix DELETE r",
                       category=category, prefix=prefix)

    async def stale_embeddings(self, dimensions, limit):
        rows = await self.run("MATCH (n:MemoryNode) WHERE size(coalesce(n.embedding,[]))>0 AND size(n.embedding)<>$dimensions "
                              "RETURN n.payload AS payload LIMIT $limit", dimensions=dimensions, limit=limit)
        return [node_from(json.loads(r["payload"])) for r in rows]

    async def update_embeddings(self, nodes):
        await self.run("UNWIND $rows AS row MATCH (n:MemoryNode {id:row.id}) SET n.embedding=row.embedding, n.payload=row.payload",
                       rows=[{"id": n.id, "embedding": n.embedding, "payload": node_properties(n)["payload"]} for n in nodes])

    async def hit(self, node_id, success=False):
        field = "successful_retrievals" if success else "navigation_hits"
        rows = await self.run(f"MATCH (n:MemoryNode {{id:$id}}) SET n.{field}=coalesce(n.{field},0)+1,n.last_accessed_at=$now "
                              "RETURN coalesce(n.successful_retrievals,0) AS hits", id=node_id, now=now())
        return rows[0]["hits"] if rows else 0
