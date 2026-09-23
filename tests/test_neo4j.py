"""Opt-in integration check against a dedicated empty test database, never user data."""
import os
from uuid import uuid4

import pytest

from graph_memory.config import Settings
from graph_memory.graph import Neo4jGraph
from graph_memory.models import Concept, Document, Edge, Relation


@pytest.mark.skipif(not os.environ.get("NEO4J_TEST_URI"), reason="Dedicated Neo4j integration database not configured")
async def test_neo4j_transaction_indexes_and_restart():
    settings = Settings(_env_file=None, neo4j_uri=os.environ.get("NEO4J_TEST_URI", ""),
                        neo4j_password=os.environ.get("NEO4J_TEST_PASSWORD", ""), embedding_dimensions=8)
    repository = Neo4jGraph(settings)
    prefix = "test-" + str(uuid4())
    ids = [prefix+"a", prefix+"b", prefix+"c", prefix+"u", prefix+"d"]
    authority = "TEST-" + prefix[-12:]
    try:
        await repository.initialize()
        a = Concept(id=ids[0], label="synthetic piano", preferred_label="synthetic piano", embedding=[1.0]+[0.0]*7)
        b = Concept(id=ids[1], label="synthetic instrument", preferred_label="synthetic instrument", embedding=[1.0]+[0.0]*7)
        await repository.put([a,b], [Edge(source=b.id,target=a.id,relation=Relation.BROADER_THAN)])
        with pytest.raises(ValueError, match="cycle"):
            await repository.put([Concept(id=ids[2], label="rollback", preferred_label="rollback")], [Edge(source=a.id,target=b.id,relation=Relation.BROADER_THAN)])
        assert await repository.get(ids[2]) is None
        assert (await repository.exact_concept("synthetic piano")).id == a.id
        assert any(n.id == a.id for n, _ in await repository.candidates("synthetic piano", a.embedding, 5))
        # Imported taxonomy concepts become retrieval candidates only once content is ABOUT them.
        official = Concept(id=ids[3], label="synthetic official", preferred_label="synthetic official", origin=authority,
                           uri="https://example.org/" + prefix, ontology_status="external")
        await repository.import_taxonomy([official], [], {"authority": authority, "fingerprint": prefix})
        assert all(n.id != official.id for n, _ in await repository.candidates("synthetic official", [], 5))
        doc = Document(id=ids[4], label="synthetic doc", text="synthetic official text", retrieval_leaf=True)
        await repository.put([doc, official], [Edge(source=doc.id, target=official.id, relation=Relation.ABOUT)])
        assert any(n.id == official.id for n, _ in await repository.candidates("synthetic official", [], 5))
        for sequence in (1, 2):
            await repository.record("event", f"{prefix}:{sequence:08}", {"sequence": sequence})
        assert await repository.records("event", prefix + ":", after=f"{prefix}:{1:08}") == [{"sequence": 2}]
        await repository.close()
        repository = Neo4jGraph(settings)
        assert (await repository.get(a.id)).label == a.label
    finally:
        await repository.run("MATCH (n:MemoryNode) WHERE n.id IN $ids DETACH DELETE n", ids=ids)
        await repository.run("MATCH (r:MemoryRecord) WHERE r.key=$authority OR r.key STARTS WITH $prefix DELETE r", authority=authority, prefix=prefix)
        await repository.close()

