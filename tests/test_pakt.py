"""Generated artifacts must not satisfy PAKT's original-source requirement."""
import pytest

from graph_memory.config import Settings
from graph_memory.demo import DemoModels
from graph_memory.graph import InMemoryGraph
from graph_memory.models import IngestRequest, QueryRequest
from graph_memory.service import Memory


@pytest.mark.asyncio
async def test_generated_artifact_does_not_satisfy_original_source_query():
    memory=Memory(Settings(memory_mode='demo',unesco_required=False,_env_file=None),InMemoryGraph(),DemoModels())
    await memory.initialize()
    try:
        await memory.ingest(IngestRequest(title='Lentils',text='Lentils contain protein and fiber.',
                                          source_type='generated',metadata={'generated':True}))
        ordinary=await memory.query(QueryRequest(query='Lentils protein fiber',allow_external=False))
        assert ordinary['evidence']
        original=await memory.query(QueryRequest(query='Lentils protein fiber',allow_external=False,original_sources_only=True))
        assert original['evidence']==[]
        assert original['coverage']['overall_status']=='INSUFFICIENT'
        await memory.ingest(IngestRequest(title='Lentils source',text='Lentils contain protein and fiber.',
                                          metadata={'original_uri':'https://example.org/lentils'}))
        sourced=await memory.query(QueryRequest(query='Lentils protein fiber',allow_external=False,original_sources_only=True))
        assert sourced['evidence']
        assert all(source['uri']=='https://example.org/lentils'
                   for e in sourced['evidence'] for source in e['provenance']['sources'])
    finally:
        await memory.close()
