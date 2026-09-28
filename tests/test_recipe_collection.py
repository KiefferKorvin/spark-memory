import pytest
from graph_memory.graph import InMemoryGraph
from graph_memory.models import IngestRequest,Source
from graph_memory.recipes import ingest,lookup,migrate,ROOT_ID


def request():
    return IngestRequest(title='Omelette',text='Public recipe',metadata={'original_uri':'https://example.org/omelette',
        'pakt_recipe':{'name':'Omelette','ingredients_text':['2 œufs'],'instructions':['Cuire.'],'yield_servings':1,'user_id':999}})


@pytest.mark.asyncio
async def test_direct_collection_is_idempotent_scoped_and_sanitized(monkeypatch):
    repo=InMemoryGraph()
    await ingest(repo,request());await ingest(repo,request())
    async def forbidden(*args,**kwargs):pytest.fail('No graph exploration')
    monkeypatch.setattr(repo,'candidates',forbidden)
    result=await lookup(repo,'omelette')
    assert result['collection_id']==ROOT_ID and len(result['recipes'])==1
    assert 'user_id' not in result['recipes'][0]['body']
    assert len(repo.nodes)==3 and len(repo.edges)==2


@pytest.mark.asyncio
async def test_old_recipe_sources_migrate_once(monkeypatch):
    repo=InMemoryGraph();r=request()
    source=Source(id='legacy',label='Omelette',source_type='api',uri='https://example.org/omelette',mime_type='text/plain',content_hash='hash',metadata=r.metadata)
    await repo.put([source],[]);await migrate(repo)
    assert len((await lookup(repo))['recipes'])==1
    async def forbidden():pytest.fail('Migration must not rescan')
    monkeypatch.setattr(repo,'recipe_sources',forbidden)
    await migrate(repo)
