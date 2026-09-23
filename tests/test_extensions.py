import base64
import io
import json
import zipfile

import httpx
import pytest

from graph_memory.config import Settings
from graph_memory.demo import DemoModels, OnlineDemoModels
from graph_memory.graph import InMemoryGraph
from graph_memory.llm import OpenRouter
from graph_memory.models import Answer, IngestRequest, SourceMetadata
from graph_memory.parsing import extract
from graph_memory.service import Memory
from graph_memory.sources import SafeFetcher
from graph_memory.taxonomies import RF2Snapshot, normalized_json
from graph_memory.unesco import parse_skos


async def test_generic_authorities_coexist_and_custom_json_roundtrip(tmp_path):
    path = tmp_path / 'ontology.ttl'
    path.write_text('@prefix s: <http://www.w3.org/2004/02/skos/core#> . <https://example.org/A> a s:Concept; s:prefLabel "Shared label"@en .', encoding='utf-8')
    a, b = parse_skos(path, 'MEDICINE'), parse_skos(path, 'CUSTOM')
    assert a.nodes[0].id != b.nodes[0].id
    repo = InMemoryGraph()
    for taxonomy in (a,b):
        await repo.import_taxonomy(taxonomy.nodes, taxonomy.edges, taxonomy.manifest)
    assert (await repo.exact_concept('Shared label', 'CUSTOM')).origin == 'CUSTOM'
    assert all(n.origin == 'MEDICINE' for n in await repo.taxonomy_candidates('Shared', authority='MEDICINE'))
    custom = tmp_path / 'custom.json'
    custom.write_text(json.dumps({'nodes':[n.model_dump() for n in b.nodes], 'edges':[], 'manifest':{}}), encoding='utf-8')
    assert normalized_json(custom, 'CUSTOM').nodes[0].uri == 'https://example.org/A'


def test_rf2_streaming_preferred_labels_attributes_and_active_only(tmp_path):
    archive = tmp_path / 'rf2.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('Release/Snapshot/Terminology/sct2_Concept_Snapshot.txt', 'id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\n1\t20260621\t1\t1\t1\n2\t20260621\t1\t1\t1\n3\t20260621\t0\t1\t1\n')
        z.writestr('Release/Snapshot/Terminology/sct2_Description_Snapshot-fr.txt', 'id\tactive\tconceptId\tlanguageCode\ttypeId\tterm\n10\t1\t1\tfr\t900000000000003001\tRacine (concept)\n20\t1\t2\tfr\t900000000000003001\tMaladie (trouble)\n21\t1\t2\tfr\t900000000000013009\tMaladie\n22\t1\t2\tfr\t900000000000013009\tTerme avec "guillemet\n')
        z.writestr('Release/Snapshot/Refset/Language/der2_cRefset_LanguageSnapshot-fr.txt', 'id\tactive\treferencedComponentId\tacceptabilityId\na\t1\t21\t900000000000548007\n')
        z.writestr('Release/Snapshot/Terminology/sct2_Relationship_Snapshot.txt', 'id\tactive\tsourceId\tdestinationId\ttypeId\trelationshipGroup\nr1\t1\t2\t1\t116680003\t0\nr2\t1\t2\t1\t999\t1\nr3\t1\t2\t1\t999\t2\n')
    staged = RF2Snapshot(archive)
    try:
        nodes, edges = list(staged.nodes()), list(staged.edges())
        assert len(nodes) == 2 and nodes[1].label == 'Maladie'
        assert len({e.id for e in edges}) == 3
        assert edges[0].source == nodes[0].id and edges[0].target == nodes[1].id
        assert edges[1].metadata['rf2']['relationshipGroup'] == '1'
    finally:
        staged.close()


async def test_generated_metadata_formatting_and_duplicates():
    class MetadataModels(DemoModels):
        async def structured(self, op, payload, schema, query_id=None):
            if op == 'source_metadata':
                return SourceMetadata(title='Generated piano note', author=None, published_at=None, publisher=None, language='en', description='A note on piano', keywords=['piano'])
            return await super().structured(op,payload,schema,query_id)
    repo = InMemoryGraph()
    memory = Memory(Settings(_env_file=None, memory_mode='demo'), repo, MetadataModels())
    raw = '# Piano\n\n**Bold** and *italic*.\n\n- one\n- two\n\n```python\nx = 2\n```\n'
    first = await memory.ingest(IngestRequest(text=raw, mime_type='text/markdown'))
    doc, source = await repo.get(first['document_id']), await repo.get(first['source_id'])
    assert doc.metadata['formatted_content'] == raw
    assert doc.label == source.label == 'Generated piano note' and source.published_at is None
    second = await memory.ingest(IngestRequest(text=raw, mime_type='text/markdown', filename='copy.md'))
    assert second['duplicate'] and (await repo.get(second['source_id'])).label == 'Generated piano note'


async def test_synthesis_uses_requested_model_and_repairs_invalid_citations():
    calls = []
    async def respond(request):
        body = json.loads(request.content); calls.append(body)
        # First answer cites nothing known ([1] is not an evidence ID), which triggers one repair round.
        answer = {'answer':'A real explanation [1].' if len(calls)==1 else 'A real explanation [E1] [see note].','evidence_ids':['E1']}
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(answer)}}]})
    online = OpenRouter(Settings(_env_file=None, memory_mode='demo', semantic_model='other-model'), InMemoryGraph(), httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    hybrid = OnlineDemoModels(online)
    result = await hybrid.structured('synthesis', {'evidence':[{'id':'E1','text':'Evidence'}]}, Answer)
    assert '[E1]' in result.answer and len(calls) == 2
    assert 'Synthesis must cite supplied evidence IDs inline' in calls[1]['messages'][-1]['content']  # the repair says what was wrong
    assert all(c['model'] == 'z-ai/glm-5.3-flash' for c in calls)
    assert not calls[0]['messages'][0]['content'].endswith('citations.')  # no answer length cap by default
    capped = OpenRouter(Settings(_env_file=None, memory_mode='demo'), InMemoryGraph(), httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    calls.clear()
    await capped.structured('synthesis', {'evidence':[{'id':'E1','text':'Evidence'}], 'answer_max_words': 60}, Answer)
    assert calls[0]['messages'][0]['content'].endswith('Your answer should not be longer than 60 words, not counting citations.')
    rules = calls[0]['messages'][2]['content']  # repeated after the evidence, where they are read last
    assert calls[0]['messages'][2]['role'] == 'user' and 'Every sentence must be supported' in rules and rules.endswith('60 words, not counting citations.')
    assert 'answer_max_words' not in calls[0]['messages'][1]['content']  # an instruction, not user data
    await capped.close()
    await hybrid.close()


def test_docx_preserves_formatting_and_embedded_metadata():
    from docx import Document
    doc = Document(); doc.core_properties.author = 'Fixture Author'
    doc.add_heading('Piano', 1); doc.add_paragraph().add_run('Important').bold = True
    table = doc.add_table(rows=2, cols=2)
    table.cell(0,0).text='Name'; table.cell(0,1).text='Value'; table.cell(1,0).text='Piano'; table.cell(1,1).text='88'
    buf = io.BytesIO(); doc.save(buf)
    text, metadata = extract(buf.getvalue(), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 1000000, True)
    assert '# Piano' in text and '**Important**' in text and '| --- | --- |' in text
    assert metadata['author'] == 'Fixture Author'


def test_pdf_pages_text_and_metadata():
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
    writer = PdfWriter(); page = writer.add_blank_page(600,800)
    font = DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):font})})
    stream = DecodedStreamObject(); stream.set_data(b'BT /F1 12 Tf 50 700 Td (Piano music) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(stream)
    writer.add_metadata({'/Title':'Fixture PDF'})
    buf=io.BytesIO(); writer.write(buf)
    text,metadata=extract(buf.getvalue(),'application/pdf',1000000,True)
    assert '<!-- page 1 -->' in text and 'Piano music' in text and metadata['/Title']=='Fixture PDF'


def test_public_web_retains_address_protection(monkeypatch):
    monkeypatch.setattr('socket.getaddrinfo',lambda *a,**k:[(None,None,None,None,('93.184.216.34',443))])
    fetcher=SafeFetcher([],1000,public_web=True)
    assert fetcher.validate('https://example.org/article')[0].hostname=='example.org'
    monkeypatch.setattr('socket.getaddrinfo',lambda *a,**k:[(None,None,None,None,('127.0.0.1',443))])
    with pytest.raises(ValueError,match='Non-public'):
        fetcher.validate('https://example.org/article')
    text,meta=extract(b'<meta name="author" content="A"><h1>Piano</h1><p><b>Music</b></p>', 'text/html', 10000, True)
    assert '**Music**' in text and meta['author']=='A'


async def test_online_retrievers_parse_and_interleave(monkeypatch):
    from graph_memory.external import build_retriever
    responses = {
        'wikipedia.org': {'query': {'pages': [
            {'title': 'Second', 'fullurl': 'https://en.wikipedia.org/wiki/Second', 'extract': 'Later hit.', 'pageid': 2, 'index': 2},
            {'title': 'Boiling point', 'fullurl': 'https://en.wikipedia.org/wiki/Boiling_point', 'extract': 'Water boils at lower temperature at altitude.', 'pageid': 1, 'index': 1}]}},
        'esearch': {'esearchresult': {'idlist': ['42']}},
        'efetch': b'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article><Journal><Title>J Phys</Title></Journal>'
                  b'<ArticleTitle>Altitude and <i>boiling</i></ArticleTitle><Abstract><AbstractText Label="RESULTS">Lower pressure lowers it.</AbstractText></Abstract>'
                  b'<AuthorList><Author><LastName>Curie</LastName><Initials>M</Initials></Author></AuthorList></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>',
        'duckduckgo': b'<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fboil&amp;rut=x">Boiling <b>guide</b></a>'
                      b'<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FX&amp;rut=y">dup</a>',
        'example.org': b'<nav>Menu</nav><h1>Guide</h1><p>At 3000 m water boils near 90 C.</p>',
    }
    async def fetch(self, url):
        key = next(k for k in responses if k in url)
        body = responses[key]
        return (json.dumps(body).encode() if isinstance(body, dict) else body), 'text/html', url
    monkeypatch.setattr(SafeFetcher, 'fetch', fetch)
    results = await build_retriever('wikipedia, pubmed, web', 100000).search('boiling point altitude', 4)
    assert [r.metadata['retriever'] for r in results] == ['wikipedia', 'pubmed', 'web', 'wikipedia']
    assert results[0].title == 'Boiling point'  # search rank order, not page-id order
    assert results[1].title == 'Altitude and boiling' and 'RESULTS: Lower pressure' in results[1].text and 'Curie M' in results[1].text
    assert results[2].url == 'https://example.org/boil' and results[2].title == 'Boiling guide'
    assert 'Menu' not in results[2].text and '90 C' in results[2].text
    with pytest.raises(ValueError, match='Unknown'):
        build_retriever('wikipedia,google', 1000)
    assert build_retriever('', 1000) is None
